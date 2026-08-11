from pathlib import Path

import numpy as np
import torch

from ._compat import patch_basicsr_torchvision_shim, preset_at
from .ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

_DDCOLOR_CACHE = {}
_ESRGAN_CACHE = {}


def _stage_dir():
    # nodes/colorize_upscale.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return Path(__file__).resolve().parents[3] / "models" / "photorevive" / "colorize_upscale"


def _get_ddcolor_pipeline(model_path: Path, input_size: int = 512):
    key = str(model_path)
    if key in _DDCOLOR_CACHE:
        return _DDCOLOR_CACHE[key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"DDColor checkpoint not found at {model_path}. "
            "Run `python scripts/download_models.py --only ddcolor` first."
        )

    model = build_ddcolor_model(DDColor, model_path=str(model_path), input_size=input_size, model_size="large")
    pipeline = ColorizationPipeline(model, input_size=input_size)
    _DDCOLOR_CACHE[key] = pipeline
    return pipeline


def _get_esrgan_upsampler(model_path: Path, netscale: int = 4):
    key = str(model_path)
    if key in _ESRGAN_CACHE:
        return _ESRGAN_CACHE[key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"Real-ESRGAN checkpoint not found at {model_path}. "
            "Run `python scripts/download_models.py --only realesrgan` first."
        )

    patch_basicsr_torchvision_shim()
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    # Matches the official RealESRGAN_x4plus.pth config.
    arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=netscale)
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=str(model_path),
        model=arch,
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),
    )
    _ESRGAN_CACHE[key] = upsampler
    return upsampler


class PhotoReviveColorizeUpscale:
    """Stage 4: colorization (DDColor -- only applied to grayscale/B&W
    photos per the analyze stage's preset, so already-color photos aren't
    re-colorized) followed by tiled super-resolution (Real-ESRGAN x4plus,
    resized to the requested factor). Runs after damage repair so
    scratches/stains aren't amplified by upscaling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
                "upscale_factor": ("INT", {"default": 2, "min": 1, "max": 4}),
            },
            "optional": {
                "ddcolor_model_path": ("STRING", {"default": ""}),
                "esrgan_model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "colorize_upscale"
    CATEGORY = "PhotoRevive"

    def colorize_upscale(self, image, preset, upscale_factor, ddcolor_model_path="", esrgan_model_path=""):
        esrgan_path = Path(esrgan_model_path) if esrgan_model_path else _stage_dir() / "RealESRGAN_x4plus.pth"
        upsampler = _get_esrgan_upsampler(esrgan_path)

        ddcolor_path = Path(ddcolor_model_path) if ddcolor_model_path else _stage_dir() / "ddcolor_paper.pth"
        colorizer = None  # lazy-loaded on first image that actually needs colorizing

        outputs = []
        for i in range(image.shape[0]):
            frame = image[i]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
            frame_bgr = np.ascontiguousarray(frame_u8[:, :, ::-1])

            if preset_at(preset, i).get("is_grayscale"):
                if colorizer is None:
                    colorizer = _get_ddcolor_pipeline(ddcolor_path)
                frame_bgr = colorizer.process(frame_bgr)

            upscaled_bgr, _ = upsampler.enhance(frame_bgr, outscale=upscale_factor)
            final_rgb = np.ascontiguousarray(upscaled_bgr[:, :, ::-1]).astype(np.float32) / 255.0
            outputs.append(final_rgb)

        final = torch.from_numpy(np.stack(outputs, axis=0))
        return (final,)
