import contextlib
from pathlib import Path

import numpy as np
import torch

from ._compat import patch_basicsr_torchvision_shim, preset_at
from .ddcolor import DDColor, ColorizationPipeline, build_ddcolor_model

_DDCOLOR_CACHE = {}
_ESRGAN_CACHE = {}
_DEOLDIFY_CACHE = {}


def _stage_dir():
    # nodes/colorize_upscale.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return Path(__file__).resolve().parents[3] / "models" / "photorevive" / "colorize_upscale"


def _deoldify_stage_dir():
    return Path(__file__).resolve().parents[3] / "models" / "photorevive" / "colorize_deoldify"


@contextlib.contextmanager
def _torch_load_permissive():
    """DeOldify's fastai checkpoint pickles a functools.partial, which
    PyTorch >=2.6's default weights_only=True load refuses. The weights come
    from DeOldify's own official release, so allow the full unpickle for the
    duration of the load only."""
    orig_load = torch.load

    def patched(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig_load


def _get_deoldify_colorizer(weights_dir: Path, render_factor: int = 35):
    key = (str(weights_dir), render_factor)
    if key in _DEOLDIFY_CACHE:
        return _DEOLDIFY_CACHE[key]

    weights_path = weights_dir / "models" / "ColorizeArtistic_gen.pth"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"DeOldify checkpoint not found at {weights_path}. "
            "Run `python scripts/download_models.py --only deoldify` first."
        )

    from deoldify import device as deoldify_device
    from deoldify.device_id import DeviceId
    from deoldify.visualize import get_image_colorizer

    deoldify_device.set(device=DeviceId.GPU0 if torch.cuda.is_available() else DeviceId.CPU)

    with _torch_load_permissive():
        colorizer = get_image_colorizer(root_folder=weights_dir, render_factor=render_factor, artistic=True)

    _DEOLDIFY_CACHE[key] = colorizer
    return colorizer


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
    """Stage 4: colorization (only applied to grayscale/B&W photos per the
    analyze stage's preset, so already-color photos aren't re-colorized)
    followed by tiled super-resolution (Real-ESRGAN x4plus, resized to the
    requested factor). Runs after damage repair so scratches/stains aren't
    amplified by upscaling.

    Two colorization backends are available: DDColor (default, fast,
    vendored, no extra deps) and DeOldify (fastai-based GAN, noticeably more
    natural/saturated colors on heavily degraded historical photos, but
    slower and pulls in the optional `deoldify`/`fastai` dependencies)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
                "upscale_factor": ("INT", {"default": 2, "min": 1, "max": 4}),
                "colorizer": (["ddcolor", "deoldify"], {"default": "ddcolor"}),
            },
            "optional": {
                "ddcolor_model_path": ("STRING", {"default": ""}),
                "esrgan_model_path": ("STRING", {"default": ""}),
                "deoldify_weights_dir": ("STRING", {"default": ""}),
                "deoldify_render_factor": ("INT", {"default": 35, "min": 7, "max": 45}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "colorize_upscale"
    CATEGORY = "PhotoRevive"

    def colorize_upscale(
        self,
        image,
        preset,
        upscale_factor,
        colorizer="ddcolor",
        ddcolor_model_path="",
        esrgan_model_path="",
        deoldify_weights_dir="",
        deoldify_render_factor=35,
    ):
        esrgan_path = Path(esrgan_model_path) if esrgan_model_path else _stage_dir() / "RealESRGAN_x4plus.pth"
        upsampler = _get_esrgan_upsampler(esrgan_path)

        ddcolor_path = Path(ddcolor_model_path) if ddcolor_model_path else _stage_dir() / "ddcolor_paper.pth"
        deoldify_dir = Path(deoldify_weights_dir) if deoldify_weights_dir else _deoldify_stage_dir()
        stage_colorizer = None  # lazy-loaded on first image that actually needs colorizing

        outputs = []
        for i in range(image.shape[0]):
            frame = image[i]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
            frame_bgr = np.ascontiguousarray(frame_u8[:, :, ::-1])

            if preset_at(preset, i).get("is_grayscale"):
                if colorizer == "deoldify":
                    if stage_colorizer is None:
                        stage_colorizer = _get_deoldify_colorizer(deoldify_dir, deoldify_render_factor)
                    frame_bgr = self._colorize_with_deoldify(stage_colorizer, frame_bgr, deoldify_render_factor)
                else:
                    if stage_colorizer is None:
                        stage_colorizer = _get_ddcolor_pipeline(ddcolor_path)
                    frame_bgr = stage_colorizer.process(frame_bgr)

            upscaled_bgr, _ = upsampler.enhance(frame_bgr, outscale=upscale_factor)
            final_rgb = np.ascontiguousarray(upscaled_bgr[:, :, ::-1]).astype(np.float32) / 255.0
            outputs.append(final_rgb)

        final = torch.from_numpy(np.stack(outputs, axis=0))
        return (final,)

    @staticmethod
    def _colorize_with_deoldify(deoldify_colorizer, frame_bgr, render_factor):
        import PIL.Image

        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        pil_image = PIL.Image.fromarray(rgb).convert("RGB")
        with _torch_load_permissive():
            filtered = deoldify_colorizer.filter.filter(
                pil_image, pil_image, render_factor=render_factor, post_process=True
            )
        out_rgb = np.array(filtered.convert("RGB"))
        return np.ascontiguousarray(out_rgb[:, :, ::-1])
