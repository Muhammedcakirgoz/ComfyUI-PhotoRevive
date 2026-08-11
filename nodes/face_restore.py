from pathlib import Path

import numpy as np
import torch

from ._compat import patch_basicsr_torchvision_shim

_RESTORER_CACHE = {}


def _default_model_path():
    # nodes/face_restore.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return (
        Path(__file__).resolve().parents[3] / "models" / "photorevive" / "face_restore" / "GFPGANv1.4.pth"
    )


def _get_restorer(model_path: Path, upscale: int = 1):
    key = (str(model_path), upscale)
    if key in _RESTORER_CACHE:
        return _RESTORER_CACHE[key]

    if not model_path.exists():
        raise FileNotFoundError(
            f"GFPGAN checkpoint not found at {model_path}. "
            "Run `python scripts/download_models.py --only gfpgan` first."
        )

    patch_basicsr_torchvision_shim()
    from gfpgan import GFPGANer

    # arch="clean" + channel_multiplier=2 matches the official GFPGANv1.4
    # release. bg_upsampler is left off since background upscaling is the
    # colorize_upscale stage's job -- doing it here would duplicate work.
    restorer = GFPGANer(
        model_path=str(model_path),
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    _RESTORER_CACHE[key] = restorer
    return restorer


class PhotoReviveFaceRestore:
    """Stage 3: face restoration (GFPGAN) at low-moderate strength to avoid
    identity/age/expression drift. GFPGAN detects, aligns, restores, and
    pastes faces back on its own; photos with no detected face pass through
    unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "model_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_face"
    CATEGORY = "PhotoRevive"

    def restore_face(self, image, preset, strength, model_path=""):
        path = Path(model_path) if model_path else _default_model_path()
        restorer = _get_restorer(path)

        outputs = []
        for i in range(image.shape[0]):
            frame = image[i]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
            frame_bgr = np.ascontiguousarray(frame_u8[:, :, ::-1])

            _, _, restored_bgr = restorer.enhance(
                frame_bgr,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
                weight=strength,
            )

            if restored_bgr is None:
                outputs.append(frame_u8.astype(np.float32) / 255.0)
                continue

            restored_rgb = np.ascontiguousarray(restored_bgr[:, :, ::-1]).astype(np.float32) / 255.0
            outputs.append(restored_rgb)

        restored = torch.from_numpy(np.stack(outputs, axis=0))
        return (restored,)
