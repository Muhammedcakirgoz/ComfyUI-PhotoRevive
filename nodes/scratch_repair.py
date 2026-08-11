from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ._compat import preset_at
from .bopbtl.detection_unet import UNet

_MODEL_CACHE = {}


def _default_checkpoint_path():
    # nodes/scratch_repair.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return (
        Path(__file__).resolve().parents[3]
        / "models"
        / "photorevive"
        / "scratch_repair"
        / "checkpoints"
        / "detection"
        / "FT_Epoch_latest.pt"
    )


def _load_detection_model(checkpoint_path: Path):
    key = str(checkpoint_path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"BOPBTL scratch detection checkpoint not found at {checkpoint_path}. "
            "Run `python scripts/download_models.py --only bopbtl_scratch` first."
        )

    # Params match Global/detection.py in microsoft/Bringing-Old-Photos-Back-to-Life.
    model = UNet(
        in_channels=1,
        out_channels=1,
        depth=4,
        conv_num=2,
        wf=6,
        padding=True,
        batch_norm=True,
        up_mode="upsample",
        with_tanh=False,
        antialiasing=True,
    )
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    _MODEL_CACHE[key] = model
    return model


class PhotoReviveScratchRepair:
    """Stage 2: scratch/damage detection (BOPBTL's detection UNet, from
    microsoft/Bringing-Old-Photos-Back-to-Life) followed by mask-guided
    inpainting + denoise. Skips undamaged photos (per the analyze stage's
    preset) to avoid unnecessary blurring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
            },
            "optional": {
                "mask_threshold": ("FLOAT", {"default": 0.4, "min": 0.05, "max": 0.95, "step": 0.05}),
                "checkpoint_path": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "repair"
    CATEGORY = "PhotoRevive"

    def repair(self, image, preset, mask_threshold=0.4, checkpoint_path=""):
        ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path()
        model = None  # lazy-loaded on first image that actually needs repair

        outputs = []
        for i in range(image.shape[0]):
            frame = image[i]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)

            if preset_at(preset, i).get("damage_level") == "low":
                outputs.append(frame_u8.astype(np.float32) / 255.0)
                continue

            if model is None:
                model = _load_detection_model(ckpt_path)

            mask = self._detect_scratches(model, frame_u8, mask_threshold)
            repaired_u8 = cv2.inpaint(frame_u8, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
            repaired_u8 = cv2.fastNlMeansDenoisingColored(repaired_u8, None, 5, 5, 7, 21)
            outputs.append(repaired_u8.astype(np.float32) / 255.0)

        repaired = torch.from_numpy(np.stack(outputs, axis=0))
        return (repaired,)

    @staticmethod
    @torch.no_grad()
    def _detect_scratches(model: UNet, frame_u8: np.ndarray, threshold: float) -> np.ndarray:
        """Runs the BOPBTL detection UNet on a grayscale, [-1, 1]-normalized,
        256-on-the-short-side (rounded to a multiple of 16) copy of the
        image -- matching Global/detection.py's preprocessing -- then
        upsamples the predicted probability map back to full resolution and
        thresholds it into a binary scratch mask."""
        h, w = frame_u8.shape[:2]
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
        tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
        tensor = (tensor - 0.5) / 0.5
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        if h < w:
            new_h, new_w = 256, w / h * 256
        else:
            new_w, new_h = 256, h / w * 256
        new_h = max(16, int(round(new_h / 16) * 16))
        new_w = max(16, int(round(new_w / 16) * 16))
        scaled = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

        prob = torch.sigmoid(model(scaled))
        prob = F.interpolate(prob, size=(h, w), mode="nearest")
        mask = (prob[0, 0].cpu().numpy() >= threshold).astype(np.uint8) * 255
        return mask
