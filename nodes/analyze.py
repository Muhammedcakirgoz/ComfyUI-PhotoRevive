import cv2
import numpy as np


class PhotoReviveAnalyze:
    """Stage 1: classify input photo (b&w vs color, damage level, face presence)
    and derive preset parameters for the downstream stages."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "PHOTOREVIVE_PRESET")
    RETURN_NAMES = ("image", "preset")
    FUNCTION = "analyze"
    CATEGORY = "PhotoRevive"

    def analyze(self, image):
        presets = []
        for i in range(image.shape[0]):
            frame = image[i]
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)

            is_grayscale = self._is_grayscale(frame_u8)
            damage_level, damage_score = self._estimate_damage(frame_u8)

            presets.append(
                {
                    "is_grayscale": is_grayscale,
                    "damage_level": damage_level,
                    "damage_score": damage_score,
                    "has_face": False,
                }
            )
        return (image, presets)

    @staticmethod
    def _is_grayscale(frame_u8, saturation_threshold=12.0):
        """A scanned B&W photo may have a slight sepia/paper color cast, so
        classify by mean saturation rather than requiring exactly equal
        R/G/B channels."""
        hsv = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2HSV)
        mean_saturation = float(hsv[..., 1].mean())
        return mean_saturation < saturation_threshold

    @staticmethod
    def _estimate_damage(frame_u8):
        """Edge density (Canny, auto-thresholded from the image's median
        intensity) as a proxy for scratches/stains/noise: clean photos have
        sparse, mostly-smooth regions, while damaged ones have a lot more
        high-frequency edge content."""
        gray = cv2.cvtColor(frame_u8, cv2.COLOR_RGB2GRAY)
        median = float(np.median(gray))
        lower = int(max(0, 0.66 * median))
        upper = int(min(255, 1.33 * median))
        edges = cv2.Canny(gray, lower, upper)
        edge_density = float(edges.mean() / 255.0)

        if edge_density < 0.02:
            level = "low"
        elif edge_density < 0.06:
            level = "medium"
        else:
            level = "high"
        return level, edge_density
