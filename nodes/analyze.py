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
        preset = {
            "is_grayscale": False,
            "damage_level": "unknown",
            "has_face": False,
        }
        return (image, preset)
