class PhotoReviveColorizeUpscale:
    """Stage 4: colorization (DDColor) + tiled upscale (ESRGAN).
    Runs after damage repair so scratches/stains aren't amplified by upscaling."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
                "upscale_factor": ("INT", {"default": 2, "min": 1, "max": 4}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("final_image",)
    FUNCTION = "colorize_upscale"
    CATEGORY = "PhotoRevive"

    def colorize_upscale(self, image, preset, upscale_factor):
        return (image,)
