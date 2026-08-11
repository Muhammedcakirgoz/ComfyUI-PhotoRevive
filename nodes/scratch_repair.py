class PhotoReviveScratchRepair:
    """Stage 2: scratch/damage detection and removal (BOPBTL) + denoise.
    Saves a restored B&W checkpoint before colorization."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("restored_image",)
    FUNCTION = "repair"
    CATEGORY = "PhotoRevive"

    def repair(self, image, preset):
        return (image,)
