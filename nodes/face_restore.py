class PhotoReviveFaceRestore:
    """Stage 3: face restoration (GFPGAN/CodeFormer) at low-moderate strength
    to avoid identity/age/expression drift."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ("PHOTOREVIVE_PRESET",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "restore_face"
    CATEGORY = "PhotoRevive"

    def restore_face(self, image, preset, strength):
        return (image,)
