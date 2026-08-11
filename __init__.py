from .nodes.analyze import PhotoReviveAnalyze
from .nodes.scratch_repair import PhotoReviveScratchRepair
from .nodes.face_restore import PhotoReviveFaceRestore
from .nodes.colorize_upscale import PhotoReviveColorizeUpscale

NODE_CLASS_MAPPINGS = {
    "PhotoRevive_Analyze": PhotoReviveAnalyze,
    "PhotoRevive_ScratchRepair": PhotoReviveScratchRepair,
    "PhotoRevive_FaceRestore": PhotoReviveFaceRestore,
    "PhotoRevive_ColorizeUpscale": PhotoReviveColorizeUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoRevive_Analyze": "PhotoRevive: Analyze",
    "PhotoRevive_ScratchRepair": "PhotoRevive: Scratch/Damage Repair",
    "PhotoRevive_FaceRestore": "PhotoRevive: Face Restore",
    "PhotoRevive_ColorizeUpscale": "PhotoRevive: Colorize + Upscale",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
