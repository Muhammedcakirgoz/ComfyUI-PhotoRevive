import sys
import types


def preset_at(preset, index: int):
    """PhotoRevive_Analyze returns one preset dict per image in the batch
    (a list). Downstream nodes accept either that list (indexing by batch
    position) or a single dict (applied to every image), so a lone
    PHOTOREVIVE_PRESET can still be wired in by hand outside a batch."""
    if isinstance(preset, list):
        return preset[index % len(preset)]
    return preset


def patch_basicsr_torchvision_shim():
    """basicsr (a GFPGAN/Real-ESRGAN dependency, unmaintained) imports
    torchvision.transforms.functional_tensor, which torchvision removed in
    0.17+. Inject a shim module re-exporting the one function it needs so
    `import gfpgan` / `import realesrgan` don't crash on current torchvision."""
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    import torchvision.transforms.functional as F

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim
