"""Downloads model weights required by the PhotoRevive pipeline
(BOPBTL scratch repair, GFPGAN/CodeFormer face restoration, DDColor
colorization, Real-ESRGAN upscale) into ComfyUI's models directory.

Usage:
    python scripts/download_models.py                # download everything
    python scripts/download_models.py --only gfpgan   # download one model
    python scripts/download_models.py --list          # list known models
    python scripts/download_models.py --models-dir /path/to/ComfyUI/models
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

# Each entry: destination is relative to <models_dir>/photorevive/<stage>/<filename>.
# Zip archives are downloaded then extracted into the stage directory; "check_file"
# is a path (relative to the stage dir) used to decide whether it's already installed.
MODELS = {
    "bopbtl_scratch": {
        "stage": "scratch_repair",
        "filename": "global_checkpoints.zip",
        "url": "https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life/releases/download/v1.0/global_checkpoints.zip",
        "sha256": None,
        "extract": True,
        "check_file": "checkpoints/detection/FT_Epoch_latest.pt",
    },
    "gfpgan": {
        "stage": "face_restore",
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "sha256": None,
    },
    "codeformer": {
        "stage": "face_restore",
        "filename": "codeformer.pth",
        "url": "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
        "sha256": None,
    },
    "ddcolor": {
        "stage": "colorize_upscale",
        "filename": "ddcolor_paper.pth",
        "url": "https://huggingface.co/piddnad/DDColor-models/resolve/main/ddcolor_paper.pth",
        "sha256": None,
    },
    "realesrgan": {
        "stage": "colorize_upscale",
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "sha256": None,
    },
}

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def report(block_num, block_size, total_size):
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = done * 100 // total_size
        sys.stdout.write(f"\r  {dest.name}: {pct}% ({done // (1024 * 1024)}MB/{total_size // (1024 * 1024)}MB)")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=report)
    sys.stdout.write("\n")
    tmp.rename(dest)


def fetch_model(key: str, spec: dict, models_dir: Path, force: bool) -> bool:
    stage_dir = models_dir / "photorevive" / spec["stage"]
    dest = stage_dir / spec["filename"]

    if spec.get("extract"):
        marker = stage_dir / spec["check_file"]
        if marker.exists() and not force:
            print(f"[skip] {key}: already present at {marker}")
            return True
    elif dest.exists() and not force:
        print(f"[skip] {key}: already present at {dest}")
        return True

    print(f"[fetch] {key}: {spec['url']}")
    try:
        download(spec["url"], dest)
    except Exception as exc:
        print(f"[error] {key}: download failed: {exc}")
        return False

    if spec.get("sha256"):
        digest = sha256sum(dest)
        if digest != spec["sha256"]:
            print(f"[error] {key}: checksum mismatch (got {digest})")
            dest.unlink(missing_ok=True)
            return False

    if spec.get("extract"):
        print(f"[extract] {key}: {dest.name} -> {stage_dir}")
        try:
            with zipfile.ZipFile(dest) as zf:
                zf.extractall(stage_dir)
        except zipfile.BadZipFile as exc:
            print(f"[error] {key}: bad zip archive: {exc}")
            return False
        dest.unlink()
        dest = stage_dir / spec["check_file"]
        if not dest.exists():
            print(f"[error] {key}: expected file missing after extraction: {dest}")
            return False

    print(f"[ok] {key}: saved to {dest}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download PhotoRevive model weights.")
    parser.add_argument("--only", nargs="*", choices=list(MODELS), help="Download only these models.")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR, help="ComfyUI models directory (default: ../../../models relative to this script, i.e. ComfyUI/models).")
    parser.add_argument("--force", action="store_true", help="Re-download even if the file already exists.")
    parser.add_argument("--list", action="store_true", help="List known models and exit.")
    args = parser.parse_args()

    if args.list:
        for key, spec in MODELS.items():
            print(f"{key:15s} -> photorevive/{spec['stage']}/{spec['filename']}")
        return

    selected = args.only or list(MODELS)
    ok = True
    for key in selected:
        ok = fetch_model(key, MODELS[key], args.models_dir, args.force) and ok

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
