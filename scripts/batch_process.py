"""Runs a whole folder of old photos through the PhotoRevive pipeline
(analyze -> scratch repair -> face restore -> colorize/upscale) without
needing a running ComfyUI server, writing restored copies to an output
folder. Files are processed one at a time -- rather than stacked into a
single batch tensor -- so an album of mixed resolutions/aspect ratios (the
normal case for scanned old photos) works without cropping or padding.

Usage:
    python scripts/batch_process.py --input photos/ --output restored/
    python scripts/batch_process.py --input photos/ --output restored/ \
        --models-dir /path/to/ComfyUI/models --quality-check
"""

import argparse
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from nodes.analyze import PhotoReviveAnalyze
from nodes.colorize_upscale import PhotoReviveColorizeUpscale
from nodes.face_restore import PhotoReviveFaceRestore
from nodes.quality_check import PhotoReviveQualityCheck
from nodes.scratch_repair import PhotoReviveScratchRepair

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _default_models_dir() -> Path:
    # scripts/batch_process.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return Path(__file__).resolve().parents[3] / "models"


def load_image_tensor(path: Path) -> torch.Tensor:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Görüntü okunamadı: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).unsqueeze(0)


def save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    frame = tensor[0]
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_u8, cv2.COLOR_RGB2BGR))


def process_one(image_tensor: torch.Tensor, checkpoints: dict, args) -> torch.Tensor:
    image, preset = PhotoReviveAnalyze().analyze(image_tensor)
    (image,) = PhotoReviveScratchRepair().repair(
        image, preset, mask_threshold=args.mask_threshold, checkpoint_path=checkpoints["bopbtl"]
    )
    (image,) = PhotoReviveFaceRestore().restore_face(
        image, preset, strength=args.face_strength, model_path=checkpoints["gfpgan"]
    )
    (image,) = PhotoReviveColorizeUpscale().colorize_upscale(
        image,
        preset,
        upscale_factor=args.upscale_factor,
        colorizer=args.colorizer,
        ddcolor_model_path=checkpoints["ddcolor"],
        esrgan_model_path=checkpoints["realesrgan"],
        deoldify_weights_dir=checkpoints["deoldify_dir"],
        deoldify_render_factor=args.deoldify_render_factor,
    )
    return image


def main():
    parser = argparse.ArgumentParser(description="Bir klasördeki eski fotoğrafları PhotoRevive pipeline'ından toplu geçirir.")
    parser.add_argument("--input", type=Path, required=True, help="Girdi fotoğraf klasörü")
    parser.add_argument("--output", type=Path, required=True, help="Çıktı klasörü")
    parser.add_argument("--models-dir", type=Path, default=None, help="ComfyUI models dizini (varsayılan: ../../../models)")
    parser.add_argument("--upscale-factor", type=int, default=2)
    parser.add_argument("--face-strength", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.4)
    parser.add_argument(
        "--colorizer",
        choices=["ddcolor", "deoldify"],
        default="ddcolor",
        help="B&W fotoğraflar için renklendirme modeli (deoldify daha gerçekçi ama daha yavaş, "
        "'deoldify'/'fastai' paketlerini gerektirir)",
    )
    parser.add_argument("--deoldify-render-factor", type=int, default=35, help="DeOldify render_factor (7-45)")
    parser.add_argument(
        "--quality-check",
        action="store_true",
        help="Her fotoğraf için öncesi/sonrası karşılaştırma görseli + rapor (.txt) de üret",
    )
    args = parser.parse_args()

    models_dir = args.models_dir or _default_models_dir()
    checkpoints = {
        "bopbtl": str(models_dir / "photorevive" / "scratch_repair" / "checkpoints" / "detection" / "FT_Epoch_latest.pt"),
        "gfpgan": str(models_dir / "photorevive" / "face_restore" / "GFPGANv1.4.pth"),
        "ddcolor": str(models_dir / "photorevive" / "colorize_upscale" / "ddcolor_paper.pth"),
        "realesrgan": str(models_dir / "photorevive" / "colorize_upscale" / "RealESRGAN_x4plus.pth"),
        "deoldify_dir": str(models_dir / "photorevive" / "colorize_deoldify"),
    }

    if not args.input.is_dir():
        print(f"Girdi klasörü bulunamadı: {args.input}")
        sys.exit(1)

    files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        print(f"'{args.input}' içinde desteklenen bir görüntü bulunamadı ({', '.join(sorted(IMAGE_EXTS))}).")
        return

    args.output.mkdir(parents=True, exist_ok=True)
    quality_checker = PhotoReviveQualityCheck() if args.quality_check else None
    quality_models_dir = str(models_dir / "photorevive" / "quality_check")

    ok, failed = 0, []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name} işleniyor...")
        t0 = time.time()
        try:
            original = load_image_tensor(path)
            restored = process_one(original, checkpoints, args)
            out_path = args.output / path.name
            save_image_tensor(restored, out_path)

            if quality_checker is not None:
                comparison, report = quality_checker.check(original, restored, models_dir=quality_models_dir)
                save_image_tensor(comparison, args.output / f"{path.stem}_comparison.png")
                (args.output / f"{path.stem}_report.txt").write_text(report, encoding="utf-8")

            ok += 1
            print(f"  tamamlandı ({time.time() - t0:.1f}s) -> {out_path}")
        except Exception as exc:  # bir dosyadaki hata tüm batch'i durdurmasın
            failed.append((path.name, str(exc)))
            print(f"  HATA: {exc}")
            traceback.print_exc()

    print(f"\nToplam: {len(files)} | Başarılı: {ok} | Başarısız: {len(failed)}")
    if failed:
        for name, err in failed:
            print(f"  - {name}: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
