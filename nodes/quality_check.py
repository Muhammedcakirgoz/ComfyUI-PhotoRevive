from pathlib import Path

import cv2
import numpy as np
import torch

from ._compat import patch_basicsr_torchvision_shim

_HELPER_CACHE = {}
_ARCFACE_CACHE = {}


def _models_dir():
    # nodes/quality_check.py -> ComfyUI-PhotoRevive -> custom_nodes -> ComfyUI
    return Path(__file__).resolve().parents[3] / "models" / "photorevive" / "quality_check"


def _get_face_helper(models_dir: Path, face_size: int = 112):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(models_dir), face_size, device)
    if key in _HELPER_CACHE:
        return _HELPER_CACHE[key]

    patch_basicsr_torchvision_shim()
    from facexlib.utils.face_restoration_helper import FaceRestoreHelper

    helper = FaceRestoreHelper(
        upscale_factor=1,
        face_size=face_size,
        det_model="retinaface_resnet50",
        device=device,
        model_rootpath=str(models_dir),
    )
    _HELPER_CACHE[key] = helper
    return helper


def _get_arcface_model(models_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    key = (str(models_dir), device)
    if key in _ARCFACE_CACHE:
        return _ARCFACE_CACHE[key]

    patch_basicsr_torchvision_shim()
    from facexlib.recognition import init_recognition_model

    try:
        model = init_recognition_model("arcface", device=device, model_rootpath=str(models_dir))
    except RuntimeError:
        # facexlib's init_recognition_model hardcodes `.to('cuda')` before
        # honoring `device`, so it raises on CPU-only machines. Degrade
        # gracefully -- the report notes identity couldn't be compared.
        model = None
    _ARCFACE_CACHE[key] = model
    return model


def _detect_and_align_face(helper, frame_bgr: np.ndarray):
    helper.clean_all()
    helper.read_image(frame_bgr)
    num_faces = helper.get_face_landmarks_5(only_center_face=True, eye_dist_threshold=5)
    if num_faces == 0:
        return None
    helper.align_warp_face()
    if not helper.cropped_faces:
        return None
    return helper.cropped_faces[0]


def _face_embedding(model, aligned_face_bgr: np.ndarray, device: str) -> np.ndarray:
    face = cv2.resize(aligned_face_bgr, (112, 112))
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(face_rgb.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return embedding[0].cpu().numpy()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


class PhotoReviveQualityCheck:
    """Stage 5: before/after quality control. Flags restorations that may
    have drifted too far from the original -- face identity (via
    facexlib's ArcFace recognition model, aligned on RetinaFace
    landmarks), or the overall background/clothing composition (via a
    dependency-free histogram + edge-map correlation) -- and produces a
    side-by-side comparison image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "restored_image": ("IMAGE",),
            },
            "optional": {
                "identity_threshold": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
                "background_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "models_dir": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("comparison_image", "report")
    FUNCTION = "check"
    CATEGORY = "PhotoRevive"

    def check(self, original_image, restored_image, identity_threshold=0.35, background_threshold=0.5, models_dir=""):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        models_path = Path(models_dir) if models_dir else _models_dir()
        helper = _get_face_helper(models_path)
        arcface = _get_arcface_model(models_path)

        batch_size = min(original_image.shape[0], restored_image.shape[0])
        comparison_rows = []
        reports = []

        for i in range(batch_size):
            orig_bgr = self._to_bgr(original_image, i)
            rest_bgr = self._to_bgr(restored_image, i)
            rest_bgr = cv2.resize(rest_bgr, (orig_bgr.shape[1], orig_bgr.shape[0]))

            orig_face = _detect_and_align_face(helper, orig_bgr)
            rest_face = _detect_and_align_face(helper, rest_bgr)

            warnings = []
            identity_score = None

            if orig_face is not None and rest_face is None:
                warnings.append("Orijinalde yüz vardı, restorasyon sonrasında yüz tespit edilemedi.")
            elif orig_face is None and rest_face is not None:
                warnings.append("Orijinalde yüz yoktu, restorasyon sonrasında yüz tespit edildi (beklenmeyen).")
            elif orig_face is not None and rest_face is not None:
                if arcface is not None:
                    emb_orig = _face_embedding(arcface, orig_face, device)
                    emb_rest = _face_embedding(arcface, rest_face, device)
                    identity_score = _cosine_similarity(emb_orig, emb_rest)
                    if identity_score < identity_threshold:
                        warnings.append(
                            f"Yüz kimliği önemli ölçüde değişmiş olabilir "
                            f"(benzerlik={identity_score:.2f} < eşik={identity_threshold})."
                        )
                else:
                    warnings.append("Kimlik karşılaştırması yapılamadı (ArcFace modeli bu sistemde kullanılamıyor).")

            background_score = self._background_similarity(orig_bgr, rest_bgr)
            if background_score < background_threshold:
                warnings.append(
                    f"Arka plan / kıyafet önemli ölçüde değişmiş olabilir "
                    f"(benzerlik={background_score:.2f} < eşik={background_threshold})."
                )

            info_lines = []
            if identity_score is not None:
                info_lines.append(f"Kimlik benzerliği: {identity_score:.3f}")
            info_lines.append(f"Arka plan benzerliği: {background_score:.3f}")

            header = f"[Görüntü {i + 1}/{batch_size}]" if batch_size > 1 else None
            lines = info_lines + (warnings or ["Önemli bir sapma tespit edilmedi."])
            reports.append("\n".join(([header] if header else []) + lines))

            comparison_rows.append(np.concatenate([orig_bgr, rest_bgr], axis=1))

        report = "\n\n".join(reports)

        # Farklı fotoğraflar farklı boyutlarda olabileceğinden, satırları
        # dikey olarak birleştirmeden önce ortak bir genişliğe getir.
        max_width = max(row.shape[1] for row in comparison_rows)
        padded_rows = [
            row if row.shape[1] == max_width else cv2.copyMakeBorder(
                row, 0, 0, 0, max_width - row.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            for row in comparison_rows
        ]
        comparison_bgr = np.concatenate(padded_rows, axis=0)
        comparison_rgb = cv2.cvtColor(comparison_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        comparison = torch.from_numpy(comparison_rgb).unsqueeze(0)

        return (comparison, report)

    @staticmethod
    def _to_bgr(image, index: int = 0) -> np.ndarray:
        frame = image[index]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        frame_u8 = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame_u8[:, :, ::-1])

    @staticmethod
    def _background_similarity(orig_bgr: np.ndarray, rest_bgr: np.ndarray) -> float:
        """Dependency-free stand-in for a structural similarity score:
        averages a grayscale-histogram correlation (overall tone/exposure
        match) with an edge-map correlation (structure/layout match)."""
        orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)
        rest_gray = cv2.cvtColor(rest_bgr, cv2.COLOR_BGR2GRAY)

        hist_orig = cv2.calcHist([orig_gray], [0], None, [64], [0, 256])
        hist_rest = cv2.calcHist([rest_gray], [0], None, [64], [0, 256])
        cv2.normalize(hist_orig, hist_orig)
        cv2.normalize(hist_rest, hist_rest)
        hist_corr = cv2.compareHist(hist_orig, hist_rest, cv2.HISTCMP_CORREL)

        edges_orig = cv2.Canny(orig_gray, 50, 150).astype(np.float32) / 255.0
        edges_rest = cv2.Canny(rest_gray, 50, 150).astype(np.float32) / 255.0
        edge_corr = float(np.corrcoef(edges_orig.flatten(), edges_rest.flatten())[0, 1])
        if np.isnan(edge_corr):
            edge_corr = 0.0

        return float(max(0.0, min(1.0, (hist_corr + edge_corr) / 2)))
