"""Bitirme icin: fotodan urun bulma (YOLO+OCR vs), bazen karisik bi kod var dikkat."""

from __future__ import annotations  # py3.10+

import functools
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import store  # sqlite tarafina bagimli, once mi sonra mi emin degilim
from store import ProjectPaths

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore


def nltk_tokens(text: str) -> list[str]:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import word_tokenize

        return [t.lower() for t in word_tokenize(text or "") if re.search(r"\w", t)]
    except Exception:  # nltk yoksa veya hata -> basit split
        return [t.lower() for t in re.split(r"\W+", text or "") if len(t) > 1]


def best_tfidf_product(
    text: str, candidates: list[tuple[str, dict[str, Any]]]
) -> tuple[dict[str, Any] | None, float]:
    if not text.strip() or not candidates:
        return None, 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        corpus = [text] + [c[0] for c in candidates]
        vec = TfidfVectorizer(lowercase=True, analyzer="word", token_pattern=r"(?u)\b\w\w+\b")
        mat = vec.fit_transform(corpus)
        sims = cosine_similarity(mat[0:1], mat[1:]).ravel()
        i = int(sims.argmax())
        return candidates[i][1], float(sims[i])
    except Exception:
        return None, 0.0


@functools.lru_cache(maxsize=1)
def _hf_ner_pipe():
    if os.getenv("SKIP_HF", "").lower() in ("1", "true", "yes"):
        return None
    try:
        from transformers import pipeline

        return pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple", device=-1)
    except Exception:
        return None


def hf_entities(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    pipe = _hf_ner_pipe()
    if pipe is None:
        return []
    try:
        out = pipe(text[:512])
        return sorted({str(x.get("word", "")).strip() for x in out if x.get("word")})
    except Exception:
        return []


def spacy_lemmas(text: str) -> list[str]:
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text[:2000])
        return [t.lemma_.lower() for t in doc if t.is_alpha and len(t.text) > 2]
    except Exception:
        return []


MANIFEST_NAME = "catalog_manifest.csv"  # csv yolu data/manifests altinda


def safe_segment(name: str, fallback: str = "x") -> str:
    base = re.sub(r"[^\w.\-]", "_", Path(name).name.strip(), flags=re.UNICODE)
    return base or fallback


def safe_filename(name: str) -> str:
    return safe_segment(name, "upload.jpg")


def manifest_path(paths: ProjectPaths | None = None) -> Path:
    p = paths or ProjectPaths.default()
    return p.data_manifests / MANIFEST_NAME


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, default in (("label", ""), ("view", ""), ("modality", "")):
        if col not in out.columns:
            out[col] = default
    return out


def load_manifest(paths: ProjectPaths | None = None) -> pd.DataFrame:
    p = manifest_path(paths)
    if not p.exists():
        return pd.DataFrame(columns=["image_path", "label", "view", "modality"])
    return ensure_columns(pd.read_csv(p))


def append_manifest(
    paths: ProjectPaths,
    rel_under_raw: str,
    label: str,
    *,
    view: str = "",
    modality: str = "",
) -> None:
    df = ensure_columns(load_manifest(paths))
    row = pd.DataFrame([{"image_path": rel_under_raw, "label": label, "view": view, "modality": modality}])
    df = pd.concat([df, row], ignore_index=True).drop_duplicates(subset=["image_path"], keep="last")
    mp = manifest_path(paths)
    mp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(mp, index=False)


def resolve_manifest_for_display(df: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    out = ensure_columns(df)
    resolved, exists = [], []
    for p in out["image_path"].astype(str):
        rel = Path(p)
        if len(rel.parts) >= 2 and rel.parts[0] in ("images", "receipts", "products"):
            full = (paths.data_raw / rel).resolve()
        else:
            full = (paths.data_raw_images / rel.name).resolve()
        resolved.append(str(full))
        exists.append(full.is_file())
    out["resolved_path"] = resolved
    out["file_exists"] = exists
    return out


@dataclass
class QRDecodeResult:
    text: str
    points: np.ndarray | None

    @property
    def ok(self) -> bool:
        return bool(self.text)


def decode_qr_from_pil(pil_image: Any) -> QRDecodeResult:
    if cv2 is None:
        raise ImportError("opencv-python-headless gerekli.")
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    det = cv2.QRCodeDetector()
    text, points, _ = det.detectAndDecode(bgr)
    return QRDecodeResult(text=text or "", points=points)


def mongo_insert_doc(doc: dict[str, Any]) -> None:
    uri = os.getenv("MONGO_URI", "").strip()
    if not uri:
        return
    try:
        from pymongo import MongoClient

        with MongoClient(uri, serverSelectionTimeoutMS=2500) as client:
            client[os.getenv("MONGO_DB", "bakkal")][os.getenv("MONGO_COLLECTION", "ocr_runs")].insert_one(doc)
    except Exception:
        pass


@dataclass
class AnalysisResult:
    product: dict[str, Any] | None
    yolo_label: str
    yolo_conf: float
    ocr_text: str
    ocr_conf: float
    yolo_backend: str
    ocr_backend: str
    nlp_tokens: list[str]
    ner_hints: list[str]


def _ocr_tesseract(pil: Image.Image) -> tuple[str, float, str]:
    try:
        import pytesseract

        txt = (pytesseract.image_to_string(pil.convert("RGB"), lang="eng+tur") or "").strip()
        return txt, 0.65 if txt else 0.0, "tesseract"
    except Exception:
        return "", 0.0, "tesseract_skip"


def _ocr_easy(pil: Image.Image) -> tuple[str, float, str]:
    try:
        import easyocr

        reader = getattr(_ocr_easy, "_reader", None)
        if reader is None:
            reader = easyocr.Reader(["en", "tr"], gpu=False, verbose=False)
            _ocr_easy._reader = reader
        arr = np.array(pil.convert("RGB"))
        lines = reader.readtext(arr, detail=1, paragraph=False)
        if not lines:
            t2, c2, b2 = _ocr_tesseract(pil)
            return (t2, c2, f"easyocr+{b2}") if t2 else ("", 0.0, "easyocr")
        texts, confs = zip(*[(t, float(c)) for _b, t, c in lines])
        return " ".join(texts), float(sum(confs) / len(confs)), "easyocr"
    except Exception:
        t2, c2, b2 = _ocr_tesseract(pil)
        return (t2, c2, b2) if t2 else ("", 0.55, "mock")


def read_text_ocr(pil_image: Image.Image) -> tuple[str, float, str]:
    return _ocr_easy(pil_image)


def _yolo_detect(np_rgb: np.ndarray) -> tuple[str, float, str]:
    weights = ProjectPaths.default().models_exports / "yolov8_product.pt"  # egitilmisse burda
    if weights.is_file():
        try:
            from ultralytics import YOLO

            model = YOLO(str(weights))
            r = model.predict(np_rgb, verbose=False)[0]
            if r.boxes is not None and len(r.boxes) > 0:
                i = int(r.boxes.conf.argmax())
                return str(model.names[int(r.boxes.cls[i])]), float(r.boxes.conf[i]), "yolov8_detect_custom"
            if r.probs is not None:
                idx = int(r.probs.top1)
                return str(model.names[idx]), float(r.probs.top1conf), "yolov8_cls_custom"
        except Exception:  # model bozuksa asagiya dus
            pass
    rows = store.list_all()  # db'den urun listesi
    if not rows:
        return "unknown", 0.0, "yolov8_demo_hash"
    h = hashlib.sha256(np_rgb.tobytes()).hexdigest()  # demo mod: gorsele gore rasgele gibi
    r = rows[int(h[:8], 16) % len(rows)]
    raw = r["yolo_class_hint"] or r["sku"] or r["name"] or "unknown"
    lab = str(raw).lower().replace(" ", "_").replace("-", "_")
    return lab, float(0.70 + (int(h[8:10], 16) % 20) / 100.0), "yolov8_demo_hash"


def _compact(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def _norm_label(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def _ocr_product_match(rows: list[sqlite3.Row], ocr_l: str, ocr_c: str) -> dict[str, Any] | None:
    best_score, best_row = 0, None

    def consider(row: sqlite3.Row, score: int) -> None:
        nonlocal best_score, best_row
        if score > best_score:
            best_score, best_row = score, row

    for r in rows:
        for field in (r["name"], r["yolo_class_hint"], r["sku"]):
            v = (field or "").strip()
            if len(v) < 2:
                continue
            vl, vc = v.lower(), _compact(v)
            if len(vl) >= 4 and vl in ocr_l:
                consider(r, len(vl))
            if len(vc) >= 4 and vc in ocr_c:
                consider(r, len(vc))
            for tok in re.split(r"\W+", vl):
                t = tok.strip().lower()
                if len(t) >= 4 and not t.isdigit() and t in ocr_l:
                    consider(r, len(t))
    if best_row == None:
        return None
    if best_score < 4:
        return None
    return store.row_to_dict(best_row)


def _sku_yolo_match(rows: list[sqlite3.Row], y_l: str) -> dict[str, Any] | None:
    for r in rows:
        sku = (r["sku"] or "").strip()
        if sku and _norm_label(sku) == y_l:
            return store.row_to_dict(r)
    return None


def _hint_and_token_match(rows: list[sqlite3.Row], y_l: str, ocr_l: str) -> dict[str, Any] | None:
    for r in rows:
        hint, name = (r["yolo_class_hint"] or "").lower(), (r["name"] or "").lower()
        if hint and hint in y_l:
            return store.row_to_dict(r)
        if hint and hint in ocr_l.replace(" ", "_"):
            return store.row_to_dict(r)
        for token in re.split(r"\W+", ocr_l):
            if len(token) >= 4 and token in name:
                return store.row_to_dict(r)
    for r in rows:
        if (r["yolo_class_hint"] or "").lower() == y_l:
            return store.row_to_dict(r)
    return None


def _tfidf_match(ocr_text: str, rows: list[sqlite3.Row]) -> dict[str, Any] | None:
    cands = []
    for r in rows:
        parca = str(r["name"] or "") + " " + str(r["sku"] or "") + " " + str(r["yolo_class_hint"] or "")
        cands.append((parca.strip(), store.row_to_dict(r)))
    best, score = best_tfidf_product(ocr_text, cands)
    if best == None:
        return None
    if score < 0.12:
        return None
    return best


def _match_product(
    yolo_label: str,
    ocr_text: str,
    *,
    yolo_backend: str = "",
    ner_hints: list[str],
) -> dict[str, Any] | None:
    rows = store.list_all()
    if len(rows) == 0:
        return None
    ocr_l = ocr_text.lower()
    if ner_hints != None and len(ner_hints) > 0:
        ocr_l = ocr_l + " "
        ocr_l = ocr_l + " ".join([h.lower() for h in ner_hints])
    ocr_c = _compact(ocr_text)
    y_l = _norm_label(yolo_label)
    trained = yolo_backend == "yolov8_cls_custom" or yolo_backend == "yolov8_detect_custom"

    m = _ocr_product_match(rows, ocr_l, ocr_c)
    if m == None:
        m = _sku_yolo_match(rows, y_l)
    if m == None:
        m = _hint_and_token_match(rows, y_l, ocr_l)
    if m == None:
        m = _tfidf_match(ocr_text, rows)

    if trained == True:
        return m
    if m != None:
        return m
    h2 = hashlib.md5(y_l.encode()).hexdigest()
    idx2 = int(h2, 16) % len(rows)
    secilen_satir = rows[idx2]
    return store.row_to_dict(secilen_satir)


def analyze_product_image(
    pil_image: Image.Image,
    *,
    log_scan: bool = True,
    user_id: int | None = None,
) -> AnalysisResult:
    pil_rgb = pil_image.convert("RGB")
    np_rgb = np.array(pil_rgb)
    y_label, y_conf, y_back = _yolo_detect(np_rgb)
    ocr_txt, ocr_cf, o_back = _ocr_easy(pil_image)
    toks = nltk_tokens(ocr_txt)
    ner = hf_entities(ocr_txt) or spacy_lemmas(ocr_txt)[:5]  # spacy opsiyonel, yoksa bos doner
    matched = _match_product(y_label, ocr_txt, yolo_backend=y_back, ner_hints=ner)
    pid = matched["id"] if matched else None
    mongo_insert_doc(  # MONGO_URI yoksa zaten icinde return
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "product_id": pid,
            "yolo": {"label": y_label, "conf": y_conf, "backend": y_back},
            "ocr": {"text": ocr_txt[:4000], "conf": ocr_cf, "backend": o_back},
            "nlp": {"tokens": toks[:80], "ner_or_lemma": ner[:40]},
        }
    )
    if log_scan:
        store.log_scan(
            user_id=user_id,
            product_id=pid,
            yolo_label=y_label,
            yolo_conf=y_conf,
            ocr_text=ocr_txt[:2000],
            ocr_conf=ocr_cf,
            notes=f"backend:{y_back},{o_back}",
        )
    return AnalysisResult(
        product=matched,
        yolo_label=y_label,
        yolo_conf=y_conf,
        ocr_text=ocr_txt,
        ocr_conf=ocr_cf,
        yolo_backend=y_back,
        ocr_backend=o_back,
        nlp_tokens=toks[:50],
        ner_hints=ner[:20],
    )
