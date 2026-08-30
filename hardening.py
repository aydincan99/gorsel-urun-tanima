"""Yükleme, hız sınırı, log ve girdi doğrulama."""

from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict, deque
from io import BytesIO

from PIL import Image, UnidentifiedImageError

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))
ALLOWED_IMAGE_EXT = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
LOGIN_LIMIT = 8
LOGIN_WINDOW = 900
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(secret|token|password|api[_-]?key|authorization|mongo_uri)\s*[:=]\s*.+"),
    re.compile(r"(?i)bearer\s+\S+"),
)

_login_hits: dict[str, deque] = defaultdict(deque)
logger = logging.getLogger("gorsel_urun")


class SecretLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for pat in SECRET_PATTERNS:
            msg = pat.sub("[REDACTED]", msg)
        record.msg = msg
        record.args = ()
        return True


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    root = logging.getLogger()
    if not any(isinstance(f, SecretLogFilter) for f in root.filters):
        root.addFilter(SecretLogFilter())


def login_allowed(key: str) -> bool:
    now = time.time()
    bucket = _login_hits[key.lower()]
    while bucket and now - bucket[0] > LOGIN_WINDOW:
        bucket.popleft()
    return len(bucket) < LOGIN_LIMIT


def login_fail(key: str) -> None:
    _login_hits[key.lower()].append(time.time())


def sanitize_text(value: str, limit: int = 200) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(value or "")).strip()
    return text[:limit]


def public_error(_: Exception) -> str:
    logger.exception("İşlem hatası")
    return "İşlem tamamlanamadı. Lütfen tekrar deneyin."


def validate_upload(file) -> Image.Image:
    name = getattr(file, "name", "") or "upload.jpg"
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXT:
        raise ValueError("Geçersiz dosya türü.")
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Dosya boş veya çok büyük.")
    try:
        image = Image.open(BytesIO(data))
        image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Geçersiz görüntü.") from exc
    image = Image.open(BytesIO(data)).convert("RGB")
    return image
