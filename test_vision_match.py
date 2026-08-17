"""OCR yazim hatalarinda dogru urun eslesmesi."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

os.environ["SKIP_HF"] = "1"

import store
import vision


@pytest.fixture()
def temp_db(monkeypatch):
    tmp = tempfile.mkdtemp()
    dbf = Path(tmp) / "t.db"

    monkeypatch.setattr(store, "db_path", lambda: dbf)
    store.init_db(with_demo_orders=False)
    yield store
    try:
        dbf.unlink(missing_ok=True)
    except OSError:
        pass


def _rows(db_mod):
    with db_mod.get_connection() as conn:
        return conn.execute("SELECT * FROM products").fetchall()


def test_fant_ocr_matches_fanta_not_coca_cola(temp_db):
    rows = _rows(temp_db)
    ocr = "FANT Gazoz 1 kutu Portakall Aromalı"
    matched = vision._ocr_product_match(rows, ocr.lower(), vision._compact(ocr))
    assert matched is not None
    assert matched["sku"] == "SKU-FANTA"


def test_low_conf_yolo_does_not_override_fanta_ocr(temp_db):
    rows = _rows(temp_db)
    ocr = "FANT Gazoz Portakall Aromalı"
    result = vision._match_product(
        "SKU-CC-330",
        ocr,
        yolo_backend="yolov8_cls_custom",
        yolo_conf=0.528,
        ner_hints=["FANT Gazoz", "Portakall Aromalı"],
    )
    assert result is not None
    assert result["sku"] == "SKU-FANTA"
