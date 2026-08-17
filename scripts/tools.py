"""Komut satiri araclari (yolo dataset, train, sklearn denemeleri falan)."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent  # proje kokune import icin
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from store import ProjectPaths

_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def embed_paths(file_paths: list[Path], batch_size: int = 16) -> np.ndarray:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor

    if not file_paths:
        return np.zeros((0, 512), dtype=np.float64)

    class _Ds(Dataset):
        def __init__(self, paths: list[Path], img_size: int = 224) -> None:
            self.paths = paths
            self.tf = Compose(
                [Resize((img_size, img_size)), ToTensor(), Normalize(_IMAGENET_MEAN, _IMAGENET_STD)]
            )

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, i: int) -> torch.Tensor:
            return self.tf(Image.open(self.paths[i]).convert("RGB"))

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=w)
    net.fc = torch.nn.Identity()
    net.eval().to(dev)
    loader = DataLoader(_Ds(file_paths), batch_size=batch_size, shuffle=False, num_workers=0)
    outs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            gpu_tensor = batch.to(dev)
            cpu_np = net(gpu_tensor).cpu().numpy().astype(np.float64)
            outs.append(cpu_np)
    return np.concatenate(outs, axis=0)


def label_enc(y_str: list[str]) -> tuple[np.ndarray, list[str]]:
    uniq = sorted(set(y_str))
    m: dict[str, int] = {}
    i = 0
    while i < len(uniq):
        m[uniq[i]] = i
        i = i + 1
    out_list: list[int] = []
    for c in y_str:
        out_list.append(m[c])
    return np.array(out_list, dtype=np.int64), uniq


def json_clean(d: Any) -> Any:
    if isinstance(d, dict):
        return {str(k): json_clean(v) for k, v in d.items()}
    if isinstance(d, list):
        return [json_clean(x) for x in d]
    if isinstance(d, (np.floating, np.integer)):
        return float(d) if isinstance(d, np.floating) else int(d)
    return d


def _dataset_yolo_root(paths: ProjectPaths | None = None) -> Path:
    p = paths or ProjectPaths.default()
    return p.root / "data" / "dataset_yolo_cls"


def _iter_labeled_images(
    split: Literal["train", "val", "all"],
    *,
    paths: ProjectPaths | None = None,
) -> Iterator[tuple[Path, str]]:
    root = _dataset_yolo_root(paths)
    splits: list[str] = ["train", "val"] if split == "all" else [split]
    for sp in splits:
        d = root / sp
        if not d.is_dir():
            continue
        for cls_dir in sorted(
            x for x in d.iterdir() if x.is_dir() and not x.name.startswith(".")
        ):
            label = cls_dir.name
            for f in sorted(cls_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in _EXT:
                    yield f, label


def _list_by_split(
    split: Literal["train", "val", "all"],
    *,
    paths: ProjectPaths | None = None,
) -> tuple[list[Path], list[str]]:
    fp: list[Path] = []
    y: list[str] = []
    for pth, lab in _iter_labeled_images(split, paths=paths):
        fp.append(pth)
        y.append(lab)
    return fp, y


def cmd_build(ap: argparse.Namespace) -> None:
    paths = ProjectPaths.default()
    src_root = paths.data_raw_products
    out_root = paths.root / "data" / "dataset_yolo_cls"
    train_dir = out_root / "train"
    val_dir = out_root / "val"
    if not src_root.is_dir():
        raise SystemExit(f"Kaynak klasör yok (önce fotoğraf ekleyin): {src_root}")
    random.seed(ap.seed)
    for d in [train_dir, val_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    class_dirs = [p for p in src_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not class_dirs:
        raise SystemExit(f"{src_root} altında ürün klasörü yok (her SKU için bir klasör).")
    n_images = 0
    had_single = False
    for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
        cls = class_dir.name
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in _EXT]
        random.shuffle(files)
        if not files:
            continue
        if len(files) == 1:
            had_single = True
            train_files = list(files)
            val_files = list(files)
        else:
            n_val = int(len(files) * ap.val_ratio)
            n_val = min(max(n_val, 1), len(files) - 1)
            val_files = files[:n_val]
            train_files = files[n_val:]
        for subset, flist in ("train", train_files), ("val", val_files):
            if subset == "val" and not flist:
                continue
            base = train_dir if subset == "train" else val_dir
            tdir = base / cls
            tdir.mkdir(parents=True, exist_ok=True)
            for f in flist:
                shutil.copy2(f, tdir / f.name)
                n_images += 1
    if n_images == 0:
        raise SystemExit("Hiç görüntü kopyalanmadı.")
    print("Sınıflar:", ", ".join(sorted(p.name for p in train_dir.iterdir() if p.is_dir())))
    print("Çıktı klasörü:", out_root, "| dosya sayısı:", n_images)
    if had_single:
        print("Not: Bazı sınıflarda tek foto var — doğrulama metriği yapay olabilir; mümkünse çoğaltın.")


def cmd_train(ap: argparse.Namespace) -> None:
    paths = ProjectPaths.default()
    data_dir = paths.root / "data" / "dataset_yolo_cls"
    train_p = data_dir / "train"
    if not train_p.is_dir() or not any(train_p.iterdir()):
        raise SystemExit("Önce: python scripts/tools.py build")
    paths.models_exports.mkdir(parents=True, exist_ok=True)
    out_weights = paths.models_exports / "yolov8_product.pt"
    if ap.dry_run:
        print("Veri:", data_dir, "| çıktı:", out_weights)
        return
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Ultralytics kurulu degil: pip install ultralytics") from e
    model = YOLO(ap.model)
    model.train(
        data=str(data_dir),
        epochs=ap.epochs,
        imgsz=ap.imgsz,
        project=str(paths.experiments),
        name="yolo_cls_products",
        exist_ok=True,
    )
    runs_dir = paths.experiments / "yolo_cls_products"
    wpt = runs_dir / "weights" / "best.pt"
    cand = [wpt] if wpt.is_file() else sorted(
        runs_dir.rglob("weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not cand:
        raise SystemExit(f"Egitim ciktisinda best.pt bulunamadi: {runs_dir}")
    shutil.copy2(cand[0], out_weights)
    print("Model kaydedildi:", out_weights)


def cmd_eval_yolo(ap: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix

    paths = ProjectPaths.default()
    wpath = Path(ap.weights) if ap.weights else paths.models_exports / "yolov8_product.pt"
    if not wpath.is_file():
        raise SystemExit(f"Agirlik dosyasi yok: {wpath}")
    val_files: list[Path] = []
    y_true: list[str] = []
    for p, lab in _iter_labeled_images("val", paths=paths):
        val_files.append(p)
        y_true.append(lab)
    if not val_files:
        raise SystemExit("Dogrulama klasoru bos. Once: python scripts/tools.py build")
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit("Ultralytics kurulu degil: pip install ultralytics") from e
    model = YOLO(str(wpath))
    y_pred: list[str] = []
    for p in val_files:
        r = model.predict(str(p), verbose=False)[0]
        if r.probs is None:
            raise SystemExit("Bu checkpoint siniflandirma degil; tools.py train ile cls egitin.")
        y_pred.append(str(model.names[int(r.probs.top1)]))
    labels = sorted(set(y_true) | set(y_pred))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out_dir = paths.experiments / "evaluation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6, len(labels) * 0.5), max(5, len(labels) * 0.45))
    )
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Gercek sinif")
    ax.set_xlabel("Tahmin")
    ax.set_title(f"YOLO cls - {wpath.name}")
    fig.tight_layout()
    png = out_dir / "yolo_val_confusion_matrix.png"
    fig.savefig(png, dpi=120)
    plt.close(fig)
    pd.DataFrame({"path": [str(p) for p in val_files], "y_true": y_true, "y_pred": y_pred}).to_csv(
        out_dir / "yolo_val_predictions.csv", index=False
    )
    print("Gorsel kaydedildi:", png)


def cmd_cv_search(ap: argparse.Namespace) -> None:
    from scipy.stats import loguniform
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (
        GridSearchCV,
        RandomizedSearchCV,
        StratifiedKFold,
        cross_val_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    paths = ProjectPaths.default()
    fps, y_str = _list_by_split("all", paths=paths)
    if len(fps) < 4:
        raise SystemExit("Cok az ornek. Once: python scripts/tools.py build")
    print("ResNet18 ile goruntu vektorleri cikariliyor...")
    X = embed_paths(fps, batch_size=16)
    y_idx, class_names = label_enc(y_str)
    _, counts = np.unique(y_idx, return_counts=True)
    n_splits = max(2, min(ap.cv_splits, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = Pipeline(
        [
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, random_state=42, solver="lbfgs")),
        ]
    )
    cv_acc = cross_val_score(pipe, X, y_idx, cv=cv, scoring="accuracy", n_jobs=None)
    print(f"Capraz dogrulama (k={n_splits}) dogruluk:", cv_acc, "ortalama:", cv_acc.mean())
    rnd = RandomizedSearchCV(
        pipe,
        param_distributions={"lr__C": loguniform(1e-3, 1e3)},
        n_iter=ap.random_iters,
        cv=cv,
        scoring="accuracy",
        random_state=42,
        n_jobs=None,
        verbose=1,
    )
    rnd.fit(X, y_idx)
    print("Rastgele arama en iyi:", rnd.best_params_, "skor:", rnd.best_score_)
    grid = GridSearchCV(
        pipe,
        param_grid={"lr__C": [0.01, 0.1, 1.0, 10.0, 100.0]},
        cv=cv,
        scoring="accuracy",
        n_jobs=None,
        verbose=1,
    )
    grid.fit(X, y_idx)
    print("Izgara arama en iyi:", grid.best_params_, "skor:", grid.best_score_)
    out_dir = paths.experiments / "evaluation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = json_clean(
        {
            "n_samples": len(fps),
            "n_classes": len(class_names),
            "class_names": class_names,
            "cv_splits": n_splits,
            "baseline_cv_accuracy_mean": float(cv_acc.mean()),
            "baseline_cv_accuracy_std": float(cv_acc.std()),
            "random_search_best": {"params": rnd.best_params_, "accuracy": float(rnd.best_score_)},
            "grid_search_best": {"params": grid.best_params_, "accuracy": float(grid.best_score_)},
        }
    )
    (out_dir / "sklearn_cv_search_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Ozet JSON:", out_dir / "sklearn_cv_search_summary.json")


def cmd_prototype(ap: argparse.Namespace) -> None:
    from sklearn.metrics import classification_report

    paths = ProjectPaths.default()
    train_fp, train_y = _list_by_split("train", paths=paths)
    val_fp, val_y = _list_by_split("val", paths=paths)
    if not train_fp or not val_fp:
        raise SystemExit("Egitim veya dogrulama klasoru bos.")
    print("Train ve val vektorleri...")
    X_tr = embed_paths(train_fp, batch_size=16)
    X_va = embed_paths(val_fp, batch_size=16)

    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)

    X_tr_n = _norm(X_tr)
    X_va_n = _norm(X_va)
    classes = sorted(set(train_y))
    protos = []
    for c in classes:
        idx = [i for i, t in enumerate(train_y) if t == c]
        mv = X_tr_n[idx].mean(axis=0)
        protos.append(mv / (np.linalg.norm(mv) + 1e-12))
    P = np.stack(protos, axis=0)
    pred = [classes[j] for j in (X_va_n @ P.T).argmax(axis=1)]
    print(classification_report(val_y, pred, labels=classes, zero_division=0))
    out_dir = paths.experiments / "evaluation_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    acc = float(np.mean(np.array(pred) == np.array(val_y)))
    meta = {
        "method": "prototype_centroid_cosine",
        "backbone": "resnet18_imagenet",
        "n_train": len(train_fp),
        "n_val": len(val_fp),
        "classes": classes,
        "val_accuracy": acc,
    }
    (out_dir / "prototype_fewshot_summary.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Ozet JSON:", out_dir / "prototype_fewshot_summary.json")


def main() -> None:
    p = argparse.ArgumentParser(description="Gorsel urun projesi - komut satiri araclari")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Ham urun klasorlerinden YOLO veri seti uret")
    b.add_argument("--val-ratio", type=float, default=0.2)
    b.add_argument("--seed", type=int, default=42)
    b.set_defaults(func=cmd_build)

    t = sub.add_parser("train", help="YOLOv8 siniflandirma egit, agirligi exports altina kopyala")
    t.add_argument("--epochs", type=int, default=40)
    t.add_argument("--imgsz", type=int, default=224)
    t.add_argument("--model", type=str, default="yolov8n-cls.pt")
    t.add_argument("--dry-run", action="store_true")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval-yolo", help="val metrigi + karmasiklik matrisi")
    e.add_argument("--weights", type=str, default="")
    e.set_defaults(func=cmd_eval_yolo)

    c = sub.add_parser("cv-search", help="ResNet + LR: capraz dogrulama ve hiperparametre aramasi")
    c.add_argument("--cv-splits", type=int, default=5)
    c.add_argument("--random-iters", type=int, default=16)
    c.set_defaults(func=cmd_cv_search)

    f = sub.add_parser("prototype", help="Prototip merkezleri ile az-ornekli siniflandirma")
    f.set_defaults(func=cmd_prototype)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
