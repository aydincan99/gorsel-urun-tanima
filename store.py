"""Sqlite magaza + yol sabitleri. Tablolari assagida stringde tutuyorum gibi bi sey."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'admin')),
    full_name TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', '+3 hours'))
);
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER NOT NULL,
    cal_100ml REAL,
    carbs_g_100ml REAL,
    sugar_g_100ml REAL,
    yolo_class_hint TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', '+3 hours'))
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    total REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'simulated_paid',
    created_at TEXT NOT NULL DEFAULT (datetime('now', '+3 hours')),
    FOREIGN KEY (user_id) REFERENCES users (id)
);
CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    unit_price REAL NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
);
CREATE TABLE IF NOT EXISTS admin_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', '+3 hours')),
    kind TEXT NOT NULL,
    message TEXT,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS scan_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL DEFAULT (datetime('now', '+3 hours')),
    user_id INTEGER,
    product_id INTEGER,
    yolo_label TEXT,
    yolo_conf REAL,
    ocr_text TEXT,
    ocr_conf REAL,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users (id),
    FOREIGN KEY (product_id) REFERENCES products (id)
);
"""


def find_project_root(start: Path | None = None) -> Path:
    if start == None:
        anchor = Path(__file__).resolve()
    else:
        anchor = start.resolve()
    if anchor.is_dir():
        cur = anchor
    else:
        cur = anchor.parent
    parents_list = [cur, *cur.parents]
    for parent in parents_list:
        req = parent / "requirements.txt"
        if req.exists():
            return parent
    kk = len(cur.parents) - 1
    if kk < 2:
        idx = kk
    else:
        idx = 2
    return cur.parents[idx]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_raw_products: Path
    data_raw_images: Path
    data_raw_receipts: Path
    data_processed: Path
    data_annotations: Path
    data_manifests: Path
    models_checkpoints: Path
    models_exports: Path
    experiments: Path

    @classmethod
    def default(cls) -> ProjectPaths:
        root = find_project_root()
        data, models, raw = root / "data", root / "models", root / "data" / "raw"
        return cls(
            root=root,
            data_raw=raw,
            data_raw_products=raw / "products",
            data_raw_images=raw / "images",
            data_raw_receipts=raw / "receipts",
            data_processed=data / "processed",
            data_annotations=data / "annotations",
            data_manifests=data / "manifests",
            models_checkpoints=models / "checkpoints",
            models_exports=models / "exports",
            experiments=root / "experiments",
        )


_TR = ZoneInfo("Europe/Istanbul")
_ITER = 120_000


def now_tr_sqlite() -> str:
    return datetime.now(_TR).strftime("%Y-%m-%d %H:%M:%S")


def format_tr_display(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        if "T" in s:
            s = s.replace("T", " ", 1)
        if len(s) >= 19:
            dt = datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%d.%m.%Y %H:%M")
    except ValueError:
        pass
    return s


def db_path() -> Path:
    p = ProjectPaths.default().data_processed
    p.mkdir(parents=True, exist_ok=True)
    return p / "app.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _cx() -> Iterator[sqlite3.Connection]:  # her seferde ac kapa, hizli degil ama anlasilir
    c = get_connection()
    try:
        yield c
    finally:
        c.close()


def init_db(with_demo_orders: bool = True) -> None:
    with _cx() as conn:
        conn.executescript(_SCHEMA)  # butun CREATEler tek seferde
        _seed_if_empty(conn)
        if with_demo_orders:
            _ensure_demo_order_history(conn)
        conn.commit()


def _seed_if_empty(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"] > 0:
        return
    for email, password, role, name in (
        ("admin@demo.local", "Admin123!", "admin", "Demo Yönetici"),
        ("user@demo.local", "User123!", "user", "Demo Müşteri"),
    ):
        salt, ph = hash_password(password)
        conn.execute(
            "INSERT INTO users (email, password_hash, salt, role, full_name, created_at) VALUES (?,?,?,?,?,?)",
            (email.lower(), ph, salt, role, name, now_tr_sqlite()),
        )
    for row in (
        ("SKU-CC-330", "Coca-Cola 330ml", 12.90, 48, 42.0, 10.6, 10.6, "coca_cola"),
        ("SKU-SP-330", "Sprite 330ml", 11.50, 52, 39.0, 10.0, 9.0, "sprite"),
        ("SKU-AYRAN", "Ayran 250ml", 8.90, 30, 55.0, 4.5, 3.5, "ayran"),
    ):
        conn.execute(
            """INSERT INTO products
            (sku, name, price, stock, cal_100ml, carbs_g_100ml, sugar_g_100ml, yolo_class_hint, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (*row, now_tr_sqlite()),
        )
    conn.execute(
        "INSERT INTO admin_logs (kind, message, payload, created_at) VALUES ('system', ?, NULL, ?)",
        ("Veritabani ilk kurulum ve demo hesaplar olusturuldu.", now_tr_sqlite()),
    )


def _ensure_demo_order_history(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"] > 0:
        return
    uid_by_email = {
        r["email"]: r["id"]
        for r in conn.execute(
            "SELECT id, email FROM users WHERE email IN (?, ?)",
            ("admin@demo.local", "user@demo.local"),
        )
    }
    if len(uid_by_email) < 2:
        return
    aid = uid_by_email.get("admin@demo.local")
    uid = uid_by_email.get("user@demo.local")
    if aid is None or uid is None:
        return
    plist = list(conn.execute("SELECT id, price FROM products ORDER BY id LIMIT 3").fetchall())
    if not plist:
        return

    def line_items(spec: list[tuple[int, int]]) -> list[tuple[int, float, int]]:
        out: list[tuple[int, float, int]] = []
        for pi, qty in spec:
            pr = plist[min(pi, len(plist) - 1)]
            out.append((int(pr["id"]), float(pr["price"]), qty))
        return out

    def ins_order(user_i: int, when: datetime, items: list[tuple[int, float, int]]) -> None:
        tot = round(sum(u * q for _, u, q in items), 2)
        ts = when.strftime("%Y-%m-%d %H:%M:%S")
        cur = conn.execute(
            "INSERT INTO orders (user_id, total, status, created_at) VALUES (?,?,?,?)",
            (user_i, tot, "simulated_paid", ts),
        )
        oid = cur.lastrowid
        for pid, unit, qty in items:
            conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?,?,?,?)",
                (oid, pid, qty, unit),
            )

    now = datetime.now(_TR)
    specs = (
        ("user", 117, 10, 12, ((0, 2),)),
        ("admin", 114, 15, 48, ((0, 1), (2, 2))),
        ("user", 109, 9, 27, ((2, 3),)),
        ("admin", 103, 18, 5, ((1, 2),)),
        ("user", 96, 11, 41, ((0, 1), (1, 1))),
        ("admin", 91, 14, 19, ((2, 4),)),
        ("user", 85, 16, 33, ((1, 3),)),
        ("admin", 79, 10, 8, ((0, 2),)),
        ("user", 72, 13, 52, ((2, 1), (0, 1))),
        ("admin", 66, 19, 22, ((1, 1), (2, 1))),
        ("user", 61, 9, 45, ((0, 3),)),
        ("admin", 55, 12, 14, ((2, 2),)),
        ("user", 49, 17, 37, ((1, 2), (0, 1))),
        ("admin", 43, 11, 29, ((0, 1),)),
        ("user", 38, 14, 6, ((2, 3),)),
        ("admin", 32, 10, 55, ((1, 4),)),
        ("user", 26, 16, 18, ((0, 2), (2, 1))),
        ("admin", 20, 13, 44, ((1, 1), (2, 2))),
        ("user", 15, 9, 21, ((2, 2),)),
        ("admin", 10, 18, 3, ((0, 2),)),
        ("user", 5, 12, 39, ((1, 1), (0, 2))),
        ("admin", 2, 15, 11, ((2, 1), (1, 1))),
    )
    for who, da, hh, mm, prod in specs:
        w = aid if who == "admin" else uid
        t = datetime.combine((now - timedelta(days=da)).date(), time(hour=hh, minute=mm), tzinfo=_TR)
        ins_order(w, t, line_items(list(prod)))


def hash_password(plain: str, salt: bytes | None = None) -> tuple[str, str]:
    if salt == None:
        salt = secrets.token_bytes(16)
    pw_bytes = plain.encode("utf-8")
    dk = hashlib.pbkdf2_hmac("sha256", pw_bytes, salt, _ITER)
    s_txt = base64.b64encode(salt).decode("ascii")
    h_txt = base64.b64encode(dk).decode("ascii")
    return s_txt, h_txt


def verify_password(plain: str, salt_b64: str, hash_b64: str) -> bool:
    salt = base64.b64decode(salt_b64.encode("ascii"))
    expected = base64.b64decode(hash_b64.encode("ascii"))
    dk = hashlib.pbkdf2_hmac("sha256", plain.encode("utf-8"), salt, _ITER)
    return secrets.compare_digest(dk, expected)


def register_user(
    email: str,
    password: str,
    full_name: str,
    *,
    as_admin: bool = False,
) -> tuple[bool, str]:
    email = email.strip().lower()
    if not email or "@" not in email:
        return False, "Geçerli bir e-posta girin."
    if len(password) < 6:
        return False, "Şifre en az 6 karakter olmalı."
    salt, ph = hash_password(password)
    role = "admin" if as_admin else "user"
    try:
        with _cx() as conn:
            conn.execute(
                "INSERT INTO users (email, password_hash, salt, role, full_name, created_at) VALUES (?,?,?,?,?,?)",
                (email, ph, salt, role, full_name.strip() or email.split("@")[0], now_tr_sqlite()),
            )
            conn.commit()
        return True, "Kayıt tamamlandı."
    except sqlite3.IntegrityError:
        return False, "Bu e-posta zaten kayıtlı."


def login_user(email: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    email = email.strip().lower()
    with _cx() as conn:
        cur = conn.execute(
            "SELECT id, email, password_hash, salt, role, full_name FROM users WHERE email = ?",
            (email,),
        )
        row = cur.fetchone()
    if row is None:
        return False, "E-posta veya şifre hatalı.", None
    if not verify_password(password, row["salt"], row["password_hash"]):
        return False, "E-posta veya şifre hatalı.", None
    return True, "Giriş başarılı.", {
        "id": row["id"],
        "email": row["email"],
        "role": row["role"],
        "full_name": row["full_name"],
    }


def list_all() -> list[sqlite3.Row]:
    with _cx() as conn:
        return conn.execute("SELECT * FROM products ORDER BY name").fetchall()


def get_by_id(pid: int) -> sqlite3.Row | None:
    with _cx() as conn:
        return conn.execute("SELECT * FROM products WHERE id = ?", (pid,)).fetchone()


def search(term: str) -> list[sqlite3.Row]:
    t = f"%{term.strip()}%"
    with _cx() as conn:
        return conn.execute(
            "SELECT * FROM products WHERE name LIKE ? OR sku LIKE ? OR yolo_class_hint LIKE ?",
            (t, t, t),
        ).fetchall()


def adjust_stock(product_id: int, delta: int) -> None:
    with _cx() as conn:
        conn.execute(
            """UPDATE products SET stock = CASE
                WHEN stock + ? < 0 THEN 0 ELSE stock + ? END
                WHERE id = ?""",
            (delta, delta, product_id),
        )
        conn.commit()


def delete_product(product_id: int) -> None:
    with _cx() as conn:
        conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()


def insert_product(
    sku: str,
    name: str,
    price: float,
    stock: int,
    *,
    cal_100ml: float | None = None,
    carbs_g_100ml: float | None = None,
    sugar_g_100ml: float | None = None,
    yolo_class_hint: str | None = None,
) -> int:
    with _cx() as conn:
        cur = conn.execute(
            """INSERT INTO products
            (sku, name, price, stock, cal_100ml, carbs_g_100ml, sugar_g_100ml, yolo_class_hint, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)""",
            (sku, name, price, stock, cal_100ml, carbs_g_100ml, sugar_g_100ml, yolo_class_hint, now_tr_sqlite()),
        )
        conn.commit()
        return int(cur.lastrowid)


def row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
    d: dict[str, Any] = {}
    for k in r.keys():
        d[k] = r[k]
    return d


def format_receipt_text(receipt: dict[str, Any]) -> str:
    lines: list[str] = [
        "================================",
        "      DİJİTAL FİŞ (SİMÜLASYON)",
        "================================",
        f"Sipariş no : #{receipt['order_id']}",
        f"Tarih/saat  : {receipt.get('issued_at_display', receipt.get('issued_at', ''))}",
        f"Müşteri     : {receipt['customer_name']}",
        "",
    ]
    for row in receipt["lines"]:
        lines.append(
            f"  {row['name']}\n    {row['qty']} adet × {row['unit']:.2f} TL = {row['subtotal']:.2f} TL"
        )
    lines += [
        "",
        "--------------------------------",
        f"  TOPLAM (KDV dahil değildir): {receipt['total']:.2f} TL",
        "================================",
        "Teşekkür ederiz.",
        "(Gerçek ödeme alınmamıştır — bitirme demosu)",
    ]
    return "\n".join(lines)


def simulate_checkout(  # gercek odeme yok sadece stok dusuruyo
    user_id: int,
    user_name: str,
    cart: list[dict[str, Any]],
) -> tuple[bool, str, dict[str, Any] | None]:
    if not cart:
        return False, "Sepet boş.", None
    with _cx() as conn:
        total = 0.0
        basket: list[tuple[int, int, float]] = []
        for item in cart:
            pid, qty = int(item["product_id"]), int(item.get("qty", 1))
            price = float(item["price"])
            row = conn.execute("SELECT stock FROM products WHERE id = ?", (pid,)).fetchone()
            if row is None or row["stock"] < qty:
                return False, f"Yetersiz stok veya ürün yok: {item.get('name', pid)}", None
            total += price * qty
            basket.append((pid, qty, price))
        cur = conn.execute(
            "INSERT INTO orders (user_id, total, status, created_at) VALUES (?,?,?,?)",
            (user_id, round(total, 2), "simulated_paid", now_tr_sqlite()),
        )
        oid = cur.lastrowid
        for pid, qty, unit_price in basket:
            conn.execute(
                "INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES (?,?,?,?)",
                (oid, pid, qty, unit_price),
            )
            conn.execute("UPDATE products SET stock = stock - ? WHERE id = ?", (qty, pid))
        payload = {"order_id": oid, "user_id": user_id, "user_name": user_name, "total": round(total, 2), "items": cart}
        conn.execute(
            "INSERT INTO admin_logs (kind, message, payload, created_at) VALUES (?,?,?,?)",
            (
                "payment_simulation",
                f"Simüle ödeme: {user_name} — {total:.2f} TL",
                json.dumps(payload, ensure_ascii=False),
                now_tr_sqlite(),
            ),
        )
        conn.commit()
        issued = now_tr_sqlite()
        receipt_lines = []
        for item in cart:
            adet = int(item.get("qty", 1))
            birim = float(item["price"])
            ara = round(birim * adet, 2)
            receipt_lines.append(
                {
                    "name": str(item.get("name", "")),
                    "qty": adet,
                    "unit": birim,
                    "subtotal": ara,
                }
            )
        receipt: dict[str, Any] = {
            "order_id": oid,
            "customer_name": user_name,
            "issued_at": issued,
            "issued_at_display": format_tr_display(issued),
            "lines": receipt_lines,
            "total": round(total, 2),
        }
        return True, f"Ödeme tamamlandı. Sipariş #{oid} — {round(total, 2)} TL (simülasyon)", receipt


def list_user_orders(user_id: int, limit: int = 25) -> list:
    with _cx() as conn:
        return conn.execute(
            "SELECT id, total, status, created_at FROM orders WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()


def admin_append(kind: str, message: str, payload: str | None = None) -> None:
    with _cx() as conn:
        conn.execute(
            "INSERT INTO admin_logs (kind, message, payload, created_at) VALUES (?,?,?,?)",
            (kind, message, payload, now_tr_sqlite()),
        )
        conn.commit()


def admin_list_by_kind(kind: str, limit: int = 50) -> list[sqlite3.Row]:
    with _cx() as conn:
        return conn.execute(
            "SELECT * FROM admin_logs WHERE kind = ? ORDER BY id DESC LIMIT ?",
            (kind, limit),
        ).fetchall()


def log_scan(
    *,
    user_id: int | None,
    product_id: int | None,
    yolo_label: str,
    yolo_conf: float,
    ocr_text: str,
    ocr_conf: float,
    notes: str = "",
) -> None:
    with _cx() as conn:
        conn.execute(
            """INSERT INTO scan_logs
            (user_id, product_id, yolo_label, yolo_conf, ocr_text, ocr_conf, notes, created_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (user_id, product_id, yolo_label, yolo_conf, ocr_text, ocr_conf, notes, now_tr_sqlite()),
        )
        conn.commit()


def scan_list_recent(limit: int = 80) -> list[sqlite3.Row]:
    with _cx() as conn:
        return conn.execute("SELECT * FROM scan_logs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
