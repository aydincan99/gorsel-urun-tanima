"""pytest — store.py icin 2 test, fazla yazmadim."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.pop("MONGO_URI", None)  # testte mongo istemiyorum

@pytest.fixture()
def temp_db(monkeypatch):
    import store as store_module

    tmp = tempfile.mkdtemp()
    dbf = Path(tmp) / "t.db"

    def _path():
        return dbf

    monkeypatch.setattr(store_module, "db_path", _path)
    store_module.init_db()
    yield store_module
    try:
        dbf.unlink(missing_ok=True)
    except OSError:
        pass


def test_hash_roundtrip(temp_db):
    s, h = temp_db.hash_password("secret123")
    assert temp_db.verify_password("secret123", s, h)
    assert not temp_db.verify_password("wrong", s, h)


def test_register_and_product(temp_db):
    ok, mesaj = temp_db.register_user("a@b.com", "pass1234", "Ali", as_admin=False)
    assert ok == True
    _ = mesaj
    pid = temp_db.insert_product("SKU-X", "Test", 1.0, 5, yolo_class_hint="x")
    row = temp_db.get_by_id(pid)
    assert row["sku"] == "SKU-X"


def test_login_lockout(temp_db):
    temp_db.register_user("lock@b.com", "pass1234", "Ali")
    for _ in range(8):
        ok, _, _ = temp_db.login_user("lock@b.com", "yanlis")
        assert ok is False
    ok, msg, _ = temp_db.login_user("lock@b.com", "pass1234")
    assert ok is False
    assert "deneme" in msg.lower()


def test_delete_account_removes_user(temp_db):
    temp_db.register_user("sil@b.com", "pass1234", "Ali")
    ok, _, user = temp_db.login_user("sil@b.com", "pass1234")
    assert ok and user
    temp_db.delete_user_account(int(user["id"]))
    assert temp_db.get_user(int(user["id"])) is None


def test_three_roles_seeded(temp_db):
    emails = {r["email"]: r["role"] for r in temp_db.get_connection().execute("SELECT email, role FROM users")}
    assert emails.get("admin@demo.local") == "admin"
    assert emails.get("staff@demo.local") == "staff"
    assert emails.get("user@demo.local") == "user"