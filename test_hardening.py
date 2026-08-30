from io import BytesIO

from hardening import login_allowed, login_fail, sanitize_text, validate_upload


def test_sanitize_and_login_limit():
    assert sanitize_text("ab\x00c") == "abc"
    key = "rate@test.com"
    for _ in range(8):
        assert login_allowed(key)
        login_fail(key)
    assert login_allowed(key) is False


def test_upload_rejects_text():
    class Fake:
        name = "a.txt"

        def getvalue(self):
            return b"not-an-image"

    try:
        validate_upload(Fake())
        raise AssertionError("should reject")
    except ValueError:
        pass
