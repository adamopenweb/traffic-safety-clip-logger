"""Phase A/D: the 404-everything gate + secret-link unlock + cookie signing."""

from __future__ import annotations

import pytest

pytest.importorskip("itsdangerous")  # auth imports it at module level (guard before that import)

from traffic_logger.web import auth  # noqa: E402

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from traffic_logger.web.app import WebSettings, create_app  # noqa: E402

TOKEN = "test-access-token-abc123"


# -- cookie + token primitives ----------------------------------------------

def test_cookie_roundtrip_and_tamper():
    signer = auth.make_signer("secret-a")
    value = auth.issue_cookie(signer)
    assert auth.cookie_valid(signer, value)
    assert not auth.cookie_valid(signer, value + "x")        # tampered
    assert not auth.cookie_valid(signer, "")                 # empty
    assert not auth.cookie_valid(auth.make_signer("secret-b"), value)  # wrong secret


def test_token_matches_constant_time():
    assert auth.token_matches("abc", "abc")
    assert not auth.token_matches("abc", "abd")
    assert not auth.token_matches("", "abc")
    assert not auth.token_matches("abc", "")


def test_new_token_is_long_and_unique():
    a, b = auth.new_token(), auth.new_token()
    assert a != b and len(a) >= 32


# -- the gate (TestClient) ---------------------------------------------------

@pytest.fixture
def client(tmp_path):
    settings = WebSettings(
        events_dir=tmp_path / "events",
        speed_log_path=str(tmp_path / "missing.sqlite"),
        timezone="America/Toronto",
        access_token=TOKEN, session_secret="sign-secret",
        unlock_prefix="k", cookie_secure=False)  # TestClient speaks http
    return TestClient(create_app(settings))


def test_everything_is_404_when_unauthenticated(client):
    for path in ["/", "/static/app.js", "/static/styles.css", "/api/now",
                 "/api/stats", "/api/stats/today", "/media/clip/x",
                 "/media/thumb/x", "/.env", "/wp-login.php", "/admin"]:
        assert client.get(path).status_code == 404, path


def test_wrong_or_empty_token_is_404(client):
    assert client.get("/k/wrong-token").status_code == 404
    assert client.get("/k/").status_code == 404


def test_unlock_link_sets_cookie_and_grants_access(client):
    r = client.get(f"/k/{TOKEN}", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/"
    set_cookie = r.headers.get("set-cookie", "")
    assert "tw=" in set_cookie and "httponly" in set_cookie.lower()
    # cookie is now in the jar -> the app is reachable
    assert client.get("/").status_code == 200
    assert client.get("/api/now").status_code == 200
    assert client.get("/static/app.js").status_code == 200


def test_logout_relocks(client):
    client.get(f"/k/{TOKEN}")              # unlock
    assert client.get("/api/now").status_code == 200
    client.post("/api/logout")
    assert client.get("/api/now").status_code == 404   # cookie cleared -> gate 404s
