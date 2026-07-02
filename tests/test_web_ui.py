"""Phase C/D: the SPA shell + assets are served once unlocked, hidden before."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from traffic_logger.web.app import WebSettings, create_app  # noqa: E402

TOKEN = "ui-token"


@pytest.fixture
def client(tmp_path):
    settings = WebSettings(
        events_dir=tmp_path / "events",
        speed_log_path=str(tmp_path / "missing.sqlite"),
        timezone="America/Toronto",
        access_token=TOKEN, session_secret="s", cookie_secure=False)
    return TestClient(create_app(settings))


def test_shell_and_assets_hidden_until_unlocked(client):
    assert client.get("/").status_code == 404
    assert client.get("/static/app.js").status_code == 404
    client.get(f"/k/{TOKEN}")  # unlock
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "Traffic Watch" in r.text
    for asset in ("/static/app.js", "/static/styles.css"):
        assert client.get(asset).status_code == 200, asset
