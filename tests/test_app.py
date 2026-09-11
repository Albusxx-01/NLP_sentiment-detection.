"""Smoke tests for the serving app (require a trained checkpoint)."""

import pytest

from src.config import ROOT, load_config

CFG = load_config()
CHECKPOINT_DIR = ROOT / CFG["model"]["checkpoint_dir"]

requires_model = pytest.mark.skipif(
    not (CHECKPOINT_DIR / "config.json").exists(),
    reason="No trained checkpoint found; run training first",
)


@requires_model
def test_demo_imports() -> None:
    import src.app.demo  # noqa: F401


@requires_model
def test_api_health_and_predict_shape() -> None:
    from src.app.api import app

    assert app.title
    assert [r.path for r in app.routes if r.path == "/predict"]
    assert [r.path for r in app.routes if r.path == "/health"]


def test_api_schema_valid() -> None:
    from src.app.api import PredictionRequest, PredictionResponse

    req = PredictionRequest(headline="test headline")
    assert req.headline == "test headline"
    resp = PredictionResponse(label=1, label_name="sarcastic", confidence=0.9)
    assert resp.label == 1
