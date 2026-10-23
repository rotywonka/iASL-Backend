"""Security test: POST /predict must reject oversized request bodies (CWE-400).

app.py loads a TensorFlow model and calls app.run() at import time, so the
heavy dependencies are stubbed and Flask.run is disabled before importing it.
"""
import base64
import importlib.util
import json
import os
import sys
import types

import flask
import numpy as np
import pytest

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def client(monkeypatch):
    class FakeModel:
        num_classes = 0

        def load_weights(self, path):
            pass

        def predict(self, x):
            return np.zeros((1, self.num_classes))

    models = types.ModuleType("tensorflow.keras.models")
    models.model_from_json = lambda s: FakeModel()
    cv2 = types.ModuleType("cv2")
    cv2.INTER_AREA = 3
    cv2.resize = lambda img, size, interpolation=None: np.zeros(size + (3,), np.uint8)
    jsonpickle = types.ModuleType("jsonpickle")
    jsonpickle.encode = json.dumps

    monkeypatch.setitem(sys.modules, "tensorflow", types.ModuleType("tensorflow"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras", types.ModuleType("tensorflow.keras"))
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", models)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    monkeypatch.setitem(sys.modules, "jsonpickle", jsonpickle)
    monkeypatch.setattr(flask.Flask, "run", lambda self, *a, **k: None)
    monkeypatch.chdir(APP_DIR)

    spec = importlib.util.spec_from_file_location("iasl_app", os.path.join(APP_DIR, "app.py"))
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    FakeModel.num_classes = len(app_module.MAP)
    return app_module.app.test_client()


def test_oversized_body_is_rejected_before_parsing(client):
    body = b"vid_stuff=" + b"A" * (2 * 1024 * 1024)
    resp = client.post("/predict", data=body,
                       content_type="application/x-www-form-urlencoded")
    assert resp.status_code == 413


def test_valid_clip_is_still_accepted(client):
    clip = np.zeros(40 * 50 * 50 * 3, np.uint8).tobytes()
    resp = client.post("/predict", data={"vid_stuff": base64.b64encode(clip)})
    assert resp.status_code == 200
