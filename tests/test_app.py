"""Unit tests for the detection helpers. Only test_model loads the model."""

import shutil
import types

import cv2
import numpy as np
import onnxruntime as ort

from anima.app import (
    MIN_CONFIDENCE,
    MODEL_PATH,
    Anima,
    dest_path,
    frames,
    has_target,
    pick_folder,
    preprocess,
    same_folder,
)


def row(score, cls):
    """Build one model output row: x1, y1, x2, y2, score, class."""
    return [0, 0, 10, 10, score, cls]


def test_has_target():
    """Only target classes at or above MIN_CONFIDENCE count as a detection."""
    assert has_target(np.array([row(0.9, 0)]))  # animal, confident
    assert has_target(np.array([row(0.9, 1)]))  # person
    assert not has_target(np.array([row(MIN_CONFIDENCE - 0.01, 0)]))  # below threshold
    assert not has_target(np.array([row(0.9, 2)]))  # vehicle is not a target
    assert not has_target(np.zeros((300, 6)))  # empty model output


def test_preprocess():
    """A Full HD frame becomes a 1x3x640x640 float tensor in [0, 1]."""
    x = preprocess(np.zeros((1080, 1920, 3), np.uint8))
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert x.max() <= 1.0


def test_pick_folder(tmp_path):
    """Only a single dropped directory is accepted."""
    (tmp_path / "a.jpg").touch()
    (tmp_path / "b").mkdir()
    assert pick_folder([str(tmp_path)]) == tmp_path
    assert pick_folder([str(tmp_path / "a.jpg")]) is None
    assert pick_folder([str(tmp_path), str(tmp_path / "b")]) is None
    assert pick_folder([]) is None


def test_frames(tmp_path):
    """A 3 s video at 10 fps gives 3 frames, a corrupt image gives none."""
    video = tmp_path / "v.avi"
    out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 64))
    for _ in range(30):
        out.write(np.zeros((64, 64, 3), np.uint8))
    out.release()
    assert len(list(frames(video))) == 3
    (tmp_path / "bad.jpg").write_text("not an image")
    assert list(frames(tmp_path / "bad.jpg")) == []


def test_dest_path(tmp_path):
    """A taken name is numbered, the same file is not copied twice."""
    src, out = tmp_path / "card" / "IMG_0001.JPG", tmp_path / "out"
    src.parent.mkdir()
    out.mkdir()
    src.write_text("card1")
    assert dest_path(src, out) == out / "IMG_0001.JPG"
    shutil.copy2(src, out)
    assert dest_path(src, out) is None  # rerun: already there
    src.write_text("another card")
    assert dest_path(src, out) == out / "IMG_0001 (1).JPG"


def test_detect_file_skips_bad_files(tmp_path):
    """A zero-byte image and a 2000x1 image are skipped, not raised."""
    # run is never reached: both files fail before the model is called
    session = types.SimpleNamespace(run=None)
    app = types.SimpleNamespace(running=True, session=session)
    (tmp_path / "empty.jpg").touch()
    assert Anima.detect_file(app, tmp_path / "empty.jpg") is False
    cv2.imwrite(str(tmp_path / "thin.png"), np.zeros((1, 2000, 3), np.uint8))
    assert Anima.detect_file(app, tmp_path / "thin.png") is False


def test_model():
    """The bundled model loads and matches what the app feeds and reads."""
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    blank = preprocess(np.full((480, 640, 3), 114, np.uint8))
    det = session.run(None, {"images": blank})[0][0]
    assert det.ndim == 2 and det.shape[1] == 6  # x1, y1, x2, y2, score, class
    assert not has_target(det)  # a plain gray frame holds nothing


def test_same_folder(tmp_path):
    """Different spellings of one folder match, a subfolder does not."""
    (tmp_path / "b").mkdir()
    assert same_folder(tmp_path, tmp_path / "b" / "..")
    assert not same_folder(tmp_path, tmp_path / "b")
