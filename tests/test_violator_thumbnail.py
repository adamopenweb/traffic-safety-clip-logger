"""Violator thumbnail: best-frame capture (per track) + the cropped-thumbnail write."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from traffic_logger.analyze.best_frame import BestFrameCache  # noqa: E402


class _Track:
    def __init__(self, tid, bbox):
        self.track_id = tid
        self.latest_bbox = bbox


def _frame(tag):  # a tiny frame stamped with a value so we can tell copies apart
    return np.full((12, 12, 3), tag, dtype=np.uint8)


# -- BestFrameCache ----------------------------------------------------------

def test_keeps_the_largest_box_frame():
    c = BestFrameCache()
    c.observe([_Track(1, (0, 0, 10, 10))], _frame(1), 0)   # area 100
    c.observe([_Track(1, (0, 0, 20, 20))], _frame(2), 1)   # area 400 -> new best
    c.observe([_Track(1, (0, 0, 5, 5))], _frame(3), 2)     # area 25  -> keep best
    frame, bbox = c.take(1)
    assert frame[0, 0, 0] == 2                              # the largest-box frame
    assert bbox == (0.0, 0.0, 20.0, 20.0)
    assert c.take(1) is None                               # popped on take


def test_take_unknown_or_none_is_safe():
    c = BestFrameCache()
    assert c.take(99) is None
    assert c.take(None) is None


def test_evicts_idle_tracks():
    c = BestFrameCache(idle_evict_frames=30)
    c.observe([_Track(1, (0, 0, 10, 10))], _frame(1), 0)
    c.observe([_Track(2, (0, 0, 10, 10))], _frame(2), 60)  # sweep (idx%30==0): tid1 idle
    assert c.take(1) is None
    assert c.take(2) is not None


# -- cropped thumbnail -------------------------------------------------------

def test_cropped_thumbnail_is_16_9_and_min_width(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from traffic_logger.events.thumbnail import save_cropped_thumbnail

    frame = np.zeros((480, 704, 3), dtype=np.uint8)
    out = save_cropped_thumbnail(frame, (300, 200, 400, 320), tmp_path / "t.jpg", min_w=360)
    assert out is not None and out.exists()
    h, w = cv2.imread(str(out)).shape[:2]
    assert abs(w / h - 16 / 9) < 0.05       # widened to a card-friendly 16:9
    assert w >= 360                          # min width honoured


def test_cropped_thumbnail_bad_bbox_returns_none(tmp_path):
    from traffic_logger.events.thumbnail import save_cropped_thumbnail

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert save_cropped_thumbnail(frame, (50, 50, 40, 40), tmp_path / "x.jpg") is None
