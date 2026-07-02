"""Tests for live-source detection (the rest of live.py needs a real stream)."""

from __future__ import annotations

from traffic_logger.analyze.live import is_stream_source


def test_stream_sources_detected():
    for s in [
        "rtsp://cam.example:554/stream",
        "rtsps://cam/stream",
        "http://host:8000/live.mjpg",
        "/dev/video0",
        "0",
        "udp://239.0.0.1:1234",
    ]:
        assert is_stream_source(s) is True


def test_file_sources_not_streams():
    for s in ["samples/street.mp4", "C:/clips/a.mov", "video.avi", "street-test.mp4"]:
        assert is_stream_source(s) is False


def test_frame_grabber_delivers_frames_and_stops():
    """The background grabber keeps the latest frame with an advancing seq."""
    import time

    from traffic_logger.analyze.live import _FrameGrabber

    class _FakeCap:
        def __init__(self):
            self._n = 0

        def isOpened(self):
            return True

        def read(self):
            self._n += 1
            time.sleep(0.001)  # mimic decode cost so the thread doesn't busy-spin
            return True, ("frame", self._n)

        def release(self):
            pass

    grab = _FrameGrabber(lambda: _FakeCap()).start()
    try:
        deadline = time.monotonic() + 2.0
        seq, frame = 0, None
        while time.monotonic() < deadline:
            seq, frame = grab.read()
            if frame is not None and seq > 0:
                break
            time.sleep(0.005)
        assert frame is not None and seq > 0
        time.sleep(0.02)
        seq2, _ = grab.read()
        assert seq2 >= seq  # thread keeps advancing the sequence
    finally:
        grab.stop()


def test_frame_grabber_stop_is_idempotent_before_start():
    from traffic_logger.analyze.live import _FrameGrabber

    grab = _FrameGrabber(lambda: None)
    grab.stop()  # never started -> must not raise
    assert grab.read() == (0, None)
