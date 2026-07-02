"""Vehicle detection.

The detector is kept abstract so the rest of the pipeline never depends on
YOLO/Ultralytics specifics (spec: "Keep detector interface abstract"). Every
detector returns a ``supervision.Detections`` so tracking, annotation, and
rules consume one common type.

* :class:`YoloDetector` wraps Ultralytics YOLO and filters to vehicle classes.
  It lazily imports ultralytics + supervision so this module imports cleanly
  without the (torch-heavy) detection stack.
* :class:`ScriptedDetector` emits predetermined detections frame-by-frame. It
  needs only supervision/numpy (no torch), which lets the whole offline
  pipeline — real ByteTrack + real debug video — run and be tested without a
  GPU or a YOLO model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Sequence, Tuple


class MissingDetectorDependency(ImportError):
    """Raised when a detector's backend (ultralytics/torch) is unavailable."""


class Detector(ABC):
    """Abstract detector: a frame in, ``supervision.Detections`` out."""

    @abstractmethod
    def detect(self, frame: Any) -> Any:
        """Run detection on a single BGR frame and return sv.Detections."""
        raise NotImplementedError


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to cuda/cpu; pass through explicit values."""
    if device and device != "auto":
        return device
    try:
        import torch  # noqa: WPS433 - optional, only present with the gpu extra

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class YoloDetector(Detector):
    """Ultralytics YOLO detector returning vehicle-class detections."""

    def __init__(self, config) -> None:
        try:
            from ultralytics import YOLO
            import supervision as sv  # noqa: F401 - ensure available for detect()
        except ImportError as exc:  # pragma: no cover - exercised in core-only envs
            raise MissingDetectorDependency(
                "YoloDetector requires the 'analyze'/'gpu' extra "
                "(ultralytics + supervision + torch)."
            ) from exc

        models = config.models
        analysis = config.analysis
        self._sv = sv
        self.model_name = models.get("yolo_model", "yolov8n.pt")
        self.confidence = float(models.get("confidence_threshold", 0.35))
        self.iou = float(models.get("iou_threshold", 0.5))
        self.imgsz = int(analysis.get("inference_input_size", 640))
        self.device = _resolve_device(str(analysis.get("device", "auto")))
        self.vehicle_classes = {c.lower() for c in models.get("vehicle_classes", [])}
        self.model = YOLO(self.model_name)

    def detect(self, frame: Any) -> Any:
        result = self.model.predict(
            frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]
        detections = self._sv.Detections.from_ultralytics(result)
        return self._filter_vehicles(detections)

    def _filter_vehicles(self, detections: Any) -> Any:
        if not self.vehicle_classes:
            return detections
        names = detections.data.get("class_name") if detections.data else None
        if names is None:
            return detections
        import numpy as np

        mask = np.array(
            [str(n).lower() in self.vehicle_classes for n in names], dtype=bool
        )
        return detections[mask]


# A frame's scripted detections: list of (x1, y1, x2, y2, confidence, class_id).
ScriptedFrame = Sequence[Tuple[float, float, float, float, float, int]]


class ScriptedDetector(Detector):
    """Returns predetermined detections per call (ignores pixel content).

    Useful for deterministic tests and for exercising the full pipeline without
    YOLO. Each ``detect`` call advances to the next scripted frame; once the
    script is exhausted it yields empty detections.
    """

    def __init__(self, frames: Sequence[ScriptedFrame]) -> None:
        import numpy as np  # noqa: F401 - needed to build Detections
        import supervision as sv

        self._np = np
        self._sv = sv
        self._frames: List[ScriptedFrame] = [list(f) for f in frames]
        self._i = 0

    def detect(self, frame: Any = None) -> Any:
        boxes = self._frames[self._i] if self._i < len(self._frames) else []
        self._i += 1
        np, sv = self._np, self._sv
        if not boxes:
            return sv.Detections.empty()
        xyxy = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=float)
        confidence = np.array([b[4] for b in boxes], dtype=float)
        class_id = np.array([b[5] for b in boxes], dtype=int)
        return sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)


_DETECTOR_TYPES = {"yolov8", "yolo", "yolov5", "yolov9", "ultralytics"}


def build_detector(config, detector: Optional[Detector] = None) -> Detector:
    """Construct the detector named by ``config.models.detector_type``.

    ``detector`` may be passed to inject a custom/fake detector (tests, demos).
    """
    if detector is not None:
        return detector
    detector_type = str(config.models.get("detector_type", "yolov8")).lower()
    if detector_type in _DETECTOR_TYPES:
        return YoloDetector(config)
    raise ValueError(f"Unknown detector_type: {detector_type!r}")
