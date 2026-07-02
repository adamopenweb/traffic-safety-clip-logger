"""Unsafe-behavior rules.

Rules consume track histories and emit candidate events. They are kept
independent of YOLO/detector details (spec: "Keep rule engine independent of
YOLO details"). The event manager decides which candidates become saved clips.
"""
