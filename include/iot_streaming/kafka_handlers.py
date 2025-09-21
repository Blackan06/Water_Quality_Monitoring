"""Reusable Kafka trigger functions for AwaitMessageTriggerFunctionSensor."""

from typing import Any


def extract_value(event=None, **kwargs):
    """
    Dùng cho AwaitMessageTriggerFunctionSensor.apply_function (bắt buộc dạng dotted-path STRING).
    Nhận record Kafka và TRẢ VỀ payload dạng string. KHÔNG trả bool.
    """
    if event is None:
        return None
    try:
        return event.value().decode("utf-8")
    except Exception:
        return str(event)

