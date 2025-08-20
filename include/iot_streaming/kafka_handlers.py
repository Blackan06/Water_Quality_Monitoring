"""Reusable Kafka trigger functions for AwaitMessageTriggerFunctionSensor."""

from typing import Any


def always_true(event: Any, **kwargs) -> bool:
    """Simple pass-through apply_function that matches any event.

    This must be importable by the Triggerer process via a dotted path.
    """
    return True


