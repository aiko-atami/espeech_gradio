from __future__ import annotations

import math
import random


SEED_MAX = 2**31 - 1


def _finite_float(value: int | float | str | None, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _finite_int(value: int | float | str | None, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def normalize_seed(seed: int | float | str | None) -> int:
    parsed = _finite_int(seed, -1)
    if parsed < 0 or parsed > SEED_MAX:
        return random.randint(0, SEED_MAX)
    return parsed


def sanitize_infer_params(
    cross_fade_duration: int | float | str | None,
    nfe_step: int | float | str | None,
    speed: int | float | str | None,
) -> tuple[float, int, float]:
    cross_fade = _finite_float(cross_fade_duration, 0.15)
    nfe = _finite_int(nfe_step, 32)
    speed_value = _finite_float(speed, 1.0)

    cross_fade = min(max(cross_fade, 0.0), 1.0)
    nfe = min(max(nfe, 4), 64)
    speed_value = min(max(speed_value, 0.3), 2.0)

    return cross_fade, nfe, speed_value
