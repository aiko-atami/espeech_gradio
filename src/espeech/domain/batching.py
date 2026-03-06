from __future__ import annotations

import re


def split_batch_lines(batch_text: str | None) -> list[str]:
    return [line.strip() for line in (batch_text or "").splitlines() if line.strip()]


def safe_filename(text: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    if not slug:
        slug = f"item-{index + 1}"
    return slug[:48]


def batch_seed(seed: int | float | str | None, index: int) -> int | float | str | None:
    try:
        parsed = int(seed)
    except (TypeError, ValueError, OverflowError):
        return seed

    if parsed == -1:
        return -1

    return parsed + index
