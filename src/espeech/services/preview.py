from __future__ import annotations

from collections.abc import Callable

from espeech.domain.batching import split_batch_lines
from espeech.domain.text_processing import process_text_with_accent
from espeech.runtime.resources import ResourceManager


def process_texts_only(
    resource_manager: ResourceManager,
    ref_text: str,
    gen_text: str,
    accent_mode: str,
    warn_fn: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    if accent_mode == "manual":
        return ref_text, gen_text

    try:
        accentizer = resource_manager.ensure_accentizer()
    except Exception as exc:
        if warn_fn is not None:
            warn_fn(f"Failed to load accent model: {exc}")
        return ref_text, gen_text

    processed_ref_text = process_text_with_accent(
        ref_text,
        accentizer,
        accent_mode,
        warn_fn=warn_fn,
    )
    processed_gen_text = process_text_with_accent(
        gen_text,
        accentizer,
        accent_mode,
        warn_fn=warn_fn,
    )
    return processed_ref_text, processed_gen_text


def preview_single_text(
    resource_manager: ResourceManager,
    ref_text: str,
    gen_text: str,
    accent_mode: str,
    warn_fn: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    return process_texts_only(
        resource_manager,
        ref_text,
        gen_text,
        accent_mode,
        warn_fn=warn_fn,
    )


def preview_batch_text(
    resource_manager: ResourceManager,
    ref_text: str,
    batch_text: str,
    accent_mode: str,
    warn_fn: Callable[[str], None] | None = None,
) -> tuple[str, str]:
    batch_lines = split_batch_lines(batch_text)
    if not batch_lines:
        return ref_text, ""

    processed_ref_text = ref_text
    processed_batch_lines: list[str] = []
    for line in batch_lines:
        processed_ref_text, processed_line = process_texts_only(
            resource_manager,
            processed_ref_text,
            line,
            accent_mode,
            warn_fn=warn_fn,
        )
        processed_batch_lines.append(processed_line)

    return processed_ref_text, "\n".join(processed_batch_lines)
