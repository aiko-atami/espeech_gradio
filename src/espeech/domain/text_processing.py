from __future__ import annotations

import re
from collections.abc import Callable


_MAX_CHUNK_CHARS = 350
_MAX_MODEL_TOKENS = 512


def _get_token_limit(accentizer) -> int:
    tokenizer = getattr(getattr(accentizer, "accent_model", None), "tokenizer", None)
    model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(model_max_length, int) and 0 < model_max_length < 100000:
        return model_max_length
    return _MAX_MODEL_TOKENS


def _get_token_count(text: str, accentizer) -> int | None:
    tokenizer = getattr(getattr(accentizer, "accent_model", None), "tokenizer", None)
    if tokenizer is None:
        return None

    encoded = tokenizer(text, add_special_tokens=True, truncation=False)
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return None

    if hasattr(input_ids, "shape"):
        if len(input_ids.shape) == 0:
            return int(input_ids)
        if len(input_ids.shape) == 1:
            return int(input_ids.shape[0])
        return int(input_ids.shape[-1])

    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _split_for_accent(text: str, max_len: int = _MAX_CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    parts = re.split(r"(\n+)", text)

    for part in parts:
        if not part:
            continue
        if part.startswith("\n"):
            chunks.append(part)
            continue

        sentences = re.split(r"(?<=[.!?…;:])\s+", part)
        current = ""
        for sentence in sentences:
            if not sentence:
                continue

            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) <= max_len:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = ""

            if len(sentence) <= max_len:
                current = sentence
                continue

            words = sentence.split(" ")
            word_buf = ""
            for word in words:
                candidate = f"{word_buf} {word}".strip() if word_buf else word
                if len(candidate) <= max_len:
                    word_buf = candidate
                    continue
                if word_buf:
                    chunks.append(word_buf)
                word_buf = word
            if word_buf:
                chunks.append(word_buf)

        if current:
            chunks.append(current)

    return chunks if chunks else [text]


def _split_longest_word(word: str) -> list[str]:
    midpoint = max(1, len(word) // 2)
    return [word[:midpoint], word[midpoint:]]


def _split_chunk_to_token_limit(
    text: str,
    accentizer,
    token_limit: int | None = None,
) -> list[str]:
    if token_limit is None:
        token_limit = _get_token_limit(accentizer)

    token_count = _get_token_count(text, accentizer)
    if token_count is None or token_count <= token_limit:
        return [text]

    initial_chunks = _split_for_accent(text)
    if len(initial_chunks) > 1:
        chunks: list[str] = []
        for chunk in initial_chunks:
            if not chunk or chunk.startswith("\n"):
                chunks.append(chunk)
                continue
            chunks.extend(_split_chunk_to_token_limit(chunk, accentizer, token_limit))
        return chunks

    words = text.split(" ")
    if len(words) > 1:
        midpoint = max(1, len(words) // 2)
        left = " ".join(words[:midpoint]).strip()
        right = " ".join(words[midpoint:]).strip()
        chunks: list[str] = []
        if left:
            chunks.extend(_split_chunk_to_token_limit(left, accentizer, token_limit))
        if right:
            chunks.extend(_split_chunk_to_token_limit(right, accentizer, token_limit))
        return chunks if chunks else [text]

    if len(text) <= 1:
        return [text]

    chunks: list[str] = []
    for piece in _split_longest_word(text):
        if piece:
            chunks.extend(_split_chunk_to_token_limit(piece, accentizer, token_limit))
    return chunks if chunks else [text]


def _safe_process_all(
    text: str,
    accentizer,
    warn_fn: Callable[[str], None] | None = None,
) -> str:
    token_limit = _get_token_limit(accentizer)
    token_count = _get_token_count(text, accentizer)

    if token_count is None or token_count <= token_limit:
        try:
            return accentizer.process_all(text)
        except Exception as exc:
            if "Attempting to broadcast an axis" not in str(exc):
                raise

    if warn_fn is not None:
        warn_fn("Accent model input is too long, processing in smaller chunks.")

    processed_chunks: list[str] = []
    for chunk in _split_chunk_to_token_limit(text, accentizer, token_limit):
        if not chunk or chunk.startswith("\n"):
            processed_chunks.append(chunk)
            continue
        try:
            processed_chunks.append(accentizer.process_all(chunk))
        except Exception:
            processed_chunks.append(chunk)
    return "".join(processed_chunks)


def process_text_with_accent(
    text: str,
    accentizer,
    accent_mode: str = "auto",
    warn_fn: Callable[[str], None] | None = None,
) -> str:
    if not text or not text.strip():
        return text

    if accent_mode == "manual":
        return text

    if accent_mode == "auto":
        if "+" in text:
            return text
        return _safe_process_all(text, accentizer, warn_fn=warn_fn)

    if accent_mode == "hybrid":
        keep_map: dict[str, str] = {}

        def _protect(match: re.Match[str]) -> str:
            key = f"ZZXKEEP{len(keep_map)}ZZX"
            keep_map[key] = match.group(0)
            return key

        protected_text = re.sub(r"\S*\+\S*", _protect, text)
        processed = _safe_process_all(
            protected_text,
            accentizer,
            warn_fn=warn_fn,
        )
        for key, value in keep_map.items():
            processed = processed.replace(key, value)
        return processed

    return text
