from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from espeech.config import DEFAULT_ACCENT_MODE
from espeech.domain.batching import split_batch_lines

DEFAULT_BATCH_CROSS_FADE = 0.15
DEFAULT_BATCH_NFE_STEP = 48
DEFAULT_BATCH_SPEED = 1.0
VALID_ACCENT_MODES = {"auto", "manual", "hybrid"}


@dataclass(slots=True, frozen=True)
class BatchReferenceConfig:
    audio_path: Path
    text: str


@dataclass(slots=True, frozen=True)
class BatchSettingsConfig:
    accent_mode: str = DEFAULT_ACCENT_MODE
    remove_silence: bool = False
    seed: int | float | str | None = -1
    cross_fade_duration: int | float | str | None = DEFAULT_BATCH_CROSS_FADE
    nfe_step: int | float | str | None = DEFAULT_BATCH_NFE_STEP
    speed: int | float | str | None = DEFAULT_BATCH_SPEED


@dataclass(slots=True, frozen=True)
class BatchOutputConfig:
    directory: Path
    zip_results: bool = False
    save_spectrograms: bool = False


@dataclass(slots=True, frozen=True)
class BatchItemConfig:
    text: str
    file_name: str | None = None
    accent_mode: str | None = None
    remove_silence: bool | None = None
    seed: int | float | str | None = None
    cross_fade_duration: int | float | str | None = None
    nfe_step: int | float | str | None = None
    speed: int | float | str | None = None


@dataclass(slots=True, frozen=True)
class BatchJobConfig:
    config_path: Path
    reference: BatchReferenceConfig
    settings: BatchSettingsConfig
    output: BatchOutputConfig
    items: list[BatchItemConfig]


def _as_mapping(value: Any, section: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Section '{section}' must be a YAML mapping.")
    return value


def _ensure_allowed_keys(
    value: dict[str, Any],
    allowed_keys: set[str],
    section: str,
) -> None:
    unexpected = sorted(set(value) - allowed_keys)
    if unexpected:
        unknown = ", ".join(unexpected)
        raise ValueError(f"Unknown key(s) in '{section}': {unknown}")


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Field '{field_name}' must be a non-empty string.")
    return value.strip()


def _resolve_path(value: Any, base_dir: Path, field_name: str) -> Path:
    path_value = _require_text(value, field_name)
    return (base_dir / path_value).expanduser().resolve()


def _resolve_existing_file(value: Any, base_dir: Path, field_name: str) -> Path:
    path = _resolve_path(value, base_dir, field_name)
    if not path.is_file():
        raise ValueError(f"File for '{field_name}' was not found: {path}")
    return path


def _read_text_file(path: Path, field_name: str) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ValueError(f"Failed to read '{field_name}' from {path}: {exc}") from exc
    if not text:
        raise ValueError(f"File for '{field_name}' is empty: {path}")
    return text


def _parse_accent_mode(value: Any, field_name: str) -> str:
    accent_mode = _require_text(value, field_name).lower()
    if accent_mode not in VALID_ACCENT_MODES:
        allowed = ", ".join(sorted(VALID_ACCENT_MODES))
        raise ValueError(f"Field '{field_name}' must be one of: {allowed}")
    return accent_mode


def _parse_bool(value: Any, field_name: str, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"Field '{field_name}' must be a boolean.")


def _parse_optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _parse_reference_config(value: Any, base_dir: Path) -> BatchReferenceConfig:
    reference = _as_mapping(value, "reference")
    _ensure_allowed_keys(reference, {"audio", "text", "text_file"}, "reference")

    audio_path = _resolve_existing_file(reference.get("audio"), base_dir, "reference.audio")

    raw_text = reference.get("text")
    text_file = reference.get("text_file")
    if raw_text is not None:
        text = _require_text(raw_text, "reference.text")
    elif text_file is not None:
        text_path = _resolve_existing_file(text_file, base_dir, "reference.text_file")
        text = _read_text_file(text_path, "reference.text_file")
    else:
        raise ValueError("Reference config must provide either 'text' or 'text_file'.")

    return BatchReferenceConfig(audio_path=audio_path, text=text)


def _parse_settings_config(value: Any) -> BatchSettingsConfig:
    settings = _as_mapping(value, "settings")
    _ensure_allowed_keys(
        settings,
        {
            "accent_mode",
            "remove_silence",
            "seed",
            "cross_fade_duration",
            "nfe_step",
            "speed",
        },
        "settings",
    )

    accent_mode = settings.get("accent_mode", DEFAULT_ACCENT_MODE)
    return BatchSettingsConfig(
        accent_mode=_parse_accent_mode(accent_mode, "settings.accent_mode"),
        remove_silence=_parse_bool(
            settings.get("remove_silence"),
            "settings.remove_silence",
            False,
        ),
        seed=settings.get("seed", -1),
        cross_fade_duration=settings.get(
            "cross_fade_duration",
            DEFAULT_BATCH_CROSS_FADE,
        ),
        nfe_step=settings.get("nfe_step", DEFAULT_BATCH_NFE_STEP),
        speed=settings.get("speed", DEFAULT_BATCH_SPEED),
    )


def _parse_output_config(value: Any, base_dir: Path, config_path: Path) -> BatchOutputConfig:
    output = _as_mapping(value, "output")
    _ensure_allowed_keys(output, {"dir", "zip", "save_spectrograms"}, "output")

    directory_value = output.get("dir", f"out/{config_path.stem}")
    directory = _resolve_path(directory_value, base_dir, "output.dir")

    return BatchOutputConfig(
        directory=directory,
        zip_results=_parse_bool(output.get("zip"), "output.zip", False),
        save_spectrograms=_parse_bool(
            output.get("save_spectrograms"),
            "output.save_spectrograms",
            False,
        ),
    )


def _parse_item(value: Any, index: int) -> BatchItemConfig:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(f"Item #{index + 1} must not be empty.")
        return BatchItemConfig(text=text)

    if not isinstance(value, dict):
        raise ValueError(f"Item #{index + 1} must be a string or mapping.")

    _ensure_allowed_keys(
        value,
        {
            "text",
            "file",
            "accent_mode",
            "remove_silence",
            "seed",
            "cross_fade_duration",
            "nfe_step",
            "speed",
        },
        f"items[{index}]",
    )

    accent_mode = value.get("accent_mode")
    return BatchItemConfig(
        text=_require_text(value.get("text"), f"items[{index}].text"),
        file_name=_parse_optional_text(value.get("file"), f"items[{index}].file"),
        accent_mode=(
            _parse_accent_mode(accent_mode, f"items[{index}].accent_mode")
            if accent_mode is not None
            else None
        ),
        remove_silence=(
            _parse_bool(
                value.get("remove_silence"),
                f"items[{index}].remove_silence",
                False,
            )
            if value.get("remove_silence") is not None
            else None
        ),
        seed=value.get("seed"),
        cross_fade_duration=value.get("cross_fade_duration"),
        nfe_step=value.get("nfe_step"),
        speed=value.get("speed"),
    )


def _parse_items_config(value: Any, items_file: Any, base_dir: Path) -> list[BatchItemConfig]:
    items: list[BatchItemConfig] = []

    if items_file is not None:
        items_path = _resolve_existing_file(items_file, base_dir, "items_file")
        batch_lines = split_batch_lines(_read_text_file(items_path, "items_file"))
        items.extend(BatchItemConfig(text=line) for line in batch_lines)

    if value is None:
        if items:
            return items
        raise ValueError("Batch config must provide 'items', 'items_file', or both.")

    if not isinstance(value, list):
        raise ValueError("Field 'items' must be a YAML list.")

    for index, item in enumerate(value):
        items.append(_parse_item(item, index))

    if not items:
        raise ValueError("Batch config resolved to an empty item list.")
    return items


def load_batch_job_config(config_path: str | Path) -> BatchJobConfig:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Batch config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Failed to read batch config from {path}: {exc}") from exc

    try:
        raw_config = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Failed to parse YAML batch config at {path}: {exc}") from exc

    if not isinstance(raw_config, dict):
        raise ValueError("Batch config root must be a YAML mapping.")

    _ensure_allowed_keys(
        raw_config,
        {"reference", "settings", "output", "items", "items_file"},
        "root",
    )

    base_dir = path.parent
    return BatchJobConfig(
        config_path=path,
        reference=_parse_reference_config(raw_config.get("reference"), base_dir),
        settings=_parse_settings_config(raw_config.get("settings")),
        output=_parse_output_config(raw_config.get("output"), base_dir, path),
        items=_parse_items_config(
            raw_config.get("items"),
            raw_config.get("items_file"),
            base_dir,
        ),
    )
