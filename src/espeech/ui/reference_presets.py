from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PRESETS_DIR = PROJECT_ROOT / "refs"


@dataclass(frozen=True, slots=True)
class ReferencePreset:
    name: str
    audio_path: str
    text: str


def list_reference_presets(
    refs_dir: Path = REFERENCE_PRESETS_DIR,
) -> list[ReferencePreset]:
    if not refs_dir.is_dir():
        return []

    presets: list[ReferencePreset] = []
    for audio_path in sorted(refs_dir.glob("*.wav")):
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.is_file():
            continue
        try:
            text = transcript_path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            continue
        presets.append(
            ReferencePreset(
                name=audio_path.stem,
                audio_path=str(audio_path),
                text=text,
            )
        )
    return presets


def reference_preset_choices(refs_dir: Path = REFERENCE_PRESETS_DIR) -> list[str]:
    return [preset.name for preset in list_reference_presets(refs_dir)]


def get_reference_preset(
    name: str | None,
    refs_dir: Path = REFERENCE_PRESETS_DIR,
) -> ReferencePreset | None:
    if not name:
        return None

    for preset in list_reference_presets(refs_dir):
        if preset.name == name:
            return preset
    return None


def reference_preset_status(refs_dir: Path = REFERENCE_PRESETS_DIR) -> str:
    if not refs_dir.is_dir():
        return (
            f"Папка `{refs_dir}` не найдена. Создайте в корне проекта пары "
            "`name.wav` + `name.txt`, чтобы они появились в списке."
        )

    presets = list_reference_presets(refs_dir)
    if not presets:
        return (
            f"В `{refs_dir}` пока нет валидных пар `name.wav` + `name.txt`. "
            "Неполные наборы в список не попадают."
        )

    return ""
