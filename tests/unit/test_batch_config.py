from pathlib import Path

import pytest

from espeech.services.batch_config import load_batch_job_config


def test_load_batch_job_config_resolves_relative_paths_and_items_file(tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"wav")
    ref_text = tmp_path / "ref.txt"
    ref_text.write_text("Reference text", encoding="utf-8")
    items_file = tmp_path / "items.txt"
    items_file.write_text(" first \n\nsecond\n", encoding="utf-8")
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
reference:
  audio: ref.wav
  text_file: ref.txt

settings:
  seed: 10
  nfe_step: 48

output:
  dir: out/demo
  zip: true

items_file: items.txt
items:
  - text: "third"
    file: "custom-third"
    speed: 0.9
""".strip(),
        encoding="utf-8",
    )

    config = load_batch_job_config(config_path)

    assert config.reference.audio_path == ref_audio.resolve()
    assert config.reference.text == "Reference text"
    assert config.settings.seed == 10
    assert config.settings.nfe_step == 48
    assert config.output.directory == (tmp_path / "out" / "demo").resolve()
    assert config.output.zip_results is True
    assert [item.text for item in config.items] == ["first", "second", "third"]
    assert config.items[2].file_name == "custom-third"
    assert config.items[2].speed == 0.9


def test_load_batch_job_config_rejects_invalid_accent_mode(tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"wav")
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
reference:
  audio: ref.wav
  text: "Reference"

settings:
  accent_mode: invalid

items:
  - text: "hello"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="settings.accent_mode"):
        load_batch_job_config(config_path)


def test_load_batch_job_config_rejects_non_boolean_flags(tmp_path: Path):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"wav")
    config_path = tmp_path / "batch.yaml"
    config_path.write_text(
        """
reference:
  audio: ref.wav
  text: "Reference"

output:
  zip: "false"

items:
  - text: "hello"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="output.zip"):
        load_batch_job_config(config_path)
