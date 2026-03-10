import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from espeech.services.batch_config import (
    BatchItemConfig,
    BatchJobConfig,
    BatchOutputConfig,
    BatchReferenceConfig,
    BatchSettingsConfig,
)
from espeech.services.batch_runner import run_batch_job
from espeech.services.synthesis import SynthesisOutcome


def test_run_batch_job_writes_outputs_and_metadata(tmp_path: Path, monkeypatch):
    ref_audio = tmp_path / "ref.wav"
    ref_audio.write_bytes(b"wav")
    spectrogram = tmp_path / "source.png"
    spectrogram.write_bytes(b"png")
    config_path = tmp_path / "batch.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    output_dir = tmp_path / "out"

    calls: list[dict[str, object]] = []

    def fake_synthesize_speech(**kwargs):
        calls.append(kwargs)
        seed = int(kwargs["seed"])
        return SynthesisOutcome(
            audio=(24000, np.array([0, 1000, -1000], dtype=np.int16)),
            spectrogram_path=str(spectrogram),
            processed_ref_text=kwargs["ref_text"],
            processed_gen_text=str(kwargs["gen_text"]).upper(),
            seed=seed,
        )

    monkeypatch.setattr(
        "espeech.services.batch_runner.synthesize_speech",
        fake_synthesize_speech,
    )

    config = BatchJobConfig(
        config_path=config_path,
        reference=BatchReferenceConfig(audio_path=ref_audio, text="Reference"),
        settings=BatchSettingsConfig(seed=42, nfe_step=48),
        output=BatchOutputConfig(
            directory=output_dir,
            zip_results=True,
            save_spectrograms=True,
        ),
        items=[
            BatchItemConfig(text="first item"),
            BatchItemConfig(text="second item", file_name="greeting_2", seed=9000),
        ],
    )

    result = run_batch_job(config, resource_manager=SimpleNamespace(), log=None)

    assert result.archive_path is not None
    assert (output_dir / "01_first-item.wav").is_file()
    assert (output_dir / "02_greeting_2.wav").is_file()
    assert (output_dir / "01_first-item.png").is_file()
    assert (output_dir / "02_greeting_2.png").is_file()
    assert (output_dir / "summary.txt").is_file()
    assert (output_dir / "results.json").is_file()
    assert (output_dir / "batch_results.zip").is_file()
    assert [call["seed"] for call in calls] == [42, 9000]

    payload = json.loads((output_dir / "results.json").read_text(encoding="utf-8"))
    assert [item["audio_path"] for item in payload["items"]] == [
        "01_first-item.wav",
        "02_greeting_2.wav",
    ]
    assert [item["status"] for item in payload["items"]] == ["ok", "ok"]
