from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from espeech.domain.batching import batch_seed, safe_filename
from espeech.runtime.resources import ResourceManager
from espeech.services.batch_config import BatchItemConfig, BatchJobConfig
from espeech.services.synthesis import SynthesisNotifications, synthesize_speech


@dataclass(slots=True)
class BatchItemRunResult:
    index: int
    text: str
    processed_text: str
    seed: int
    status: str
    audio_path: str | None = None
    spectrogram_path: str | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchRunResult:
    output_dir: str
    summary_path: str
    results_path: str
    archive_path: str | None
    items: list[BatchItemRunResult]


def _log_message(log: Callable[[str], None] | None, message: str) -> None:
    if log is not None:
        log(message)


def _effective_value(item_value, default_value):
    return default_value if item_value is None else item_value


def _item_seed(config: BatchJobConfig, item: BatchItemConfig, index: int):
    if item.seed is not None:
        return item.seed
    return batch_seed(config.settings.seed, index)


def _output_stem(item: BatchItemConfig, processed_text: str, index: int) -> str:
    preferred = (item.file_name or "").strip()
    if preferred:
        slug = re.sub(r"[^\w.-]+", "-", Path(preferred).stem.lower()).strip("._-")
        if slug:
            return slug[:64]
    return safe_filename(processed_text, index)


def _copy_spectrogram(
    spectrogram_path: str | None,
    output_dir: Path,
    stem: str,
    index: int,
) -> str | None:
    if not spectrogram_path:
        return None

    source_path = Path(spectrogram_path)
    if not source_path.is_file():
        return None

    destination = output_dir / f"{index + 1:02d}_{stem}.png"
    shutil.copyfile(source_path, destination)
    return destination.name


def _write_summary(
    output_dir: Path,
    items: list[BatchItemRunResult],
) -> tuple[Path, Path]:
    summary_path = output_dir / "summary.txt"
    results_path = output_dir / "results.json"

    summary_lines = [f"Generated {len(items)} item(s) into {output_dir}:"]
    for item in items:
        if item.status == "ok":
            summary_lines.append(
                f"{item.index + 1}. ok | seed={item.seed} | {item.audio_path}"
            )
        else:
            summary_lines.append(
                f"{item.index + 1}. failed | seed={item.seed} | {item.error or item.text}"
            )

    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    results_path.write_text(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "items": [
                    {
                        "index": item.index,
                        "text": item.text,
                        "processed_text": item.processed_text,
                        "seed": item.seed,
                        "status": item.status,
                        "audio_path": item.audio_path,
                        "spectrogram_path": item.spectrogram_path,
                        "error": item.error,
                    }
                    for item in items
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary_path, results_path


def _write_archive(output_dir: Path) -> Path:
    archive_path = output_dir / "batch_results.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(output_dir.iterdir()):
            if path == archive_path or not path.is_file():
                continue
            archive.write(path, arcname=path.name)
    return archive_path


def run_batch_job(
    config: BatchJobConfig,
    resource_manager: ResourceManager | None = None,
    log: Callable[[str], None] | None = print,
) -> BatchRunResult:
    manager = resource_manager or ResourceManager(log=log)
    output_dir = config.output.directory
    output_dir.mkdir(parents=True, exist_ok=True)

    item_results: list[BatchItemRunResult] = []

    for index, item in enumerate(config.items):
        seed = _item_seed(config, item, index)
        item_messages: list[str] = []

        def warn(message: str) -> None:
            item_messages.append(message)
            _log_message(log, f"[warn] item {index + 1}: {message}")

        def info(message: str) -> None:
            _log_message(log, f"[info] item {index + 1}: {message}")

        outcome = synthesize_speech(
            resource_manager=manager,
            ref_audio=str(config.reference.audio_path),
            ref_text=config.reference.text,
            gen_text=item.text,
            accent_mode=_effective_value(item.accent_mode, config.settings.accent_mode),
            remove_silence=bool(
                _effective_value(
                    item.remove_silence,
                    config.settings.remove_silence,
                )
            ),
            seed=seed,
            cross_fade_duration=_effective_value(
                item.cross_fade_duration,
                config.settings.cross_fade_duration,
            ),
            nfe_step=_effective_value(item.nfe_step, config.settings.nfe_step),
            speed=_effective_value(item.speed, config.settings.speed),
            notifications=SynthesisNotifications(
                warn=warn,
                info=info,
            ),
        )

        if outcome.audio is None:
            error = "; ".join(item_messages) or "Synthesis returned no audio."
            item_results.append(
                BatchItemRunResult(
                    index=index,
                    text=item.text,
                    processed_text=outcome.processed_gen_text,
                    seed=outcome.seed,
                    status="failed",
                    error=error,
                )
            )
            _log_message(log, f"[fail] item {index + 1}: {error}")
            continue

        sample_rate, waveform = outcome.audio
        stem = _output_stem(item, outcome.processed_gen_text, index)
        audio_path = output_dir / f"{index + 1:02d}_{stem}.wav"
        sf.write(audio_path, waveform, sample_rate)

        saved_spectrogram = None
        if config.output.save_spectrograms:
            saved_spectrogram = _copy_spectrogram(
                outcome.spectrogram_path,
                output_dir,
                stem,
                index,
            )

        item_results.append(
            BatchItemRunResult(
                index=index,
                text=item.text,
                processed_text=outcome.processed_gen_text,
                seed=outcome.seed,
                status="ok",
                audio_path=audio_path.name,
                spectrogram_path=saved_spectrogram,
            )
        )
        _log_message(log, f"[ok] item {index + 1}: {audio_path.name}")

    summary_path, results_path = _write_summary(output_dir, item_results)
    archive_path = _write_archive(output_dir) if config.output.zip_results else None

    return BatchRunResult(
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        results_path=str(results_path),
        archive_path=str(archive_path) if archive_path else None,
        items=item_results,
    )
