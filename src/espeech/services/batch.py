from __future__ import annotations

import os
import tempfile
import traceback
import zipfile
from dataclasses import dataclass

import soundfile as sf

from espeech.domain.batching import batch_seed, safe_filename, split_batch_lines
from espeech.runtime.resources import ResourceManager
from espeech.services.synthesis import (
    SynthesisNotifications,
    synthesize_speech,
)


@dataclass(slots=True)
class BatchSynthesisOutcome:
    zip_path: str | None
    summary: str
    processed_ref_text: str
    processed_batch_text: str


def _warn(notifications: SynthesisNotifications, message: str) -> None:
    if notifications.warn is not None:
        notifications.warn(message)


def synthesize_batch(
    resource_manager: ResourceManager,
    ref_audio: str | None,
    ref_text: str,
    batch_text: str,
    accent_mode: str,
    remove_silence: bool,
    seed: int | float | str | None,
    cross_fade_duration: int | float | str | None = 0.15,
    nfe_step: int | float | str | None = 32,
    speed: int | float | str | None = 1.0,
    notifications: SynthesisNotifications | None = None,
) -> BatchSynthesisOutcome:
    notifications = notifications or SynthesisNotifications()
    batch_lines = split_batch_lines(batch_text)
    if not batch_lines:
        _warn(notifications, "Please enter one text per line for batch generation.")
        return BatchSynthesisOutcome(None, "", ref_text, "")

    batch_dir = tempfile.mkdtemp(prefix="espeech_batch_")
    zip_path = os.path.join(batch_dir, "espeech_batch_results.zip")
    summary_lines = [f"Generated {len(batch_lines)} item(s):"]
    processed_ref_preview = ref_text
    processed_batch_lines: list[str] = []

    try:
        with zipfile.ZipFile(
            zip_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            for index, line in enumerate(batch_lines):
                result = synthesize_speech(
                    resource_manager=resource_manager,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    gen_text=line,
                    accent_mode=accent_mode,
                    remove_silence=remove_silence,
                    seed=batch_seed(seed, index),
                    cross_fade_duration=cross_fade_duration,
                    nfe_step=nfe_step,
                    speed=speed,
                    notifications=notifications,
                )

                processed_ref_preview = result.processed_ref_text
                processed_batch_lines.append(result.processed_gen_text)

                if result.audio is None:
                    summary_lines.append(f"{index + 1}. failed: {line}")
                    continue

                sample_rate, waveform = result.audio
                item_name = safe_filename(result.processed_gen_text, index)
                audio_path = os.path.join(batch_dir, f"{index + 1:02d}_{item_name}.wav")
                sf.write(audio_path, waveform, sample_rate)
                archive.write(audio_path, arcname=os.path.basename(audio_path))

                if result.spectrogram_path and os.path.exists(result.spectrogram_path):
                    spectrogram_name = f"{index + 1:02d}_{item_name}.png"
                    archive.write(result.spectrogram_path, arcname=spectrogram_name)

                summary_lines.append(
                    f"{index + 1}. ok | seed={result.seed} | {result.processed_gen_text}"
                )
    except Exception as exc:
        _warn(notifications, f"Batch generation failed: {exc}")
        traceback.print_exc()
        return BatchSynthesisOutcome(
            None,
            "\n".join(summary_lines),
            processed_ref_preview,
            "\n".join(processed_batch_lines),
        )

    return BatchSynthesisOutcome(
        zip_path,
        "\n".join(summary_lines),
        processed_ref_preview,
        "\n".join(processed_batch_lines),
    )
