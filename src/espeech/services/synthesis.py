from __future__ import annotations

import gc
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import soundfile as sf
import torch
import torchaudio
from f5_tts.infer.utils_infer import (
    infer_process,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
    save_spectrogram,
    tempfile_kwargs,
)

from espeech.domain.synthesis_params import normalize_seed, sanitize_infer_params
from espeech.domain.text_processing import process_text_with_accent
from espeech.runtime.resources import ResourceManager


def _noop(_: str) -> None:
    return None


@dataclass(slots=True)
class SynthesisNotifications:
    warn: Callable[[str], None] | None = None
    info: Callable[[str], None] | None = None
    progress_factory: Callable[[], Any] | None = None


@dataclass(slots=True)
class SynthesisOutcome:
    audio: tuple[int, Any] | None
    spectrogram_path: str | None
    processed_ref_text: str
    processed_gen_text: str
    seed: int


def _warn(notifications: SynthesisNotifications, message: str) -> None:
    if notifications.warn is not None:
        notifications.warn(message)


def _save_spectrogram(combined_spectrogram) -> str | None:
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            **tempfile_kwargs,
        ) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(combined_spectrogram, spectrogram_path)
            return spectrogram_path
    except Exception as exc:
        print("Save spectrogram failed:", exc)
        return None


def synthesize_speech(
    resource_manager: ResourceManager,
    ref_audio: str | None,
    ref_text: str,
    gen_text: str,
    accent_mode: str,
    remove_silence: bool,
    seed: int | float | str | None,
    cross_fade_duration: int | float | str | None = 0.15,
    nfe_step: int | float | str | None = 32,
    speed: int | float | str | None = 1.0,
    notifications: SynthesisNotifications | None = None,
) -> SynthesisOutcome:
    notifications = notifications or SynthesisNotifications()

    if not ref_audio:
        _warn(notifications, "Please provide reference audio.")
        return SynthesisOutcome(None, None, ref_text, gen_text, normalize_seed(seed))

    normalized_seed = normalize_seed(seed)
    cross_fade_duration, nfe_step, speed = sanitize_infer_params(
        cross_fade_duration,
        nfe_step,
        speed,
    )
    torch.manual_seed(normalized_seed)

    if not gen_text or not gen_text.strip():
        _warn(notifications, "Please enter text to generate.")
        return SynthesisOutcome(None, None, ref_text, gen_text, normalized_seed)

    accentizer = None
    if accent_mode != "manual":
        try:
            accentizer = resource_manager.ensure_accentizer()
        except Exception as exc:
            _warn(notifications, f"Failed to load accent model: {exc}")
            return SynthesisOutcome(None, None, ref_text, gen_text, normalized_seed)

    processed_ref_text = process_text_with_accent(
        ref_text,
        accentizer,
        accent_mode,
        warn_fn=notifications.warn,
    )
    processed_gen_text = process_text_with_accent(
        gen_text,
        accentizer,
        accent_mode,
        warn_fn=notifications.warn,
    )

    try:
        model = resource_manager.ensure_model()
    except Exception as exc:
        _warn(notifications, f"Failed to load model: {exc}")
        return SynthesisOutcome(
            None,
            None,
            processed_ref_text,
            processed_gen_text,
            normalized_seed,
        )

    try:
        vocoder = resource_manager.ensure_vocoder()
    except Exception as exc:
        _warn(notifications, f"Failed to load vocoder: {exc}")
        return SynthesisOutcome(
            None,
            None,
            processed_ref_text,
            processed_gen_text,
            normalized_seed,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info_fn = notifications.info or _noop

    with resource_manager.inference_lock:
        try:
            if device.type == "cuda":
                try:
                    model.to(device)
                    vocoder.to(device)
                except Exception as exc:
                    print("Warning: failed to move model/vocoder to cuda:", exc)

            try:
                ref_audio_proc, processed_ref_text_final = preprocess_ref_audio_text(
                    ref_audio,
                    processed_ref_text,
                    show_info=info_fn,
                )
            except Exception as exc:
                _warn(notifications, f"Preprocess failed: {exc}")
                traceback.print_exc()
                return SynthesisOutcome(
                    None,
                    None,
                    processed_ref_text,
                    processed_gen_text,
                    normalized_seed,
                )

            try:
                infer_kwargs = {
                    "cross_fade_duration": cross_fade_duration,
                    "nfe_step": nfe_step,
                    "speed": speed,
                    "show_info": info_fn,
                }
                if notifications.progress_factory is not None:
                    infer_kwargs["progress"] = notifications.progress_factory()
                final_wave, final_sample_rate, combined_spectrogram = infer_process(
                    ref_audio_proc,
                    processed_ref_text_final,
                    processed_gen_text,
                    model,
                    vocoder,
                    **infer_kwargs,
                )
            except Exception as exc:
                _warn(notifications, f"Infer failed: {exc}")
                traceback.print_exc()
                return SynthesisOutcome(
                    None,
                    None,
                    processed_ref_text,
                    processed_gen_text,
                    normalized_seed,
                )

            if remove_silence:
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav",
                        **tempfile_kwargs,
                    ) as temp_file:
                        temp_path = temp_file.name
                        sf.write(temp_path, final_wave, final_sample_rate)
                        remove_silence_for_generated_wav(temp_path)
                        final_wave_tensor, _ = torchaudio.load(temp_path)
                        final_wave = final_wave_tensor.squeeze().cpu().numpy()
                except Exception as exc:
                    print("Remove silence failed:", exc)

            spectrogram_path = _save_spectrogram(combined_spectrogram)

            import numpy as np

            if isinstance(final_wave, np.ndarray) and final_wave.dtype in (
                np.float64,
                np.float32,
            ):
                final_wave = np.clip(final_wave, -1.0, 1.0)
                final_wave = (final_wave * 32767).astype(np.int16)

            return SynthesisOutcome(
                (final_sample_rate, final_wave),
                spectrogram_path,
                processed_ref_text_final,
                processed_gen_text,
                normalized_seed,
            )
        finally:
            if device.type == "cuda":
                try:
                    model.to("cpu")
                    vocoder.to("cpu")
                    torch.cuda.empty_cache()
                    gc.collect()
                except Exception as exc:
                    print("Warning during cuda cleanup:", exc)
