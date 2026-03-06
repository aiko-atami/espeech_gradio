from __future__ import annotations

from collections.abc import Callable

import numpy as np
import soundfile as sf
import torch
import torchaudio


def soundfile_torchaudio_load(
    path: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
):
    del normalize
    start = int(frame_offset) if frame_offset else 0
    frames = int(num_frames) if num_frames and num_frames > 0 else -1
    waveform_np, sample_rate = sf.read(
        path,
        start=start,
        frames=frames,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(waveform_np)
    if channels_first:
        waveform = waveform.transpose(0, 1)
    return waveform, sample_rate


def patch_torchaudio_load_if_needed(
    logger: Callable[[str], None] | None = None,
) -> None:
    try:
        import torchcodec  # noqa: F401
    except Exception as exc:
        if torchaudio.load is soundfile_torchaudio_load:
            return
        if logger is not None:
            logger(
                f"TorchCodec unavailable, patching torchaudio.load with soundfile: {exc}"
            )
        torchaudio.load = soundfile_torchaudio_load


def patch_ruaccent_token_type_ids(accentizer_obj) -> None:
    accent_model = accentizer_obj.accent_model
    if getattr(accent_model, "_espeech_token_type_ids_patched", False):
        return

    original_put_accent = accent_model.put_accent

    def put_accent_with_fallback(word: str):
        try:
            return original_put_accent(word)
        except ValueError as exc:
            if "token_type_ids" not in str(exc):
                raise

            lower_word = word.lower()
            inputs = accent_model.tokenizer(lower_word, return_tensors="np")
            if "token_type_ids" not in inputs and "input_ids" in inputs:
                inputs["token_type_ids"] = np.zeros_like(inputs["input_ids"])
            inputs = {key: value.astype(np.int64) for key, value in inputs.items()}

            outputs = accent_model.session.run(None, inputs)
            output_names = {
                output_key.name: idx
                for idx, output_key in enumerate(accent_model.session.get_outputs())
            }
            logits = outputs[output_names["logits"]]
            probabilities = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probabilities = probabilities / probabilities.sum(axis=-1, keepdims=True)
            scores = np.max(probabilities, axis=-1)[0]
            labels = np.argmax(logits, axis=-1)[0]
            predictions = [
                {
                    "label": accent_model.id2label[str(label)],
                    "score": float(score),
                }
                for label, score in zip(labels, scores)
            ]
            return accent_model.render_stress(word, predictions)

    accent_model.put_accent = put_accent_with_fallback
    accent_model._espeech_token_type_ids_patched = True
