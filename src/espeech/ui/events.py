from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from espeech.runtime.resources import ResourceManager
from espeech.services.preview import preview_single_text
from espeech.services.synthesis import SynthesisNotifications, synthesize_speech
from espeech.ui.reference_presets import (
    get_reference_preset,
    reference_preset_choices,
    reference_preset_status,
)

if TYPE_CHECKING:
    from espeech.ui.app import UIComponents


def _notifications() -> SynthesisNotifications:
    return SynthesisNotifications(
        warn=gr.Warning,
        info=gr.Info,
        progress_factory=gr.Progress,
    )


def _preview_single_handler(
    resource_manager: ResourceManager, open_accordion: bool = False
):
    def handler(ref_text: str, gen_text: str, accent_mode: str):
        if not ref_text or not ref_text.strip():
            gr.Warning("Reference text is empty.")
            return (
                (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(interactive=True, value="Preview text"),
                )
                if open_accordion
                else (gr.update(), gr.update())
            )
        if not gen_text or not gen_text.strip():
            gr.Warning("Text to generate is empty.")
            return (
                (
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(interactive=True, value="Preview text"),
                )
                if open_accordion
                else (gr.update(), gr.update())
            )

        res = preview_single_text(
            resource_manager,
            ref_text,
            gen_text,
            accent_mode,
            warn_fn=gr.Warning,
        )
        if open_accordion:
            return (
                res[0],
                res[1],
                gr.update(open=True),
                gr.update(interactive=True, value="Preview text"),
            )
        return res[0], res[1]

    return handler


def _synthesize_handler(resource_manager: ResourceManager):
    def handler(
        ref_audio: str | None,
        ref_text: str,
        gen_text: str,
        accent_mode: str,
        remove_silence: bool,
        seed: int | float | str | None,
        cross_fade_duration: int | float | str | None = 0.15,
        nfe_step: int | float | str | None = 32,
        speed: int | float | str | None = 1.0,
    ):
        result = synthesize_speech(
            resource_manager=resource_manager,
            ref_audio=ref_audio,
            ref_text=ref_text,
            gen_text=gen_text,
            accent_mode=accent_mode,
            remove_silence=remove_silence,
            seed=seed,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
            notifications=_notifications(),
        )
        return (
            result.audio,
            result.spectrogram_path,
        )

    return handler


def _select_reference_preset_handler():
    def handler(
        preset_name: str | None,
        current_ref_audio: str | None,
        current_ref_text: str | None,
    ):
        preset = get_reference_preset(preset_name)
        if preset is None:
            return (
                gr.update(visible=False),
                None,
                "",
                current_ref_audio,
                current_ref_text or "",
            )

        return (
            gr.update(visible=True),
            preset.audio_path,
            preset.text,
            preset.audio_path,
            preset.text,
        )

    return handler


def _refresh_reference_presets_on_tab_handler():
    def handler():
        choices = reference_preset_choices()
        status = reference_preset_status()
        return (
            gr.update(choices=choices),
            gr.update(value=status, visible=bool(status)),
        )

    return handler


def bind_events(components: UIComponents, resource_manager: ResourceManager) -> None:
    components.ref_preset_tab.select(
        _refresh_reference_presets_on_tab_handler(),
        outputs=[
            components.ref_preset_selector,
            components.ref_preset_status,
        ],
        queue=False,
        show_progress="hidden",
    )

    components.ref_preset_selector.change(
        _select_reference_preset_handler(),
        inputs=[
            components.ref_preset_selector,
            components.ref_audio_input,
            components.ref_text_input,
        ],
        outputs=[
            components.ref_preset_preview_row,
            components.ref_preset_audio_preview,
            components.ref_preset_text_preview,
            components.ref_audio_input,
            components.ref_text_input,
        ],
        queue=False,
        show_progress="hidden",
    )

    components.process_text_button.click(
        lambda: (
            gr.update(interactive=False, value="Processing..."),
            gr.update(open=True),
        ),
        outputs=[components.process_text_button, components.processed_text_accordion],
        queue=False,
    ).then(
        _preview_single_handler(resource_manager, open_accordion=True),
        inputs=[
            components.ref_text_input,
            components.gen_text_input,
            components.accent_mode_input,
        ],
        outputs=[
            components.processed_ref_preview,
            components.processed_gen_preview,
            components.processed_text_accordion,
            components.process_text_button,
        ],
    )

    components.generate_button.click(
        _synthesize_handler(resource_manager),
        inputs=[
            components.ref_audio_input,
            components.ref_text_input,
            components.gen_text_input,
            components.accent_mode_input,
            components.remove_silence_input,
            components.seed_input,
            components.cross_fade_slider,
            components.nfe_slider,
            components.speed_slider,
        ],
        outputs=[
            components.audio_output,
            components.spectrogram_output,
        ],
    ).then(
        _preview_single_handler(resource_manager, open_accordion=False),
        inputs=[
            components.ref_text_input,
            components.gen_text_input,
            components.accent_mode_input,
        ],
        outputs=[
            components.processed_ref_preview,
            components.processed_gen_preview,
        ],
        show_progress="hidden",
    )
