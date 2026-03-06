from __future__ import annotations

from dataclasses import dataclass
from inspect import cleandoc
from typing import Any

import gradio as gr

from espeech.config import APP_TITLE, DEFAULT_ACCENT_MODE
from espeech.runtime.resources import ResourceManager
from espeech.ui.events import bind_events
from espeech.ui.reference_presets import (
    reference_preset_choices,
    reference_preset_status,
)

INTRO_MARKDOWN = cleandoc(
    """
    - Загрузите reference (аудио + текст).
    - Введите текст для синтеза, выберите режим ударений и подготовьте текст.
    - Запускайте синтез.

    Первый запрос после старта может выполняться дольше: модель, вокодер и акцентайзер загружаются лениво.
    """
)

REFERENCE_MARKDOWN = cleandoc(
    """
    Референс до 12 секунд, чистый голос без музыки и шумов. Чем чище запись и точнее transcript, тем стабильнее голос.
    """
)

ACCENT_MODES_MARKDOWN = cleandoc(
    """
    **Auto** — акцентайзер расставит ударения, если в тексте нет `+`<br>
    **Manual** — только ваши `+`, автомат не трогает текст<br>
    **Hybrid** — автомат и ваши `+` сохраняются
    """
)


TEXT_PREPARATION_MARKDOWN = cleandoc(
    """
    ## Text Preparation
    Введите текст для генерации. Для ручного ударения используйте `+` перед гласной, например: `прив+ет`.
    """
)


@dataclass(slots=True)
class UIComponents:
    ref_audio_input: Any
    ref_text_input: Any
    ref_preset_tab: Any
    ref_preset_selector: Any
    ref_preset_preview_row: Any
    ref_preset_audio_preview: Any
    ref_preset_text_preview: Any
    ref_preset_status: Any
    accent_mode_input: Any
    gen_text_input: Any
    process_text_button: Any
    processed_text_accordion: Any
    generate_button: Any
    remove_silence_input: Any
    seed_input: Any
    speed_slider: Any
    nfe_slider: Any
    cross_fade_slider: Any
    processed_ref_preview: Any
    processed_gen_preview: Any
    audio_output: Any
    spectrogram_output: Any


def create_app(resource_manager: ResourceManager | None = None) -> gr.Blocks:
    manager = resource_manager or ResourceManager()
    preset_choices = reference_preset_choices()
    preset_status = reference_preset_status()

    with gr.Blocks(title=APP_TITLE) as app:
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(
            INTRO_MARKDOWN,
            container=True,
        )

        with gr.Group():
            gr.Markdown("## Reference", container=True, elem_classes="padded-text")
            with gr.Tabs():
                with gr.Tab("Upload / Record"):
                    with gr.Row(elem_id="ref-inputs-row", equal_height=True):
                        with gr.Column():
                            ref_audio_input = gr.Audio(
                                label="Reference Audio",
                                type="filepath",
                                sources=["upload", "microphone"],
                                elem_id="ref-audio",
                            )
                        with gr.Column():
                            ref_text_input = gr.Textbox(
                                label="Reference Text",
                                lines=8,
                                placeholder="Text corresponding to reference audio",
                                elem_id="ref-text",
                            )
                    gr.Markdown(
                        REFERENCE_MARKDOWN,
                        container=True,
                        elem_classes="padded-text",
                    )

                with gr.Tab("Saved refs") as ref_preset_tab:
                    ref_preset_status = gr.Markdown(
                        preset_status,
                        visible=bool(preset_status),
                        container=True,
                        elem_classes="padded-text",
                    )
                    with gr.Row():
                        ref_preset_selector = gr.Radio(
                            choices=preset_choices,
                            value=None,
                            label="Choose reference preset",
                        )
                    with gr.Row(
                        equal_height=True, visible=False
                    ) as ref_preset_preview_row:
                        with gr.Column():
                            ref_preset_audio_preview = gr.Audio(
                                label="Preset Audio",
                                type="filepath",
                                interactive=False,
                            )
                        with gr.Column():
                            ref_preset_text_preview = gr.Textbox(
                                label="Preset Text",
                                lines=8,
                                interactive=False,
                            )

        with gr.Group():
            gr.Markdown(
                TEXT_PREPARATION_MARKDOWN,
                container=True,
                elem_classes="padded-text",
            )
            gen_text_input = gr.Textbox(
                label="Text to generate",
                lines=8,
                max_lines=20,
                placeholder="Enter text to synthesize...",
            )
            accent_mode_input = gr.Radio(
                choices=["auto", "manual", "hybrid"],
                value=DEFAULT_ACCENT_MODE,
                label="Accent Mode",
            )
            gr.Markdown(
                ACCENT_MODES_MARKDOWN,
                container=True,
                elem_classes="padded-text",
            )
            process_text_button = gr.Button(
                "Preview text", variant="secondary", size="md"
            )

            with gr.Accordion(
                "Processed text", open=False, elem_classes="no-border-accordion"
            ) as processed_text_accordion:
                processed_ref_preview = gr.Textbox(
                    label="Processed Reference Text",
                    lines=3,
                    interactive=False,
                )
                processed_gen_preview = gr.Textbox(
                    label="Processed Generation Text",
                    lines=6,
                    interactive=False,
                )

        with gr.Group():
            with gr.Accordion(
                "Advanced Settings", open=False, elem_classes="no-border-accordion"
            ):
                with gr.Column():
                    seed_input = gr.Number(
                        label="🎲 Seed",
                        value=-1,
                        precision=0,
                        info="Укажите -1 для случайного значения.",
                    )
                    nfe_slider = gr.Slider(
                        label="🎛️ NFE Steps",
                        minimum=4,
                        maximum=64,
                        value=48,
                        step=2,
                        info="Больше шагов — лучше качество, но дольше генерация.",
                    )

                    speed_slider = gr.Slider(
                        label="⚡ Speed",
                        minimum=0.3,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="Ниже 1.0: медленнее/выразительнее. Выше 1.0: быстрее.",
                    )
                    cross_fade_slider = gr.Slider(
                        label="🎞️ Cross-Fade (s)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.15,
                        step=0.01,
                        info="Сглаживает переходы между склеенными сегментами.",
                    )

                    remove_silence_input = gr.Checkbox(
                        label="✂️ Remove Silences",
                        value=False,
                        info="Автоматическое удаление тишины из итогового аудио.",
                    )

            generate_button = gr.Button(
                "Generate speech",
                variant="primary",
                size="md",
            )

            audio_output = gr.Audio(label="Generated Audio", type="numpy")
            with gr.Accordion(
                "Diagnostics", open=False, elem_classes="no-border-accordion"
            ):
                spectrogram_output = gr.Image(
                    label="Spectrogram",
                    type="filepath",
                )

        components = UIComponents(
            ref_audio_input=ref_audio_input,
            ref_text_input=ref_text_input,
            ref_preset_tab=ref_preset_tab,
            ref_preset_selector=ref_preset_selector,
            ref_preset_preview_row=ref_preset_preview_row,
            ref_preset_audio_preview=ref_preset_audio_preview,
            ref_preset_text_preview=ref_preset_text_preview,
            ref_preset_status=ref_preset_status,
            accent_mode_input=accent_mode_input,
            gen_text_input=gen_text_input,
            process_text_button=process_text_button,
            processed_text_accordion=processed_text_accordion,
            generate_button=generate_button,
            remove_silence_input=remove_silence_input,
            seed_input=seed_input,
            speed_slider=speed_slider,
            nfe_slider=nfe_slider,
            cross_fade_slider=cross_fade_slider,
            processed_ref_preview=processed_ref_preview,
            processed_gen_preview=processed_gen_preview,
            audio_output=audio_output,
            spectrogram_output=spectrogram_output,
        )
        bind_events(components, manager)

    return app
