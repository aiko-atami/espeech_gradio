import gradio as gr

from espeech.ui.app import create_app
from espeech.ui.styles import APP_CSS

demo = create_app()


def main() -> None:
    demo.launch(theme=gr.themes.Default(), css=APP_CSS)


if __name__ == "__main__":
    main()
