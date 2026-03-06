from types import SimpleNamespace

from espeech.ui.app import create_app


def test_create_app_builds_gradio_blocks():
    app = create_app(resource_manager=SimpleNamespace())

    assert app is not None
    assert callable(app.launch)
