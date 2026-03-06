from espeech.ui.styles import APP_CSS


def test_app_css_avoids_gradio_internal_selectors():
    assert ".section-copy" in APP_CSS
    assert "#ref-inputs-row" in APP_CSS
    assert ".wrap" not in APP_CSS
    assert ".empty-container" not in APP_CSS
    assert ".record-container" not in APP_CSS
    assert ".waveform-container" not in APP_CSS
    assert ".audio-container" not in APP_CSS
