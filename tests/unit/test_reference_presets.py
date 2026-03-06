from espeech.ui.reference_presets import (
    get_reference_preset,
    list_reference_presets,
    reference_preset_choices,
    reference_preset_status,
)


def test_list_reference_presets_keeps_only_complete_pairs(tmp_path):
    (tmp_path / "bravo.wav").write_bytes(b"bravo")
    (tmp_path / "bravo.txt").write_text("Bravo text", encoding="utf-8")
    (tmp_path / "alpha.wav").write_bytes(b"alpha")
    (tmp_path / "alpha.txt").write_text("Alpha text", encoding="utf-8")
    (tmp_path / "audio_only.wav").write_bytes(b"partial")
    (tmp_path / "text_only.txt").write_text("partial", encoding="utf-8")

    presets = list_reference_presets(tmp_path)

    assert [preset.name for preset in presets] == ["alpha", "bravo"]
    assert reference_preset_choices(tmp_path) == ["alpha", "bravo"]
    assert reference_preset_status(tmp_path) == ""
    assert presets[0].audio_path == str(tmp_path / "alpha.wav")
    assert presets[0].text == "Alpha text"


def test_reference_preset_helpers_handle_missing_directory(tmp_path):
    missing_refs_dir = tmp_path / "refs"

    assert list_reference_presets(missing_refs_dir) == []
    assert reference_preset_choices(missing_refs_dir) == []
    assert get_reference_preset("alpha", missing_refs_dir) is None
    assert "не найдена" in reference_preset_status(missing_refs_dir)
