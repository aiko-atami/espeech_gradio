from espeech.domain.batching import batch_seed, safe_filename, split_batch_lines


def test_split_batch_lines_ignores_empty_lines_and_whitespace():
    result = split_batch_lines(" first \n\nsecond\n   \n third ")

    assert result == ["first", "second", "third"]


def test_safe_filename_uses_index_fallback_for_empty_slug():
    assert safe_filename("!!!", 2) == "item-3"


def test_safe_filename_normalizes_and_truncates():
    name = safe_filename("Hello, Мир! one two three four five six seven eight", 0)

    assert name.startswith("hello-one-two-three")
    assert len(name) <= 48


def test_batch_seed_keeps_random_marker():
    assert batch_seed(-1, 3) == -1


def test_batch_seed_increments_fixed_seed():
    assert batch_seed(42, 3) == 45
