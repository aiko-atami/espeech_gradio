from espeech.domain.text_processing import process_text_with_accent


class FakeAccentizer:
    def __init__(self, token_limit=None, fail_on_too_long=False):
        self.calls = []
        self.token_limit = token_limit
        self.fail_on_too_long = fail_on_too_long
        tokenizer = type(
            "FakeTokenizer",
            (),
            {
                "model_max_length": token_limit or 512,
                "__call__": lambda _, text, add_special_tokens=True, truncation=False: (
                    self._tokenize(text, add_special_tokens, truncation)
                ),
            },
        )()
        self.accent_model = type("AccentModel", (), {"tokenizer": tokenizer})()

    def _tokenize(self, text, add_special_tokens=True, truncation=False):
        del truncation
        token_count = len(text.split())
        if add_special_tokens:
            token_count += 2
        return {"input_ids": list(range(token_count))}

    def process_all(self, text):
        self.calls.append(text)
        if self.fail_on_too_long and self.token_limit is not None:
            token_count = len(self._tokenize(text)["input_ids"])
            if token_count > self.token_limit:
                raise RuntimeError(
                    "Attempting to broadcast an axis by a dimension other than 1. "
                    f"512 by {token_count}"
                )
        return f"ACCENT({text})"


def test_manual_mode_returns_input_without_processing():
    accentizer = FakeAccentizer()
    text = "редкое слово"

    result = process_text_with_accent(text, accentizer, "manual")

    assert result == text
    assert accentizer.calls == []


def test_auto_mode_skips_when_plus_exists():
    accentizer = FakeAccentizer()
    text = "редк+ое слово"

    result = process_text_with_accent(text, accentizer, "auto")

    assert result == text
    assert accentizer.calls == []


def test_auto_mode_processes_without_plus():
    accentizer = FakeAccentizer()
    text = "редкое слово"

    result = process_text_with_accent(text, accentizer, "auto")

    assert result == "ACCENT(редкое слово)"
    assert accentizer.calls == [text]


def test_hybrid_mode_preserves_plus_tokens():
    accentizer = FakeAccentizer()
    text = "редк+ое слово"

    result = process_text_with_accent(text, accentizer, "hybrid")

    assert result == "ACCENT(редк+ое слово)"
    assert accentizer.calls == ["ZZXKEEP0ZZX слово"]


def test_hybrid_mode_preserves_multiple_plus_tokens():
    accentizer = FakeAccentizer()
    text = "редк+ое слово и втор+ое"

    result = process_text_with_accent(text, accentizer, "hybrid")

    assert result == "ACCENT(редк+ое слово и втор+ое)"
    assert accentizer.calls == ["ZZXKEEP0ZZX слово и ZZXKEEP1ZZX"]


def test_auto_mode_splits_long_text_by_token_limit():
    accentizer = FakeAccentizer(token_limit=8, fail_on_too_long=True)
    text = "раз два три четыре пять шесть семь восемь девять десять"

    result = process_text_with_accent(text, accentizer, "auto")

    assert (
        result
        == "ACCENT(раз два три четыре пять)ACCENT(шесть семь восемь девять десять)"
    )
    assert accentizer.calls == [
        "раз два три четыре пять шесть семь восемь девять десять",
        "раз два три четыре пять",
        "шесть семь восемь девять десять",
    ]


def test_auto_mode_warns_when_falling_back_to_chunking():
    accentizer = FakeAccentizer(token_limit=6, fail_on_too_long=True)
    warnings = []

    result = process_text_with_accent(
        "раз два три четыре пять шесть",
        accentizer,
        "auto",
        warn_fn=warnings.append,
    )

    assert result == "ACCENT(раз два три)ACCENT(четыре пять шесть)"
    assert warnings == ["Accent model input is too long; processing in smaller chunks."]
