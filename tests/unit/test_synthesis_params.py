from espeech.domain.synthesis_params import normalize_seed, sanitize_infer_params


def test_normalize_seed_uses_fallback_for_invalid_value():
    seed = normalize_seed(float("nan"))

    assert isinstance(seed, int)
    assert seed >= 0


def test_normalize_seed_keeps_valid_seed():
    assert normalize_seed(123) == 123


def test_sanitize_infer_params_clamps_values():
    cross_fade, nfe_step, speed = sanitize_infer_params(-5, 999, 0)

    assert cross_fade == 0.0
    assert nfe_step == 64
    assert speed == 0.3


def test_sanitize_infer_params_defaults_on_bad_inputs():
    cross_fade, nfe_step, speed = sanitize_infer_params("bad", None, "nan")

    assert cross_fade == 0.15
    assert nfe_step == 32
    assert speed == 1.0
