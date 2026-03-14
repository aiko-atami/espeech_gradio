from espeech.domain.greeting import greet


def test_greet_with_world():
    assert greet("World") == "Hello, World!"


def test_greet_with_empty_name():
    assert greet("") == "Hello, !"


def test_greet_with_name():
    assert greet("Alice") == "Hello, Alice!"
