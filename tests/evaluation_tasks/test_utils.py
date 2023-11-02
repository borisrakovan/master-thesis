import pytest

from src.evaluation_tasks.utils import split_into_words


@pytest.mark.parametrize("input_text,expected_result", [
    ("Hello, world!", ["Hello", "world"]),
    ("It's a beautiful day.", ["It's", "a", "beautiful", "day"]),
    ("High-quality, well-written code.", ["High-quality", "well-written", "code"]),
    ("Spaces    and tabs\tare ignored.", ["Spaces", "and", "tabs", "are", "ignored"]),
    ("End-of-sentence. Another start?", ["End-of-sentence", "Another", "start"]),
    ("Hyphenated-words are tricky.", ["Hyphenated-words", "are", "tricky"]),
    ("Special characters & symbols are #1!", ["Special", "characters", "symbols", "are", "1"]),
    ("Based on the review, I would classify it as Negative.", ["Based", "on", "the", "review", "I", "would", "classify", "it", "as", "Negative"]),
])
def test_split_into_words(input_text, expected_result):
    assert split_into_words(input_text) == expected_result
