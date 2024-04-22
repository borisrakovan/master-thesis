import pytest

from src.evaluation_tasks.utils import split_into_words, parse_qa_label_from_response


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


@pytest.mark.parametrize(
    "response, expected",
    [
        ("The correct answer is D: Albumen.", "D"),
        ("The correct answer is D, because...", "D"),
        ("A is the correct answer", "A"),
        ("The correct answer is E", "E"),
        ("B: a line", "B"),
        ("C: The text of the answer", "C"),
        ("A reasonable choice would be to say that the answer is B: some text", "B"),
        ("The correct answer is (D)", "D"),
        ("The correct answer is (A), because...", "A"),
        ("This is some text that doesn't contain a label", None),
        ("I don't know!", None),
        ("I chose F", None),
    ]
)
def test_parse_qa_label_from_response(response, expected):
    assert parse_qa_label_from_response(response) == expected

