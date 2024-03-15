import pytest

from src.evaluation_tasks.arc_multiple_choice_qa import parse_label_from_response


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
def test_parse_label_from_response(response, expected):
    assert parse_label_from_response(response) == expected
