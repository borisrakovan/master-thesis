import re


def split_into_words(text: str) -> list[str]:
    """Splits the text into words, preserving internal apostrophes and hyphens. Returns a list of words."""
    return re.findall(r'\b[\w\']+(?:-\w+)*\b', text)


def first_word_occurrence(text: str, words: list[str]) -> str | None:
    """Returns the first word from the list that occurs in the text or None if none of the words are in the text"""

    text_words = split_into_words(text)
    # Convert list to set for faster lookup
    words_set = set(word.lower() for word in words)

    for word in text_words:
        if word.lower() in words_set:
            return word
    return None


def parse_qa_label_from_response(response: str) -> str | None:
    # First, look for patterns like 'X:'
    pattern_1 = re.compile(r'\b([ABCDE]):')
    match = pattern_1.search(response)
    if match:
        return match.group(1)

    # If not found, look for patterns like '(X)'
    pattern_2 = re.compile(r'\(([ABCDE])\)')
    match = pattern_2.search(response)
    if match:
        return match.group(1)

    # If not found, look for 'X' with specific follow-up characters to minimize false positives
    pattern_2 = re.compile(fr'\b([ABCDE])([\s,.])?')
    match = pattern_2.search(response)
    if match:
        return match.group(1)

    return None
