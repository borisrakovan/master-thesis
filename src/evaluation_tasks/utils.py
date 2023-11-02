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
