from enum import StrEnum


class TaskType(StrEnum):
    """Type of the evaluation benchmark"""

    TWEET_SENTIMENT_ANALYSIS = "TWEET_SENTIMENT_ANALYSIS"
    IMDB_SENTIMENT_ANALYSIS = "IMDB_SENTIMENT_ANALYSIS"
    ARC_MULTIPLE_CHOICE_QA = "ARC_MULTIPLE_CHOICE_QA"
