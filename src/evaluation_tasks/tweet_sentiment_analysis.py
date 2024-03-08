import csv
from typing import Iterator

import requests
from src.constants import DATA_DIRECTORY
from src.llm.schema import ChatCompletionParameters
from src.llm.enums import ChatModel
from src.llm.prompt_template import PromptTemplate
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.exceptions import InvalidModelResponseError
from src.evaluation_tasks.utils import first_word_occurrence
from src.logger import get_logger


logger = get_logger(__name__)


class TweetSentimentAnalysis(EvaluationTask):
    """Twitter Airlines Sentiment Analysis classification task"""

    _SOURCE_URL = (
        "https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis/download?datasetVersionNumber=2"
    )  # noqa
    _CLASSES = ["Positive", "Negative", "Neutral"]
    _DATASET_PATH = DATA_DIRECTORY / "twitter_airline_sentiment_analysis" / "Tweets.csv"
    _NUM_DATASET_SAMPLES = 14800

    def _download(self) -> None:
        """Download the dataset if it's not already downloaded"""
        logger.info(f"Downloading {TweetSentimentAnalysis.__name__} dataset to {self._DATASET_PATH}")
        if not self._DATASET_PATH.exists():
            self._DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
            response = requests.get(self._SOURCE_URL)
            response.raise_for_status()
            with self._DATASET_PATH.open("wb") as f:
                f.write(response.content)

    @property
    def num_samples(self) -> int:
        return self._NUM_DATASET_SAMPLES

    def iter_samples(self) -> Iterator[tuple[str, str]]:
        self._download()
        with self._DATASET_PATH.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Capitalize the class name because LLMs are more likely to return a capitalized output
                sample = row["text"]
                label = row["airline_sentiment"].capitalize()
                yield sample, label

    async def evaluate_sample(self, sample: str, model: ChatModel, prompt: PromptTemplate) -> str:
        response = await self._llm.create_chat_completion(
            messages=[self._system_message, prompt.format(sample_text=sample)],
            parameters=ChatCompletionParameters(
                model=model,
                # Give the model enough tokens to output the class label
                max_tokens=30,
                temperature=0,
            ),
        )

        predicted_class = first_word_occurrence(response.content, self._CLASSES)

        if predicted_class is None:
            raise InvalidModelResponseError(model, response.content)

        return predicted_class
