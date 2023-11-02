import csv
from typing import Iterator

import structlog
from src.constants import DATA_DIRECTORY
from src.evaluation_tasks.utils import first_word_occurrence
from src.llm.enums import ChatModel
from src.llm.prompt_template import PromptTemplate
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.exceptions import InvalidModelResponseError

logger = structlog.get_logger(__name__)


class IMDbSentimentAnalysis(EvaluationTask):
    """Twitter Airlines Sentiment Analysis classification task"""

    _SOURCE_URL = "https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format/download?datasetVersionNumber=1"  # noqa
    _CLASSES = ["Positive", "Negative"]
    # We use the test set file because it has fewer samples
    _DATASET_PATH = DATA_DIRECTORY / "imdb_sentiment_analysis" / "test.csv"
    _NUM_DATASET_SAMPLES = 5000

    @property
    def num_samples(self) -> int:
        return self._NUM_DATASET_SAMPLES

    def iter_samples(self) -> Iterator[tuple[str, str]]:
        with self._DATASET_PATH.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = row["text"]
                label = "Positive" if int(row["label"]) == 1 else "Negative"
                yield sample, label

    async def evaluate(self, sample: str, model: ChatModel, prompt: PromptTemplate) -> str:
        response = await self._llm_client.chat_completion(
            messages=[self._system_message, prompt.format(sample_text=sample)],
            model=model,
            # Give the model enough tokens to output the class label
            max_tokens=30,
            temperature=0,
        )

        # Convert the gold label to the class name
        predicted_class = first_word_occurrence(response.content, self._CLASSES)

        if predicted_class not in self._CLASSES:
            raise InvalidModelResponseError(model, response.content)

        return predicted_class
