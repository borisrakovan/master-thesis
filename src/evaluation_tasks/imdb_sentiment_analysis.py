import csv
from typing import Iterator

from src.constants import DATA_DIRECTORY
from src.evaluation_tasks.schema import Sample, SampleResult
from src.evaluation_tasks.utils import first_word_occurrence
from src.llm.schema import ChatCompletionParameters
from src.llm.enums import ChatModel
from src.llm.prompt_template import PromptTemplate
from src.evaluation_tasks.base import EvaluationTask
from src.logger import get_logger


logger = get_logger(__name__)


class IMDbSentimentAnalysis(EvaluationTask[str, str]):
    """IMDB Sentiment Analysis classification task"""

    _SOURCE_URL = "https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format/download?datasetVersionNumber=1"  # noqa
    _CLASSES = ["Positive", "Negative"]
    # We use the test set file because it has fewer samples
    _DATASET_PATH = DATA_DIRECTORY / "imdb_sentiment_analysis" / "test.csv"
    _NUM_DATASET_SAMPLES = 5000

    @property
    def num_samples(self) -> int:
        return self._NUM_DATASET_SAMPLES

    def iter_samples(self) -> Iterator[Sample[str, str]]:
        with self._DATASET_PATH.open(encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Sample(
                    input=row["text"],
                    target="Positive" if int(row["label"]) == 1 else "Negative",
                )

    async def evaluate_sample(
        self,
        sample: Sample[str, str],
        model: ChatModel,
        prompt: PromptTemplate
    ) -> SampleResult[str, str]:
        response = await self._llm.create_chat_completion(
            messages=[self._system_message, prompt.format(sample_text=sample.input)],
            parameters=ChatCompletionParameters(
                model=model,
                # Give the model enough tokens to output the class label
                max_tokens=30,
                temperature=0,
            )
        )

        prediction = first_word_occurrence(response.content, self._CLASSES)

        return SampleResult(
            sample=sample,
            prediction=prediction,
            llm_response=response.content,
        )
