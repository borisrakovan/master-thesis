import csv
from functools import cached_property
from typing import Iterator, TypeAlias, Literal

from pydantic import BaseModel

from src.constants import DATA_DIRECTORY
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.schema import Sample, SampleResult
from src.evaluation_tasks.utils import parse_qa_label_from_response
from src.llm.enums import ChatModel
from src.llm.messages import Message
from src.llm.prompt_template import PromptTemplate
from src.llm.schema import ChatCompletionParameters
from src.llm.service import LlmService


CosmosTarget: TypeAlias = Literal['A', 'B', 'C', 'D']


class CosmosInput(BaseModel):
    context: str
    question: str
    choices: dict[CosmosTarget, str]


class CosmosMultipleChoiceQA(EvaluationTask[CosmosInput, CosmosTarget]):

    _DATASET_PATH = DATA_DIRECTORY / "cosmos_qa" / "train.csv"

    def __init__(self, llm: LlmService, system_message: Message):
        super().__init__(llm, system_message)

    @cached_property
    def num_samples(self) -> int:
        with self._DATASET_PATH.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)

    def iter_samples(self) -> Iterator[Sample[CosmosInput, CosmosTarget]]:

        label_map = "ABCD"

        # utf-8-sig
        with self._DATASET_PATH.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield Sample(
                    input=CosmosInput(
                        context=row["context"],
                        question=row["question"],
                        choices={
                            "A": row["answer0"],
                            "B": row["answer1"],
                            "C": row["answer2"],
                            "D": row["answer3"],
                        },
                    ),
                    target=label_map[int(row["label"])],
                )

    async def evaluate_sample(
        self,
        sample: Sample[CosmosInput, CosmosTarget],
        model: ChatModel,
        prompt: PromptTemplate
    ) -> SampleResult[CosmosInput, CosmosTarget]:
        formatted_prompt = prompt.format(input=sample.input)
        response = await self._llm.create_chat_completion(
            messages=[self._system_message, formatted_prompt],
            parameters=ChatCompletionParameters(
                model=model,
                max_tokens=100,
                temperature=0.,
            ),
        )

        prediction = parse_qa_label_from_response(response.content.strip())

        return SampleResult(
            sample=sample,
            prediction=prediction,
            llm_response=response.content,
        )
