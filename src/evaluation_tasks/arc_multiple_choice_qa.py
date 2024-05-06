from typing import Iterator, TypeAlias, Literal

from datasets import load_dataset, concatenate_datasets
from pydantic import BaseModel

from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.schema import Sample, SampleResult
from src.evaluation_tasks.utils import parse_qa_label_from_response
from src.llm.enums import ChatModel
from src.llm.messages import Message
from src.llm.prompt_template import PromptTemplate
from src.llm.schema import ChatCompletionParameters
from src.llm.service import LlmService


ArcTarget: TypeAlias = Literal['A', 'B', 'C', 'D', 'E']


class ArcInput(BaseModel):
    question: str
    choices: dict[ArcTarget, str]


class ARCMultipleChoiceQA(EvaluationTask[ArcInput, ArcTarget]):

    _possible_labels = ['A', 'B', 'C', 'D', 'E']

    def __init__(self, llm: LlmService, system_message: Message):
        super().__init__(llm, system_message)
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        self.dataset = concatenate_datasets(
            [dataset["train"], dataset["test"], dataset["validation"]]
        )

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def iter_samples(self) -> Iterator[Sample[ArcInput, ArcTarget]]:

        label_map = dict(zip('12345', self._possible_labels))

        def map_label(label: str) -> str:
            new_label = label_map.get(label, label)
            if new_label not in self._possible_labels:
                raise ValueError(f"Invalid label: {label}")
            return new_label

        def map_labels(labels: list[str]) -> list[str]:
            return list(map(map_label, labels))

        for item in self.dataset:
            # Map the target labels to make them consistent
            targets = map_labels(item["choices"]["label"])
            yield Sample(
                input=ArcInput(
                    question=item["question"],
                    choices=dict(zip(targets, item["choices"]["text"]))
                ),
                target=map_label(item["answerKey"]),
            )

    async def evaluate_sample(
        self,
        sample: Sample[ArcInput, ArcTarget],
        model: ChatModel,
        prompt: PromptTemplate
    ) -> SampleResult[str, str]:
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
