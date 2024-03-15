import re
from typing import Iterator, TypeAlias

from datasets import load_dataset
from pydantic import BaseModel

from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.schema import Sample, SampleResult
from src.llm.enums import ChatModel
from src.llm.messages import Message
from src.llm.prompt_template import PromptTemplate
from src.llm.schema import ChatCompletionParameters
from src.llm.service import LlmService


ArcTarget: TypeAlias = str


class ArcInput(BaseModel):
    question: str
    choices: dict[ArcTarget, str]


def parse_label_from_response(response: str) -> str | None:
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


class ARCMultipleChoiceQA(EvaluationTask[ArcInput, ArcTarget]):

    _possible_labels = ['A', 'B', 'C', 'D', 'E']

    def __init__(self, llm: LlmService, system_message: Message):
        super().__init__(llm, system_message)
        self.dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")["train"]

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

        prediction = parse_label_from_response(response.content.strip())

        return SampleResult(
            sample=sample,
            prediction=prediction,
            llm_response=response.content,
        )
