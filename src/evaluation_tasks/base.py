from abc import abstractmethod, ABC
from typing import Iterator, TypeVar, Generic

from src.evaluation_tasks.schema import SampleResult, Sample
from src.llm.service import LlmService
from src.llm.enums import ChatModel
from src.llm.messages import Message
from src.llm.prompt_template import PromptTemplate


TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


class EvaluationTask(ABC, Generic[TInput, TTarget]):
    """Abstract class representing an evaluation benchmark"""

    def __init__(self, llm: LlmService, system_message: Message):
        self._llm = llm
        self._system_message = system_message

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Return the number of samples in the task"""
        ...

    @abstractmethod
    def iter_samples(self) -> Iterator[Sample[TInput, TTarget]]:
        """Return an iterator over the individual task samples"""
        ...

    @abstractmethod
    async def evaluate_sample(
        self,
        sample: Sample[TInput, TTarget],
        model: ChatModel,
        prompt: PromptTemplate
    ) -> SampleResult[TInput, TTarget]:
        """Evaluate the model on a single sample using the given prompt"""
        ...
