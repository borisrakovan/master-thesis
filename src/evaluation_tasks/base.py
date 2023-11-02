from abc import abstractmethod, ABC
from typing import Iterator, Any

from src.llm.client import LlmClient
from src.llm.enums import ChatModel
from src.llm.messages import Message
from src.llm.prompt_template import PromptTemplate

Sample = Any
Label = Any


class EvaluationTask(ABC):
    """Abstract class representing an evaluation benchmark"""

    def __init__(self, llm_client: LlmClient, system_message: Message):
        self._llm_client = llm_client
        self._system_message = system_message

    @property
    @abstractmethod
    def num_samples(self) -> int:
        """Return the number of samples in the task"""
        ...

    @abstractmethod
    def iter_samples(self) -> Iterator[tuple[Sample, Label]]:
        """Return an iterator over the individual task samples"""
        ...

    @abstractmethod
    async def evaluate(self, sample: Sample, model: ChatModel, prompt: PromptTemplate) -> Label:
        """Evaluate the model on a single sample using the given prompt and return boolean indicating success"""
        ...
