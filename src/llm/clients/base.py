from abc import ABC, abstractmethod

from src.llm.messages import Message, AIMessage
from src.llm.schema import ChatCompletionParameters
from src.llm.statistics import LlmStatistics


class LlmClient(ABC):

    @abstractmethod
    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        ...

    @property
    @abstractmethod
    def statistics(self) -> LlmStatistics:
        ...


