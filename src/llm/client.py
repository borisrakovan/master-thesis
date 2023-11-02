from abc import ABC, abstractmethod

import structlog
from openai import AsyncOpenAI, OpenAIError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, ConfigDict

from src.llm.cache import file_cache
from src.llm.enums import ChatModel
from src.llm.exceptions import LlmApiError
from src.llm.messages import AIMessage, Message
from src.llm.statistics import LlmStatistics
from src.llm.retrying import openai_retry
from src.settings import settings

logger = structlog.get_logger(__name__)


class ChatCompletionParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ChatModel
    max_tokens: int
    temperature: float
    stop: list[str] | None = None


class LlmClient(ABC):

    @abstractmethod
    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        ...


class LlmClientDispatcher(LlmClient):

    def __init__(self):
        openai_client = OpenAIClient()
        anyscale_client = AnyscaleClient()
        self._clients = {
            ChatModel.GPT_35: openai_client,
            ChatModel.GPT_4: openai_client,
            ChatModel.LLAMA_7B: anyscale_client,
            ChatModel.LLAMA_13B: anyscale_client,
            ChatModel.LLAMA_70B: anyscale_client,
        }

    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        try:
            client = self._clients[parameters.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {parameters.model}")

        return await client.create_chat_completion(messages, parameters)


class OpenAISDKClient(LlmClient):

    def __init__(self, api_url: str, api_key: str):
        self._statistics = LlmStatistics()
        self._client = AsyncOpenAI(
            base_url=api_url,
            api_key=api_key,
        )

    @property
    def statistics(self) -> LlmStatistics:
        return self._statistics

    @file_cache(namespace="llm_client.chat_completion", is_method=True)
    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        """Fetch a chat completion using the specified parameters"""

        @openai_retry
        async def chat_completion() -> ChatCompletion:
            return await self._client.chat.completions.create(
                messages=[message.model_dump() for message in messages],
                model=parameters.model,
                max_tokens=parameters.max_tokens,
                temperature=parameters.temperature,
                stop=parameters.stop,
                timeout=settings.openai_chat_completion_timeout,
            )

        try:
            completion = await chat_completion()
        except OpenAIError as exc:
            logger.error(f"Error during chat completion: {exc}", exc_info=True)
            raise LlmApiError() from exc

        self._statistics.update(
            parameters.model,
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )

        return AIMessage(content=completion.choices[0].message.content)


class OpenAIClient(OpenAISDKClient):
    def __init__(self):
        super().__init__(settings.openai_api_base, settings.openai_api_key)


class AnyscaleClient(OpenAISDKClient):
    def __init__(self):
        super().__init__(settings.anyscale_api_base, settings.anyscale_api_key)