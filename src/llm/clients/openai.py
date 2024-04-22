from openai import AsyncOpenAI, RateLimitError, OpenAIError, InternalServerError, APITimeoutError, APIConnectionError
from openai.types.chat import ChatCompletion
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.llm.clients.base import LlmClient
from src.llm.clients.utils import log_retry_error_attempt
from src.llm.schema import ChatCompletionParameters
from src.llm.exceptions import LlmRateLimitError, LlmApiError
from src.llm.messages import Message, AIMessage
from src.llm.statistics import LlmStatistics
from src.logger import get_logger
from src.settings import settings

logger = get_logger(__name__)


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

    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        """Fetch a chat completion using the specified parameters."""

        @openai_retry
        async def chat_completion() -> ChatCompletion:
            return await self._client.chat.completions.create(
                messages=[message.model_dump() for message in messages],
                model=parameters.model,
                max_tokens=parameters.max_tokens,
                temperature=parameters.temperature,
                stop=parameters.stop,
                timeout=settings.llm_timeout,
            )

        try:
            completion = await chat_completion()
        except RateLimitError as exc:
            logger.error(
                f"Rate limit error during {self.api_name} chat completion: {exc}",
                exc_info=True
            )
            raise LlmRateLimitError(parameters.model) from exc
        except OpenAIError as exc:
            logger.error(
                f"Error during {self.api_name} chat completion: {exc}",
                exc_info=True
            )
            raise LlmApiError(parameters.model) from exc

        self._statistics.update(
            parameters.model,
            completion.usage.prompt_tokens, completion.usage.completion_tokens
        )

        return AIMessage(content=completion.choices[0].message.content)

    @property
    def api_name(self) -> str:
        return self.__class__.__name__.rstrip("Client")


class OpenAIClient(OpenAISDKClient):
    def __init__(self):
        super().__init__(settings.openai_api_base, settings.openai_api_key)


class AnyscaleClient(OpenAISDKClient):
    def __init__(self):
        super().__init__(settings.anyscale_api_base, settings.anyscale_api_key)


openai_retry = retry(
    retry=retry_if_exception_type((RateLimitError, InternalServerError, APITimeoutError, APIConnectionError)),
    stop=stop_after_attempt(settings.llm_max_retries),
    # This will wait 1s, 2s, 4s, 8s and so on
    wait=wait_exponential(min=1, max=60),
    after=log_retry_error_attempt,
    reraise=True,
)
