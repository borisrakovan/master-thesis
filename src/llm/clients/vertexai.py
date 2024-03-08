import functools

import vertexai
from google.api_core.exceptions import ServerError, GoogleAPIError, TooManyRequests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from vertexai.generative_models import GenerativeModel, Content, Part, GenerationConfig, \
    HarmCategory, HarmBlockThreshold

from src.llm.clients.base import LlmClient
from src.llm.clients.utils import log_retry_error_attempt
from src.llm.enums import ChatModel
from src.llm.exceptions import LlmRateLimitError, LlmApiError
from src.llm.schema import ChatCompletionParameters
from src.llm.messages import Message, AIMessage, MessageRole
from src.llm.statistics import LlmStatistics
from src.logger import get_logger
from src.settings import settings


logger = get_logger(__name__)


class VertexAIClient(LlmClient):

    def __init__(self):
        vertexai.init(
            project=settings.vertexai_project_id,
            location=settings.vertexai_location,
        )
        # TODO: figure out how to implement this
        self._statistics = LlmStatistics()

    async def create_chat_completion(
        self,
        messages: list[Message],
        parameters: ChatCompletionParameters
    ) -> AIMessage:
        model = self._create_model(parameters.model)

        @vertexai_retry
        async def chat_completion() -> str:
            response = await model.generate_content_async(
                contents=self._convert_messages(messages),
                generation_config=GenerationConfig(
                    temperature=parameters.temperature,
                    max_output_tokens=parameters.max_tokens,
                    stop_sequences=parameters.stop,
                )
            )
            return response.text

        try:
            completion = await chat_completion()
        except TooManyRequests as exc:
            logger.error(
                f"Rate limit error during VertexAI chat completion: {exc}",
                exc_info=True
            )
            raise LlmRateLimitError(parameters.model) from exc
        except GoogleAPIError as exc:
            logger.error(f"Error during VertexAI chat completion: {exc}", exc_info=True)
            raise LlmApiError(parameters.model) from exc

        return AIMessage(content=completion)

    @classmethod
    @functools.cache
    def _create_model(cls, model: ChatModel) -> GenerativeModel:
        safety_config = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }
        return GenerativeModel(model, safety_settings=safety_config)


    @classmethod
    def _convert_messages(cls, messages: list[Message]) -> list[Content]:

        assert len(messages) > 0, "Messages must not be empty"

        if (system_message := messages[0]).role == MessageRole.SYSTEM:
            assert len(messages) > 1, "System message must be followed by a user message"
            messages = messages[1:]
            # VertexAI doesn't support system message,
            # so we append the system message to the beginning of the user message
            messages[0].content = f"{system_message}\n\n{messages[0].content}"
        return [
            Content(
                role=cls._convert_message_role(message.role),
                parts=[Part.from_text(message.content)]
            ) for message in messages
        ]

    @staticmethod
    def _convert_message_role(role: MessageRole):
        if role == MessageRole.USER:
            return "user"
        elif role == MessageRole.ASSISTANT:
            return "model"
        else:
            raise ValueError(f"Unsupported message role: {role}")

    @property
    def statistics(self) -> LlmStatistics:
        return self._statistics


vertexai_retry = retry(
    # Sometimes GCP returns IndexError: list index out of range
    # on return self.candidates[0].text (probably a wrongly handled rate limit error?)
    retry=retry_if_exception_type((ServerError, TooManyRequests, IndexError)),
    stop=stop_after_attempt(settings.llm_max_retries),
    # This will wait 1s, 2s, 4s, 8s and so on
    wait=wait_exponential(min=1, max=60),
    after=log_retry_error_attempt,
    reraise=True,
)
