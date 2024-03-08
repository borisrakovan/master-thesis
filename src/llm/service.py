from src.llm.cache import file_cache
from src.llm.schema import ChatCompletionParameters
from src.llm.clients.openai import OpenAIClient, AnyscaleClient
from src.llm.clients.vertexai import VertexAIClient
from src.llm.enums import ChatModel
from src.llm.messages import Message, AIMessage
from src.llm.statistics import LlmStatistics, merge_multiple_llm_statistics


class LlmService:
    """
    Service for interacting with the LLM APIs.
    This class is also responsible for response caching and statistics collection.
    """

    def __init__(self):
        self._openai_client = OpenAIClient()
        self._anyscale_client = AnyscaleClient()
        self._vertexai_client = VertexAIClient()

        self._clients = {
            ChatModel.GPT_35: self._openai_client,
            ChatModel.GPT_4: self._openai_client,
            ChatModel.LLAMA_7B: self._anyscale_client,
            ChatModel.LLAMA_13B: self._anyscale_client,
            ChatModel.LLAMA_70B: self._anyscale_client,
            ChatModel.GEMINI_PRO: self._vertexai_client,
            ChatModel.PaLM_2: self._vertexai_client,
        }

    @file_cache(namespace="llm_client.chat_completion", is_method=True)
    async def create_chat_completion(
        self, messages: list[Message], parameters: ChatCompletionParameters
    ) -> AIMessage:
        try:
            client = self._clients[parameters.model]
        except KeyError:
            raise ValueError(f"Unsupported model: {parameters.model}")

        return await client.create_chat_completion(messages, parameters)

    def collect_statistics(self) -> LlmStatistics:
        return merge_multiple_llm_statistics(
            self._openai_client.statistics, self._anyscale_client.statistics
        )
