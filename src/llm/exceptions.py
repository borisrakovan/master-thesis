from src.llm.enums import ChatModel


class LlmApiError(Exception):
    def __init__(self, model: ChatModel, message: str | None = None):
        self.model = model
        super().__init__(message or f"Connection to LLM API failed for {model}")


class LlmRateLimitError(LlmApiError):
    def __init__(self, model: ChatModel, message: str | None = "Rate limit reached"):
        super().__init__(model, message or f"Rate limit reached for {model}")
