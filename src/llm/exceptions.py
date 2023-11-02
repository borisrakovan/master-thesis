class LlmApiError(Exception):
    def __init__(self, message: str | None = "Connection to LLM API failed"):
        super().__init__(message)
