from pydantic import BaseModel, ConfigDict

from src.llm.enums import ChatModel


class ChatCompletionParameters(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ChatModel
    max_tokens: int
    temperature: float
    stop: list[str] | None = None
