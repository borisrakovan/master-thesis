from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    role: MessageRole
    content: str


class AIMessage(Message):
    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT


class UserMessage(Message):
    role: Literal[MessageRole.USER] = MessageRole.USER


class SystemMessage(Message):
    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM
