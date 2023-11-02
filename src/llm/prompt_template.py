from abc import ABC, abstractmethod
from typing import Self, Any
from jinja2 import Environment, StrictUndefined, Template, FileSystemLoader

from src.constants import PROJECT_ROOT
from src.llm.messages import Message, MessageRole

_BASE_ENV = Environment(
    loader=FileSystemLoader(
        searchpath=[
            PROJECT_ROOT / "experiments" / "templates"
        ]
    ),
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


class PromptTemplate(ABC):
    """Class used for rendering Jinja2 prompt templates"""

    def __init__(self, template: Template):
        self._template = template

    def format(self, **kwargs: Any) -> Message:
        """Format the prompt with the specified arguments"""
        message_content = self._template.render(**kwargs)
        return Message(role=self.message_role, content=message_content)

    @property
    @abstractmethod
    def message_role(self) -> MessageRole:
        pass

    @classmethod
    def from_file(cls, filename: str) -> Self:
        """Load a jinja prompt template from file"""
        template = _BASE_ENV.get_template(filename)
        return cls(template)


class SystemPromptTemplate(PromptTemplate):

    message_role = MessageRole.SYSTEM


class UserPromptTemplate(PromptTemplate):

    message_role = MessageRole.USER


class PromptTemplateSequence:
    """Defines a sequence of prompts with their templates"""

    def __init__(self, prompt_templates: list[PromptTemplate]):
        self._prompt_templates = prompt_templates

    def format(self, **kwargs: Any) -> list[Message]:
        """Format the prompt templates with the specified arguments"""
        return [template.format(**kwargs) for template in self._prompt_templates]
