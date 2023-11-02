from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, Field

from src.experiment.result import ModelResult

from src.llm.enums import ChatModel
from src.llm.prompt_template import SystemPromptTemplate, UserPromptTemplate
from src.llm.statistics import LlmStatistics
from src.evaluation_tasks.enums import TaskType


class ExperimentType(StrEnum):
    """Classification of experiment based on the evaluated prompt engineering recommendation"""

    INSTRUCTION_CONTEXT_SEPARATION = "INSTRUCTION_CONTEXT_SEPARATION"
    DO_VS_DONT_FORMULATION = "DO_VS_DONT_FORMULATION"
    INSTRUCTION_ITEMIZATION = "INSTRUCTION_ITEMIZATION"


class UserPrompt(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    template: UserPromptTemplate = Field(exclude=True)


class ExperimentPrompts(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    system: SystemPromptTemplate = Field(exclude=True)
    user: list[UserPrompt]


class ExperimentDefinition(BaseModel):
    """A definition of a single experiment"""

    slug: str
    name: str
    experiment_type: ExperimentType
    task_type: TaskType
    models: list[ChatModel]
    prompts: ExperimentPrompts

    @classmethod
    def from_file(cls, file: Path) -> Self:

        try:
            with file.open(encoding="utf-8") as inp:
                file_data = yaml.safe_load(inp)
        except FileNotFoundError as exc:
            raise ValueError(f"Experiment definition file {file=} does not exist") from exc

        try:
            experiment_data = {
                **file_data,
                "slug": file.stem,
                "prompts": {
                    "system": SystemPromptTemplate.from_file(
                        filename=file_data["prompts"]["system"]
                    ),
                    "user": [
                        {
                            "name": user_data["name"],
                            "template": UserPromptTemplate.from_file(
                                filename=user_data["file"]
                            )
                        }
                        for user_data in file_data["prompts"]["user"]
                    ]
                }
            }
        except KeyError as exc:
            raise ValueError(f"Invalid experiment definition file {file=}") from exc

        try:
            return cls.model_validate(experiment_data)
        except ValidationError as exc:
            raise ValueError(f"Invalid experiment definition file {file=}") from exc


class ExperimentRunnerOptions(BaseModel):
    random_seed: int
    num_samples: int


class ExperimentRun(BaseModel):
    """A single run of an experiment"""

    experiment: ExperimentDefinition
    options: ExperimentRunnerOptions
    results: dict[ChatModel, ModelResult]
    llm_stats: LlmStatistics
    timestamp: datetime
