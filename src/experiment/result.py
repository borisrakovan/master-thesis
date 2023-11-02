from functools import cached_property
from typing import Any, Annotated

from pydantic import BaseModel, computed_field, WrapSerializer, ConfigDict
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from src.experiment.utils import compute_confidence_interval, compute_initial_sample_size
from src.settings import settings
from src.evaluation_tasks.base import Sample, Label


class ValidSampleResult(BaseModel):
    sample: Sample
    label: Label
    predicted: Label


class InvalidSampleResult(BaseModel):
    sample: Sample
    label: Label
    response: str


SampleResult = ValidSampleResult | InvalidSampleResult


def _round_float(value: Any, nxt: SerializerFunctionWrapHandler) -> float:
    return round(nxt(value), 2)


RoundedFloat = Annotated[float, WrapSerializer(_round_float, when_used='json')]


class ConfidenceInterval(BaseModel):

    bounds: tuple[RoundedFloat, RoundedFloat]
    confidence_level: float

    @computed_field
    @property
    def margin_of_error(self) -> RoundedFloat:
        return (self.bounds[1] - self.bounds[0]) / 2


class EvaluationResult(BaseModel):

    model_config = ConfigDict(frozen=True)

    correct_samples: list[SampleResult]
    incorrect_samples: list[SampleResult]
    invalid_samples: list[SampleResult]

    @computed_field
    @cached_property
    def num_correct(self) -> int:
        return len(self.correct_samples)

    @computed_field
    @cached_property
    def num_incorrect(self) -> int:
        return len(self.incorrect_samples)

    @computed_field
    @cached_property
    def num_invalid(self) -> int:
        return len(self.invalid_samples)

    @computed_field
    @cached_property
    def total(self) -> int:
        return self.num_correct + self.num_incorrect + self.num_invalid

    @computed_field(return_type=RoundedFloat)
    @cached_property
    def accuracy(self) -> float:
        return self.num_correct / self.total if self.total else 0

    @computed_field(return_type=RoundedFloat)
    @cached_property
    def pct_invalid(self) -> float:
        return self.num_invalid / self.total if self.total else 0

    @computed_field
    @cached_property
    def confidence_interval(self) -> ConfidenceInterval:
        """Computes the confidence interval for the accuracy at a predefined confidence level."""
        return ConfidenceInterval(
            bounds=compute_confidence_interval(
                self.num_correct,
                self.total,
                confidence_level=settings.accuracy_confidence_interval_level
            ),
            confidence_level=settings.accuracy_confidence_interval_level,
        )

    @computed_field
    @cached_property
    def recommended_sample_sizes(self) -> dict[str, int]:
        """
        Computes recommended sample sizes for selected margins of error
        at a predefined confidence level.
        """
        margins_of_error = [0.01, 0.03, 0.05]
        sample_sizes = {
            f"±{int(moe * 100)}%": compute_initial_sample_size(
                estimated_proportion=self.accuracy,
                margin_of_error=moe,
                confidence_level=settings.accuracy_confidence_interval_level
            )
            for moe in margins_of_error
        }
        return sample_sizes


class ModelResult(BaseModel):
    prompt_results: dict[str, EvaluationResult]
