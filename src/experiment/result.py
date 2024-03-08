from functools import cached_property
from typing import Any, Annotated

from pydantic import BaseModel, computed_field, WrapSerializer, ConfigDict
from pydantic_core.core_schema import SerializerFunctionWrapHandler

from src.evaluation_tasks.schema import SampleResult
from src.experiment.utils import compute_confidence_interval, compute_initial_sample_size
from src.settings import settings


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


class EvaluationStatistics(BaseModel):

    num_correct: int
    num_incorrect: int
    num_invalid: int
    total: int
    accuracy: RoundedFloat
    pct_invalid: RoundedFloat
    confidence_interval: ConfidenceInterval
    recommended_sample_sizes: dict[str, int]


class EvaluatedSamples(BaseModel):
    correct: list[SampleResult]
    incorrect: list[SampleResult]
    invalid: list[SampleResult]


class EvaluationResult(BaseModel):

    model_config = ConfigDict(frozen=True)

    samples: EvaluatedSamples

    @computed_field
    @cached_property
    def evaluation_statistics(self) -> EvaluationStatistics:
        return EvaluationStatistics(
            num_correct=self.num_correct,
            num_incorrect=self.num_incorrect,
            num_invalid=self.num_invalid,
            total=self.total,
            accuracy=self.accuracy,
            pct_invalid=self.pct_invalid,
            confidence_interval=self.confidence_interval,
            recommended_sample_sizes=self.recommended_sample_sizes
        )

    @cached_property
    def num_correct(self) -> int:
        return len(self.samples.correct)

    @cached_property
    def num_incorrect(self) -> int:
        return len(self.samples.incorrect)

    @cached_property
    def num_invalid(self) -> int:
        return len(self.samples.invalid)

    @cached_property
    def total(self) -> int:
        return self.num_correct + self.num_incorrect + self.num_invalid

    @cached_property
    def accuracy(self) -> float:
        return self.num_correct / self.total if self.total else 0

    @cached_property
    def pct_invalid(self) -> float:
        return self.num_invalid / self.total if self.total else 0

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
