from typing import Generic, TypeVar

from pydantic import BaseModel


TInput = TypeVar("TInput")
TTarget = TypeVar("TTarget")


class Sample(BaseModel, Generic[TInput, TTarget]):
    """Generic class representing a sample in an evaluation task"""

    input: TInput
    target: TTarget


class SampleResult(BaseModel, Generic[TInput, TTarget]):
    sample: Sample[TInput, TTarget]
    # None if model generated invalid response
    prediction: TTarget | None
    model_response: str

    @property
    def is_valid(self) -> bool:
        return self.prediction is not None

    @property
    def is_correct(self) -> bool:
        return self.prediction == self.sample.target
