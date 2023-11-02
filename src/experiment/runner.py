import asyncio
from collections import defaultdict
from datetime import datetime

import numpy as np
import structlog

from src.experiment.schema import ExperimentDefinition, ExperimentRun, ExperimentRunnerOptions, UserPrompt
from src.experiment.utils import batched, random_sample
from src.llm.client import LlmClient
from src.llm.enums import ChatModel
from src.llm.prompt_template import PromptTemplate
from src.experiment.result import EvaluationResult, ModelResult, ValidSampleResult, \
    InvalidSampleResult
from src.evaluation_tasks.base import EvaluationTask, Label, Sample
from src.evaluation_tasks.exceptions import InvalidModelResponseError
from src.evaluation_tasks.factory import create_evaluation_task

logger = structlog.get_logger(__name__)


class ExperimentRunner:
    # Anyscale currently supports up to 30 concurrent requests
    # We're calling three models concurrently, so we need to limit the concurrency to 10
    # https://docs.endpoints.anyscale.com/text-generation/query-a-model/
    _MAX_MODEL_API_CONCURRENCY = defaultdict(
        lambda: 10,
        {
            ChatModel.GPT_35: 1,
            ChatModel.GPT_4: 5,
        },
    )

    def __init__(self, llm_client: LlmClient, options: ExperimentRunnerOptions):
        self._llm_client = llm_client
        self._options = options
        self._rng = np.random.default_rng(options.random_seed)

    async def run(self, experiment: ExperimentDefinition) -> ExperimentRun:
        evaluation_task = create_evaluation_task(experiment, llm_client=self._llm_client)

        sample_iterator = evaluation_task.iter_samples()
        samples = list(
            random_sample(
                iterator=sample_iterator,
                iterator_size=evaluation_task.num_samples,
                rng=self._rng,
                sample_size=self._options.num_samples
            )
        )

        model_results = await asyncio.gather(
            *[
                self._run_for_model(model, evaluation_task, samples, experiment.prompts.user)
                for model in experiment.models
            ]
        )

        results = dict(zip(experiment.models, model_results))

        return ExperimentRun(
            timestamp=datetime.now(),
            options=self._options,
            experiment=experiment,
            results=results,
            llm_stats=self._llm_client.statistics,
        )

    async def _run_for_model(
        self,
        model: ChatModel,
        evaluation_task: EvaluationTask,
        samples: list[tuple[Sample, Label]],
        user_prompts: list[UserPrompt],
    ) -> ModelResult:
        prompt_results: dict[str, EvaluationResult] = {}
        for prompt in user_prompts:
            prompt_result = await self._run_for_prompt(model, evaluation_task, samples, prompt.template)
            prompt_results[prompt.name] = prompt_result

        return ModelResult(prompt_results=prompt_results)

    async def _run_for_prompt(
        self,
        model: ChatModel,
        evaluation_task: EvaluationTask,
        samples: list[tuple[Sample, Label]],
        prompt: PromptTemplate,
    ) -> EvaluationResult:
        # Batch the samples into groups executed concurrently
        sample_batches = batched(samples, batch_size=self._MAX_MODEL_API_CONCURRENCY[model])

        invalid_samples, correct_samples, incorrect_samples = [], [], []

        for sample_batch in sample_batches:
            samples, labels = zip(*sample_batch)
            # TODO: use asyncio.TaskGroup (handle partial exceptions better)
            results: tuple[Label | BaseException] = await asyncio.gather(
                *[evaluation_task.evaluate(sample, model, prompt) for sample in samples],
                return_exceptions=True,
            )
            # TODO: move this part to evaluation task (maybe inherit from classification task later
            for idx, result in enumerate(results):
                # Propagate unexpected exceptions
                if isinstance(result, Exception) and not isinstance(result, InvalidModelResponseError):
                    raise result

                if isinstance(result, InvalidModelResponseError):
                    sample_result = InvalidSampleResult(
                        sample=samples[idx],
                        label=labels[idx],
                        response=result.response,
                    )
                    invalid_samples.append(sample_result)
                else:
                    sample_result = ValidSampleResult(
                        sample=samples[idx],
                        label=labels[idx],
                        predicted=result,
                    )
                    if result == labels[idx]:
                        correct_samples.append(sample_result)
                    else:
                        incorrect_samples.append(sample_result)

        return EvaluationResult(
            correct_samples=correct_samples,
            incorrect_samples=incorrect_samples,
            invalid_samples=invalid_samples,
        )
