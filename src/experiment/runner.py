import asyncio
from collections import defaultdict
from datetime import datetime

import numpy as np

from src.evaluation_tasks.schema import Sample, SampleResult
from src.experiment.schema import ExperimentDefinition, ExperimentRun, ExperimentRunnerOptions, UserPrompt
from src.experiment.utils import batched, random_sample
from src.llm.service import LlmService
from src.llm.enums import ChatModel
from src.llm.prompt_template import PromptTemplate
from src.experiment.result import EvaluationResult, ModelResult, EvaluatedSamples
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.factory import create_evaluation_task
from src.logger import get_logger

logger = get_logger(__name__)


class ExperimentRunner:
    # Anyscale currently supports up to 30 concurrent requests
    # We're calling three models concurrently, so we need to limit the concurrency to 10
    # https://docs.endpoints.anyscale.com/text-generation/query-a-model/
    _MAX_MODEL_API_CONCURRENCY = defaultdict(
        lambda: 10,
        {
            ChatModel.GPT_35: 10,
            ChatModel.GPT_4: 5,
            ChatModel.GEMINI_PRO: 5,
        },
    )

    def __init__(self, llm: LlmService, options: ExperimentRunnerOptions):
        self._llm = llm
        self._options = options
        self._rng = np.random.default_rng(options.random_seed)

    async def run(self, experiment: ExperimentDefinition) -> ExperimentRun:
        evaluation_task = create_evaluation_task(experiment, llm=self._llm)

        if evaluation_task.num_samples < self._options.num_samples:
            raise ValueError(
                f"Dataset size {evaluation_task.num_samples} is lower "
                f"than the number of experiment samples {self._options.num_samples}."
            )

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
            llm_stats=self._llm.collect_statistics(),
        )

    async def _run_for_model(
        self,
        model: ChatModel,
        evaluation_task: EvaluationTask,
        samples: list[Sample],
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
        samples: list[Sample],
        prompt: PromptTemplate,
    ) -> EvaluationResult:
        # Batch the samples into groups executed concurrently
        sample_batches = batched(samples, batch_size=self._MAX_MODEL_API_CONCURRENCY[model])

        invalid_samples, correct_samples, incorrect_samples = [], [], []
        num_processed = 0

        for sample_batch in sample_batches:
            batch_list = list(sample_batch)
            logger.info(f"Running batch of {len(batch_list)} samples for model {model}")
            results: tuple[SampleResult] = await asyncio.gather(
                *[evaluation_task.evaluate_sample(sample, model, prompt) for sample in batch_list]
            )

            for idx, result in enumerate(results):

                if not result.is_valid:
                    invalid_samples.append(result)
                elif not result.is_correct:
                    incorrect_samples.append(result)
                else:
                    correct_samples.append(result)

            num_processed += len(results)
            logger.info(f"Processed {num_processed} samples for model {model}")

        return EvaluationResult(
            samples=EvaluatedSamples(
                correct=correct_samples,
                incorrect=incorrect_samples,
                invalid=invalid_samples,
            )
        )
