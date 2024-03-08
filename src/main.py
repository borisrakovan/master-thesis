import asyncio
import logging

from tabulate import tabulate

from src import constants
from src.constants import PROJECT_ROOT
from src.experiment.runner import ExperimentRunner
from src.experiment.schema import ExperimentRun, ExperimentDefinition, ExperimentRunnerOptions
from src.llm.service import LlmService

logging.basicConfig(level=logging.INFO)


def store_experiment_run(experiment_run: ExperimentRun, result_table: str) -> None:
    timestamp = experiment_run.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = constants.RESULT_DIRECTORY / experiment_run.experiment.slug
    exp_path = exp_dir / f"{timestamp}.json"
    table_path = exp_dir / f"{timestamp}.txt"

    exp_dir.mkdir(parents=True, exist_ok=True)

    with exp_path.open("w") as out:
        out.write(experiment_run.model_dump_json(indent=4))

    with table_path.open("w") as out:
        out.write(result_table)

    print(f"Stored experiment run to {exp_path}")


def make_experiment_result_table(experiment_run: ExperimentRun) -> str:
    table_data = []
    user_prompts = [prompt.name for prompt in experiment_run.experiment.prompts.user]

    headers = ["Model"] + user_prompts

    for model, model_result in experiment_run.results.items():

        row = [model.name]
        for evaluation_result in model_result.prompt_results.values():
            cell_data = f"{evaluation_result.accuracy:.1%}"
            cell_data += f" ± {evaluation_result.confidence_interval.margin_of_error:.1%}"
            if evaluation_result.pct_invalid > 1e-3:
                cell_data += f" ({evaluation_result.pct_invalid:.1%} invalid)"

            row.append(cell_data)

        table_data.append(row)

    return tabulate(table_data, headers=headers, tablefmt="grid")


async def main() -> None:

    llm = LlmService()
    runner_options = ExperimentRunnerOptions(random_seed=46, num_samples=1)
    runner = ExperimentRunner(llm=llm, options=runner_options)
    experiment = ExperimentDefinition.from_file(
    # todo finish this one
        # PROJECT_ROOT / "experiments" / "instruction_context_separation_imdb.yml"
        PROJECT_ROOT / "experiments" / "instruction_itemization_imdb.yml"
    )
    experiment_run = await runner.run(experiment)

    result_table = make_experiment_result_table(experiment_run)
    print(result_table)
    store_experiment_run(experiment_run, result_table)


if __name__ == "__main__":
    asyncio.run(main())
