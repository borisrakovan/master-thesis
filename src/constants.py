from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIRECTORY = PROJECT_ROOT / "data"
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

CACHE_DIRECTORY = PROJECT_ROOT / ".cache"
CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)

EXPERIMENT_DIRECTORY = PROJECT_ROOT / "experiments"
EXPERIMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)

RESULT_DIRECTORY = EXPERIMENT_DIRECTORY / "results"
RESULT_DIRECTORY.mkdir(parents=True, exist_ok=True)
