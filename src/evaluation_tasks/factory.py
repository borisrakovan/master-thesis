from src.experiment.schema import ExperimentDefinition
from src.llm.client import LlmClient
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.enums import TaskType
from src.evaluation_tasks.imdb_sentiment_analysis import IMDbSentimentAnalysis
from src.evaluation_tasks.tweet_sentiment_analysis import TweetSentimentAnalysis


def create_evaluation_task(experiment: ExperimentDefinition, llm_client: LlmClient) -> EvaluationTask:
    """Create an evaluation task for a given task type"""
    system_message = experiment.prompts.system.format()
    if experiment.task_type == TaskType.TWEET_SENTIMENT_ANALYSIS:
        return TweetSentimentAnalysis(llm_client=llm_client, system_message=system_message)
    elif experiment.task_type == TaskType.IMDB_SENTIMENT_ANALYSIS:
        return IMDbSentimentAnalysis(llm_client=llm_client, system_message=system_message)
    else:
        raise NotImplementedError(f"Task type {experiment.task_type} is not implemented")
