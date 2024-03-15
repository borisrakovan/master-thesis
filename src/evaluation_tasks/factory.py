from src.evaluation_tasks.arc_multiple_choice_qa import ARCMultipleChoiceQA
from src.evaluation_tasks.base import EvaluationTask
from src.evaluation_tasks.enums import TaskType
from src.evaluation_tasks.imdb_sentiment_analysis import IMDbSentimentAnalysis
from src.evaluation_tasks.tweet_sentiment_analysis import TweetSentimentAnalysis
from src.experiment.schema import ExperimentDefinition
from src.llm.service import LlmService


def create_evaluation_task(experiment: ExperimentDefinition, llm: LlmService) -> EvaluationTask:
    """Create an evaluation task for a given task type"""
    system_message = experiment.prompts.system.format()
    if experiment.task_type == TaskType.TWEET_SENTIMENT_ANALYSIS:
        return TweetSentimentAnalysis(llm=llm, system_message=system_message)
    elif experiment.task_type == TaskType.IMDB_SENTIMENT_ANALYSIS:
        return IMDbSentimentAnalysis(llm=llm, system_message=system_message)
    elif experiment.task_type == TaskType.ARC_MULTIPLE_CHOICE_QA:
        return ARCMultipleChoiceQA(llm=llm, system_message=system_message)
    else:
        raise NotImplementedError(f"Task type {experiment.task_type} is not implemented")
