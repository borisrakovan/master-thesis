from openai import RateLimitError, InternalServerError
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.logger import get_logger
from src.settings import settings

logger = get_logger(__name__)


def _log_retry_error_attempt(retry_state: RetryCallState):
    """Log a retry attempt"""

    if not retry_state.outcome:
        return

    exc = retry_state.outcome.exception()
    message = f"Error: {exc.__class__.__name__}: {exc}"
    if retry_state.attempt_number < settings.llm_max_retries:
        message += "\nRetrying..."
    logger.warning(message)


openai_retry = retry(
    retry=retry_if_exception_type((RateLimitError, InternalServerError)),
    stop=stop_after_attempt(settings.llm_max_retries),
    wait=wait_exponential(multiplier=2),
    after=_log_retry_error_attempt,
    reraise=True,
)
