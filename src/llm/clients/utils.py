from tenacity import (
    RetryCallState,
)

from src.logger import get_logger
from src.settings import settings

logger = get_logger(__name__)


def log_retry_error_attempt(retry_state: RetryCallState):
    """Log a retry attempt"""

    if not retry_state.outcome:
        return

    exc = retry_state.outcome.exception()
    message = f"Error: {exc.__class__.__name__}: {exc}"
    if retry_state.attempt_number < settings.llm_max_retries:
        message += "\nRetrying..."
    logger.warning(message)


