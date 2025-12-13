import time
from typing import Callable, Any


def retry_call(func: Callable[[], Any], retries: int = 3, backoff: float = 1.0, exceptions: tuple = (Exception,)) -> Any:
    """Call `func` with retries and exponential backoff.

    Args:
        func: Zero-argument callable to execute.
        retries: Number of attempts (including the first).
        backoff: Initial backoff in seconds; doubles each retry.
        exceptions: Exception types that should trigger a retry.

    Returns:
        The result of `func()` if successful.

    Raises:
        The last exception raised by `func()` if all retries fail.
    """
    attempt = 0
    delay = backoff
    last_exc = None
    while attempt < retries:
        try:
            return func()
        except exceptions as e:
            last_exc = e
            attempt += 1
            if attempt >= retries:
                break
            time.sleep(delay)
            delay *= 2
    # If we get here, all retries failed
    raise last_exc
