import time
from threading import Lock
from typing import Optional


class RateLimiter:
    """Simple token-bucket rate limiter.

    Example:
        limiter = RateLimiter(calls=60, period=60)  # 60 calls per 60 seconds
        limiter.acquire()
    """

    def __init__(self, calls: int = 60, period: int = 60):
        self.calls = max(1, int(calls))
        self.period = max(1, int(period))
        self._tokens = self.calls
        self._last = time.monotonic()
        self._lock = Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last
        if elapsed <= 0:
            return
        refill_tokens = (elapsed / self.period) * self.calls
        if refill_tokens >= 1:
            self._tokens = min(self.calls, self._tokens + int(refill_tokens))
            self._last = now

    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """Acquire permission to proceed according to rate limits.

        If `blocking` is True, will sleep until a token is available (or timeout).
        Returns True if acquired, False otherwise.
        """
        start = time.monotonic()
        while True:
            with self._lock:
                self._refill()
                if self._tokens > 0:
                    self._tokens -= 1
                    return True
            if not blocking:
                return False
            if timeout is not None and (time.monotonic() - start) >= timeout:
                return False
            time.sleep(0.05)
