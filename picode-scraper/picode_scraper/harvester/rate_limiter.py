"""Per-domain rate limiting."""

import threading
import time
from urllib.parse import urlparse


class DomainRateLimiter:
    """Rate limit requests per domain.

    Thread-safe rate limiter that ensures minimum delay between
    requests to the same domain.
    """

    def __init__(self, default_delay: float = 1.0) -> None:
        """Initialize rate limiter.

        Args:
            default_delay: Minimum seconds between requests to same domain
        """
        self.default_delay = default_delay
        self._last_request: dict[str, float] = {}
        self._lock = threading.Lock()

    def wait(self, url: str) -> None:
        """Wait if necessary to respect rate limit for domain.

        Args:
            url: URL being requested (domain extracted automatically)
        """
        domain = urlparse(url).netloc

        with self._lock:
            last = self._last_request.get(domain, 0)
            elapsed = time.time() - last
            wait_time = self.default_delay - elapsed

            if wait_time > 0:
                time.sleep(wait_time)

            self._last_request[domain] = time.time()
