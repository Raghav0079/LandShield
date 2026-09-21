from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, TypeVar

import requests

T = TypeVar("T")
LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def chunks(values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


class CachedHttpClient:
    """Small JSON HTTP client with deterministic disk caching and retries."""

    def __init__(
        self,
        cache_dir: Path,
        timeout: int = 90,
        retries: int = 3,
        user_agent: str = "ner-landslide/0.1 (research decision-support)",
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        cache_namespace: str = "http",
        refresh: bool = False,
    ) -> Any:
        key = stable_hash({"url": url, "params": params})
        cache_path = self.cache_dir / cache_namespace / f"{key}.json"
        if cache_path.exists() and not refresh:
            with cache_path.open(encoding="utf-8") as stream:
                return json.load(stream)

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                payload = response.json()
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = cache_path.with_suffix(".tmp")
                with temporary.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream)
                temporary.replace(cache_path)
                return payload
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if attempt + 1 < self.retries:
                    delay = 2**attempt
                    LOGGER.warning("Request failed; retrying in %ss: %s", delay, error)
                    time.sleep(delay)
        raise RuntimeError(f"Request failed after {self.retries} attempts: {url}") from last_error
