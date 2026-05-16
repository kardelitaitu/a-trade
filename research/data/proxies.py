"""
Proxy manager for QuantumEdge — round-robin rotation with retry and dead proxy detection.

Format expected in proxies.txt:
    ip:port:username:password
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PROXIES_FILE = Path(__file__).resolve().parents[2] / "proxies.txt"
MAX_RETRIES = 3
MAX_FAILURES = 3


@dataclass
class ProxyEntry:
    """A single proxy with failure tracking."""
    host: str
    port: int
    username: str
    password: str
    failures: int = 0
    alive: bool = True

    @property
    def url(self) -> str:
        """Full proxy URL for requests library."""
        return f"http://{self.username}:{self.password}@{self.host}:{self.port}"

    @property
    def dict_format(self) -> dict:
        """Format for `proxies=` parameter in requests."""
        return {"http": self.url, "https": self.url}


class ProxyManager:
    """
    Round-robin proxy manager with automatic retry and dead proxy detection.

    Usage:
        pm = ProxyManager()
        proxy = pm.get_proxy()          # next proxy in rotation
        resp = pm.fetch("https://...")  # auto-retry with next proxy on failure
    """

    def __init__(
        self,
        proxies_file: str | Path = PROXIES_FILE,
        max_retries: int = MAX_RETRIES,
        shuffle: bool = True,
    ):
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._index = 0
        self._proxies: list[ProxyEntry] = []
        self._load(proxies_file, shuffle)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_proxy(self) -> Optional[ProxyEntry]:
        """Get the next alive proxy (round-robin)."""
        with self._lock:
            if not self._proxies:
                return None
            alive = [p for p in self._proxies if p.alive]
            if not alive:
                return None

            # Advance index through alive proxies only
            for _ in range(len(self._proxies)):
                idx = self._index % len(self._proxies)
                self._index += 1
                proxy = self._proxies[idx]
                if proxy.alive:
                    return proxy

            return None

    def fetch(
        self,
        url: str,
        method: str = "GET",
        timeout: int = 30,
        **kwargs,
    ) -> requests.Response:
        """
        Make an HTTP request with automatic proxy rotation and retry.

        On failure (connection error, timeout, status >= 500), marks the
        proxy as failed and retries with the next proxy.
        Raises after exhausting all alive proxies.
        """
        if not self._proxies:
            return requests.request(method, url, timeout=timeout, **kwargs)

        last_error = None
        for attempt in range(self.max_retries):
            proxy = self.get_proxy()
            if proxy is None:
                raise ConnectionError(
                    f"All proxies exhausted after {attempt} attempt(s). "
                    f"Last error: {last_error}"
                )

            try:
                resp = requests.request(
                    method, url,
                    proxies=proxy.dict_format,
                    timeout=timeout,
                    **kwargs,
                )
                # Server-side errors count as proxy failure
                if resp.status_code >= 500:
                    self._mark_failed(proxy)
                    last_error = f"HTTP {resp.status_code}"
                    continue

                # Success — reset failure count
                with self._lock:
                    proxy.failures = 0
                return resp

            except (requests.ConnectionError, requests.Timeout) as e:
                self._mark_failed(proxy)
                last_error = str(e)
                continue

        raise ConnectionError(
            f"Request failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def status(self) -> dict:
        """Return proxy pool status."""
        with self._lock:
            total = len(self._proxies)
            alive = sum(1 for p in self._proxies if p.alive)
            dead = total - alive
            return {
                "total": total,
                "alive": alive,
                "dead": dead,
                "index": self._index,
            }

    def reset_dead(self) -> int:
        """Re-enable all dead proxies. Returns count revived."""
        with self._lock:
            revived = 0
            for p in self._proxies:
                if not p.alive:
                    p.alive = True
                    p.failures = 0
                    revived += 1
            return revived

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self, path: str | Path, shuffle: bool) -> None:
        """Parse proxies.txt and load entries."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Proxies file not found: {path}. Running without proxy.")
            return

        lines = path.read_text().strip().splitlines()
        entries = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":")
            if len(parts) == 4:
                entries.append(ProxyEntry(
                    host=parts[0],
                    port=int(parts[1]),
                    username=parts[2],
                    password=parts[3],
                ))
            else:
                logger.debug(f"Skipping malformed proxy line: {line[:40]}...")

        if shuffle:
            random.shuffle(entries)

        self._proxies = entries
        logger.info(f"Loaded {len(entries)} proxies from {path}")

    def _mark_failed(self, proxy: ProxyEntry) -> None:
        """Increment failure count; mark dead if threshold reached."""
        with self._lock:
            proxy.failures += 1
            if proxy.failures >= MAX_FAILURES:
                proxy.alive = False
                logger.debug(f"Proxy {proxy.host}:{proxy.port} marked dead "
                             f"({proxy.failures} failures)")
