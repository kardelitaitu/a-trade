"""Tests for research/data/proxies.py."""

import pytest
from pathlib import Path
from research.data.proxies import ProxyManager, ProxyEntry


SAMPLE_PROXIES = """
1.2.3.4:8080:user1:pass1
5.6.7.8:9090:user2:pass2
"""


@pytest.fixture
def proxy_file(tmp_path):
    p = tmp_path / "proxies.txt"
    p.write_text(SAMPLE_PROXIES.strip())
    return p


class TestProxyEntry:

    def test_url_format(self):
        p = ProxyEntry(host="1.2.3.4", port=8080, username="user", password="pass")
        assert p.url == "http://user:pass@1.2.3.4:8080"

    def test_dict_format(self):
        p = ProxyEntry(host="1.2.3.4", port=8080, username="u", password="p")
        d = p.dict_format
        assert d["http"] == "http://u:p@1.2.3.4:8080"
        assert d["https"] == "http://u:p@1.2.3.4:8080"


class TestProxyManager:

    def test_load(self, proxy_file):
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False)
        assert pm.status()["total"] == 2

    def test_round_robin(self, proxy_file):
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False)
        p1 = pm.get_proxy()
        p2 = pm.get_proxy()
        p3 = pm.get_proxy()
        assert p1 is not None
        assert p2 is not None
        assert p3 is not None
        # Round-robin should cycle back
        assert p1.host == p3.host

    def test_empty_file(self, tmp_path):
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        pm = ProxyManager(proxies_file=empty)
        assert pm.status()["total"] == 0

    def test_missing_file(self):
        pm = ProxyManager(proxies_file="/nonexistent/proxies.txt")
        assert pm.status()["total"] == 0

    def test_no_proxy_fallback(self, proxy_file):
        """get_proxy should return None when no alive proxies."""
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False)
        # Kill all proxies
        for p in pm._proxies:
            p.alive = False
        assert pm.get_proxy() is None

    def test_mark_failed(self, proxy_file):
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False, max_retries=1)
        proxy = pm._proxies[0]
        pm._mark_failed(proxy)
        assert proxy.failures == 1
        assert proxy.alive is True  # not yet at threshold (3)
        pm._mark_failed(proxy)
        pm._mark_failed(proxy)
        assert proxy.alive is False  # reached threshold

    def test_status(self, proxy_file):
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False)
        s = pm.status()
        assert s["total"] == 2
        assert s["alive"] == 2
        assert s["dead"] == 0

    def test_reset_dead(self, proxy_file):
        pm = ProxyManager(proxies_file=proxy_file, shuffle=False)
        pm._proxies[0].alive = False
        revived = pm.reset_dead()
        assert revived == 1
        assert pm._proxies[0].alive is True
