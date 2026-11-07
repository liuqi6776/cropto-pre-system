"""Owned connection pools for bounded, unauthenticated Polymarket public GETs.

One lease owns a Session until the response and its receipt are complete. Owners
must close their own transport; observation and execution must use separate
instances. Connections may be reused, but responses, cookies and clocks are not.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
import threading
import time
from urllib.parse import urlencode

import requests
from requests.adapters import HTTPAdapter


SOURCE = "POLYMARKET_PUBLIC_READ_ONLY"
MAX_RESPONSE_BYTES = 2_000_000
ENDPOINTS = {
    "market": ("https://gamma-api.polymarket.com/markets", "slug"),
    "book": ("https://clob.polymarket.com/book", "token_id"),
}


def public_url(kind, value=None):
    """No caller-supplied URL, method, headers, credentials or query vocabulary."""
    if kind == "time":
        if value is not None:
            raise ValueError("clock route has no parameters")
        return "https://clob.polymarket.com/time"
    if kind not in ENDPOINTS:
        raise ValueError("endpoint is not allowed")
    value = str(value)
    if kind == "market":
        if not re.fullmatch(r"(?:btc|eth|sol|bnb)-updown-(?:5|15)m-[0-9]{10}", value):
            raise ValueError("invalid market lookup")
    elif not re.fullmatch(r"[1-9][0-9]{0,77}", value):
        raise ValueError("invalid outcome identifier")
    base, key = ENDPOINTS[kind]
    return base + "?" + urlencode({key: value})


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class PublicPolymarketTransport:
    """A terminal-close pool of one execution or four observation Sessions.

    Every GET is attempted at most once. An error discards only its leased
    Session; a later explicit GET may create a replacement. Waiting callers are
    rejected after close begins, while already leased requests finish normally.
    """

    def __init__(self, pool_size=1):
        if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size not in (1, 4):
            raise ValueError("public pool size must be one or four")
        self.pool_size = pool_size
        self._condition = threading.Condition()
        self._sessions = [None] * pool_size
        self._available = list(range(pool_size))
        self._active = 0
        self._closing = False
        self._closed = False

    def __enter__(self):
        with self._condition:
            if self._closing:
                raise ValueError("PUBLIC_TRANSPORT_CLOSED")
        return self

    def __exit__(self, *args):
        self.close()

    def _acquire(self):
        queued = time.monotonic()
        with self._condition:
            while not self._available and not self._closing:
                self._condition.wait()
            if self._closing:
                raise ValueError("PUBLIC_TRANSPORT_CLOSED")
            index = self._available.pop()
            self._active += 1
        return index, time.monotonic() - queued

    def _release(self, index):
        with self._condition:
            self._active -= 1
            self._available.append(index)
            self._condition.notify_all()

    def _discard(self, index):
        session, self._sessions[index] = self._sessions[index], None
        if session is not None:
            try:
                session.close()
            except Exception:
                # Never print arbitrary transport exceptions or mask the receipt
                # of the original failure with a cleanup error.
                pass

    def close(self):
        """Reject queued/new requests, wait for active receipts, then close once."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()
            while self._active:
                self._condition.wait()
            if not self._closed:
                for index in range(self.pool_size):
                    self._discard(index)
                self._closed = True

    def _session(self, index):
        session = self._sessions[index]
        if session is None:
            session = requests.Session()
            self._sessions[index] = session
            session.mount("https://", HTTPAdapter(max_retries=0, pool_connections=2,
                                                 pool_maxsize=1, pool_block=True))
        # A persistent connection must never persist server cookies or inherit
        # .netrc, proxy credentials, client certificates, hooks or default auth.
        session.trust_env = False
        session.auth = None
        session.proxies.clear()
        session.cookies.clear()
        session.headers.clear()
        session.headers.update(requests.utils.default_headers())
        session.params.clear()
        session.hooks = {"response": []}
        session.verify = True
        session.cert = None
        return session

    def get(self, kind, value=None):
        """Return (decoded JSON or None, raw receipt, safe error code or None)."""
        url = public_url(kind, value)
        index, queued_seconds = self._acquire()
        try:
            return self._get_leased(index, kind, url, queued_seconds)
        finally:
            self._release(index)

    def _get_leased(self, index, kind, url, queued_seconds):
        receipt = {"source": SOURCE, "url": url, "method": "GET",
                   "requested_at": None, "received_at": None, "http_status": None,
                   "sha256": None, "raw_text": None, "body_complete": False,
                   "transport_timing": {"basis": "MONOTONIC",
                       "queue_wait_seconds": queued_seconds,
                       "request_elapsed_seconds": None, "headers_elapsed_seconds": None,
                       "body_elapsed_seconds": None}}
        timing = receipt["transport_timing"]
        begin = None
        session = None
        failed = False
        try:
            session = self._session(index)
            headers = {"Accept": "application/json", "User-Agent": "argon4-forward-public/1.0"}
            if kind == "time":
                headers["Cache-Control"] = "no-cache"
            # These timestamps bracket the actual GET, excluding lease waiting.
            receipt["requested_at"] = _utc_now()
            begin = time.monotonic()
            with session.get(url, timeout=(2, 2) if kind == "time" else (2, 3),
                             allow_redirects=False, stream=True, headers=headers) as response:
                receipt["http_status"] = response.status_code
                headers_at = time.monotonic()
                timing["headers_elapsed_seconds"] = headers_at - begin
                raw = bytearray()
                for chunk in response.iter_content(64 if kind == "time" else 32768):
                    raw.extend(chunk)
                    if ((kind == "time" and len(raw) >= 64)
                            or (kind != "time" and len(raw) > MAX_RESPONSE_BYTES)
                            or time.monotonic() - headers_at > 6):
                        raise ValueError("RESPONSE_LIMIT")
                receipt["received_at"] = _utc_now()
                finished = time.monotonic()
                timing["request_elapsed_seconds"] = finished - begin
                timing["body_elapsed_seconds"] = finished - headers_at
                # A failed UTF-8 decode cannot claim a complete text receipt.
                text = bytes(raw).decode("utf-8")
                receipt.update(body_complete=True, sha256=hashlib.sha256(raw).hexdigest(), raw_text=text)
                if response.status_code != 200:
                    failed = True
                    return None, receipt, f"HTTP_{response.status_code}"
                return json.loads(text), receipt, None
        except (requests.RequestException, ValueError, UnicodeError) as exc:
            failed = True
            if receipt["received_at"] is None:
                receipt["received_at"] = _utc_now()
            if begin is not None and timing["request_elapsed_seconds"] is None:
                timing["request_elapsed_seconds"] = time.monotonic() - begin
            # Do not stringify arbitrary errors: their text may contain secrets.
            code = "RESPONSE_LIMIT" if type(exc) is ValueError and exc.args == ("RESPONSE_LIMIT",) else type(exc).__name__
            return None, receipt, code
        except BaseException:
            failed = True
            raise
        finally:
            if session is not None:
                try:
                    session.cookies.clear()
                except Exception:
                    failed = True
            if failed:
                self._discard(index)
