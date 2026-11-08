"""Cohort-local immutable feature inputs and source-hash compiled formulas.

The frozen ``model.py`` remains the reference/training implementation. This
adapter changes input acquisition and compilation only: all observations,
formula order, cutoffs and validation rules are retained. No input is shared
between cohorts and no global mutable reader is patched.
"""
from __future__ import annotations

import ast
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import sys
import threading
import types
import uuid

import numpy as np
import pandas as pd

from . import model as reference

_CODE_LOCK = threading.Lock()
_CODE_CACHE = OrderedDict()


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def required_paths(data_root, symbols=None):
    root = Path(data_root).resolve()
    coins = tuple(symbols or reference.SYMBOLS)
    paths = []
    for coin in coins:
        old = root / "intraday" / f"{coin}_5m_futures_2023_2026.parquet"
        paths.append(old if old.exists() else root / "intraday" / f"{coin}_5m_futures.parquet")
    for coin in coins:
        paths.extend(root / "auxiliary" / name for name in (
            f"{coin}_8h_funding.parquet", f"{coin}_1h_mark.parquet", f"{coin}_1h_futures.parquet"))
    paths.extend(root / "auxiliary" / name for name in
                 ("binance_basis_4h.parquet", "deribit_dvol_daily_2021_2026.parquet"))
    return tuple(paths)


def _compiled_sources():
    blobs = {}
    for name, expected in reference.SOURCE_HASHES.items():
        raw = (reference.ORIGINAL / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != expected:
            raise reference.ModelInputError("frozen source hash mismatch")
        blobs[name] = raw
    key = tuple(sorted(reference.SOURCE_HASHES.items()))
    with _CODE_LOCK:
        if key in _CODE_CACHE:
            _CODE_CACHE.move_to_end(key)
            return _CODE_CACHE[key], True
        base_name = "step2_standalone_clean_room_master.py"
        expert_name = "coin_specific_v4_divergence.py"
        tree = ast.parse(blobs[base_name].decode("utf-8"))
        builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_clean_room_bundle")
        cut = next(i for i, n in enumerate(builder.body) if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "o_raw" for t in n.targets))
        builder.body = builder.body[:cut] + ast.parse(
            "return {'index': idx, 'features': features_259, 'spec_features': spec_df}").body

        class RedirectReader(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "pd" and node.func.attr in ("read_parquet", "read_feather")):
                    node.func = ast.Name(id="_guarded_read", ctx=ast.Load())
                return node

        tree = RedirectReader().visit(tree)
        ast.fix_missing_locations(tree)
        codes = (compile(tree, str(reference.ORIGINAL / base_name), "exec"),
                 compile(blobs[expert_name].decode("utf-8"), str(reference.ORIGINAL / expert_name), "exec"))
        _CODE_CACHE[key] = codes
        while len(_CODE_CACHE) > 4:
            _CODE_CACHE.popitem(last=False)
        return codes, False


@contextmanager
def _fresh_modules(codes, reader, root):
    # Dataclass processing resolves the defining module during exec. Unique
    # names avoid concurrent builds sharing namespaces; remove them afterwards.
    name = "argon4_cohort_features_" + uuid.uuid4().hex
    expert_name = name + "_expert"
    base, expert = types.ModuleType(name), types.ModuleType(expert_name)
    base.__file__ = str(reference.ORIGINAL / "step2_standalone_clean_room_master.py")
    expert.__file__ = str(reference.ORIGINAL / "coin_specific_v4_divergence.py")
    sys.modules[name], sys.modules[expert_name] = base, expert
    try:
        exec(codes[0], base.__dict__)
        base._locate_data_root = lambda: root
        base._guarded_read = reader
        exec(codes[1], expert.__dict__)
        yield base, expert
    finally:
        sys.modules.pop(name, None)
        sys.modules.pop(expert_name, None)


class FeatureSnapshot:
    """One fixed cutoff and one byte snapshot shared by this cohort's streams.

    ``expected_sources`` can be the collector's source metadata dictionary.
    Whenever supplied, every feature file must match a collector hash. Capture
    rejects concurrent replacement; after capture it uses the captured bytes,
    even if the next observation is subsequently published on disk.
    """
    def __init__(self, data_root, *, as_of, expected_sources=None, symbols=None):
        self._root = Path(data_root).resolve()
        self._cutoff = reference._timestamp(as_of)
        if not self.root.is_relative_to(reference.ROOT):
            raise reference.ModelInputError("unsupported local feature source or symbol")
        if symbols is None and isinstance(expected_sources, dict) and any(str(k).startswith("XRPUSDT:") for k in expected_sources):
            symbols = (*reference.SYMBOLS, "XRPUSDT")
        self._symbols = tuple(symbols or reference.SYMBOLS)
        self._lock = threading.RLock()
        self._closed = False
        self._raw, self._validated = {}, {}
        self._stats = {"physical_reads": 0, "parquet_decodes": 0, "validated_cache_hits": 0,
                       "builds": 0, "input_history_truncated": False}
        expected = {}
        if expected_sources is not None:
            for entry in expected_sources.values():
                if isinstance(entry, dict) and entry.get("path") and entry.get("sha256"):
                    path = Path(entry["path"]).resolve()
                    if path in expected and expected[path] != entry["sha256"]:
                        raise reference.ModelInputError("inconsistent expected source hashes")
                    expected[path] = entry["sha256"]
        identities = {}
        for path in required_paths(self.root, symbols=self._symbols):
            path = path.resolve()
            if not path.is_relative_to(self.root) or path.suffix != ".parquet":
                raise reference.ModelInputError("only explicit local parquet sources are accepted")
            identities[path] = _identity(path)
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if expected_sources is not None and expected.get(path) != digest:
                raise reference.ModelInputError("feature snapshot differs from validated collector source")
            if _identity(path) != identities[path]:
                raise reference.ModelInputError("feature source changed during snapshot capture")
            self._raw[path] = (raw, digest)
            self._stats["physical_reads"] += 1
        if any(_identity(path) != identity for path, identity in identities.items()):
            raise reference.ModelInputError("feature sources changed during snapshot capture")
        self._codes, self._stats["compile_cache_hit"] = _compiled_sources()

    @property
    def root(self):
        return self._root

    @property
    def cutoff(self):
        return self._cutoff

    @property
    def stats(self):
        with self._lock:
            return {**self._stats, "cutoff_utc": self.cutoff.isoformat(),
                    "source_hashes": {p.relative_to(self.root).as_posix(): digest for p, (_, digest) in self._raw.items()}}

    def close(self):
        with self._lock:
            self._closed = True
            self._raw.clear()
            self._validated.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _read_validated(self, path):
        p = Path(path).resolve()
        with self._lock:
            if self._closed:
                raise reference.ModelInputError("feature snapshot already closed")
            if p not in self._raw:
                raise reference.ModelInputError("source not captured in this cohort snapshot")
            if p in self._validated:
                self._stats["validated_cache_hits"] += 1
                frame, availability, fingerprint = self._validated[p]
                return frame.copy(), deepcopy(availability), dict(fingerprint)
            raw, digest = self._raw[p]
            frame = pd.read_parquet(io.BytesIO(raw))
            self._stats["parquet_decodes"] += 1
            cutoff = self.cutoff
            received_field = "first_received_at" if "first_received_at" in frame.columns else "available_at" if "available_at" in frame.columns else None
            if received_field:
                published = pd.to_datetime(frame[received_field], utc=True, errors="raise")
                if published.isna().any():
                    raise reference.ModelInputError("missing observation availability")
                frame = frame.loc[published <= cutoff].copy()
            fingerprint = {"bytes": len(raw), "sha256": digest,
                "availability_policy": "OBSERVED_FIRST_RECEIVED_AT" if received_field == "first_received_at" else "OBSERVED_AVAILABLE_AT" if received_field else "HISTORICAL_PUBLICATION_ASSUMPTION"}
            availability = {}
            name = p.name
            if p.parent.name == "intraday":
                if not isinstance(frame.index, pd.DatetimeIndex):
                    frame = frame.set_index(pd.to_datetime(frame["open_time"], utc=True))
                frame.index = pd.to_datetime(frame.index, utc=True)
                frame = frame.loc[frame.index + pd.Timedelta(minutes=5) <= cutoff]
                frame = frame.loc[frame.index >= pd.Timestamp("2024-01-01", tz="UTC")]
                if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
                    raise reference.ModelInputError("unsorted or duplicate raw candles")
                if len(frame) < 1728 or np.any(np.diff(frame.index.asi8) != 300_000_000_000):
                    raise reference.ModelInputError("at least 1728 contiguous observed 5m candles required")
                reference._validate_ohlc(frame, name, volume=True)
                availability[name] = {"last_available_time": str(frame.index[-1] + pd.Timedelta(minutes=5)), "rows": len(frame)}
            elif "_8h_funding" in name:
                stamps = pd.to_datetime(frame["date"], utc=True)
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps.loc[stamps <= cutoff]
                if stamps.empty or stamps.duplicated().any() or not stamps.is_monotonic_increasing or not np.isfinite(frame["funding_rate"].to_numpy(float)).all():
                    raise reference.ModelInputError("invalid observed funding events")
                availability[name] = {"last_available_time": str(stamps.iloc[-1]), "rows": len(frame)}
            elif "_1h_" in name:
                stamps = pd.to_datetime(frame["date"], utc=True) + pd.Timedelta(hours=1)
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps.loc[stamps <= cutoff]
                if stamps.empty or stamps.duplicated().any() or not stamps.is_monotonic_increasing:
                    raise reference.ModelInputError("invalid hourly observations")
                reference._validate_ohlc(frame, name)
                availability[name] = {"last_available_time": str(stamps.iloc[-1]), "rows": len(frame)}
            elif name == "binance_basis_4h.parquet":
                frame.index = pd.to_datetime(frame.index, utc=True)
                stamps = frame.index + pd.Timedelta(hours=4)
                frame = frame.loc[stamps <= cutoff]
                stamps = stamps[stamps <= cutoff]
                if len(frame) < 4 or frame.index.has_duplicates or not frame.index.is_monotonic_increasing or not np.isfinite(frame[list(reference.SYMBOLS[:3])].to_numpy(float)).all():
                    raise reference.ModelInputError("invalid observed 4h basis")
                availability[name] = {"last_available_time": str(stamps[-1]), "rows": len(frame)}
            elif name == "deribit_dvol_daily_2021_2026.parquet":
                stamps = pd.to_datetime(frame["bucket_utc"], utc=True) + pd.Timedelta(days=1, minutes=5)
                frame = frame.loc[stamps <= cutoff]
                for coin in ("BTC", "ETH"):
                    sub = frame.loc[frame.symbol == coin]
                    dates = pd.to_datetime(sub.bucket_utc, utc=True)
                    if len(sub) < 9 or dates.duplicated().any() or not dates.is_monotonic_increasing:
                        raise reference.ModelInputError("insufficient daily observed DVOL history")
                    reference._validate_ohlc(sub, name)
                    availability[f"DVOL_{coin}"] = {"last_available_time": str(dates.iloc[-1] + pd.Timedelta(days=1, minutes=5)), "rows": len(sub)}
            else:
                raise reference.ModelInputError("unrecognized feature source")
            self._validated[p] = (frame.copy(), deepcopy(availability), dict(fingerprint))
            return frame, availability, fingerprint

    def build(self, symbol="BTCUSDT", *, tail_rows=1):
        if symbol not in self._symbols:
            raise reference.ModelInputError("unsupported local feature source or symbol")
        if self._closed:
            raise reference.ModelInputError("feature snapshot already closed")
        frames, availability, fingerprints = {}, {}, {}

        def reader(path):
            p = Path(path).resolve()
            frame, meta, fingerprint = self._read_validated(p)
            availability.update(meta)
            fingerprints[str(p.relative_to(reference.ROOT))] = fingerprint
            return frame

        intraday_paths = tuple(self._raw)[:len(self._symbols)]
        target_paths = intraday_paths[:len(reference.SYMBOLS)]
        if symbol not in reference.SYMBOLS:
            target_paths = (*target_paths, next(p for p in intraday_paths if p.name.startswith(symbol + "_")))
        for p in target_paths:
            coin = p.name.split("_")[0]
            frames[coin] = reader(p)
        idx = frames[symbol].index
        if any(not frame.index.equals(idx) for frame in frames.values()):
            raise reference.ModelInputError("four observed candle timelines must match exactly")
        effective_close = idx[-1] + pd.Timedelta(minutes=5)
        if not pd.Timedelta(0) <= self.cutoff - effective_close <= pd.Timedelta(minutes=5):
            raise reference.ModelInputError("latest observed closed candle is stale")
        with _fresh_modules(self._codes, reader, self.root) as (base_module, expert_module):
            if symbol not in getattr(base_module, "CORE_SYMBOLS", reference.SYMBOLS):
                base_module.CORE_SYMBOLS = (*reference.SYMBOLS, symbol)
            bundle = base_module.build_clean_room_bundle(symbol)
            base = pd.concat([bundle["features"], bundle["spec_features"]], axis=1).astype(np.float32)
            ofi = sum((2 * frame.taker_buy_quote_volume - frame.quote_volume) / (frame.quote_volume + 1e-12) for frame in tuple(frames.values())[:4]) / 4
            expert = expert_module.build_coin_own_structural_and_v4_mag_gap(frames[symbol], frames["BTCUSDT"], ofi)
        if base.shape[1] != 311 or expert.shape[1] != 65 or not base.index.equals(expert.index):
            raise reference.ModelInputError("unexpected verified feature schema")
        if not np.isfinite(base.to_numpy()).all() or not np.isfinite(expert.to_numpy()).all():
            raise reference.ModelInputError("nonfinite model features")
        for name, meta in availability.items():
            age = effective_close - reference._timestamp(meta["last_available_time"])
            maximum = (pd.Timedelta(hours=8, minutes=5) if "funding" in name else
                       pd.Timedelta(hours=1, minutes=5) if "_1h_" in name else
                       pd.Timedelta(hours=4, minutes=5) if "basis" in name else
                       pd.Timedelta(days=1, minutes=10) if name.startswith("DVOL_") else pd.Timedelta(minutes=5))
            if age > maximum:
                raise reference.ModelInputError(f"stale source input: {name}")
            meta["age_seconds_at_feature_close"] = age.total_seconds()
        raw_target = frames[symbol][["open", "close"]].copy()
        if tail_rows is not None:
            if not isinstance(tail_rows, int) or tail_rows < 1:
                raise reference.ModelInputError("positive tail_rows required")
            base, expert, raw_target = base.tail(tail_rows), expert.tail(tail_rows), raw_target.tail(tail_rows)
        self._stats["builds"] += 1
        return {"base": base, "expert": expert, "index": base.index, "raw_target": raw_target,
                "availability": availability, "provenance": fingerprints,
                "classification": reference.CLASSIFICATION, "source_sha256": dict(reference.SOURCE_HASHES)}


def build_feature_frames(data_root=reference.DATA_ROOT, symbol="BTCUSDT", *, as_of, tail_rows=1, snapshot=None):
    """Compatible convenience entry; pass one snapshot to share within a cohort."""
    if snapshot is not None:
        if snapshot.root != Path(data_root).resolve() or snapshot.cutoff != reference._timestamp(as_of):
            raise reference.ModelInputError("feature snapshot scope mismatch")
        return snapshot.build(symbol, tail_rows=tail_rows)
    with FeatureSnapshot(data_root, as_of=as_of) as captured:
        return captured.build(symbol, tail_rows=tail_rows)
