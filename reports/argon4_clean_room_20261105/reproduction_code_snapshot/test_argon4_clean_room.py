"""Integrity and causality checks for the attachment reproduction wrapper."""
import ast
import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.argon4_clean_room_fetch import COLS, fetch, read_partition, validate
from scripts.argon4_clean_room_reproduce import Capture, ORIGINAL, load_original
from scripts.argon4_clean_room_seven_streams import load_v4


def test_supplied_sources_match_frozen_hashes():
    assert len(load_original()["files"]) == 3


def test_instrumentation_only_adds_export_hooks():
    source = (ORIGINAL / "step2_standalone_clean_room_master.py").read_text(encoding="utf8")
    original = ast.parse(source)
    c = Capture()
    instrumented = c.visit(ast.parse(source))
    assert c.counts == {"binary": 1, "positions": 1, "done": 1}

    class Strip(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_capture":
                return None
            return self.generic_visit(node)

    assert ast.dump(Strip().visit(instrumented)) == ast.dump(original)


def test_network_allowlist_rejects_order_routes():
    with pytest.raises(ValueError, match="allowlist"):
        fetch("https://fapi.binance.com/fapi/v1/order")


def test_archive_microsecond_spot_timestamps(tmp_path):
    row = [1735689600000000, 10, 12, 9, 11, 3, 1735703999999999, 33, 2, 1, 11, 0]
    p = tmp_path / "spot.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("spot.csv", ",".join(map(str, row)) + "\n")
    d = read_partition(p)
    assert d.open_time.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert d.close.iloc[0] == 11


def observed():
    return pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01 01:00"], format="mixed", utc=True),
                         "open": [10., 10.], "high": [12., 12.], "low": [9., 9.], "close": [11., 11.], "volume": [2., 3.]})


def test_gaps_reported_without_imputation():
    d = observed()
    d.loc[1, "date"] += pd.Timedelta("1h")
    before = d.copy(deep=True)
    report = validate(d, "date", "1h")
    assert len(report["gaps"]) == 1
    pd.testing.assert_frame_equal(d, before)


@pytest.mark.parametrize("problem", ["nan", "duplicate", "ohlc", "negative_volume"])
def test_invalid_observations_fail_closed(problem):
    d = observed()
    if problem == "nan":
        d.loc[0, "close"] = np.nan
    elif problem == "duplicate":
        d.loc[1, "date"] = d.loc[0, "date"]
    elif problem == "ohlc":
        d.loc[0, "high"] = 8
    else:
        d.loc[0, "volume"] = -1
    with pytest.raises(ValueError):
        validate(d, "date", "1h")


def test_funding_preserves_observed_irregular_events():
    d = pd.DataFrame({"date": pd.to_datetime(["2025-01-01", "2025-01-01 08:00", "2025-01-01 12:00"], format="mixed", utc=True),
                      "funding_rate": [.0001, -.0003, .0002]})
    report = validate(d, "date")
    assert report["rows"] == 3
    assert set(report["interval_counts"]) == {"0 days 08:00:00", "0 days 04:00:00"}
    assert d.funding_rate.tolist() == [.0001, -.0003, .0002]


def test_v4_features_do_not_depend_on_future_candles():
    # Synthetic unit-test fixture only; never used for research results.
    n = 700
    t = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    c = 100 + np.sin(np.arange(n) / 7) + np.arange(n) * .01
    d = pd.DataFrame({"open": c - .1, "close": c, "high": c + .2, "low": c - .3,
                      "volume": 100 + np.arange(n) % 11, "quote_volume": c * 100,
                      "taker_buy_quote_volume": c * (48 + np.arange(n) % 7), "count": 20 + np.arange(n) % 13}, index=t)
    ofi = pd.Series(np.sin(np.arange(n) / 3), index=t)
    f = load_v4().build_coin_own_structural_and_v4_mag_gap
    full = f(d, d, ofi)
    prefix = f(d.iloc[:600], d.iloc[:600], ofi.iloc[:600])
    assert full.shape[1] == 65
    pd.testing.assert_frame_equal(full.iloc[:600], prefix)
