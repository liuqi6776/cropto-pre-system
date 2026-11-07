"""Research isolation, causal selection and observed-data validation contracts."""
import json

import numpy as np
import pandas as pd
import pytest

from tools import argon4_research11 as study


def candles():
    # Explicit synthetic test fixture; never saved as observed study data.
    t = pd.date_range("2026-09-01", periods=4, freq="5min", tz="UTC")
    return pd.DataFrame({"open_time": t, "close_time": (t+pd.Timedelta(minutes=5)-pd.Timedelta(milliseconds=1)).asi8//10**6,
                         "open": 1., "high": 2., "low": .5, "close": 1.1, "volume": 100.,
                         "quote_volume": 100., "count": 5, "taker_buy_quote_volume": 50.})


def test_candle_validation_never_fills_gap_or_changes_close():
    assert study.candle_audit(candles())["rows"] == 4
    with pytest.raises(ValueError, match="gap"):
        study.candle_audit(candles().drop(index=2))
    f = candles()
    f.loc[2, "close_time"] += 1
    with pytest.raises(ValueError, match="close"):
        study.candle_audit(f)
    with pytest.raises(ValueError, match="incomplete"):
        study.candle_audit(candles(), month="2026-09")


def test_zero_price_missing_volume_and_nonfinite_are_refused():
    for column, value in [("open", 0), ("volume", np.nan), ("quote_volume", -1)]:
        f = candles()
        f.loc[0, column] = value
        with pytest.raises(ValueError):
            study.candle_audit(f)


def test_fee_adjusted_proxy_retains_ties_small_returns_and_unsettled():
    result = study.payout_proxy([2, -2, 2, 2, 0], [0, 0, 1e-15, np.nan, 1], 1)
    assert result["signals"] == 4
    assert result["settled_proxy_labels"] == 3
    assert result["wins"] == 2
    assert result["ties_retained"] == 2
    assert result["unsettled_proxy_labels"] == 1
    assert result["break_even_win_rate"] == pytest.approx(.53248425)
    assert result["hypothetical_pnl_one_share_per_signal"] == pytest.approx(2-3*.53248425)


def test_threshold_selection_cannot_use_test_labels():
    index = pd.date_range("2024-01-01", "2026-09-30 23:55", freq="5min", tz="UTC")
    labels = np.full(len(index), .001)
    first = study.model._split(index, labels, study.model.FOLDS["oos_2026"], 3)
    changed = labels.copy()
    changed[first[2]] = np.nan
    second = study.model._split(index, changed, study.model.FOLDS["oos_2026"], 3)
    assert all(np.array_equal(a, b) for a,b in zip(first, second))
    tr, va, te = first
    assert index[tr[-1]]+pd.Timedelta(minutes=20) < index[va[0]]
    assert index[va[-1]]+pd.Timedelta(minutes=20) < index[te[0]]


def test_inventory_blocks_training_before_any_fit_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "DATA", tmp_path/"data")
    monkeypatch.setattr(study, "RAW", tmp_path/"data/raw_data")
    monkeypatch.setattr(study, "OUTPUT", tmp_path/"models")
    monkeypatch.setattr(study.model, "_fit_family", lambda *a, **kw: pytest.fail("fit must not run"))
    assert len(study.inventory()["missing_5m_months"]) == 66
    with pytest.raises(ValueError, match="missing"):
        study.prepare()
    with pytest.raises(ValueError, match="manifest missing"):
        study.train()
    assert not (tmp_path/"models").exists()
    assert not (tmp_path/"data").exists()


def test_output_and_download_allowlists(tmp_path, monkeypatch):
    monkeypatch.setattr(study, "DATA", tmp_path/"data")
    monkeypatch.setattr(study, "OUTPUT", tmp_path/"models")
    with pytest.raises(ValueError, match="isolated"):
        study.write_json(tmp_path/"production.json", {})
    with pytest.raises(ValueError):
        study.archive_url("BTCUSDT", "2026-09")
    with pytest.raises(ValueError):
        study.archive_url("XRPUSDT", "2026-11")
    with pytest.raises(ValueError):
        study._public_get("https://fapi.binance.com/fapi/v1/order", 100)
    path = tmp_path/"data/item.json"
    study.write_json(path, {"value": 1})
    with pytest.raises(ValueError, match="already exists"):
        study.write_json(path, {"value": 2})
    assert json.loads(path.read_text())["value"] == 1


def test_original_four_targets_use_exact_original_builder(monkeypatch):
    sentinel = object()
    called = []
    def original(root, symbol):
        called.append((root, symbol))
        return sentinel
    monkeypatch.setattr(study.model, "build_feature_frames", original)
    for symbol in study.CORE:
        assert study.build_features(symbol) is sentinel
    assert [x[1] for x in called] == list(study.CORE)
    assert len(study.CANDIDATES) == 11
    assert "BNB_5m" not in study.CANDIDATES
    assert tuple(study.model.SYMBOLS) == study.CORE


def test_new_target_features_keep_original_four_coin_ofi_and_own_prices(tmp_path, monkeypatch):
    """Synthetic formula fixture; never a source dataset or a score result claim."""
    raw = tmp_path/"raw"
    frames = {}
    timeline = pd.date_range("2024-01-12", periods=2000, freq="5min", tz="UTC")
    for i, symbol in enumerate(study.CORE + ("XRPUSDT",)):
        prices = 100 + (i+1)*np.sin(np.arange(2000)/13) + np.arange(2000)*.001
        f = pd.DataFrame({"open": prices, "high": prices+2, "low": prices-2, "close": prices+.1,
                          "volume": 100., "quote_volume": 100., "count": 3,
                          "taker_buy_quote_volume": 20+i*10+np.sin(np.arange(2000)/17)}, index=timeline)
        frames[f"{symbol}_5m_futures.parquet"] = f
        hourly = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), timeline[-1], freq="1h")
        for suffix in ("1h_mark", "1h_futures"):
            frames[f"{symbol}_{suffix}.parquet"] = pd.DataFrame({"date": hourly, "open": 1., "high": 2., "low": .5, "close": 1., "volume": 0.})
        funding = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), timeline[-1], freq="8h")
        frames[f"{symbol}_8h_funding.parquet"] = pd.DataFrame({"date": funding, "funding_rate": .0001})
    basis_index = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), timeline[-1], freq="4h")
    frames["binance_basis_4h.parquet"] = pd.DataFrame({s: .001 for s in study.CORE[:3]}, index=basis_index)
    days = pd.date_range(pd.Timestamp("2024-01-01", tz="UTC"), timeline[-1], freq="1d")
    frames["deribit_dvol_daily_2021_2026.parquet"] = pd.concat([pd.DataFrame({"bucket_utc": days, "symbol": s,
        "open": 50., "high": 51., "low": 49., "close": 50.}) for s in ("BTC", "ETH")], ignore_index=True)
    for name in frames:
        path = raw/("intraday" if "_5m_" in name else "auxiliary")/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()  # Paths for the mocked in-memory test reader only.
    monkeypatch.setattr(study, "RAW", raw)
    monkeypatch.setattr(study.model, "ROOT", tmp_path)
    monkeypatch.setattr(pd, "read_parquet", lambda path: frames[path.name].copy())
    original_sources = study.model._source_modules
    seen = []
    def spy(reader, root):
        base, expert = original_sources(reader, root)
        original_expert = expert.build_coin_own_structural_and_v4_mag_gap
        def calc(target, btc, ofi):
            seen.append((target, ofi))
            return original_expert(target, btc, ofi)
        expert.build_coin_own_structural_and_v4_mag_gap = calc
        return base, expert
    monkeypatch.setattr(study.model, "_source_modules", spy)
    result = study.build_features("XRPUSDT")
    expected = sum((2*frames[f"{s}_5m_futures.parquet"].taker_buy_quote_volume-100)/(100+1e-12) for s in study.CORE)/4
    expected.name = None
    pd.testing.assert_series_equal(seen[0][1], expected)
    pd.testing.assert_frame_equal(seen[0][0], frames["XRPUSDT_5m_futures.parquet"])
    assert result["base"].shape == (2000, 311)
    assert result["expert"].shape == (2000, 65)
    assert tuple(study.model.SYMBOLS) == study.CORE
    assert result["research_mapping"]["partner"] == "BTCUSDT"


def test_daily_mark_merge_only_adds_observed_missing_rows_and_rejects_conflict(tmp_path, monkeypatch):
    # Synthetic test partition; not an observed research dataset.
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "DATA", tmp_path/"research11")
    directory = study.DATA/"supplemental"
    directory.mkdir(parents=True)
    observed = pd.DataFrame({"date": pd.date_range("2026-06-29", periods=24, freq="1h", tz="UTC"),
                             "open": 1., "high": 2., "low": .5, "close": 1., "volume": 0.})
    path, source = directory/"fixture.parquet", directory/"fixture.zip"
    observed.to_parquet(path, index=False)
    source.write_bytes(b"explicit synthetic test archive")
    receipt = {"path": str(path.relative_to(tmp_path)), "source_path": str(source.relative_to(tmp_path)),
               "sha256": study.model.sha256(path), "official_sha256": study.model.sha256(source)}
    (directory/"XRPUSDT_mark_2026-06-29.receipt.json").write_text(json.dumps(receipt), encoding="utf8")
    original = observed.iloc[[0, 23]].copy()
    merged, evidence = study.merge_mark_supplement(original, "XRPUSDT")
    pd.testing.assert_frame_equal(merged, observed)
    assert evidence["added_rows"] == 22
    bad = original.copy()
    bad.loc[0, "close"] = 1.5
    with pytest.raises(AssertionError):
        study.merge_mark_supplement(bad, "XRPUSDT")
    source.write_bytes(b"changed archive")
    with pytest.raises(ValueError, match="provenance"):
        study.merge_mark_supplement(original, "XRPUSDT")


def test_mark_registration_emits_safe_relative_posix_paths_and_verified_byte_counts(tmp_path, monkeypatch):
    # Existing synthetic receipts exercise the actual proposal generator without
    # any request or changes to the stored input bytes.
    monkeypatch.setattr(study, "ROOT", tmp_path)
    monkeypatch.setattr(study, "DATA", tmp_path/"research_data/study")
    monkeypatch.setattr(study, "_public_get", lambda *a, **k: pytest.fail("receipt reuse must stay offline"))
    directory = study.DATA/"supplemental"
    directory.mkdir(parents=True)
    before = {}
    for symbol in study.NEW_SYMBOLS:
        source = directory/(symbol+".zip")
        path = directory/(symbol+".parquet")
        source.write_bytes(b"synthetic archived receipt")
        pd.DataFrame({"close": [1.0]}).to_parquet(path, index=False)
        before[path] = path.read_bytes()
        receipt = {"symbol": symbol, "url": "https://data.binance.vision/synthetic-test-only.zip",
                   "source_path": str(source.relative_to(tmp_path)), "path": str(path.relative_to(tmp_path)),
                   "official_sha256": study.model.sha256(source), "sha256": study.model.sha256(path)}
        (directory/f"{symbol}_mark_2026-06-29.receipt.json").write_text(json.dumps(receipt), encoding="utf8")
    result = study.fetch_mark_repair()
    proposal = json.loads(study.Path(result["proposal"]).read_text())
    assert len(proposal["mark_research_references"]) == 2
    for ref in proposal["mark_research_references"]:
        relative = study.Path(ref["local_path"])
        assert ref["local_path"] == relative.as_posix()
        assert not relative.is_absolute() and not relative.drive and ".." not in relative.parts
        path = tmp_path/relative
        assert ref["bytes"] == len(before[path]) > 0
        assert study.model.sha256(path) == ref["sha256"]
        assert path.read_bytes() == before[path]
    # The same checked metadata builder is also used for the full hourly mark
    # proposal. Neither a changed input nor an outside path can be registered.
    path = next(iter(before))
    digest = study.model.sha256(path)
    path.write_bytes(b"synthetic changed input")
    with pytest.raises(ValueError, match="hash mismatch"):
        study._mark_reference_file_metadata(path, digest)
    outside = tmp_path/"other.parquet"
    outside.write_bytes(b"outside isolated study")
    with pytest.raises(ValueError, match="isolated"):
        study._mark_reference_file_metadata(outside, study.model.sha256(outside))


def test_study_registry_gate_does_not_resolve_unrelated_descriptive_templates(tmp_path):
    path = tmp_path/"observed.parquet"
    records = {"datasets": [
        {"id": "legacy", "source_path": "C:/source/{SYMBOL}_1h.parquet | {SYMBOL}_1d.parquet"},
        {"id": study.STUDY+"_XRPUSDT_klines_5m", "source_path": str(path)},
        {"id": "some_other_study", "source_path": str(tmp_path/"other.parquet")},
    ]}
    assert study._registered_research_paths(records) == {str(path.resolve())}
