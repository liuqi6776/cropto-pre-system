"""Read-only Chainlink 60s TWAP vs. Binance 5m candle settlement label alignment analyzer.

Compares Polymarket official resolution outcomes (OBSERVED_OFFICIAL_RESOLUTION)
against local Binance 5m futures candles over [window_start, window_end):
  1. binance_point_outcome: "UP" if close(window_end) >= open(window_start) else "DOWN"
  2. binance_vwap_outcome:  "UP" if sum(quote_volume)/sum(volume) >= open(window_start) else "DOWN"
  3. abs_ret_bps:           abs(close_end - open_start) / open_start * 10000

Outputs label agreement rates by stream, timeframe (5m vs 15m), and return
magnitude bucket (<1_bps, 1_to_3_bps, 3_to_10_bps, >10_bps), plus realized win
rate under official resolution vs. Binance point close.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RETURN_BUCKETS = ("<1_bps", "1_to_3_bps", "3_to_10_bps", ">10_bps")
ASK_BUCKETS = ("DeepDiscount (<0.35)", "MidDiscount (0.35-0.44)", "NearPar (0.44-0.515)")
ADVERSE_SELECTION_ZONES = ("NORMAL_ASK (>=0.37)", "TOXIC_DISCOUNT (<=0.36)")
RV12_BUCKETS = ("UltraLow (<3_bps)", "Low (3_to_6_bps)", "Normal (6_to_15_bps)", "High (>15_bps)")


def classify_return_bucket(abs_ret_bps: float) -> str:
    val = float(abs_ret_bps)
    if val < 1.0:
        return "<1_bps"
    if val < 3.0:
        return "1_to_3_bps"
    if val < 10.0:
        return "3_to_10_bps"
    return ">10_bps"


def classify_ask_bucket(ask_price: float | None) -> str | None:
    if ask_price is None:
        return None
    val = float(ask_price)
    if val < 0.35:
        return "DeepDiscount (<0.35)"
    if val < 0.44:
        return "MidDiscount (0.35-0.44)"
    return "NearPar (0.44-0.515)"


def classify_adverse_selection_zone(ask_price: float | None) -> str | None:
    if ask_price is None:
        return None
    return "TOXIC_DISCOUNT (<=0.36)" if float(ask_price) < 0.365 else "NORMAL_ASK (>=0.37)"


def classify_rv12_bucket(rv12_bps: float | None) -> str | None:
    if rv12_bps is None:
        return None
    val = float(rv12_bps)
    if val < 3.0:
        return "UltraLow (<3_bps)"
    if val < 6.0:
        return "Low (3_to_6_bps)"
    if val < 15.0:
        return "Normal (6_to_15_bps)"
    return "High (>15_bps)"


def _symbol_for_position(stream: str | None = None, underlying: str | None = None, symbol: str | None = None) -> str:
    if symbol:
        text = str(symbol).upper()
        return text if text.endswith("USDT") else f"{text}USDT"
    if underlying:
        text = str(underlying).upper()
        return text if text.endswith("USDT") else f"{text}USDT"
    if stream and "_" in str(stream):
        base = str(stream).split("_", 1)[0].upper()
        return f"{base}USDT"
    raise ValueError("Unable to infer Binance symbol from stream/underlying")


def compute_settlement_alignment_row(
    data_root: str | Path,
    *,
    stream: str,
    window_start: str,
    window_end: str,
    official_outcome: str,
    market_id: str | None = None,
    prediction_side: str | None = None,
    symbol: str | None = None,
    ask_price: float | None = None,
    effective_price: float | None = None,
    realized_pnl_usd: float | None = None,
) -> dict[str, Any]:
    official = str(official_outcome).upper()
    if official not in ("UP", "DOWN"):
        raise ValueError(f"Unsupported official_outcome: {official_outcome}")
    sym = _symbol_for_position(stream=stream, symbol=symbol)
    parquet_path = Path(data_root) / "intraday" / f"{sym}_5m_futures.parquet"
    if not parquet_path.is_file():
        return {
            "status": "MISSING_CANDLE_FILE",
            "stream": stream,
            "symbol": sym,
            "market_id": market_id,
            "window_start": window_start,
            "window_end": window_end,
            "official_outcome": official,
            "prediction_side": prediction_side,
        }
    frame = pd.read_parquet(parquet_path)
    if "date" not in frame.columns and frame.index.name == "date":
        frame = frame.reset_index()
    dates = pd.to_datetime(frame["date"], utc=True)
    start_ts = pd.Timestamp(window_start).tz_convert("UTC") if pd.Timestamp(window_start).tzinfo else pd.Timestamp(window_start, tz="UTC")
    end_ts = pd.Timestamp(window_end).tz_convert("UTC") if pd.Timestamp(window_end).tzinfo else pd.Timestamp(window_end, tz="UTC")
    window_bars = frame.loc[(dates >= start_ts) & (dates < end_ts)].copy()
    if window_bars.empty:
        return {
            "status": "MISSING_WINDOW_BARS",
            "stream": stream,
            "symbol": sym,
            "market_id": market_id,
            "window_start": window_start,
            "window_end": window_end,
            "official_outcome": official,
            "prediction_side": prediction_side,
        }
    window_bars = window_bars.sort_values("date")
    pre_bars = frame.loc[dates < start_ts].sort_values("date").tail(12)
    ex_ante_rv12_bps: float | None = None
    if len(pre_bars) > 0:
        p_opens = pre_bars["open"].astype(float)
        p_closes = pre_bars["close"].astype(float)
        if (p_opens > 0).all():
            ex_ante_rv12_bps = float(((p_closes - p_opens).abs() / p_opens * 10000.0).mean())
    open_start = float(window_bars.iloc[0]["open"])
    close_end = float(window_bars.iloc[-1]["close"])
    total_vol = float(window_bars["volume"].sum()) if "volume" in window_bars.columns else 0.0
    total_quote = (
        float(window_bars["quote_volume"].sum())
        if "quote_volume" in window_bars.columns
        else close_end * total_vol
    )
    vwap = total_quote / total_vol if total_vol > 0.0 else close_end
    binance_point_outcome = "UP" if close_end >= open_start else "DOWN"
    binance_vwap_outcome = "UP" if vwap >= open_start else "DOWN"
    abs_ret_bps = abs(close_end - open_start) / open_start * 10000.0 if open_start > 0.0 else 0.0
    ret_bucket = classify_return_bucket(abs_ret_bps)
    timeframe = stream.split("_", 1)[1] if "_" in str(stream) else "unknown"
    side = str(prediction_side).upper() if prediction_side in ("UP", "DOWN", "up", "down") else None
    row: dict[str, Any] = {
        "status": "ALIGNED_CHECKED",
        "stream": stream,
        "timeframe": timeframe,
        "symbol": sym,
        "market_id": market_id,
        "window_start": start_ts.isoformat(),
        "window_end": end_ts.isoformat(),
        "bar_count": int(len(window_bars)),
        "open_start": open_start,
        "close_end": close_end,
        "vwap": vwap,
        "abs_ret_bps": abs_ret_bps,
        "return_bucket": ret_bucket,
        "ex_ante_rv12_bps": ex_ante_rv12_bps,
        "rv12_bucket": classify_rv12_bucket(ex_ante_rv12_bps),
        "ask_price": float(ask_price) if ask_price is not None else None,
        "effective_price": float(effective_price) if effective_price is not None else None,
        "ask_bucket": classify_ask_bucket(ask_price),
        "adverse_selection_zone": classify_adverse_selection_zone(ask_price),
        "realized_pnl_usd": float(realized_pnl_usd) if realized_pnl_usd is not None else None,
        "official_outcome": official,
        "binance_point_outcome": binance_point_outcome,
        "binance_vwap_outcome": binance_vwap_outcome,
        "point_aligned": official == binance_point_outcome,
        "vwap_aligned": official == binance_vwap_outcome,
        "prediction_side": side,
        "official_win": (side == official) if side else None,
        "binance_point_win": (side == binance_point_outcome) if side else None,
        "binance_vwap_win": (side == binance_vwap_outcome) if side else None,
    }
    return row


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [r for r in rows if r.get("status") == "ALIGNED_CHECKED"]
    n = len(checked)
    if n == 0:
        return {
            "count": 0,
            "point_agreement_rate": None,
            "vwap_agreement_rate": None,
            "official_win_rate": None,
            "binance_point_win_rate": None,
            "win_rate_delta": None,
            "realized_pnl_usd": 0.0,
        }
    point_hits = sum(1 for r in checked if r.get("point_aligned"))
    vwap_hits = sum(1 for r in checked if r.get("vwap_aligned"))
    with_side = [r for r in checked if r.get("prediction_side") in ("UP", "DOWN")]
    if with_side:
        off_win = sum(1 for r in with_side if r.get("official_win")) / len(with_side)
        pt_win = sum(1 for r in with_side if r.get("binance_point_win")) / len(with_side)
        delta = off_win - pt_win
    else:
        off_win = pt_win = delta = None
    total_pnl = sum(float(r["realized_pnl_usd"]) for r in checked if r.get("realized_pnl_usd") is not None)
    return {
        "count": n,
        "point_agreement_rate": point_hits / n,
        "vwap_agreement_rate": vwap_hits / n,
        "official_win_rate": off_win,
        "binance_point_win_rate": pt_win,
        "win_rate_delta": delta,
        "realized_pnl_usd": round(total_pnl, 4),
    }


def summarize_settlement_alignments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [r for r in rows if r.get("status") == "ALIGNED_CHECKED"]
    streams = sorted({str(r.get("stream")) for r in checked if r.get("stream")})
    timeframes = sorted({str(r.get("timeframe")) for r in checked if r.get("timeframe")})
    by_stream = {s: _group_metrics([r for r in checked if r.get("stream") == s]) for s in streams}
    by_timeframe = {tf: _group_metrics([r for r in checked if r.get("timeframe") == tf]) for tf in timeframes}
    by_bucket = {
        b: _group_metrics([r for r in checked if r.get("return_bucket") == b])
        for b in RETURN_BUCKETS
    }
    by_ask_bucket = {
        b: _group_metrics([r for r in checked if r.get("ask_bucket") == b])
        for b in ASK_BUCKETS
    }
    by_adverse_zone = {
        z: _group_metrics([r for r in checked if r.get("adverse_selection_zone") == z])
        for z in ADVERSE_SELECTION_ZONES
    }
    by_rv12_bucket = {
        b: _group_metrics([r for r in checked if r.get("rv12_bucket") == b])
        for b in RV12_BUCKETS
    }
    guarded_subset = [
        r for r in checked
        if (r.get("ask_price") is None or float(r["ask_price"]) >= 0.365)
        and (r.get("ex_ante_rv12_bps") is None or float(r["ex_ante_rv12_bps"]) >= 3.0)
    ]
    disagreements = [r for r in checked if not r.get("point_aligned")]
    return {
        "total_records": len(rows),
        "checked_records": len(checked),
        "overall": _group_metrics(checked),
        "by_stream": by_stream,
        "by_timeframe": by_timeframe,
        "by_return_bucket": by_bucket,
        "by_ask_bucket": by_ask_bucket,
        "by_adverse_selection_zone": by_adverse_zone,
        "by_rv12_bucket": by_rv12_bucket,
        "combined_guards_simulation": _group_metrics(guarded_subset),
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:50],
    }


def analyze_runtime_settlement_alignment(
    runtime_root: str | Path,
    *,
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    runtime = Path(runtime_root)
    raw_root = Path(data_root) if data_root is not None else runtime / "features" / "raw_data"
    account_db = runtime / "account.sqlite3"
    if not account_db.is_file():
        account_db = runtime / "paper_account.sqlite3"
    rows: list[dict[str, Any]] = []
    if account_db.is_file():
        with sqlite3.connect(account_db.resolve().as_uri() + "?mode=ro", uri=True, timeout=1.0) as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
            if {"orders", "resolutions", "signals"} <= tables:
                settled_rows = conn.execute(
                    "SELECT o.market_id, o.outcome, o.window_end, o.payload, s.payload, r.payload "
                    "FROM orders o "
                    "JOIN resolutions r ON o.market_id = r.market_id "
                    "LEFT JOIN signals s ON o.signal_id = s.signal_id "
                    "ORDER BY o.fill_at ASC"
                ).fetchall()
                for market_id, pred_side, w_end_col, o_raw, s_raw, r_raw in settled_rows:
                    o_pay = json.loads(o_raw) if o_raw else {}
                    s_pay = json.loads(s_raw) if s_raw else {}
                    r_pay = json.loads(r_raw) if r_raw else {}
                    stream = s_pay.get("stream") or o_pay.get("stream")
                    window_start = s_pay.get("window_start") or o_pay.get("venue_market", {}).get("window_start")
                    window_end = s_pay.get("window_end") or w_end_col or o_pay.get("venue_market", {}).get("window_end")
                    official_outcome = r_pay.get("outcome") or r_pay.get("winning_outcome")
                    qty = float(o_pay.get("quantity", 0.0) or 0.0)
                    cost_usd = float(o_pay.get("cost_usd", 0.0) or 0.0)
                    fee_usd = float(o_pay.get("fee_usd", 0.0) or 0.0)
                    ask_px = (cost_usd / qty) if qty > 0 else None
                    eff_px = ((cost_usd + fee_usd) / qty) if qty > 0 else None
                    pnl_usd = (qty - (cost_usd + fee_usd)) if (pred_side == official_outcome) else -(cost_usd + fee_usd)
                    if stream and window_start and window_end and official_outcome in ("UP", "DOWN"):
                        rows.append(
                            compute_settlement_alignment_row(
                                raw_root,
                                stream=stream,
                                window_start=window_start,
                                window_end=window_end,
                                official_outcome=official_outcome,
                                market_id=str(market_id),
                                prediction_side=pred_side,
                                ask_price=ask_px,
                                effective_price=eff_px,
                                realized_pnl_usd=pnl_usd,
                            )
                        )
            elif "settlements" in tables:
                settled_rows = conn.execute(
                    "SELECT s.market_id, s.outcome, s.payload, p.payload "
                    "FROM settlements s LEFT JOIN positions p ON s.position_id = p.position_id "
                    "ORDER BY s.id ASC"
                ).fetchall()
                for market_id, winning_outcome, s_payload_raw, p_payload_raw in settled_rows:
                    s_payload = json.loads(s_payload_raw) if s_payload_raw else {}
                    p_payload = json.loads(p_payload_raw) if p_payload_raw else {}
                    stream = p_payload.get("stream") or s_payload.get("stream")
                    window_start = p_payload.get("window_start") or s_payload.get("window_start")
                    window_end = p_payload.get("window_end") or s_payload.get("window_end")
                    prediction_side = p_payload.get("outcome")
                    official_outcome = s_payload.get("winning_outcome") or s_payload.get("outcome") or winning_outcome
                    if stream and window_start and window_end and official_outcome in ("UP", "DOWN"):
                        rows.append(
                            compute_settlement_alignment_row(
                                raw_root,
                                stream=stream,
                                window_start=window_start,
                                window_end=window_end,
                                official_outcome=official_outcome,
                                market_id=str(market_id),
                                prediction_side=prediction_side,
                            )
                        )
    summary = summarize_settlement_alignments(rows)
    return {
        "runtime_root": str(runtime),
        "data_root": str(raw_root),
        "rows": rows,
        "summary": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Chainlink TWAP vs Binance 5m settlement label alignment.")
    parser.add_argument("--runtime", type=Path, default=ROOT / "paper_runtime" / "argon4_forward_20261105")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    report = analyze_runtime_settlement_alignment(args.runtime, data_root=args.data_root)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
