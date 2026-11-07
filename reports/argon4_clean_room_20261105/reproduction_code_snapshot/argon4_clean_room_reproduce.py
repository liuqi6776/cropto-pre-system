"""Instrument the supplied source without changing its model calculations.

All original inputs are freshly downloaded public observations. AST hooks only
export intermediate results. Missing economic components remain explicitly
unimplemented, rather than being inferred from the attachment's headline numbers.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.argon4_clean_room_fetch import DATA, STUDY, SYMBOLS, sha, write_json

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

ORIGINAL = ROOT / "research" / STUDY / "original"
OUT = ROOT / "reports" / STUDY


def load_original():
    manifest = json.loads((ORIGINAL / "extraction_manifest.json").read_text(encoding="utf8"))
    for name, expected in manifest["files"].items():
        if sha(ORIGINAL / name) != expected:
            raise ValueError(f"Supplied source changed: {name}")
    return manifest


def verify_data():
    manifest = json.loads((DATA / "data_manifest.json").read_text(encoding="utf8"))
    for item in manifest["datasets"]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError(f"Input changed: {item['path']}")
    return manifest


def hook(kind):
    return ast.parse(f"_capture('{kind}', locals())").body[0]


class Capture(ast.NodeTransformer):
    """Insert side-effect-only exports at three unambiguous source anchors."""
    def __init__(self):
        self.counts = {"binary": 0, "positions": 0, "done": 0}

    def visit_Assign(self, node):
        if len(node.targets) == 1:
            t = node.targets[0]
            if isinstance(t, ast.Name) and t.id == "th":
                self.counts["binary"] += 1
                return [node, hook("binary")]
            if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "pos_by_fold":
                self.counts["positions"] += 1
                return [node, hook("positions")]
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.name == "run_clean_room_pipeline":
            node.body.append(hook("done"))
            self.counts["done"] += 1
        return node


def load_module(instrument=True):
    path = ORIGINAL / "step2_standalone_clean_room_master.py"
    spec = importlib.util.spec_from_file_location("argon4_supplied_clean_room", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    tree = ast.parse(path.read_text(encoding="utf8"))
    if instrument:
        c = Capture()
        tree = c.visit(tree)
        if c.counts != {"binary": 1, "positions": 1, "done": 1}:
            raise ValueError(f"Unexpected source instrumentation anchors: {c.counts}")
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    # Explicit local independent input root; no automatic fallback to legacy data.
    mod._locate_data_root = lambda: DATA / "raw_data"
    return mod


def export_binary(l, out, label):
    fold = l["fold"]
    name = l["name"].split()[0]
    idx = l["b_obj"]["index"]
    test = l["te_idx"]
    z = l["te_z"]
    d = pd.DataFrame({"decision_bar_open_utc": idx[test], "decision_close_utc": idx[test] + pd.Timedelta("5min"),
                      "label_end_utc": idx[test] + pd.Timedelta(minutes=5 + 5 * l["k_bars"]),
                      "z": z, "threshold": l["th"], "selected": np.abs(z) >= l["th"],
                      "side": np.where(z > 0, 1, -1), "forward_return": l["y_ret"][test]})
    d["correct"] = np.sign(d.forward_return) == d.side
    d.to_parquet(out / f"{label}_{name}_{fold.year}_predictions.parquet", index=False)
    meta = {"stream": name, "year": fold.year, "train_rows": len(l["tr_idx"]),
            "validation_rows": len(l["va_idx"]), "test_rows": len(test), "features": l["X_mat"].shape[1],
            "threshold": float(l["th"]), "train_last_label_end_utc": str(idx[l["tr_idx"][-1]] + pd.Timedelta(minutes=5 + 5 * l["k_bars"])),
            "validation_first_decision_utc": str(idx[l["va_idx"][0]] + pd.Timedelta("5min")),
            "last_test_label_end_utc": str(d.label_end_utc.iloc[-1]), "label": label}
    write_json(out / f"{label}_{name}_{fold.year}_fit.json", meta)
    print(f"FITTED {label} {name} {fold.year}: {d.selected.sum()} selected", flush=True)
    return d


def prepare_monthly_marks():
    """Retain the exact monthly-only observations the supplied crawler would read.

    Gaps remain gaps. The original model's *derived basis feature* may carry a
    stale value across them; no observed price is filled or created here.
    """
    from scripts.argon4_clean_room_fetch import read_partition, validate, save_frame, registry_entry
    data = verify_data()
    items = []
    mapping = {}
    for sym in SYMBOLS:
        receipts = sorted([r for r in data["archives"] if r["spec"] == ["futures/um", "markPriceKlines", sym, "1h"]], key=lambda r: r["month"])
        for r in receipts:
            if sha(ROOT / r["path"]) != r["official_sha256"]:
                raise ValueError("Original monthly archive changed")
        d = pd.concat([read_partition(ROOT / r["path"]) for r in receipts], ignore_index=True)
        d = d.rename(columns={"open_time": "date"})[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)
        p = DATA / "monthly_only_marks" / f"{sym}_1h_mark.parquet"
        item = {**save_frame(d, p, validate(d, "date", "1h")), "symbol": sym}
        items.append(item)
        mapping[(DATA / "raw_data/auxiliary" / p.name).resolve()] = p
    manifest = {"status": "OBSERVED_MONTHLY_ONLY_WITH_REPORTED_GAPS", "parent_manifest_sha256": sha(DATA / "data_manifest.json"), "datasets": items}
    write_json(DATA / "monthly_only_marks/manifest.json", manifest)
    rp = ROOT / "archive/catalog/dataset_registry.json"
    reg = json.loads(rp.read_text(encoding="utf8"))
    entries = [registry_entry(f"literal_monthly_mark_{i['symbol']}", ROOT / i["path"], "futures_usds_m_perpetual", "markPriceKlines", [i["symbol"]], "1h", f"Exact observed Binance monthly markPriceKlines, ZIP .CHECKSUM verified. Gaps retained; no daily repair. SHA256={i['sha256']}; research_data/{STUDY}/monthly_only_marks/manifest.json", True) for i in items]
    ids = {i["id"] for i in entries}
    reg["datasets"] = [i for i in reg["datasets"] if i["id"] not in ids] + entries
    write_json(rp, reg)
    rp = ROOT / "archive/catalog/mark_candle_registry.json"
    reg = json.loads(rp.read_text(encoding="utf8"))
    refs = [{"dataset_id": f"{STUDY}_literal_monthly_mark_{i['symbol']}", "pair": f"{i['symbol'][:-4]}/USDT:USDT", "timeframe": "1h", "candle_type": "mark", "local_path": i["path"], "extension": "parquet", "sha256": i["sha256"], "rows": i["rows"], "first_utc": i["first_utc"], "last_utc": i["last_utc"], "verification_status": "VERIFIED_PERPETUAL_LABEL", "provenance_evidence": "Official monthly-only markPriceKlines, checksum verified. Missing days retained and reported in study monthly_only_marks manifest; never a readiness claim.", "installed": False, "research_only": True, "gate_effect": "NONE"} for i in items]
    ids = {i["dataset_id"] for i in refs}
    reg["research_references"] = [i for i in reg.get("research_references", []) if i["dataset_id"] not in ids] + refs
    write_json(rp, reg)
    return mapping, manifest


def run_original(threads=4, monthly_only=False):
    load_original()
    data = verify_data()
    out = OUT / ("literal_monthly" if monthly_only else "literal")
    out.mkdir(parents=True, exist_ok=True)
    mod = load_module()
    build = mod.build_clean_room_bundle
    bundles_meta = {}

    def bundle(sym="BTCUSDT"):
        b = build(sym)
        bundles_meta[sym] = {"rows": len(b["index"]), "base_features": list(b["features"]), "structural_features": list(b["spec_features"]),
                             "base_count": b["features"].shape[1], "structural_count": b["spec_features"].shape[1],
                             "first_utc": str(b["index"][0]), "last_utc": str(b["index"][-1])}
        write_json(out / "feature_inventory.json", bundles_meta)
        # Reusable *newly computed* feature cache, never an imported model/cache.
        b["features"].to_parquet(out / f"{sym}_base_features.parquet")
        b["spec_features"].to_parquet(out / f"{sym}_structural_features.parquet")
        return b

    def capture(kind, l):
        if kind == "binary":
            export_binary(l, out, "literal")
        elif kind == "positions":
            f = l["fold"]
            pd.DataFrame({"decision_bar_open_utc": l["idx_b"][l["te_m"]], "position": l["pos"],
                          "e1": l["p_e1"], "e2": l["p_e2"], "e3_15": l["p_e3_15"], "e3_30": l["p_e3_30"], "e4": l["p_e4"]}).to_parquet(out / f"positions_{f.year}.parquet", index=False)
            print(f"PERP FITTED {f.year}", flush=True)
        elif kind == "done":
            pd.DataFrame(l["rows"]).to_csv(out / "binary_summary.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(l["perp_rows"]).to_csv(out / "perp_summary.csv", index=False, encoding="utf-8-sig")

    mod.build_clean_room_bundle = bundle
    mod._capture = capture
    start = time.time()
    mapping, monthly = prepare_monthly_marks() if monthly_only else ({}, None)
    metadata = {"status": "RUNNING", "classification": "LITERAL_CODE_REPRODUCTION_NOT_VALIDATED",
                "input_variant": "EXACT_MONTHLY_ONLY_WITH_GAPS" if monthly_only else "VERIFIED_DATA_WITH_OBSERVED_DAILY_MARK_REPAIRS",
                "monthly_only_input_manifest": monthly,
                "started_utc": pd.Timestamp.now(tz="UTC").isoformat(), "original": load_original(),
                "data_manifest_sha256": sha(DATA / "data_manifest.json"), "runner_sha256": sha(Path(__file__)),
                "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__},
                "threads": threads, "pretrained_models_used": False,
                "adaptations": ["Explicit newly downloaded data root", "AST result-export hooks only", "Numerical/model parameters unchanged"],
                "limitations": ["Original omits funding cashflows, final liquidation fee and liquidation path", "Same-time next-open fills are idealized", "Binary outcomes use Binance candles, not prediction-market oracle settlements", "Zero-return labels are excluded using future outcomes", "Derived feature NaNs are zero-filled by supplied code; observed data is not imputed", "Original hardcodes 631 days; calendar window is 632 days", "2024-2026 historical windows are previously touched, not untouched holdouts"]}
    write_json(out / "run.json", metadata)
    original_read = pd.read_parquet
    def read_selected(path, *args, **kwargs):
        chosen = mapping.get(Path(path).resolve(), path) if isinstance(path, (str, Path)) else path
        return original_read(chosen, *args, **kwargs)
    from unittest.mock import patch
    try:
        with threadpool_limits(limits=threads), patch.object(pd, "read_parquet", read_selected):
            mod.run_clean_room_pipeline()
        verify_data()
        if monthly:
            for item in monthly["datasets"]:
                if sha(ROOT / item["path"]) != item["sha256"]:
                    raise ValueError("Monthly-only inputs changed during run")
        metadata["status"] = "COMPLETED"
    except Exception:
        metadata["status"] = "FAILED"
        metadata["error"] = traceback.format_exc()
        raise
    finally:
        metadata["elapsed_seconds"] = time.time() - start
        metadata["finished_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        metadata["artifacts"] = {p.name: sha(p) for p in out.glob("*") if p.is_file() and p.name != "run.json"}
        write_json(out / "run.json", metadata)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--monthly-only-marks", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        load_original()
        mod = load_module()
        print(json.dumps({"status": "PLAN_ONLY", "original_folds": [f.__dict__ for f in mod.FOLDS], "data_root": str(DATA), "output": str(OUT)}))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ("literal_monthly_stdout.txt" if a.monthly_only_marks else "literal_stdout.txt")).open("w", encoding="utf8") as log:
        class Tee:
            def write(self, text):
                log.write(text)
                sys.__stdout__.write(text)
            def flush(self):
                log.flush()
                sys.__stdout__.flush()
        with redirect_stdout(Tee()):
            run_original(a.threads, a.monthly_only_marks)


if __name__ == "__main__":
    main()
