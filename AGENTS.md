# Source archive rules

- Public market-data reads and isolated local simulation only. Never transmit, cancel or enable real exchange orders.
- Never commit credentials, private keys, personal account state, model binaries, datasets or ledgers.
- Do not invent candles, funding settlements, fill events or historical orders. Report missing data and preserve provenance.
- Keep model/code/data hash checks, decision deadlines, risk caps and the configured stage priority intact.
- Frozen sources use byte-level SHA256 checks. Preserve exact bytes; .gitattributes disables line ending conversion.
- This is a source-only export. Empty registry templates are not verified data, and missing artifacts must fail closed.
- After changes, run relevant tests using python -B -m pytest tests -q. PyTorch-only research tests require the optional dependency.
- Historical report helper scripts may depend on omitted snapshots and the original larger workspace. Do not represent them as current deployment gates.
- Functional tests do not establish strategy profitability or authorize live trading.
