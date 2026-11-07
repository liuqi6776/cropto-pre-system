# Eleven-stream research candidate

This is an isolated research preparation path, not a deployed policy. The
running five streams, production model bytes, thresholds, order, price limit,
capital limits and execution rules are unchanged by these commands. No live
orders or account endpoints are available in this module.

## What exists and what is missing

Checked on 2026-11-07:

- The attachment contains text only, not the claimed full training/data code.
- Seven existing local model bundles are available. No XRP/DOGE ARGON bundles
  were found in the inspected local model locations.
- XRP/DOGE hourly trade, hourly mark and funding data for January 2024 through
  September 2026 exist locally: 198 monthly partitions, 102,360 rows. Both the
  normalized SHA256 and source ZIP SHA256 match official acquisition receipts.
- Two September 2026 five-minute samples were downloaded from Binance's official
  USD-M archive: 17,280 rows, 739,925 compressed bytes. Both official checksums,
  full-month grids and exact candle close timestamps passed validation.
- The subsequently authorized full collection completed all 66 five-minute
  symbol/month partitions: 24,856,015 compressed bytes, each official checksum
  and full-month schema/grid verified. The initial sample-only training block
  was tested before completing these inputs.
- A full coverage check found one day missing from each old monthly mark source:
  2026-06-29. Separate official daily archives supplied 24 actual hourly records
  per coin. Their checksums and timestamps were verified, and only missing
  timestamps were added to the isolated view. Original monthly sources remain
  unchanged; conflicting overlaps are refused.
- The prepared 26-source research view occupies 146,846,640 bytes. Its generated
  source and mark registration proposals must be merged before training. This
  input preparation is not a measured eleven-stream profit or saturation result.

The complete ZIP acquisition took 69.17 seconds in this run. Preparing the
isolated view additionally copied 108,433,642 bytes of existing shared inputs,
plus the new normalized data.
Feature matrices alone use roughly 0.44 GB per symbol for 311+65 float32 columns;
fitting creates additional float64 arrays. Plan for several GB of free RAM and
run one symbol at a time with two threads. Prior seven-stream training took
about three minutes on this machine, but that is context, not an eleven-stream
completion estimate. Time and memory must be measured when complete inputs exist.

## Commands

Run from the workspace root:

```powershell
.venv\Scripts\python.exe -m tools.argon4_research11 inventory
.venv\Scripts\python.exe -m tools.argon4_research11 audit-auxiliary
.venv\Scripts\python.exe -m tools.argon4_research11 fetch-month --month 2026-09
.venv\Scripts\python.exe -m tools.argon4_research11 fetch-mark-repair
.venv\Scripts\python.exe -m tools.argon4_research11 prepare
.venv\Scripts\python.exe -m tools.argon4_research11 train --threads 2
.venv\Scripts\python.exe -m pytest tests/test_argon4_research11.py -q
```

Use the project interpreter: its Python 3.12.11 / scikit-learn 1.7.2 match the
frozen models. A trial using the system interpreter was correctly rejected at
the model version check before any fitting; the version guard was not bypassed.

`fetch-month` accepts one explicitly selected month in January 2024–September
2026 and exactly the two new symbols. Each ZIP is capped at 8 MiB, only the
official public archive URL is allowed, redirects are refused, and provider
checksums are verified. Files and receipts are immutable; failed or partial
preparations require review rather than being silently overwritten. No broad
historical download is scheduled automatically.

`fetch-mark-repair` is restricted to the diagnosed 2026-06-29 hourly mark gap
for these two symbols. It writes separate immutable daily ZIPs, normalized
supplements, receipts and mark registration proposals. It cannot fetch an
arbitrary day or fabricate the missing observations.

Downloads live under `research_data/argon4_research11_20261107/`. Their generated
`registration_proposal_YYYY-MM.json` entries must be merged into
`archive/catalog/dataset_registry.json` as research sources with
`convertible: false`. The script does not edit the shared registry.

Once all months are present, `prepare` verifies and creates a separate 26-source
view, preserving the original 18 files byte for byte and adding eight new input
files. It writes `prepared_manifest.json` and
`prepared_registration_proposal.json`; the latter includes eight source entries
and two mark research references. Merge these into the dataset and mark
registries before training. `train` checks the entire manifest, file hashes and
registrations and refuses missing sources. Models and comparisons go only into
`research/argon4_research11_20261107/`. They cannot be loaded by the production
loader without a separate reviewed integration.

Funding timestamps keep their observed milliseconds and irregular spacing;
they are never snapped to a grid. The legacy `8h_funding` mapping is allowed
only while all provider-declared funding intervals are eight hours. Original
derived-feature startup behavior is retained; raw candles and settlements are
never synthesized or filled.

## Feature and training contract

For the original four coins, the production feature function is called without
alteration. Existing seven model bundles are loaded read-only with their existing
weights; the candidate comparison only derives a separate base-score threshold
from validation. It never overwrites their original thresholds or manifests.

For XRP/DOGE, an isolated copy of the guarded feature function adds the actual
target's own price, funding and mark/trade inputs. The frozen source formulas
remain hash-checked. The partner is BTC. Target/partner dispersion columns mean
target, BTC, ETH and BNB, as dictated by the existing target-relative formula.
The specialist's shared OFI basket remains BTC/ETH/SOL/BNB divided by four;
XRP/DOGE are not added to it. No coin is relabelled as another. Tests execute the
real frozen formulas on explicitly synthetic test fixtures and verify the
target mapping, 311+65 schema and unchanged four-coin OFI context. Those fixtures
are not study observations or profit evidence.

| Stream | Candidate validation quantile |
| --- | ---: |
| BTC 15m | 0.80 |
| ETH 5m | 0.88 |
| SOL 15m | 0.85 |
| SOL 5m | 0.88 |
| BTC 5m | 0.90 |
| ETH 15m | 0.85 |
| BNB 15m | 0.88 |
| XRP 5m / DOGE 5m | 0.88 |
| XRP 15m / DOGE 15m | 0.85 |

These are prespecified candidate hypotheses from the attachment, not approved
production settings or a funding priority. The quantile is fitted to absolute
**base** validation scores and applied to the frozen blended score, matching the
existing frozen-threshold convention. It is not a blend-score reset quantile.
The new-target weight grid uses validation only. Normalizers use validation
only; scalers and estimators use training only. No test score or outcome selects
a threshold or weight.

The comparison reuses the local `oos_2026` split: training through September 2025,
validation October–December 2025, comparison January–September 2026, with the
existing label purges at boundaries. This historical period has already been
researched and is **not untouched holdout evidence**. It does not reproduce the
attachment's alleged January 2025 start or 631-day result. An independent future
evaluation period must be frozen before any effectiveness claim or switch.

Test signal selection uses scores only. Exact ties, very small returns and
unsettled labels remain visible. The original training-only return clipping and
tie exclusion do not become test-time hindsight filters. Each newly trained
bundle is reloaded and required to reproduce its pre-save scores exactly.

## Local run results

The full local comparison completed on 2026-11-07 using two threads in 82.70
seconds, with peak process RSS 5,340,770,304 bytes (about 4.97 GiB). All four new
research bundles passed exact saved-score parity. All 42 existing model-related
files remained byte-identical. Seven original-threshold signal counts, wins and
accuracy matched the previous local report exactly. Ten targeted tests passed
at that stage. A later compatibility check found four new mark references used
Windows path separators and lacked file-size metadata. The generator and those
four entries were corrected to verified relative POSIX paths and actual byte
sizes; data and model hashes stayed unchanged. The resulting eleven research
tests and the affected foreign-reference rejection test all passed (12 total).
Production monitoring and other validation were running concurrently; this is
a measured run duration, not an isolated performance benchmark.

The columns below refer to the previously researched January–September 2026
comparison period. Payout means the all-signals-filled, **one share per signal**
scenario specified below, including its fee assumption; it is not the 10,000 USD
paper account's return. A dash means there is no existing local baseline model.

| Stream | Baseline signals | Candidate signals | Baseline accuracy | Candidate accuracy | Baseline hypothetical payout | Candidate hypothetical payout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BTC 15m | 4,417 | 5,823 | 56.26% | 55.52% | 133.02 | 132.34 |
| BTC 5m | 10,738 | 12,770 | 53.97% | 53.81% | 77.18 | 71.18 |
| ETH 5m | 6,769 | 9,980 | 55.62% | 55.19% | 160.61 | 193.81 |
| ETH 15m | 2,606 | 3,889 | 53.91% | 54.02% | 17.35 | 30.17 |
| SOL 15m | 2,029 | 3,852 | 55.25% | 54.96% | 40.59 | 65.87 |
| SOL 5m | 6,460 | 9,834 | 54.50% | 54.47% | 81.15 | 120.55 |
| BNB 15m | 1,389 | 2,250 | 53.06% | 54.22% | -2.62 | 21.91 |
| XRP 5m | — | 14,047 | — | 52.99% | — | -36.81 |
| XRP 15m | — | 5,113 | — | 53.80% | — | 28.41 |
| DOGE 5m | — | 8,841 | — | 52.98% | — | -23.69 |
| DOGE 15m | — | 4,018 | — | 52.71% | — | -21.52 |

The baseline seven produce 34,408 signals and 507.28 hypothetical payout units;
the candidate seven produce 48,398 and 635.83. Adding the four new flows produces
80,417 signals but reduces this scenario to 582.21: the added four contribute
**−53.61**. Three of the four new flows are negative even before introducing
actual fill, deadline and capital constraints. BTC 15m and BTC 5m also have a
slightly lower hypothetical total after their threshold changes. More signals
are therefore not sufficient evidence for either blanket threshold relaxation
or eleven-stream deployment.

Only historical `oos_2026` research bundles were created for the new targets.
There are no new current-period deployment bundles, production manifests,
policy pins or stream registrations. No live or paper production switch was
made. The small positive XRP 15m proxy is a further-research hypothesis, not
evidence of an effective or currently deployable strategy.

## What the payout comparison means

The output includes direction accuracy plus an explicitly hypothetical
one-share-per-signal payout scenario at price 0.515 and fee rate 0.07. The
official taker fee formula is `shares × feeRate × price × (1 − price)`, so this
scenario has per-share cost 0.53248425 and break-even accuracy 53.248425%, before
other effects. Actual fees are market-specific and must be queried for actual
execution; no maker rebate is assumed. See the
[official fee documentation](https://docs.polymarket.com/trading/fees).

This is not an execution backtest: it assumes every selected signal fills and
uses Binance next-open/window-close direction, rather than the market's
Chainlink TWAP resolution. It has no observed queue, book capacity, deadline,
settlement delay, cash-overlap or correlated-risk model. It therefore cannot
support the attachment's dollar-profit, Sharpe, saturation-time or withdrawal
claims. Risk caps and profit withdrawals are not implemented here.

Official market pages establish that these market types exist:
[XRP 5m](https://polymarket.com/event/xrp-updown-5m-1794114300),
[XRP 15m](https://polymarket.com/event/xrp-updown-15m-1793780100),
[DOGE 5m](https://polymarket.com/event/doge-updown-5m-1794114600), and
[DOGE 15m](https://polymarket.com/event/doge-updown-15m-1789470000).
They do not establish continuous listings or executable depth in future windows.
Archive provenance follows the
[official Binance public-data repository](https://github.com/binance/binance-public-data).

The mandatory local `strategy-validation-gates/SKILL.md` could not be located.
No strategy efficacy gate has passed. Before a switch, restore that review
process, complete registered data and model tests, freeze independent validation,
verify official labels and real execution constraints, and review the proposed
policy separately. Paper operation alone is not evidence of profitability.
