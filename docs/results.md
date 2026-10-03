# Results

All fleet numbers come from `artifacts/model_report.json` (full training run, 200 history and 240 live devices). Re-running `python -m bms_copilot.train` regenerates them; the simulator is seeded, so they reproduce.

> **Read this first.** The fleet is simulated. These results show the pipeline works end to end, against ground truth that real data cannot provide. They are **not** field accuracy. The real-laptop section is real data from one battery.

---

## Health estimation

### State of health

| Evaluation set | Snapshots | AI (GP) MAE | 95 % interval coverage | Cycle-counting gauge MAE |
|---|---:|---:|---:|---:|
| Held-out history devices | 5,833 | **0.27 %** | 93.0 % | 5.44 % |
| In-service fleet | 10,317 | **0.32 %** | 92.0 % | 4.80 % |

The GP is **15–20× more accurate** than the legacy gauge, and its error bars are close to calibrated: a perfect 95 % interval covers 95 % of truths.

### Degradation modes from routine charging data

| Mode | R² | MAE (fraction of capacity) |
|---|---:|---:|
| Loss of lithium inventory (LLI) | 0.999 | 0.10 pp |
| Cathode active-material loss (LAM<sub>pe</sub>) | 0.952 | 0.33 pp |
| Anode active-material loss (LAM<sub>ne</sub>) | 0.887 | 0.46 pp |

LLI is easy because it shifts every dQ/dV peak. Anode loss is the hardest because its signature overlaps with LLI. It is still accurate enough to screen supplier lots across the fleet without a dedicated diagnostic capture.

### Remaining useful life (days to 80 % SoH)

| Evaluation set | n | MAE | Median relative error | 95 % coverage | Mean interval width |
|---|---:|---:|---:|---:|---:|
| Held-out history devices | 2,529 | **45 days** | 7.9 % | 91.3 % | n/a |
| … final year before end of life | n/a | **28 days** | n/a | n/a | n/a |
| In-service fleet (all) | 178 | 61 days | 9.2 % | 93.3 % | 327 days |
| … adaptive-charging cohort | 75 | 71 days | 6.3 % | 100 % | 457 days |
| … legacy-charging cohort | 103 | 53 days | 11.2 % | 88.3 % | 232 days |

Predictions sharpen as end of life approaches, which is when they matter most. Adaptive-charging devices have wider intervals because they are further from end of life, so there is more future to be uncertain about.

---

## Digital twin and state of charge

| Metric | Physics only | Hybrid (physics + learned residual) |
|---|---:|---:|
| Terminal-voltage RMSE (12 test runs) | 30.7 mV | **9.4 mV** (−69 %) |

| SoC estimator on an aged cell (2.25 h, 4,056 steps) | MAE | Max error | Final error |
|---|---:|---:|---:|
| Legacy coulomb counting | 6.29 % | 14.1 % | 14.1 % |
| EKF, beginning-of-life parameters | 1.83 % | 3.9 % | 0.8 % |
| EKF, recalibrated from the twin | **1.44 %** | **2.3 %** | 0.9 % |

Coulomb counting drifts without bound because its capacity figure is stale. The voltage feedback in the EKF keeps the error bounded, and the twin's recalibration halves the worst case.

---

## Early-warning detector

| Metric | Result |
|---|---:|
| Healthy snapshots tested | 8,818 |
| False warning rate | **0.2 %** |
| False critical rate | 0 % |
| Defective devices detected (13 swelling, 2 micro-short) | 15 / 15 |
| Mechanism attributed correctly | 15 / 15 |
| Median delay, onset → first warning | **10 days** |
| Median lead time, warning → critical | **112 days** |

Swelling cells were flagged 7–18 days after onset, with 98–112 days of lead time before they would trip a hard limit. The two micro-shorts were flagged 4 and 16 days after onset, with 98 and 84 days of lead time.

---

## Adaptive charging

150 simulated overnight sessions across profiles and scenarios, each run on identical safety-gated physics.

| Policy | Capacity loss per session | Annualised | vs legacy | Targets met | Hours above 95 % | Peak temp. | Energy cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legacy (charge to 100 % at once) | 0.0091 % | 3.31 %/yr | n/a | 98.7 % | 5.6 h | 32.3 °C | €0.0076 |
| **MILP optimiser** | 0.0032 % | **1.16 %/yr** | **−65 %** | **100 %** | 0.1 h | 29.5 °C | €0.0053 |
| RL policy (Q-learning) | 0.0046 % | 1.66 %/yr | −50 % | 98.7 % | 0.3 h | 32.4 °C | €0.0061 |

The optimiser wins by spending almost no time near full charge and finishing just before unplug. It also runs about 3 °C cooler and costs less, because it skips unnecessary top-ups and favours cheaper hours.

**RL training:** about 900,000 episodes; 4,286 of 4,320 states visited.

### Fleet-level effect

Matched by usage profile (5 profiles), devices on adaptive charging fade **1.75 %** per 100 days vs **2.25 %** on legacy charging, so they **age 22 % slower**.

---

## TinyML compression

| Stage | Size | Agreement with teacher (on-policy) |
|---|---:|---:|
| Distilled MLP (FP32) | 19,732 B | 74.3 % |
| + DAgger, round 1 | n/a | 74.9 % |
| + DAgger, round 2 | n/a | 74.8 % |
| Structured pruning 50 % (before fine-tune) | n/a | 37.0 % |
| Pruned + fine-tuned (FP32) | 5,780 B | 74.9 % |
| Post-training INT8 (no QAT) | 1,940 B | 73.6 % |
| **Quantisation-aware INT8** | **1,940 B** | **74.1 %** |

The final model is **10× smaller** than the distilled FP32 network and takes about **48 µs** per decision in NumPy.

| ONNX export | Size | Decision agreement with NumPy |
|---|---:|---:|
| FP32 | 6,295 B | 100 % |
| INT8 (QDQ) | 3,142 B | 99.9 % |

### Closed loop (75 sessions): what actually matters

| Policy controlling the charger | Targets met |
|---|---:|
| Legacy | 96.0 % |
| RL teacher | 97.3 % |
| INT8 student | 96.0 % |
| RL teacher + deadline guard | 97.3 % |
| **INT8 student + deadline guard** | **97.3 %** |

About 74 % agreement sounds low, but most disagreements fall between neighbouring charge rates with nearly equal value. With the deadline guard, the 1.9 KB student matches the full teacher's closed-loop result. See [lessons learned](lessons-learned.md) for how this gap was closed.

---

## Agentic workforce (L5 run)

One pass of all six agents at autonomy level 5 over the 240-device in-service fleet:

| Metric | Result |
|---|---:|
| Devices flagged | 123 |
| Actions created | 466 (447 executed, 19 awaiting user consent) |
| Active swelling escalated as a safety issue | **4 / 4** |
| Active micro-short escalated as a safety issue | **1 / 1** |
| Active anode defects identified as manufacturing defects | **8 / 9** |
| Healthy devices wrongly called manufacturing defects | 2 (both with evidence attached for human review) |
| Bad supplier lot L2317 detected | **Yes**: 5 of 24 devices are outliers, p ≈ 5.6 × 10⁻⁴ (next lot: p = 0.086) |

| Action type | Count |
|---|---:|
| OTA recalibration | 101 |
| Diagnostic capture | 87 |
| User notification | 79 |
| Enable adaptive charging | 74 |
| Proactive replacement offer | 62 |
| Warranty claim | 21 |
| Replacement order | 21 |
| Safety derate | 16 |
| Design insight | 3 |
| Logistics shipment | 2 |

The three design insights it produced:

1. The largest ageing driver is cycling throughput.
2. The adaptive-charging cohort fades 22 % slower than legacy.
3. Supplier lot L2317 shows excess anode active-material loss.

---

## The real laptop

ASUS laptop, Li-ion, 63.0 Wh design capacity. Snapshot taken 3 Oct 2026.

| Metric | Value |
|---|---|
| Full-charge capacity | 44.3 Wh (**70.4 %** of design) |
| Cycle count | 384 |
| Runtime on a full charge | about 3.0 h now vs 4.3 h new (77 min lost) |
| Capacity history | 52 entries, Oct 2025 → Oct 2026; 82.3 % at cycle 219 → 70.4 % at cycle 384 |
| Corrupt entries excluded | 1 (592,623 h of standby logged in a single day) |
| Fade rate (GP trend) | 2.6 % of design per 100 days |
| Forecast: 60 % of design | median **27 Oct 2027**; 95 % interval about 7–26 months from now |
| Time on AC (last 45 days) | 92 % |
| Energy drawn from the battery | about 18 Wh/day (41 % of a full charge) |
| Charge limit detected | **Yes, about 60 %** (firmware "battery care" mode) |
| Calibrated ageing factor *k* | 5.5 (the fleet ageing model must run 5.5× faster to reproduce this battery's observed fade) |
| What-if: days to 60 % | current habits **420** · always at 100 % **210** |
| Tonight's plan | no charge needed (60 % covers typical use): **83 % less wear** than charging to 100 % |
| Data-quality insight | full-charge capacity flat for 45 days under the charge limit, which points to a stale fuel gauge |

The calibration factor of 5.5 is itself a finding. Part of it is a real difference between this cell and the modelled one; part is information Windows doesn't expose, such as cell temperature and chemistry, which the projection has to assume. Either way, a fleet model needs per-device calibration before its projections can be trusted, and this one gets it from the battery's own history.
