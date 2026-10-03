# Lessons learned

These are the problems that took the most thought, what I tried and what finally worked. Most of them come down to the same idea: **a metric is only useful if it measures the thing you care about.**

---

## 1. Imitation accuracy isn't closed-loop performance

**Problem.** The first distilled charging policy agreed with its RL teacher on about 74 % of decisions, which looked fine. Put in control of the charger, it met only **62 %** of morning targets.

**Why.** A charging session is a chain of about 30–40 decisions. A student that is slightly too timid early in the night ends up in states the teacher never visits, such as "6 a.m., still at 40 %". It never learned what to do there, so the errors compound.

**What worked.**
- **DAgger**: roll the student out, have the teacher label the states the *student* actually reaches, and retrain on those.
- **A deterministic deadline guard**: a few-line rule that forces just enough current to reach the target in the time left. It is trivially verifiable and catches the rare cases where the network hesitates.

Together they lifted the INT8 student to **97.3 %**, the same as the full teacher.

**Takeaway.** For any policy that acts over time, evaluate in the closed loop. Agreement with the teacher is a debugging signal, not a success metric.

---

## 2. An optimiser's plan is a hypothesis; verify it

**Problem.** The MILP's plans sometimes finished a few percent short of the target.

**Why.** Near full charge a lithium-ion cell enters a constant-voltage phase, where current tapers non-linearly. The MILP can only express this with linear chords, and graphite's staging makes the true curve wavy. When the chords were slightly optimistic, the safety supervisor clamped the current and the plan fell short.

**What worked.** *Plan → verify → re-plan.* Every plan is rolled out on the safety-gated twin. Any shortfall is added to the internal target and the problem is re-solved, or a gentle top-up fills the idle steps before unplug. The result: **100 %** of targets met, with no change to the optimiser's core formulation.

**Takeaway.** A cheap simulation check after a fast approximate solver is often better than an exact but slow formulation.

---

## 3. In a hot room, safety stops the plan unless the plan knows about it

**Problem.** In hot-room scenarios, plans that were fine on paper stalled partway through the night.

**Why.** At a warm ambient temperature, a moderate charge current self-heats the cell past the 45 °C charge limit; a 1.2 C step in a 38 °C room is enough. The safety supervisor, correctly, stopped charging, and the plan never recovered.

**What worked.** A **thermal ceiling** in the MILP: the maximum current at which steady-state I²R heating keeps the cell below the limit, given the forecast ambient temperature. The optimiser now plans within what the supervisor will allow.

**Takeaway.** When a hard safety layer sits downstream of an optimiser, the optimiser needs a model of that layer. Otherwise the two fight each other.

---

## 4. False alarms came from trends, not noise

**Problem.** The first early-warning detector raised a warning on about **21 %** of healthy snapshots, which made it useless.

**Why.** Healthy batteries drift too. Internal resistance rises, strain grows as electrodes swell slightly with age, and cell spread widens. A baseline fitted on young batteries flags every old battery as anomalous.

**What worked.**
- Fitting and **removing each signal's slow ageing trend** before computing the Mahalanobis distance.
- Restricting CUSUM drift detection to signals that should *not* drift in a healthy cell (self-discharge, thermal residual, strain excess).

False warnings fell to **0.2 %**, and every injected swelling and micro-short case was still caught (15 of 15).

A related fix: cell-voltage divergence alone usually means imbalance from ageing, not danger. Routing it to diagnostics instead of a safety escalation removed a class of false safety calls.

**Takeaway.** "Anomalous" has to mean "unexpected for a battery of this age", not "different from a new one".

---

## 5. A simulation that's too clean flatters the model

**Problem.** Early SoH error was **0.10 %**. That was too good to believe.

**Why.** The simulated sensors were perfect apart from random noise. Real current sensors have a fixed gain error per device, which biases every capacity measurement the same way and doesn't average out.

**What worked.** Adding a per-device current-sensor gain error (σ = 0.6 %) and cell-to-cell manufacturing spread. SoH error rose to **0.27 %**. That is a less impressive number and a more honest one, and it is still 20× better than the cycle-counting baseline.

**Takeaway.** If a model looks too accurate on simulated data, the simulator is missing a source of error. Add realism until the results stop surprising you.

---

## 6. Real data is messy in ways simulations aren't

**Problem.** The real-laptop pipeline produced absurd usage statistics, and the health trend went flat for no physical reason.

**Why.**
- Windows had logged one day containing **592,623 hours** of standby time.
- The laptop's firmware had a 60 % charge limit. Fuel gauges re-learn capacity only after near-full charges, so the reported capacity stopped updating; the battery itself had not stopped ageing.

**What worked.** Explicit sanity checks: an entry can't contain more time than the period it covers. The forecast also detects the charge limit and flags the stale gauge as an insight in its own right, which turns out to be a strong argument for sensing health on the device from partial charges.

**Takeaway.** Validate inputs against physical limits before any model sees them, and treat the anomalies you find as findings, not just noise.

---

## 7. Safety by construction beats safety by testing

**Decision.** Rather than trying to prove that the RL policy, the MILP and the agents never request unsafe currents, all of them route through a single deterministic supervisor that:

- clamps every command to the certified envelope (JEITA temperature bands, voltage and current ceilings, CV headroom);
- accepts requests to *tighten* limits and rejects any request to loosen them;
- runs hardware trips independently of any AI state.

A fuzz test throws random requests at it and asserts that the output never leaves the envelope.

**Takeaway.** When AI components can be wrong in unpredictable ways, put the guarantee in a small, simple component you *can* reason about completely, and make it impossible to bypass.

---

## 8. Read the regulation, not the summary

**Problem.** The brief I started from described the EU battery passport as mandatory for "consumer electronics".

**What I found.** Article 77 of Regulation (EU) 2023/1542 mandates the passport from 18 February 2027 only for light-means-of-transport batteries, industrial batteries above 2 kWh and EV batteries. Portable laptop batteries are out of scope.

**What I did.** I built the passport as a **voluntary** record that follows the Annex XIII data categories, and I said so in the code, the UI and the docs. Values only a manufacturer can declare are explicit placeholders rather than invented numbers.

**Takeaway.** For compliance claims, go to the primary source. A confident summary is not evidence.

---

## Smaller lessons

- **Browser caches hide fixes.** A chart fix looked broken in screenshots because the browser kept serving the old script. Adding `Cache-Control: no-cache` to the dashboard's own files means updates always show up, and ETags keep it cheap.
- **Fail soft on missing credentials.** Without an API key, the Anthropic SDK raised an error at the first request rather than at start-up. The copilot now checks for credentials up front and falls back to its offline responder, so the dashboard never breaks for someone who just wants to try it.
- **Vectorise early.** Simulating 440 devices × 3 cells day by day for years is only practical because every physics function accepts arrays. Retrofitting that later would have meant rewriting everything.
- **Hand-rolled is fine when the maths is the point.** Writing the GP, Q-learning and quantisation from scratch cost time, but it made every number explainable and kept the install small.
