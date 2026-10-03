"""TinyML pipeline: RL teacher -> compact INT8 student for the laptop NPU.

1. Knowledge distillation  - an MLP student learns the Q-learning teacher's
                             action preferences (soft targets, per-state temperature)
2. Structured pruning      - drop whole hidden neurons with the smallest weight
                             norm, then fine-tune (dense, NPU-friendly shapes)
3. Quantization-aware      - fine-tune with fake-quantised INT8 weights
   training (QAT)            (per-output-channel) and activations (per-tensor),
                             straight-through estimator
4. Export                  - ONNX (FP32 and INT8 QDQ). ONNX Runtime executes it;
                             on a device the QNN / OpenVINO / DirectML execution
                             providers offload the graph to the NPU.

Everything is numpy; ONNX export is optional (needs `onnx`, verified with
`onnxruntime` when installed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..charging.rl import ACTIONS, SOH_LEVELS, QPolicy, encode, onpolicy_states

INPUT_NAMES = ["soc", "hours_left_div12", "ambient_minus25_div10", "target_soc", "soh", "elapsed_1h"]


def featurize(soc, hours_left, amb, target, soh, elapsed):
    return np.column_stack(
        [soc, np.asarray(hours_left) / 12.0, (np.asarray(amb) - 25.0) / 10.0, target, soh, np.asarray(elapsed, float)]
    ).astype(np.float32)


def distillation_dataset(q: QPolicy, n: int, seed: int, onpolicy_share: float = 0.75):
    """Mostly on-policy states (what the device will see) plus uniform coverage.

    Returns (features, soft targets, teacher argmax, teacher Q rows, on-policy mask).
    """
    rng = np.random.default_rng(seed)
    n_on = int(n * onpolicy_share)
    on = onpolicy_states(q, n_episodes=max(500, n_on // 12), seed=seed)
    pick = rng.choice(len(on["soc"]), size=n_on, replace=len(on["soc"]) < n_on)
    n_u = n - n_on
    soc = np.r_[on["soc"][pick], rng.uniform(0.0, 1.0, n_u)]
    hours = np.r_[on["hours"][pick], rng.uniform(0.25, 11.0, n_u)]
    amb = np.r_[on["amb"][pick], rng.uniform(8.0, 38.0, n_u)]
    target = np.r_[on["target"][pick], rng.choice([0.8, 0.9, 1.0], n_u)]
    soh = np.r_[on["soh"][pick], rng.uniform(SOH_LEVELS.min(), 1.0, n_u)]
    elapsed = np.r_[on["elapsed"][pick], rng.random(n_u) < 0.6]
    qv = q.q[encode(soc, hours, amb, target, soh, elapsed)]
    span = qv.max(1, keepdims=True) - qv.min(1, keepdims=True)
    tau = np.maximum(0.05 * span, 1e-4)
    soft = np.exp((qv - qv.max(1, keepdims=True)) / tau)
    soft /= soft.sum(1, keepdims=True)
    X = featurize(soc, hours, amb, target, soh, elapsed)
    return X, soft.astype(np.float32), qv.argmax(1), qv, np.arange(len(X)) < n_on


# ----------------------------------------------------------------- the MLP
def _quant_weight(w: np.ndarray):
    """Symmetric per-output-channel INT8 (w: in x out)."""
    scale = np.maximum(np.abs(w).max(axis=0), 1e-8) / 127.0
    return np.clip(np.round(w / scale), -127, 127).astype(np.int8), scale.astype(np.float32)


def _fake_quant_w(w):
    q, s = _quant_weight(w)
    return q.astype(np.float32) * s


def _fake_quant_a(x, amax):
    s = max(amax, 1e-8) / 127.0
    return np.clip(np.round(x / s), -128, 127) * s


@dataclass
class MLP:
    W: list
    b: list
    act_max: list  # calibrated activation ranges (inputs of each layer)

    @classmethod
    def init(cls, sizes, seed=0):
        rng = np.random.default_rng(seed)
        W = [rng.normal(0, np.sqrt(2.0 / a), (a, b)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        return cls(W, [np.zeros(b, np.float32) for b in sizes[1:]], [1.0] * (len(sizes) - 1))

    def forward(self, x, qat=False, cache=False):
        acts, h = [], x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            if qat:
                h = _fake_quant_a(h, self.act_max[i])
                W = _fake_quant_w(W)
            acts.append(h)
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z
        return (h, acts) if cache else h

    def n_params(self) -> int:
        return int(sum(w.size + b.size for w, b in zip(self.W, self.b)))

    def calibrate(self, x):
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            self.act_max[i] = float(np.percentile(np.abs(h), 99.9))
            z = h @ W + b
            h = np.maximum(z, 0.0) if i < len(self.W) - 1 else z


def _train(mlp: MLP, X, Y, epochs, lr, seed, qat=False, masks=None):
    rng = np.random.default_rng(seed)
    m = [np.zeros_like(w) for w in mlp.W] + [np.zeros_like(b) for b in mlp.b]
    v = [np.zeros_like(p) for p in m]
    step = 0
    params = mlp.W + mlp.b
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        for k in range(0, len(X), 512):
            idx = perm[k : k + 512]
            x, y = X[idx], Y[idx]
            logits, acts = mlp.forward(x, qat=qat, cache=True)
            p = np.exp(logits - logits.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            g = (p - y) / len(x)
            grads_W, grads_b = [None] * len(mlp.W), [None] * len(mlp.W)
            for i in reversed(range(len(mlp.W))):
                grads_W[i] = acts[i].T @ g
                grads_b[i] = g.sum(0)
                if i:
                    W = _fake_quant_w(mlp.W[i]) if qat else mlp.W[i]
                    g = (g @ W.T) * (acts[i] > 0)
            step += 1
            for j, gr in enumerate(grads_W + grads_b):
                m[j] = 0.9 * m[j] + 0.1 * gr
                v[j] = 0.999 * v[j] + 0.001 * gr * gr
                mh, vh = m[j] / (1 - 0.9**step), v[j] / (1 - 0.999**step)
                params[j] -= lr * mh / (np.sqrt(vh) + 1e-8)
            if masks is not None:
                for i, mk in enumerate(masks):
                    if mk is not None:
                        mlp.W[i] *= mk
    return mlp


def _prune_structured(mlp: MLP, keep_frac: float) -> MLP:
    """Remove whole hidden neurons (columns of W_i and rows of W_{i+1})."""
    W, b = [w.copy() for w in mlp.W], [x.copy() for x in mlp.b]
    for i in range(len(W) - 1):
        score = np.linalg.norm(W[i], axis=0) * np.linalg.norm(W[i + 1], axis=1)
        keep = np.sort(np.argsort(score)[::-1][: max(4, int(round(keep_frac * len(score))))])
        W[i], b[i] = W[i][:, keep], b[i][keep]
        W[i + 1] = W[i + 1][keep, :]
    return MLP(W, b, list(mlp.act_max))


# --------------------------------------------------------------- INT8 model
@dataclass
class Int8Model:
    Wq: list
    w_scale: list
    b: list
    a_scale: list

    @classmethod
    def from_mlp(cls, mlp: MLP) -> Int8Model:
        Wq, ws = zip(*[_quant_weight(w) for w in mlp.W])
        return cls(
            list(Wq), list(ws), [x.astype(np.float32) for x in mlp.b], [max(a, 1e-8) / 127.0 for a in mlp.act_max]
        )

    def forward(self, x):
        h = np.asarray(x, np.float32)
        for i in range(len(self.Wq)):
            xq = np.clip(np.round(h / self.a_scale[i]), -128, 127).astype(np.int32)  # ONNX int8 range
            acc = xq @ self.Wq[i].astype(np.int32)  # integer MAC
            z = acc.astype(np.float32) * (self.a_scale[i] * self.w_scale[i]) + self.b[i]
            h = np.maximum(z, 0.0) if i < len(self.Wq) - 1 else z
        return h

    def size_bytes(self) -> int:
        return int(
            sum(w.size for w in self.Wq)
            + 4 * sum(s.size for s in self.w_scale)
            + 4 * sum(x.size for x in self.b)
            + 4 * len(self.a_scale)
        )

    def act(self, feats) -> np.ndarray:
        return ACTIONS[np.argmax(self.forward(feats), axis=1)]


# ------------------------------------------------------------------ export
def export_onnx(mlp: MLP, int8: Int8Model, path_fp32, path_int8) -> dict:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return {"exported": False, "reason": "onnx not installed"}

    def graph(quantised: bool):
        nodes, inits = [], []
        cur = "features"
        for i in range(len(mlp.W)):
            if quantised:
                inits += [
                    numpy_helper.from_array(int8.Wq[i], f"W{i}_q"),
                    numpy_helper.from_array(int8.w_scale[i].astype(np.float32), f"W{i}_s"),
                    numpy_helper.from_array(np.zeros(int8.Wq[i].shape[1], np.int8), f"W{i}_zp"),
                    numpy_helper.from_array(np.array(int8.a_scale[i], np.float32), f"A{i}_s"),
                    numpy_helper.from_array(np.array(0, np.int8), f"A{i}_zp"),
                ]
                nodes += [
                    helper.make_node("QuantizeLinear", [cur, f"A{i}_s", f"A{i}_zp"], [f"A{i}_q"]),
                    helper.make_node("DequantizeLinear", [f"A{i}_q", f"A{i}_s", f"A{i}_zp"], [f"A{i}_dq"]),
                    helper.make_node("DequantizeLinear", [f"W{i}_q", f"W{i}_s", f"W{i}_zp"], [f"W{i}"], axis=1),
                ]
                inp = f"A{i}_dq"
            else:
                inits.append(numpy_helper.from_array(mlp.W[i].astype(np.float32), f"W{i}"))
                inp = cur
            inits.append(numpy_helper.from_array(mlp.b[i].astype(np.float32), f"b{i}"))
            nodes += [
                helper.make_node("MatMul", [inp, f"W{i}"], [f"mm{i}"]),
                helper.make_node("Add", [f"mm{i}", f"b{i}"], [f"z{i}"]),
            ]
            if i < len(mlp.W) - 1:
                nodes.append(helper.make_node("Relu", [f"z{i}"], [f"h{i}"]))
                cur = f"h{i}"
            else:
                cur = f"z{i}"
        nodes.append(helper.make_node("Identity", [cur], ["action_logits"]))
        g = helper.make_graph(
            nodes,
            "bms_charge_policy" + ("_int8" if quantised else ""),
            [helper.make_tensor_value_info("features", TensorProto.FLOAT, [None, len(INPUT_NAMES)])],
            [helper.make_tensor_value_info("action_logits", TensorProto.FLOAT, [None, len(ACTIONS)])],
            inits,
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)], producer_name="bms-copilot")
        model.ir_version = 8
        model.doc_string = "inputs: " + ", ".join(INPUT_NAMES) + " | actions (C-rate): " + str(ACTIONS.tolist())
        onnx.checker.check_model(model)
        return model

    info = {"exported": True}
    for path, q in ((path_fp32, False), (path_int8, True)):
        m = graph(q)
        onnx.save(m, str(path))
        info["bytes_" + ("int8" if q else "fp32")] = len(m.SerializeToString())
    return info


def verify_onnx(path_fp32, path_int8, X, mlp: MLP, int8: Int8Model) -> dict:
    try:
        import onnxruntime as ort
    except ImportError:
        return {"verified": False, "reason": "onnxruntime not installed"}
    out = {"verified": True}
    for name, path, ref in (("fp32", path_fp32, mlp.forward(X)), ("int8", path_int8, int8.forward(X))):
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        y = sess.run(None, {"features": X.astype(np.float32)})[0]
        out[name] = {
            "max_abs_diff_vs_numpy": float(np.abs(y - ref).max()),
            "argmax_agreement": float(np.mean(y.argmax(1) == ref.argmax(1))),
        }
    out["execution_providers_available"] = ort.get_available_providers()
    return out


def run_pipeline(q: QPolicy, artifacts_dir, seed: int = 0, rollout_sessions: int = 80) -> tuple[Int8Model, dict]:
    import time

    X, Y, _, _, _ = distillation_dataset(q, 80_000, seed)
    Xv, _, hard_v, qv, onv = distillation_dataset(q, 20_000, seed + 1)
    span = np.maximum(qv.max(1) - qv.min(1), 1e-9)

    def metrics(fn) -> dict:
        act = fn(Xv).argmax(1)
        rows = np.arange(len(act))
        regret = (qv[rows, hard_v] - qv[rows, act]) / span
        return {
            "agreement_onpolicy": float(np.mean(act[onv] == hard_v[onv])),
            "agreement_uniform": float(np.mean(act[~onv] == hard_v[~onv])),
            "normalised_value_regret_onpolicy": float(np.mean(regret[onv])),
        }

    dense = MLP.init([len(INPUT_NAMES), 64, 64, len(ACTIONS)], seed)
    _train(dense, X, Y, epochs=30, lr=3e-3, seed=seed)
    stages = [
        {
            "stage": "distilled MLP (FP32)",
            "params": dense.n_params(),
            "bytes": 4 * dense.n_params(),
            **metrics(dense.forward),
        }
    ]

    # DAgger: label the states the *student* reaches, add them, retrain.
    for rnd in range(2):
        Xd, Yd = _dagger_round(q, dense, seed + 10 + rnd)
        X, Y = np.vstack([X, Xd]), np.vstack([Y, Yd])
        _train(dense, X, Y, epochs=8, lr=1e-3, seed=seed + 20 + rnd)
        stages.append(
            {
                "stage": f"DAgger round {rnd + 1} (FP32)",
                "params": dense.n_params(),
                "added_states": int(len(Xd)),
                **metrics(dense.forward),
            }
        )

    pruned = _prune_structured(dense, keep_frac=0.5)
    stages.append(
        {"stage": "structured pruning 50 % (before fine-tune)", "params": pruned.n_params(), **metrics(pruned.forward)}
    )
    _train(pruned, X, Y, epochs=12, lr=1e-3, seed=seed + 2)
    stages.append(
        {
            "stage": "pruned + fine-tuned (FP32)",
            "params": pruned.n_params(),
            "bytes": 4 * pruned.n_params(),
            **metrics(pruned.forward),
        }
    )

    pruned.calibrate(X)  # activation ranges before QAT
    ptq = Int8Model.from_mlp(pruned)
    stages.append({"stage": "post-training INT8 (no QAT)", "bytes": ptq.size_bytes(), **metrics(ptq.forward)})
    _train(pruned, X, Y, epochs=6, lr=3e-4, seed=seed + 3, qat=True)
    int8 = Int8Model.from_mlp(pruned)
    stages.append({"stage": "QAT INT8", "bytes": int8.size_bytes(), **metrics(int8.forward)})

    t0 = time.perf_counter()
    for _ in range(200):
        int8.forward(Xv[:1])
    latency_us = (time.perf_counter() - t0) / 200 * 1e6

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    p32, p8 = artifacts_dir / "charge_policy_fp32.onnx", artifacts_dir / "charge_policy_int8.onnx"
    exp = export_onnx(pruned, int8, p32, p8)
    ver = verify_onnx(p32, p8, Xv[:2000], pruned, int8) if exp.get("exported") else {}
    report = {
        "teacher": "tabular Q-learning policy (RL)",
        "stages": stages,
        "compression_vs_fp32_distilled": 4 * dense.n_params() / int8.size_bytes(),
        "numpy_int8_latency_us": latency_us,
        "closed_loop": closed_loop_parity(q, int8, rollout_sessions, seed + 7),
        "onnx": {**exp, **ver},
        "inputs": INPUT_NAMES,
        "actions_c_rate": ACTIONS.tolist(),
    }
    return int8, report


def _dagger_round(q: QPolicy, model: MLP, seed: int, n_episodes: int = 3000):
    def act_fn(soc, left, amb, tgt, soh, el):
        return np.argmax(model.forward(featurize(soc, left, amb, tgt, soh, el)), axis=1)

    st = onpolicy_states(q, n_episodes=n_episodes, eps=0.0, seed=seed, act_fn=act_fn)
    qv = q.q[encode(st["soc"], st["hours"], st["amb"], st["target"], st["soh"], st["elapsed"])]
    span = qv.max(1, keepdims=True) - qv.min(1, keepdims=True)
    soft = np.exp((qv - qv.max(1, keepdims=True)) / np.maximum(0.05 * span, 1e-4))
    soft /= soft.sum(1, keepdims=True)
    X = featurize(st["soc"], st["hours"], st["amb"], st["target"], st["soh"], st["elapsed"])
    return X, soft.astype(np.float32)


def int8_session_policy(model: Int8Model, soh: float):
    from ..charging.session import DT_H

    def policy(t, soc, temp, ctx):
        f = featurize(
            [soc], [ctx.n_steps * DT_H - t * DT_H], [ctx.ambient_c], [ctx.target_soc], [soh], [t * DT_H >= 1.0]
        )
        return float(model.act(f)[0])

    return policy


def closed_loop_parity(q: QPolicy, model: Int8Model, n: int, seed: int) -> dict:
    """RL teacher vs INT8 student driving the same sessions through the physics."""
    from ..charging.rl import _sample, level_models
    from ..charging.session import (
        DT_H,
        CellModelSet,
        ChargeContext,
        ChargePhysics,
        legacy_policy,
        rollout,
        with_deadline_guard,
    )

    rng = np.random.default_rng(seed)
    ep = _sample(rng, n)
    models = level_models()
    res = {"legacy": [], "rl_teacher": [], "int8_student": [], "rl_teacher+guard": [], "int8_student+guard": []}
    for i in range(n):
        lvl = int(ep["level"][i])
        ctx = ChargeContext(
            float(ep["soc"][i]),
            float(ep["n"][i] * DT_H),
            float(ep["target"][i]),
            22.0,
            float(ep["amb"][i]),
            models.states[lvl],
        )
        phys = ChargePhysics(CellModelSet([models.states[lvl]]))
        soh = float(models.soh[lvl])
        res["legacy"].append(rollout(ctx, legacy_policy(), phys))
        cap = float(models.capacity[lvl])
        res["rl_teacher"].append(rollout(ctx, q.as_session_policy(soh), phys))
        res["int8_student"].append(rollout(ctx, int8_session_policy(model, soh), phys))
        res["rl_teacher+guard"].append(rollout(ctx, with_deadline_guard(q.as_session_policy(soh), cap), phys))
        res["int8_student+guard"].append(rollout(ctx, with_deadline_guard(int8_session_policy(model, soh), cap), phys))
    out = {
        k: {
            "damage_pct_capacity_mean": float(np.mean([r["damage_pct_capacity"] for r in v])),
            "met_target_rate": float(np.mean([r["met_target"] for r in v])),
        }
        for k, v in res.items()
    }
    out["sessions"] = n
    return out


def _calibrated(mlp: MLP, X) -> MLP:
    m = MLP([w.copy() for w in mlp.W], [b.copy() for b in mlp.b], list(mlp.act_max))
    m.calibrate(X)
    return m


def save_int8(model: Int8Model, path) -> None:
    arrays = {}
    for i in range(len(model.Wq)):
        arrays.update({f"Wq{i}": model.Wq[i], f"ws{i}": model.w_scale[i], f"b{i}": model.b[i]})
    arrays["a_scale"] = np.array(model.a_scale, np.float32)
    np.savez(path, **arrays)


def load_int8(path) -> Int8Model:
    z = np.load(path)
    n = len([k for k in z.files if k.startswith("Wq")])
    return Int8Model(
        [z[f"Wq{i}"] for i in range(n)],
        [z[f"ws{i}"] for i in range(n)],
        [z[f"b{i}"] for i in range(n)],
        z["a_scale"].tolist(),
    )
