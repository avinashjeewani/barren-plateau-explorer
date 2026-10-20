"""Run the experiments defined in a YAML config and write results/gradients.csv.

Each experiment sweeps the number of qubits for several "series" (for example,
local vs. global cost). For every point we draw many random initializations,
compute the exact gradient of the first parameter, and record its mean squared
value E[g^2] with a bootstrap 95% confidence interval.

Why E[g^2] rather than Var[g]: for random (uniform) initialization the mean gradient
is zero, so the two are equal and E[g^2] is the quantity in the barren-plateau
literature. For deterministic starts such as identity blocks, every sample gives the
same gradient, so Var[g] = 0 even though the gradient itself is large. E[g^2] measures
how big the gradient typically is in both cases.

Usage:
    python -m bpx.experiments --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .circuits import gradient_first_param, make_cost, sample_params


def resolve_layers(spec, n_qubits: int) -> int:
    """Layers can be a fixed int or a string like 'n' or '2n' (scaled by qubit count)."""
    if isinstance(spec, int):
        return spec
    spec = str(spec).strip()
    if spec.endswith("n"):
        mult = spec[:-1] or "1"
        return int(float(mult) * n_qubits)
    return int(spec)


def bootstrap_ci(g: np.ndarray, rng, n_boot: int = 300):
    """95% bootstrap CI for E[g^2]."""
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    m = (g[idx] ** 2).mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def measure(n_qubits, n_layers, cost, init, n_samples, seed, sigma=0.1):
    rng = np.random.default_rng(seed)
    identity = init == "identity_block"
    if identity and n_layers % 2:
        n_layers += 1  # identity blocks need an even layer count
    C = make_cost(n_qubits, cost, identity_block=identity)
    params = sample_params(rng, n_samples, n_layers, n_qubits, init, sigma=sigma)
    g = gradient_first_param(C, params)
    lo, hi = bootstrap_ci(g, rng)
    return {
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "cost": cost,
        "init": init,
        "grad_sq_mean": float((g ** 2).mean()),
        "grad_variance": float(g.var(ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "grad_mean": float(g.mean()),
        "n_samples": n_samples,
        "seed": seed,
    }


def run(cfg: dict) -> pd.DataFrame:
    rows = []
    qubits = list(range(cfg["qubits"]["min"], cfg["qubits"]["max"] + 1))
    for exp_name, exp in cfg["experiments"].items():
        for label, s in exp["series"].items():
            for n in qubits:
                t0 = time.perf_counter()
                r = measure(
                    n_qubits=n,
                    n_layers=resolve_layers(s["layers"], n),
                    cost=s["cost"],
                    init=s["init"],
                    n_samples=cfg["n_samples"],
                    seed=cfg["seed"] + 1000 * n,
                    sigma=cfg.get("small_angle_sigma", 0.1),
                )
                r.update(experiment=exp_name, series=label, seconds=time.perf_counter() - t0)
                rows.append(r)
                print(f"{exp_name:6s} {label:32s} n={n:2d} L={r['n_layers']:3d} "
                      f"E[g^2]={r['grad_sq_mean']:.3e} ({r['seconds']:.1f}s)")
    cols = ["experiment", "series", "n_qubits", "n_layers", "cost", "init", "grad_sq_mean", "grad_variance",
            "ci_low", "ci_high", "grad_mean", "n_samples", "seed", "seconds"]
    return pd.DataFrame(rows)[cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out or cfg.get("out_dir", "results"))
    out.mkdir(parents=True, exist_ok=True)
    df = run(cfg)
    df.to_csv(out / "gradients.csv", index=False)

    from .report import write_report

    write_report(df, cfg, out)
    print(f"\nWrote {out}/gradients.csv, {out}/summary.md and {out}/barren_plateaus.png")


if __name__ == "__main__":
    main()
