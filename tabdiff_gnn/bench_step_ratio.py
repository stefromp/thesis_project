#!/usr/bin/env python3
"""
Relative compute cost of each Route-A AttnGNN (d_model 256, real data shapes)
against the GNN reference configuration whose A100 step time was measured
(tabdiff_study/make_jobs.measure: gcn -> adult L2 d128, 14 nodes; gatv2 ->
default L3 d128 h8, 23 nodes). CPU timing of one training step (forward +
backward + AdamW, batch 1024, GATv2 gradient checkpointing as in training) and
one no-grad forward; each configuration is timed right after its reference,
ROUNDS times alternating, and the median ratio is kept (laptop timings drift). The ratios replace make_jobs.py's pessimistic
quadratic-in-nodes x quadratic-in-d_model extrapolation; the A100 itself is
measured by the cluster timing probe (README).

    python tabdiff_gnn/bench_step_ratio.py          # -> hpc_transfer/jobs_tabdiff_gnn/step_ratio_cpu.csv
    python tabdiff_gnn/bench_step_ratio.py --rules  # -> .../step_ratio_rules_cpu.csv: each adaptive
                                                    #    rule vs fixed, same dataset, same protocol
"""
from __future__ import annotations

import csv
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import config as C  # noqa: E402

OUT = os.path.join(C.REPO, "hpc_transfer", "jobs_tabdiff_gnn", "step_ratio_cpu.csv")
OUT_RULES = os.path.join(C.REPO, "hpc_transfer", "jobs_tabdiff_gnn", "step_ratio_rules_cpu.csv")
B, REPS, THREADS, ROUNDS = 1024, 3, 6, 3   # 6 = performance cores of the local M-series CPU
# reference runs' graphs: same node counts and layer/width/heads as the measured runs
REFS = {"gatv2": dict(d_num=14, cats=[2] + [10] * 8, gnn="gatv2", L=3, d=128, k=5, h=8),   # default, 23 nodes
        "gcn":   dict(d_num=6, cats=[2] + [10] * 7, gnn="gcn", L=2, d=128, k=5, h=4)}      # adult, 14 nodes


def bench(d_num, cats, gnn, L, d, k, h, topk_mode="fixed"):
    import torch
    from denoiser import AttnGNNDenoiser
    torch.manual_seed(0)
    extra = {} if topk_mode == "fixed" else {"topk_mode": topk_mode, "k_min": C.RULE_SETTINGS["k_min"],
                                             "topk_warmup_steps": C.RULE_SETTINGS["topk_warmup_steps"]}
    m = AttnGNNDenoiser(d_num, [c + 1 for c in cats], d, L, gnn, h, k, dim_t=C.DIM_T, **extra)
    opt = torch.optim.AdamW(m.parameters(), 1e-3)
    x_num = torch.randn(B, d_num)
    x_cat = torch.cat([torch.nn.functional.one_hot(torch.randint(0, c + 1, (B,)), c + 1).float() for c in cats], 1)
    c = torch.randn(B)

    def step():
        opt.zero_grad()
        a, b = m(x_num, x_cat, c)
        (a.pow(2).mean() + b.pow(2).mean()).backward()
        opt.step()
    m.train()
    step()
    t = time.time()
    for _ in range(REPS):
        step()
    tr = (time.time() - t) / REPS
    m.eval()
    with torch.no_grad():
        m(x_num, x_cat, c)
        t = time.time()
        for _ in range(REPS):
            m(x_num, x_cat, c)
    return tr, (time.time() - t) / REPS


def main_rules() -> int:
    """Each adaptive rule against fixed on the same dataset (d_model 256, real shapes),
    alternating ROUNDS times, median ratio."""
    import statistics
    import torch
    import params
    torch.set_num_threads(THREADS)
    rows = []
    for ds in C.ORDER:
        g, _ = C.gnn_config(ds)
        d_num, cats, _ = params.shapes(C.target_name(ds))
        cfg = (d_num, cats, g["gnn_type"], g["n_layers"], g["d_model"], g["top_k"], g["n_heads"])
        for mode in [m for m in C.TOPK_MODES if m != "fixed"]:
            tr, ev = [], []
            for _ in range(ROUNDS):
                tf, ef = bench(*cfg)
                ta, ea = bench(*cfg, topk_mode=mode)
                tr.append(ta / tf)
                ev.append(ea / ef)
            med = statistics.median
            rows.append({"dataset": ds, "topk_mode": mode, "gnn_type": g["gnn_type"],
                         "nodes": d_num + len(cats), "train_ratio_vs_fixed": round(med(tr), 3),
                         "eval_ratio_vs_fixed": round(med(ev), 3),
                         "train_ratio_spread": f"{min(tr):.2f}-{max(tr):.2f}"})
            print(f"  {ds:9s} {mode:17s} train x{rows[-1]['train_ratio_vs_fixed']:.2f} "
                  f"({rows[-1]['train_ratio_spread']})  forward x{rows[-1]['eval_ratio_vs_fixed']:.2f}", flush=True)
    with open(OUT_RULES, "w", newline="") as fh:
        fh.write(f"# CPU {platform.processor() or platform.machine()}, torch {torch.__version__}, "
                 f"{THREADS} threads, batch {B}, {REPS} reps x {ROUNDS} alternating rounds (median), "
                 f"rule vs fixed on the same dataset\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_RULES}")
    return 0


def main() -> int:
    if "--rules" in sys.argv[1:]:
        return main_rules()
    import torch
    import params
    torch.set_num_threads(THREADS)
    import statistics
    rows = []
    for ds in C.ORDER:
        g, _ = C.gnn_config(ds)
        d_num, cats, _ = params.shapes(C.target_name(ds))
        rcfg = REFS[g["gnn_type"]]
        rr, rt_l, re_l, tr_l, ev_l = [], [], [], [], []
        for _ in range(ROUNDS):
            rt, re_ = bench(**rcfg)
            tr, ev = bench(d_num, cats, g["gnn_type"], g["n_layers"], g["d_model"], g["top_k"], g["n_heads"])
            rt_l.append(rt); re_l.append(re_); tr_l.append(tr); ev_l.append(ev)
        med = statistics.median
        rows.append({"dataset": ds, "gnn_type": g["gnn_type"], "n_layers": g["n_layers"], "d_model": g["d_model"],
                     "nodes": d_num + len(cats), "train_ms": round(med(tr_l) * 1e3, 1),
                     "eval_ms": round(med(ev_l) * 1e3, 1), "ref_train_ms": round(med(rt_l) * 1e3, 1),
                     "ref_eval_ms": round(med(re_l) * 1e3, 1),
                     "train_ratio": round(med([t / r for t, r in zip(tr_l, rt_l)]), 3),
                     "eval_ratio": round(med([e / r for e, r in zip(ev_l, re_l)]), 3),
                     "eval_over_train": round(med([e / t for e, t in zip(ev_l, tr_l)]), 3),
                     "train_ratio_spread": f"{min(t / r for t, r in zip(tr_l, rt_l)):.2f}-"
                                           f"{max(t / r for t, r in zip(tr_l, rt_l)):.2f}"})
        print(f"  {ds:9s} {rows[-1]['nodes']:3d} nodes  train x{rows[-1]['train_ratio']:5.2f} "
              f"({rows[-1]['train_ratio_spread']})  eval x{rows[-1]['eval_ratio']:5.2f}  "
              f"eval/train {rows[-1]['eval_over_train']:.2f}", flush=True)
    with open(OUT, "w", newline="") as fh:
        fh.write(f"# CPU {platform.processor() or platform.machine()}, torch {torch.__version__}, "
                 f"{THREADS} threads, batch {B}, {REPS} reps x {ROUNDS} alternating rounds (median); refs {REFS}\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
