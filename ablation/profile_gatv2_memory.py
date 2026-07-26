"""
Profile GATv2 denoiser GPU memory: grad_checkpoint OFF vs ON.

Answers two questions the CPU estimates cannot:
  1. Where does the retained ~11 GiB actually go?  A forward hook logs
     torch.cuda.memory_allocated() after every block, so you see the
     accumulation build up block-by-block (OFF) vs stay flat (ON).
  2. What is the real peak (fwd+bwd) before/after the fix, and does the
     failed config now fit in 15 GiB?

Runs on a CUDA GPU (Kaggle T4).  On CPU it prints a notice and exits -- the
numbers are only meaningful on the actual device.

Examples:
    python ablation/profile_gatv2_memory.py                 # d=128/h8 and d=512/h4, noattn
    python ablation/profile_gatv2_memory.py --d_model 512 --n_heads 8
    python ablation/profile_gatv2_memory.py --attention     # use_attention=True
    python ablation/profile_gatv2_memory.py --record        # + memory_viz snapshot pickle

The --record snapshots are viewable at https://pytorch.org/memory_viz to see
exactly which tensors dominate.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tab_ddpm.gnn_layers import GATv2Layer, GATV2_PAIR_BUDGET_BYTES
from tab_ddpm.graph_denoiser import GraphAwareDenoiser

# news-like feature graph: 45 numerical + 3 categorical = 48 nodes, d_in = 59.
D_NUM = 45
CAT_SIZES = [5, 5, 4]
D_IN = D_NUM + sum(CAT_SIZES)
N_NODES = D_NUM + len(CAT_SIZES)

DEFAULT_CONFIGS = [
    dict(d_model=128, n_heads=8),   # the config that OOM'd on Kaggle
    dict(d_model=512, n_heads=4),   # the original 18 GiB (untiled) config
]


def human(n_bytes: int) -> str:
    return f"{n_bytes / 1024**3:6.2f} GiB"


def build(d_model, n_heads, n_layers, use_attention, grad_ckpt, device):
    m = GraphAwareDenoiser(
        d_in=D_IN, num_classes=0, is_y_cond=False, d_num=D_NUM, cat_sizes=CAT_SIZES,
        d_model=d_model, n_layers=n_layers, n_heads=n_heads, graph_mode="dynamic",
        top_k=5, gnn_type="gatv2", dropout=0.0, use_attention=use_attention,
    ).to(device).train()
    for layer in m.modules():
        if isinstance(layer, GATv2Layer):
            layer.grad_checkpoint = grad_ckpt
    return m


def run_once(d_model, n_heads, n_layers, use_attention, batch, grad_ckpt, device, record):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    m = build(d_model, n_heads, n_layers, use_attention, grad_ckpt, device)
    x = torch.randn(batch, D_IN, device=device)
    t = torch.randint(0, 1000, (batch,), device=device)

    per_block = []
    handles = [
        b.register_forward_hook(
            lambda mod, inp, out, i=i: per_block.append((i, torch.cuda.memory_allocated()))
        )
        for i, b in enumerate(m.blocks)
    ]

    if record:
        torch.cuda.memory._record_memory_history(max_entries=200_000)

    out = m(x, t)
    out.pow(2).mean().backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()

    for h in handles:
        h.remove()
    if record:
        fn = f"snapshot_d{d_model}_h{n_heads}_ckpt{int(grad_ckpt)}.pickle"
        torch.cuda.memory._dump_snapshot(fn)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"      snapshot -> {fn}  (open at https://pytorch.org/memory_viz)")

    del m, x, t, out
    torch.cuda.empty_cache()
    return peak, per_block


def profile_config(cfg, n_layers, batch, use_attention, device, record):
    dh = cfg["d_model"] // cfg["n_heads"]
    untiled_pair = batch * cfg["n_heads"] * N_NODES * N_NODES * dh * 4
    print("\n" + "=" * 70)
    print(f"  d_model={cfg['d_model']}  n_heads={cfg['n_heads']}  n_layers={n_layers}  "
          f"batch={batch}  attention={use_attention}")
    print(f"  N={N_NODES} nodes | untiled pair/layer would be {human(untiled_pair)} | "
          f"tile budget {human(GATV2_PAIR_BUDGET_BYTES)}")
    print("=" * 70)

    for grad_ckpt in (False, True):
        tag = "grad_checkpoint ON " if grad_ckpt else "grad_checkpoint OFF"
        try:
            peak, per_block = run_once(
                d_model=cfg["d_model"], n_heads=cfg["n_heads"], n_layers=n_layers,
                use_attention=use_attention, batch=batch, grad_ckpt=grad_ckpt,
                device=device, record=record,
            )
            print(f"\n  [{tag}]  PEAK fwd+bwd = {human(peak)}")
            print(f"      retained after each block (forward accumulation):")
            for i, mem in per_block:
                print(f"        block {i}: {human(mem)}")
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  [{tag}]  OOM -> {str(e).splitlines()[0]}")
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=None)
    ap.add_argument("--n_heads", type=int, default=None)
    ap.add_argument("--n_layers", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--attention", action="store_true",
                    help="use_attention=True (default: noattn, matching the failed run)")
    ap.add_argument("--record", action="store_true",
                    help="dump a memory_viz snapshot pickle per run")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA device. Memory numbers are only meaningful on the actual GPU "
              "(e.g. Kaggle T4) -- run this there.")
        return

    torch.cuda.reset_peak_memory_stats()
    props = torch.cuda.get_device_properties(0)
    print(f"{props.name} | total {human(props.total_memory)} | torch {torch.__version__}")

    if args.d_model is not None and args.n_heads is not None:
        configs = [dict(d_model=args.d_model, n_heads=args.n_heads)]
    else:
        configs = DEFAULT_CONFIGS

    for cfg in configs:
        profile_config(cfg, args.n_layers, args.batch, args.attention, "cuda", args.record)


if __name__ == "__main__":
    main()
