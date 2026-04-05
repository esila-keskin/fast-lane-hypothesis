import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import copy
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg
from tasks.social_task import make_loaders, build_dataset
from models.ven_circuit import VENCircuit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS_POP = {"pyr": "#4477AA", "ven": "#EE7733"}


def train_one(ven_pct, seed, n_pyr=2000, epochs=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_ven = max(0, int(round(n_pyr * ven_pct / 100)))
    model = VENCircuit(
        n_input=cfg.TASK["n_input"], n_pyramidal=n_pyr, n_ven=n_ven,
        n_classes=cfg.TASK["n_classes"], circuit_cfg=cfg.CIRCUIT,
        neuron_cfg=cfg.NEURON, seed=seed
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    tr, va, _ = make_loaders(cfg.TASK, {**cfg.TRAIN, "num_workers": 0})
    best_val = 0
    best_state = None
    for ep in range(epochs):
        model.train()
        for spk, lbl in tr:
            spk, lbl = spk.to(device), lbl.to(device)
            optimizer.zero_grad()
            logits, _ = model(spk)
            loss = nn.functional.cross_entropy(logits, lbl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        c2 = t2 = 0
        with torch.no_grad():
            for spk, lbl in va:
                spk, lbl = spk.to(device), lbl.to(device)
                logits, _ = model(spk)
                c2 += (logits.argmax(1) == lbl).sum().item()
                t2 += len(lbl)
        val_acc = c2 / t2
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
        scheduler.step()
    model.load_state_dict(best_state)
    print(f"  Trained {ven_pct}% VENs (n_pyr={n_pyr}): best_val={best_val:.4f}")
    return model


def ablate(m):
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        for attr in ["inp_to_ven", "ven_to_out"]:
            layer = getattr(m2, attr, None)
            if layer is not None:
                layer.weight.data.zero_()
                if layer.bias is not None:
                    layer.bias.data.zero_()
    return m2


def run(n_pyr=2000, seed=0, out_path="figures/fig_latency_all_conditions.pdf"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("Training models for latency analysis...")
    m_typical = train_one(2.0, seed=seed, n_pyr=n_pyr)
    m_autism  = train_one(0.4, seed=seed, n_pyr=n_pyr)
    m_ftd     = ablate(m_typical)

    spk_te, lbl_te = build_dataset(cfg.TASK["n_test"], cfg.TASK, seed=9999)
    spk_te = spk_te.to(device)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"Spike Dynamics Across Clinical Conditions (Npyr={n_pyr}, seed={seed})",
                 fontweight="bold", fontsize=13)

    for col, (cname, model) in enumerate([
        ("Typical", m_typical),
        ("Autism-like", m_autism),
        ("FTD-like", m_ftd),
    ]):
        model.eval()
        with torch.no_grad():
            pyr_spk, ven_spk = model.get_internal_spikes(spk_te)

        pyr_np = pyr_spk.cpu().numpy()
        T, B, N_pyr = pyr_np.shape

        ax = axes[0, col]
        ax.plot(range(T), pyr_np.mean(axis=(1, 2)),
                color=COLORS_POP["pyr"], label=f"Pyramidal (N={N_pyr})", lw=2)
        if ven_spk is not None:
            ven_np = ven_spk.cpu().numpy()
            ax.plot(range(T), ven_np.mean(axis=(1, 2)),
                    color=COLORS_POP["ven"], label=f"VEN (N={model.n_ven})", lw=2)
        ax.set_title(cname, fontsize=11)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mean spike rate")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1, col]
        n_trials = min(B, 300)
        pyr_first = []
        for b in range(n_trials):
            for n in range(N_pyr):
                fired = np.where(pyr_np[:, b, n] > 0)[0]
                if len(fired):
                    pyr_first.append(fired[0])

        ax.hist(pyr_first, bins=50, range=(0, T), density=True,
                color=COLORS_POP["pyr"], alpha=0.7, label="Pyramidal")

        if ven_spk is not None and model.n_ven > 0:
            ven_np = ven_spk.cpu().numpy()
            ven_first = []
            for b in range(n_trials):
                for nv in range(model.n_ven):
                    fired = np.where(ven_np[:, b, nv] > 0)[0]
                    if len(fired):
                        ven_first.append(fired[0])
            ax.hist(ven_first, bins=50, range=(0, T), density=True,
                    color=COLORS_POP["ven"], alpha=0.7, label="VEN")
            print(f"  {cname}: pyr median={np.median(pyr_first):.1f}ms  "
                  f"ven median={np.median(ven_first):.1f}ms")
        else:
            print(f"  {cname}: pyr median={np.median(pyr_first):.1f}ms  VENs absent/ablated")

        ax.set_xlabel("First-spike latency (ms)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()