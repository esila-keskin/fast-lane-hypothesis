import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import copy
import json
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import config as cfg
from tasks.social_task import make_loaders, build_dataset
from models.ven_circuit import VENCircuit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = {"typical": "#3A7FC1", "autism": "#52C869", "ftd": "#E05252"}
COND_LABELS = {"typical": "Typical (2%)", "autism": "Autism-like (0.4%)", "ftd": "FTD (ablated)"}
N_SEEDS = 5
N_PYR = 2000
THETAS = [1, 2, 3, 4, 5]


def train_model(ven_pct, seed, n_pyr=2000, epochs=30):
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


@torch.no_grad()
def get_spikes(model, spikes):
    model.eval()
    _, s = model(spikes.to(device))
    return s.cpu().numpy()


def rt_fixed(spk, labels, theta):
    T, N, C = spk.shape
    cum = np.cumsum(spk, 0)
    rts = np.full(N, float(T))
    corr = np.zeros(N)
    for i in range(N):
        for t in range(T):
            if cum[t, i].max() >= theta:
                rts[i] = t + 1
                corr[i] = int(cum[t, i].argmax() == int(labels[i]))
                break
        else:
            corr[i] = int(cum[-1, i].argmax() == int(labels[i]))
    return rts, corr


def run(out_dir="figures", results_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    theta_res = {c: {th: {"rt": [], "acc": []} for th in THETAS}
                 for c in ["typical", "autism", "ftd"]}

    for seed in range(N_SEEDS):
        print(f"\nSeed {seed + 1}/{N_SEEDS}:", flush=True)
        m_t = train_model(2.0, seed, N_PYR)
        m_a = train_model(0.4, seed, N_PYR)
        m_f = ablate(m_t)
        spk_te, lbl_te = build_dataset(cfg.TASK["n_test"], cfg.TASK, seed=seed + 9999)
        lbl_np = lbl_te.numpy()
        for cname, model in [("typical", m_t), ("autism", m_a), ("ftd", m_f)]:
            so = get_spikes(model, spk_te)
            for th in THETAS:
                rts, corr = rt_fixed(so, lbl_np, th)
                theta_res[cname][th]["rt"].append(float(rts.mean()))
                theta_res[cname][th]["acc"].append(float(corr.mean()))
        print(f"  seed {seed + 1} done", flush=True)

    print("\n" + "=" * 60)
    print("THRESHOLD SENSITIVITY SUMMARY (mean ± SD)")
    print("=" * 60)
    for th in THETAS:
        print(f"\nθ={th}:")
        for c in ["typical", "autism", "ftd"]:
            rt  = np.array(theta_res[c][th]["rt"])
            acc = np.array(theta_res[c][th]["acc"])
            print(f"  {c:<10} RT={rt.mean():.2f}±{rt.std():.2f}  acc={acc.mean():.4f}±{acc.std():.4f}")

    print("\nPaired t-tests on RT at each threshold:")
    for th in THETAS:
        print(f"\n  θ={th}:")
        for a, b in [("typical", "autism"), ("typical", "ftd"), ("autism", "ftd")]:
            t_s, p = stats.ttest_rel(
                np.array(theta_res[a][th]["rt"]),
                np.array(theta_res[b][th]["rt"])
            )
            sig = "*" if p < 0.05 else "ns"
            print(f"    {a} vs {b}: t={t_s:.3f}, p={p:.4f} [{sig}]")

    json_path = os.path.join(results_dir, "theta_results.json")
    with open(json_path, "w") as f:
        json.dump(theta_res, f, indent=2)
    print(f"\nSaved: {json_path}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, ylabel, scale in zip(
        axes, ["rt", "acc"],
        ["Mean RT (ms)", "Threshold-crossing accuracy (%)"],
        [1, 100]
    ):
        for cname in ["typical", "autism", "ftd"]:
            means = [np.mean(theta_res[cname][th][metric]) * scale for th in THETAS]
            sds   = [np.std( theta_res[cname][th][metric]) * scale for th in THETAS]
            ax.plot(THETAS, means, "o-", color=COLORS[cname],
                    label=COND_LABELS[cname], lw=2.5, markersize=7)
            ax.fill_between(THETAS,
                            [m - s for m, s in zip(means, sds)],
                            [m + s for m, s in zip(means, sds)],
                            color=COLORS[cname], alpha=0.15)
        ax.set_xlabel("Decision threshold (θ)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.35)
        ax.set_xticks(THETAS)

    fig.suptitle(f"Threshold Sensitivity — Npyr={N_PYR}, {N_SEEDS} seeds (mean ± SD)",
                 fontweight="bold")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, "fig_threshold_sensitivity.pdf")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    run()