"""
run_experiment.py
Reproduces the main clinical conditions experiment from:
"The Fast Lane Hypothesis: Von Economo Neurons Implement a
Biological Speed-Accuracy Tradeoff"
"""
import sys, torch, numpy as np, copy, json
import torch.nn as nn
from scipy import stats
import config as cfg
from tasks.social_task import make_loaders, build_dataset
from models.ven_circuit import VENCircuit

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SEEDS = 10
N_PYR   = 2000
THETA   = 3

def train_model(ven_pct, seed, n_pyr=2000, epochs=30):
    torch.manual_seed(seed); np.random.seed(seed)
    n_ven = max(0, int(round(n_pyr * ven_pct / 100)))
    model = VENCircuit(
        n_input=cfg.TASK["n_input"], n_pyramidal=n_pyr, n_ven=n_ven,
        n_classes=cfg.TASK["n_classes"], circuit_cfg=cfg.CIRCUIT,
        neuron_cfg=cfg.NEURON, seed=seed
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    tr, va, _ = make_loaders(cfg.TASK, {**cfg.TRAIN, "num_workers": 0})
    best_val = 0.0; best_state = None
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
        model.eval(); c2 = t2 = 0
        with torch.no_grad():
            for spk, lbl in va:
                spk, lbl = spk.to(device), lbl.to(device)
                logits, _ = model(spk)
                c2 += (logits.argmax(1)==lbl).sum().item(); t2 += len(lbl)
        val_acc = c2/t2
        if val_acc > best_val:
            best_val = val_acc; best_state = copy.deepcopy(model.state_dict())
        if (ep+1) % 10 == 0:
            print(f"  ep{ep+1:02d} val={val_acc:.4f}", flush=True)
        scheduler.step()
    model.load_state_dict(best_state)
    return model, best_val

def ablate(m):
    m2 = copy.deepcopy(m)
    with torch.no_grad():
        for attr in ["inp_to_ven","ven_to_out"]:
            layer = getattr(m2, attr, None)
            if layer is not None:
                layer.weight.data.zero_()
                if layer.bias is not None: layer.bias.data.zero_()
    return m2

@torch.no_grad()
def get_spikes(model, spikes):
    model.eval(); _, s = model(spikes.to(device)); return s.cpu().numpy()

def rt_fixed(spk, labels, theta):
    T, N, C = spk.shape; cum = np.cumsum(spk, 0)
    rts = np.full(N, float(T)); corr = np.zeros(N)
    for i in range(N):
        for t in range(T):
            if cum[t,i].max() >= theta:
                rts[i]=t+1; corr[i]=int(cum[t,i].argmax()==int(labels[i])); break
        else: corr[i]=int(cum[-1,i].argmax()==int(labels[i]))
    return rts, corr

results = {c: {"rt":[],"acc":[],"val_acc":[]} for c in ["typical","autism","ftd"]}

print(f"Running experiment: Npyr={N_PYR}, {N_SEEDS} seeds, θ={THETA}")
for seed in range(N_SEEDS):
    print(f"\nSeed {seed+1}/{N_SEEDS}", flush=True)
    m_t, va_t = train_model(2.0, seed, N_PYR)
    m_a, va_a = train_model(0.4, seed, N_PYR)
    m_f = ablate(m_t)
    spk_te, lbl_te = build_dataset(cfg.TASK["n_test"], cfg.TASK, seed=seed+9999)
    for cname, model, va in [("typical",m_t,va_t),("autism",m_a,va_a),("ftd",m_f,va_t)]:
        so = get_spikes(model, spk_te)
        rts, corr = rt_fixed(so, lbl_te.numpy(), THETA)
        results[cname]["rt"].append(float(rts.mean()))
        results[cname]["acc"].append(float(corr.mean()))
        results[cname]["val_acc"].append(va)
        print(f"  {cname:<8} val={va:.4f} RT={rts.mean():.2f} acc={corr.mean():.4f}")

print("\n=== RESULTS ===")
for c in ["typical","autism","ftd"]:
    rt  = np.array(results[c]["rt"])
    acc = np.array(results[c]["acc"])
    print(f"  {c}: RT={rt.mean():.2f}±{rt.std():.2f}  acc={acc.mean():.4f}±{acc.std():.4f}")

print("\nPaired t-tests (RT):")
for a, b in [("typical","autism"),("typical","ftd"),("autism","ftd")]:
    t_s, p = stats.ttest_rel(np.array(results[a]["rt"]),np.array(results[b]["rt"]))
    print(f"  {a} vs {b}: t={t_s:.3f}, p={p:.4f} [{'*' if p<0.05 else 'ns'}]")

with open("results_multiseed.json","w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results_multiseed.json")
