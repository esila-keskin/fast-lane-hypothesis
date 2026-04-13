# The Fast Lane Hypothesis

**Von Economo Neurons Implement a Biological Speed-Accuracy Tradeoff**

> A computational account of social intuition, autism, and frontotemporal dementia

> arxiv preprint: **https://arxiv.org/abs/2604.09229**

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/framework-SpikingJelly-orange.svg)](https://github.com/fangwei123456/spikingjelly)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

Von Economo neurons (VENs) are large, fast projection neurons found exclusively in species with complex social cognition. Their selective depletion in frontotemporal dementia (FTD) and altered development in autism implicate them in rapid social decision-making, yet no computational model of VEN function previously existed.

This repository contains the full implementation of the **Fast Lane Hypothesis**: VENs implement a biological speed-accuracy tradeoff by providing a sparse, fast projection pathway that enables rapid social decisions.

---

## Key Results

| Finding | Result |
|---------|--------|
| Asymptotic accuracy | All VEN fractions achieve ~99.7% - VENs modulate **speed**, not capacity |
| First-spike latency | VENs fire **4ms earlier** than pyramidal neurons (median) |
| FTD vs Typical | FTD significantly slower at θ=1–4 (p < 0.001) |
| Clinical asymmetry | Autism-like vs FTD-like distinction emerges from architecture alone |

---

## Architecture
```
Input (100-dim Poisson spikes)
├── Pyramidal LIF neurons  (τ=20ms, fan-in=80, recurrent p=0.15)
└── VEN LIF neurons        (τ=5ms,  fan-in=8,  no recurrence)
            └──────────────┬──────────────────┘
                   Output readout (2 classes)
```

---

## Clinical Conditions

| Condition | VEN Fraction | Mechanism |
|-----------|-------------|-----------|
| Typical | 2.0% (40 neurons) | Normal training |
| Autism-like | 0.4% (8 neurons) | Reduced VENs from initialisation |
| FTD-like | 2.0% → 0% | Post-training VEN ablation |

The autism-like model compensates during training by routing more information through the pyramidal pathway. The FTD-like model is trained with full VENs then ablated - producing a larger, less-compensable deficit. This developmental vs degenerative asymmetry mirrors the clinical literature precisely, and emerges without any condition-specific tuning.

## Repository Structure
```
fast-lane-hypothesis/
├── models/
│   ├── __init__.py
│   └── ven_circuit.py              # Core SNN architecture (VENCircuit)
├── tasks/
│   ├── __init__.py
│   └── social_task.py              # Social discrimination task
├── analysis/
│   ├── latency_analysis.py         # First-spike latency (Fig 3)
│   └── threshold_sensitivity.py    # SAT sweep, 5 seeds (Fig 4)
├── figures/                        # Generated PDF figures
├── results/                        # JSON results from multi-seed runs
├── paper/
│   └── fast_lane_hypothesis.tex    # Full manuscript (LaTeX)
├── config.py                       # All hyperparameters
├── run_experiment.py               # Main experiment entry point
├── requirements.txt
└── README.md
```

---

## Installation
```bash
git clone https://github.com/esila-keskin/fast-lane-hypothesis.git
cd fast-lane-hypothesis
pip install -r requirements.txt
```

---

## Usage
```bash
# Reproduce the main clinical conditions experiment (10 seeds, Npyr=2000)
python run_experiment.py

# Reproduce Figure 3 - first-spike latency analysis
python analysis/latency_analysis.py

# Reproduce Figure 4 - threshold sensitivity (5 seeds)
python analysis/threshold_sensitivity.py
```

---

## Paper

**The Fast Lane Hypothesis: Von Economo Neurons Implement a Biological Speed-Accuracy Tradeoff**  
Esila Keskin - University of the West of England, Bristol (2026)

*Preprint coming soon.*

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- SpikingJelly ≥ 0.0.0.0.14
- NumPy, SciPy, Matplotlib

---

## License

MIT
