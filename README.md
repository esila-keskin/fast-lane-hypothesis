# The Fast Lane Hypothesis

**Von Economo Neurons Implement a Biological Speed-Accuracy Tradeoff**

> A computational account of social intuition, autism, and frontotemporal dementia

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://python.org)
[![Framework](https://img.shields.io/badge/framework-SpikingJelly-orange.svg)](https://github.com/fangwei123456/spikingjelly)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Overview

Von Economo neurons (VENs) are large, fast projection neurons found exclusively 
in species with complex social cognition. Their selective depletion in 
frontotemporal dementia (FTD) and altered development in autism implicate them 
in rapid social decision-making, yet no computational model of VEN function 
previously existed.

This repository contains the full implementation of the **Fast Lane Hypothesis**: 
VENs implement a biological speed-accuracy tradeoff by providing a sparse, fast 
projection pathway that enables rapid social decisions.

## Key Results

- All VEN fractions achieve equivalent asymptotic accuracy (~99%), confirming 
  VENs modulate **speed**, not representational capacity
- VENs fire with a **4ms earlier median first-spike latency** than pyramidal neurons
- FTD-like ablation is **consistently the slowest** decision-maker across all 
  decision thresholds (θ = 1–5)
- The developmental (autism-like) vs degenerative (FTD-like) asymmetry emerges 
  naturally from the architecture — without condition-specific tuning

## Architecture
Input (100-dim Poisson spikes)
├── Pyramidal LIF neurons (τ=20ms, fan-in=80, recurrent p=0.15)
└── VEN LIF neurons       (τ=5ms,  fan-in=8,  no recurrence)
└──────────────┬──────────────────┘
Output readout (2 classes)

## Clinical Conditions Modelled

| Condition | VEN fraction | Method |
|-----------|-------------|--------|
| Typical | 2.0% | Normal training |
| Autism-like | 0.4% | Reduced VENs during training |
| FTD-like | 2.0% → 0% | Post-training VEN ablation |

## Installation
```bash
git clone https://github.com/esilakeskin/fast-lane-hypothesis.git
cd fast-lane-hypothesis
pip install -r requirements.txt
```

## Usage
```bash
# Train and evaluate all clinical conditions (10 seeds, Npyr=2000)
python run_experiment.py

# Reproduce individual figures
python analysis/latency_analysis.py
python analysis/threshold_sensitivity.py
```

## Repository Structure
fast-lane-hypothesis/
├── models/
│   └── ven_circuit.py        # Core SNN architecture
├── tasks/
│   └── social_task.py        # Social discrimination task
├── analysis/
│   ├── latency_analysis.py   # First-spike latency (Fig 3)
│   └── threshold_sensitivity.py  # SAT sweep (Fig 4)
├── config.py                 # All hyperparameters
├── run_experiment.py         # Main experiment entry point
├── requirements.txt
└── README.md

## Paper

**The Fast Lane Hypothesis: Von Economo Neurons Implement a Biological 
Speed-Accuracy Tradeoff**  
Esila Keskin, University of the West of England, Bristol (2026)

*Preprint coming soon.*

## Requirements

- Python 3.10+
- PyTorch 2.0+
- SpikingJelly
- NumPy, SciPy, Matplotlib

## License

MIT
