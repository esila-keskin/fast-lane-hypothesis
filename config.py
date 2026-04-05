import os
ROOT_DIR = "/kaggle/working/ven_hypothesis"
CKPT_DIR = os.path.join(ROOT_DIR,"outputs","checkpoints")
FIG_DIR  = os.path.join(ROOT_DIR,"outputs","figures")
RES_DIR  = os.path.join(ROOT_DIR,"outputs","results")
for _d in [CKPT_DIR,FIG_DIR,RES_DIR]:
    os.makedirs(_d,exist_ok=True)

TASK = dict(
    n_input=100, n_classes=2, T=50, dt=1.0,
    n_train=4000, n_val=500, n_test=1000, seed=42,
    threatening_hz=(40,90), threatening_burst_prob=0.7,
    friendly_hz=(5,20),     friendly_burst_prob=0.15,
)
NEURON = dict(
    pyramidal_tau=20.0, pyramidal_v_thresh=1.0,
    pyramidal_v_reset=0.0, pyramidal_tau_s=5.0,
    ven_tau=5.0, ven_v_thresh=1.5,
    ven_v_reset=0.0, ven_tau_s=2.0,
)
CIRCUIT = dict(
    n_pyramidal=200,
    pyramidal_fan_in=80,
    ven_fan_in=8,
    recurrent_prob=0.15,
    pyramidal_to_output_prob=0.3,
    ven_to_output_prob=1.0,
    ven_fractions=[0.0,0.005,0.01,0.02,0.03,0.05,0.08,0.10],
    typical_ven_frac=0.02,
    autism_ven_frac=0.004,
    ftd_ven_frac=0.02,
)
TRAIN = dict(
    batch_size=64, lr=1e-3, epochs=30,
    device="cuda", num_workers=2,
    grad_clip=1.0, save_every=10,
)
SAT = dict(decision_threshold=5, max_rt=50)
EVOLUTION = dict(
    complexities=[2,4,6,8,12,16],
    ven_fractions=[0.001,0.005,0.01,0.02,0.03,0.05,0.08,0.10],
)
