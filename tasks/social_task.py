import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def _poisson_burst_spikes(n_neurons,T,base_hz,burst_prob,dt_ms=1.0,rng=None):
    if rng is None: rng=np.random.default_rng()
    dt_s=dt_ms*1e-3
    spikes=np.zeros((T,n_neurons),dtype=np.float32)
    burst_active=False; burst_countdown=0
    for t in range(T):
        if not burst_active and rng.random()<burst_prob*dt_s*10:
            burst_active=True; burst_countdown=int(rng.integers(3,10))
        rate=base_hz*(5.0 if burst_active else 1.0)
        p=rate*dt_s
        spikes[t]=(rng.random(n_neurons)<p).astype(np.float32)
        if burst_active:
            burst_countdown-=1
            if burst_countdown<=0: burst_active=False
    return spikes

def generate_stimulus(label,n_input,T,task_cfg,rng):
    if label==0:
        hz_lo,hz_hi=task_cfg["threatening_hz"]; burst_prob=task_cfg["threatening_burst_prob"]
    else:
        hz_lo,hz_hi=task_cfg["friendly_hz"]; burst_prob=task_cfg["friendly_burst_prob"]
    hz=float(rng.uniform(hz_lo,hz_hi))
    return _poisson_burst_spikes(n_input,T,hz,burst_prob,task_cfg["dt"],rng)

def build_dataset(n_samples,task_cfg,seed=42):
    rng=np.random.default_rng(seed)
    n_input=task_cfg["n_input"]; T=task_cfg["T"]
    all_spikes,all_labels=[],[]
    for i in range(n_samples):
        label=i%task_cfg["n_classes"]
        sp=generate_stimulus(label,n_input,T,task_cfg,rng)
        all_spikes.append(sp); all_labels.append(label)
    spikes=torch.from_numpy(np.stack(all_spikes,axis=0))
    labels=torch.tensor(all_labels,dtype=torch.long)
    return spikes,labels

def make_loaders(task_cfg,train_cfg):
    spikes_train,labels_train=build_dataset(task_cfg["n_train"],task_cfg,seed=task_cfg["seed"])
    spikes_val,  labels_val  =build_dataset(task_cfg["n_val"],  task_cfg,seed=task_cfg["seed"]+1000)
    spikes_test, labels_test =build_dataset(task_cfg["n_test"], task_cfg,seed=task_cfg["seed"]+2000)
    bs=train_cfg["batch_size"]; nw=train_cfg["num_workers"]
    train_loader=DataLoader(TensorDataset(spikes_train,labels_train),
                            batch_size=bs,shuffle=True,num_workers=nw,
                            pin_memory=True,drop_last=True)
    val_loader  =DataLoader(TensorDataset(spikes_val,  labels_val),
                            batch_size=bs,shuffle=False,num_workers=nw)
    test_loader =DataLoader(TensorDataset(spikes_test, labels_test),
                            batch_size=bs,shuffle=False,num_workers=nw)
    return train_loader,val_loader,test_loader
