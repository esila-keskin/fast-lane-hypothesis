import torch
import torch.nn as nn
import numpy as np
import spikingjelly.activation_based.surrogate as surrogate
from spikingjelly.activation_based import neuron, functional


class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, fan_in, seed=0, weight_scale=1.0):
        super().__init__()
        rng = np.random.default_rng(seed)
        k = min(fan_in, in_features)
        scale = weight_scale * np.sqrt(2.0 / k)
        weight = np.zeros((out_features, in_features), dtype=np.float32)
        for i in range(out_features):
            idx = rng.choice(in_features, size=k, replace=False)
            weight[i, idx] = rng.normal(0, scale, size=k)
        self.register_buffer("mask", torch.from_numpy((weight != 0).astype(np.float32)))
        self.weight = nn.Parameter(torch.from_numpy(weight))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask, self.bias)


class RecurrentSparseLinear(nn.Module):
    def __init__(self, n_neurons, prob=0.15, seed=0):
        super().__init__()
        rng = np.random.default_rng(seed)
        scale = 0.1 / np.sqrt(n_neurons)
        weight = np.zeros((n_neurons, n_neurons), dtype=np.float32)
        for i in range(n_neurons):
            candidates = [j for j in range(n_neurons) if j != i]
            k = max(1, int(prob * n_neurons))
            idx = rng.choice(candidates, size=min(k, len(candidates)), replace=False)
            weight[i, idx] = rng.normal(0, scale, size=len(idx))
        self.register_buffer("mask", torch.from_numpy((weight != 0).astype(np.float32)))
        self.weight = nn.Parameter(torch.from_numpy(weight))

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.mask)


class VENCircuit(nn.Module):
    """
    Biologically parameterised spiking neural network modelling Von Economo neurons.
    """

    def __init__(self, n_input, n_pyramidal, n_ven, n_classes,
                 circuit_cfg, neuron_cfg, seed=0):
        super().__init__()
        self.n_pyr = n_pyramidal
        self.n_ven = n_ven
        self.n_classes = n_classes
        self.T = 50

        pyr_fan_in = min(circuit_cfg["pyramidal_fan_in"], n_input)
        ven_fan_in = min(circuit_cfg["ven_fan_in"], n_input)

        # Pyramidal population
        self.inp_to_pyr = SparseLinear(
            n_input, n_pyramidal, fan_in=pyr_fan_in, seed=seed, weight_scale=3.0)
        self.pyr_recurrent = RecurrentSparseLinear(
            n_pyramidal, prob=circuit_cfg["recurrent_prob"], seed=seed + 1)
        self.pyr_neurons = neuron.LIFNode(
            tau=neuron_cfg["pyramidal_tau"], v_threshold=0.5, v_reset=0.0,
            surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="s")

        # VEN population
        if n_ven > 0:
            self.inp_to_ven = SparseLinear(
                n_input, n_ven, fan_in=ven_fan_in, seed=seed + 2, weight_scale=3.0)
            self.ven_neurons = neuron.LIFNode(
                tau=neuron_cfg["ven_tau"], v_threshold=0.5, v_reset=0.0,
                surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="s")
            self.ven_to_out = nn.Linear(n_ven, n_classes)
            nn.init.normal_(self.ven_to_out.weight, 0, 0.1 / max(1, n_ven * 0.015))
            nn.init.zeros_(self.ven_to_out.bias)
        else:
            self.inp_to_ven = None
            self.ven_neurons = None
            self.ven_to_out = None

        # Output readout
        self.pyr_to_out = nn.Linear(n_pyramidal, n_classes)
        nn.init.normal_(self.pyr_to_out.weight, 0, 50.0 / n_pyramidal)
        nn.init.zeros_(self.pyr_to_out.bias)

        self.out_neurons = neuron.LIFNode(
            tau=20.0, v_threshold=0.1, v_reset=0.0,
            surrogate_function=surrogate.ATan(), detach_reset=True, step_mode="s")

    def forward(self, x):
        functional.reset_net(self)
        B, T, N = x.shape
        pyr_spk_prev = torch.zeros(B, self.n_pyr, device=x.device)
        spike_record = []
        for t in range(T):
            xt = x[:, t, :]
            pyr_in = self.inp_to_pyr(xt) + self.pyr_recurrent(pyr_spk_prev)
            pyr_spk = self.pyr_neurons(pyr_in)
            pyr_spk_prev = pyr_spk.detach()
            out_in = self.pyr_to_out(pyr_spk)
            if self.n_ven > 0:
                ven_spk = self.ven_neurons(self.inp_to_ven(xt))
                out_in = out_in + self.ven_to_out(ven_spk)
            out_spk = self.out_neurons(out_in)
            spike_record.append(out_spk)
        spike_record = torch.stack(spike_record, dim=0)
        logits = spike_record.sum(dim=0)
        return logits, spike_record

    def get_internal_spikes(self, x):
        """Returns (pyr_spikes, ven_spikes) tensors of shape (T, B, N)."""
        functional.reset_net(self)
        B, T, N = x.shape
        pyr_spk_prev = torch.zeros(B, self.n_pyr, device=x.device)
        pyr_rec, ven_rec = [], []
        for t in range(T):
            xt = x[:, t, :]
            pyr_in = self.inp_to_pyr(xt) + self.pyr_recurrent(pyr_spk_prev)
            pyr_spk = self.pyr_neurons(pyr_in)
            pyr_spk_prev = pyr_spk.detach()
            pyr_rec.append(pyr_spk)
            if self.n_ven > 0:
                ven_spk = self.ven_neurons(self.inp_to_ven(xt))
                ven_rec.append(ven_spk)
        pyr_out = torch.stack(pyr_rec, dim=0)
        ven_out = torch.stack(ven_rec, dim=0) if self.n_ven > 0 else None
        return pyr_out, ven_out