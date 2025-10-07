import numpy as np
import torch

import sys
import os

from DikinSamplers import RandomWalk, Langevin, DikinLangevin

sampler_name = sys.argv[1].lower()

# Define log pdf
d = 10
A = np.concatenate([np.eye(d), -np.eye(d)], axis=0)
v = 10 ** np.linspace(-2, 0, d)[::-1]
v = np.concatenate([v, v])
A /= v[:, None]

v_tensor = torch.tensor(v[:d], dtype=torch.float64)
x0 = (torch.zeros(d, dtype=torch.float64) + 0.5) * v_tensor

# v_scale = 0.25 * torch.linspace(1, 0.01, d, dtype=torch.float64) ** 1.5
v_scale = 0.5 * v_tensor ** 1.5

log_pdf = lambda x: -0.5 * torch.linalg.norm((x - x0) / v_scale, axis=-1) ** 2


# Run Samplers
num_iters = 100_000
num_indepent_runs = 200

for i in range(num_indepent_runs):
    torch.manual_seed(len(sampler_name) + i)
    print(f"Starting {sampler_name} trial {i}")
    filename = f"metropolis_results/{sampler_name}_trial_{i}.npy"
    accept_filename = f"metropolis_results/{sampler_name}_trial_{i}_accept.npy"
    
    if os.path.exists(filename):
        continue

    if sampler_name == "randomwalk":
        sampler = RandomWalk(0.0045, torch.tensor(A, dtype=torch.float64), log_pdf, e=1e-5)
    elif sampler_name == "langevin":
        sampler = Langevin(4e-7, torch.tensor(A, dtype=torch.float64), log_pdf)
    elif sampler_name == "dikinlangevin":
        sampler = DikinLangevin(0.018, torch.tensor(A, dtype=torch.float64), log_pdf, e=1e-5)
    else:
        raise ValueError(f"Sampler name {sampler_name} is not defined. Try one of randomwalk, langevin, dikinlangevin")

    history, accept_ratio = sampler.run(num_iters)
    np.save(filename, history)
    np.save(accept_filename, accept_ratio)