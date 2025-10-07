import numpy as np
import torch

import sys
import os

from DikinSamplers import RandomWalk, Langevin, DikinLangevin

sampler_name = sys.argv[1].lower()
if len(sys.argv) > 2:
    start_index = int(sys.argv[2])
else:
    start_index = 0

d = 10
A = np.concatenate([np.eye(d), -np.eye(d)], axis=0)

c = 1
x0 = torch.zeros(d, dtype=torch.float64) + 0.5;
x1 = torch.zeros(d, dtype=torch.float64) - 0.5;
v_tensor = torch.ones_like(x0)
log_pdf = lambda x: torch.logaddexp(-2.5 * torch.linalg.norm((x - x0), axis=-1) ** 2,
                                    -2.5 * torch.linalg.norm((x - x1), axis=-1) ** 2)


num_iters = 20_000
tol = 1e-5
num_indepent_runs = 200

for i in range(start_index, num_indepent_runs):
    torch.manual_seed(len(sampler_name) + i)
    print(f"Starting {sampler_name} trial {i}")
    filename = f"new_multimodal_results/{sampler_name}_trial_{i}.npy"
    accept_filename = f"new_multimodal_results/{sampler_name}_trial_{i}_accept.npy"
    
    if os.path.exists(filename):
        continue
    
    if sampler_name == "dikinlangevin":
        sampler = DikinLangevin(0.04, torch.tensor(A, dtype=torch.float64), log_pdf, beta=1, e=1e-5)
    else:
        sampler = RandomWalk(0.022, torch.tensor(A, dtype=torch.float64), log_pdf, e=1e-5)
        
    history, accept_ratio = sampler.run(num_iters)
    
    all_positive = (history > tol).all(axis=-1).astype(int)
    all_negative = (history < -tol).all(axis=-1).astype(int)
    
    labels = all_positive - all_negative
    switches = labels[labels != 0]

    print(f"Iteration {i}: {all_positive.mean()}, {all_negative.mean()}, {np.sum(switches[1:] != switches[:-1])}")
    
    np.save(filename, history)
    np.save(accept_filename, accept_ratio)
