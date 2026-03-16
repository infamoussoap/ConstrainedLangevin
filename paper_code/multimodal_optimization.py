import numpy as np
import torch
import matplotlib.pyplot as plt

import sys
import os

sys.path.append("..")
from Optimizers import InteractingDikinLangevin, InteractingLangevin 


class Schedule:
    def __init__(self, dt, t0=3):
        self.dt = dt
        self.iterations = 0
        self.t0 = t0
    
    def __call__(self, beta):
        self.iterations += 1
        if self.iterations < 35_000:
            return beta / np.log(self.t0 + self.dt * self.iterations)
        elif self.iterations < 39_000:
            return beta / np.log(self.t0 + self.dt * self.iterations) ** 4
        return 0
    
d = 10
A = np.concatenate([np.eye(d), -np.eye(d)], axis=0)
A_torch = torch.tensor(A, dtype=torch.float64)
b_torch = torch.ones(len(A), dtype=torch.float64)

log_pdf = lambda x: (torch.cos(6 * np.pi * x ** 2) - x ** 2).sum(axis=-1)


# Run Optimization
num_iters = 40_000
num_indepent_runs = 100
dt = 0.01
num_chains = 5

for i in range(num_indepent_runs, 1, -1):
    torch.manual_seed(i)
    print(f"Starting trial {i}")
    sampler_name = "InteractingDikinLangevin_NoResample"
    filename = f"optimization_results/{sampler_name}_chains_{num_chains}_trial_{i}.npy"
    accept_filename = f"optimization_results/{sampler_name}_chains_{num_chains}_trial_{i}_accept.npy"
    
    if os.path.exists(filename):
        continue
    
    x0 = torch.sign(torch.rand(10) - 0.5).to(torch.float64) * 0.8156;
    sampler = InteractingDikinLangevin(dt, A_torch, b_torch, log_pdf, beta=2,
                                       schedule=Schedule(2*dt), e=1e-6, x0=x0, num_chains=num_chains, 
                                       resample_iters=100_000)

    history, accept_ratio = sampler.run(num_iters)
    np.save(filename, history)
    np.save(accept_filename, accept_ratio)
