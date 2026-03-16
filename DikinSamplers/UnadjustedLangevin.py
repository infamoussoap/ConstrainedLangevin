import numpy as np
import torch

from tqdm import tqdm

from Optimizers.utils import to_torch_with_grad


class UnadjustedLangevin:
    def __init__(self, h, A, b, log_pdf, beta=1, schedule=None, e=1e-6, include_lazification=False, x0=None):
        # Constraints Ax <= b
        self.h = h
        self.log_pdf = log_pdf

        self.A = A
        self.b = b

        self.beta = beta
        self.e = e

        self.dim = A.size(1)

        if x0 is None:
            self.x = torch.zeros(self.dim, dtype=torch.float64, requires_grad=True)
        else:
            self.x = to_torch_with_grad(x0, dtype=torch.float64, device=None)
            
        self.eye = torch.eye(len(self.x), dtype=torch.float64)
        
        self.include_lazification = include_lazification
        
        if schedule is None:
            self.schedule = lambda x: x
        else:
            self.schedule = schedule

    def run(self, num_iters):
        history = torch.zeros(num_iters, self.dim)
        A = self.A
        num_accept = 0
        
        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
            temp_beta = self.schedule(self.beta)
            if (np.random.rand() > 0.5) and self.include_lazification:
                history[i] = self.x.detach().clone()
                continue

            val = self.log_pdf(self.x)
            val.backward()

            h = np.random.rand() * self.h
            with torch.no_grad():
                temp_particle = self.x + h * self.x.grad + np.sqrt(2 * temp_beta * h) * torch.randn_like(self.x)

            # a_temp = self.acceptance_ratio(temp_particle, self.x, h)
            # if torch.rand(1) < a_temp:
            if self.is_valid(temp_particle.detach()):
                num_accept += 1
                with torch.no_grad():
                    self.x[:] = temp_particle

            history[i] = self.x.detach().clone()

            self.x.grad.zero_()
            pbar.set_postfix_str(f"Acc. Prob {num_accept / (i + 1):.3f} - Val: {val:.3f}")

        return history.detach().clone().numpy(), num_accept / num_iters

    def is_valid(self, x):
        return (self.A @ x <= self.b).all()
