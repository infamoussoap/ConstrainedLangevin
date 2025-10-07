import numpy as np
import torch

from tqdm import tqdm


class RandomWalk:
    def __init__(self, h, A, log_pdf, e=0, include_lazification=False):
        self.h = h
        self.log_pdf = log_pdf
        self.A = A
        self.e = e
        self.include_lazification = include_lazification

        self.dim = A.size(1)

        self.x = torch.zeros(self.dim, dtype=torch.float64)
        self.eye = torch.eye(len(self.x), dtype=torch.float64)

    def run(self, num_iters):
        history = torch.zeros(num_iters, self.dim)
        A = self.A
        num_accept = 0
        
        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
            if (np.random.rand() > 0.5) and self.include_lazification:
                history[i] = self.x.detach().clone()
                continue

            with torch.no_grad():
                H = (A[:, :, None] * A[:, None, :]) / ((1 - A @ self.x) ** 2)[:, None, None]
                H = H.sum(axis=0)

                Sigma_inv = (H + self.e * self.eye) / self.h

                eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
                sqrt_Sigma = eig_vecs @ torch.diag(1 / torch.sqrt(eig_vals)) @ eig_vecs.T

                temp_particle = self.x + sqrt_Sigma @ torch.randn_like(self.x)

            a_temp = self.acceptance_ratio(temp_particle, self.x)
            if torch.rand(1) < a_temp:
                num_accept += 1
                with torch.no_grad():
                    self.x[:] = temp_particle

            history[i] = self.x.detach().clone()
            
            pbar.set_postfix_str(f"Acc. Prob {num_accept / (i + 1):.3f}")
            
        return history.detach().clone().numpy(), num_accept / num_iters

    @staticmethod
    def log_proposal(y, x, A, e, h):
        with torch.no_grad():
            H = (A[:, :, None] * A[:, None, :]) / ((1 - A @ x) ** 2)[:, None, None]
            Sigma_inv = (H.sum(axis=0) + e * torch.eye(len(x), dtype=torch.float64)) / h

            eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
            log_det_sigma = torch.log(1 / eig_vals).sum()

            out = -0.5 * (y - x) @ Sigma_inv @ (y - x) - 0.5 * log_det_sigma
        return out

    def acceptance_ratio(self, y, x):
        with torch.no_grad():
            temp = self.A @ y

        if (temp > 1).any():
            return 0

        log_numerator = self.log_pdf(y) + self.log_proposal(x, y, self.A, self.e, self.h)
        log_denominator = self.log_pdf(x) + self.log_proposal(y, x, self.A, self.e, self.h)

        return torch.exp(log_numerator - log_denominator)
