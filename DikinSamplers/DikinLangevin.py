import numpy as np
import torch

from tqdm import tqdm


class DikinLangevin:
    def __init__(self, h, A, log_pdf, e=0, beta=1, include_lazification=False):
        self.h = h
        self.log_pdf = log_pdf
        self.A = A
        self.e = e
        self.beta = beta

        self.dim = A.size(1)
        
        self.x = torch.zeros(self.dim, dtype=torch.float64, requires_grad=True)
            
        self.eye = torch.eye(len(self.x), dtype=torch.float64)
        
        self.include_lazification = include_lazification

    def run(self, num_iters):
        history = torch.zeros(num_iters, self.dim)
        A = self.A
        num_accept = 0
        
        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
            if (np.random.rand() > 0.5) and self.include_lazification:
                history[i] = self.x.detach().clone()
                continue

            val = self.log_pdf(self.x)
            val.backward()

            h = np.random.rand() * self.h
            with torch.no_grad():
                hessian = (A[:, :, None] * A[:, None, :]) / ((1 - A @ self.x) ** 2)[:, None, None]
                Sigma_inv = hessian.sum(axis=0) + self.e * self.eye

                eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
                sqrt_Sigma = eig_vecs @ torch.diag(1 / torch.sqrt(eig_vals)) @ eig_vecs.T
                Sigma = eig_vecs @ torch.diag(1 / eig_vals) @ eig_vecs.T
                
                temp_particle = self.x + h * Sigma @ self.x.grad \
                                + np.sqrt(2 * self.beta * h) * sqrt_Sigma @ torch.randn_like(self.x)

            a_temp = self.acceptance_ratio(temp_particle, self.x, h)
            if torch.rand(1) < a_temp:
                num_accept += 1
                with torch.no_grad():
                    self.x[:] = temp_particle

            history[i] = self.x.detach().clone()

            self.x.grad.zero_()
            pbar.set_postfix_str(f"Acc. Prob {num_accept / (i + 1):.3f}")

        return history.detach().clone().numpy(), num_accept / num_iters

    def log_proposal(self, y, x, A, e, h):
        with torch.no_grad():
            hessian = (A[:, :, None] * A[:, None, :]) / ((1 - A @ x) ** 2)[:, None, None]
            Sigma_inv = hessian.sum(axis=0) + self.e * self.eye

            eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
            log_det_sigma = torch.log(1 / eig_vals).sum()
            Sigma = eig_vecs @ torch.diag(1 / eig_vals) @ eig_vecs.T

        temp_x = x.detach().clone().requires_grad_(True)
        val = self.log_pdf(temp_x)
        val.backward()
        
        mu = temp_x + h * Sigma @ temp_x.grad

        with torch.no_grad():
            out = -0.5 * (y - mu) @ (Sigma_inv / (2 * self.beta * h)) @ (y - mu) - 0.5 * log_det_sigma
        return out

    def acceptance_ratio(self, y, x, h):
        with torch.no_grad():
            temp = self.A @ y

        if (temp > 1).any():
            return 0

        log_numerator = self.log_pdf(y) + self.log_proposal(x, y, self.A, self.e, h)
        log_denominator = self.log_pdf(x) + self.log_proposal(y, x, self.A, self.e, h)

        return torch.exp(log_numerator - log_denominator)
