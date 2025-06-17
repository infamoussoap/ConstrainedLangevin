import numpy as np
import torch


class LangevinSampler:
    def __init__(self, h, A, log_pdf, e=0, include_lazification=True):
        self.h = h
        self.log_pdf = log_pdf
        self.A = A
        self.e = e
        self.include_lazification = include_lazification

        self.dim = A.size(1)

        self.x = torch.zeros(self.dim, dtype=torch.float64, requires_grad=True)
        self.eye = torch.eye(len(self.x), dtype=torch.float64)

    def run(self, num_iters):
        history = torch.zeros(num_iters, self.dim)
        A = self.A
        for i in range(num_iters):
            if (np.random.rand() > 0.5) and self.include_lazification:
                history[i] = self.x.detach().clone()
                continue

            val = self.log_pdf(self.x)
            val.backward()

            with torch.no_grad():
                H = (A[:, :, None] * A[:, None, :]) / ((1 - A @ self.x) ** 2)[:, None, None]
                Sigma_inv = (H.sum(axis=0) + self.e * self.eye) / (2 * self.h)

                eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
                sqrt_Sigma = eig_vecs @ torch.diag(1 / torch.sqrt(eig_vals)) @ eig_vecs.T

                temp_particle = self.x + self.h * self.x.grad \
                                + sqrt_Sigma @ torch.randn_like(self.x)

            a_temp = self.acceptance_ratio(temp_particle, self.x)
            if torch.rand(1) < a_temp:
                with torch.no_grad():
                    self.x[:] = temp_particle

            history[i] = self.x.detach().clone()

            self.x.grad.zero_()

        return history.detach().clone().numpy()

    def log_proposal(self, y, x, A, e, h):
        with torch.no_grad():
            H = (A[:, :, None] * A[:, None, :]) / ((1 - A @ x) ** 2)[:, None, None]
            Sigma_inv = (H.sum(axis=0) + e * torch.eye(len(x), dtype=torch.float64)) / (2 * h)

            eig_vals, eig_vecs = torch.linalg.eigh(Sigma_inv)
            log_det_sigma = torch.log(1 / eig_vals).sum()

        temp_x = x.detach().clone().requires_grad_(True)
        val = self.log_pdf(temp_x)
        val.backward()

        mu = temp_x + h * temp_x.grad

        with torch.no_grad():
            out = -0.5 * (y - mu) @ Sigma_inv @ (y - mu) - 0.5 * log_det_sigma
        return out

    def acceptance_ratio(self, y, x):
        with torch.no_grad():
            temp = self.A @ y

        if (temp > 1).any():
            return 0

        log_numerator = self.log_pdf(y) + self.log_proposal(x, y, self.A, self.e, self.h)
        log_denominator = self.log_pdf(x) + self.log_proposal(y, x, self.A, self.e, self.h)

        return torch.exp(log_numerator - log_denominator)