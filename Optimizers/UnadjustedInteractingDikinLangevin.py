import math
import numpy as np
import torch

from tqdm import tqdm


class UnadjustedInteractingDikinLangevin:
    def __init__(
        self,
        h,
        A,
        b,
        log_pdf,
        beta=1,
        schedule=None,
        e=1e-6,
        include_lazification=False,
        x0=None,
        num_chains=1,
        resample_iters=1000,
    ):
        # Constraints Ax <= b
        self.h = h
        self.log_pdf = log_pdf
        self.beta = beta
        self.e = e
        self.include_lazification = include_lazification
        self.num_chains = num_chains
        self.resample_iters = resample_iters

        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.b = torch.as_tensor(b, dtype=torch.float64, device=self.A.device)

        self.dim = self.A.size(1)
        self.device = self.A.device
        self.dtype = self.A.dtype

        if x0 is None:
            x_init = torch.zeros(self.dim, dtype=self.dtype, device=self.device)
        else:
            x_init = torch.as_tensor(x0, dtype=self.dtype, device=self.device)

        if x_init.ndim == 1:
            x_init = x_init.unsqueeze(0).repeat(self.num_chains, 1)
        elif x_init.ndim == 2 and x_init.shape == (self.num_chains, self.dim):
            x_init = x_init.clone()
        else:
            raise ValueError(
                f"x0 must have shape ({self.dim},) or ({self.num_chains}, {self.dim})"
            )

        self.x = x_init.detach().clone().requires_grad_(True)
        self.eye = torch.eye(self.dim, dtype=self.dtype, device=self.device)

        if schedule is None:
            self.schedule = lambda x: x
        else:
            self.schedule = schedule

    def _evaluate_log_pdf(self, x):
        try:
            values = self.log_pdf(x)
            if not torch.is_tensor(values):
                values = torch.as_tensor(values, dtype=x.dtype, device=x.device)
            else:
                values = values.to(dtype=x.dtype, device=x.device)

            if values.ndim == 0:
                per_chain = values.expand(x.shape[0])
                total = values * x.shape[0]
            elif values.ndim == 1 and values.shape[0] == x.shape[0]:
                per_chain = values
                total = values.sum()
            else:
                raise ValueError("Batched log_pdf returned an unexpected shape.")
        except Exception:
            per_chain = torch.stack([
                self.log_pdf(xi).reshape(()) if torch.is_tensor(self.log_pdf(xi))
                else torch.as_tensor(self.log_pdf(xi), dtype=x.dtype, device=x.device).reshape(())
                for xi in x
            ])
            total = per_chain.sum()

        return total, per_chain

    def _legacy_resample_chains(self):
        with torch.no_grad():
            _, per_chain_val = self._evaluate_log_pdf(self.x.detach())
            weights = torch.softmax(per_chain_val, dim=0)
            indices = torch.multinomial(weights, num_samples=self.num_chains, replacement=True)
            self.x = self.x[indices].detach().clone().requires_grad_(True)

    def _resample_chains(self):
        with torch.no_grad():
            _, per_chain_val = self._evaluate_log_pdf(self.x.detach())
            weights = torch.softmax(per_chain_val, dim=0)

            best_idx = torch.argmax(per_chain_val)
            if self.num_chains == 1:
                indices = best_idx.unsqueeze(0)
            else:
                sampled_idx = torch.multinomial(
                    weights, num_samples=self.num_chains - 1, replacement=True
                )
                indices = torch.cat((best_idx.unsqueeze(0), sampled_idx), dim=0)

            self.x = self.x[indices].detach().clone().requires_grad_(True)

    def run(self, num_iters):
        history = torch.zeros(
            num_iters,
            self.num_chains,
            self.dim,
            dtype=self.x.dtype,
            device=self.x.device,
        )
        A = self.A
        num_accept = 0
        num_attempt = 0

        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
            temp_beta = self.schedule(self.beta)

            if self.include_lazification:
                active_mask = torch.rand(self.num_chains, device=self.x.device) <= 0.5
            else:
                active_mask = torch.ones(
                    self.num_chains, dtype=torch.bool, device=self.x.device
                )

            if active_mask.any():
                total_val, per_chain_val = self._evaluate_log_pdf(self.x)
                total_val.backward()

                h = self.h
                with torch.no_grad():
                    s = self.b.unsqueeze(0) - self.x @ A.T
                    w = 1.0 / s.square()

                    Sigma_inv = torch.einsum("ki,ck,kj->cij", A, w, A)
                    Sigma_inv = Sigma_inv + self.e * self.eye.unsqueeze(0)

                    L = torch.linalg.cholesky(Sigma_inv)
                    Sigma = torch.cholesky_inverse(L)

                    # div C(x) = -2 C(x) sum_i ((a_i^T C(x) a_i) / (b_i - a_i^T x)^3) a_i
                    AC = torch.einsum("ki,cij->ckj", A, Sigma)
                    quad = (AC * A.unsqueeze(0)).sum(dim=2)
                    alpha = quad / s.pow(3)
                    div_C = -2.0 * torch.einsum("cij,cj->ci", Sigma, alpha @ A)

                    grad_term = torch.einsum("cij,cj->ci", Sigma, self.x.grad)
                    drift = h * (grad_term + temp_beta * div_C)

                    z = torch.randn_like(self.x)
                    diffusion = math.sqrt(2 * temp_beta * h) * torch.linalg.solve_triangular(
                        L, z.unsqueeze(-1), upper=False
                    ).squeeze(-1)

                    temp_particle = self.x + drift + diffusion

                    valid_mask = self.is_valid(temp_particle.detach())
                    accept_mask = active_mask & valid_mask

                    num_accept += int(accept_mask.sum().item())
                    num_attempt += int(active_mask.sum().item())

                    self.x[accept_mask] = temp_particle[accept_mask]

                if self.x.grad is not None:
                    self.x.grad.zero_()
            else:
                _, per_chain_val = self._evaluate_log_pdf(self.x.detach())

            if (i + 1) % self.resample_iters == 0:
                self._resample_chains()

            history[i] = self.x.detach().clone()

            acc_prob = num_accept / max(num_attempt, 1)
            if (i + 1) % 25 == 0 or i + 1 == num_iters:
                pbar.set_postfix_str(
                    f"Acc. Prob {acc_prob:.3f} - Mean Val: {per_chain_val.mean().item():.3f}"
                )

        return history.detach().clone().cpu().numpy(), num_accept / max(num_attempt, 1)

    def is_valid(self, x):
        if x.ndim == 1:
            return (self.A @ x <= self.b).all()
        return ((x @ self.A.T) <= self.b.unsqueeze(0)).all(dim=1)
