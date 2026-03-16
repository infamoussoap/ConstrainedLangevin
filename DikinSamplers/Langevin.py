import math
import torch
from tqdm import tqdm

from .BaseSampler import BaseSampler


class Langevin(BaseSampler):
    def __init__(self, h, A, b, log_pdf, e=0.0, beta=1.0, x0=None):
        super().__init__(h, A, b, log_pdf, e=e, beta=beta, x0=x0)

    def _log_pdf_and_grad(self, x):
        x_req = x.detach().clone().requires_grad_(True)
        logp = self.log_pdf(x_req)
        (grad,) = torch.autograd.grad(logp, x_req)
        return logp.detach(), grad.detach()

    def _state_stats(self, x):
        x_detached = x.detach().clone()
        logp, grad = self._log_pdf_and_grad(x_detached)
        return {
            "x": x_detached,
            "logp": logp,
            "grad": grad,
        }

    def _proposal_mean(self, state, h):
        return state["x"] + h * state["grad"]

    def _propose(self, current_state, h):
        noise = torch.randn(self.dim, dtype=self.x.dtype, device=self.x.device)
        return self._proposal_mean(current_state, h) + math.sqrt(2.0 * self.beta * h) * noise

    def log_proposal(self, y, state, h):
        mu = self._proposal_mean(state, h)
        diff = y - mu
        return -diff.dot(diff) / (4.0 * self.beta * h)

    def acceptance_ratio(self, proposal_state, current_state, h):
        log_numerator = proposal_state["logp"] + self.log_proposal(current_state["x"], proposal_state, h)
        log_denominator = current_state["logp"] + self.log_proposal(proposal_state["x"], current_state, h)
        log_alpha = log_numerator - log_denominator
        return torch.exp(torch.minimum(log_alpha, self.x.new_tensor(0.0)))

    def sample(self, num_iters):
        return self.run(num_iters)

    def run(self, num_iters):
        history = torch.empty(
            num_iters,
            self.dim,
            dtype=self.x.dtype,
            device=self.x.device,
        )
        num_accept = 0

        current_state = self._state_stats(self.x)
        self.x = current_state["x"].clone()

        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
            h = self.h
            proposal = self._propose(current_state, h)

            if self.is_valid(proposal):
                proposal_state = self._state_stats(proposal)
                alpha = self.acceptance_ratio(proposal_state, current_state, h)
            else:
                proposal_state = None
                alpha = self.x.new_tensor(0.0)

            if torch.rand((), dtype=self.x.dtype, device=self.x.device) < alpha:
                num_accept += 1
                current_state = proposal_state
                self.x = current_state["x"].clone()
            else:
                self.x = current_state["x"]

            history[i] = self.x
            if (i + 1) % 25 == 0 or i + 1 == num_iters:
                pbar.set_postfix_str(f"Acc. Prob {num_accept / (i + 1):.3f}")

        return history.cpu().numpy(), num_accept / num_iters
