import torch
from tqdm import tqdm

from .BaseSampler import BaseSampler


class DikinWalk(BaseSampler):
    """Metropolis Dikin walk with proposal covariance h (H(x) + e I)^{-1}.

    This version inherits the shared geometry utilities from BaseSampler,
    while keeping the fixed-step behaviour of the original Dikin walk.
    """

    def __init__(self, h, A, b, log_pdf, e=0.0, beta=1.0, x0=None):
        super().__init__(h=h, A=A, b=b, log_pdf=log_pdf, e=e, beta=beta, x0=x0)

    def _state_stats(self, x):
        x = torch.as_tensor(x, dtype=self.x.dtype, device=self.x.device)
        metric = self.dikin_hessian(x) + self.e * self.eye
        L_metric = torch.linalg.cholesky(metric)
        log_det_metric = 2.0 * torch.log(torch.diagonal(L_metric)).sum()

        return {
            "x": x.clone(),
            "metric": metric,
            "L_metric": L_metric,
            "log_det_metric": log_det_metric,
            "log_pdf": self.log_pdf(x),
        }

    def _propose(self, current_state, h):
        h_t = torch.as_tensor(h, dtype=self.x.dtype, device=self.x.device)
        z = torch.randn(self.dim, dtype=self.x.dtype, device=self.x.device)
        diffusion = torch.linalg.solve_triangular(
            current_state["L_metric"], z.unsqueeze(-1), upper=False
        ).squeeze(-1)
        return current_state["x"] + torch.sqrt(h_t) * diffusion

    def _log_proposal(self, y, state, h):
        h_t = torch.as_tensor(h, dtype=self.x.dtype, device=self.x.device)
        dx = y - state["x"]
        quad = (dx @ (state["metric"] @ dx)) / h_t
        log_det_sigma = self.dim * torch.log(h_t) - state["log_det_metric"]
        return -0.5 * quad - 0.5 * log_det_sigma

    def acceptance_ratio(self, proposal_state, current_state, h):
        log_alpha = (
            proposal_state["log_pdf"]
            + self._log_proposal(current_state["x"], proposal_state, h)
            - current_state["log_pdf"]
            - self._log_proposal(proposal_state["x"], current_state, h)
        )
        zero = torch.zeros((), dtype=self.x.dtype, device=self.x.device)
        return torch.exp(torch.minimum(log_alpha, zero))

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
        h = self.h

        pbar = tqdm(range(num_iters), total=num_iters, ncols=110, mininterval=0.1)
        for i in pbar:
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
