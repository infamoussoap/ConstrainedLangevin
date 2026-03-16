import torch
from tqdm import tqdm


class BaseSampler:
    def __init__(self, h, A, b, log_pdf, e=0.0, beta=1.0, x0=None):
        self.h = h
        self._log_pdf = log_pdf
        self.A = A
        self.b = b
        self.e = e
        self.beta = beta

        self.dim = A.size(1)
        self.x = torch.zeros(self.dim, dtype=A.dtype, device=A.device)
        self.eye = torch.eye(self.dim, dtype=A.dtype, device=A.device)

        if x0 is not None:
            self.x = torch.as_tensor(x0, dtype=A.dtype, device=A.device).clone().requires_grad_(False)

    def log_pdf(self, x):
        return self._log_pdf(x)

    def dikin_hessian(self, x):
        slack = self.b - self.A @ x
        weight = slack.reciprocal().square()
        return self.A.T @ (self.A * weight.unsqueeze(1))

    def is_valid(self, x):
        if x.ndim == 1:
            return (self.A @ x <= self.b).all()
        return ((x @ self.A.T) <= self.b.unsqueeze(0)).all(dim=1)

    def _state_stats(self, x):
        raise NotImplementedError

    def _propose(self, current_state, h):
        raise NotImplementedError

    def acceptance_ratio(self, proposal_state, current_state, h):
        raise NotImplementedError

    def sample(self, num_iters):
        raise NotImplementedError

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
            h = torch.rand((), dtype=self.x.dtype, device=self.x.device).item() * self.h
            proposal = self._propose(current_state, h)

            if self.is_valid(proposal):
                proposal_state = self._state_stats(proposal)
                alpha = self.acceptance_ratio(proposal_state, current_state, h)
            else:
                proposal_state = None
                alpha = self.x.new_tensor(0.0)

            if torch.rand((), device=self.x.device) < alpha:
                num_accept += 1
                current_state = proposal_state
                self.x = current_state["x"].clone()
            else:
                self.x = current_state["x"]

            history[i] = self.x
            if (i + 1) % 25 == 0 or i + 1 == num_iters:
                pbar.set_postfix_str(f"Acc. Prob {num_accept / (i + 1):.3f}")

        return history.cpu().numpy(), num_accept / num_iters
