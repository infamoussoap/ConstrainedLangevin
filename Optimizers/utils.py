import numpy as np
import torch


def to_torch_with_grad(x, dtype=torch.float64, device=None):
    """
    Convert a NumPy array or PyTorch tensor to a leaf PyTorch tensor
    with requires_grad=True.

    Parameters
    ----------
    x : numpy.ndarray or torch.Tensor
        Input array/tensor.
    dtype : torch.dtype, optional
        Output dtype. Defaults to torch.float64.
    device : torch.device or str, optional
        Target device. If None, keeps the tensor's current device when
        possible, otherwise uses the default CPU device.

    Returns
    -------
    torch.Tensor
        A leaf tensor with requires_grad=True.
    """
    if isinstance(x, np.ndarray):
        t = torch.as_tensor(x, dtype=dtype, device=device)
    elif isinstance(x, torch.Tensor):
        if device is None:
            device = x.device
        t = x.to(dtype=dtype, device=device)
    else:
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor")

    # Make sure the output is a leaf tensor with gradients enabled
    return t.detach().clone().requires_grad_(True)
