import numpy as np
import torch


def project_vectors(v, K, e=1e-6):
    norm = torch.linalg.norm(v, axis=1) + e
    C = torch.min(norm, K * torch.ones_like(norm))
    
    return C[:, None] * v / norm[:, None]


def replace_invalid_particles(particles, valid_samples, replace=True, w=None):
    if w is None:
        w = valid_samples.to(torch.float32).detach().numpy()
        w = w / w.sum()
    
    invalid_samples = ~valid_samples
    if (invalid_samples).all():
        raise ValueError("No valid samples")
        
    num_invalid = invalid_samples.to(int).sum().item()
    if num_invalid == 0:
        return
    
    chosen_indices = np.random.choice(np.arange(len(valid_samples)), size=num_invalid, replace=replace, p=w)
    
    with torch.no_grad():
        particles[invalid_samples] = particles[chosen_indices]
