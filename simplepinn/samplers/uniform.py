import torch


class UniformSampler:
    def sample(self, domain, vars, n_points):
        coords = {}

        for i, (low, high) in enumerate(domain):
            values = torch.rand(n_points, 1) * (high - low) + low
            values.requires_grad_(True)
            coords[vars[i]] = values

        return coords
