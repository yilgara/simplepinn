import torch


class LatinHypercubeSampler:
    def sample(self, domain, vars, n_points):
        coords = {}

        for i, (low, high) in enumerate(domain):
            edges = torch.linspace(0.0, 1.0, n_points + 1)
            points = edges[:-1].reshape(-1, 1)
            points = points + torch.rand(n_points, 1) / n_points
            points = points[torch.randperm(n_points)]
            values = points * (high - low) + low
            values.requires_grad_(True)
            coords[vars[i]] = values

        return coords
