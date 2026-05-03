# simplepinn

**Physics-Informed Neural Networks for people who don't have a PhD (yet).**

`simplepinn` is a beginner-friendly Python toolkit for building and training Physics-Informed Neural Networks (PINNs) on top of PyTorch. Most PINN libraries are written by researchers, for researchers. `simplepinn` is written for anyone who knows basic PyTorch and wants to solve real physics problems with neural networks — without reading three papers first.

---

## Why simplepinn?

Standard neural networks learn purely from data. PINNs go further — they bake physical laws (PDEs, boundary conditions, conservation equations) directly into the training process, so the model is *penalised* for violating physics. This means they can solve problems with very sparse data, which is a big deal in science and engineering.

The catch: existing PINN libraries are powerful but painful to use. `simplepinn` fixes that.

```python
# Existing libraries: ~80 lines of boilerplate
# simplepinn: this
import simplepinn as sp

problem = sp.Problem()
problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
problem.add_initial(sp.boundaries.Sinusoidal())

model = sp.PINN(problem)
model.fit(epochs=5000)
model.plot()
```

---

## Features

- **Declarative PDE definition** — describe your physics, not your autodiff graph
- **Built-in equation library** — heat, wave, Burgers', Poisson, and more out of the box
- **Smart collocation sampling** — uniform, Latin hypercube, and adaptive point sampling
- **Auto-weighted loss** — handles the `λ_pde · L_pde + λ_bc · L_bc` balancing automatically (or you tune it manually)
- **One-line visualisation** — `model.plot()` generates publication-quality solution plots
- **Pure PyTorch** — no new paradigms, no magic, plays well with everything you already know

---

## Installation

```bash
pip install simplepinn
```

Or from source:

```bash
git clone https://github.com/yourusername/simplepinn.git
cd simplepinn
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+

---

## Quickstart

### Solve the 1D heat equation

```python
import simplepinn as sp

# u_t = 0.01 * u_xx  on [0,1] x [0,1]
# u(x, 0) = sin(πx),  u(0,t) = u(1,t) = 0

problem = sp.Problem(domain=[(0, 1), (0, 1)], vars=["x", "t"])

problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
problem.add_boundary(sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0))
problem.add_initial(sp.boundaries.Function(lambda x: torch.sin(torch.pi * x)))

model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
model.fit(epochs=10000, lr=1e-3)

model.plot(t=0.5)  # plot solution at t=0.5
```

### Define a custom PDE

```python
@sp.equation
def my_pde(u, x, t):
    # Burgers' equation: u_t + u*u_x = 0.01 * u_xx
    u_t = sp.grad(u, t)
    u_x = sp.grad(u, x)
    u_xx = sp.grad(u, x, order=2)
    return u_t + u * u_x - 0.01 * u_xx

problem.add_pde(my_pde)
```

### Solve an inverse problem (recover unknown parameters)

```python
# You have noisy measurements but don't know alpha
alpha = torch.nn.Parameter(torch.tensor(0.001))  # learnable!

problem.add_pde(sp.equations.HeatEquation(alpha=alpha))
problem.add_data(x_measured, t_measured, u_measured)

model = sp.PINN(problem)
model.fit(epochs=10000)

print(f"Recovered alpha: {alpha.item():.4f}")  # should be ~0.01
```

---

## Built-in equation library

| Equation | Class | Domain |
|---|---|---|
| Heat equation | `sp.equations.HeatEquation` | Heat transfer |
| Wave equation | `sp.equations.WaveEquation` | Acoustics, elasticity |
| Burgers' equation | `sp.equations.BurgersEquation` | Fluid dynamics |
| Poisson equation | `sp.equations.PoissonEquation` | Electrostatics, gravity |
| Laplace equation | `sp.equations.LaplaceEquation` | Steady-state heat, potential flow |
| Advection equation | `sp.equations.AdvectionEquation` | Transport problems |

Custom equations are first-class citizens — see the [custom equations guide](docs/custom_equations.md).

---

## How it works

A PINN is a neural network `u(x, t; θ)` trained to satisfy:

1. **PDE residual** — the equation holds at randomly sampled interior points
2. **Boundary conditions** — the solution matches known values on the boundary
3. **Initial conditions** — the solution matches the known state at `t=0`
4. **Data** (optional) — the solution matches any sparse measurements you have

The total loss is:

```
L = λ_pde · L_pde + λ_bc · L_bc + λ_ic · L_ic + λ_data · L_data
```

`simplepinn` handles the automatic differentiation, collocation point sampling, and loss balancing so you can focus on the physics.

---

## Project structure

```
simplepinn/
├── core/
│   ├── problem.py       # Problem definition API
│   ├── pinn.py          # PINN model + training loop
│   ├── grad.py          # Autodiff utilities (sp.grad)
│   └── sampler.py       # Collocation point samplers
├── equations/
│   ├── heat.py
│   ├── wave.py
│   ├── burgers.py
│   └── ...
├── boundaries/
│   ├── dirichlet.py
│   ├── neumann.py
│   └── ...
└── viz/
    └── plot.py          # Visualisation utilities
```

---

## Roadmap

- [ ] 2D and 3D spatial domains
- [ ] Adaptive collocation point refinement
- [ ] Neural operator support (DeepONet)
- [ ] Export solutions to standard formats (VTK, HDF5)
- [ ] GPU multi-device training
- [ ] Interactive solution explorer (Gradio)

---

## Contributing

Contributions are very welcome — especially new equations, better sampling strategies, and tutorials. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## Citation

If you use `simplepinn` in your research, please cite:

```bibtex
@software{simplepinn,
  title  = {simplepinn: Beginner-friendly Physics-Informed Neural Networks in PyTorch},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/yourusername/simplepinn}
}
```

---

## Acknowledgements

`simplepinn` is inspired by [DeepXDE](https://github.com/lululululucky/deepxde), [PINA](https://github.com/mathLab/PINA), and the original PINN paper by Raissi, Perdikaris & Karniadakis (2019).

---

## License

MIT
