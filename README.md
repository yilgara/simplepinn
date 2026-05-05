# simplepinn

**Physics-Informed Neural Networks for people who don't have a PhD yet.**

`simplepinn` is a beginner-friendly Python framework for building and training
Physics-Informed Neural Networks (PINNs) on top of PyTorch. It is designed for
people who know basic PyTorch and want to solve real physics problems without
writing a full autodiff and training stack from scratch.

PINNs train neural networks to satisfy physical laws. Instead of learning only
from data, the model is penalized when it violates a PDE, boundary condition,
initial condition, or sparse measurement.

## Why simplepinn?

Most PINN libraries are powerful but research-heavy. `simplepinn` aims for a
small, readable API:

```python
import simplepinn as sp

problem = sp.Problem()
problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
problem.add_initial(sp.boundaries.Sinusoidal())

model = sp.PINN(problem)
model.fit(epochs=5000)
model.plot(t=0.5)
```

For the common heat-equation case, you can go even higher level:

```python
import simplepinn as sp

model = sp.solve(equation="heat", alpha=0.01, epochs=2000)
model.plot(t=0.5)
```

## Features

- **Declarative problem API**: define PDEs, boundaries, initials, and data
- **Built-in equations**: heat, wave, Burgers, Poisson, Laplace, advection
- **Custom PDE decorator**: write residual functions with `@sp.equation`
- **Boundary and initial conditions**: Dirichlet, Neumann, function, sinusoidal
- **Data constraints**: fit sparse observations alongside physics residuals
- **Inverse problems**: learn unknown physics parameters with `torch.nn.Parameter`
- **Samplers**: uniform, Latin hypercube, and adaptive API placeholder
- **Loss controls**: manual lambdas and optional `auto_weight=True`
- **Plotting and prediction**: `model.plot(...)`, `model.predict(...)`
- **Interactive GUI demo**: Gradio app for heat/Burgers workflows
- **Pure PyTorch**: models, gradients, parameters, and optimizers stay familiar

## Installation

From source:

```bash
git clone https://github.com/yilgara/simplepinn.git
cd simplepinn
pip install -e .
```

For the GUI demo:

```bash
pip install -e .[gui]
```

For development and tests:

```bash
pip install -e .[dev]
pytest
```

**Requirements:** Python 3.9+, PyTorch 2.0+

`simplepinn` is not published to PyPI yet, so `pip install simplepinn` is a
future release step.

## Quickstart

### Solve the 1D Heat Equation

```python
import torch
import simplepinn as sp

# u_t = 0.01 * u_xx on [0,1] x [0,1]
# u(x, 0) = sin(pi*x), u(0,t) = u(1,t) = 0

problem = sp.Problem(domain=[(0, 1), (0, 1)], vars=["x", "t"])

problem.add_pde(sp.equations.HeatEquation(alpha=0.01))
problem.add_boundary(sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0))
problem.add_initial(sp.boundaries.Function(lambda x: torch.sin(torch.pi * x)))

model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
model.summary()
model.fit(epochs=2000, lr=1e-3)

model.plot(t=0.5)
```

### Define a Custom PDE

```python
import simplepinn as sp


@sp.equation
def my_pde(u, x, t):
    u_t = sp.grad(u, t)
    u_x = sp.grad(u, x)
    u_xx = sp.grad(u, x, order=2)
    return u_t + u * u_x - 0.01 * u_xx


problem = sp.Problem()
problem.add_pde(my_pde)
```

See [docs/custom_equations.md](docs/custom_equations.md).

### Solve an Inverse Problem

Recover an unknown heat coefficient from sparse observations:

```python
import torch
import simplepinn as sp

alpha = torch.nn.Parameter(torch.tensor(0.05))

problem = sp.Problem()
problem.add_pde(sp.equations.HeatEquation(alpha=alpha))
problem.add_boundary(sp.boundaries.Dirichlet(value=0.0))
problem.add_initial(sp.boundaries.Sinusoidal())
problem.add_data(x_measured, t_measured, u_measured)

model = sp.PINN(problem)
model.lambda_data = 10.0
model.fit(epochs=3000)

print(f"Recovered alpha: {alpha.item():.4f}")
```

Run the full example:

```bash
python examples/inverse_heat_alpha.py
```

### Run the GUI

```bash
python gui/app.py
```

Then open:

```text
http://127.0.0.1:7860/
```

The GUI includes equation selection, comma-decimal input support, progress
feedback, solution plots, and loss curves.

## Built-In Equation Library

| Equation | Class | Current scope |
|---|---|---|
| Heat equation | `sp.equations.HeatEquation` | 1D space + time |
| Wave equation | `sp.equations.WaveEquation` | 1D space + time |
| Burgers equation | `sp.equations.BurgersEquation` | 1D space + time |
| Poisson equation | `sp.equations.PoissonEquation` | 1D |
| Laplace equation | `sp.equations.LaplaceEquation` | 1D |
| Advection equation | `sp.equations.AdvectionEquation` | 1D space + time |

## Examples

```text
examples/heat_1d.py
examples/heat_easy.py
examples/solve_heat.py
examples/burgers_1d.py
examples/wave_1d.py
examples/advection_1d.py
examples/poisson_1d.py
examples/laplace_1d.py
examples/inverse_heat_alpha.py
```

## How It Works

A PINN is a neural network `u(x, t; theta)` trained to satisfy:

1. PDE residuals at collocation points
2. boundary conditions
3. initial conditions
4. optional sparse data observations

The total loss is:

```text
L = lambda_pde * L_pde
  + lambda_bc * L_bc
  + lambda_ic * L_ic
  + lambda_data * L_data
```

You can tune lambdas manually:

```python
model.lambda_pde = 1.0
model.lambda_bc = 1.0
model.lambda_ic = 1.0
model.lambda_data = 10.0
```

Or use simple automatic balancing:

```python
model.fit(auto_weight=True)
```

## Project Structure

```text
simplepinn/
├── core/
│   ├── problem.py
│   ├── pinn.py
│   ├── network.py
│   ├── grad.py
│   └── equation.py
├── equations/
│   ├── heat.py
│   ├── wave.py
│   ├── burgers.py
│   ├── poisson.py
│   ├── laplace.py
│   └── advection.py
├── boundaries/
│   ├── dirichlet.py
│   ├── neumann.py
│   ├── function.py
│   └── sinusoidal.py
├── samplers/
│   ├── uniform.py
│   ├── latin_hypercube.py
│   └── adaptive.py
├── viz/
│   └── plot.py
└── presets.py
```

## Known Limitations

- The framework is 1D-first today.
- Poisson and Laplace support basic 1D workflows.
- The adaptive sampler currently preserves the public API but uses uniform
  sampling internally; residual-based refinement is planned.
- The GUI is a demo, not a full production application.
- PyPI publishing is not done yet.
- Multi-device GPU training, DeepONet, VTK/HDF5 export, and 2D/3D domains are
  roadmap items.

## Tests

```bash
pytest
```

The current tests cover:

- public API exports
- built-in equation residuals
- README-style problem setup
- custom `@sp.equation`
- data constraints
- samplers
- prediction
- learnable physics parameters

## Roadmap

- [ ] Publish `simplepinn` to PyPI
- [ ] Residual-based adaptive collocation
- [ ] 2D and 3D spatial domains
- [ ] More inverse-problem examples
- [ ] Neural operator support
- [ ] Export solutions to VTK/HDF5
- [ ] GPU multi-device training
- [ ] More polished GUI workflows

## Contributing

Contributions are welcome, especially new equations, better samplers, examples,
and tutorials. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{simplepinn,
  title  = {simplepinn: Beginner-friendly Physics-Informed Neural Networks in PyTorch},
  author = {Mohammad Asghari and Contribution},
  year   = {2026},
  url    = {https://github.com/yilgara/simplepinn}
}
```

## Acknowledgements

`simplepinn` is inspired by DeepXDE, PINA, and the original PINN paper by Raissi,
Perdikaris, and Karniadakis.

## License

MIT
