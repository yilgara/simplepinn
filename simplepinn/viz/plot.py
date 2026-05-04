import torch
import matplotlib


def _create_axes():
    try:
        import matplotlib.pyplot as plt

        return plt, plt.subplots()
    except Exception:
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt, plt.subplots()


def plot_1d(model, problem, t_value=0.0, n_points=200):
    """
    Plot a 1D PINN solution u(x, t) at a fixed time value.
    """
    if len(problem.domain) < 2 or len(problem.vars) < 2:
        raise ValueError("plot_1d expects a problem with variables like ['x', 't']")

    x_low, x_high = problem.domain[0]

    x = torch.linspace(float(x_low), float(x_high), n_points).reshape(-1, 1)
    t = torch.full_like(x, float(t_value))
    inputs = torch.cat([x, t], dim=1)

    was_training = model.training
    model.eval()

    with torch.no_grad():
        u = model(inputs).detach().cpu().numpy()

    if was_training:
        model.train()

    plt, (fig, ax) = _create_axes()
    ax.plot(x.detach().cpu().numpy(), u)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title(f"PINN solution at t = {t_value}")
    ax.grid(True)

    plt.show()

    return fig, ax
