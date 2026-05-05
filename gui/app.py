import math

import torch
import gradio as gr
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import simplepinn as sp


def _to_float(value):
    return float(str(value).replace(",", "."))


def run_heat(alpha, epochs, n_pde, t_plot, progress=gr.Progress()):
    alpha = _to_float(alpha)
    t_plot = _to_float(t_plot)
    epochs = int(epochs)
    n_pde = int(n_pde)

    progress(0, desc="Preparing problem...")

    problem = sp.Problem(
        domain=[(0, 1), (0, 1)],
        vars=["x", "t"]
    )

    problem.add_pde(
        sp.equations.HeatEquation(alpha=alpha)
    )

    problem.add_boundary(
        sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0)
    )

    problem.add_initial(
        sp.boundaries.Function(lambda x: torch.sin(torch.pi * x))
    )

    model = sp.PINN(problem, hidden_layers=4, hidden_units=64)

    def update_progress(epoch, total_epochs, history):
        fraction = min((epoch + 1) / total_epochs, 1.0)
        progress(fraction, desc=f"Training epoch {epoch + 1}/{total_epochs}")

    model.fit(
        epochs=epochs,
        lr=1e-3,
        n_pde=n_pde,
        progress_callback=update_progress,
    )

    progress(0.95, desc="Computing exact solution and plots...")

    x = torch.linspace(0, 1, 200).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model.predict(x, t=t_plot)

    u_exact = math.exp(-alpha * math.pi**2 * t_plot) * torch.sin(math.pi * x)

    error = torch.mean(torch.abs(u_pred - u_exact)).item()

    solution_fig, ax = plt.subplots()
    ax.plot(x.numpy(), u_pred.numpy(), label="PINN")
    ax.plot(x.numpy(), u_exact.numpy(), "--", label="Exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Heat equation solution at t={t_plot}")
    ax.grid(True)
    ax.legend()

    loss_fig, loss_ax = plt.subplots()
    loss_ax.plot(model.history["epoch"], model.history["total"], label="Total loss")
    loss_ax.plot(model.history["epoch"], model.history["pde"], label="PDE loss", alpha=0.8)
    loss_ax.plot(model.history["epoch"], model.history["bc"], label="Boundary loss", alpha=0.8)
    loss_ax.plot(model.history["epoch"], model.history["ic"], label="Initial loss", alpha=0.8)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Training loss curve")
    loss_ax.grid(True)
    loss_ax.legend()

    progress(1, desc="Done")

    return solution_fig, loss_fig, f"Mean absolute error: {error:.6f}"


demo = gr.Interface(
    fn=run_heat,
    inputs=[
        gr.Number(value=0.01, label="Alpha"),
        gr.Slider(100, 5000, value=1000, step=100, label="Epochs"),
        gr.Slider(100, 5000, value=1000, step=100, label="Collocation points"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Plot time t"),
    ],
    outputs=[
        gr.Plot(label="PINN vs Exact Solution"),
        gr.Plot(label="Training Loss Curve"),
        gr.Textbox(label="Error"),
    ],
    title="simplepinn Heat Equation Demo",
    description="Train a Physics-Informed Neural Network for the 1D heat equation. Progress appears while the model trains.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
