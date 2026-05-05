import math

import torch
import gradio as gr
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

import simplepinn as sp


def _to_float(value):
    return float(str(value).replace(",", "."))


def run_equation(equation, alpha, epochs, n_pde, t_plot, progress=gr.Progress()):
    equation_key = str(equation).lower()
    coefficient = _to_float(alpha)
    t_plot = _to_float(t_plot)
    epochs = int(epochs)
    n_pde = int(n_pde)

    if equation_key == "poisson":
        raise gr.Error("Poisson is visible in the UI, but its preset is not added yet.")

    progress(0, desc="Preparing problem...")

    def update_progress(epoch, total_epochs, history):
        fraction = min((epoch + 1) / total_epochs, 1.0)
        progress(fraction, desc=f"Training epoch {epoch + 1}/{total_epochs}")

    solve_kwargs = dict(
        equation=equation_key,
        epochs=epochs,
        n_pde=n_pde,
        progress_callback=update_progress,
    )

    if equation_key == "heat":
        solve_kwargs["alpha"] = coefficient
        solve_kwargs["compute_error"] = False
    elif equation_key == "burgers":
        solve_kwargs["nu"] = coefficient

    model = sp.solve(**solve_kwargs)

    progress(0.95, desc="Computing exact solution and plots...")

    x = torch.linspace(0, 1, 200).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model.predict(x, t=t_plot)

    solution_fig, ax = plt.subplots()
    ax.plot(x.numpy(), u_pred.numpy(), label="PINN")

    if equation_key == "heat":
        u_exact = math.exp(-coefficient * math.pi**2 * t_plot) * torch.sin(math.pi * x)
        error = torch.mean(torch.abs(u_pred - u_exact)).item()
        ax.plot(x.numpy(), u_exact.numpy(), "--", label="Exact")
        status = f"Mean absolute error: {error:.6f}"
    else:
        status = "No analytical reference shown for Burgers yet."

    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"{equation} solution at t={t_plot}")
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

    return solution_fig, loss_fig, status


demo = gr.Interface(
    fn=run_equation,
    inputs=[
        gr.Dropdown(["Heat", "Burgers", "Poisson"], value="Heat", label="Equation"),
        gr.Textbox(value="0.01", label="Coefficient (alpha for Heat, nu for Burgers)"),
        gr.Slider(100, 5000, value=1000, step=100, label="Epochs"),
        gr.Slider(100, 5000, value=1000, step=100, label="Collocation points"),
        gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Plot time t"),
    ],
    outputs=[
        gr.Plot(label="PINN vs Exact Solution"),
        gr.Plot(label="Training Loss Curve"),
        gr.Textbox(label="Error"),
    ],
    title="simplepinn Multi-Physics Demo",
    description="Train a Physics-Informed Neural Network for Heat or Burgers equations. Poisson is listed as the next preset.",
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
