import torch
import gradio as gr
import matplotlib.pyplot as plt

import simplepinn as sp


def run_heat(alpha, epochs, n_pde, t_plot):
    problem = sp.Problem(
        domain=[(0, 1), (0, 1)],
        vars=["x", "t"]
    )

    problem.add_pde(
        sp.equations.HeatEquation(alpha=float(alpha))
    )

    problem.add_boundary(
        sp.boundaries.Dirichlet(edges=["left", "right"], value=0.0)
    )

    problem.add_initial(
        sp.boundaries.Function(lambda x: torch.sin(torch.pi * x))
    )

    model = sp.PINN(problem, hidden_layers=4, hidden_units=64)
    model.fit(epochs=int(epochs), lr=1e-3, n_pde=int(n_pde))

    x = torch.linspace(0, 1, 200).reshape(-1, 1)

    with torch.no_grad():
        u_pred = model.predict(x, t=float(t_plot))

    u_exact = torch.exp(
        torch.tensor(-float(alpha) * torch.pi**2 * float(t_plot))
    ) * torch.sin(torch.pi * x)

    error = torch.mean(torch.abs(u_pred - u_exact)).item()

    fig, ax = plt.subplots()
    ax.plot(x.numpy(), u_pred.numpy(), label="PINN")
    ax.plot(x.numpy(), u_exact.numpy(), "--", label="Exact")
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(f"Heat equation solution at t={t_plot}")
    ax.grid(True)
    ax.legend()

    return fig, f"Mean absolute error: {error:.6f}"


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
        gr.Textbox(label="Error"),
    ],
    title="simplepinn Heat Equation Demo",
    description="Train a Physics-Informed Neural Network for the 1D heat equation.",
)

if __name__ == "__main__":
    demo.launch()
