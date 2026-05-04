import torch


def grad(u, x, order=1):
    """
    Compute derivatives of u with respect to x using PyTorch autograd.

    Parameters:
        u     : tensor (output of neural network)
        x     : tensor (input variable, must have requires_grad=True)
        order : int (1 = first derivative, 2 = second derivative, ...)

    Returns:
        derivative tensor of same shape as u
    """

    if not x.requires_grad:
        raise ValueError("Input tensor x must have requires_grad=True")

    result = u

    for _ in range(order):
        result = torch.autograd.grad(
            outputs=result,
            inputs=x,
            grad_outputs=torch.ones_like(result),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

    return result
