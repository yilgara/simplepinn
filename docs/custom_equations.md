# Custom Equations

Use `@simplepinn.equation` to turn a residual function into a PDE object.

```python
import simplepinn as sp


@sp.equation
def burgers(u, x, t):
    u_t = sp.grad(u, t)
    u_x = sp.grad(u, x)
    u_xx = sp.grad(u, x, order=2)
    return u_t + u * u_x - 0.01 * u_xx


problem = sp.Problem()
problem.add_pde(burgers)
```

The first argument must be `u`. The remaining argument names must match the
variables in the problem, such as `x` and `t`.
