class BaseEquation:
    """
    Base class for all PDE equations.
    Every equation must implement residual(model, coords).
    """

    def residual(self, model, coords):
        raise NotImplementedError(
            "Each equation must implement a residual(model, coords) method."
        )