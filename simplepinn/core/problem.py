class Problem:
    """
    Stores the full physics problem definition.

    A Problem contains:
    - domain
    - variables
    - PDEs
    - boundary conditions
    - initial conditions
    - data constraints
    """

    def __init__(self, domain=None, vars=None, sampler="uniform"):
        if domain is None:
            domain = [(0, 1), (0, 1)]

        if vars is None:
            vars = ["x", "t"]

        if not isinstance(domain, list):
            raise ValueError("domain must be a list of tuples")

        if not all(isinstance(bounds, tuple) and len(bounds) == 2 for bounds in domain):
            raise ValueError("domain entries must be tuples like (low, high)")

        if not isinstance(vars, list):
            raise ValueError("vars must be a list of variable names")

        if len(vars) != len(domain):
            raise ValueError("vars must match domain dimensions")

        self.domain = domain
        self.vars = vars
        self.sampler = sampler

        self.pdes = []
        self.boundaries = []
        self.initials = []
        self.data = []

    def add_pde(self, pde):
        self.pdes.append(pde)
        return self

    def add_boundary(self, boundary):
        self.boundaries.append(boundary)
        return self

    def add_initial(self, initial):
        self.initials.append(initial)
        return self

    def add_data(self, *data):
        self.data.append(data)
        return self

    def summary(self):
        return {
            "domain": self.domain,
            "vars": self.vars,
            "num_pdes": len(self.pdes),
            "num_boundaries": len(self.boundaries),
            "num_initials": len(self.initials),
            "num_data_constraints": len(self.data),
            "sampler": self.sampler,
        }
