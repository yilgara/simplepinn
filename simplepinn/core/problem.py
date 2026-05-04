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
