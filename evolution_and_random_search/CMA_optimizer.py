import cma


class CMA_optimizer:
    def __init__(self, starting_params, sigma0=0.5, pop_size=None):
        self.opt = cma.CMAEvolutionStrategy(starting_params, sigma0)
        print("cma popsize", self.opt.popsize)

    def ask(self):
        return self.opt.ask()

    def tell(self, candidates, rewards):
        print("cma sigma", self.opt.sigma)
        return self.opt.tell(candidates, -rewards)

    @property
    def result(self):
        return self.opt.result
