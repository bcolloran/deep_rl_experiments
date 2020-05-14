import numpy as np

# based on paper 1803.07055.pdf from arxiv
# implements ask/tell part of api from https://github.com/CMA-ES/pycma
class ARS_optimizer:
    def __init__(
        self,
        starting_params,
        learning_rate=0.02,
        pop_size=16,
        std_dev_exploration_noise=0.03,
    ):
        self.name = "ARS"
        self.params = starting_params
        self.dim = starting_params.shape[0]
        self.N = pop_size
        self.nu = std_dev_exploration_noise
        self.lr = learning_rate  # alpha in the paper

    def ask(self):
        param_deltas = [np.random.randn(self.dim) for _ in range(self.N)]

        candidates_pos = [self.params + D * self.nu for D in param_deltas]
        candidates_neg = [self.params - D * self.nu for D in param_deltas]

        return candidates_pos + candidates_neg

    def tell(self, candidates, rewards):
        candidates_pos = candidates[0 : self.N]
        reward_pos = rewards[: self.N]
        reward_neg = rewards[self.N :]
        reward_diffs = reward_pos - reward_neg

        std_rewards = np.std(rewards)

        param_adjustment = np.zeros_like(self.params)

        # param_adjustment derived from 1803.07055.pdf eq 7 in alg 2
        # by setting C_k = M_j - nu*delta_k , which implies
        # delta_k = (C_k - M_j)/nu , and then cranking through the algebra
        for j, C in enumerate(candidates_pos):
            param_adjustment += C * reward_diffs[j]

        param_adjustment -= self.params * np.sum(reward_diffs)

        self.params += (self.lr / (std_rewards * self.N * self.nu)) * param_adjustment

    @property
    def result(self):
        return [self.params]
