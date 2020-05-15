import numpy as np
from state_standardizer import StateStandardizer


class StandardizingLinearAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_trajectory_steps=1000,
        min_normalization_std=0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_steps = max_trajectory_steps
        self.min_normalization_std = min_normalization_std

        self.model_params = np.zeros(action_dim * state_dim)
        self.standardizer = StateStandardizer(state_dim, min_normalization_std)

    def load_agent_state_dict(self, data_dict):
        self.model_params = data_dict["model_params"]
        self.standardizer.load_state_dict(data_dict["standardizer_state"])

    def get_agent_state_dict(self):
        data_dict = {}
        data_dict["model_params"] = self.model_params
        data_dict["standardizer_state"] = self.standardizer.get_state_dict()
        data_dict["max_trajectory_steps"] = self.max_trajectory_steps
        data_dict["min_normalization_std"] = self.min_normalization_std
        return data_dict

    def get_model_param_vector(self):
        return self.model_params

    def update_param_vector(self, param_vector):
        self.model_params = param_vector

    def act(self, state, params=None):
        if params is None:
            params = self.model_params

        model_matrix = params.reshape(self.action_dim, self.state_dim)

        std_state = self.standardizer.standardize_state(state)

        return np.matmul(model_matrix, std_state).ravel()

    def run_trajectory_in_env(self, env, testing=False, params=None):
        state = env.reset()
        total_reward = 0
        for steps in range(self.max_trajectory_steps):
            if not testing:
                self.standardizer.observe_state(state)
            action = self.act(state, params)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break
        if testing:
            return total_reward, steps, done
        else:
            return total_reward
