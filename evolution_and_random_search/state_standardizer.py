import numpy as np


class StateStandardizer:
    def __init__(
        self, state_dim, min_normalization_std=0.01,
    ):
        self.state_dim = state_dim
        self.min_normalization_std = min_normalization_std

        self.observed_state_mean = np.zeros(state_dim)
        self.observed_state_std = np.zeros(state_dim)
        self.welford_var_agg = np.ones(state_dim)
        self.num_states_observed = 0

    def load_state_dict(self, data_dict):
        self.observed_state_mean = data_dict["observed_state_mean"]
        self.observed_state_std = data_dict["observed_state_std"]
        self.welford_var_agg = data_dict["welford_var_agg"]
        self.num_states_observed = data_dict["num_states_observed"]

    def get_state_dict(self):
        data_dict = {}
        data_dict["observed_state_mean"] = self.observed_state_mean
        data_dict["observed_state_std"] = self.observed_state_std
        data_dict["welford_var_agg"] = self.welford_var_agg
        data_dict["num_states_observed"] = self.num_states_observed
        data_dict["min_normalization_std"] = self.min_normalization_std
        return data_dict

    def standardize_state(self, state):
        if self.num_states_observed == 0:
            raise ValueError(
                "num_states_observed==0; must observe at least one state before standardizing"
            )
        return (state - self.observed_state_mean) / self.observed_state_std

    def observe_state(self, state):
        self.num_states_observed += 1

        next_mean = (
            self.observed_state_mean
            + (state - self.observed_state_mean) / self.num_states_observed
        )

        self.welford_var_agg += (state - next_mean) * (state - self.observed_state_mean)

        self.observed_state_mean = next_mean

        self.observed_state_std = np.sqrt(
            self.welford_var_agg / self.num_states_observed
        ).clip(min=self.min_normalization_std)


class CenteredLinearRescaler:
    def __init__(self, min_vals, max_vals, scale=1):
        self.min_vals = np.array(min_vals)
        self.max_vals = np.array(max_vals)
        self.scale = scale

    def load_state_dict(self, data_dict):
        self.min_vals = data_dict["min_vals"]
        self.max_vals = data_dict["max_vals"]
        self.scale = data_dict["scale"]

    def get_state_dict(self):
        data_dict = {}
        data_dict["min_vals"] = self.min_vals
        data_dict["max_vals"] = self.max_vals
        data_dict["scale"] = self.scale
        return data_dict

    def standardize_state(self, state):
        # rescale into [0,1]
        X = (state - self.min_vals) / (self.max_vals - self.min_vals)
        # recenter areound zero; double to map into [-1,1], then apply scaling
        return (X - 0.5) * 2 * self.scale

    def observe_state(self, state):
        pass


class IdentityRescaler:
    def __init__(self):
        pass

    def load_state_dict(self, data_dict):
        pass

    def get_state_dict(self):
        return {}

    def standardize_state(self, state):
        return state

    def observe_state(self, state):
        pass
