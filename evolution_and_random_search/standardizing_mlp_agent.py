import numpy as np
from state_standardizer import StateStandardizer


def relu(x):
    return np.maximum(x, 0)


def layer_end_indices(size_in, size_hid, size_out):
    W1_end = size_hid * size_in
    b1_end = W1_end + size_hid
    W2_end = b1_end + size_out * size_hid
    return (W1_end, b1_end, W2_end)


class StandardizingMLPAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_trajectory_steps=1000,
        min_normalization_std=0.01,
        activation="tanh",
        hidden_layer_size=None,
    ):
        self.name = "StandardizingLinearAgent"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_steps = max_trajectory_steps
        self.min_normalization_std = min_normalization_std

        self.standardizer = StateStandardizer(state_dim, min_normalization_std)

        if hidden_layer_size is None:
            hidden_layer_size = int(np.ceil((state_dim + action_dim) / 2))

        self.hidden_layer_size = hidden_layer_size

        self.activation = relu if activation == "relu" else np.tanh

        num_params = layer_end_indices(state_dim, hidden_layer_size, action_dim)[-1]

        self.model_params = np.zeros(num_params)

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

    def get_model_weights_and_biases(self, params=None):
        if params is None:
            params = self.model_params

        h_size = self.hidden_layer_size

        (W1_end, b1_end, W2_end) = layer_end_indices(
            self.state_dim, self.hidden_layer_size, self.action_dim
        )

        W1 = params[:W1_end].reshape(h_size, self.state_dim)
        b1 = params[W1_end:b1_end].reshape(h_size, 1)
        W2 = params[b1_end:W2_end].reshape(self.action_dim, h_size)

        return (W1, b1, W2)

    def act(self, state, params=None):
        if params is None:
            params = self.model_params

        # model_matrix = params.reshape(self.action_dim, self.state_dim)

        std_state = self.standardizer.standardize_state(state)

        (W1, b1, W2) = self.get_model_weights_and_biases(params)

        x = self.activation(np.matmul(W1, std_state) + b1)
        action = np.matmul(W2, x)
        return action.ravel()

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
