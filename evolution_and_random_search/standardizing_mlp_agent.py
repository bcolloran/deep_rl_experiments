import numpy as np
from state_standardizer import StateStandardizer, CenteredLinearRescaler


def relu(x):
    return np.maximum(x, 0)


# def layer_end_indices(size_in, size_hid, size_out):
#     W1_end = size_hid * size_in
#     b1_end = W1_end + size_hid
#     W2_end = b1_end + size_out * size_hid
#     return (W1_end, b1_end, W2_end)


# def layer_end_indices(size_in, sizes_hid, size_out):
#     layer_sizes = [size_in] + sizes_hid + [size_out]
#     indices = []

#     index_so_far = 0
#     for i in range(len(layer_sizes) - 2):
#         # weight matrix
#         weight_size = layer_sizes[i] * layer_sizes[i + 1]
#         index_so_far += weight_size
#         indices.append(index_so_far)
#         # bias vector
#         bias_size = layer_sizes[i + 1]
#         index_so_far += bias_size
#         indices.append(index_so_far)

#     indices.append(index_so_far + layer_sizes[-2] * layer_sizes[-1])

#     return indices


def num_model_params(layer_sizes):
    weight_inds, bias_inds = weight_and_bias_indices(layer_sizes)
    return bias_inds[-1][-1]


def weight_and_bias_indices(layer_sizes):
    # layer_sizes = [size_in] + sizes_hidden + [size_out]
    weight_inds = []
    bias_inds = []

    index_so_far = 0
    for i in range(len(layer_sizes) - 1):
        # weight matrix
        weight_size = layer_sizes[i] * layer_sizes[i + 1]
        weight_inds.append((index_so_far, index_so_far + weight_size))
        index_so_far += weight_size
        # bias vector
        bias_size = layer_sizes[i + 1]
        bias_inds.append((index_so_far, index_so_far + bias_size))
        index_so_far += bias_size

    return weight_inds, bias_inds


def get_weights_and_biases_as_param_vects(param_vector, weight_inds, bias_inds):
    if len(weight_inds) != len(bias_inds):
        raise ValueError(
            f"number of weight matrices ({len(weight_inds)}) != "
            f"number of bias vectors ({len(bias_inds)})"
        )
    weights = []
    biases = []
    for i in range(len(weight_inds)):
        w0, w1 = weight_inds[i]
        b0, b1 = bias_inds[i]
        weights.append(param_vector[w0:w1])
        biases.append(param_vector[b0:b1])
    return weights, biases


def get_weights_and_biases(param_vector, layer_sizes):
    weight_inds, bias_inds = weight_and_bias_indices(layer_sizes)
    weights_v, biases_v = get_weights_and_biases_as_param_vects(
        param_vector, weight_inds, bias_inds
    )
    weights = []
    biases = []
    for i, w in enumerate(weights_v):
        weights.append(w.reshape(layer_sizes[i + 1], layer_sizes[i]))

    for b in biases_v:
        biases.append(b.reshape(-1, 1))

    return weights, biases


def apply_layers(weights, biases, state, activation):
    v = state
    for i, W in enumerate(weights[:-1]):
        b = biases[i]
        v = activation(np.matmul(W, v) + b)
    return np.matmul(weights[-1], v)


class StandardizingMLPAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_trajectory_steps=1000,
        min_normalization_std=0.01,
        activation="tanh",
        hidden_layer_sizes=None,
        standardizer=None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_trajectory_steps = max_trajectory_steps
        self.min_normalization_std = min_normalization_std

        if standardizer is None:
            standardizer = StateStandardizer(state_dim, min_normalization_std)
        self.standardizer = standardizer

        if hidden_layer_sizes is None:
            hidden_layer_sizes = [int(np.ceil((state_dim + action_dim) / 2))]

        self.layer_sizes = [state_dim] + hidden_layer_sizes + [action_dim]

        self.activation = relu if activation == "relu" else np.tanh

        num_params = num_model_params(self.layer_sizes)

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

    def act(self, state, params=None):
        if params is None:
            params = self.model_params

        std_state = self.standardizer.standardize_state(state)

        weights, biases = get_weights_and_biases(self.model_params, self.layer_sizes)
        action = apply_layers(weights, biases, std_state, self.activation)
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


def PendulumAgent(
    state_dim,
    action_dim,
    max_trajectory_steps=1000,
    activation="relu",
    hidden_layer_sizes=[9, 9],
    # standardizer=CenteredLinearRescaler,
):
    return StandardizingMLPAgent(
        state_dim,
        action_dim,
        max_trajectory_steps=1000,
        min_normalization_std=0.01,
        activation=activation,
        hidden_layer_sizes=hidden_layer_sizes,
        standardizer=CenteredLinearRescaler([-1, -1, -8], [1, 1, 8]),
    )

    #     state_dim,
    #     action_dim,
    #     max_trajectory_steps=max_trajectory_steps,
    #     activation=activation,
    #     hidden_layer_sizes=hidden_layer_sizes,
    #     standardizer=standardizer,
    # )
