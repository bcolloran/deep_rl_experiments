import jax.numpy as np

from jax_nn_utils import relu_MLP
from replay_buffer import ReplayBuffer
from damped_spring_noise import dampedSpringNoiseStep
from trees import Tree

from collections import namedtuple


nodeData = namedtuple(
    "NodeData", ["action_inbound", "state", "noise_state", "value_est", "step_reward"]
)


class Agent:
    def __init__(
        self,
        state,
        dynamics_layer_sizes,
        reward_layer_sizes,
        value_layer_sizes,
        obs_dim,
        act_dim,
        memory_size,
        batch_size,
        target_num_leaves,
        cross_val_ratio=0.1,
        rand_state_0=(0, 0, 0.5, 0.05, 0.01),
    ):

        self.dynamics = relu_MLP(dynamics_layer_sizes)
        self.reward = relu_MLP(reward_layer_sizes)
        self.value = relu_MLP(value_layer_sizes)

        self.memory_train = ReplayBuffer(obs_dim, act_dim, memory_size, batch_size)
        self.memory_test = ReplayBuffer(obs_dim, act_dim, memory_size, batch_size)

        self.target_num_leaves = target_num_leaves
        new_state_node = nodeData(
            action_inbound=None,
            state=state,
            noise_state=rand_state_0,
            value_est=self.value.predict(state),
            step_reward=0,
        )
        self.T = Tree(new_state_node)

    def act(self, state):
        while self.T.num_leaves() < self.target_num_leaves:
            state_node = self.T.softmax_sample_leaves()
            state = state_node.state
            noise_state = state_node.noise_state
            noise_state, action = dampedSpringNoiseStep(state.noise_state)
            state_est = state + self.dynamics.predict(np.vstack(state, action))
            reward_est = self.reward.predict(np.vstack(state_est, action))
            value_est = self.value.predict(state_est)

            new_state_node = nodeData(
                action_inbound=action,
                state=state_est,
                noise_state=noise_state,
                value_est=value_est,
                step_reward=reward_est,
            )

            self.T.add_state(
                parent=state_node, child=new_state_node,
            )
        target_node = self.T.softmax_sample_leaf_states()
        next_node = self.T.step_toward_and_reroot(target_node)
        action = next_node.action

    def update(self, last_state, action, reward, state):
        pass
