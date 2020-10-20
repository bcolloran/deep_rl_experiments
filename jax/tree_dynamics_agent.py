import jax.numpy as np

from jax_nn_utils import relu_MLP
from replay_buffer import ReplayBuffer
from noise_procs import dampedSpringNoiseStep
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
        # NOTE: idea: after taking action, check to see whether the child nodes from this action
        # are reachable fron the current node; track an estimate of required action to get from
        # one state to the next:
        # action_needed = a(s_t,s_t+1)
        action = next_node.action

    def update(self, last_state, action, reward, state):
        pass


# # %%
# tda.Agent(
#     state=np.array([[1], [2]]),
#     dynamics_layer_sizes=[3, 10, 2],
#     reward_layer_sizes=[3, 5, 5, 1],
#     value_layer_sizes=[2, 5, 1],
#     obs_dim=2,
#     act_dim=1,
#     memory_size=100000,
#     batch_size=64,
#     target_num_leaves=20,
# )


# # %%
# delta_S

# rand_state_0 = (0.0, 0.0, sigma, theta, phi, key)

# for i in range(num_episodes):
#     state = env.reset()

#     value_est = Value(state)

#     new_state_node = Node(
#         action=None, state=state, noise_state=rand_state_0, value_est=value_est
#     )
#     T = tree(init_root=new_state_node)

#     for t in range(max_timesteps):
#         while T.num_leaves() < target_num_leaves:
#             state_node = T.softmax_sample_leaf_states()
#             state = state_node.state
#             noise_state = state_node.noise_state
#             noise_state, action = dampedSpringNoiseStep(state.noise_state)
#             state_est = state + DynamicsFn.est(state, action)
#             reward_est = RewardFn.est(state_est, action)
#             value_est = ValueFn.est(state_est)

#             new_state_node = Node(
#                 action=action,
#                 state=state_est,
#                 noise_state=noise_state,
#                 value_est=value_est,
#             )

#             T.add_state(parent=state_node, child=new_state_node, edge_reward=reward_est)

#         target_node = T.softmax_sample_leaf_states()
#         next_node = T.step_toward_and_reroot(target_node)
#         action = next_node.action
#         # NOTE does there need to be a "done" estimator?
#         last_state = state
#         state, reward, done, _ = env.step(action)

#         agent.update(last_state, action, reward, state)
#         agent.learn()
#         DynamicsFn.update(state, last_state, action)
#         RewardFn.update(reward, state, action)
#         ValueFn.update(reward, state)  # ?
