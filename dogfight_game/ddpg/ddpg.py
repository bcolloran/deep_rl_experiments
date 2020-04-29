import gc
import logging
import os
import torch
import numpy as np

import torch.nn.functional as F
from torch.optim import Adam

from .nets import Actor, Critic
from .replay_buffer import ReplayBuffer

logger = logging.getLogger("ddpg")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# if gpu is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def noise(self):
        x = (
            self.x_prev
            + self.theta * (self.mu - self.x_prev) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(
            self.mu, self.sigma
        )


class DDPG(object):
    def __init__(
        self,
        gamma,
        tau,
        hidden_size,
        num_inputs,
        action_space,
        noise_process=None,
        checkpoint_dir="./saved_models/",
        buffer_size=2 ** 16,
        batch_size=64,
        device="cpu",
        seed=123,
        eps=0.1,
    ):
        """
        Deep Deterministic Policy Gradient
        Read the detail about it here:
        https://arxiv.org/abs/1509.02971

        Arguments:
            gamma:          Discount factor
            tau:            Update factor for the actor and the critic
            hidden_size:    Number of units in the hidden layers
                of the actor and critic. Must be of length 2.
            num_inputs:     Size of the input states
            action_space:   The action space of the used environment.
                Used to clip the actions and to distinguish the
                number of outputs. action_vars by 2 np.array
            checkpoint_dir: Path as String to the directory to save the networks. 
                            If None then "./saved_models/" will be used
        """

        self.gamma = gamma
        self.tau = tau
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.eps = eps

        # Define the actor
        self.actor = Actor(hidden_size, num_inputs, self.action_space).to(self.device)
        self.actor_target = Actor(hidden_size, num_inputs, self.action_space).to(
            self.device
        )

        # Define the critic
        self.critic = Critic(hidden_size, num_inputs, self.action_space).to(self.device)
        self.critic_target = Critic(hidden_size, num_inputs, self.action_space).to(
            device
        )

        # Define the optimizers for both networks
        # optimizer for the actor network
        self.actor_optimizer = Adam(self.actor.parameters(), lr=1e-4)
        # optimizer for the critic network
        self.critic_optimizer = Adam(
            self.critic.parameters(), lr=1e-3, weight_decay=1e-2
        )

        self.memory = ReplayBuffer(
            action_space.shape[0], self.buffer_size, self.batch_size, self.device, seed
        )
        self.t_step = 0

        self.noise_process = noise_process

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        print("init - saving all...")
        logger.info("Saving all checkpoints to {}".format(self.checkpoint_dir))

    def act(self, state, explore=False):
        """
        Evaluates the action to perform in a given state

        Arguments:
            state: State to perform the action on in the env.
            action_noise: If not None, the noise to apply on the evaluated action
        """
        x = torch.Tensor(state).to(self.device)

        # Get the continous action value to perform in the env
        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)
        self.actor.train()  # Sets the actor in training mode
        mu = mu.cpu().data.numpy()

        # During training we add noise for exploration
        # must clip to [-1,1] interval
        if self.noise_process is not None and explore:
            # noise = self.noise_process.noise()
            if np.random.rand(1) < self.eps:
                mu = np.random.rand(mu.shape[0]) * 2 - 1
        return mu

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        # print("memory to add", state, action, reward, next_state, done)
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
        This means we ...
            1. Compute the targets
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update

        Arguments:
            batch:  Batch to perform the training of the parameters
        """
        # Get tensors from the batch

        (state_batch, action_batch, reward_batch, next_state_batch, done_batch) = batch
        # print("action_batch", action_batch.size())
        # print("state_batch", state_batch.size())
        # print("next_state_batch", next_state_batch.size())
        # print("reward_batch", reward_batch.size())
        # print("done_batch", done_batch.size())

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        # print("next_action_batch", next_action_batch.size())
        next_state_action_values = self.critic_target(
            next_state_batch, next_action_batch.detach()
        )

        # Compute the target
        expected_values = (
            reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values
        )
        # print("next_state_action_values", next_state_action_values.size())

        # TODO: Clipping the expected values here?
        # expected_value = torch.clamp(expected_value, min_value, max_value)

        # Update the critic network
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)

        # print("expected_values -- TARGET", expected_values.size())

        # print("state_action_batch -- INPUT", state_action_batch.size())

        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        # print("LOSS", value_loss)

        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor network
        self.actor_optimizer.zero_grad()
        policy_loss = -self.critic(state_batch, self.actor(state_batch))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        """
        checkpoint_name = self.checkpoint_dir + f"/ep_{self.t_step}.pth.tar"
        logger.info("Saving checkpoint...")
        checkpoint = {
            "last_timestep": self.t_step,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_buffer": self.memory,
        }

        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        logger.info(f"Saved model at timestep {self.t_step} to {self.checkpoint_dir}")

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [
            file
            for file in os.listdir(self.checkpoint_dir)
            if (file.endswith(".pt") or file.endswith(".tar"))
        ]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path. If the given path is None
        then the latest saved file in 'checkpoint_dir' will be used.

        Arguments:
            checkpoint_path:    File to load the model from

        """

        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info("Loading checkpoint...({})".format(checkpoint_path))
            key = "cuda" if torch.cuda.is_available() else "cpu"

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint["last_timestep"] + 1
            self.actor.load_state_dict(checkpoint["actor"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.actor_target.load_state_dict(checkpoint["actor_target"])
            self.critic_target.load_state_dict(checkpoint["critic_target"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
            replay_buffer = checkpoint["replay_buffer"]

            gc.collect()
            logger.info(
                "Loaded model at timestep {} from {}".format(
                    start_timestep, checkpoint_path
                )
            )
            return start_timestep, replay_buffer
        else:
            raise OSError("Checkpoint not found")

    def set_eval(self):
        """
        Sets the model in evaluation mode

        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode

        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == "Actor":
            return self.actor
        elif name == "Critic":
            return self.critic
        else:
            raise NameError("name '{}' is not defined as a network".format(name))
