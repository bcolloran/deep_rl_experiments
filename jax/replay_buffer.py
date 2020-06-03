import numpy as np


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, batch_size=64):
        # pass
        print(obs_dim, act_dim, size, batch_size)
        self.batch_size = batch_size
        self.obs_buf = np.zeros(size, obs_dim)
        self.obs2_buf = np.zeros(size, obs_dim)
        self.act_buf = np.zeros(size, act_dim)
        self.rew_buf = np.zeros(size)
        self.done_buf = np.zeros(size)
        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch
