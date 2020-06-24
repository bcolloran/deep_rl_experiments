import numpy as np


def SARnSn_from_SAR(S, A, R, n):
    S1 = S[:-n]
    A1 = A[:-n]
    # R1n = np.vstack([R[m : -(n - m)] for m in range(n)]).T
    R1n = np.hstack([R[m : -(n - m)] for m in range(n)])
    Sn = S[n:]
    return S1, A1, R1n, Sn


class ReplayBuffer:
    """
    A FIFO experience replay buffer that can accomodate n-step rewards

    """

    def __init__(self, obs_dim, act_dim, size, reward_steps=1, batch_size=64):
        # pass
        self.batch_size = batch_size

        self.S1 = np.zeros((size, obs_dim))  # current state
        self.Sn = np.zeros((size, obs_dim))  # state in n steps

        self.A1 = np.zeros((size, act_dim))  # action at current time
        self.R1n = np.zeros((size, reward_steps))  # reward at times t to n

        self.Done1 = np.zeros(size)  # episode reaches terminal state after this action?

        self.ptr = 0
        self.size = 0
        self.max_size = size

    def store(self, obs, act, rew, next_obs, done):
        self.S1[self.ptr] = obs
        self.Sn[self.ptr] = next_obs
        self.A1[self.ptr] = act
        self.R1n[self.ptr] = rew
        self.Done1[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def store_many(self, obs, act, rew, next_obs, done):
        N = obs.shape[0]
        p = self.ptr
        n = p + N
        if p + N <= self.max_size:
            self.S1[p:n] = obs
            self.Sn[p:n] = next_obs
            self.A1[p:n] = act
            self.R1n[p:n] = rew
            self.Done1[p:n] = done
            self.ptr = n
            self.size = min(self.size + N, self.max_size)
        else:
            overflow = (p + N) - self.max_size
            self.S1[p:] = obs[:-overflow]
            self.Sn[p:] = next_obs[:-overflow]
            self.A1[p:] = act[:-overflow]
            self.R1n[p:] = rew[:-overflow]
            self.Done1[p:] = done[:-overflow]

            self.S1[:overflow] = obs[-overflow:]
            self.Sn[:overflow] = next_obs[-overflow:]
            self.A1[:overflow] = act[-overflow:]
            self.R1n[:overflow] = rew[-overflow:]
            self.Done1[:overflow] = done[-overflow:]

            self.ptr = overflow
            self.size = N

    def sample_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.S1[idxs],
            self.Sn[idxs],
            self.A1[idxs],
            self.R1n[idxs],
            self.Done1[idxs],
        )
