import pytest
import replay_buffer as RB
import numpy as np


def test_init():
    rb = RB.ReplayBuffer(2, 1, 100)
    assert rb.obs_buf.shape == (100, 2)
    assert rb.act_buf.shape == (100,)


def test_store_many_1d():
    rb = RB.ReplayBuffer(1, 1, 100)
    S, A, R, Sn, D = [np.random.randn(10) for i in range(5)]
    rb.store_many(S, A, R, Sn, D)
    assert rb.obs_buf.shape == (100,)
    assert rb.obs2_buf.shape == (100,)
    assert rb.act_buf.shape == (100,)
    assert rb.rew_buf.shape == (100,)
    assert rb.done_buf.shape == (100,)


def test_store_many_3d():
    rb = RB.ReplayBuffer(3, 3, 100, reward_steps=3)
    S, A, R, Sn = [np.random.randn(10, 3) for i in range(4)]
    D = np.random.randn(10)
    rb.store_many(S, A, R, Sn, D)

    assert rb.obs_buf.shape == (100, 3)
    assert rb.obs2_buf.shape == (100, 3)
    assert rb.act_buf.shape == (100, 3)
    assert rb.rew_buf.shape == (100, 3)
    assert rb.done_buf.shape == (100,)


def test_store_many_3d_repeatedly():
    rb = RB.ReplayBuffer(3, 3, 100, reward_steps=3)
    for t in range(50):
        S, A, R, Sn = [np.random.randn(10, 3) for i in range(4)]
        D = np.random.randn(10)
        rb.store_many(S, A, R, Sn, D)

    assert rb.obs_buf.shape == (100, 3)
    assert rb.obs2_buf.shape == (100, 3)
    assert rb.act_buf.shape == (100, 3)
    assert rb.rew_buf.shape == (100, 3)
    assert rb.done_buf.shape == (100,)


def test_store_many_3d_repeatedly_and_sample():
    rb = RB.ReplayBuffer(3, 3, 100, reward_steps=3)
    for t in range(50):

        S, A, R, Sn = [np.random.randn(10, 3) for i in range(4)]
        D = np.random.randn(10)
        rb.store_many(S, A, R, Sn, D)
    S, A, R, Sn, D = rb.sample_batch()
    assert S.shape == (64, 3)
    assert A.shape == (64, 3)
    assert R.shape == (64, 3)
    assert Sn.shape == (64, 3)
    assert D.shape == (64,)


def test_custom_batch_size():
    rb = RB.ReplayBuffer(3, 3, 100, reward_steps=3, batch_size=17)
    for t in range(50):

        S, A, R, Sn = [np.random.randn(10, 3) for i in range(4)]
        D = np.random.randn(10)
        rb.store_many(S, A, R, Sn, D)
    S, A, R, Sn, D = rb.sample_batch()
    assert S.shape == (17, 3)
    assert A.shape == (17, 3)
    assert R.shape == (17, 3)
    assert Sn.shape == (17, 3)
    assert D.shape == (17,)


def test_buffer_wrapping():
    N = 11
    rb = RB.ReplayBuffer(3, 3, 21, reward_steps=3)
    for t in range(2):
        S, A, R, Sn = [np.linspace((3, 4, 5), (8.5, 8.5, 8.5), N) for i in range(4)]
        D = np.random.randn(N)
        rb.store_many(S, A, R, Sn, D)

    S, A, R, Sn = rb.obs_buf, rb.act_buf, rb.rew_buf, rb.obs2_buf
    x = np.array((8.5, 8.5, 8.5))

    assert np.all(S[0, :] == x)
    assert np.all(A[0, :] == x)
    assert np.all(R[0, :] == x)
    assert np.all(Sn[0, :] == x)
