from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import datetime
import sac_openai_cuda_nets as core
from spinup.utils.logx import EpochLogger

# from collections import deque
import matplotlib.pyplot as plt

from IPython import display
import pickle
import os


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def elapsed_time_string(elapsed_time):
    if elapsed_time < 60:
        time_str = f"{elapsed_time:.2f} sec"
    elif elapsed_time < 60 * 60:
        time_str = f"{elapsed_time/60:.2f} min"
    else:
        time_str = f"{elapsed_time/(60*60):.4f} hr"
    return time_str


def timestamp():
    return datetime.datetime.now().isoformat().replace(":", "_")


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device="cpu"):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr = 0
        self.size = 0
        self.max_size = size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32).to(self.device)
            for k, v in batch.items()
        }


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0.0, theta=0.05, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.seed = np.random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = np.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state


def sac(
    env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=int(1e6),
    gamma=0.99,
    polyak=0.995,
    lr=1e-3,
    alpha=0.2,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=1000,
    logger_kwargs=dict(),
    use_logger=True,
    save_freq=1,
    device="cpu",
    logger=None,
    epoch_plot_fn=None,
    st_plot_fn=None,
    add_noise=False,
):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    run_datetime_str = timestamp()
    run_params = {
        "seed": seed,
        "steps_per_epoch": steps_per_epoch,
        "epochs": epochs,
        "replay_size": replay_size,
        "gamma": gamma,
        "polyak": polyak,
        "lr": lr,
        "alpha": alpha,
        "batch_size": batch_size,
        "start_steps": start_steps,
        "update_after": update_after,
        "update_every": update_every,
        "num_test_episodes": num_test_episodes,
        "max_ep_len": max_ep_len,
        "use_logger": use_logger,
        "save_freq": save_freq,
        "device": device,
    }

    if use_logger:
        logger = EpochLogger(**logger_kwargs)
        logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac.to(device)
    ac_targ = deepcopy(ac)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device
    )

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])

    if add_noise:
        noise_process = OUNoise(act_dim, seed)
    if use_logger:
        logger.log(
            "\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" % var_counts
        )

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(
            Q1Vals=q1.detach().cpu().numpy(), Q2Vals=q2.detach().cpu().numpy()
        )

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data["obs"]
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    if use_logger:
        # Set up model saving
        logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        if use_logger:
            logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        if use_logger:
            logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device), deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            if use_logger:
                logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    start_time = time.time()

    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch

    # episode_rewards = []
    # recent_episode_rewards = deque(maxlen=50)  # last 100 scores
    # avg_recent_episode_rewards = []
    epoch = 0
    episode_num = 0

    # step_rewards = []
    # recent_step_rewards = deque(maxlen=1000)  # last 1000 scores
    # avg_recent_step_rewards = []

    step_log = {
        "episode num": [],
        "elapsed time": [],
        "reward": [],
        "step time": [],
        "act time": [],
        "train time": [],
        "env time": [],
        "plot time": [],
    }

    episode_log = {
        "episode num": [],
        "elapsed time": [],
        "reward": [],
        "episode steps": [],
        "episode done": [],
    }

    figure, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    ax1.set_xlabel("episode")
    ax1.set_ylabel("reward")

    ax2.set_xlim([0, total_steps])
    ax2.set_xlabel("step")
    ax2.set_ylabel("reward")

    ax3.set_xlim([0, total_steps])
    ax3.set_xlabel("step")
    ax3.set_ylabel("time")
    ax3.set_yscale("log")

    # l_step, _ = ax3.plot(step_log["step time"])
    # l_step.set_label("step")
    # l_act, _ = ax3.plot(step_log["act time"])
    # l_act.set_label("act")
    # l_train, _ = ax3.plot(step_log["train time"])
    # l_train.set_label("train")
    # l_env, _ = ax3.plot(step_log["env time"])
    # l_env.set_label("env")

    first_plot = True

    steps_to_average = 10000
    episodes_to_average = 100

    def update_plot(first_plot):
        t = len(step_log["reward"])

        elapsed_time = np.round(time.time() - start_time)

        plot_title = (
            f"episode {episode_num};  step {t}"
            f"\nRecent avg step score: {np.mean(step_log['reward'][-steps_to_average:]):.2f}"
            f"\ntime: {elapsed_time_string(elapsed_time)}"
            f" {(time.time() - start_time) / t:.4f}s/step)"
        )

        ax1.set_title(plot_title)
        ax1.plot(episode_log["reward"], "g")
        ax1.plot(running_mean(episode_log["reward"], -episodes_to_average), "k")

        ax2.plot(step_log["reward"], "g")
        ax2.plot(running_mean(step_log["reward"], -steps_to_average), "k")

        # l_step.set_ydata(step_log["step time"])
        # l_act.set_ydata(step_log["act time"])
        # l_train.set_ydata(step_log["train time"])
        # l_env.set_ydata(step_log["env time"])

        (l_step,) = ax3.plot(step_log["step time"], color="k")
        (l_act,) = ax3.plot(step_log["act time"], color="b")
        (l_train,) = ax3.plot(step_log["train time"], color="r")
        (l_env,) = ax3.plot(step_log["env time"], color="g")

        if first_plot:
            first_plot = False
            l_step.set_label("step")
            l_act.set_label("act")
            l_train.set_label("train")
            l_env.set_label("env")
            ax3.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            # plt.show(block=False)

        if st_plot_fn is not None:
            st_plot_fn(figure)

        else:
            display.clear_output(wait=True)
            display.display(figure)
        # return first_plot

    add_noise_this_episode = False
    for t in range(total_steps):
        # print("steps", t)
        step_start_time = time.perf_counter()
        episode_start_time = time.perf_counter()

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.

        act_time = time.perf_counter()
        if t > start_steps:
            a = get_action(o)
            if add_noise_this_episode:
                a = np.clip(a + noise_process.sample(), -1, 1)
        else:
            a = env.action_space.sample()
        act_time_this_step = time.perf_counter() - act_time

        # Step the env
        env_time = time.perf_counter()
        o2, r, d, _ = env.step(a)
        env_time_this_step = time.perf_counter() - env_time

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            episode_num += 1
            if use_logger:
                logger.store(EpRet=ep_ret, EpLen=ep_len)

            episode_log["episode num"].append(episode_num)
            episode_log["elapsed time"].append(time.perf_counter() - episode_start_time)
            episode_log["reward"].append(ep_ret)

            episode_start_time = time.perf_counter()
            o, ep_ret, ep_len = env.reset(), 0, 0
            if add_noise and np.random.rand() < 0.02:
                add_noise_this_episode = True
            else:
                add_noise_this_episode = False

        # Update handling
        train_time = time.perf_counter()
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
        train_time_this_step = time.perf_counter() - train_time

        step_log["episode num"].append(episode_num)
        step_log["elapsed time"].append(time.time() - start_time)
        step_log["reward"].append(r)
        step_log["step time"].append(time.perf_counter() - step_start_time)
        step_log["train time"].append(train_time_this_step)
        step_log["env time"].append(env_time_this_step)
        step_log["act time"].append(act_time_this_step)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            elapsed_time = np.round(time.time() - start_time)

            print(
                f"\rEpoch {epoch}; {episode_num} episodes; {t} steps;"
                f"     elapsed time: {elapsed_time_string(elapsed_time)}"
            )

            log_path = f"{os.getcwd()}/model_runs/{run_datetime_str}"
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            model_filename = f"/model_epoch-{epoch}.pt"
            log_filename = f"/log.pkl"

            torch.save(ac.state_dict(), log_path + model_filename)
            with open(log_path + log_filename, "wb") as pickle_file:
                pickle.dump(
                    {
                        "run params": run_params,
                        "step log": step_log,
                        "episode log": episode_log,
                    },
                    pickle_file,
                )

            first_plot = update_plot(first_plot)
        #     # Save model
        #     if use_logger:
        #         if (epoch % save_freq == 0) or (epoch == epochs):
        #             logger.save_state({"env": env}, None)

        #     # Test the performance of the deterministic version of the agent.
        #     test_agent()

        #     # Log info about epoch
        #     if use_logger:
        #         logger.log_tabular("Epoch", epoch)
        #         logger.log_tabular("EpRet", with_min_and_max=True)
        #         logger.log_tabular("TestEpRet", with_min_and_max=True)
        #         logger.log_tabular("EpLen", average_only=True)
        #         logger.log_tabular("TestEpLen", average_only=True)
        #         logger.log_tabular("TotalEnvInteracts", t)
        #         logger.log_tabular("Q1Vals", with_min_and_max=True)
        #         logger.log_tabular("Q2Vals", with_min_and_max=True)
        #         logger.log_tabular("LogPi", with_min_and_max=True)
        #         logger.log_tabular("LossPi", average_only=True)
        #         logger.log_tabular("LossQ", average_only=True)
        #         logger.log_tabular("Time", time.time() - start_time)
        #         logger.dump_tabular()
    return ac, step_log, episode_log


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v2")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", "-s", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--exp_name", type=str, default="sac")
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac(
        lambda: gym.make(args.env),
        actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
        gamma=args.gamma,
        seed=args.seed,
        epochs=args.epochs,
        logger_kwargs=logger_kwargs,
    )
