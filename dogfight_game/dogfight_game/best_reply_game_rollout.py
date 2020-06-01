import numpy as np
from .game_model import GameEnv


def best_reply_game_rollout(
    N_agents,
    time_steps,
    action_options,
    random_seed_initial_conditions,
    random_seed_dynamics,
    agent0_2ndReply,
    agent0_lookahead,
):
    env = GameEnv(
        N_agents=N_agents,
        enemy_type="straight",
        seed=int(random_seed_initial_conditions % 1e8),
    )

    for i in range(time_steps):
        np.random.seed(random_seed_dynamics)

        default_actions = env.pickDefaultActions()
        actions = np.zeros_like(default_actions)
        # all agents chooses the best reply to all others taking the default action
        for agent in range(actions.shape[1]):
            action, bestReward = env.pickBestAgentRewardsForActions(
                agent, action_options, default_actions
            )

            actions[:, agent] = action

        if agent0_2ndReply and not agent0_lookahead:
            # agent 0 chooses the best reply to the others' best reply
            action, bestReward = env.pickBestAgentRewardsForActions(
                0, action_options, actions
            )
            actions[:, 0] = action

        if agent0_lookahead:
            next_actions = env.pickDefaultActions()
            # this will end up with a random action if nothing gives reward > 0
            best_action_now = action_options[np.random.randint(0, len(action_options))]
            best_reward = -np.inf
            for i, a1 in enumerate(action_options):
                actions[:, 0] = a1
                positions, headings, health, hits, r1 = env.act(actions)
                for j, a2 in enumerate(action_options):
                    next_actions[:, 0] = a2
                    state_tup = (positions[:, :, -1], headings[:, -1], health[:, -1])
                    _, _, _, _, r2 = env.act(next_actions, state_tup)
                    if r1[0, 0] + r2[0, 0] > best_reward:
                        best_reward = r1[0, 0] + r2[0, 0]
                        best_action_now = a1
            actions[:, 0] = best_action_now

        next_state, reward, done, _ = env.step_actions_for_all(actions)
        if done:
            return env
    return env
