from numba import jit, njit
import numpy as np
from numpy import cos, sin, pi, ones, zeros


_2pi = 2 * pi

randn = np.random.randn
rand = np.random.rand

np.random.seed(123)

N_agents = 20


def intitialState():
    return (rand(2, N_agents) * 10 - 5, rand(N_agents) * (2 * pi), ones(N_agents))


steps_per_turn = 120

dt = 1 / steps_per_turn

min_dist_per_turn = 1
max_dist_per_turn = 4

max_yaw_per_turn = (1 / 2) * pi

actionSpaceBounds = np.array(
    [[min_dist_per_turn, max_dist_per_turn], [-max_yaw_per_turn, max_yaw_per_turn]]
)

deg = pi * 1 / 180

hit_angle = 45 / 2 * deg
hit_dist = 1.5

turns_being_hit_to_die = 0.3
damage_per_hit = (1 / steps_per_turn) / turns_being_hit_to_die
damage_for_distance = 0.25 * damage_per_hit

field_radius_squared = 10 ** 2
wall_distance = 10

death_penalty = -20
damage_penalty = -5.1 * damage_per_hit
hitting_bonus = 0.1 * damage_per_hit
kill_bonus = 10


def randomActions(N):
    # (distance, yaw)
    return np.vstack(
        [
            min_dist_per_turn + rand(N) * (max_dist_per_turn - min_dist_per_turn),
            (rand(N) - 0.5) * 2 * max_yaw_per_turn,
        ]
    )


@jit(nopython=True)
def smallestAngle(a, b):
    # works for angles withing 2pi of each other
    c = a - b if a > b else b - a
    return c if c < pi else _2pi - c


@jit(nopython=True)
def envStep(pos, heading, actions):
    action_dist = actions[0, :]
    action_yaw = actions[1, :]
    # # no edge handling
    # heading_next = np.mod(action_yaw / steps_per_turn + heading, _2pi)

    # bounce off of walls
    heading_next = np.zeros_like(heading)
    for i in range(len(heading)):
        h = heading[i]
        if np.abs(pos[0, i]) > wall_distance:
            h = np.arctan2(sin(h), -cos(h))
        if np.abs(pos[1, i]) > wall_distance:
            h = np.arctan2(-sin(h), cos(h))
        heading_next[i] = np.mod(action_yaw[i] / steps_per_turn + h, _2pi)

    pos_next = pos + (
        np.vstack((cos(heading_next), sin(heading_next)))
        * (action_dist / steps_per_turn)
    )

    return pos_next, heading_next


@jit(nopython=True)
def anglesWithinAlpha(a, b, alpha):
    return smallestAngle(a, b) < alpha


@jit(nopython=True)
def handleHitDamageAndReward_(j, i, health, reward):
    # for j hitting i
    health[i] -= damage_per_hit
    reward[i] += damage_penalty
    reward[j] += hitting_bonus
    if health[i] <= 0:
        health[i] = 0
        reward[i] += death_penalty
        reward[j] += kill_bonus


@jit(nopython=True)
def checkHits(pos, heading, health, reward):
    N = pos.shape[1]
    hit_matrix = np.zeros((N, N))
    healthOut = health.copy()
    for i in range(N):
        if healthOut[i] <= 0:
            continue

        # # take damage if too far from origin
        # if pos[0, i] ** 2 + pos[1, i] ** 2 > field_radius_squared:
        #     # take damage if you agent is too far from origin
        #     healthOut[i] -= damage_for_distance
        #     if healthOut[i] <= 0:
        #         reward[i] += death_penalty
        #         healthOut[i] = 0
        #     reward[i] += damage_penalty

        for j in range(i + 1, N):
            if healthOut[j] <= 0:
                continue
            # x and y components of the vect from j to i
            x = pos[0, i] - pos[0, j]
            y = pos[1, i] - pos[1, j]
            if np.hypot(x, y) < hit_dist:
                # the points are close enough, so see if either
                # agent is pointing in a close enough heading to the
                # angle from j to i; for i this will be rotated by 180deg
                angle = np.arctan2(y, x) + pi

                if anglesWithinAlpha(heading[j], np.mod(angle + pi, _2pi), hit_angle):
                    # j hits i
                    hit_matrix[j, i] = 1
                    handleHitDamageAndReward_(j, i, healthOut, reward)

                if anglesWithinAlpha(heading[i], angle, hit_angle):
                    # i hits j
                    hit_matrix[i, j] = 1
                    handleHitDamageAndReward_(i, j, healthOut, reward)

    # for i in range(N):
    #     if healthOut[i] <= 0:
    #         healthOut[i] = 0

    return hit_matrix, healthOut


@jit(nopython=True)
def doTurn(game_state, actions):
    # game_state: : tuple of (position array, heading array)
    # actions: tuple of (distances array , yaw change array)
    pos_next, heading_next, health_next = game_state
    N = pos_next.shape[1]

    # init containers
    positions = np.zeros((2, N, steps_per_turn))
    headings = np.zeros((N, steps_per_turn))
    health = np.zeros((N, steps_per_turn))
    hits = np.zeros((N, N, steps_per_turn))

    # this is mutated cumulatively for the whole turn.
    reward = np.zeros((N, 1))

    for t in range(0, steps_per_turn):
        pos_next, heading_next = envStep(pos_next, heading_next, actions)
        hits_next, health_next = checkHits(pos_next, heading_next, health_next, reward)

        positions[:, :, t] = pos_next
        headings[:, t] = heading_next
        health[:, t] = health_next
        hits[:, :, t] = hits_next

    return positions, headings, health, hits, reward


@njit
def rescaleColumns(X, A, B):
    # rescale the columns of X, in which the column vectors
    # should have values in intervals given
    # by the rows of A, to the proportional values
    # in interval given by rows of B
    A_0col = A[:, 0:1]
    B_0col = B[:, 0:1]
    return (X - A_0col) / (A[:, 1:2] - A_0col) * (B[:, 1:2] - B_0col) + B_0col


class GameEnv:
    def __init__(self, **kwargs):
        self.__KWARGS = kwargs
        self._init(**self.__KWARGS)
        # print("init game env now", kwargs)

    def _init(self, N_agents=8, enemy_type="random", seed=0):
        self.seed = seed
        np.random.seed(seed)
        self.N_agents = N_agents
        self.enemy_type = enemy_type

        self.latestPositions = (rand(2, N_agents) - 0.5) * 2 * wall_distance
        self.latestHeadings = rand(N_agents) * (2 * pi)
        self.latestHealth = ones(N_agents)

        self.positions = zeros((2, N_agents, 0))
        self.headings = zeros((N_agents, 0))
        self.health = zeros((N_agents, 0))
        self.hits = zeros((N_agents, N_agents, 0))
        self.rewards = zeros((N_agents, 0))

        self.saved_states = zeros((self.getStateDimension(), 0))
        self.saved_actions = zeros((2, N_agents, 0))

        self.turnsSoFar = 0

        self.observation_space = np.zeros_like(self.getLatestTurnEndStateVector())
        self.action_space = actionSpaceBounds

    def pickDefaultActions(self):
        N = self.N_agents
        if self.enemy_type == "random":
            return randomActions(N)
        if self.enemy_type == "stationary":
            return np.vstack([np.zeros(N), np.zeros(N)])
        if self.enemy_type == "straight":
            return np.vstack([ones(N), np.zeros(N)])
        print("WARNING -- unknown enemy type given:", self.enemy_type)
        return randomActions(N)

    def getAgentRewardsForActions(
        self, agent, action_options, all_actions, state_tup=None
    ):

        if state_tup is None:
            state_tup = self.getLatestTurnEndStateTup()
        rewards = []
        for agent_action in action_options:
            all_actions[:, agent] = agent_action
            _, _, _, _, reward = self.act(all_actions, state_tup)
            rewards.append(reward[agent, 0])
        return np.array(rewards)

    def pickBestAgentRewardsForActions(
        self, agent, action_options, all_actions, state_tup=None
    ):
        # ensure that the same agent takes the same action at the same time
        # (other factors being equal)
        np.random.seed(
            int((132 + agent) * (137 + self.turnsSoFar) * (541 + self.seed) % 1e8)
        )

        if state_tup is None:
            state_tup = self.getLatestTurnEndStateTup()
        rewards = self.getAgentRewardsForActions(agent, action_options, all_actions)
        i = np.argmax(rewards)
        bestAction = action_options[i]
        # if no action produces >0 reward, choose from among
        # action_options that produce 0 reward
        if rewards[i] == 0:
            zeroActions = [a for j, a in enumerate(action_options) if rewards[i] == 0]
            bestAction = zeroActions[np.random.randint(0, len(zeroActions))]
        return bestAction, rewards[i]

    def act(self, actions, state_tup=None):
        if state_tup is None:
            state_tup = self.getLatestTurnEndStateTup()
        """
        agent_0_action elements are always in [-1,1],
        and need to be rescaled for the game
        """

        scaledAction = rescaleColumns(
            actions, np.array([[-1, 1], [-1, 1]]), self.getActionSpaceBounds(),
        )

        return doTurn(state_tup, scaledAction)

    def step_actions_for_all(self, actions):
        """
        agent_0_action elements are always in [-1,1],
        and need to be rescaled for the game
        """
        # print("self.saved_states", self.saved_states.shape)
        # print(
        #     "self.getLatestTurnEndStateVector",
        #     self.getLatestTurnEndStateVector().reshape(-1, 1).shape,
        # )
        self.saved_states = np.concatenate(
            [self.saved_states, self.getLatestTurnEndStateVector().reshape(-1, 1)], 1
        )
        self.saved_actions = np.concatenate(
            [self.saved_actions, actions.reshape(2, self.N_agents, 1)], 2
        )
        positions, headings, health, hits, reward = self.act(actions)

        self.latestPositions = positions[:, :, -1]
        self.latestHeadings = headings[:, -1]
        self.latestHealth = health[:, -1]

        self.positions = np.concatenate([self.positions, positions], 2)
        self.headings = np.concatenate([self.headings, headings], 1)
        self.health = np.concatenate([self.health, health], 1)
        self.hits = np.concatenate([self.hits, hits], 2)
        self.rewards = np.concatenate([self.rewards, reward], 1)

        done = health[0, -1] <= 0 or np.all(health[1:, -1] <= 0)

        next_state = self.getLatestTurnEndStateVector()

        self.turnsSoFar += 1

        return next_state, reward[0, 0], done, None

    # NOTE: agent0 version, deprecated
    # def step(self, action):
    #     """
    #     agent_0_action elements are always in [-1,1],
    #     and need to be clipped rescaled for the game
    #     """
    #     actions = self.pickDefaultActions()

    #     actions[:, 0] = np.clip(action, -1, 1)

    #     return self.step_actions_for_all(actions)

    def getLatestTurnEndStateTup(self):
        return self.latestPositions, self.latestHeadings, self.latestHealth

    def getLatestTurnEndStateVector(self):
        pos, heading, health = self.getLatestTurnEndStateTup()
        return np.hstack((np.ravel(pos), heading, health))

    def getDeathTimes(self):
        mask = self.health == 0
        return np.where(mask.any(axis=1), mask.argmax(axis=1), np.NaN)

    def getTurnDataTup(self):
        return self.positions, self.headings, self.health, self.hits, self.rewards

    def reset(self, positions=None, headings=None, health=None):
        self._init(**self.__KWARGS)
        if positions is not None:
            self.latestPositions = positions
        if headings is not None:
            self.latestHeadings = headings
        if health is not None:
            self.latestHealth = health
        return self.getLatestTurnEndStateVector()

    def getStateDimension(self):
        return self.getLatestTurnEndStateVector().shape[0]

    def getActionSpaceBounds(self):
        return actionSpaceBounds
