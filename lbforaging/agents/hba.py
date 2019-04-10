from . import QAgent
from foraging import Env
import random
import numpy as np
from agents import H1, H2, H3, H4
from itertools import product
from collections import defaultdict
from functools import reduce
import operator


class HBAAgent(QAgent):
    name = "HBA"

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.type_space = [H1, H2, H3, H4]

        N = 2  # todo N is number of agents
        D = len(self.type_space)

        self.prev_likelihood = []
        self.belief = np.ones((N, D)) / D
        self.prev_obs = None

    def gtw(self, a, b, c):
        def f(x):
            return max(0, a - b * (x - 1) ** c)

        return f

    def choose_action(self, state, obs):
        player_no = next((i for i, item in enumerate(obs.players) if item.is_self))
        self.Q.check_state_exist(state)

        env = Env.from_obs(obs)
        moves = self.generate_typespace_moves(env)

        actions = []
        beliefs = []
        for i, p in enumerate(obs.players):
            if p.is_self:
                actions.append(obs.actions)
                beliefs.append(np.ones(len(self.type_space)))
            else:
                actions.append(moves[i, :])
                beliefs.append(self.belief[i, :])
        joint_actions = list(product(*actions))
        probs = list(product(*beliefs))

        exppay = defaultdict(int)

        for action, prob in zip(joint_actions, probs):
            v = self.Q.q_table.at[state, action] * reduce(operator.mul, prob)
            exppay[action[player_no]] += v

        return max(exppay.items(), key=operator.itemgetter(1))[0]

    def step(self, obs):
        if obs.current_step > 0:
            self.update_belief(obs)
        val = super().step(obs)
        self.prev_obs = obs
        return val

    def generate_typespace_moves(self, env, exclude_player=None):

        moves = np.empty((len(env.players), len(self.type_space)), dtype=object)
        for i, player in enumerate(env.players):
            # if i == exclude_player:  # todo this player can be excluded (because it's us)
            #     continue
            obs = env._make_obs(env.players[i])
            for j, t in enumerate(self.type_space):
                agent = t(env.players[i])
                action = agent._step(obs)
                moves[i, j] = action
        return moves

    def update_belief(self, obs):
        player_no = next((i for i, item in enumerate(obs.players) if item.is_self))

        env = Env.from_obs(self.prev_obs)
        moves = self.generate_typespace_moves(env, player_no)
        truth = np.array(
            [[p.history[-1] for p in obs.players]] * len(self.type_space)
        ).T

        likelihood = np.equal(truth, moves).astype(float)
        likelihood[likelihood == 0] = 0.01

        self.prev_likelihood.insert(0, likelihood)
        if len(self.prev_likelihood) > 15:
            self.prev_likelihood.pop()

        gtw = self.gtw(10, 0.05, 3)

        for i, p in enumerate(obs.players):
            if i == player_no:
                continue
            L = np.zeros(len(self.type_space))
            for t, l in enumerate(self.prev_likelihood):
                L += l[i] * gtw(t)

            self.belief[i, :] = self.belief[i, :] * L
            self.belief[i, :] = self.belief[i, :] / sum(self.belief[i, :])

    def expand(self, obs, depth):
        player_no = next((i for i, item in enumerate(obs.players) if item.is_self))

        env = Env.from_obs(obs)

        observations = [env._make_obs(p) for p in env.players]

        for i, player in enumerate(env.players):
            if i == player_no:
                continue  # we will control this player ourselves
            else:
                likely_type = self.type_space[np.argmax(self.belief[i, :])]
                player.set_controller(likely_type(player))

        for _ in range(depth):
            actions = []

            for i, player in enumerate(env.players):

                if i == player_no:
                    if random.random() > self.e_2:
                        action = self.Q.choose_action(
                            self._make_state(observations[i])
                        )[player_no]
                    else:
                        action = random.choice(observations[i].actions)
                else:
                    action = player.step(observations[i])

                # make sure the action is valid (if not replace with random action):
                action = (
                    action
                    if action in observations[i].actions
                    else random.choice(observations[i].actions)
                )
                actions.append(action)

            prev_state = self._make_state(observations[player_no])
            joint_action = tuple(actions)

            past_score = observations[player_no].players[player_no].score

            observations = env.step(actions)  # perform the joint action

            reward = observations[player_no].players[player_no].score - past_score

            state = self._make_state(observations[player_no])

            self.Q.learn(prev_state, joint_action, reward, state)
            # import time
            # env.render("EXPANSION STAGE")
            # time.sleep(0.4)
            # input()

            if env.game_over:
                break
