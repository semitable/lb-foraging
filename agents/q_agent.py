import os
from itertools import chain, repeat, product

import numpy as np
import pandas as pd
import random
from foraging import Agent, Env
from foraging.environment import Action
from agents import H1, H2, H3, H4

_CACHE = None


class QLearningTable:
    _DATA_FILE = 'qtable.gz'

    def __init__(self, actions):
        self.actions = actions  # a list

        self.beta = 0.2
        self.gamma = 0.9  # reward decay
        self.lambda_ = 0.9
        self.e_min = 0.1

        self.lr_w = lambda t: 1 / (1000 + t / 10)
        self.lr_l = lambda t: 2 / (1000 + t / 10)

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.e_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation, e):
        self.check_state_exist(observation)

        if np.random.uniform() <= 1 - e:
            # choose best action
            state_action = self.q_table.loc[observation, :]

            # some actions have the same value
            # state_action = state_action.reindex(np.random.permutation(state_action.index))
            max_reward = state_action.max()

            action = random.choice(state_action[state_action == max_reward].index)
        # if e==0:
        # 	print("CHOOSING BEST ACTION {} for STATE {} with R: {}".format(action, observation, max(state_action)))
        else:

            # choose random action
            action = self.actions[np.random.randint(len(self.actions))]
        # print("RANDOM ACTION", action)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        # pd.set_option('display.width', 1000)
        # print("=========")
        # print("Learning: {}".format(s))
        # print("Next State: {}".format(s_))
        # print("Reward: {}".format(r))
        # print("Action Taken: {}".format(a))

        q_predict = self.q_table.at[s, a]
        # print("Prev Prediction: ", q_predict)

        max_exp_pay = self.q_table.loc[s_, :].max()

        delta = self.beta * (r + self.gamma * max_exp_pay - q_predict)

        self.e_table.at[s, a] = 1

        eligibility = list(self.e_table[self.e_table >= self.e_min].stack().index)
        # print('Delta: ', delta)
        for sn, an in eligibility:
            cur_e = self.e_table.at[sn, an]
            new_q = self.q_table.at[sn, an] + delta * cur_e
            # print('    State: ', sn)
            # print('    Action: ', an)
            # print('    Eligibility: ', cur_e)
            # print('    New Q: ', new_q)
            # print('    -------')
            self.q_table.at[sn, an] = new_q
            self.e_table.at[sn, an] = self.lambda_ * self.e_table.at[sn, an]

    # print("-------")

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
            self.e_table = self.e_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QAgent(Agent):
    name = "Q Agent"

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

        self.Q = None
        self._prev_score = 0
        self._prev_state = None

        self.e_1 = 0
        self.e_2 = 0.2  # expand stage

    def expand(self, obs, depth):

        player_no = next((i for i, item in enumerate(obs.players) if item.is_self))

        env = Env.from_obs(obs)

        observations = [env._make_obs(p) for p in env.players]

        for i, player in enumerate(env.players):
            if i == player_no:
                continue  # we will control this player ourselves
            else:
                player.set_controller(H1(player))

        for _ in range(depth):
            actions = []

            for i, player in enumerate(env.players):

                if i == player_no:
                    rl_action = self.Q.choose_action(self._make_state(observations[i]), self.e_2)[player_no]
                    action = rl_action if rl_action in observations[i].actions else random.choice(
                        observations[i].actions)
                else:
                    action = player.step(observations[i])

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

    def step(self, obs):

        if self.Q == None:
            self.Q = QLearningTable(actions=list(product(*repeat(Action, len(obs.players)))))

        # observe current state s
        state = self._make_state(obs)

        if self.history and self._prev_state:
            reward = self.score - self._prev_score
            joint_action = tuple([p.history[-1] for p in obs.players])
            self.Q.learn(self._prev_state, joint_action, reward, state)

        if obs.game_over:
            return None

        eligibility = self.Q.e_table.copy()
        for _ in range(3):
            self.Q.e_table = eligibility.copy()
            self.expand(obs, depth=20)

        rl_action = self.Q.choose_action(state, self.e_1)[0]

        action = rl_action if rl_action in obs.actions else random.choice(obs.actions)

        self._prev_score = self.score
        self._prev_state = state

        return action
