import os
from itertools import chain

import numpy as np
import pandas as pd

from foraging import Agent
from foraging.environment import Action

_CACHE = None


class QLearningTable:
	_DATA_FILE = 'qtable.gz'

	def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
		self.actions = actions  # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		if os.path.isfile(self._DATA_FILE):
			self.load()
		else:
			self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

	def save(self):
		global _CACHE
		_CACHE = self.q_table
		self.q_table.to_pickle(self._DATA_FILE, compression='gzip')

	def load(self):
		if _CACHE is None:
			self.q_table = pd.read_pickle(self._DATA_FILE, compression='gzip')
		else:
			self.q_table = _CACHE

	def choose_action(self, observation):
		self.check_state_exist(observation)

		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.loc[observation, :]

			# some actions have the same value
			state_action = state_action.reindex(np.random.permutation(state_action.index))

			action = state_action.idxmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)

		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		self.check_state_exist(s)

		q_predict = self.q_table.ix[s, a]

		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.loc[s_, :].max()
		else:
			q_target = r  # next state is terminal
		# update
		self.q_table.loc[s, a] += self.lr * (q_target - q_predict)

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class QAgent(Agent):
	name = "Q Agent"

	def __init__(self, *kargs, **kwargs):
		super().__init__(*kargs, **kwargs)

		self.Q = QLearningTable(actions=Action)
		self._prev_score = 0
		self._prev_state = None


	def step(self, obs):
		state = self._make_state(obs)

		u, s = self.simulate(obs, [Action.NONE, Action.NONE])
		print(u, s)

		if self.history and self._prev_state:
			reward = self.score - self._prev_score
			self.Q.learn(self._prev_state, self.history[-1], reward, state)

		rl_action = self.Q.choose_action(state)
		action = rl_action if rl_action in obs.actions else Action.NONE

		self._prev_score = self.score
		self._prev_state = state

		return action

	def cleanup(self):
		self.Q.save()
