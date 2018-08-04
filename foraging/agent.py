import logging
from . import Env, Player
import numpy as np
from copy import deepcopy
from itertools import chain

_MAX_INT = 999999


class Agent:
	name = "Prototype Agent"

	def __repr__(self):
		return self.name

	def __init__(self, player):
		self.logger = logging.getLogger(__name__)
		self.player = player

	def __getattr__(self, item):
		return getattr(self.player, item)

	def _step(self, obs):
		self.observed_position = next((x for x in obs.players if x.is_self), None).position

		#saves the action to the history
		action = self.step(obs)
		self.history.append(action)

		return action

	def step(self, obs):
		raise NotImplemented("You must implement an agent")

	def _closest_food(self, obs, max_food_level=None, start=None):

		if start is None:
			x, y = self.observed_position
		else:
			x, y = start

		field = np.copy(obs.field)

		if max_food_level:
			field[field > max_food_level] = 0

		r, c = np.nonzero(field)
		try:
			min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
		except ValueError:
			return None

		return r[min_idx], c[min_idx]

	def _make_state(self, obs):
		state = np.concatenate((
			obs.field.flatten(),
			[*self.observed_position, self.level],
			list(chain(*sorted(
				[(a.position[0], a.position[1], a.level) for a in obs.players if not a.is_self],
				key=lambda x: x[0]))
				 ),
		))
		return hash(tuple(state))

	def simulate(self, obs, actions):
		assert len(actions) == len(obs.players)

		player_no = next(i for i, x in enumerate(obs.players) if x.is_self)

		env = Env.from_obs(obs)
		env.step(actions)

		reward = env.players[player_no].score - self.score
		new_state = self._make_state(env._make_obs(env.players[player_no]))
		print(reward)
		return reward, new_state



	def cleanup(self):
		pass