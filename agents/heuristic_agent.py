import random

from foraging import Agent
from foraging.environment import Action


class HeuristicAgent(Agent):
	name = "Heuristic Agent"

	def _move_towards(self, target, allowed):

		y, x = self.observed_position
		r, c = target

		if r < y and Action.NORTH in allowed:
			return Action.NORTH
		elif r > y and Action.SOUTH in allowed:
			return Action.SOUTH
		elif c > x and Action.EAST in allowed:
			return Action.EAST
		elif c < x and Action.WEST in allowed:
			return Action.WEST
		else:
			raise ValueError("No simple path found")

	def step(self, obs):
		try:
			r, c = self._closest_food(obs, self.level)
		except TypeError:
			return random.choice(obs.actions)

		y, x = self.observed_position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		try:
			return self._move_towards((r, c), obs.actions)
		except ValueError:
			return random.choice(obs.actions)
