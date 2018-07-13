from foraging import Agent
from foraging.environment import Action
import random

class RushAgent(Agent):
	def step(self, obs):
		try:
			r, c = self._closest_food(obs, self.level)
		except TypeError:
			return Action.NONE

		y, x = self.position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		if r < y and Action.NORTH in obs.actions:
			return Action.NORTH
		elif r > y and Action.SOUTH in obs.actions:
			return Action.SOUTH
		elif c > x and Action.EAST in obs.actions:
			return Action.EAST
		elif c < x and Action.WEST in obs.actions:
			return Action.WEST
		else:
			return random.choice(obs.actions)
