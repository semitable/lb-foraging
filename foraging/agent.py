import numpy as np
_MAX_INT = 999999

class Agent:
	def __init__(self, position, level):
		self.visibility = _MAX_INT
		self.position = position
		self.level = level

	def step(self, obs):
		raise NotImplemented("You must implement an agent")

	def _closest_food(self, obs):
		x, y = self.position
		r, c = np.nonzero(obs.field)
		min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
		return r[min_idx], c[min_idx]