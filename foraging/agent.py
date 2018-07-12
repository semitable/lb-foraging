import numpy as np
_MAX_INT = 999999

class Agent:
	def __init__(self, position, level):
		self.visibility = _MAX_INT
		self.position = position
		self.level = level
		self.score = 0

	def step(self, obs):
		raise NotImplemented("You must implement an agent")

	def _closest_food(self, obs, max_food_level=0):
		x, y = self.position
		field = obs.field

		if max_food_level > 0:
			field[field > max_food_level] = 0

		r, c = np.nonzero(field)
		try:
			min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
		except ValueError:
			return None

		return r[min_idx], c[min_idx]