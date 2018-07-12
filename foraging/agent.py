_MAX_INT = 999999

class Agent:
	def __init__(self, position, level):
		self.visibility = _MAX_INT
		self.position = position
		self.level = level

	def step(self, obs):
		raise NotImplemented("You must implement an agent")
