import random

from foraging import Agent


class RandomAgent(Agent):

	name = "Random Agent"

	def step(self, obs):
		return random.choice(obs.actions)
