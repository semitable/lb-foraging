import random

from foraging import Agent


class RandomAgent(Agent):
	def step(self, obs):
		return random.choice(obs.actions)
