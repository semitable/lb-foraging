import random

from lbforaging.agents import BaseAgent


class RandomAgent(BaseAgent):
    name = "Random Agent"

    def step(self, obs):
        return random.choice(obs.actions)
