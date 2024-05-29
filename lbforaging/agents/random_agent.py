import random

from lbforaging.foraging import Agent


class RandomAgent(Agent):
    name = "Random Agent"

    def _act(self, obs):
        return random.choice(obs.actions)
