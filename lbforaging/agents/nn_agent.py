import random

from ..foraging import Agent


class NNAgent(Agent):
    name = "NN Agent"

    def _act(self, obs):
        return random.choice(obs.actions)
