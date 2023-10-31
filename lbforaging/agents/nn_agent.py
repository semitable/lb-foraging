import random

from lbforaging.foraging.agent import Agent


class NNAgent(Agent):
    name = "NN Agent"

    def step(self, obs):
        return random.choice(obs.actions)
