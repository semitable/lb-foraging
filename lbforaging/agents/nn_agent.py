import random

from lbforaging.agents.agent import BaseAgent


class NNAgent(BaseAgent):
    name = "NN Agent"

    def step(self, obs):
        return random.choice(obs.actions)
