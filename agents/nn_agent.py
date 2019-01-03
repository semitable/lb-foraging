from foraging import Agent, Env
import random
import tensorflow as tf


class NNAgent(Agent):
    name = "NN Agent"

    def step(self, obs):
        return random.choice(obs.actions)

