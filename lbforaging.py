from collections import namedtuple
from enum import Enum
from random import randint
import random
import numpy as np

from foraging import Env
from agents import RandomAgent, RushAgent
_MAX_STEPS = 100


def main():
	env = Env(agents=(RushAgent,), max_agent_level=4, field_size=(12, 8), max_food=8, max_food_level=4)
	obs = env.reset()

	for _ in range(_MAX_STEPS):
		actions = []
		for i, agent in enumerate(env.agents):
			actions.append(agent.step(obs[i]))
		obs = env.step(actions)
		env.render()


if __name__ == '__main__':
	main()
