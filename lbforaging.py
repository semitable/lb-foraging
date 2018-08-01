import argparse
import time
from collections import defaultdict
from tqdm import tqdm
import numpy as np

from agents import H1, H2, H3, H4, QAgent
from foraging import Env

_MAX_STEPS = 100


def _game_loop(env, render):
	obs = env.reset()

	for _ in range(_MAX_STEPS):
		actions = []
		for i, agent in enumerate(env.agents):
			actions.append(agent._step(obs[i]))
		obs = env.step(actions)

		if render:
			time.sleep(1)
			env.render()

		if env.game_over:
			break

	# loop once more so that agents record the game over event
	for i, agent in enumerate(env.agents):
		agent._step(obs[i])


def main(game_count=1, render=False):
	env = Env(agents=(H1, H2, H3, H4, QAgent), max_agent_level=4, field_size=(8, 8), max_food=8, sight=5)

	efficiency = defaultdict(list)

	for _ in tqdm(range(game_count)):
		_game_loop(env, render)

		for agent in env.agents:
			efficiency[agent.name].append(agent.score / env.current_step)

		env.reset()

	for k, v in efficiency.items():
		print("Agent: {} - Efficiency: {}".format(k, np.mean(v)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Play the level foraging game.')

	parser.add_argument('--render', action='store_true')
	parser.add_argument('--times', type=int, default=1, help='How many times to run the game')

	args = parser.parse_args()
	main(args.times, args.render)
