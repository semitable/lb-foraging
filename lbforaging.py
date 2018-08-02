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

	for player in env.players:
		player.set_controller(H1(player))

	for _ in range(_MAX_STEPS):
		actions = []
		for i, player in enumerate(env.players):
			actions.append(player.step(obs[i]))
		obs = env.step(actions)

		if render:
			time.sleep(1)
			env.render()

		if env.game_over:
			break

	# loop once more so that agents record the game over event
	for i, player in enumerate(env.players):
		player.step(obs[i])

def evaluate(agent, types, max_agent_level=5, field_size=(8,8), food_count=5, sight=None):

	if sight == None:
		sight = max(*field_size)
	pass


def main(game_count=1, render=False):
	env = Env(player_count=2, max_player_level=4, field_size=(8, 8), max_food=5, sight=5)

	efficiency = defaultdict(list)

	pbar = tqdm if game_count > 1 else (lambda x: x) # use tqdm for game_count >1

	for _ in pbar(range(game_count)):
		_game_loop(env, render)

		for player in env.players:
			efficiency[player.name].append(player.score / env.current_step)


	for k, v in efficiency.items():
		print("Agent: {} - Efficiency: {}".format(k, np.mean(v)))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Play the level foraging game.')

	parser.add_argument('--render', action='store_true')
	parser.add_argument('--times', type=int, default=1, help='How many times to run the game')

	args = parser.parse_args()
	main(args.times, args.render)
