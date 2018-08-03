import argparse
import random
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from agents import H1, H2, H3, H4, QAgent
from foraging import Env, Player

_MAX_STEPS = 100


class TypeSpacePlayer(Player):
	"""
	Changes the controller (Agent) every change_every() times to a random type_space
	"""

	def __init__(self, type_space, change_every):
		self.type_space = type_space
		self.change_every = change_every
		self.next_change = 0
		super().__init__()

	def step(self, obs):
		if self.next_change <= 0:
			self.set_controller(random.choice(self.type_space)(self))
			self.next_change = self.change_every()

		self.next_change -= 1
		return super().step(obs)

	@property
	def name(self):
		return str([a.name for a in self.type_space])


def _game_loop(env, render):
	"""
	"""
	obs = env.reset()

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


def evaluate(players, game_count, render, max_agent_level=5, field_size=(8, 8), food_count=5, sight=None):
	if sight == None:
		sight = max(*field_size)

	env = Env(players=players, max_player_level=4, field_size=(8, 8), max_food=5, sight=sight)

	efficiency = 0
	flexibility = 0

	pbar = tqdm if game_count > 1 else (lambda x: x)  # use tqdm for game_count >1

	for _ in pbar(range(game_count)):
		_game_loop(env, render)

		for player in env.players:
			efficiency += player.score / env.current_step
			if env.current_step < _MAX_STEPS:
				flexibility += 1


	print("{} - Efficiency: {} - Flexibility: {}".format(players[0].name, np.mean(efficiency), np.mean(flexibility)))


def main(game_count=1, render=False):
	p0 = Player()
	p0.set_controller(QAgent(p0))

	p1 = TypeSpacePlayer([H1, H2, H3, H4], lambda: random.randint(10, 20))

	evaluate([p0, p1], game_count, render)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Play the level foraging game.')

	parser.add_argument('--render', action='store_true')
	parser.add_argument('--times', type=int, default=1, help='How many times to run the game')

	args = parser.parse_args()
	main(args.times, args.render)
