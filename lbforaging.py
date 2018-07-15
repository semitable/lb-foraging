import argparse
import time
from collections import defaultdict
from agents import HeuristicAgent, RandomAgent
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


def main(game_count=1, render=False):
	env = Env(agents=(HeuristicAgent, RandomAgent), max_agent_level=4, field_size=(12, 8), max_food=8,
			  max_food_level=4, sight=5)

	scores = defaultdict(lambda:0)

	for _ in range(game_count):
		_game_loop(env, render)

		for agent in env.agents:
			scores[agent.name] += agent.score

		env.reset()

	for k, v in scores.items():
		print("Agent: {} - Score: {}".format(k, v))



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Play the level foraging game.')

	parser.add_argument('--render', action='store_true')
	parser.add_argument('--times', type=int, default=1, help='How many times to run the game')

	args = parser.parse_args()
	main(args.times, args.render)
