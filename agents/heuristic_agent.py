import random
import numpy as np
from foraging import Agent
from foraging.environment import Action


class HeuristicAgent(Agent):
	name = "Heuristic Agent"

	def _center_of_agents(self, agents):
		coords = np.array([agent.position for agent in agents])
		return np.rint(coords.mean(axis=0))

	def _move_towards(self, target, allowed):

		y, x = self.observed_position
		r, c = target

		if r < y and Action.NORTH in allowed:
			return Action.NORTH
		elif r > y and Action.SOUTH in allowed:
			return Action.SOUTH
		elif c > x and Action.EAST in allowed:
			return Action.EAST
		elif c < x and Action.WEST in allowed:
			return Action.WEST
		else:
			raise ValueError("No simple path found")

	def step(self, obs):
		raise NotImplemented("Heuristic agent is implemented by H1-H4")


class H1(HeuristicAgent):
	"""
	H1 agent always goes to the closest food
	"""
	name = "H1"

	def step(self, obs):
		try:
			r, c = self._closest_food(obs)
		except TypeError:
			return random.choice(obs.actions)
		y, x = self.observed_position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		try:
			return self._move_towards((r, c), obs.actions)
		except ValueError:
			return random.choice(obs.actions)


class H2(HeuristicAgent):
	"""
	H2 Agent goes to the one visible food which is closest to the centre of visible players
	"""
	name = "H2"

	def step(self, obs):

		agents_center = self._center_of_agents(obs.agents)

		try:
			r, c = self._closest_food(obs, None, agents_center)
		except TypeError:
			return random.choice(obs.actions)
		y, x = self.observed_position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		try:
			return self._move_towards((r, c), obs.actions)
		except ValueError:
			return random.choice(obs.actions)


class H3(HeuristicAgent):
	"""
	H3 Agent always goes to the closest food with compatible level
	"""
	name = "H3"

	def step(self, obs):

		try:
			r, c = self._closest_food(obs, self.level)
		except TypeError:
			return random.choice(obs.actions)
		y, x = self.observed_position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		try:
			return self._move_towards((r, c), obs.actions)
		except ValueError:
			return random.choice(obs.actions)


class H4(HeuristicAgent):
	"""
	H4 Agent goes to the one visible food which is closest to all visible players
	 such that the sum of their and H4's level is sufficient to load the food
	"""
	name = "H4"

	def step(self, obs):

		agents_center = self._center_of_agents(obs.agents)
		agents_sum_level = sum([a.level for a in obs.agents])

		try:
			r, c = self._closest_food(obs, agents_sum_level, agents_center)
		except TypeError:
			return random.choice(obs.actions)
		y, x = self.observed_position

		if (abs(r - y) + abs(c - x)) == 1:
			return Action.LOAD

		try:
			return self._move_towards((r, c), obs.actions)
		except ValueError:
			return random.choice(obs.actions)
