from collections import namedtuple
from enum import Enum
from random import randint

import numpy as np


class Action(Enum):
	NORTH = N = 0
	SOUTH = S = 1
	WEST = W = 2
	EAST = E = 3
	LOAD = 4


class Env:
	"""
	A class that contains rules/actions for the game level-based foraging.
	"""

	action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
	Observation = namedtuple("Observation", ['field', 'actions', 'agents'])
	AgentObservation = namedtuple("AgentObservation", ['position', 'level'])

	def __init__(self, agents, max_agent_level, field_size, max_food, max_food_level):
		self.agent_classes = agents
		self.agents = []
		self.field = np.zeros(field_size, np.int32)

		self.max_food = max_food
		self.max_food_level = max_food_level
		self.max_agent_level = max_agent_level

	@property
	def field_size(self):
		return self.field.shape

	@property
	def rows(self):
		return self.field_size[0]

	@property
	def cols(self):
		return self.field_size[1]

	def neighborhood(self, row, col, distance=1):

		return self.field[
			   max(row - distance, 0):min(row + distance + 1, self.rows),
			   max(col - distance, 0):min(col + distance + 1, self.cols)
			   ]

	def spawn_food(self, max_food, max_level):

		food_count = 0
		attempts = 0

		while food_count < max_food and attempts < 1000:
			row = randint(1, self.rows - 2)
			col = randint(1, self.cols - 2)

			# check if it has neighbors:
			if self.neighborhood(row, col).sum() > 0:
				attempts += 1
				continue

			self.field[row, col] = randint(1, max_level)

			food_count += 1
			attempts += 1

	def _is_empty_location(self, row, col):

		if self.field[row, col] != 0:
			return False
		for a in self.agents:
			if row == a.position[0] and col == a.position[1]:
				return False

		return True

	def spawn_agents(self, max_agent_level):
		self.agents = []

		for agent_cls in self.agent_classes:

			attempts = 0

			while attempts < 1000:
				row = randint(0, self.rows - 1)
				col = randint(0, self.cols - 1)
				if self._is_empty_location(row, col):
					agent = agent_cls((row, col), randint(1, max_agent_level))
					self.agents.append(agent)
					break
				attempts += 1

	def _is_valid_action(self, agent, action):

		if action == Action.NORTH:
			return agent.position[0] > 0 and self.field[agent.position[0] - 1, agent.position[1]] == 0
		elif action == Action.SOUTH:
			return agent.position[0] < self.rows - 1 and self.field[agent.position[0] + 1, agent.position[1]] == 0
		elif action == Action.WEST:
			return agent.position[1] > 0 and self.field[agent.position[0], agent.position[1] - 1] == 0
		elif action == Action.EAST:
			return agent.position[1] < self.cols - 1 and self.field[agent.position[0], agent.position[1] + 1] == 0
		elif action == Action.LOAD:
			return self.neighborhood(agent.position[0], agent.position[1]).sum() > 0

		raise ValueError("Undefined action")

	def _make_obs(self, agent):

		return self.Observation(
			actions=[action for action in Action if self._is_valid_action(agent, action)],
			agents=[self.AgentObservation(position=a.position, level=a.level) for a in self.agents],
			field=self.field
		)

	def reset(self):
		self.field = np.zeros(self.field_size, np.int32)
		self.spawn_food(self.max_food, self.max_food_level)
		self.spawn_agents(self.max_agent_level)

		return [self._make_obs(agent) for agent in self.agents]

	def step(self, actions):
		for agent, action in zip(self.agents, actions):
			if not self._is_valid_action(agent, action):
				raise ValueError("Invalid action attempted")

			if action == Action.NORTH:
				agent.position = (agent.position[0] - 1, agent.position[1])
			elif action == Action.SOUTH:
				agent.position = (agent.position[0] + 1, agent.position[1])
			elif action == Action.WEST:
				agent.position = (agent.position[0], agent.position[1] - 1)
			elif action == Action.EAST:
				agent.position = (agent.position[0], agent.position[1] + 1)

		return [self._make_obs(agent) for agent in self.agents]

	def render(self):
		print('==========')
		for r in range(self.rows):
			print('|', end='')
			for c in range(self.cols):
				if self._is_empty_location(r, c):
					print(' ', end='')
				elif self.field[r, c] != 0:
					print(self.field[r, c], end='')
				else:
					print('x', end='')
			print('|')
		print('==========')
