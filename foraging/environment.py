from collections import namedtuple, defaultdict
from enum import Enum
from random import randint

import numpy as np
import pygame

# Define some colors
_BLACK = (0, 0, 0)
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (255, 0, 0)


class Action(Enum):
	NONE = 0
	NORTH = N = 1
	SOUTH = S = 2
	WEST = W = 3
	EAST = E = 4
	LOAD = 5


class Env:
	"""
	A class that contains rules/actions for the game level-based foraging.
	"""

	action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
	Observation = namedtuple("Observation", ['field', 'actions', 'agents', 'game_over'])
	AgentObservation = namedtuple("AgentObservation", ['position', 'level'])

	def __init__(self, agents, max_agent_level, field_size, max_food, max_food_level):
		self.agent_classes = agents
		self.agents = []
		self.field = np.zeros(field_size, np.int32)

		self.max_food = max_food
		self.max_food_level = max_food_level
		self.max_agent_level = max_agent_level
		self._game_over = None

		self._rendering_initialized = False

	@property
	def field_size(self):
		return self.field.shape

	@property
	def rows(self):
		return self.field_size[0]

	@property
	def cols(self):
		return self.field_size[1]

	@property
	def game_over(self):
		return self._game_over

	def neighborhood(self, row, col, distance=1):

		return self.field[
			   max(row - distance, 0):min(row + distance + 1, self.rows),
			   max(col - distance, 0):min(col + distance + 1, self.cols)
			   ]

	def adjacent_food(self, row, col):
		return (self.field[max(row - 1, 0):min(row + 2, self.rows), col].sum()
				+ self.field[row, max(col - 1, 0):min(col + 2, self.cols)].sum()
				)

	def adjacent_food_location(self, row, col):
		if row > 1 and self.field[row - 1, col] > 0:
			return row - 1, col
		elif row < self.rows - 1 and self.field[row + 1, col] > 0:
			return row + 1, col
		elif col > 1 and self.field[row, col - 1] > 0:
			return row, col - 1
		elif col < self.cols - 1 and self.field[row, col + 1] > 0:
			return row, col + 1

	def adjacent_agents(self, row, col):
		return [agent for agent in self.agents if
				abs(agent.position[0] - row) == 1 and agent.position[1] == col or
				abs(agent.position[1] - col) == 1 and agent.position[0] == row
				]

	def spawn_food(self, max_food, max_level):

		food_count = 0
		attempts = 0

		while food_count < max_food and attempts < 10000:
			attempts += 1
			row = randint(1, self.rows - 2)
			col = randint(1, self.cols - 2)

			# check if it has neighbors:
			if self.neighborhood(row, col, distance=2).sum() > 0:
				continue

			self.field[row, col] = randint(1, max_level)
			food_count += 1

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
		if action == Action.NONE:
			return True
		if action == Action.NORTH:
			return agent.position[0] > 0 and self.field[agent.position[0] - 1, agent.position[1]] == 0
		elif action == Action.SOUTH:
			return agent.position[0] < self.rows - 1 and self.field[agent.position[0] + 1, agent.position[1]] == 0
		elif action == Action.WEST:
			return agent.position[1] > 0 and self.field[agent.position[0], agent.position[1] - 1] == 0
		elif action == Action.EAST:
			return agent.position[1] < self.cols - 1 and self.field[agent.position[0], agent.position[1] + 1] == 0
		elif action == Action.LOAD:
			return self.adjacent_food(*agent.position) > 0

		raise ValueError("Undefined action")

	def _make_obs(self, agent):
		return self.Observation(
			actions=[action for action in Action if self._is_valid_action(agent, action)],
			agents=[self.AgentObservation(position=a.position, level=a.level) for a in self.agents],
			field=np.copy(self.field),
			game_over=self.game_over
		)

	def reset(self):
		self.field = np.zeros(self.field_size, np.int32)
		self.spawn_food(self.max_food, self.max_food_level)
		self.spawn_agents(self.max_agent_level)
		self.current_step = 0
		self._game_over = False

		return [self._make_obs(agent) for agent in self.agents]

	def step(self, actions):
		self.current_step += 1

		# check if actions are valid
		for agent, action in zip(self.agents, actions):
			if not self._is_valid_action(agent, action):
				raise ValueError("Invalid action attempted")

		loading_agents = set()

		# move agents
		# if two or more agents try to move to the same location they all fail
		collisions = defaultdict(list)

		# so check for collisions
		for agent, action in zip(self.agents, actions):
			if action == Action.NONE:
				collisions[agent.position].append(agent)
			elif action == Action.NORTH:
				collisions[(agent.position[0] - 1, agent.position[1])].append(agent)
			elif action == Action.SOUTH:
				collisions[(agent.position[0] + 1, agent.position[1])].append(agent)
			elif action == Action.WEST:
				collisions[(agent.position[0], agent.position[1] - 1)].append(agent)
			elif action == Action.EAST:
				collisions[(agent.position[0], agent.position[1] + 1)].append(agent)
			elif action == Action.LOAD:
				collisions[agent.position].append(agent)
				loading_agents.add(agent)

		# and do movements for non colliding agents

		for k, v in collisions.items():
			if len(v) > 1:  # make sure no more than an agent will arrive at location
				continue
			v[0].position = k

		# finally process the loadings:
		while loading_agents:
			# find adjacent food
			agent = loading_agents.pop()
			frow, fcol = self.adjacent_food_location(*agent.position)
			food = self.field[frow, fcol]

			adj_agents = self.adjacent_agents(frow, fcol)

			adj_agent_level = sum([a.level for a in adj_agents])

			loading_agents = loading_agents - set(adj_agents)

			if adj_agent_level < food:
				# failed to load
				continue

			# else the food was loaded and each agent scores points
			for a in adj_agents:
				a.score += food
			# and the food is removed
			self.field[frow, fcol] = 0

		self._game_over = self.field.sum() == 0

		return [self._make_obs(agent) for agent in self.agents]

	def _init_render(self):

		self.grid_size = 50
		self.font_size = 20
		pygame.init()
		self._screen = pygame.display.set_mode(
			(self.cols * self.grid_size + 1,
			 self.rows * self.grid_size + 1)
		)
		self._font = pygame.font.SysFont("monospace", self.font_size)
		self._rendering_initialized = True

	def _draw_grid(self):
		for r in range(self.rows + 1):
			pygame.draw.line(self._screen, _WHITE, (0, self.grid_size * r),
							 (self.grid_size * self.cols, self.grid_size * r))
		for c in range(self.cols + 1):
			pygame.draw.line(self._screen, _WHITE, (self.grid_size * c, 0),
							 (self.grid_size * c, self.grid_size * self.rows))

	def _draw_food(self):
		for r in range(self.rows):
			for c in range(self.cols):
				if self._is_empty_location(r, c):
					pass
				elif self.field[r, c] != 0:
					self._screen.blit(
						self._font.render(str(self.field[r, c]), 1, _GREEN),
						(self.grid_size * c + self.grid_size // 3, self.grid_size * r + self.grid_size // 3)
					)

	def _draw_agents(self):
		for agent in self.agents:
			r, c = agent.position
			self._screen.blit(
				self._font.render(str(agent.level), 1, _RED),
				(self.grid_size * c + self.grid_size // 3, self.grid_size * r + self.grid_size // 3)
			)

	def render(self):
		if not self._rendering_initialized:
			self._init_render()

		self._screen.fill(_BLACK)
		self._draw_grid()
		self._draw_food()
		self._draw_agents()

		pygame.display.flip()
