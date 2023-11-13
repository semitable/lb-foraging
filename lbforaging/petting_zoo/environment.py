import functools
import logging
from collections import defaultdict
from copy import copy
from enum import Enum
import gymnasium
from gymnasium.utils import seeding
import numpy as np
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils import parallel_to_aec
from PIL import ImageColor


class Action(Enum):
    NONE = 0
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    LOAD = 5


class CellEntity(Enum):
    # entity encodings for grid observations
    OUT_OF_BOUNDS = 0
    EMPTY = 1
    FOOD = 2
    AGENT = 3

def env(**kwargs):
    env = raw_env(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def parallel_env(**kwargs):
    env = ForagingEnvLite(**kwargs)
    return env

def raw_env(**kwargs):
    env = parallel_env(**kwargs)
    env = parallel_to_aec(env)
    return env

class ForagingEnvLite(ParallelEnv):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {
        "name": "lbforaging_v2",
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        }

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    def __init__(
            self,
            n_players=2,
            max_player_level=3,
            field_size=(8,8),
            max_food=3,
            sight=8,
            max_cycles=50,
            force_coop=False,
            player_levels=[],
            food_levels=[],
            agent_colors=[],
            normalize_reward=True,
            grid_observation=False,
            penalty=0.0,
            render_mode="rgb_array",
            render_style="simple",
        ):
        # TODO sight = None, etc
        self.logger = logging.getLogger(__name__)
        self.seed()

        self.possible_agents = [f"player_{i}" for i in range(n_players)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}
        self.agents = []
        self.pos = {}
        self.specified_agent_levels = defaultdict(lambda: None)
        for i, level in enumerate(player_levels):
            if i >= n_players:
                break
            self.specified_agent_levels[self.possible_agents[i]] = level
        self.agent_levels = {}
        # TODO set agent colors
        self.agent_colors = defaultdict(lambda: (0, 0, 0))
        for i, agent_color in enumerate(agent_colors):
            if i >= n_players:
                break
            if isinstance(agent_color, list) or isinstance(agent_color, tuple):
                self.agent_colors[self.possible_agents[i]] = agent_color
            else:
                self.agent_colors[self.possible_agents[i]] = ImageColor.getrgb(agent_color)


        self.field = np.zeros(field_size, np.int32)

        self.penalty = penalty
        self.max_food = max_food
        self.specified_food_levels = [None] * self.max_food
        self.specified_food_levels[:len(food_levels)] = food_levels
        self._food_spawned = 0.0
        self.max_agent_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        self._rendering_initialized = False
        self._valid_actions = None
        self._max_cycles = max_cycles

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.viewer = None
        self.render_mode = render_mode
        self.render_style = render_style

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x = self.field.shape[1]
            field_y = self.field.shape[0]
            # field_size = field_x * field_y

            max_food = self.max_food
            max_food_level = self.max_agent_level * len(self.possible_agents)

            min_obs_food = [-1, -1, 0]
            max_obs_food = [field_y-1, field_x-1, max_food_level]
            min_obs_agents = [-1, -1, 0]
            max_obs_agents = [field_y-1, field_x-1, self.max_agent_level]

            min_obs = min_obs_food * max_food + min_obs_agents * len(self.possible_agents)
            max_obs = max_obs_food * max_food + max_obs_agents * len(self.possible_agents)
        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_agent_level

            # foods layer: foods level
            max_food_level = self.max_agent_level * len(self.possible_agents)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])
        return gymnasium.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @property
    def observation_spaces(self):
        return {agent: self.observation_space(agent) for agent in self.possible_agents}

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return gymnasium.spaces.Discrete(6)

    @property
    def action_spaces(self):
        return {agent: self.action_space(agent) for agent in self.possible_agents}

    @property
    def field_size(self):
        return self.field.shape

    @property
    def field_length(self):
        return self.field.size

    @property
    def rows(self):
        return self.field_size[0]

    @property
    def cols(self):
        return self.field_size[1]

    @property
    def game_over(self):
        return self._game_over

    def _gen_valid_moves(self):
        self._valid_actions = {
            agent: [
                action for action in Action if self._is_valid_action(agent, action)
            ]
            for agent in self.agents
        }

    def _action_mask(self, agent):
        return np.array([
            1 if Action(i) in self._valid_actions[agent] else 0
            for i in range(self.action_space(agent).n)
            ], dtype=np.int8)

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def adjacent_food(self, row, col):
        return (
            self.field[max(row - 1, 0), col]
            + self.field[min(row + 1, self.rows - 1), col]
            + self.field[row, max(col - 1, 0)]
            + self.field[row, min(col + 1, self.cols - 1)]
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
        return [agent
                for agent in self.agents
                if abs(self.pos[agent][0] - row) == 1
                and self.pos[agent][1] == col
                or abs(self.pos[agent][1] - col) == 1
                and self.pos[agent][0] == row
                ]

    def spawn_food(self, max_level):
        attempts = 0
        min_level = max_level if self.force_coop else 1
        for food_level in self.specified_food_levels:
            while attempts < 1000:
                attempts += 1
                row = self.np_random.integers(1, self.rows - 1)
                col = self.np_random.integers(1, self.cols - 1)

                # check if it has neighbors:
                if (
                    self.neighborhood(row, col).sum() > 0
                    or self.neighborhood(row, col, distance=2, ignore_diag=True) > 0
                    or not self._is_empty_location(row, col)
                ):
                    continue

                self.field[row, col] = (food_level 
                                        if food_level is not None 
                                        else self.np_random.integers(min_level, max_level+1)
                                        )
                break
        self._food_spawned = self.field.sum()

    def _is_empty_location(self, row, col):
        if self.field[row, col] != 0:
            return False
        for pos in self.pos.values():
            if pos[0] == row and pos[1] == col:
                return False
        return True

    def spawn_agents(self, max_agent_level):
        possible_indices = np.arange(self.field_length)[self.field.flatten()==0]
        num_agents_to_spawn = len(self.agents)
        spawn_indices = self.np_random.choice(possible_indices,
                                              size=num_agents_to_spawn,
                                              replace=False)
        unraveled_indices = np.unravel_index(spawn_indices, shape=self.field_size)
        unraveled_indices = list(zip(*unraveled_indices))
        for i, agent in enumerate(self.agents):
            self.pos[agent] = unraveled_indices[i]
            if self.specified_agent_levels[agent] is None:
                self.agent_levels[agent] = self.np_random.integers(1, max_agent_level + 1)
            else:
                self.agent_levels[agent] = min(self.specified_agent_levels[agent], max_agent_level)

    def _is_valid_action(self, agent, action):
        if action == Action.NONE:
            return True
        row_pos = self.pos[agent][0]
        col_pos = self.pos[agent][1]
        row_pos, col_pos = self.pos[agent]
        row_pos_min, col_pos_min = (0, 0)
        row_pos_max, col_pos_max = (self.rows-1, self.cols-1)
        if action == Action.NORTH:
            return (
                row_pos > row_pos_min
                and self.field[row_pos-1, col_pos] == 0
            )
        if action == Action.SOUTH:
            return (
                row_pos < row_pos_max
                and self.field[row_pos+1, col_pos] == 0
            )
        if action == Action.WEST:
            return (
                col_pos > col_pos_min
                and self.field[row_pos, col_pos-1] == 0
            )
        if action == Action.EAST:
            return (
                col_pos < col_pos_max
                and self.field[row_pos, col_pos+1] == 0
            )
        if action == Action.LOAD:
            return self.adjacent_food(*self.pos[agent]) > 0

        self.logger.error("Undefined action {} from {}".format(action, agent))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self, agent):
        # TODO
        return self._valid_actions[agent]

    def reset(self, seed=None, return_info=False, options=None):
        if seed is not None:
            self.seed(seed=seed)
        self.field = np.zeros(self.field_size, np.int32)
        self.agents = copy(self.possible_agents)
        self.spawn_agents(self.max_agent_level)
        self.spawn_food(
            max_level=sum(self.agent_levels.values())
            )
        self.current_step = 0
        self._game_over = False
        self.terminated = False
        self.truncated = False
        self._gen_valid_moves()

        observations = {agent: self.observe(agent) for agent in self.agents}
        infos = {agent: {"action_mask": self._action_mask(agent)}
                 for agent in self.agents}
        return observations, infos

    def step(self, actions):
        self.current_step += 1

        rewards = {agent: 0.0 for agent in self.agents}
        actions = {agent: (Action(a) if Action(a) in self._valid_actions[agent] else Action.NONE)
                   for agent, a in actions.items()}

        loading_agents = set()
        # move agents
        # if two or more agents try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for agent, action in actions.items():
            if action == Action.NONE:
                collisions[tuple(self.pos[agent])].append(agent)
            elif action == Action.NORTH:
                collisions[(self.pos[agent][0] - 1, self.pos[agent][1])].append(agent)
            elif action == Action.SOUTH:
                collisions[(self.pos[agent][0] + 1, self.pos[agent][1])].append(agent)
            elif action == Action.WEST:
                collisions[(self.pos[agent][0], self.pos[agent][1] - 1)].append(agent)
            elif action == Action.EAST:
                collisions[(self.pos[agent][0], self.pos[agent][1] + 1)].append(agent)
            elif action == Action.LOAD:
                collisions[tuple(self.pos[agent])].append(agent)
                loading_agents.add(agent)

        # and do movements for non colliding agents
        for pos, agents in collisions.items():
            if len(agents) > 1:  # make sure no more than an agents will arrive at location
                continue
            self.pos[agents[0]] = pos

        # finally process the loadings:
        while loading_agents:
            # find adjacent food
            agent = loading_agents.pop()
            frow, fcol = self.adjacent_food_location(*self.pos[agent])
            food = self.field[frow, fcol]

            adj_agents = self.adjacent_agents(frow, fcol)
            adj_agents = [
                a for a in adj_agents if a in loading_agents or a is agent
            ]

            adj_agent_level = sum([self.agent_levels[a] for a in adj_agents])

            loading_agents = loading_agents - set(adj_agents)

            if adj_agent_level < food:
                # failed to load
                for a in adj_agents:
                    rewards[a] -= self.penalty
                continue

            # else the food was loaded and each agent scores points
            for a in adj_agents:
                rewards[a] = float(self.agent_levels[a] * food)
                if self._normalize_reward:
                    rewards[a] = rewards[a] / float(
                        adj_agent_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

        # TODO when pettingzoo distinguishes between 'done' and 'terminated/truncated' will need to update
        self.terminated = self.field.sum == 0
        self.truncated = self._max_cycles <= self.current_step
        terminated = {agent: self.terminated for agent in self.agents}
        truncated = {agent: self.truncated for agent in self.agents}
        self._game_over = self.terminated or self.truncated
        dones = {agent: self._game_over for agent in self.agents}

        observations = {agent: self.observe(agent) for agent in self.agents}

        self._gen_valid_moves()
        infos = {agent: {"action_mask": self._action_mask(agent),
                         "terminated": self.terminated,
                         "truncated": self.truncated,}
                 for agent in self.agents}
        self.agents = [agent for agent in self.agents if not dones[agent]]
        return observations, rewards, terminated, truncated, infos

    def _get_global_grid_layers(self):
        grid_shape_x, grid_shape_y = self.field_size
        grid_shape_x += 2 * self.sight
        grid_shape_y += 2 * self.sight
        grid_shape = (grid_shape_x, grid_shape_y)

        # Agents layer: level & position of agents
        agents_layer = np.zeros(grid_shape, dtype=np.float32)
        for agent in self.agents:
            row, col = self.pos[agent]
            agents_layer[self.sight + row, self.sight + col] = self.agent_levels[agent]
        
        # Foods layer: level & position of foods
        foods_layer = np.zeros(grid_shape, dtype=np.float32)
        foods_layer[self.sight:-self.sight, self.sight:-self.sight] = self.field.copy()

        # Access layer: 1 if grid cells are accessible
        access_layer = np.ones(grid_shape, dtype=np.float32)
        # out of bounds not accessible
        access_layer[:self.sight, :] = 0.0
        access_layer[-self.sight:, :] = 0.0
        access_layer[:, :self.sight] = 0.0
        access_layer[:, -self.sight:] = 0.0
        # agent locations are not accessible
        for agent in self.agents:
            row, col = self.pos[agent]
            access_layer[self.sight + row, self.sight + col] = 0.0
        # food locations are not accessible
        for row, col in zip(*self.field.nonzero()):
            access_layer[self.sight + row, self.sight + col] = 0.0
        
        return np.stack([agents_layer, foods_layer, access_layer])

    def _get_grid_obs(self, agent):
        global_grid_layers = self._get_global_grid_layers()
        row, col = self.pos[agent]
        start_row, end_row = row, row + 2*self.sight+1 
        start_col, end_col = col, col + 2*self.sight+1
        return global_grid_layers[:, start_row:end_row, start_col:end_col]

    def _get_array_obs(self, agent):
        obs = np.zeros(self.observation_space(agent).shape, dtype=np.float32)
        local_field = self.neighborhood(*self.pos[agent], distance=self.sight)
        obs[:3*self.max_food] = np.tile([-1, -1, 0], reps=self.max_food)
        for i, (row, col) in enumerate(zip(*np.nonzero(local_field))):
            obs[(3*i):(3*i+3)] = [row, col, local_field[row, col]]

        obs[3*self.max_food:] = np.tile([-1, -1, 0], reps=len(self.possible_agents))
        # self agent is always first
        ordered_agents = [agent] + [a for a in self.possible_agents if a != agent]
        for i, other_agent in enumerate(ordered_agents):
            relative_pos = self._transform_to_neighborhood(self.pos[agent],
                                                           self.sight,
                                                           self.pos[other_agent])
            if self._in_sight(relative_pos):
                idx = 3*self.max_food + 3*i
                obs[idx:idx+3] = [*relative_pos, self.agent_levels[other_agent]]
        return obs

    def _in_sight(self, relative_pos):
        lower_bound = np.array([0, 0])
        upper_bound = np.array([2*self.sight, 2*self.sight])
        rpos = np.array(relative_pos)
        return np.any((lower_bound < rpos) & (rpos < upper_bound))

    def observe(self, agent):
        if self._grid_observation:
            obs = self._get_grid_obs(agent)
        else:
            obs = self._get_array_obs(agent)
        assert self.observation_space(agent).contains(obs), \
            f"obs space error: obs: {obs}, obs_space: {self.observation_space(agent)}"
        return obs

    def _init_render(self):
        if self.render_style == "full":
            from .rendering import Viewer
            self.viewer = Viewer((self.rows, self.cols))
        elif self.render_style == "simple":
            from .simple_render import render
            self.simple_render = render
        self._rendering_initialized = True

    def render(self):
        if not self._rendering_initialized:
            self._init_render()
        if self.render_style == "full":
            return self.viewer.render(self, return_rgb_array=(self.render_mode=="rgb_array"))
        elif self.render_style == "simple":
            return self.simple_render(self)

    def close(self):
        if self.viewer:
            self.viewer.close()
