import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from typing import List, Tuple

# from gym import Env
import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from lbforaging.utils import to_one_hot


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


class Player:
    def __init__(self):
        self.controller = None
        self.position = None
        self.level = None
        self.preferences = None
        self.features = None
        self.field_size = None
        self.score = 0.0
        self.reward = 0.0
        self.history = None
        self.current_step = None

    def setup(self, position, level, preferences, features, field_size):
        self.history = []
        self.position = position
        self.level = level
        self.preferences = preferences
        self.features = features
        self.field_size = field_size
        self.reward = 0.0
        self.score = 0.0

    def set_controller(self, controller):
        self.controller = controller

    def step(self, obs):
        return self.controller._step(obs)

    def food_utils(self, food_features):
        assert self.preferences is not None
        assert self.preferences.shape == food_features.shape
        return np.dot(self.preferences, food_features)

    def set_reward(self, r: float):
        self.reward = r
        self.score += r

    @property
    def name(self):
        if self.controller:
            return self.controller.name
        else:
            return "Player"


class ForagingEnv(gym.Env):
    """
    A class that contains rules/actions for the game level-based foraging.
    """

    metadata = {"render.modes": ["human"]}

    action_set = [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.LOAD]
    Observation = namedtuple(
        "Observation",
        ["field", "actions", "players", "game_over", "sight", "current_step"],
    )
    PlayerObservation = namedtuple(
        "PlayerObservation",
        ["position", "level", "features", "history", "reward", "is_self"],
    )  # reward is available only if is_self

    def __init__(
        self,
        players: int,
        max_player_level: int,
        field_size: Tuple[int],
        max_food: int,
        food_feature_dim: int,
        agent_feature_dim: int,
        sight: int,
        max_episode_steps: int,
        force_coop: bool,
        normalize_reward=True,
        grid_observation=False,
        penalty=0.0,
        render_mode=None,
    ):
        self.logger = logging.getLogger(__name__)
        self.seed()

        self.field_size = field_size
        self.food_feature_dim = food_feature_dim
        self.agent_feature_dim = agent_feature_dim
        self.init_field()

        self.players = [Player() for _ in range(players)]
        self.n_agents = len(self.players)
        self.penalty = penalty
        self.max_food = max_food
        self._food_spawned = 0.0
        self.max_player_level = max_player_level
        self.sight = sight
        self.force_coop = force_coop
        self._game_over = None

        # self.render_mode = render_mode
        self._rendering_initialized = False
        self._valid_actions = None
        self._max_episode_steps = max_episode_steps

        self._normalize_reward = normalize_reward
        self._grid_observation = grid_observation

        self.action_space = gym.spaces.Tuple(
            tuple([gym.spaces.Discrete(6)] * len(self.players))
        )
        self.observation_space = gym.spaces.Tuple(
            tuple([self._get_observation_space()] * len(self.players))
        )

        self.viewer = None

    def init_field(self):
        self.field = np.zeros(self.field_size + (1 + self.food_feature_dim,), np.int32)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_observation_space(self):
        """The Observation Space for each agent.
        - all of the board (board_size^2) with foods
        - player description (x, y, level)*player_count
        """
        if not self._grid_observation:
            field_x, field_y, _ = self.field.shape
            min_loc, max_loc = [-1, -1], [field_x - 1, field_y - 1]
            max_food_level = self.max_player_level * len(self.players)

            min_fruit_obs = [*min_loc, 0] + [0] * self.food_feature_dim
            max_fruit_obs = [*max_loc, max_food_level] + [1] * self.food_feature_dim

            min_agent_obs = [*min_loc, 0] + [0] * self.agent_feature_dim
            max_agent_obs = [*max_loc, self.max_player_level] + [
                1
            ] * self.agent_feature_dim

            min_obs = min_fruit_obs * self.max_food + min_agent_obs * len(self.players)
            max_obs = max_fruit_obs * self.max_food + max_agent_obs * len(self.players)

        else:
            # grid observation space
            grid_shape = (1 + 2 * self.sight, 1 + 2 * self.sight)

            # agents layer: agent levels
            agents_min = np.zeros(grid_shape, dtype=np.float32)
            agents_max = np.ones(grid_shape, dtype=np.float32) * self.max_player_level

            # foods layer: foods level
            max_food_level = self.max_player_level * len(self.players)
            foods_min = np.zeros(grid_shape, dtype=np.float32)
            foods_max = np.ones(grid_shape, dtype=np.float32) * max_food_level

            # access layer: i the cell available
            access_min = np.zeros(grid_shape, dtype=np.float32)
            access_max = np.ones(grid_shape, dtype=np.float32)

            # total layer
            min_obs = np.stack([agents_min, foods_min, access_min])
            max_obs = np.stack([agents_max, foods_max, access_max])

        return gym.spaces.Box(np.array(min_obs), np.array(max_obs), dtype=np.float32)

    @classmethod
    def from_obs(cls, obs):
        players = []
        for p in obs.players:
            player = Player()
            player.setup(p.position, p.level, np.array([1]), obs.field.shape)
            player.score = p.score if p.score else 0
            players.append(player)

        env = cls(players, None, None, None, None)
        env.field = np.copy(obs.field)

        frows, fcols = env.field[:, :, 0].nonzero()
        num_foods = len(frows)
        for p in env.players:
            p.preferences = np.ones(num_foods)

        env.current_step = obs.current_step
        env.sight = obs.sight
        env._gen_valid_moves()

        return env

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
            player: [
                action for action in Action if self._is_valid_action(player, action)
            ]
            for player in self.players
        }

    def neighborhood(self, row, col, distance=1, ignore_diag=False):
        if not ignore_diag:
            return self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows),
                max(col - distance, 0) : min(col + distance + 1, self.cols),
            ]

        # max note: check this
        return (
            self.field[
                max(row - distance, 0) : min(row + distance + 1, self.rows), col
            ].sum()
            + self.field[
                row, max(col - distance, 0) : min(col + distance + 1, self.cols)
            ].sum()
        )

    def is_food(self, row, col):
        return self.food_level(row, col) > 0

    def food_level(self, row, col):
        return self.field[row, col][0]

    def food_feature(self, row, col):
        return self.field[row, col][1:]

    def food_levels(self):
        return self.field[:, :, 0]

    def food_features(self):
        return self.field[:, :, 1:]

    def adjacent_food(self, row, col):
        return (
            self.food_level(max(row - 1, 0), col)
            + self.food_level(min(row + 1, self.rows - 1), col)
            + self.food_level(row, max(col - 1, 0))
            + self.food_level(row, min(col + 1, self.cols - 1))
        )

    def adjacent_food_location(self, row, col):
        if row > 1 and self.food_level(row - 1, col) > 0:
            return row - 1, col
        elif row < self.rows - 1 and self.food_level(row + 1, col) > 0:
            return row + 1, col
        elif col > 1 and self.food_level(row, col - 1) > 0:
            return row, col - 1
        elif col < self.cols - 1 and self.food_level(row, col + 1) > 0:
            return row, col + 1

    def adjacent_players(self, row, col):
        return [
            player
            for player in self.players
            if abs(player.position[0] - row) == 1
            and player.position[1] == col
            or abs(player.position[1] - col) == 1
            and player.position[0] == row
        ]

    def spawn_food(self, max_food, max_level):
        food_count, attempts = 0, 0
        min_level = max_level if self.force_coop else 1

        while food_count < max_food and attempts < 1000:
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

            # food_level = (
            #     min_level
            #     if min_level == max_level
            #     # ! this is excluding food of level `max_level` but is kept for
            #     # ! consistency with prior LBF versions
            #     else self.np_random.integers(min_level, max_level)
            # )

            food_level = self.np_random.integers(1, 3)
            food_feature_vec = to_one_hot(food_count, self.food_feature_dim)
            self.field[row, col] = [food_level, *food_feature_vec]
            food_count += 1
        self._food_spawned = self.field[:, :, 0].sum()

    def remove_food(self, row, col):
        self.field[row, col] = [0] * self.field.shape[2]

    def _is_empty_location(self, row, col):
        if self.food_level(row, col) > 0:
            return False
        for a in self.players:
            if a.position and row == a.position[0] and col == a.position[1]:
                return False

        return True

    def spawn_players(self, max_player_level):
        for m, player in enumerate(self.players):
            attempts = 0

            # prefs = np.array([1.0, 1.0, -1.5]) if m == 0 else np.array([-1.5, 1.0, 1.0])
            prefs = np.array([1.0, 2.0, 0.0])
            feature = to_one_hot(m + 1, self.agent_feature_dim)

            while attempts < 1000:
                row = self.np_random.integers(0, self.rows)
                col = self.np_random.integers(0, self.cols)
                if self._is_empty_location(row, col):
                    player.setup(
                        (row, col),
                        1,
                        # self.np_random.integers(1, max_player_level + 1),
                        prefs,
                        feature,
                        self.field_size,
                    )
                    break
                attempts += 1

    def _is_valid_action(self, player, action):
        if action == Action.NONE:
            return True
        elif action == Action.NORTH:
            return player.position[0] > 0 and not self.is_food(
                player.position[0] - 1, player.position[1]
            )
        elif action == Action.SOUTH:
            return player.position[0] < self.rows - 1 and not self.is_food(
                player.position[0] + 1, player.position[1]
            )
        elif action == Action.WEST:
            return player.position[1] > 0 and not self.is_food(
                player.position[0], player.position[1] - 1
            )
        elif action == Action.EAST:
            return player.position[1] < self.cols - 1 and not self.is_food(
                player.position[0], player.position[1] + 1
            )
        elif action == Action.LOAD:
            return self.adjacent_food(*player.position) > 0

        self.logger.error("Undefined action {} from {}".format(action, player.name))
        raise ValueError("Undefined action")

    def _transform_to_neighborhood(self, center, sight, position):
        return (
            position[0] - center[0] + min(sight, center[0]),
            position[1] - center[1] + min(sight, center[1]),
        )

    def get_valid_actions(self) -> list:
        return list(product(*[self._valid_actions[player] for player in self.players]))

    def _make_obs(self, player):
        return self.Observation(
            actions=self._valid_actions[player],
            players=[
                self.PlayerObservation(
                    position=self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    ),
                    level=a.level,
                    features=a.features,
                    is_self=a == player,
                    history=a.history,
                    reward=a.reward if a == player else None,
                )
                for a in self.players
                if min(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                >= 0
                and max(
                    self._transform_to_neighborhood(
                        player.position, self.sight, a.position
                    )
                )
                <= 2 * self.sight
            ],
            # todo also check max?
            field=np.copy(self.neighborhood(*player.position, self.sight)),
            game_over=self.game_over,
            sight=self.sight,
            current_step=self.current_step,
        )

    def _make_gym_obs(self):
        def make_obs_array(observation):
            obs = np.zeros(self.observation_space[0].shape, dtype=np.float32)

            # self player is always first
            seen_players = [p for p in observation.players if p.is_self] + [
                p for p in observation.players if not p.is_self
            ]

            default_fruit_repr = [-1, -1, 0, 0]
            obs[: 4 * self.max_food] = default_fruit_repr * self.max_food

            food_levels, food_features = (
                observation.field[:, :, 0],
                observation.field[:, :, 1:],
            )

            food_chunk_len = 3 + self.food_feature_dim
            agent_chunk_len = 3 + self.agent_feature_dim

            for i, (y, x) in enumerate(zip(*np.nonzero(food_levels))):
                start, end = food_chunk_len * i, food_chunk_len * (i + 1)
                f = food_features[y, x].tolist()
                obs[start:end] = [y, x, food_levels[y, x], *f]

            default_player_repr = [-1, -1, 0] + [0] * self.agent_feature_dim
            obs[self.max_food * food_chunk_len : len(obs)] = default_player_repr * len(
                self.players
            )

            for i, p in enumerate(seen_players):
                offset = self.max_food * food_chunk_len
                start, end = offset + (agent_chunk_len * i), offset + (
                    agent_chunk_len * (i + 1)
                )
                obs[start:end] = [*p.position, p.level] + p.features.tolist()

            return obs

        # max note: update this
        def make_global_grid_arrays():
            """
            Create global arrays for grid observation space
            """
            grid_shape_x, grid_shape_y = self.field_size
            grid_shape_x += 2 * self.sight
            grid_shape_y += 2 * self.sight
            grid_shape = (grid_shape_x, grid_shape_y)

            agents_layer = np.zeros(grid_shape, dtype=np.float32)
            for player in self.players:
                player_x, player_y = player.position
                agents_layer[player_x + self.sight, player_y + self.sight] = (
                    player.level
                )

            foods_layer = np.zeros(grid_shape, dtype=np.float32)
            foods_layer[self.sight : -self.sight, self.sight : -self.sight] = (
                self.field.copy()
            )

            access_layer = np.ones(grid_shape, dtype=np.float32)
            # out of bounds not accessible
            access_layer[: self.sight, :] = 0.0
            access_layer[-self.sight :, :] = 0.0
            access_layer[:, : self.sight] = 0.0
            access_layer[:, -self.sight :] = 0.0
            # agent locations are not accessible
            for player in self.players:
                player_x, player_y = player.position
                access_layer[player_x + self.sight, player_y + self.sight] = 0.0
            # food locations are not accessible
            foods_x, foods_y = self.field.nonzero()
            for x, y in zip(foods_x, foods_y):
                access_layer[x + self.sight, y + self.sight] = 0.0

            return np.stack([agents_layer, foods_layer, access_layer])

        def get_agent_grid_bounds(agent_x, agent_y):
            return (
                agent_x,
                agent_x + 2 * self.sight + 1,
                agent_y,
                agent_y + 2 * self.sight + 1,
            )

        def get_player_reward(observation):
            for p in observation.players:
                if p.is_self:
                    return p.reward

        observations = [self._make_obs(player) for player in self.players]

        if self._grid_observation:
            layers = make_global_grid_arrays()
            agents_bounds = [
                get_agent_grid_bounds(*player.position) for player in self.players
            ]
            nobs = tuple(
                [
                    layers[:, start_x:end_x, start_y:end_y]
                    for start_x, end_x, start_y, end_y in agents_bounds
                ]
            )
        else:
            nobs = tuple([make_obs_array(obs) for obs in observations])
        nreward = [get_player_reward(obs) for obs in observations]
        ndone = [obs.game_over for obs in observations]
        ntrunc = [False] * len(ndone)
        info = {"full_observations": observations}

        # check the space of obs
        for i, obs in enumerate(nobs):
            assert self.observation_space[i].contains(
                obs
            ), f"obs space error: obs: {obs}, obs_space: {self.observation_space[i]}"

        return nobs, nreward, ndone, ntrunc, info

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)

        self.init_field()
        self.spawn_players(self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(self.max_food, max_level=sum(player_levels[:3]))
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        return self._make_gym_obs()[0]

    def step(self, actions):
        self.current_step += 1

        for i, a in enumerate(actions):
            act = Action(a)
            self.players[i].set_reward(0 if act == Action.NONE else -self.penalty)
            actions[i] = (
                act if act in self._valid_actions[self.players[i]] else Action.NONE
            )

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players

        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food_level, food_feature = self.food_level(frow, fcol), self.food_feature(
                frow, fcol
            )

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]
            loading_players = loading_players - set(adj_players)

            adj_player_level = sum([a.level for a in adj_players])
            if adj_player_level < food_level:
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                u_food = a.food_utils(food_feature)
                r = float(a.level * food_level * u_food)
                if self._normalize_reward:
                    r /= float(adj_player_level * self._food_spawned)
                a.set_reward(r)
            # and the food is removed
            self.remove_food(frow, fcol)

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        return self._make_gym_obs()

    def _init_render(self):
        from .rendering import Viewer

        self.viewer = Viewer((self.rows, self.cols))
        self._rendering_initialized = True

    def render(self, mode="human"):
        if not self._rendering_initialized:
            self._init_render()

        return self.viewer.render(self, return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
