import logging
import random
from typing import Any, Optional

import numpy as np

_MAX_INT = 999999


class Agent:
    name = "Prototype Agent"

    def __repr__(self):
        return self.name

    def __init__(self, player):
        self.logger = logging.getLogger(__name__)
        self.player = player

    def __getattr__(self, item):
        return getattr(self.player, item)

    def act(self, obs):
        if hasattr(obs, "players"):
            self.observed_position = next(
                (x for x in obs.players if x.is_self), None
            ).position

        try:
            action = self._act(obs)
        except Exception as e:
            # self.logger.error(f"Error in agent {self.name} act method: {e}")
            action = random.choice(obs.actions)

        self.history.append(action)
        return action

    def _act(self, obs) -> Any:
        raise NotImplemented("You must implement an agent")

    def _closest_food(self, obs, max_food_level=None, start=None):
        x, y = start if start else self.observed_position

        field = np.copy(obs.field)
        food_lvls, food_feats = field[:, :, 0], field[:, :, 1]

        if max_food_level:
            food_lvls[food_lvls > max_food_level] = 0

        r, c = np.nonzero(food_lvls)
        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]

    def _make_state(self, obs):

        state = str(obs.field)
        for c in ["]", "[", " ", "\n"]:
            state = state.replace(c, "")

        for a in obs.players:
            state = state + str(a.position[0]) + str(a.position[1]) + str(a.level)

        return int(state)

    def cleanup(self):
        pass


class PrefsAgent(Agent):
    name = "Prototype preferences agent"

    @property
    def preferences(self):
        return self.player.preferences

    def _get_food_info(self, obs, max_food_level=None):
        field = np.copy(obs.field)
        food_lvls, food_feats = field[:, :, 0], field[:, :, 1]

        if max_food_level:
            idx = food_lvls > max_food_level
            food_lvls[idx] = 0
            food_feats[idx] = 0

        return food_lvls, food_feats

    def _get_food_vals(self, feature_grid):
        return np.concatenate([[0], self.preferences])[feature_grid.astype(int)]

    def _closest_positive_food(self, obs, max_food_level=None, start=None):
        x, y = start if start else self.observed_position

        _, food_feats = self._get_food_info(obs, max_food_level)
        pos_food_vals = self._get_food_vals(food_feats) > 0
        if not np.any(pos_food_vals):
            return None

        r, c = np.nonzero(pos_food_vals)

        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]

    def _closest_best_food(self, obs, max_food_level=None, start=None):
        x, y = start if start else self.observed_position

        _, food_feats = self._get_food_info(obs, max_food_level)
        food_vals = self._get_food_vals(food_feats)

        max_val = food_vals.max()
        if max_val <= 0:
            return None

        best_foods = food_vals == food_vals.max()
        r, c = np.nonzero(best_foods)

        try:
            min_idx = ((r - x) ** 2 + (c - y) ** 2).argmin()
        except ValueError:
            return None

        return r[min_idx], c[min_idx]
