import random
import numpy as np

from lbforaging.foraging.agent import PrefsAgent
from ..foraging import Agent
from ..foraging.environment import Action


class HeuristicAgent(Agent):
    name = "Heuristic Agent"

    def _center_of_players(self, players):
        coords = np.array([player.position for player in players])
        center = np.rint(coords.mean(axis=0))
        return (center[0], center[1])

    def _move_towards(self, target, allowed):

        y, x = self.observed_position
        r, c = target

        poss_actions = []

        if r < y and Action.NORTH in allowed:
            poss_actions.append(Action.NORTH)
        elif r > y and Action.SOUTH in allowed:
            poss_actions.append(Action.SOUTH)
        if c > x and Action.EAST in allowed:
            poss_actions.append(Action.EAST)
        elif c < x and Action.WEST in allowed:
            poss_actions.append(Action.WEST)

        if len(poss_actions) == 0:
            return Action.NONE

        return random.choice(poss_actions)

    def _load_or_move_towards(self, obs, target):
        r, c = target
        y, x = self.observed_position
        if (abs(r - y) + abs(c - x)) == 1:
            return Action.LOAD
        return self._move_towards((r, c), obs.actions)


class H1(HeuristicAgent):
    """
    H1 agent always goes to the closest food
    """

    name = "H1"

    def _act(self, obs):
        target = self._closest_food(obs)
        return self._load_or_move_towards(obs, target)


class H1PrefsSatisfice(HeuristicAgent, PrefsAgent):
    """
    Variant of H1: always goes to the closest food with nonzero value under its feature preferences
    """

    name = "H1PrefsSatisfice"

    def _act(self, obs):
        target = self._closest_positive_food(obs)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H1PrefsOptim(HeuristicAgent, PrefsAgent):
    """
    Variant of H1: always goes to the closest food with maximal value under its feature preferences
    """

    name = "H1PrefsOptim"

    def _act(self, obs):
        target = self._closest_best_food(obs)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H2(HeuristicAgent):
    """
    H2 Agent goes to the one visible food which is closest to the centre of visible players
    """

    name = "H2"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        target = self._closest_food(obs, None, players_center)
        return self._load_or_move_towards(obs, target)


class H2PrefsSatisfice(HeuristicAgent, PrefsAgent):
    """
    Variant of H2: always goes to the food with nonzero value under its
    feature preferences that's closest to the centre of visible players
    """

    name = "H2PrefsSatisfice"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        target = self._closest_positive_food(obs, None, players_center)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H2PrefsOptim(HeuristicAgent, PrefsAgent):
    """
    Variant of H2: always goes to the food with maximal value under its
    feature preferences that's closest to the centre of visible players
    """

    name = "H2PrefsOptim"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        target = self._closest_best_food(obs, None, players_center)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H3(HeuristicAgent):
    """
    H3 Agent always goes to the closest food with compatible level
    """

    name = "H3"

    def _act(self, obs):
        target = self._closest_food(obs, self.level)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards


class H3PrefsSatisfice(HeuristicAgent, PrefsAgent):
    """
    Variant of H3: always goes to the closest food with nonzero value under its
    feature preferences and compatible level
    """

    name = "H3PrefsSatisfice"

    def _act(self, obs):
        target = self._closest_positive_food(obs, self.level)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H3PrefsOptim(HeuristicAgent, PrefsAgent):
    """
    Variant of H3: always goes to the closest food with maximal value under its
    feature preferences and compatible level
    """

    name = "H3PrefsOptim"

    def _act(self, obs):
        target = self._closest_best_food(obs, self.level)
        if target is None:
            return Action.NONE
        return self._load_or_move_towards(obs, target)


class H4(HeuristicAgent):
    """
    H4 Agent goes to the one visible food which is closest to all visible players
    such that the sum of their and H4's level is sufficient to load the food
    """

    name = "H4"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        try:
            target = self._closest_food(obs, players_sum_level, players_center)
        except TypeError:
            return Action.NONE

        return self._load_or_move_towards(obs, target)


class H4PrefsSatisfice(HeuristicAgent, PrefsAgent):
    """
    Variant of H4: always goes to the food with nonzero value under its feature
    preferences that's closest to all visible players and has level such that the
    sum of those agents' levels and H4's level is sufficient to load the food
    """

    name = "H4PrefsSatisfice"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        target = self._closest_positive_food(obs, players_sum_level, players_center)
        if target is None:
            return Action.NONE

        return self._load_or_move_towards(obs, target)


class H4PrefsOptim(HeuristicAgent, PrefsAgent):
    """
    Variant of H4: always goes to the food with maximal value under its feature
    preferences that's closest to all visible players and has level such that the
    sum of those agents' levels and H4's level is sufficient to load the food
    """

    name = "H4PrefsOptim"

    def _act(self, obs):
        players_center = self._center_of_players(obs.players)
        players_sum_level = sum([a.level for a in obs.players])

        target = self._closest_best_food(obs, players_sum_level, players_center)
        if target is None:
            return Action.NONE

        return self._load_or_move_towards(obs, target)
