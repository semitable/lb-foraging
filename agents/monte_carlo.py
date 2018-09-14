import os
from itertools import chain, repeat, product
import logging
from copy import copy, deepcopy
import math
import time
import numpy as np
import pandas as pd
import random
from foraging import Agent, Env
from foraging.environment import Action
from agents import H1, H2, H3, H4
import networkx as nx

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, state: Env):

        self.root = None

        self.move = None
        self.visits = 0
        self.reward = 0
        self.parent = None
        self.children = []

        self.state = state

        self.available_moves = set(state.get_valid_actions())
        self.tried_moves = set()

        self.is_terminal = False

    def not_expanded(self):
        return len(self.available_moves - self.tried_moves) > 0

    def non_terminal(self):
        return not self.is_terminal

    def expand(self):
        unchecked_moves = self.available_moves - self.tried_moves

        move = random.sample(unchecked_moves, 1)[0]
        u_new = self.add_child(move)

        return u_new

    def add_child(self, move):

        new_state = deepcopy(self.state)

        new_is_terminal = False

        observations = new_state.step(move)

        if new_state.game_over:
            new_is_terminal = True

        u_new = Node(new_state)

        u_new.is_terminal = new_is_terminal

        u_new.parent = self
        u_new.move = move
        u_new.root = self.root

        self.tried_moves.add(move)
        self.children.append(u_new)

        return u_new

    def best_child(self, c=2, h=10):

        my_id = 0  # todo fix this

        ucb1 = lambda u: (u.reward / u.visits
                          + c * math.sqrt(math.log(self.root.visits / u.visits))
                          + h * u.state.players[my_id].score / u.visits
                          )
        best = max(self.children, key=ucb1)

        return best

    def most_visited_child(self):
        most_visited = lambda u: u.visits
        best = max(self.children, key=most_visited)
        return best


class MonteCarloAgent(Agent):
    name = "Monte Carlo Agent"

    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)
        pass

    def step(self, obs):

        my_id = 0  # todo fix this
        env = Env.from_obs(obs)
        root = self.uct_search(env)

        move = root.most_visited_child().move
        print(move)
        print(root.most_visited_child().visits)

        return move[my_id]

    def uct_search(self, state: Env, timeout=30):
        graph = nx.DiGraph()

        root = Node(state)
        root.root = root

        graph.add_node(root)

        future = timeout + time.time()

        while time.time() < future:

            # selection & expansion
            u_next = self.tree_policy(root)

            if u_next not in graph:
                graph.add_node(u_next)
                graph.add_edge(u_next.parent, u_next, action=str(u_next.move))

            # simulation
            delta = self.default_policy(u_next)

            # back propagation
            self.backup(u_next, delta)

        return root

    def backup(self, u: Node, delta: float):
        while u is not None:
            u.visits += 1
            u.reward += delta
            u = u.parent

    def tree_policy(self, u: Node):
        while u.non_terminal():
            if u.not_expanded():
                return u.expand()
            else:
                u = u.best_child()
        return u

    def random_play(self, state: Env):
        actions = state.get_valid_actions()
        a = random.choice(actions)
        state.step(a)

    def default_policy(self, u: Node):
        if u.non_terminal():
            new_state = deepcopy(u.state)

            while not new_state.game_over:
                self.random_play(new_state)

        else:
            new_state = u.state

        my_id = 0  # todo fix this

        return new_state.players[my_id].score