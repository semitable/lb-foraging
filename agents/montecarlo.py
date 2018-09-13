import os
from itertools import chain, repeat, product
import logging
import copy

logger = logging.getLogger(__name__)
import time
import numpy as np
import pandas as pd
import random
from foraging import Agent, Env
from foraging.environment import Action
from agents import H1, H2, H3, H4
import networkx as nx

class Node:
    def __init__(self, state: Env):

        self.root = None

        self.move = None
        self.visits = 0
        self.reward = 0
        self.parent = None
        self.children = []

        self.state = state

        self.available_moves = set(state.get_moves())
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

        new_state = self.state.clone()

        new_is_terminal = False
        try:
            new_state.do_move(move)
        except GameOver:
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

        ucb1 = lambda u: (u.reward / u.visits
                          + c * math.sqrt(math.log(self.root.visits / u.visits))
                          + h * u.state.get_score(u.state.game.players[u.root.state.current_player_id]) / u.visits
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
        action = random.choice(obs.actions)
        return action

    def uct_search(state: Env, timeout=20):
        graph = nx.DiGraph()

        root = Node(state)
        root.root = root

        graph.add_node(root)

        future = timeout + time.time()

        while time.time() < future:

            # selection & expansion
            u_next = tree_policy(root)

            if u_next not in graph:
                graph.add_node(u_next)
                graph.add_edge(u_next.parent, u_next, action=str(u_next.move))

            # simulation
            delta = default_policy(u_next)

            # back propagation
            backup(u_next, delta)

        return root

    def backup(u: Node, delta: float):
        while u is not None:
            u.visits += 1
            u.reward += delta
            u = u.parent

    def tree_policy(u: Node):
        while u.non_terminal():
            if u.not_expanded():
                return u.expand()
            else:
                u = u.best_child()
        return u

    def default_policy(u: Node):
        if u.non_terminal():
            new_state = u.state.clone()
            try:
                while True:
                    random_play(new_state.game)
            except GameOver:
                pass
        else:
            new_state = u.state

        my_id = u.root.state.current_player_id

        if new_state.game.players[my_id].playstate == PlayState.WON:
            return 1
        elif new_state.game.players[my_id].playstate == PlayState.LOST:
            return 0
        elif new_state.game.players[my_id].playstate == PlayState.TIED:
            return 0
        else:
            raise ValueError
