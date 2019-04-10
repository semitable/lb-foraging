import logging
import math
import pickle
import random
import time

import networkx as nx
import plotly.graph_objs as go
from networkx.drawing.nx_pydot import graphviz_layout

from foraging import Agent, Env

MCTS_DEPTH = 15

logger = logging.getLogger(__name__)


def plot_graph(G):
    pos = graphviz_layout(G, prog="dot")

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=go.Line(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    my_annotations = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)
        my_annotations.append(
            dict(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                xref="x",
                yref="y",
                text="{}".format(G.get_edge_data(edge[0], edge[1])["action"]),
                showarrow=False,
                arrowhead=2,
                ax=0,
                ay=0,
            )
        )

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=go.Marker(
            showscale=False,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale="YIGnBu",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line=dict(width=2),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"].append(x)
        node_trace["y"].append(y)

        node_info = "Visits: +{0}<br>Rewards: {1}<br>Score: {2}".format(
            node.visits, node.reward, node.state.players[0].score
        )

        node_trace["text"].append(node_info)

    fig = go.Figure(
        data=go.Data([edge_trace, node_trace]),
        layout=go.Layout(
            title="<br>Network graph made with Python",
            titlefont=dict(size=16),
            showlegend=False,
            width=650,
            height=650,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=my_annotations,
            xaxis=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=go.YAxis(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    return fig


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

        move = random.choice(tuple(unchecked_moves))
        u_new = self.add_child(move)

        return u_new

    def add_child(self, move):
        new_state = pickle.loads(pickle.dumps(self.state, pickle.HIGHEST_PROTOCOL))
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

        ucb1 = lambda u: (
            u.reward / u.visits
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

        return move[my_id]

    def uct_search(self, state: Env, timeout=0.5):
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

        # py.plot(plot_graph(graph))
        # print(root.visits)
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
            new_state = pickle.loads(pickle.dumps(u.state, pickle.HIGHEST_PROTOCOL))
            depth = 0
            while not new_state.game_over and depth < MCTS_DEPTH:
                self.random_play(new_state)
                depth += 1

        else:
            new_state = u.state

        my_id = 0  # todo fix this

        return new_state.players[my_id].score
