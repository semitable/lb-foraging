# Based on: https://github.com/proroklab/VectorizedMultiAgentSimulator/blob/main/vmas/interactive_rendering.py
"""
Use this script to interactively play LBF

You can control the interaction with the following keys:
- Arrow keys: move the current agent
- L: load food
- K: load food and keep the agent loading
- SPACE: do nothing
- TAB: change the current agent
- R: reset the environment
- H: show help
- D: display agent info (per step)
- ESC: exit
"""
from argparse import ArgumentParser
import warnings

import numpy as np
import gymnasium as gym

from lbforaging.foraging.environment import Action


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="Foraging-8x8-2p-2f-v3",
        help="Environment to use",
    )
    parser.add_argument(
        "--display_info",
        action="store_true",
        help="Display agent info per step",
    )
    return parser.parse_args()


class InteractiveLBFEnv:
    """
    Use this script to interactively play with scenarios

    You can change agent by pressing TAB
    You can reset the environment by pressing R
    You can control agent actions with the arrow keys for movement and P/C for picking up/ collecting food
    """

    def __init__(
        self,
        env: str,
        display_info: bool = True,
    ):
        self.env = gym.make(env, render_mode="human")
        self.n_agents = self.env.unwrapped.n_agents
        self.running = True
        self.current_agent_index = 0
        self.current_action = None

        self.loading_agents = []
        self.t = 0
        self.ep_returns = np.zeros(self.n_agents)
        self.reset = False

        self.display_info = display_info

        obss, _ = self.env.reset()
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self._key_press

        if self.display_info:
            self._display_info(obss, [0] * self.n_agents, False)

        self._cycle()

    def _help(self):
        print("Use the arrow keys to move the current agent")
        print("Use the L key to load food")
        print("Use the K key to load food and keep the agent loading")
        print("Use the SPACE key to do nothing")
        print("Press TAB to change the current agent")
        print("Press R to reset the environment")
        print("Press H to show help")
        print("Press D to display agent info")
        print("Press ESC to exit")
        print()

    def _format_pos(self, pos):
        return f"row {pos[0] + 1}, col {pos[1] + 1}"

    def _display_info(self, obss, rews, done):
        print(f"Step {self.t}:")
        agent_level = self.env.unwrapped.players[self.current_agent_index].level
        agent_position = self.env.unwrapped.players[self.current_agent_index].position
        print(f"\tSelected: Agent {self.current_agent_index + 1} (Level {agent_level}, at {self._format_pos(agent_position)})")
        if self.loading_agents:
            print(f"\tLoading: {[i + 1 for i in self.loading_agents]}")
        print(f"\tObs: {obss[self.current_agent_index]}")
        print(f"\tRew: {round(rews[self.current_agent_index], 3)}")
        print(f"\tDone: {done}")
        # print(f"\tEp returns: {round(self.ep_returns[self.current_agent_index], 3)}")
        print()

    def _increment_current_agent_index(self, index: int):
        index += 1
        if index == self.n_agents:
            index = 0
        return index

    def _key_press(self, k, mod):
        from pyglet.window import key

        if k == key.LEFT:
            self.current_action = Action.WEST
        elif k == key.RIGHT:
            self.current_action = Action.EAST
        elif k == key.DOWN:
            self.current_action = Action.SOUTH
        elif k == key.UP:
            self.current_action = Action.NORTH
        elif k == key.L:
            self.current_action = Action.LOAD
        elif k == key.K:
            self.current_action = Action.LOAD
            self.loading_agents.append(self.current_agent_index)
        elif k == key.SPACE:
            self.current_action = Action.NONE
        elif k == key.TAB:
            self.current_action = None
            self.current_agent_index = self._increment_current_agent_index(
                self.current_agent_index
            )
            if self.display_info:
                agent_level = self.env.unwrapped.players[self.current_agent_index].level
                agent_position = self.env.unwrapped.players[self.current_agent_index].position
                print(f"Now selected: Agent {self.current_agent_index + 1} (Level {agent_level}, at {self._format_pos(agent_position)})")
        elif k == key.R:
            self.current_action = None
            self.reset = True
        elif k == key.H:
            self.current_action = None
            self._help()
        elif k == key.D:
            self.current_action = None
            self.display_info = not self.display_info
        elif k == key.ESCAPE:
            self.running = False
        else:
            self.current_action = None
            warnings.warn(f"Key {k} not recognized")

        if k in [key.LEFT, key.RIGHT, key.DOWN, key.UP, key.L, key.SPACE]:
            if self.current_agent_index in self.loading_agents:
                self.loading_agents.remove(self.current_agent_index)

    def _cycle(self):
        while self.running:
            if self.reset:
                if self.display_info:
                    print(f"Finished episode with episodic returns: {[round(ret, 3) for ret in self.ep_returns]}")
                    print()
                obss, _ = self.env.reset()
                self.reset = False
                self.ep_returns = np.zeros(self.n_agents)
                self.loading_agents = []
                self.t = 0

                if self.display_info:
                    self._display_info(obss, [0] * self.n_agents, False)

            if self.current_action is not None:
                actions = [Action.NONE if i not in self.loading_agents else Action.LOAD for i in range(self.n_agents)]
                actions[self.current_agent_index] = self.current_action
                obss, rews, done, trunc, info = self.env.step([act.value for act in actions])
                self.ep_returns += np.array(rews)
                self.t += 1

                if self.display_info:
                    self._display_info(obss, rews, done or trunc)

                if done or trunc:
                    self.reset = True

                self.current_action = None
            self.env.render()
        self.env.close()



if __name__ == "__main__":
    args = parse_args()
    InteractiveLBFEnv(env=args.env, display_info=args.display_info)
