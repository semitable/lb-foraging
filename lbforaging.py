import argparse
import logging
import time

import gymnasium as gym
import numpy as np

import lbforaging  # noqa

logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """ """
    obss, _ = env.reset()
    done = False

    returns = np.zeros(env.n_agents)

    if render:
        env.render()
        time.sleep(0.5)

    while not done:
        actions = env.action_space.sample()

        obss, rewards, done, _, _ = env.step(actions)
        returns += rewards

        if render:
            env.render()
            time.sleep(0.5)

    print("Returns: ", returns)


def main(episodes=1, render=False):
    # lbforaging.register_envs()
    env = gym.make("Foraging-simple-v3")

    for episode in range(episodes):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--episodes", type=int, default=1, help="How many episodes to run"
    )

    args = parser.parse_args()
    main(args.episodes, args.render)
