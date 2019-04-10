import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging.foraging


logger = logging.getLogger(__name__)

EPISODES = 500

def _game_loop(env, render):
    """
    """
    obs = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(1)

    while not done:

        actions = []
        for i, player in enumerate(env.players):
            actions.append(env.action_space.sample())
        print(actions)
        nobs, nreward, ndone, _ = env.step(actions)
        print(actions)

        if render:
            env.render()
            time.sleep(1)

        done = np.all(ndone)


def main(game_count=1, render=False):
    env = gym.make("Foraging-v0")
    obs = env.reset()

    for episode in range(EPISODES):
        _game_loop(env, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
