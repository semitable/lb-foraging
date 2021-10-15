import argparse
import logging
import random
import time
import gym
import numpy as np
import lbforaging


logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """
    """
    obs = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:

        actions = env.action_space.sample()

        nobs, nreward, ndone, _ = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)
    # print(env.players[0].score, env.players[1].score)


def main(game_count=1, render=False):
    env = gym.make("Foraging-8x8-2p-2f-v2")
    obs = env.reset()

    for episode in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
