'''Basic flow to see if the base install worked over one environment.'''
import argparse
import logging
import time
import numpy as np
import gymnasium as gym
import lbforaging


logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """
    """
    _, _ = env.reset()
    done = False

    if render:
        env.render()
        time.sleep(0.5)

    while not done:

        actions = env.action_space.sample()

        _, nreward, ndone, _, _ = env.step(actions)
        if sum(nreward) > 0:
            print(nreward)

        if render:
            env.render()
            time.sleep(0.5)

        done = np.all(ndone)
    # print(env.players[0].score, env.players[1].score)


def main(game_count=1, render=False):
    env = gym.make("Foraging-8x8-2p-2f-v2")

    _, info = env.reset()
    assert info == {}

    for _ in range(game_count):
        _game_loop(env, render)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)

    print("Done. NO RUNTIME ERRORS.")
