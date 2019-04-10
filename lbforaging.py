import argparse
import logging
import random
import time
import gym
import lbforaging.foraging


logger = logging.getLogger(__name__)


def _game_loop(env, render):
    """
    """
    obs = env.reset()

    if render:
        env.render()
        time.sleep(1)

    for _ in range(_MAX_STEPS):
        actions = []
        for i, player in enumerate(env.players):
            actions.append(player.step(obs[i]))
        obs = env.step(actions)

        if render:
            env.render()
            time.sleep(1)

        if env.game_over:
            break

    # loop once more so that agents record the game over event
    for i, player in enumerate(env.players):
        player.step(obs[i])


def evaluate(
    players,
    game_count,
    render,
    max_player_level=None,
    field_size=(8, 8),
    food_count=5,
    sight=None,
):
    if sight is None:
        sight = max(*field_size)

    if max_player_level is None:
        max_player_level = len(
            players
        )  # max player level becomes the number of players

    env = Env(
        players=players,
        max_player_level=max_player_level,
        field_size=field_size,
        max_food=food_count,
        sight=sight,
    )

    efficiency = 0
    flexibility = 0

    pbar = tqdm if game_count > 1 else (lambda x: x)  # use tqdm for game_count >1

    for _ in pbar(range(game_count)):
        _game_loop(env, render)

        efficiency += env.players[0].score / env.current_step
        if env.current_step < _MAX_STEPS:
            flexibility += 1

    for player in env.players:
        player.controller.cleanup()

    efficiency /= game_count
    flexibility /= game_count

    logger.warning(
        "{} - Efficiency: {} - Flexibility: {}".format(
            players[0].name, efficiency, flexibility
        )
    )


def main(game_count=1, render=False):
    env = gym.make("Foraging-v0")
    obs = env.reset()
    print(env.action_space)
    print(env.observation_space)
    print(obs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play the level foraging game.")

    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--times", type=int, default=1, help="How many times to run the game"
    )

    args = parser.parse_args()
    main(args.times, args.render)
