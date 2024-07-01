import gymnasium as gym
import numpy as np
import pytest


import lbforaging  # noqa
from lbforaging.foraging.environment import Action


lbforaging.register_envs()


def manhattan_distance(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


@pytest.fixture
def simple2p1f():
    env = gym.make("Foraging-8x8-2p-1f-v3")
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._food_spawned = env.field.sum()

    env.unwrapped.players[0].position = (4, 3)
    env.unwrapped.players[1].position = (4, 5)

    env.unwrapped.players[0].level = 2
    env.unwrapped.players[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()

    return env


@pytest.fixture
def simple2p1f_sight1():
    env = gym.make("Foraging-8x8-2p-1f-v3", sight=1)
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._food_spawned = env.field.sum()

    env.unwrapped.players[0].position = (4, 3)
    env.unwrapped.players[1].position = (4, 5)

    env.unwrapped.players[0].level = 2
    env.unwrapped.players[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()
    return env


@pytest.fixture
def simple2p1f_sight2():
    env = gym.make("Foraging-8x8-2p-1f-v3", sight=2)
    env.reset()

    env.unwrapped.field[:] = 0

    env.unwrapped.field[4, 4] = 2
    env.unwrapped._food_spawned = env.field.sum()

    env.unwrapped.players[0].position = (4, 3)
    env.unwrapped.players[1].position = (4, 5)

    env.unwrapped.players[0].level = 2
    env.unwrapped.players[1].level = 2
    assert env.unwrapped.test_gen_valid_moves()
    return env


def test_make():
    names = [
        "Foraging-8x8-2p-1f-v3",
        "Foraging-5x5-2p-1f-v3",
        "Foraging-8x8-3p-1f-v3",
        "Foraging-8x8-3p-1f-coop-v3",
    ]
    for name in names:
        env = gym.make(name)
        assert env is not None
        env.reset()


def test_spaces():
    pass


def test_seed():
    env = gym.make("Foraging-8x8-2p-2f-v3")
    for seed in range(10):
        obs1 = []
        obs2 = []
        env.seed(seed)
        for r in range(10):
            obs, _ = env.reset()
            obs1.append(obs)
        env.seed(seed)
        for r in range(10):
            obs, _ = env.reset()
            obs2.append(obs)

    for o1, o2 in zip(obs1, obs2):
        assert np.array_equal(o1, o2)


def test_food_spawning_0():
    env = gym.make("Foraging-6x6-2p-2f-v3")

    for i in range(1000):
        env.reset()

        foods = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 2 foods
        assert len(foods) == 2

        # foods must not be within 2 steps of each other
        assert manhattan_distance(foods[0], foods[1]) > 2

        # food cannot be placed in first or last col/row
        assert foods[0][0] not in [0, 7]
        assert foods[0][1] not in [0, 7]
        assert foods[1][0] not in [0, 7]
        assert foods[1][1] not in [0, 7]


def test_food_spawning_1():
    env = gym.make("Foraging-8x8-2p-3f-v3")

    for i in range(1000):
        env.reset()

        foods = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 3 foods
        assert len(foods) == 3

        # foods must not be within 2 steps of each other
        assert manhattan_distance(foods[0], foods[1]) > 2
        assert manhattan_distance(foods[0], foods[2]) > 2
        assert manhattan_distance(foods[1], foods[2]) > 2


def test_reward_0(simple2p1f):
    _, rewards, _, _, _ = simple2p1f.step([5, 5])
    assert rewards[0] == 0.5
    assert rewards[1] == 0.5


def test_reward_1(simple2p1f):
    _, rewards, _, _, _ = simple2p1f.step([0, 5])
    assert rewards[0] == 0
    assert rewards[1] == 1


def test_reward_2(simple2p1f):
    _, rewards, _, _, _ = simple2p1f.step([5, 0])
    assert rewards[0] == 1
    assert rewards[1] == 0


def test_partial_obs_1(simple2p1f_sight1):
    env = simple2p1f_sight1
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1


def test_partial_obs_2(simple2p1f_sight2):
    env = simple2p1f_sight2
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1


def test_partial_obs_3(simple2p1f):
    env = simple2p1f
    obs = env.unwrapped.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1


def test_reproducibility(simple2p1f):
    env = simple2p1f
    episodes_per_seed = 5
    for seed in range(5):
        obss1 = []
        field1 = []
        player_positions1 = []
        player_levels1 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss1.append(np.array(obss).copy())
            field1.append(env.unwrapped.field.copy())
            player_positions1.append([p.position for p in env.unwrapped.players])
            player_levels1.append([p.level for p in env.unwrapped.players])

        obss2 = []
        field2 = []
        player_positions2 = []
        player_levels2 = []
        env.seed(seed)
        for _ in range(episodes_per_seed):
            obss, _ = env.reset()
            obss2.append(np.array(obss).copy())
            field2.append(env.unwrapped.field.copy())
            player_positions2.append([p.position for p in env.unwrapped.players])
            player_levels2.append([p.level for p in env.unwrapped.players])

        print("Seed: ", seed)
        for obs1, obs2 in zip(obss1, obss2):
            print(obs1)
            print(obs2)
            print(np.array_equal(obs1, obs2))
            print()

        for i, (obs1, obs2) in enumerate(zip(obss1, obss2)):
            assert np.array_equal(
                obs1, obs2
            ), f"Observations of env not identical for episode {i} with seed {seed}"
        for i, (field1, field2) in enumerate(zip(field1, field2)):
            assert np.array_equal(
                field1, field2
            ), f"Fields of env not identical for episode {i} with seed {seed}"
        for i, (player_positions1, player_positions2) in enumerate(
            zip(player_positions1, player_positions2)
        ):
            assert (
                player_positions1 == player_positions2
            ), f"Player positions of env not identical for episode {i} with seed {seed}"
        for i, (player_levels1, player_levels2) in enumerate(
            zip(player_levels1, player_levels2)
        ):
            assert (
                player_levels1 == player_levels2
            ), f"Player levels of env not identical for episode {i} with seed {seed}"
