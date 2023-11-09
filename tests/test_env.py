import pytest
import numpy as np
import lbforaging
from lbforaging.foraging.environment import Action
import gymnasium as gym


def manhattan_distance(x,y):
    return sum(abs(a-b) for a,b in zip(x,y))

@pytest.fixture
def simple2p1f():
    env = gym.make("Foraging-8x8-2p-1f-v2")
    env.reset()

    env.field[:] = 0

    env.field[4,4] = 2
    env._food_spawned = env.field.sum()

    env.players[0].position = (4,3)
    env.players[1].position = (4,5)

    env.players[0].level = 2
    env.players[1].level = 2
    assert env.test_gen_valid_moves()
    return env

@pytest.fixture
def simple2p1f_sight1():
    env = gym.make("Foraging-8x8-2p-1f-v2", sight=1)
    env.reset()

    env.field[:] = 0

    env.field[4,4] = 2
    env._food_spawned = env.field.sum()

    env.players[0].position = (4,3)
    env.players[1].position = (4,5)

    env.players[0].level = 2
    env.players[1].level = 2
    assert env.test_gen_valid_moves()
    return env

@pytest.fixture
def simple2p1f_sight2():
    env = gym.make("Foraging-8x8-2p-1f-v2", sight=2)
    env.reset()

    env.field[:] = 0

    env.field[4,4] = 2
    env._food_spawned = env.field.sum()

    env.players[0].position = (4,3)
    env.players[1].position = (4,5)

    env.players[0].level = 2
    env.players[1].level = 2
    assert env.test_gen_valid_moves()
    return env


def test_make():
    ''' Tests that we can make and reset the environments. '''
    enames = ["Foraging-8x8-2p-1f-v2",
              "Foraging-5x5-2p-1f-v2",
              "Foraging-8x8-3p-1f-v2",
              "Foraging-8x8-3p-1f-coop-v2"]
    for ename in enames:
        env = gym.make(ename)
        assert env is not None
        env.reset()


def test_spaces():
    pass


def test_seed():
    env = gym.make("Foraging-8x8-2p-2f-v2")
    for seed in range(10):
        obs1 = []
        obs2 = []
        env.seed(seed)
        for _ in range(10):
            temp_obs, _ = env.reset()
            obs1.append(temp_obs)
        env.seed(seed)
        for _ in range(10):
            temp_obs, _ = env.reset()
            obs2.append(temp_obs)

    for o1, o2 in zip(obs1, obs2):
        assert np.array_equal(o1, o2)


def test_food_spawning_0():
    env = gym.make("Foraging-6x6-2p-2f-v2")

    for _ in range(1000):
        env.reset()

        foods = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 2 foods
        assert len(foods) == 2

        #foods must not be within 2 steps of each other
        assert manhattan_distance(foods[0], foods[1]) > 2

        # food cannot be placed in first or last col/row
        assert foods[0][0] not in [0, 7]
        assert foods[0][1] not in [0, 7]
        assert foods[1][0] not in [0, 7]
        assert foods[1][1] not in [0, 7]

def test_food_spawning_1():
    env = gym.make("Foraging-8x8-2p-3f-v2")

    for _ in range(1000):
        env.reset()

        foods = [np.array(f) for f in zip(*env.field.nonzero())]
        # we should have 3 foods
        assert len(foods) == 3

        #foods must not be within 2 steps of each other
        assert manhattan_distance(foods[0], foods[1]) > 2
        assert manhattan_distance(foods[0], foods[2]) > 2
        assert manhattan_distance(foods[1], foods[2]) > 2

def test_reward_0(simple2p1f):
    _, rewards, _, _, _ = simple2p1f.step([5, 5])
    assert rewards[0] == 1.0
    assert rewards[1] == 1.0

def test_reward_1(simple2p1f):
    _, rewards, _, _, _ = simple2p1f.step([0, 5])
    assert rewards[0] == 0
    assert rewards[1] == 2.0

def test_partial_obs_1(simple2p1f_sight1):
    env = simple2p1f_sight1
    obs, _, _, _, _ = env.test_make_gym_obs()

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1

def test_partial_obs_2(simple2p1f_sight2):
    env = simple2p1f_sight2
    obs, _, _, _, _ = env.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] == -1
    assert obs[1][-2] == -1

def test_partial_obs_3(simple2p1f):
    env = simple2p1f
    obs, _, _, _, _ = env.test_make_gym_obs()

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1

    obs, _, _, _, _ = env.step([Action.WEST, Action.NONE])

    assert obs[0][-2] > -1
    assert obs[1][-2] > -1
