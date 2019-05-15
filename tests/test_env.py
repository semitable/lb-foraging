import pytest
import numpy as np
import lbforaging
import gym


def test_make():
    env = gym.make("Foraging-8x8-2p-1f-v0")
    env = gym.make("Foraging-5x5-2p-1f-v0")
    env = gym.make("Foraging-8x8-3p-1f-v0")


def test_spaces():
    pass

def test_seed():
    env = gym.make("Foraging-8x8-2p-2f-v0")
    env.seed(1)
    obs1 = env.reset()
    env = gym.make("Foraging-8x8-2p-2f-v0")
    env.seed(1)
    obs2 = env.reset()
    print(obs1)
    print(obs2)

    assert np.array_equal(obs1, obs2)
