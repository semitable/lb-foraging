from gym.envs.registration import registry, register, make, spec

register(
    id="Foraging-8x8-2p-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 3,
        "sight": 8,
        "max_episode_steps": 50,
    },
)

register(
    id="Foraging-5x5-2p-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 3,
        "field_size": (5, 5),
        "max_food": 1,
        "sight": 5,
        "max_episode_steps": 50,
    },
)

register(
    id="Foraging-8x8-3p-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 3,
        "max_player_level": 3,
        "field_size": (8, 8),
        "max_food": 3,
        "sight": 8,
        "max_episode_steps": 50,
    },
)
