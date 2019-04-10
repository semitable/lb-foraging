from gym.envs.registration import registry, register, make, spec

register(
    id="Foraging-v0",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "max_player_level": 2,
        "field_size": (8, 8),
        "max_food": 4,
        "sight": 8,
    },
    max_episode_steps=250,
)
