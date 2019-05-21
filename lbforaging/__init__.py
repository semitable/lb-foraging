from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 11)
players = range(2, 5)
foods = range(1, 5)

for s, p, f in product(sizes, players, foods):
    register(
        id="Foraging-{0}x{0}-{1}p-{2}f-v0".format(s, p, f),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": p,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 50,
        },
    )
