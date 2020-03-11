from gym.envs.registration import registry, register, make, spec
from itertools import product

sizes = range(5, 11)
# players = range(2, 5)
foods = range(1, 5)
coop = [True, False]

for s, f, c in product(sizes, foods, coop):
    register(
        id="Foraging-{0}x{0}-{1}f{2}-v0".format(s, f, "-coop" if c else ""),
        entry_point="lbforaging.foraging:ForagingEnv",
        kwargs={
            "players": 20,
            "max_player_level": 3,
            "field_size": (s, s),
            "max_food": f,
            "sight": s,
            "max_episode_steps": 50,
            "force_coop": c,
        },
    )
