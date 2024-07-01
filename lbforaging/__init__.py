from itertools import product

from gymnasium import register


sizes = range(5, 20)
players = range(2, 20)
foods = range(1, 10)
max_food_level = [None, 1]
coop = [True, False]
partial_obs = [True, False]
pens = [True, False]


register(
    id="Foraging-simple-v3",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (8, 8),
        "min_food_level": 1,
        "max_food_level": None,
        "max_num_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": False,
        "grid_observation": False,
    },
)

register(
    id="Foraging-2s-simple-v3",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (8, 8),
        "min_food_level": 1,
        "max_food_level": None,
        "max_num_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": False,
        "grid_observation": False,
    },
)

register(
    id="Foraging-coop-v3",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (8, 8),
        "min_food_level": 1,
        "max_food_level": None,
        "max_num_food": 2,
        "sight": 8,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)

register(
    id="Foraging-2s-coop-v3",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 2,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (8, 8),
        "min_food_level": 1,
        "max_food_level": None,
        "max_num_food": 2,
        "sight": 2,
        "max_episode_steps": 50,
        "force_coop": True,
        "grid_observation": False,
    },
)


def register_envs():
    for s, p, f, mfl, c, po, pen in product(
        sizes, players, foods, max_food_level, coop, partial_obs, pens
    ):
        register(
            id="Foraging{4}-{0}x{0}-{1}p-{2}f{3}{5}{6}-v3".format(
                s,
                p,
                f,
                "-coop" if c else "",
                "-2s" if po else "",
                "-ind" if mfl else "",
                "-pen" if pen else "",
            ),
            entry_point="lbforaging.foraging:ForagingEnv",
            kwargs={
                "players": p,
                "min_player_level": 1,
                "max_player_level": 2,
                "field_size": (s, s),
                "min_food_level": 1,
                "max_food_level": mfl,
                "max_num_food": f,
                "sight": 2 if po else s,
                "max_episode_steps": 50,
                "force_coop": c,
                "grid_observation": False,
                "penalty": 0.1 if pen else 0.0,
            },
        )


def register_grid_envs():
    for s, p, f, mfl, c in product(sizes, players, foods, max_food_level, coop):
        for sight in range(1, s + 1):
            register(
                id="Foraging-grid{4}-{0}x{0}-{1}p-{2}f{3}{5}-v3".format(
                    s,
                    p,
                    f,
                    "-coop" if c else "",
                    "" if sight == s else f"-{sight}s",
                    "-ind" if mfl else "",
                ),
                entry_point="lbforaging.foraging:ForagingEnv",
                kwargs={
                    "players": p,
                    "min_player_level": 1,
                    "max_player_level": 2,
                    "field_size": (s, s),
                    "min_food_level": 1,
                    "max_food_level": mfl,
                    "max_num_food": f,
                    "sight": sight,
                    "max_episode_steps": 50,
                    "force_coop": c,
                    "grid_observation": True,
                },
            )
