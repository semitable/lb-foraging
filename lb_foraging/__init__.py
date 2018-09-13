from gym.envs.registration import register
from .agent import Agent
from lb_foraging.envs.lb_foraging import Action, LBForagingEnv
register(
    id='lb-foraging-simple-v0',
    entry_point='lb_foraging.envs:LBForagingEnv',
    kwargs={
        'player_count': 2,
        'max_player_level': 2,
        'field_size': (8,8),
        'max_food': 5,
        'sight': 8,
    }
)
