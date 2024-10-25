from environment import ForagingEnv

env = ForagingEnv(
        grid_observation=True,
        players=2,
        max_player_level=2,
        field_size=(8,8),
        max_num_food=2,
        sight=8,
        force_coop=True,
        min_player_level=1,
        min_food_level=1,
        max_food_level=10,
        max_episode_steps=100)

nobs, infos = env.reset()
actions = [5,5,0]
nobs, rewards, done, truncated, info   = env.step(actions=actions)
print(f"nobs: \n {nobs[0].shape}")