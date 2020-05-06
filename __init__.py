from gym.envs.registration import register

register(id="AX_12-v0", entry_point="gym_cog_ml_tasks.envs:AX_12_ENV")
register(id="AX_S_12-v0", entry_point="gym_cog_ml_tasks.envs:AX_S_12_ENV")
register(id="AX_CPT-v0", entry_point="gym_cog_ml_tasks.envs:AX_CPT_ENV")
register(id="Saccade-v0", entry_point="gym_cog_ml_tasks.envs:Saccade_ENV")