COMMON_PORT_CONFIG = {
    8: 60000,
    101: 70000,
    "test": 80000,
}

COMMON_ACTION_CONFIG = {
    "walk_speed": 8,
}

COMMON_STOP_CONFIG = {
    "timesteps_total": 5e7,
}

COMMON_TRAIN_CONFIG = {
    "framework": "torch",
    "num_workers": 80,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "evaluation_interval": 10,
    "evaluation_duration": 100,
    "evaluation_duration_unit": "episodes",
    "evaluation_num_workers": 50,
    "num_gpus":0,
}

COMMON_ENV_CONFIG = {
    "num_agents": 4,
    "random_seed": 0,
    "num_envs_per_worker": 1,
    "episode_timeout": 180,
    "dmp_far": 200,
    "dmp_width": 42,
    "dmp_height": 42,
    "map_dir": "/root/map-data",
    "engine_dir": "/root/game-engine",
    "trigger_range": 2,
}

TRAIN_START_RANGE = 50
TRAIN_EPISODE_TIMEOUT = 180
