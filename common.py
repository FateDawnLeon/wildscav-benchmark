COMMON_PORT_CONFIG = {
    8: 60000,
    101: 70000,
    "test": 80000,
}

COMMON_ACTION_CONFIG = {
    "walk_speed": 8,
}

COMMON_STOP_CONFIG = {
    "timesteps_total": 2e7,
}

COMMON_TRAIN_CONFIG = {
    "framework": "torch",
    "num_workers": 4,
    "num_cpus_per_worker": 1,
    "num_envs_per_worker": 1,
    "evaluation_interval": None,
    "evaluation_duration": 10,
    "evaluation_duration_unit": "episodes",
    "evaluation_num_workers": 2,
}

COMMON_ENV_CONFIG = {
    "num_agents": 4,
    "random_seed": 0,
    "num_envs_per_worker": 1,
    "episode_timeout": 120,
    "dmp_far": 200,
    "dmp_width": 42,
    "dmp_height": 42,
    "map_dir": "/mnt/d/Codes/cog-local/map-data-benchmark",
    "engine_dir": "/mnt/d/Codes/cog-local/fps_linux_benchmark",
    "trigger_range": 2,
}

TRAIN_START_RANGE = 50
TRAIN_EPISODE_TIMEOUT = 30
