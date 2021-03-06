import random
import numpy as np
from gym import spaces
from typing import Tuple
from ray.rllib.env import MultiAgentEnv, EnvContext
from ray.rllib.utils.typing import MultiAgentDict
from inspirai_fps import ActionVariable, Game
from inspirai_fps.gamecore import AgentState
from inspirai_fps.utils import get_distance, get_position
from pprint import pprint


class CooperativeSupplyGatherEnv(MultiAgentEnv):

    from common import COMMON_ACTION_CONFIG

    walk_speed = COMMON_ACTION_CONFIG["walk_speed"]

    ACTION_POOL = {
        "StopAndWalk": [
            [(ActionVariable.WALK_SPEED, 0)],
            [(ActionVariable.WALK_SPEED, walk_speed), (ActionVariable.WALK_DIR, 0)],
            [(ActionVariable.WALK_SPEED, walk_speed), (ActionVariable.WALK_DIR, 90)],
            [(ActionVariable.WALK_SPEED, walk_speed), (ActionVariable.WALK_DIR, 180)],
            [(ActionVariable.WALK_SPEED, walk_speed), (ActionVariable.WALK_DIR, 270)],
        ],
        "RotateAndJump": [
            [(ActionVariable.JUMP, True)],
            [(ActionVariable.TURN_LR_DELTA, -2)],
            [(ActionVariable.TURN_LR_DELTA, 0)],
            [(ActionVariable.TURN_LR_DELTA, 2)],
        ],
    }

    def __init__(self, env_config: EnvContext) -> None:
        super().__init__()
        self.env_config = env_config

        num_envs_per_worker = env_config["num_envs_per_worker"]
        self.game_port = (
            env_config["base_port"]
            + env_config.worker_index * num_envs_per_worker
            + env_config.vector_index
        )
        seed = env_config["random_seed"] + self.game_port
        self.seed(seed)

        self._agent_ids = set(range(env_config["num_agents"]))

        # build action and observation space
        self.action_space = spaces.Dict(
            {key: spaces.Discrete(len(val)) for key, val in self.ACTION_POOL.items()}
        )

        far = env_config["dmp_far"]
        width = env_config["dmp_width"]
        height = env_config["dmp_height"]

        self.observation_space = spaces.Dict(
            {
                "self_pos": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,)),
                "nearest_supply_pos": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,)),
                "nearby_supply_num": spaces.Box(low=0, high=np.Inf, shape=(1,)),
                "depth_map": spaces.Box(low=0, high=far, shape=(height, width)),
            }
        )

        # build game backend and set game parameters
        map_dir = env_config["map_dir"]
        engine_dir = env_config["engine_dir"]
        game = Game(map_dir, engine_dir, server_port=self.game_port)
        game.set_map_id(env_config["map_id"])
        game.set_game_mode(Game.MODE_SUP_BATTLE)
        game.set_random_seed(seed)
        game.set_episode_timeout(env_config["episode_timeout"])
        game.set_depth_map_size(width, height, far)
        game.turn_on_depth_map()
        for _ in range(1, env_config["num_agents"]):
            game.add_agent()

        # set supply distribution parameters
        game.set_supply_heatmap_center([0, 0])
        game.set_supply_heatmap_radius(env_config["supply_heatmap_radius"])
        game.set_supply_indoor_richness(0)
        game.set_supply_outdoor_richness(50)
        game.set_supply_indoor_quantity_range(1, 1)
        game.set_supply_outdoor_quantity_range(1, 1)
        game.set_supply_spacing(5)

        game.init()

        # store all game backend variables
        self.game = game
        self.valid_locations = self._get_valid_locations()

        # initialize key variables
        self.num_steps = 0
        self.num_supply_dict = {agent_id: 0 for agent_id in self._agent_ids}

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        processed_action_dict = {}
        for agent_id, action in action_dict.items():
            act = [(ActionVariable.PICKUP, True)]
            for a_type, a_idx in action.items():
                act.extend(self.ACTION_POOL[a_type][a_idx])
            processed_action_dict[agent_id] = act

        self.game.make_action_by_list(processed_action_dict)

        self.state_all = self.game.get_state_all()

        obs = {
            agent_id: self._get_obs(self.state_all[agent_id])
            for agent_id in action_dict
        }
        rewards = {agent_id: 0 for agent_id in action_dict}
        dones = {agent_id: False for agent_id in action_dict}
        infos = {agent_id: {} for agent_id in action_dict}

        for agent_id in action_dict:
            state = self.state_all[agent_id]
            if state.num_supply > self.num_supply_dict[agent_id]:
                rewards[agent_id] = 1
                self.num_supply_dict[agent_id] = state.num_supply

        dones["__all__"] = self.game.is_episode_finished()

        self.num_steps += 1
        
        if dones["__all__"]:
            worker_idx = self.env_config.worker_index
            supply_all = sum(self.num_supply_dict.values())
            print(f"[worker={worker_idx}] num_supply_total={supply_all}")

        return obs, rewards, dones, infos

    def reset(self):
        # sample start locations
        for agent_id in self._agent_ids:
            if agent_id != "__all__":
                start_location = self._sample_start_location()
                self.game.set_start_location(start_location, agent_id)

        self.game.new_episode()

        # reset key variables
        self.num_steps = 0
        self.num_supply_dict = {}
        for agent_id in self._agent_ids:
            if agent_id != "__all__":
                self.num_supply_dict[agent_id] = 0

        # get initial state
        self.state_all = self.game.get_state_all()

        print(f"[game_port={self.game_port}] reset")

        return {
            agent_id: self._get_obs(state) for agent_id, state in self.state_all.items()
        }

    def close(self) -> None:
        self.game.close()
        return super().close()

    def _get_obs(self, state: AgentState):
        self_pos = np.asarray(get_position(state))

        def _distance(supply_state):
            supply_pos = np.asarray(get_position(supply_state))
            return np.linalg.norm(self_pos - supply_pos)

        if len(state.supply_states) > 0:
            nearest_supply = min(state.supply_states.values(), key=_distance)
            nearest_supply_pos = np.asarray(get_position(nearest_supply))
        else:
            nearest_supply_pos = np.asarray([0, -100, 0])

        return {
            "self_pos": self_pos,
            "nearest_supply_pos": nearest_supply_pos,
            "nearby_supply_num": np.asarray([len(state.supply_states)]),
            "depth_map": state.depth_map,
        }

    def _sample_start_location(self):
        return random.choice(self.valid_locations)

    def _get_valid_locations(self):
        valid_locations = self.game.get_valid_locations()["outdoor"]
        if self.env_config["map_id"] == 101:

            def _in_map(location):
                x, _, z = location
                return -100 <= x <= 100 and -100 <= z <= 100

            return list(filter(_in_map, valid_locations))
        return valid_locations


if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.rllib.agents import ppo, a3c, impala

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-id", type=int, required=True, choices=[8, 101])
    parser.add_argument(
        "--algo", type=str, required=True, choices=["a3c", "ppo", "impala"]
    )
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    from common import (
        COMMON_ENV_CONFIG,
        COMMON_STOP_CONFIG,
        COMMON_PORT_CONFIG,
        TRAIN_EPISODE_TIMEOUT
    )

    base_port = COMMON_PORT_CONFIG[args.map_id]
    if args.test:
        base_port = COMMON_PORT_CONFIG["test"]

    train_env_config = COMMON_ENV_CONFIG.copy()
    train_env_config.update(
        {
            "map_id": args.map_id,
            "base_port": base_port,
            "episode_timeout": TRAIN_EPISODE_TIMEOUT,
            "in_evaluation": False,
            "supply_heatmap_radius": 80 if args.map_id == 101 else 200,
        }
    )

    train_config = {
        "env": CooperativeSupplyGatherEnv,
        "env_config": train_env_config,
        "framework": "torch",
        "num_workers": 160,
        "num_gpus": 0,
        "num_cpus_per_worker": 1,
        "num_envs_per_worker": 1,
    }

    pprint(train_config)

    ray.init()

    if args.algo == "a3c":
        run_algo = a3c.A3CTrainer
    elif args.algo == "ppo":
        run_algo = ppo.APPOTrainer
    elif args.algo == "impala":
        run_algo = impala.ImpalaTrainer
    else:
        raise ValueError(f"Unknown algo: {args.algo}")

    if args.test:
        trainer = run_algo(train_config)
        trainer.restore(args.checkpoint)
        result = trainer.evaluate()
        pprint(result)
        exit(0)

    analysis = tune.run(
        run_or_experiment=run_algo,
        name=f"Task=coop_supply-Algo={args.algo}-Map={args.map_id}",
        config=train_config,
        stop=COMMON_STOP_CONFIG,
        local_dir="results_coop_supply",
        checkpoint_freq=1,
        checkpoint_at_end=True,
    )
    pprint(analysis.dataframe())

    ray.shutdown()
