import random
import numpy as np
from inspirai_fps import ActionVariable
from inspirai_fps.utils import get_distance, get_position, get_picth_yaw


def policy_1(ts, state, **kwargs):
    return [
        (ActionVariable.WALK_DIR, 0),
        (ActionVariable.WALK_SPEED, 2),
        (ActionVariable.TURN_LR_DELTA, 1),
    ]


def policy_2(ts, state, **kwargs):
    if state.pitch < 45:
        look_ud_delta = -1
    else:
        look_ud_delta = 0

    return [
        (ActionVariable.TURN_LR_DELTA, 3),
        (ActionVariable.LOOK_UD_DELTA, look_ud_delta),
    ]


def policy_3(ts, state, **kwargs):
    self_loc = np.asarray(get_position(state))
    target_loc = np.asarray(kwargs["target_loc"])
    direction = target_loc - self_loc
    walk_dir = get_picth_yaw(*direction)[1]

    return [
        (ActionVariable.WALK_DIR, walk_dir),
        (ActionVariable.WALK_SPEED, 5),
    ]


def policy_4(ts, state, **kwargs):
    # compute walking direction
    self_loc = np.asarray(get_position(state))
    target_loc = np.asarray(kwargs["target_loc"])
    direction = target_loc - self_loc
    walk_dir = get_picth_yaw(*direction)[1]

    def get_supply_distance(supply_state):
        supply_loc = np.asarray(get_position(supply_state))
        return np.linalg.norm(supply_loc - self_loc)

    # compute supply direction
    supplies = state.supply_states.values()
    if supplies:
        nearest_supply = min(supplies, key=get_supply_distance)
        supply_loc = np.asarray(get_position(nearest_supply))
        supply_dir = supply_loc - self_loc
        walk_dir = get_picth_yaw(*supply_dir)[1]

    return [
        (ActionVariable.WALK_DIR, walk_dir),
        (ActionVariable.WALK_SPEED, 5),
        (ActionVariable.TURN_LR_DELTA, 1),
        (ActionVariable.PICKUP, True),
    ]


class SupplyBattlePolicy:
    def __init__(self, **kwargs):
        self.target_locations = kwargs["target_locations"]
        self.agent_name = kwargs["agent_name"]
        self.curr_target_loc = None
        self.last_self_loc = None
        self.num_supply = 0
        self.last_pickup_timestep = None
        self.explore_step_budget = 0

    def _sample_target(self):
        self.curr_target_loc = np.asarray(random.choice(self.target_locations))

    def _compute_walk_dir(self, self_loc):
        direction = self.curr_target_loc - self_loc
        walk_dir = get_picth_yaw(*direction)[1]
        return walk_dir

    def _policy_explore(self, ts, state, **kwargs):
        assert self.explore_step_budget > 0
        self.explore_step_budget -= 1

        self._sample_target()

        self_loc = np.asarray(get_position(state))
        walk_dir = self._compute_walk_dir(self_loc)
        walk_speed = 10
        fire = ts % (50 * 3) == 0
        turn_lr_delta = 1

        self.last_self_loc = self_loc

        return [
            (ActionVariable.WALK_DIR, walk_dir),
            (ActionVariable.WALK_SPEED, walk_speed),
            (ActionVariable.TURN_LR_DELTA, turn_lr_delta),
            (ActionVariable.ATTACK, fire),
        ]

    def _policy_goto_supply(self, ts, state, **kwargs):
        self_loc = np.asarray(get_position(state))
        
        def get_supply_distance(supply_state):
            supply_loc = get_position(supply_state)
            return get_distance(self_loc, supply_loc)

        if state.supply_states:
            supplies = state.supply_states.values()
            nearest_supply = min(supplies, key=get_supply_distance)
            self.curr_target_loc = np.asarray(get_position(nearest_supply))
            walk_dir = self._compute_walk_dir(self_loc)
            walk_speed = 5
        else:
            walk_dir = 0
            walk_speed = 0

        self.last_self_loc = self_loc
        
        return [
            (ActionVariable.WALK_DIR, walk_dir),
            (ActionVariable.WALK_SPEED, walk_speed),
            (ActionVariable.PICKUP, True),
        ]

    def __call__(self, ts, state, **kwargs):
        if state.num_supply > self.num_supply:
            self.num_supply = state.num_supply
            self.last_pickup_timestep = ts
            print(f"[{self.agent_name}] #Supply={self.num_supply} @ timestep={ts}")

        if self.explore_step_budget > 0:
            return self._policy_explore(ts, state, **kwargs)
        
        self_loc = np.asarray(get_position(state))

        if self.last_self_loc is not None and np.linalg.norm(self_loc - self.last_self_loc) < 0.1:
            self.explore_step_budget = 50
            return self._policy_explore(ts, state, **kwargs)

        return self._policy_goto_supply(ts, state, **kwargs)


if __name__ == "__main__":
    import os
    import argparse
    from inspirai_fps import Game

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--map-dir", type=str, default="/mnt/d/Codes/cog-local/map-data"
    )
    parser.add_argument(
        "--engine-dir", type=str, default="/mnt/d/Codes/cog-local/fps_linux_train"
    )
    parser.add_argument(
        "--replay-file-store-dir",
        type=str,
        default="/mnt/d/COG2022/offline_training/fps_win_replay/FPSGameUnity_Data/StreamingAssets/Replay",
    )
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--map-id", type=int, default=1)
    parser.add_argument("--start-location", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--target-location", type=float, nargs=3, default=[1, 0, 1])
    parser.add_argument("--random-start-location", action="store_true")
    parser.add_argument("--random-target-location", action="store_true")
    parser.add_argument("--policy", type=str, default="policy_1")
    parser.add_argument("--replay-suffix", type=str, default=None)
    parser.add_argument("--num-agents", type=int, default=1)
    args = parser.parse_args()

    game = Game(engine_dir=args.engine_dir, map_dir=args.map_dir, server_port=66666)
    game.init()

    game.set_map_id(args.map_id)
    game.set_game_mode(Game.MODE_SUP_BATTLE)
    game.set_episode_timeout(args.timeout)
    game.set_start_location(args.start_location)
    game.set_target_location(args.target_location)
    
    for id in range(1, args.num_agents):
        game.add_agent()
    
    if args.random_start_location:
        for id in range(args.num_agents):
            game.random_start_location(agent_id=id)
    
    if args.random_target_location:
        game.random_target_location()

    game.set_supply_heatmap_center([0, 0])
    game.set_supply_heatmap_radius(150)
    game.set_supply_outdoor_richness(30)
    game.set_supply_indoor_richness(70)
    game.set_supply_indoor_quantity_range(10, 100)
    game.set_supply_outdoor_quantity_range(1, 10)
    game.set_supply_spacing(5)

    from attrs import define

    @define
    class Location:
        x: float = 0
        y: float = 0
        z: float = 0

    others = {
        "target_loc": game.get_target_location(),
    }

    game.turn_on_record()
    game.set_game_replay_suffix(args.replay_suffix)
    game.new_episode()
    
    policies = {}
    if args.policy == "sup_battle":
        game_info = {
            "target_locations": game.get_valid_locations()["outdoor"],
        }
        for agent_id in range(args.num_agents):
            game_info["agent_name"] = game.get_agent_name(agent_id)
            policies[agent_id] = SupplyBattlePolicy(**game_info)
    else:
        for agent_id in range(args.num_agents):
            policies[agent_id] = eval(args.policy)
    
    while not game.is_episode_finished():
        ts = game.get_time_step()
        state_all = game.get_state_all()
        action_all = {}
        for agent_id, state in state_all.items():
            action_all[agent_id] = policies[agent_id](ts, state, **others)
        game.make_action_by_list(action_all)

        # state = state_all[0]
        # loc = Location(x=state.position_x, y=state.position_y, z=state.position_z)
        # print(f"{ts=} {loc=}")

    game.close()

    import time

    time.sleep(3)

    replay_dir = os.path.join(args.engine_dir, "fps_Data/StreamingAssets/Replay")

    # find the last created replay file under the replay_dir
    import glob

    replay_files = glob.glob(os.path.join(replay_dir, "*.bin"))
    replay_files.sort(key=os.path.getmtime)
    replay_file = replay_files[-1]

    # copy the replay file to the the replay file store dir
    import shutil

    shutil.copy(replay_file, args.replay_file_store_dir)
