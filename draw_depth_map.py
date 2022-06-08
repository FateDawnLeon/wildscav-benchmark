from inspirai_fps import Game, ActionVariable


class PolicyPool:
    @staticmethod
    def policy_1(ts, state):
        return [
            (ActionVariable.WALK_DIR, 0),
            (ActionVariable.WALK_SPEED, 2),
            (ActionVariable.TURN_LR_DELTA, 1),
        ]

    @staticmethod
    def policy_2(ts, state):
        if state.pitch < 45:
            look_ud_delta = -1
        else:
            look_ud_delta = 0

        return [
            (ActionVariable.TURN_LR_DELTA, 3),
            (ActionVariable.LOOK_UD_DELTA, look_ud_delta),
        ]

    @staticmethod
    def policy_3(ts, state):
        if state.pitch < 30:
            look_ud_delta = -1
        else:
            look_ud_delta = 0

        return [
            (ActionVariable.TURN_LR_DELTA, 0),
            (ActionVariable.LOOK_UD_DELTA, look_ud_delta),
        ]
    
    @staticmethod
    def policy_4(ts, state):
        if state.pitch < 15:
            look_ud_delta = -1
        else:
            look_ud_delta = 1

        return [
            (ActionVariable.LOOK_UD_DELTA, look_ud_delta),
        ]


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="dmp")
    parser.add_argument(
        "--map-dir", type=str, default="/mnt/d/Codes/cog-local/map-data"
    )
    parser.add_argument(
        "--engine-dir", type=str, default="/mnt/d/Codes/cog-local/fps_linux_train"
    )
    parser.add_argument("--dmp-width", type=int, default=42)
    parser.add_argument("--dmp-height", type=int, default=42)
    parser.add_argument("--dmp-far", type=int, default=200)
    parser.add_argument("--render-scale", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--map-id", type=int, default=1)
    parser.add_argument("--start-location", type=int, nargs=3, default=(0, 0, 0))
    parser.add_argument("--random-start-location", action="store_true")
    parser.add_argument("--policy", type=str, default="policy_1")
    args = parser.parse_args()

    import cv2
    from record_depth_video import capture_frame

    game = Game(map_dir=args.map_dir, engine_dir=args.engine_dir, server_port=88888)
    game.set_map_id(args.map_id)
    game.set_episode_timeout(args.timeout)
    game.turn_on_depth_map()
    game.set_depth_map_size(args.dmp_width, args.dmp_height, args.dmp_far)
    game.set_start_location(args.start_location)
    if args.random_start_location:
        game.random_start_location()

    game.init()
    game.new_episode()
    w, h, f = game.get_depth_map_size()
    r = args.render_scale

    os.makedirs(args.save_dir, exist_ok=True)

    policy = getattr(PolicyPool, args.policy)

    while not game.is_episode_finished():
        ts = game.get_time_step()
        state = game.get_state()
        action = policy(ts, state)
        game.make_action_by_list({0: action})
        img = capture_frame(state.depth_map, render_scale=r, far=f)
        img_name = f"map={args.map_id}_{w=}_{h=}_{f=}_{r=}_{ts=}.png"
        save_path = os.path.join(args.save_dir, img_name)
        cv2.imwrite(save_path, img)
        print("saved image:", save_path)

    game.close()
