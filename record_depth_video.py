import cv2
import numpy as np
from abc import abstractmethod
from inspirai_fps import Game, ActionVariable
from inspirai_fps.gamecore import AgentState
from typing import List, Tuple, Dict

ActionType = List[Tuple[str, int or float or bool]]


class Policy:
    def __init__(self, game_config: Dict) -> None:
        pass

    @abstractmethod
    def __call__(self, ts: int, state: AgentState) -> ActionType:
        raise NotImplementedError


def capture_frame(depth_map: np.ndarray, render_scale=1, far=100):
    img = (depth_map / far * 255).astype(np.uint8)
    h, w = [x * render_scale for x in img.shape]
    img = cv2.resize(img, (w, h))
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def create_video(images, video_name, fps=10):
    height, width = images[0].shape[:2]

    video = cv2.VideoWriter(video_name, 0, fps, (width, height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()


def record_one_episode(policy: Policy, game: Game, save_path: str, render_scale=1):
    game.new_episode()
    far = game.get_depth_map_size()[-1]

    images = []

    while not game.is_episode_finished():
        ts = game.get_time_step()
        state = game.get_state()
        action = policy(ts, state)
        game.make_action_by_list({0: action})
        img = capture_frame(state.depth_map, render_scale=render_scale, far=far)
        images.append(img)
        print("num_images:", len(images))

    create_video(images, save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dmp-width", type=int, default=42)
    parser.add_argument("--dmp-height", type=int, default=42)
    parser.add_argument("--dmp-far", type=int, default=200)
    parser.add_argument("--render-scale", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--map-id", type=int, default=1)
    parser.add_argument("--video-path", type=str, default="test_video.avi")
    parser.add_argument(
        "--map-dir", type=str, default="/mnt/d/Codes/cog-local/map-data"
    )
    parser.add_argument(
        "--engine-dir", type=str, default="/mnt/d/Codes/cog-local/fps_linux_train"
    )
    args = parser.parse_args()

    class MyPolicy(Policy):
        def __init__(self, game_config: Dict = None) -> None:
            super().__init__(game_config)

        def __call__(self, ts: int, state: AgentState) -> ActionType:
            return [
                (ActionVariable.WALK_DIR, 0),
                (ActionVariable.WALK_SPEED, 2),
                (ActionVariable.TURN_LR_DELTA, 1),
            ]

    policy = MyPolicy()
    game = Game(
        map_dir=args.map_dir,
        engine_dir=args.engine_dir,
        server_port=90000,
    )
    game.set_map_id(args.map_id)
    game.set_game_mode(Game.MODE_SUP_BATTLE)
    game.set_episode_timeout(args.timeout)
    game.turn_on_depth_map()
    game.set_depth_map_size(args.dmp_width, args.dmp_height, args.dmp_far)
    game.init()

    record_one_episode(
        policy, game, save_path=args.video_path, render_scale=args.render_scale
    )

    game.close()
