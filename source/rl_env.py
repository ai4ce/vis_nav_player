"""
A Gymnasium environment on the local simulator, for reinforcement learning.

    uv run source/rl_env.py --mazes 7,8,9      # random policy, prints rewards

The simulator runs in this process (`vis_nav_sim`), so an episode is a few thousand frames
a second and a new maze is a millisecond. The reward uses the robot's true position, which
only exists locally; a policy trained here is deployed on the server with `rl_train.py`'s
`RLAgent`, which sees exactly what any agent sees: frames.

    observation   uint8 (6, 60, 80): the camera frame and target view 0, downscaled,
                  channels first. Change `observe()` to give the policy something else.
    action        Discrete(4): FORWARD, BACKWARD, LEFT, RIGHT, each held `ticks` ticks.
    reward        metres closed toward the goal this step, +10 on arrival (distance <= 0.2 m
                  and heading <= 30 degrees, the rubric's goal), -0.01 per step.
    episode       ends on arrival or after `max_decisions` steps. reset() picks the next
                  maze from `mazes` (domain randomization) and a fresh motion noise seed.

Requires gymnasium and vis-nav-sim: `uv sync --extra rl`.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

import cv2
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    raise SystemExit("rl_env needs gymnasium: run `uv sync --extra rl`") from None

import vis_nav_sim as sim

ACTIONS = (sim.FORWARD, sim.BACKWARD, sim.LEFT, sim.RIGHT)
ACTION_NAMES = ("FORWARD", "BACKWARD", "LEFT", "RIGHT")
GOAL_DISTANCE_M = 0.2
GOAL_HEADING_DEG = 30.0
ARRIVAL_BONUS = 10.0
STEP_PENALTY = 0.01


def observe(frame: np.ndarray, target: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """The policy's input from the current frame and target view 0: both downscaled to
    `size` (width, height) and stacked channels-first. BGR, as the SDK gives them."""
    small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    goal = cv2.resize(target, size, interpolation=cv2.INTER_AREA)
    return np.concatenate([small, goal], axis=2).transpose(2, 0, 1).copy()


class VisNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        mazes: Sequence[int] = (7,),
        *,
        textures: sim.Textures | None = None,
        ticks: int = 4,
        size: tuple[int, int] = (80, 60),
        max_decisions: int = 500,
        motion_noise: bool = True,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.mazes = list(mazes)
        self.ticks = ticks
        self.size = size
        self.max_decisions = max_decisions
        self.render_mode = render_mode
        self._textures = textures or sim.Textures.find()
        self._noise = sim.DEFAULT_NOISE if motion_noise else None
        self._world: sim.Simulator | None = None
        self._episode = 0
        self._decisions = 0
        self._distance = 0.0
        self._target0: np.ndarray | None = None
        self._frame: np.ndarray | None = None
        width, height = size
        self.observation_space = spaces.Box(0, 255, shape=(6, height, width), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(ACTIONS))

    # -- gymnasium

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        maze_seed = (options or {}).get("maze_seed")
        if maze_seed is None:
            maze_seed = self.mazes[self._episode % len(self.mazes)]
        motion_seed = int(self.np_random.integers(0, 2**31))
        if self._world is None:
            self._world = sim.Simulator(
                self._textures, maze_seed, motion_noise=self._noise, motion_seed=motion_seed
            )
            self._frame = self._world.initial_frame()
        else:
            self._frame = self._world.reset(seed=maze_seed, motion_seed=motion_seed)
        self._target0 = self._world.targets()[0]
        self._episode += 1
        self._decisions = 0
        self._distance = self._world.distance_heading()[0]
        return observe(self._frame, self._target0, self.size), self._info()

    def step(self, action: int):
        assert self._world is not None and self._target0 is not None, "reset() first"
        self._frame = self._world.step(ACTIONS[int(action)], self.ticks)[-1]
        self._decisions += 1
        distance, heading = self._world.distance_heading()
        reward = (self._distance - distance) - STEP_PENALTY
        self._distance = distance
        arrived = distance <= GOAL_DISTANCE_M and heading <= GOAL_HEADING_DEG
        if arrived:
            reward += ARRIVAL_BONUS
        truncated = self._decisions >= self.max_decisions
        observation = observe(self._frame, self._target0, self.size)
        return observation, reward, arrived, truncated, self._info()

    def render(self):
        return None if self._frame is None else self._frame[:, :, ::-1]  # RGB for gym

    # -- the truth, for reward and diagnostics

    def _info(self) -> dict:
        assert self._world is not None
        x, y, yaw = self._world.pose()
        distance, heading = self._world.distance_heading()
        return {
            "maze": self._world.seed,
            "x": x,
            "y": y,
            "yaw_deg": math.degrees(yaw),
            "distance_m": distance,
            "heading_deg": heading,
            "steps": self._world.steps,
            "oracle_steps": self._world.oracle_steps,
        }

    @property
    def world(self) -> sim.Simulator:
        assert self._world is not None
        return self._world


def main() -> None:
    parser = argparse.ArgumentParser(description="drive a random policy through the environment")
    parser.add_argument("--mazes", default="7", help="comma-separated maze seeds")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=200)
    args = parser.parse_args()
    env = VisNavEnv([int(s) for s in args.mazes.split(",")], max_decisions=args.max_decisions)
    print("textures:", env._textures)
    for _ in range(args.episodes):
        obs, info = env.reset()
        total = 0.0
        done = False
        while not done:
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        print(
            f"maze {info['maze']}: {env._decisions} decisions, {info['steps']} ticks, "
            f"return {total:+.2f}, ended {info['distance_m']:.2f} m from the goal"
            + (" (arrived)" if terminated else "")
        )


if __name__ == "__main__":
    main()
