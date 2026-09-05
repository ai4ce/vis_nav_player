"""
Train a policy on the local simulator, then drive it on the server.

    uv run source/rl_train.py train --mazes 7,8,9,10 --steps 200000   # -> models/ppo.zip
    uv run source/rl_train.py play  --local 11                         # a maze it never saw
    uv run source/rl_train.py play  --challenge <id>                   # one real attempt

`train` runs PPO (stable-baselines3) on `rl_env.VisNavEnv` over the listed mazes; the
reward uses the robot's true position, which the local simulator has and the server does
not. `play` wraps the saved policy in an `Agent` that sees only frames, and hands it to
`vis_nav_sdk.run()` like any other agent: `--local` for another maze here, `--challenge`
for an attempt on the server, where the same frames come from the same renderer.

This is a starting point, not a solution: 200k steps of PPO on raw pixels learns to move
and turn, not to navigate a maze it has never seen. The environment, the observation and
the reward are yours to change.

Requires `uv sync --extra rl` (gymnasium, stable-baselines3, and with it torch).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cli
import numpy as np
from rl_env import ACTION_NAMES, ACTIONS, VisNavEnv, observe
from vis_nav_sdk import Action, Agent, Observation, SessionInfo, run

MODEL = Path("models") / "ppo.zip"


def _sb3():
    try:
        import stable_baselines3 as sb3
    except ImportError:
        raise SystemExit("training needs stable-baselines3: run `uv sync --extra rl`") from None
    return sb3


def train(args: argparse.Namespace) -> None:
    sb3 = _sb3()

    mazes = [int(s) for s in args.mazes.split(",")]
    # The observation is already channels-first uint8, which SB3's CnnPolicy takes as is.
    env = VisNavEnv(mazes, ticks=args.ticks)
    if args.resume and MODEL.exists():
        model = sb3.PPO.load(MODEL, env=env)
        print(f"resuming from {MODEL}")
    else:
        model = sb3.PPO("CnnPolicy", env, verbose=1, n_steps=512, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=args.steps)
    MODEL.parent.mkdir(exist_ok=True)
    model.save(MODEL)
    print(f"saved {MODEL}")


class RLAgent(Agent):
    """The trained policy as an agent: frames in, one of four movements out, `ticks` at a
    time. It checks in when the policy's own confidence says nothing useful, which is to
    say never; add a stopping rule (a place-recognition match to the target, a step
    budget) before spending attempts on it."""

    def __init__(
        self, model_path: Path = MODEL, ticks: int = 4, size=(80, 60), max_decisions: int = 500
    ):
        sb3 = _sb3()
        if not model_path.exists():
            raise SystemExit(f"no model at {model_path}; train first")
        self.model = sb3.PPO.load(model_path)
        self.ticks = ticks
        self.size = size
        self.max_decisions = max_decisions
        self.decisions = 0
        self.target0: np.ndarray | None = None
        self.last = "-"

    def setup(self, info: SessionInfo) -> None:
        self.target0 = info.targets[0]
        self.decisions = 0

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        assert self.target0 is not None
        if self.decisions >= self.max_decisions:
            return Action.CHECKIN
        self.decisions += 1
        observation = observe(obs.image, self.target0, self.size)
        action, _ = self.model.predict(observation, deterministic=True)
        self.last = ACTION_NAMES[int(action)]
        return Action(ACTIONS[int(action)]), self.ticks

    def hud(self) -> list[str]:
        return [f"policy: {self.last}   decision {self.decisions}"]


def play(args: argparse.Namespace) -> None:
    run(RLAgent(ticks=args.ticks), cli.challenge(args), **cli.run_options(args))


def main() -> None:
    parser = cli.parser(__doc__)
    parser.add_argument("command", choices=("train", "play"))
    parser.add_argument("--mazes", default="7,8,9,10", help="training mazes, comma-separated seeds")
    parser.add_argument("--steps", type=int, default=200_000, help="PPO timesteps")
    parser.add_argument("--ticks", type=int, default=4, help="ticks per decision")
    parser.add_argument("--resume", action="store_true", help="continue from models/ppo.zip")
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    else:
        play(args)


if __name__ == "__main__":
    main()
