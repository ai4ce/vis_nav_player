"""
Experimental: train a policy on the local simulator with skrl's PPO.

    uv run source/rl/train.py --mazes 7,8,9,10 --timesteps 200000    # -> models/policy.pt

The environment is `rl/env.py`, over the listed mazes in turn (a new maze is a millisecond);
the networks are `rl/models.py`. Logs and skrl checkpoints go under runs/; the policy's
weights alone go to models/policy.pt, which `rl/play.py` wraps in an Agent that sees only
frames and drives it on another local maze or on the server.

This is a demo, not a solution: two hundred thousand steps of PPO on raw pixels learns to
move and turn, not to navigate a maze it has never seen. The course's reference agent is
source/baseline_agent.py. The observation, the reward and the networks are the parts to
change if you want to take this further.

Requires `uv sync --group rl` (gymnasium, skrl and with it torch).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/, for cli and vlad

import torch
from env import VisNavEnv
from models import Policy, Value
from skrl.agents.torch.base import ExperimentCfg
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer, SequentialTrainerCfg

POLICY = Path("models") / "policy.pt"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mazes", default="7,8,9,10", help="training mazes, comma-separated seeds")
    parser.add_argument("--timesteps", type=int, default=200_000, help="environment steps")
    parser.add_argument("--ticks", type=int, default=4, help="simulator ticks per decision")
    parser.add_argument("--rollouts", type=int, default=1024, help="steps between PPO updates")
    parser.add_argument("--resume", action="store_true", help="start from models/policy.pt")
    parser.add_argument("--runs", default="runs", help="where skrl writes logs and checkpoints")
    args = parser.parse_args()

    mazes = [int(s) for s in args.mazes.split(",")]
    env = wrap_env(VisNavEnv(mazes, ticks=args.ticks), verbose=False)
    device = env.device
    policy = Policy(env.observation_space, env.action_space, device)
    value = Value(env.observation_space, env.action_space, device)
    if args.resume and POLICY.exists():
        policy.load_state_dict(torch.load(POLICY, map_location=device))
        print(f"resuming from {POLICY}")

    cfg = PPO_CFG(
        rollouts=args.rollouts,
        learning_epochs=4,
        mini_batches=8,
        learning_rate=3e-4,
        entropy_loss_scale=0.01,
        grad_norm_clip=0.5,
        experiment=ExperimentCfg(
            directory=args.runs,
            experiment_name="ppo",
            write_interval="auto",
            checkpoint_interval="auto",
        ),
    )
    agent = PPO(
        models={"policy": policy, "value": value},
        memory=RandomMemory(memory_size=args.rollouts, num_envs=env.num_envs, device=device),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
        cfg=cfg,
    )
    trainer = SequentialTrainer(
        env=env, agents=agent, cfg=SequentialTrainerCfg(timesteps=args.timesteps, headless=True)
    )
    trainer.train()

    POLICY.parent.mkdir(exist_ok=True)
    torch.save(policy.state_dict(), POLICY)
    print(f"saved the policy to {POLICY}; skrl's own checkpoints are under {args.runs}/")


if __name__ == "__main__":
    main()
