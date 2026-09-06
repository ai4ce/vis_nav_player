"""
Experimental: drive a trained policy on another local maze, or on the server for a real attempt.

    uv run source/rl/play.py --local 11          # a maze it never saw, free
    uv run source/rl/play.py --challenge <id>    # one attempt on the server

`RLAgent` is an ordinary `vis_nav_sdk.Agent`: it is given frames and returns movements,
`ticks` at a time, from the policy `rl/train.py` saved. It has no idea where it is, which
is exactly the situation on the server. It checks in after `--decisions` decisions; a real
stopping rule (a place-recognition match to the target, say) is the first thing to add.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # source/, for cli

import cli
import numpy as np
import torch
from env import ACTION_NAMES, ACTIONS, VisNavEnv, observe
from models import Policy
from vis_nav_sdk import Action, Agent, Observation, SessionInfo, run

POLICY = Path("models") / "policy.pt"


class RLAgent(Agent):
    def __init__(self, policy_path: Path = POLICY, *, ticks: int = 4, decisions: int = 500) -> None:
        if not policy_path.exists():
            raise SystemExit(f"no policy at {policy_path}; train one with source/rl/train.py")
        spaces = VisNavEnv.spaces()
        self.policy = Policy(*spaces, device="cpu")
        self.policy.load_state_dict(torch.load(policy_path, map_location="cpu"))
        self.policy.eval()
        self.size = VisNavEnv.SIZE
        self.ticks = ticks
        self.decisions = decisions
        self.taken = 0
        self.target0: np.ndarray | None = None
        self.last = "-"

    def setup(self, info: SessionInfo) -> None:
        self.target0 = info.targets[0]
        self.taken = 0

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        assert self.target0 is not None
        if self.taken >= self.decisions:
            return Action.CHECKIN
        self.taken += 1
        image = torch.as_tensor(observe(obs.image, self.target0, self.size)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.policy.compute({"observations": image.flatten(1)})
        choice = int(logits.argmax(dim=1).item())
        self.last = ACTION_NAMES[choice]
        return Action(ACTIONS[choice]), self.ticks

    def hud(self) -> list[str]:
        return [f"policy: {self.last}   decision {self.taken}/{self.decisions}"]


def main() -> None:
    parser = cli.parser(__doc__)
    parser.add_argument("--policy", type=Path, default=POLICY)
    parser.add_argument("--ticks", type=int, default=4, help="simulator ticks per decision")
    parser.add_argument("--decisions", type=int, default=500, help="decisions before checking in")
    args = parser.parse_args()
    agent = RLAgent(args.policy, ticks=args.ticks, decisions=args.decisions)
    run(agent, cli.challenge(args), **cli.run_options(args))


if __name__ == "__main__":
    main()
