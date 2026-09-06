# Visual Navigation Challenge — starter kit

NYU **ROB-UY 3203 Robot Vision** and **ROB-GY 6203 Robot Perception**, run by the
[AI4CE Lab](https://ai4ce.github.io/).

A robot sits in a maze on the course server. Your code receives its camera frames and four
photos of the goal, chooses each move, and checks in when it thinks it has arrived. This
repository is what you fork: a keyboard agent to drive the maze yourself, a place-recognition
baseline, and the skeleton your own agent goes in.

**The tutorial is on the course site:** <https://visual-navigation-challenge.ai4ce.dev/>
Environment setup, a first drive, the agent interface, how the baseline works, and how runs
are scored, step by step. Everything below is the short version.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh      # Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
uv sync
```

## Credentials

Copy `.env.example` to `.env` and fill in your API key (from the course site, under the
account menu) and the challenge id (the last part of a challenge page's URL). Every script
reads the file; git ignores it.

```
VIS_NAV_API_KEY=...
VIS_NAV_CHALLENGE=...
```

## Run

```bash
uv run source/keyboard_agent.py   # drive by hand: arrows move, space checks in, esc quits
uv run source/baseline_agent.py   # drive with the baseline's hints
uv run source/my_agent.py         # your agent (source/my_agent.py)
```

Every script accepts `--challenge` and `--api-key` in place of `.env`, plus `--yes`,
`--no-browser`, `--no-check` and, where relevant, `--data <dir>`. When a new SDK version is
out the scripts say so and offer to update.

## On your own machine

The server's simulator is also a Python package. With it, any script runs on a maze on your
machine instead of a challenge: no attempt, no key, no network, the same code.

```bash
uv sync --group local                        # once; the first run also downloads the textures (123 MB)
uv run source/keyboard_agent.py --local 7    # maze 7: the same maze on every machine
uv run source/my_agent.py --local 7
```

A seed is a maze; share one like a challenge id. The exploration data for a local maze is
recorded on first use into `data/local-<seed>/`, so the baseline works there too.

For reinforcement learning, `source/rl/` is a Gymnasium environment on the simulator, with
the robot's true position for the reward (which only exists locally), skrl's PPO to train on
a list of mazes, and the trained policy as an agent that sees only frames:

```bash
uv sync --group rl          # gymnasium, skrl, and PyPI's torch: CPU on macOS and Windows, CUDA on Linux
uv sync --group rl-cuda     # ...or the CUDA build, for an NVIDIA card on Windows or Linux
uv sync --group rl-cpu      # ...or the CPU build everywhere: the small download
uv run source/rl/env.py --mazes 7,8,9                         # a random policy, to see the environment
uv run source/rl/train.py --mazes 7,8,9,10 --timesteps 200000 # -> models/policy.pt
uv run source/rl/play.py --local 11                           # a maze it never saw
uv run source/rl/play.py --challenge <id>                     # one real attempt
```

## What is here

| file | what |
|---|---|
| `source/my_agent.py` | the skeleton: `__init__`, `setup`, `act`, `finish`, `hud`, `panel`, each with a comment saying when it runs |
| `source/keyboard_agent.py` | drive with the arrow keys |
| `source/baseline_agent.py`, `source/vlad.py` | RootSIFT + VLAD place recognition over the exploration frames, a graph of them, and the next move along the shortest path |
| `source/cli.py` | the shared command line, `.env` loading, `--local` |
| `source/rl/` | `env.py` the Gymnasium environment, `models.py` the networks, `train.py` PPO with skrl, `play.py` the policy as an agent |

The SDK's own documentation (`connect()`, `Session`, the REST client) is at
<https://visual-navigation-challenge.ai4ce.dev/sdk>.
