# Visual Navigation Challenge — starter kit

NYU ROB-GY 6203 Robot Perception (AI4CE lab).

A robot sits in a maze on the course server. You are shown four photos taken from the goal.
Your code receives the robot's camera frame, sends a movement, receives the next frame, and
so on, until it decides it has arrived and checks in. The server then measures how far from
the goal it really is, and grades the run with the challenge's rubric.

This repository is what you fork: an agent skeleton to fill in, a keyboard agent to get a
feel for the maze, and a baseline that shows one way to use the exploration data.

## 1. Install

You need two tools: [mise](https://mise.jdx.dev), which installs the right Python, and
[uv](https://docs.astral.sh/uv/), which installs everything else. No conda, no `pip`.

**macOS / Linux**

```bash
curl https://mise.run | sh
echo 'eval "$(~/.local/bin/mise activate zsh)"' >> ~/.zshrc     # bash: use bash and ~/.bashrc
exec $SHELL
```

**Windows** — use [WSL](https://learn.microsoft.com/windows/wsl/install) and follow the Linux
steps inside it.

Then:

```bash
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
mise install        # python 3.12 + uv, pinned in mise.toml
uv sync             # creates .venv with everything in pyproject.toml, including the course SDK
```

That is the whole install. `uv run <script>` runs a script inside `.venv`; you never activate
anything.

## 2. Your credentials

You need two strings. Your **API key** is the one you were sent for the course and use to
sign in to the course site — treat it like a password. The **challenge id** is in the URL of
the challenge page (`/challenges/<id>`). Put them in your shell, not in your code:

```bash
export VIS_NAV_API_KEY="..."
export VIS_NAV_CHALLENGE="..."
```

Add those two lines to `~/.zshrc` (or `~/.bashrc`) so you do not retype them.

## 3. Drive it yourself

```bash
uv run source/keyboard_agent.py
```

It shows the challenge and how many attempts you have, asks before starting one, and opens
a window: the camera on the left, the four goal views and the step count on the right.
Arrows move (hold two for an arc), **space** checks in, **escape** quits. When you check in,
the result appears in the terminal and on the challenge page, which the script opens for
you.

Each run is one **attempt**. Attempts may be limited per challenge and one is spent the
moment your code connects; quitting or closing the window spends it too. Your best attempt
is the one that counts. If your connection drops mid-run, the session waits five minutes
for you: run the same command again and it picks up where it left off.

## 4. Write your agent

`source/my_agent.py` is the skeleton. Every method you can implement is there, with a
comment saying when it is called and what it may return; only `act()` is required:

```python
class MyAgent(Agent):
    def __init__(self, data_dir):        # load exploration data, build your index
    def setup(self, info):               # once per session: info.targets, info.camera, info.limits
    def act(self, obs):                  # every step: obs.image -> Action
    def finish(self, result):            # after the session ended
    def hud(self):                       # optional: text for the viewer window
    def panel(self):                     # optional: images for the viewer window
```

```bash
uv run source/my_agent.py
```

On first run it downloads the challenge's **exploration data** into `data/<challenge>/`:
frames and action labels from earlier drives through this maze, plus `target.jpg`. This is
the only imagery you have of the maze before a session starts. Build whatever you like from
it — a place-recognition index, a topological map, a learned model.

`act()` returns an `Action` (`FORWARD`, `BACKWARD`, `LEFT`, `RIGHT`; combine with `|` for
an arc), or `(Action, n)` to hold it for `n` ticks in one round trip, or `Action.CHECKIN`
when you believe you are at the goal. Every tick counts toward your step total.

Before connecting, `run()` tries your agent on random frames. If `act()` crashes there, no
attempt is spent — so do the heavy lifting in `__init__` and keep `act()` fast: the
server measures how long your code takes per step.

### What your agent may use

Everything is on this list. There is nothing else; the robot's pose, the map and the goal
position live on the server and are never sent.

| when | what |
|---|---|
| before a session | the exploration data: `traj_i/k.jpg` frames, `traj_i/data_info.json` (the action taken after each frame), `target.jpg` |
| `setup(info)` | `info.targets` — four goal views (front, left, back, right); `info.camera` — size and intrinsics; `info.limits` — step budget, attempts |
| `act(obs)` | `obs.image` — `(240, 320, 3)` `uint8`, BGR; `obs.step`, `obs.steps_left` |

Movement is applied as you request it, with a little noise on most challenges, and the next
frame is the only feedback. Working out where you are from those pixels is the assignment.

## 5. The baseline

```bash
uv run source/baseline_agent.py
```

You still drive; the baseline says where it thinks you are and which way to go, from a
RootSIFT + VLAD index over the exploration frames and a graph of who-follows-whom
(`source/vlad.py`, `source/baseline_agent.py`). Building the index takes about a minute the
first time and is cached in `cache/<challenge>/`. It is a starting point, not a solution.

## 6. Keeping up to date

The course SDK (`vis-nav-sdk`) is installed from the course site. When a new version is
out, the scripts tell you. Update with:

```bash
uv lock --upgrade-package vis-nav-sdk && uv sync
```

If the server refuses your SDK version outright, that is the same fix.

## Command-line options

Every script accepts `--challenge` and `--api-key` (instead of the environment variables),
`--server`, `--yes` (start without asking), `--no-browser`, `--no-check` (skip the
pre-flight), and `--data <dir>` for the exploration data where relevant.

## Prefer to drive the loop yourself?

```python
from vis_nav_sdk import Action, connect

with connect(CHALLENGE_ID) as session:
    obs = session.initial_observation
    obs = session.step(Action.FORWARD, repeat=4)
    print(session.checkin())
```

SDK reference: https://visual-navigation-challenge.ai4ce.dev/sdk/README.md
