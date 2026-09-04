# Visual Navigation Game — example agents

Course project platform for NYU ROB-GY 6203 Robot Perception (AI4CE lab, cfeng at nyu dot
edu).

A robot sits in a maze on the course server. You are shown four photos taken from the goal.
Your code receives the robot's camera frame, sends a movement, receives the next frame, and
so on, until it decides it has arrived and checks in. The server then measures how far from
the goal it really is.

## Setup

We use [mise](https://mise.jdx.dev) to install tools and [uv](https://docs.astral.sh/uv/) to
manage Python. No conda.

```bash
curl https://mise.run | sh            # once; then restart your shell
git clone https://github.com/ai4ce/vis_nav_player.git
cd vis_nav_player
mise install                          # python + uv, pinned in mise.toml
uv sync                               # creates .venv with everything in pyproject.toml
```

`uv sync` installs the course SDK (`vis-nav-sdk`) from the course site's package index,
`https://visual-navigation-challenge.ai4ce.dev/sdk/simple/`, alongside the usual scientific
stack. When a new SDK version is announced: `uv lock --upgrade-package vis-nav-sdk && uv sync`.

Get your **API key** and the **challenge id** from the course site and put them in your
shell (never in code you commit):

```bash
export VIS_NAV_API_KEY="..."
export VIS_NAV_CHALLENGE="..."
```

Every script also takes `--api-key`, `--challenge`, `--server`, `--yes` (do not ask) and
`--no-browser`.

## Starting a run

Every script begins the same way. It checks the challenge — whether your team already has a
run going, whether you left a session unconnected earlier — and asks before starting attempt
_n_ of _m_. Then it opens the challenge page, where you can watch the robot's camera and your
step count live and see the result when you check in, and tells you where to submit your
report if the challenge asks for one. Nothing is spent until your code connects.

## Drive it yourself

```bash
uv run source/keyboard_agent.py
```

Arrows move (hold two for an arc), **space** checks in, **escape** quits. The window shows
the camera, the four views from the goal, and what each step costs: the round trip, the
server's share, the network's share, and your own code's time.

Each run is one **attempt**, and attempts may be limited per challenge; it is spent when
your code connects. Quitting, closing the window or losing the connection also spends it.
Your best attempt counts.

## Baseline

```bash
uv run source/baseline_agent.py
```

You still drive; the baseline says where it thinks you are and which way to go. On first run
it downloads the exploration data into `data/<challenge>/` and builds its index into
`cache/<challenge>/` (a minute or so; cached afterwards).

How it works (`source/vlad.py`, `source/baseline_agent.py`):

1. **RootSIFT** descriptors for every exploration frame
2. **k-means** codebook (k = 128)
3. **VLAD** vector per frame, with intra- and power normalisation
4. **Graph**: consecutive frames joined by the recorded action; the most similar-looking
   distant pairs joined by visual shortcut edges
5. **Localise and plan**: match the live frame to its nearest node, the goal to the node
   most like the target's front view, Dijkstra between them

The strip under the camera shows the best match, the goal frame, and the next nodes along
the path with the action that gets you there.

## What you have to work with

Everything your agent may use is on this list. There is nothing else; the robot's pose,
the map and the goal position live on the server and are never sent.

**Before a session — the exploration data** (`Client().download_exploration_data(id)`,
or let the baseline fetch it). A zip of one or more drives through the maze:

```
target.jpg                     the four goal views side by side
traj_0/0.jpg, 1.jpg, ...       camera frames from one drive
traj_0/data_info.json          [{"step": k, "image": "k.jpg", "action": ["FORWARD"]}, ...]
traj_1/...
```

Frames are consecutive, and `action` is the movement the robot made *after* that frame. Use
them to learn what the maze looks like, build a place-recognition index, estimate how far a
movement takes you, or anything else you can get out of images and action labels.

**When a session opens** — `setup(info)`:

- `info.targets`: the four goal views, from the goal pose facing front, left, back, right
- `info.camera`: image size and intrinsic matrix
- `info.limits`: the step budget and your attempt count

**Every step** — `act(obs)`:

- `obs.image`: the camera frame, `(240, 320, 3)` `uint8`, BGR
- `obs.step`, `obs.steps_left`: ticks spent and remaining

And that is all. Movement is applied as you request it — with a bit of noise on some
challenges — and the frame you get back is the only feedback. Working out where you are and
how far you have moved from those pixels is the assignment.

## Write your own

```python
from vis_nav_sdk import Agent, Action, run


class MyAgent(Agent):
    def __init__(self): ...  # load exploration data, build your index

    def setup(self, info):
        self.goal = info.targets[0]  # once per session

    def act(self, obs):
        if self.at_goal(obs.image):
            return Action.CHECKIN  # scores the run and ends it
        return Action.FORWARD, 4  # hold for 4 ticks in one round trip


run(MyAgent(), CHALLENGE_ID)  # key from $VIS_NAV_API_KEY
```

Actions are a bit field — `FORWARD | LEFT` is an arc. Every tick counts toward
`nav_steps`, which ranks you once you have reached the goal; `(action, n)` applies it for
`n` ticks in one round trip, which changes how long you wait on the network and nothing
else. Do your heavy lifting in `__init__`: nothing there touches the server, and `run()`
tries your agent on random frames before connecting, so a crash costs no attempt.

Prefer to drive the loop yourself?

```python
from vis_nav_sdk import connect, Action

with connect(CHALLENGE_ID) as session:
    obs = session.initial_observation
    obs = session.step(Action.FORWARD, repeat=4)
    print(session.checkin())
```

SDK reference: https://visual-navigation-challenge.ai4ce.dev/sdk/README.md
