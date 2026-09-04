# vis-nav-sdk

Client for the visual navigation challenge. The simulator runs on the course server; your
code opens a session, sends actions, gets camera frames back, and checks in when it thinks
it has reached the goal. The robot's true pose never leaves the server — inferring it is the
assignment.

The SDK is the `vis_nav_sdk` package in this directory; `uv sync` at the repository root
installs it (editable) with everything it needs. To use it from a project of your own, add
this directory as a dependency: `uv add path/to/vis_nav_player/sdk`, or
`pip install -e path/to/vis_nav_player/sdk[viewer]`.

When the course announces a protocol change, `git pull` here and `uv sync` again.

## Configuration

Two settings, looked up in this order: explicit argument, environment variable, default.

| setting | argument | environment | default |
|---|---|---|---|
| API key | `api_key=` | `VIS_NAV_API_KEY` | — (required) |
| server | `server=` | `VIS_NAV_SERVER` | the course server |

Your key is on the course site. Treat it like a password: put it in the environment, not in
code you commit.

## An agent

```python
from vis_nav_sdk import Agent, Action, run


class MyAgent(Agent):
    def setup(self, info):
        # Once per session. info.targets: four views from the goal (front, left, back,
        # right). info.camera.intrinsic_matrix, info.limits.max_steps.
        self.goal = info.targets[0]

    def act(self, obs):
        # obs.image: (240, 320, 3) uint8 BGR. obs.step, obs.steps_left.
        if self.looks_like(obs.image, self.goal):
            return Action.CHECKIN  # score me here
        return Action.FORWARD, 4  # hold FORWARD for 4 ticks, one round trip


result = run(MyAgent(), "vns_...")          # the token from Start
print(result.goal_tier, result.trans_error, result.nav_steps)
```

`run()` opens a window (if pygame is installed) showing the camera, the four targets, and
what every step costs — the round trip, how much of it was the server, how much the
network, JPEG decode, and how long your own `act()` took. With a window the loop is paced
to 30 steps/s so a human can follow it; headless (`viewer=False`) it runs as fast as the
server answers. `fps=` overrides either.

Lifecycle:

| method | when | notes |
|---|---|---|
| `__init__` | before anything | load data, build indexes. Nothing here touches the server. |
| `setup(info)` | session open | cheap work only — the session clock is running |
| `act(obs)` | every step | return `Action`, `(Action, repeat)`, or `WAIT` |
| `finish(result)` | session over | `result` is `None` if it ended unscored |
| `hud()` / `panel()` | each redraw | optional: text lines, and tiles `(image, label[, rgb])` or one BGR image, for the window |

`run()` first calls `setup()` and `act()` with random frames, *before* redeeming the
token, so a crash in your code costs no attempt and the token still works. `setup()` therefore runs twice per run — do not
cache anything derived from the targets across calls. `check=False` skips this.

`WAIT` means "call me again with the same frame, do not step" — for interactive agents.
Waiting is free; after 90 s of it `run()` burns one `IDLE` tick so the server's 120 s idle
timeout does not end your session.

Interactive agents read the keyboard through the window: `self.viewer.keys()` returns the
key names currently `held` and those `tapped` since the last call (`"up"`, `"space"`, …).
`self.viewer` is `None` when headless.

## What you get, and what you do not

Before a session: the challenge's **exploration data** — camera frames from drives through
the maze with the movement taken after each frame, and the goal views
(`Client().download_exploration_data(id)`). When a session opens: the four **target views**
from the goal pose (front, left, back, right), the **camera intrinsics**, and the **step
budget**. Every step: the **camera frame**, and how many ticks you have spent and have left.

Nothing else. The robot's pose, the map and the goal position stay on the server. An
observation carries no position, no velocity and no collision flag; how far a movement took
you is something you infer from the pixels, and movement may carry noise on some
challenges (`info.limits.motion_noise`).

## Actions

A bit field; combinations are legal and useful.

| Action | Value | Effect |
|---|---|---|
| `IDLE` | 0 | nothing, but burns a tick |
| `FORWARD` | 1 | drive forward one tick |
| `BACKWARD` | 2 | drive backward one tick |
| `LEFT` | 4 | turn left one tick |
| `RIGHT` | 8 | turn right one tick |

`FORWARD | LEFT` drives an arc; opposing bits cancel. `CHECKIN` and `QUIT` are returned
from `act()` (or, on a `Session`, are `checkin()` and `abort()`), not sent as steps.

**`repeat` buys wall clock, never score.** Every tick counts toward `nav_steps`. A typical
successful run is ~1,500 steps; at one round trip per tick that is minutes of pure latency
on campus wifi. Drive in bursts of 4–8, which is how the maze was meant to be driven.

## The session, directly

```python
from vis_nav_sdk import connect, Action

with connect("vns_...") as session:
    session.targets  # four BGR arrays
    obs = session.initial_observation  # free, costs no step
    while obs.steps_left > 0:
        obs = session.step(Action.FORWARD, repeat=4)
        if done(obs.image):
            break
    result = session.checkin()

session.telemetry.summary()  # rtt / server / network / think medians
```

`session.step_all(action, repeat, frames="all")` returns every intermediate frame instead
of the last one (~15 KB each; do not do this for a whole run).

Nothing in `Session` or `Observation` is hidden from you — there is simply nothing more
on the wire than what is documented here.

## The rest of the API

```python
from vis_nav_sdk import Client

client = Client()                                  # key and server from the environment
client.me()                                        # {"email", "name", "course", ...}
client.challenge(CHALLENGE_ID)                     # name, deadline, max_steps, ...
client.quota(CHALLENGE_ID)                         # attempts_used / attempts_allowed
client.sessions(CHALLENGE_ID)                      # your past runs
client.session(SESSION_ID)                         # one run, live: status, steps, result
client.download_exploration_data(CHALLENGE_ID)     # -> Path("data/<id>"), cached
```

The exploration data is `target.jpg` plus `traj_<i>/<k>.jpg` with a `data_info.json` per
trajectory listing the action taken at each frame.

## Scoring, attempts, timing

`checkin()` scores the distance from the goal at the current pose: `goal_tier` is
`perfect`, `partial` or `failed`; ranking is `(goal_tier, nav_steps, trans_error)`.

Each session is one **attempt**; attempts may be limited per challenge. The attempt is
spent when your code connects, not when you press Start. Aborting, quitting, running out of
steps, closing the window and dropping the connection all spend it. Your best attempt is the
one that counts. There is no resume: a token is single-use, press Start for the next run.

The server measures how long your code takes per step, isolated from the network: the SDK
acknowledges each frame from its IO thread the instant it is decoded, before your code
runs, so the server can subtract two arrival times on the same path. Time inside `act()`
(including plotting and logging) is yours; JPEG decode is not. It is reported in
`result.think`; it does not affect your rank.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `ConfigError: no session token` | press Start on the challenge page; pass the `vns_…` token or set `VIS_NAV_SESSION` |
| `ConfigError: … is not a session token` | you passed your API key; the token comes from Start |
| `HTTP 401` on connect | token already used or expired (15 min) — press Start again |
| `HTTP 403` | the challenge stopped accepting runs since you pressed Start |
| `HTTP 409` | your team has a session running; the token stays valid, retry when it ends |
| `HTTP 503` | no simulation worker; the token stays valid, retry shortly, tell a TA if it persists |
| `ConfigError: no API key` (REST only) | set `VIS_NAV_API_KEY` or pass `Client(api_key=)` |
| `no JPEG decoder available` | `uv add opencv-python-headless` (or Pillow) |
| `the gateway speaks protocol vN` | `git pull` in `vis_nav_player`, then `uv sync` |
| `the viewer needs pygame` | `uv add pygame`, or `run(..., viewer=False)` |
