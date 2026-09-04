"""
Baseline: you drive, it tells you where you are and which way the goal is.

    RootSIFT -> k-means codebook -> VLAD -> similarity graph -> Dijkstra

Every exploration frame is a node. Consecutive frames are joined by the recorded action;
the most similar-looking distant pairs are joined by "visual shortcut" edges. The current
camera view is matched to its nearest node, the goal is the node most like the target's
front view, and the shortest path between them gives the next action to take.

Same keys as keyboard_agent.py. The strip under the camera shows the best-matching frame,
the goal frame, and the next few nodes along the path with the action that gets you there.
"""

from __future__ import annotations

from pathlib import Path

import cli
import cv2
import networkx as nx
import numpy as np
from keyboard_agent import KeyboardAgent
from vis_nav_sdk import Action, Observation, SessionInfo, run
from vlad import REVERSE, VLADExtractor, load_frames

TEMPORAL_WEIGHT = 1.0
VISUAL_WEIGHT_BASE = 2.0
VISUAL_WEIGHT_SCALE = 3.0
MIN_SHORTCUT_GAP = 50
PREVIEW_NODES = 5

MATCH = (76, 209, 132)
GOAL = (245, 177, 76)
PATH = (108, 180, 255)
JUMP = (178, 140, 255)


class BaselineAgent(KeyboardAgent):
    def __init__(
        self,
        data_dir: Path,
        *,
        n_clusters: int = 128,
        subsample: int = 5,
        top_k_shortcuts: int = 30,
        cache_dir: Path = Path("cache"),
    ) -> None:
        self.frames, bounds = load_frames(data_dir, subsample)
        print(f"{len(self.frames)} frames from {len(bounds)} trajectories")
        self.extractor = VLADExtractor(n_clusters, cache_dir)
        self.database = self.extractor.fit(self.frames)
        self.graph = self._build_graph(bounds, top_k_shortcuts)

        self.goal: int | None = None
        self.current: int | None = None
        self.match_distance = 0.0
        self.path: list[int] = []
        self._panel: list[tuple[np.ndarray, str, tuple[int, int, int]]] = []
        self._localised_step = -1

    # -- Agent

    def setup(self, info: SessionInfo) -> None:
        similarity = self.database @ self.extractor.extract(info.targets[0])
        self.goal = int(np.argmax(similarity))
        self.current = None
        self.path = []
        self._panel = []
        self._localised_step = -1
        print(f"goal: node {self.goal} (d={_distance(similarity[self.goal]):.3f})")

    def act(self, obs: Observation) -> Action | tuple[Action, int]:
        if obs.step != self._localised_step:
            self._localise(obs.image)
            self._localised_step = obs.step
        return super().act(obs)

    def hud(self) -> list[str]:
        lines = super().hud()
        if self.current is None or self.goal is None:
            return lines
        hops = len(self.path) - 1
        hint = self._hint()
        lines += [
            f"node {self.current} (d={self.match_distance:.3f})  goal {self.goal}",
            f"{hops} hops   next: {hint}",
        ]
        if hops <= 5:
            lines.append("near the goal: press space when the view matches")
        return lines

    def panel(self) -> list[tuple[np.ndarray, str, tuple[int, int, int]]]:
        return self._panel

    def _render_panel(self) -> list[tuple[np.ndarray, str, tuple[int, int, int]]]:
        assert self.current is not None and self.goal is not None
        tiles = [
            (self._image(self.current), f"best match · node {self.current}", MATCH),
            (self._image(self.goal), f"goal · node {self.goal}", GOAL),
        ]
        for a, b in list(zip(self.path[:-1], self.path[1:], strict=True))[:PREVIEW_NODES]:
            if self.graph[a][b].get("visual", False):
                tiles.append((self._image(b), f"node {b} · visual jump", JUMP))
            else:
                tiles.append(
                    (self._image(b), f"node {b} · {self._edge_action(a, b).lower()}", PATH)
                )
        return tiles

    # -- the index

    def _build_graph(self, bounds: list[tuple[int, int]], top_k: int) -> nx.Graph:
        n = len(self.database)
        graph = nx.Graph()
        graph.add_nodes_from(range(n))
        for start, end in bounds:
            for i in range(start, end - 1):
                graph.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT)

        similarity = self.database @ self.database.T
        for i in range(n):
            similarity[i, max(0, i - MIN_SHORTCUT_GAP) : i + MIN_SHORTCUT_GAP + 1] = -2
        similarity[np.tril_indices(n)] = -2
        flat = similarity.ravel()
        for index in np.argpartition(flat, -top_k)[-top_k:]:
            i, j = divmod(int(index), n)
            d = _distance(float(flat[index]))
            graph.add_edge(i, j, weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d, visual=True)
        print(f"graph: {n} nodes, {graph.number_of_edges()} edges ({top_k} visual)")
        return graph

    def _localise(self, image: np.ndarray) -> None:
        similarity = self.database @ self.extractor.extract(image)
        self.current = int(np.argmax(similarity))
        self.match_distance = _distance(float(similarity[self.current]))
        try:
            self.path = nx.shortest_path(self.graph, self.current, self.goal, weight="weight")
        except nx.NetworkXNoPath:
            self.path = [self.current]
        self._panel = self._render_panel()

    def _hint(self) -> str:
        if len(self.path) < 2:
            return "at the goal"
        a, b = self.path[0], self.path[1]
        if self.graph[a][b].get("visual", False):
            return "visual jump -- look around"
        return self._edge_action(a, b)

    def _edge_action(self, a: int, b: int) -> str:
        if b == a + 1:
            return self.frames[a].action
        if b == a - 1:
            return REVERSE[self.frames[b].action]
        return "?"

    def _image(self, node: int) -> np.ndarray:
        image = cv2.imread(str(self.frames[node].path))
        return image if image is not None else np.zeros((240, 320, 3), np.uint8)


def _distance(similarity: float) -> float:
    return float(np.sqrt(max(0.0, 2 - 2 * similarity)))


if __name__ == "__main__":
    parser = cli.parser(__doc__)
    parser.add_argument("--data", help="exploration data directory (default: download)")
    parser.add_argument("--subsample", type=int, default=5, help="keep every Nth frame")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size")
    parser.add_argument("--top-k", type=int, default=30, help="visual shortcut edges")
    args = parser.parse_args()

    challenge = cli.challenge(args)
    agent = BaselineAgent(
        cli.exploration_data(args, args.data),
        n_clusters=args.n_clusters,
        subsample=args.subsample,
        top_k_shortcuts=args.top_k,
        cache_dir=Path("cache") / challenge,
    )
    run(agent, challenge, **cli.run_options(args))
