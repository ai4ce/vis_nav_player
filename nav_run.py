"""
run.py
----------------------
This is the main challenge-time navigation file.

The goal of this file is to take the cleaned map from the offline SLAM stage,
the exploration images, and the live FPV camera stream, and turn that into a
fully autonomous navigation policy.

Main idea of the pipeline:
- Use SIFT / VLAD to match the current camera view against the
  exploration images.
- Use the cleaned metric map and stored frame poses from SLAM to localize the
  robot in world coordinates.
- Build a safe A* route to the target on that metric map.
- Follow the route with a simple controller that turns toward the path, checks
  whether forward motion is safe, avoids obstacles, and replans if the pose 
  drifts too far.

In short, this file is the online navigation brain of the project.
"""

from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import numpy as np
import os
import json
import heapq
import pickle
import networkx as nx
from sklearn.cluster import KMeans
from tqdm import tqdm
import time

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_DIR = "cache"
IMAGE_DIR = "data/exploration_images/traj_0"
DATA_INFO_PATH = "data/data_info.json"
PATH_CACHE_FILE = os.path.join(CACHE_DIR, "astar_path_cache.pkl")

# Graph construction settings.
# Temporal edges follow the exploration order, while visual edges add a few
# strong long-range shortcuts between visually similar frames.
TEMPORAL_WEIGHT = 1.0       # edge weight for consecutive frames
VISUAL_WEIGHT_BASE = 2.0    # base weight for visual shortcut edges
VISUAL_WEIGHT_SCALE = 3.0   # weight += scale * vlad_distance
MIN_SHORTCUT_GAP = 50       # minimum trajectory index gap for shortcuts

os.makedirs(CACHE_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Geometry / Motion Helpers
# ---------------------------------------------------------------------------
class Odometry:
    """Simple planar odometry state used during navigation."""

    def __init__(self, initial_x=0.0, initial_y=0.0, initial_theta=0.0):
        self.x = initial_x
        self.y = initial_y
        self.theta = initial_theta

    def update(self, v, w, dt):
        """Integrate one small motion step into the current pose estimate."""
        if dt <= 0: return
        self.x += v * dt * np.cos(self.theta)
        self.y += v * dt * np.sin(self.theta)
        self.theta += w * dt
        self.theta = (self.theta + np.pi) % (2 * np.pi) - np.pi

class WallDetector:
    """
    Extract rough local wall points from the FPV image.

    We use this as a lightweight geometric cue so the robot is not relying only
    on image retrieval. The detector finds the sky / wall boundary and projects
    those pixels into approximate robot-centric wall coordinates.
    """

    def __init__(self, K, true_camera_height=0.21, wall_height=0.30, max_depth=0.3):
        self.K = K
        self.cam_h = true_camera_height 
        self.wall_h = wall_height 
        self.max_depth = max_depth
        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]

    def extract_wall_points(self, fpv):
        """
        Return local wall points in robot coordinates.

        Detect the first non-sky pixel in sampled image columns, then 
        back-project it into a rough 2D wall point.
        """
        h, w = fpv.shape[:2]
        gray = cv2.cvtColor(fpv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        mask = np.zeros((h + 2, w + 2), np.uint8)
        seed_point = (w // 2, 5)
        cv2.floodFill(blurred, mask, seed_point, 255, 2, 2, cv2.FLOODFILL_MASK_ONLY)
        sky_mask = mask[1:-1, 1:-1]
        
        all_local_points = []
        step = max(1, w // 60)
        
        for u in range(step // 2, w, step):
            for v in range(5, int(self.cy) - 5):
                if sky_mask[v, u] == 0:
                    delta_h = self.wall_h - self.cam_h 
                    denominator = max((self.cy - v), 1)
                    Z_cam = (self.fy * delta_h) / denominator
                    
                    if Z_cam > self.max_depth or Z_cam <= 0.1: break
                    X_cam = ((u - self.cx) * Z_cam) / self.fx
                    X_robot, Y_robot = Z_cam, -X_cam
                    all_local_points.append((X_robot, Y_robot))
                    break 
                    
        return all_local_points


# ---------------------------------------------------------------------------
# VLAD Feature Extraction
# ---------------------------------------------------------------------------
class VLADExtractor:
    """
    RootSIFT + VLAD descriptor pipeline.

    This is the main visual matching block. It converts each exploration image
    and each live frame into a compact descriptor so we can do fast place matching.
    """

    def __init__(self, n_clusters: int = 128):
        self.n_clusters = n_clusters
        self.sift = cv2.SIFT_create()
        self.codebook = None
        self._sift_cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        """Return the final VLAD descriptor size."""
        return self.n_clusters * 128

    @staticmethod
    def _root_sift(des: np.ndarray) -> np.ndarray:
        """Apply RootSIFT normalization on top of raw SIFT descriptors."""
        des = des / np.sum(des, axis=1, keepdims=True)
        return np.sqrt(des)

    def _des_to_vlad(self, des: np.ndarray) -> np.ndarray:
        """
        Aggregate local descriptors into one VLAD vector.

        We use VLAD because it keeps matching compact while still preserving more
        structure than using only one local descriptor or a raw nearest-neighbor setup.
        """
        labels = self.codebook.predict(des)
        centers = self.codebook.cluster_centers_
        k = self.codebook.n_clusters
        vlad = np.zeros((k, des.shape[1]))
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                vlad[i] = np.sum(des[mask] - centers[i], axis=0)
                norm = np.linalg.norm(vlad[i])
                if norm > 0:
                    vlad[i] /= norm
        vlad = vlad.ravel()
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))
        norm = np.linalg.norm(vlad)
        if norm > 0:
            vlad /= norm
        return vlad

    def load_sift_cache(self, file_list: list[str], subsample_rate: int):
        """
        Load cached SIFT descriptors if available; otherwise extract and store them.

        This keeps startup time practical, since recomputing SIFT over the whole
        exploration set every run would be wasteful.
        """
        cache_file = os.path.join(CACHE_DIR, f"sift_ss{subsample_rate}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached SIFT from {cache_file}")
            with open(cache_file, "rb") as f:
                self._sift_cache = pickle.load(f)
            if all(fname in self._sift_cache for fname in file_list):
                return
            print("  Cache incomplete, re-extracting...")

        self.frame_count = 0
        print(f"Extracting SIFT for {len(file_list)} images...")
        self._sift_cache = {}
        for fname in tqdm(file_list, desc="SIFT"):
            img = cv2.imread(os.path.join(IMAGE_DIR, fname))
            _, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                self._sift_cache[fname] = self._root_sift(des)
        with open(cache_file, "wb") as f:
            pickle.dump(self._sift_cache, f)
        print(f"  Saved {len(self._sift_cache)} descriptors -> {cache_file}")

    def build_vocabulary(self, file_list: list[str]):
        """
        Build or load the VLAD codebook.

        The codebook is the visual vocabulary that lets us turn many local SIFT
        features into one compact global descriptor per frame.
        """
        cache_file = os.path.join(CACHE_DIR, f"codebook_k{self.n_clusters}.pkl")
        if os.path.exists(cache_file):
            print(f"Loading cached codebook from {cache_file}")
            with open(cache_file, "rb") as f:
                self.codebook = pickle.load(f)
            return

        all_des = np.vstack([self._sift_cache[f] for f in file_list if f in self._sift_cache])
        print(f"Fitting KMeans (k={self.n_clusters}) on {len(all_des)} descriptors...")
        self.codebook = KMeans(
            n_clusters=self.n_clusters, init='k-means++',
            n_init=3, max_iter=300, tol=1e-4, verbose=1, random_state=42,
        ).fit(all_des)
        print(f"  {self.codebook.n_iter_} iters, inertia={self.codebook.inertia_:.0f}")
        with open(cache_file, "wb") as f:
            pickle.dump(self.codebook, f)

    def extract(self, img: np.ndarray) -> np.ndarray:
        """Extract one VLAD descriptor from a single image."""
        _, des = self.sift.detectAndCompute(img, None)
        if des is None or len(des) == 0:
            return np.zeros(self.dim)
        return self._des_to_vlad(self._root_sift(des))

    def extract_batch(self, file_list: list[str]) -> np.ndarray:
        """Extract VLAD descriptors for the full exploration database."""
        vectors = []
        for fname in tqdm(file_list, desc="VLAD"):
            if fname in self._sift_cache and len(self._sift_cache[fname]) > 0:
                vectors.append(self._des_to_vlad(self._sift_cache[fname]))
            else:
                vectors.append(np.zeros(self.dim))
        return np.array(vectors)

# ---------------------------------------------------------------------------
# Main Navigation Agent
# ---------------------------------------------------------------------------
class KeyboardPlayerPyGame(Player):
    """
    Main challenge-time navigation agent.

    This class ties the whole online pipeline together:
    visual matching, pose estimation, path planning, control, relocalization,
    and the debugging / map views.
    """

    def __init__(self, n_clusters: int = 128, subsample_rate: int = 5, top_k_shortcuts: int = 30, viz: bool = True):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.viz = viz

        # The cleaned map produced by the offline stage is the geometric base for
        # both path planning and wall-validity checks.
        self.occupancy_map = cv2.imread("cache/slam_map_walls_cleaned.png", cv2.IMREAD_GRAYSCALE)
        self.map_scale = 60.0 
        self.map_offset = 400 

        # Precompute the free-space mask and distance-to-wall map once.
        # This avoids rebuilding them every time A* is called.
        if self.occupancy_map is not None:
            self.free_mask = np.zeros_like(self.occupancy_map, dtype=np.uint8)
            self.free_mask[self.occupancy_map > 150] = 255

            kernel = np.ones((5, 5), np.uint8)
            self.free_mask = cv2.erode(self.free_mask, kernel, iterations=1)

            self.clearance_map = cv2.distanceTransform(self.free_mask, cv2.DIST_L2, 5)
        else:
            self.free_mask = None
            self.clearance_map = None

        # Core path-following state.
        self.global_path = []              # list of (world_x, world_y)
        self.goal_world_coords = None      # (gx, gy)
        self.is_autonomous = False
        self.lookahead_dist = 0.45
        self.goal_reach_dist = 0.0125
        self.path_replan_dist = 0.40

        super().__init__()

        self.subsample_rate = subsample_rate
        self.top_k_shortcuts = top_k_shortcuts

        # Camera model and wall detector used during live correction.
        CAMERA_W, CAMERA_H = 320, 240
        CAMERA_F = 92
        self.K = np.array([[CAMERA_F, 0, CAMERA_W / 2.0],
                           [0, CAMERA_F, CAMERA_H / 2.0],
                           [0, 0, 1]])
        
        self.wall_detector = WallDetector(self.K)
        self.historic_v_walls = []
        self.historic_h_walls = []
        self.MATCH_THRESHOLD = 0.15

        # Load the exploration trajectory and keep only pure single-action frames.
        # We subsample them because we want a compact but still useful visual database.
        self.motion_frames = []
        self.file_list = []
        if os.path.exists(DATA_INFO_PATH):
            with open(DATA_INFO_PATH) as f:
                raw = json.load(f)
            pure = {'FORWARD', 'LEFT', 'RIGHT', 'BACKWARD'}
            all_motion = [
                {'step': d['step'], 'image': d['image'], 'action': d['action'][0]}
                for d in raw if len(d['action']) == 1 and d['action'][0] in pure
            ]
            self.motion_frames = all_motion[::subsample_rate]
            self.file_list = [m['image'] for m in self.motion_frames]
            print(f"Frames: {len(all_motion)} total, {len(self.motion_frames)} after {subsample_rate}x subsample")

        self.extractor = VLADExtractor(n_clusters=n_clusters)
        self.database = None
        self.G = None
        self.goal_node = None

        # Runtime map / pose state.
        self.slam_map = None
        self.frame_poses = {}
        self.odom = None          
        self.last_time = None     
        self.frame_count = 0

        if os.path.exists("cache/slam_map_walls_cleaned.png") and os.path.exists("cache/frame_poses.json"):
            self.slam_map = cv2.imread("cache/slam_map_walls_cleaned.png")
            with open("cache/frame_poses.json") as f:
                self.frame_poses = json.load(f)
            print("Loaded SLAM map and world coordinates.")
        else:
            print("Warning: SLAM map/poses not found in cache. Run SLAM script first.")

    def reset(self):
        """Reset runtime navigation state before a new attempt starts."""
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.odom = None
        self.last_time = None
        self.frame_count = 0
        self.historic_v_walls = []
        self.historic_h_walls = []
        self.global_path = []

        # We default to autonomous mode because we want the runs to be
        # fully automatic
        self.is_autonomous = True

        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
        }

    def world_to_pixel(self, world_x, world_y):
        """Convert metric world coordinates into map pixel coordinates."""
        px = int(self.map_offset + (world_x * self.map_scale))
        py = int(self.map_offset - (world_y * self.map_scale))
        return px, py

    def pixel_to_world(self, px, py):
        """Convert map pixel coordinates back into world coordinates."""
        x = (px - self.map_offset) / self.map_scale
        y = (self.map_offset - py) / self.map_scale
        return x, y

    def plan_astar_path(self):
        """
        Build the metric A* route from the current odometry pose to the goal.

        The path is computed on the cleaned maze map, not on the raw retrieval graph.
        That is important because we want a physically safe route through free space,
        not just a chain of visually similar frames.
        """
        if self.odom is None or self.goal_world_coords is None or self.occupancy_map is None:
            self.global_path = []
            return

        # Convert the current robot pose and goal pose from world coordinates into
        # map pixels, because the search is done directly on the map image grid.
        start_px, start_py = self.world_to_pixel(self.odom.x, self.odom.y)
        goal_px, goal_py = self.world_to_pixel(self.goal_world_coords[0], self.goal_world_coords[1])

        TURN_COST = 30  # penalize repeated turns

        # ------------------------------------------------------
        # Load cached A* result if this exact start/goal pair
        # has already been solved before.
        # ------------------------------------------------------
        cache_key = ("turn_penalty_v2", TURN_COST, start_px, start_py, goal_px, goal_py)
        if os.path.exists(PATH_CACHE_FILE):
            try:
                with open(PATH_CACHE_FILE, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("key") == cache_key:
                    self.global_path = cached["path"]
                    print("[A*] Loaded cached path.")
                    return
            except Exception as e:
                print(f"[A*] Cache load failed: {e}")

        # Use cached maps instead of rebuilding them on every call.
        if self.free_mask is None or self.clearance_map is None:
            self.global_path = []
            return

        # Crop the search to a local ROI around start and goal.
        # This makes the initial A* much faster.
        margin = 100
        x0 = max(0, min(start_px, goal_px) - margin)
        x1 = min(self.free_mask.shape[1], max(start_px, goal_px) + margin + 1)
        y0 = max(0, min(start_py, goal_py) - margin)
        y1 = min(self.free_mask.shape[0], max(start_py, goal_py) + margin + 1)

        free_mask = self.free_mask[y0:y1, x0:x1]
        clearance = self.clearance_map[y0:y1, x0:x1]

        start_px -= x0
        start_py -= y0
        goal_px -= x0
        goal_py -= y0

        h, w = free_mask.shape

        def heuristic(a, b):
            """
            Manhattan-distance heuristic for A*.

            We use this because our search only moves in the 4 cardinal directions.
            So Manhattan distance is a natural and cheap estimate of how far a cell is
            from the goal.
            """
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # We keep the search 4-connected instead of diagonal.
        # That matches the maze geometry better and avoids unrealistic corner cutting.
        directions = [(0,1),(0,-1),(1,0),(-1,0)]

        # Direction-aware A*:
        # state = (x, y, prev_dir_idx)
        # This lets us add an extra cost whenever direction changes.
        open_set = []
        start_state = (start_px, start_py, None)
        heapq.heappush(open_set, (0.0, start_state))

        came_from = {}
        g_score = {start_state: 0.0}

        goal_state = None
        path_found = False

        while open_set:
            _, (curr_x, curr_y, prev_dir) = heapq.heappop(open_set)
            curr_state = (curr_x, curr_y, prev_dir)

            # We allow a tiny pixel tolerance near the goal.
            if heuristic((curr_x, curr_y), (goal_px, goal_py)) < 3:
                goal_state = curr_state
                path_found = True
                break

            for dir_idx, (dx, dy) in enumerate(directions):
                nx_, ny_ = curr_x + dx, curr_y + dy

                # Only expand neighbors that stay inside the image and lie in free space.
                if 0 <= nx_ < w and 0 <= ny_ < h and free_mask[ny_, nx_] > 0:
                    # Base step cost is 1 per grid move.
                    # Then we add a wall penalty so cells near walls become more expensive.
                    wall_penalty = 12.0 / max(clearance[ny_, nx_], 1.0)
                    step_cost = 1.0 + wall_penalty

                    # Extra cost whenever we change direction.
                    turn_penalty = 0.0
                    if prev_dir is not None and dir_idx != prev_dir:
                        # we add a turn penalty so any turns become more expensive.
                        turn_penalty = TURN_COST

                    next_state = (nx_, ny_, dir_idx)
                    tentative_g = g_score[curr_state] + step_cost + turn_penalty

                    # Standard A* relaxation.
                    if next_state not in g_score or tentative_g < g_score[next_state]:
                        g_score[next_state] = tentative_g
                        priority = tentative_g + heuristic((nx_, ny_), (goal_px, goal_py))
                        heapq.heappush(open_set, (priority, next_state))
                        came_from[next_state] = curr_state

        if not path_found:
            self.global_path = []
            return

        # Reconstruct the path by backtracking from the final goal state.
        path_pixels = []
        curr_state = goal_state
        while curr_state in came_from:
            x, y, _ = curr_state
            path_pixels.append((x, y))
            curr_state = came_from[curr_state]
            
        path_pixels.reverse()

        # Shift ROI-local pixels back to full-map pixels.
        path_pixels = [(px + x0, py + y0) for (px, py) in path_pixels]

        # Mild sparsification keeps the overall route the same, but removes a lot of
        # tiny pixel-by-pixel jitter. That makes the downstream controller smoother
        # and usually faster without changing the actual maze route.
        if len(path_pixels) > 1:
            sparse = path_pixels[::4]
            if sparse[-1] != path_pixels[-1]:
                sparse.append(path_pixels[-1])
            path_pixels = sparse

        # Store the final path back in world coordinates because the controller and
        # odometry run in metric space, not image-pixel space.
        self.global_path = [self.pixel_to_world(px, py) for (px, py) in path_pixels]

        # ---------------------------------------------------------------
        # Save solved path for future runs.
        # ---------------------------------------------------------------
        try:
            with open(PATH_CACHE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "key": cache_key,
                        "path": self.global_path,
                    },
                    f,
                )
            print("[A*] Saved path to cache.")
        except Exception as e:
            print(f"[A*] Cache save failed: {e}")

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi] for stable heading comparisons."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _distance_to_path(self):
        """Return the current robot-to-path distance in world coordinates."""
        if self.odom is None or not self.global_path:
            return np.inf
        robot = np.array([self.odom.x, self.odom.y], dtype=np.float32)
        return min(np.linalg.norm(np.array(p, dtype=np.float32) - robot) for p in self.global_path)

    def _is_forward_safe(self, samples=6, step_dt=0.01):
        """
        Roll out a short forward motion on the map and reject it if it enters a wall.

        This is the small safety check that lets the controller stay simple while still
        respecting the occupancy map.
        """
        if self.odom is None:
            return False

        test_x = self.odom.x
        test_y = self.odom.y
        step = 2.9462 * step_dt

        for _ in range(samples):
            test_x += step * np.cos(self.odom.theta)
            test_y += step * np.sin(self.odom.theta)
            if not self.is_state_valid(test_x, test_y):
                return False
        return True

    def get_autonomous_action(self):
        """
        Core path-following controller.

        The controller is intentionally simple:
        it tracks the current A* path, turns toward the next useful point, checks if
        a forward rollout is safe, and replans if the pose drifts too far away.
        """
        if self.odom is None or self.goal_world_coords is None:
            return Action.IDLE

        # If no path is currently stored, build one from the latest pose estimate.
        if not self.global_path:
            self.plan_astar_path()
            if not self.global_path:
                return Action.IDLE

        robot = np.array([self.odom.x, self.odom.y], dtype=np.float32)
        goal = np.array(self.goal_world_coords, dtype=np.float32)

        # 1. true metric distance to the goal, and
        # 2. distance to the last point of the current A* path.
        #
        # The second one helps because the path endpoint can sometimes be a slightly
        # more stable stopping reference than the raw goal point itself.
        goal_dist = np.linalg.norm(goal - robot)

        path_end_dist = np.inf
        if self.global_path:
            path_end = np.array(self.global_path[-1], dtype=np.float32)
            path_end_dist = np.linalg.norm(path_end - robot)

        # Either being at the actual goal or effectively at the final path endpoint is
        # treated as good enough for check-in.
        if goal_dist < self.goal_reach_dist or path_end_dist < 0.035:
            print(f"[AUTO] Goal reached. CHECKIN. goal_dist={goal_dist:.3f}, path_end_dist={path_end_dist:.3f}")
            self.is_autonomous = False
            return Action.CHECKIN

        # If visual relocalization or drift has pulled the pose too far away from the
        # current route, rebuild the path from the new pose instead of stubbornly
        # following an outdated one.
        if self._distance_to_path() > self.path_replan_dist:
            self.plan_astar_path()
            if not self.global_path:
                return Action.IDLE

        # Prune waypoints we have already reached
        while len(self.global_path) > 1:
            wp0 = np.array(self.global_path[0], dtype=np.float32)
            if np.linalg.norm(wp0 - robot) < 0.08:
                self.global_path.pop(0)
            else:
                break

        # Near the goal, it is cleaner to drive directly toward the goal point.
        # Farther away, we use the path itself and sometimes look one point ahead
        # so the robot does not get stuck making tiny turns at every small waypoint.
        if goal_dist < 0.08:
            target_pt = goal
        elif len(self.global_path) > 1 and np.linalg.norm(np.array(self.global_path[0], dtype=np.float32) - robot) < 0.12:
            target_pt = np.array(self.global_path[1], dtype=np.float32)
        else:
            target_pt = np.array(self.global_path[0], dtype=np.float32)

        dx = float(target_pt[0] - self.odom.x)
        dy = float(target_pt[1] - self.odom.y)
        target_heading = np.arctan2(dy, dx)
        heading_error = self._wrap_angle(target_heading - self.odom.theta)

        # Turn tolerance controls how precisely we force heading alignment before
        # moving forward.
        #
        # Near the goal we use a tighter tolerance for better final accuracy.
        # Away from the goal we use a looser tolerance so the robot can follow the
        # route more fluidly instead of over-rotating at every small bend.
        if goal_dist < 0.08:
            TURN_TOL = np.deg2rad(40)
        else:
            TURN_TOL = np.deg2rad(4)

        # Before committing to forward motion, do a short rollout on the map.
        # If that rollout would hit a wall, rotate first instead of pushing forward.
        if not self._is_forward_safe(samples=5, step_dt=0.01):
            return Action.LEFT if heading_error >= 0 else Action.RIGHT

        # Turn in place until reasonably aligned to the next path segment
        if heading_error > TURN_TOL:
            self.frame_count += 1       # weight less on the rotation due to slow rotate
            return Action.LEFT
        elif heading_error < -TURN_TOL:
            self.frame_count += 1       # weight less on the rotation due to slow rotate
            return Action.RIGHT
        else:
            self.frame_count += 10      # weight more on the forward movement
            return Action.FORWARD

    def act(self):
        """
        Handle keyboard events and switch between manual and autonomous control.

        Manual key presses immediately disable auto mode. Pressing A toggles auto mode.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.is_autonomous = not self.is_autonomous
                    self.last_act = Action.IDLE
                    print(f"[AUTO] {'ON' if self.is_autonomous else 'OFF'}")
                    if self.is_autonomous and self.odom is not None:
                        self.plan_astar_path()
                elif event.key in self.keymap:
                    self.is_autonomous = False
                    self.last_act |= self.keymap[event.key]
                else:
                    self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]

        if self.is_autonomous:
            self.last_act = self.get_autonomous_action()

        return self.last_act

    def see(self, fpv):
        """
        Main perception / update loop during navigation.

        This is where the online pipeline actually runs:
        startup localization, odometry integration, local geometric correction,
        periodic visual relocalization, and visualization updates.
        """
        if fpv is None or len(fpv.shape) < 3:
            return
        self.fpv = fpv
        current_time = time.time()

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("KeyboardPlayer:fpv")

        if self._state and self._state[1] == Phase.NAVIGATION:
            
            # -------------------------------------------------------------------
            # 1. Global localization at the start of navigation
            # -------------------------------------------------------------------
            # We only do this once. The first live frame is matched against the
            # exploration database, and the matched frame pose becomes our initial
            # metric pose in the map.
            if self.odom is None:
                feat = self.extractor.extract(self.fpv)
                best_idx = int(np.argmax(self.database @ feat))
                best_file = self.file_list[best_idx]
                
                if best_file in self.frame_poses:
                    wx, wy, wtheta = self.frame_poses[best_file]
                    self.odom = Odometry(wx, wy, wtheta)
                    print(f"\n[!] VLAD Initialized Position: X={wx:.2f}, Y={wy:.2f}, Theta={np.rad2deg(wtheta):.0f}deg")
                else:
                    # Fallback only in case pose lookup fails.
                    self.odom = Odometry(0.0, 0.0, 0.0)
                self.last_time = current_time

                # Build the A* path once from the initial localized pose
                if self.goal_world_coords is not None and not self.global_path:
                    self.plan_astar_path() 

            # ---------------------------------------------------------------
            # 2. Kinematic integration from the previous action.
            # ---------------------------------------------------------------
            elif self.last_time is not None:
                dt = 0.01 
                v, w = 0.0, 0.0
                base_v = 2.9462
                base_w = 4.27 
                
                # Reconstruct the commanded linear / angular motion from the
                # last discrete action.
                is_back = bool(self.last_act & Action.BACKWARD)
                if self.last_act & Action.FORWARD: v += base_v
                if is_back: v -= base_v
                if self.last_act & Action.LEFT: w += -base_w if is_back else base_w
                if self.last_act & Action.RIGHT: w += base_w if is_back else -base_w
                
                next_x = self.odom.x + v * dt * np.cos(self.odom.theta)
                next_y = self.odom.y + v * dt * np.sin(self.odom.theta)
                
                # In auto mode we force the full step to be valid before moving.
                # This avoids the robot "sliding" into walls one axis at a time.
                if self.is_autonomous and abs(v) > 0:
                    if self.is_state_valid(next_x, next_y):
                        self.odom.x = next_x
                        self.odom.y = next_y
                else:
                    # Manual mode keeps the older sliding behavior so the user
                    # still has direct interactive control.
                    if self.is_state_valid(next_x, self.odom.y):
                        self.odom.x = next_x
                    if self.is_state_valid(self.odom.x, next_y):
                        self.odom.y = next_y
                self.odom.theta += w * dt
                self.odom.theta = (self.odom.theta + np.pi) % (2 * np.pi) - np.pi

            # ---------------------------------------------------------------
            # 3. Local geometric correction from wall structure.
            # ---------------------------------------------------------------
            # Every few frames, use the wall detector to estimate the dominant wall
            # direction and softly pull the heading back toward the nearest Manhattan
            # orientation. This keeps the pose from slowly rotating away from the maze.
            if self.frame_count >= 200 and self.odom is not None:   # skip 200 frames for fast processing
                all_local = self.wall_detector.extract_wall_points(self.fpv)
                self.frame_count -= 200                             # revert 200 back for the next processing
                
                if len(all_local) > 15:
                    pts = np.array(all_local, dtype=np.float32)
                    step_size = 3
                    dx = pts[step_size:, 0] - pts[:-step_size, 0]
                    dy = pts[step_size:, 1] - pts[:-step_size, 1]
                    
                    local_angles = np.arctan2(dy, dx)
                    hist, bin_edges = np.histogram(local_angles, bins=36, range=(-np.pi, np.pi))
                    dominant_bin = np.argmax(hist)
                    dominant_local_angle = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2.0
                    
                    # Keep only wall segments that agree with the dominant local direction.
                    angle_diffs = np.abs(np.arctan2(np.sin(local_angles - dominant_local_angle), 
                                                    np.cos(local_angles - dominant_local_angle)))
                    inlier_mask_segments = angle_diffs < np.deg2rad(15)
                    
                    if np.sum(inlier_mask_segments) > 5:
                        precise_local_angle = np.arctan2(
                            np.mean(np.sin(local_angles[inlier_mask_segments])), 
                            np.mean(np.cos(local_angles[inlier_mask_segments]))
                        )
                        
                        # Estimate the dominant wall angle in global coordinates,
                        # then snap that angle to the nearest Manhattan direction.
                        global_wall_angle = self.odom.theta + precise_local_angle
                        target_global_angle = np.round(global_wall_angle / (np.pi/2.0)) * (np.pi/2.0)
                        heading_error = target_global_angle - global_wall_angle
                        
                        # Soft heading correction only. This keeps the correction stable
                        # and avoids sudden jumps.
                        self.odom.theta += heading_error * 0.15
                        self.odom.theta = (self.odom.theta + np.pi) % (2 * np.pi) - np.pi
                        
                        # In manual mode only, also allow gentle x/y correction based on
                        # recurring wall locations. In auto mode we avoid this because it
                        # can make the map marker move while the real robot is actually stuck.
                        if self.is_autonomous:
                            inlier_mask_pts = np.append(inlier_mask_segments, [False]*step_size)
                            valid_local_pts = pts[inlier_mask_pts]
                            
                            if len(valid_local_pts) > 0:
                                global_X = self.odom.x + (valid_local_pts[:, 0] * np.cos(self.odom.theta) - valid_local_pts[:, 1] * np.sin(self.odom.theta))
                                global_Y = self.odom.y + (valid_local_pts[:, 0] * np.sin(self.odom.theta) + valid_local_pts[:, 1] * np.cos(self.odom.theta))
                                
                                mean_wall_x = np.mean(global_X)
                                mean_wall_y = np.mean(global_Y)
                                
                                norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                                is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                                is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                                translation_gain = 0.6
                                GRID_SIZE = 0.4

                                # Normalize target angle to stay within -pi to pi
                                norm_target = (target_global_angle + np.pi) % (2 * np.pi) - np.pi
                                # Classify the wall as horizontal or vertical
                                is_horizontal = np.isclose(abs(norm_target), 0.0, atol=0.1) or np.isclose(abs(norm_target), np.pi, atol=0.1)
                                is_vertical = np.isclose(abs(norm_target), np.pi/2.0, atol=0.1)
                                
                                if is_horizontal:
                                    # Snap the wall to the nearest absolute 0.4m grid line
                                    target_y = np.round(mean_wall_y / GRID_SIZE) * GRID_SIZE
                                    if abs(target_y - mean_wall_y) < 0.18:
                                        # Gently adjust the robot's y position toward the idead y position
                                        self.odom.y += (target_y - mean_wall_y) * translation_gain
                                            
                                elif is_vertical:
                                    # Snap the wall to the nearest absolute 0.4m grid line
                                    target_x = np.round(mean_wall_x / GRID_SIZE) * GRID_SIZE
                                    if abs(target_x - mean_wall_x) < 0.18:
                                        # Gently adjust the robot's x position toward the idead x position
                                        self.odom.x += (target_x - mean_wall_x) * translation_gain
                if self.viz:
                    self.display_global_map()

        # Always refresh the live FPV display, even if we are not currently navigating.
        rgb = fpv[:, :, ::-1]
        surface = pygame.image.frombuffer(rgb.tobytes(), rgb.shape[1::-1], 'RGB')
        self.screen.blit(surface, (0, 0))
        pygame.display.update()

    def is_state_valid(self, x, y):
        """Check whether a world-coordinate point lies in free space on the cleaned map."""
        if self.occupancy_map is None: 
            return True # Failsafe if map isn't loaded
            
        px = int(self.map_offset + (x * self.map_scale))
        py = int(self.map_offset - (y * self.map_scale))
        
        h, w = self.occupancy_map.shape
        if 0 <= px < w and 0 <= py < h:
            # Walls are usually near 0 (black).
            return self.occupancy_map[py, px] > 50
        return False # Out of bounds is invalid
    
    # -----------------------------------------------------------------------
    # Setup and Visualizer Methods
    # -----------------------------------------------------------------------
    def set_target_images(self, images):
        """Store the target views and open the target image window."""
        super().set_target_images(images)
        self.show_target_images()

    def pre_navigation(self):
        """Build the visual database, the retrieval graph, and the goal estimate before navigation starts."""
        super().pre_navigation()
        self._build_database()
        self._build_graph()
        self._setup_goal()

    def _build_database(self):
        """Load / build the full VLAD database over the exploration images."""
        if self.database is not None:
            return
        self.extractor.load_sift_cache(self.file_list, self.subsample_rate)
        self.extractor.build_vocabulary(self.file_list)
        self.database = self.extractor.extract_batch(self.file_list)

    def _build_graph(self):
        """Build the temporal + visual shortcut graph used for retrieval-side reasoning and debugging."""
        if self.G is not None: return
        n = len(self.database)
        self.G = nx.DiGraph() 
        self.G.add_nodes_from(range(n))

        for i in range(n - 1):
            self.G.add_edge(i, i + 1, weight=TEMPORAL_WEIGHT, edge_type="temporal")

        sim = self.database @ self.database.T
        np.fill_diagonal(sim, -2)

        for i in range(n):
            lo = max(0, i - MIN_SHORTCUT_GAP)
            hi = min(n, i + MIN_SHORTCUT_GAP + 1)
            sim[i, lo:hi] = -2
        sim[~np.triu(np.ones((n, n), dtype=bool), k=1)] = -2

        flat = sim.ravel()
        top_k = self.top_k_shortcuts
        top_idx = np.argpartition(flat, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(-flat[top_idx])]

        for rank, fi in enumerate(top_idx):
            i, j = divmod(int(fi), n)
            s = float(flat[fi])
            d = float(np.sqrt(max(0, 2 - 2 * s)))
            self.G.add_edge(i, j, weight=VISUAL_WEIGHT_BASE + VISUAL_WEIGHT_SCALE * d, edge_type="visual")

    def _setup_goal(self):
        if self.goal_node is not None:
            return

        targets = self.get_target_images()
        if not targets:
            return

        front = targets[0]   # only use front view
        feat = self.extractor.extract(front)
        sims = self.database @ feat

        smooth = sims.copy()
        for i in range(1, len(smooth) - 1):
            smooth[i] = 0.25 * sims[i - 1] + 0.5 * sims[i] + 0.25 * sims[i + 1]

        self.goal_node = int(np.argmax(smooth))

        goal_file = self.file_list[self.goal_node]
        if goal_file in self.frame_poses:
            gx, gy, _ = self.frame_poses[goal_file]
            self.goal_world_coords = (gx, gy)
            print(f"[GOAL] node={self.goal_node}, world=({gx:.2f}, {gy:.2f})")
        else:
            print(f"[GOAL] Warning: {goal_file} not found in frame_poses")

    def _load_img(self, idx: int) -> np.ndarray | None:
        """Load an exploration image by node index."""
        if 0 <= idx < len(self.file_list):
            return cv2.imread(os.path.join(IMAGE_DIR, self.file_list[idx]))
        return None

    def _get_current_node(self) -> int:
        """Return the best-matching exploration node for the current FPV frame."""
        feat = self.extractor.extract(self.fpv)
        return int(np.argmax(self.database @ feat))

    def _get_path(self, start: int) -> list[int]:
        """Return a shortest path on the retrieval graph from the current node to the goal node."""
        try:
            return nx.shortest_path(self.G, start, self.goal_node, weight="weight")
        except nx.NetworkXNoPath:
            return [start]

    def _edge_action(self, a: int, b: int) -> str:
        """Translate a temporal graph edge into a forward/back/left/right label for the debug panel."""
        REVERSE = {'FORWARD': 'BACKWARD', 'BACKWARD': 'FORWARD', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}
        if b == a + 1 and a < len(self.motion_frames):
            return self.motion_frames[a]['action']
        elif b == a - 1 and b < len(self.motion_frames):
            return REVERSE.get(self.motion_frames[b]['action'], '?')
        return '?'

    def show_target_images(self):
        """Display the four target views in one window for quick reference."""
        targets = self.get_target_images()
        if not targets: return
        top = cv2.hconcat(targets[:2])
        bot = cv2.hconcat(targets[2:])
        img = cv2.vconcat([top, bot])
        h, w = img.shape[:2]
        cv2.line(img, (w // 2, 0), (w // 2, h), (0, 0, 0), 2)
        cv2.line(img, (0, h // 2), (w, h // 2), (0, 0, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        for label, pos in [('Front', (10, 25)), ('Right', (w//2+10, 25)),
                           ('Back', (10, h//2+25)), ('Left', (w//2+10, h//2+25))]:
            cv2.putText(img, label, pos, font, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow('Target Images', img)
        cv2.waitKey(1)

    def display_global_map(self):
        """Show the global metric map with the current pose, target, and A* path."""
        if self.slam_map is None or self.goal_node is None: return
        display_map = self.slam_map.copy()

        goal_file = self.file_list[self.goal_node]
        if goal_file in self.frame_poses:
            gx, gy, _ = self.frame_poses[goal_file]
            gpx, gpy = self.world_to_pixel(gx, gy)
            cv2.circle(display_map, (gpx, gpy), 10, (0, 0, 255), -1) 
            cv2.putText(display_map, "TARGET", (gpx + 15, gpy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw A* geometric shortest path (blue/cyan style)
        if self.global_path:
            for i in range(len(self.global_path) - 1):
                p1 = self.world_to_pixel(self.global_path[i][0], self.global_path[i][1])
                p2 = self.world_to_pixel(self.global_path[i + 1][0], self.global_path[i + 1][1])
                cv2.line(display_map, p1, p2, (255, 200, 0), 2)
                cv2.circle(display_map, p1, 2, (255, 200, 0), -1)

        # Draw current path start marker from odometry
        if self.odom is not None:
            rpx, rpy = self.world_to_pixel(self.odom.x, self.odom.y)
            cv2.circle(display_map, (rpx, rpy), 6, (0, 255, 0), -1)
            hx = int(rpx + 15 * np.cos(self.odom.theta))
            hy = int(rpy - 15 * np.sin(self.odom.theta))
            cv2.line(display_map, (rpx, rpy), (hx, hy), (0, 255, 0), 2)

        mode_txt = "AUTO" if self.is_autonomous else "MANUAL"
        mode_col = (0, 255, 0) if self.is_autonomous else (0, 0, 255)
        cv2.putText(display_map, "A* PATH", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 0), 2, cv2.LINE_AA)
        cv2.putText(display_map, mode_txt, (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_col, 2, cv2.LINE_AA)

        cv2.imshow("Global Metric Map (Live Odometry + Constraints)", display_map)
        cv2.waitKey(1)

if __name__ == "__main__":
    """Run the navigation player directly with command-line configuration options."""
    import argparse
    import vis_nav_game

    parser = argparse.ArgumentParser()
    parser.add_argument("--subsample", type=int, default=5, help="Take every Nth motion frame (default: 5)")
    parser.add_argument("--n-clusters", type=int, default=128, help="VLAD codebook size (default: 128)")
    parser.add_argument("--top-k", type=int, default=30, help="Number of global visual shortcut edges (default: 30)")
    parser.add_argument("--viz", type=int, default=1, help="Shows the SLAM map (1=True, 0=False)")

    args = parser.parse_args()

    vis_nav_game.play(the_player=KeyboardPlayerPyGame(
        n_clusters=args.n_clusters,
        subsample_rate=args.subsample,
        top_k_shortcuts=args.top_k,
        viz=bool(args.viz),
    ))
