import os
import cv2
import json
import torch
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

try:
    from locate_walls import create_wall_mask
except ImportError:
    print("[WARNING] locate_walls.py not found. Tier 2 wall matching will fail.")
    create_wall_mask = None

try:
    from vis_nav_game import Player, Action, Phase
    import pygame
except ImportError:
    Player, Action, Phase = object, None, None
    pygame = None


class VisualLocalizer:
    def __init__(self, graph_path, embeddings_path, img_dir, device=None):
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
            
        self.img_dir = img_dir
        
        with open(graph_path, 'r') as f:
            raw_graph = json.load(f)
            self.graph = {int(k): v for k, v in raw_graph.items()}
            
        self.total_nodes = len(self.graph)
        self.embeddings = torch.load(embeddings_path, map_location=self.device)
        
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = nn.Sequential(*list(model.children())[:-1]).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.sift = cv2.SIFT_create()
        
        # Build an undirected graph for accurate topological radius searching
        self.undirected_graph = {n: set() for n in self.graph.keys()}
        for node, data in self.graph.items():
            if node > 0:
                self.undirected_graph[node].add(node - 1)
                self.undirected_graph[node - 1].add(node)
                
            for edge in data.get("edges", []):
                self.undirected_graph[node].add(edge)
                if edge in self.undirected_graph:
                    self.undirected_graph[edge].add(node)

    def _get_all_nodes_ordered_by_distance(self, start_node):
        visited = {start_node}
        queue = [start_node]
        ordered_nodes = [start_node]
        
        while queue:
            curr = queue.pop(0)
            for neighbor in self.undirected_graph.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    ordered_nodes.append(neighbor)
                    
        for node in self.graph.keys():
            if node not in visited:
                ordered_nodes.append(node)
                
        return ordered_nodes

    def _get_masked_ransac_inliers(self, img1_gray, mask1, img2_path):
        img2 = cv2.imread(img2_path)
        if img2 is None:
            return 0, None
            
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        _, mask2 = create_wall_mask(img2_path)
        if mask2 is None:
            return 0, None

        kp1, des1 = self.sift.detectAndCompute(img1_gray, mask1)
        kp2, des2 = self.sift.detectAndCompute(img2_gray, mask2)
        
        if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
            return 0, None

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        
        good_matches = []
        max_angle_diff = 30.0 
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                angle1 = kp1[m.queryIdx].angle
                angle2 = kp2[m.trainIdx].angle
                diff = abs(angle1 - angle2)
                diff = min(diff, 360.0 - diff) 
                if diff <= max_angle_diff:
                    good_matches.append(m)
                
        if len(good_matches) < 4:
            return 0, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if inlier_mask is None:
            return 0, None
            
        return int(np.sum(inlier_mask)), M

    def localize(self, current_frame_bgr, last_known_node, expected_path=None, 
                 high_sim_thresh=0.98, global_sim_thresh=0.85, frame_retriever=None):
        
        # NEW WARP LOGIC: Shift the "guess" forward by 4 nodes along the path
        if expected_path and last_known_node in expected_path:
            idx = expected_path.index(last_known_node)
            guess_idx = min(len(expected_path) - 1, idx + 4)
            guessed_node = expected_path[guess_idx]
            print(f"[LOCALIZER] Warping guess +4 nodes along path: Node {last_known_node} -> Node {guessed_node}")
        else:
            guessed_node = last_known_node + 4
            print(f"[LOCALIZER] Warping guess +4 nodes naively: Node {last_known_node} -> Node {guessed_node}")

        # Update window_nodes to center around the warped guessed_node
        window_nodes = []
        if expected_path and guessed_node in expected_path:
            curr_idx = expected_path.index(guessed_node)
            start_idx = max(0, curr_idx - 5)
            end_idx = min(len(expected_path), curr_idx + 16)
            window_nodes = [expected_path[i] for i in range(start_idx, end_idx)]
        else:
            search_start = max(0, guessed_node - 5)
            search_end = min(self.total_nodes, guessed_node + 16)
            window_nodes = list(range(search_start, search_end))

        # ==========================================
        # TIER 1: High-Confidence Route ResNet Match
        # ==========================================
        frame_rgb = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB)
        current_feat = self._extract_feature(frame_rgb)
        similarities = torch.matmul(self.embeddings, current_feat.T).squeeze().cpu().numpy()

        best_sim = -1.0
        best_sim_node = None
        
        for node in window_nodes:
            if similarities[node] > best_sim:
                best_sim = similarities[node]
                best_sim_node = node

        if best_sim >= high_sim_thresh:
            print(f"[TIER 1] Localized to Node {best_sim_node} on planned route (Sim: {best_sim:.3f} >= {high_sim_thresh})")
            return best_sim_node

        # ==========================================
        # TIER 2: Wall-Masked SIFT/RANSAC Fallback
        # ==========================================
        print(f"[TIER 2] ResNet fell below {high_sim_thresh}. Initiating Wall Matching...")
        
        temp_img_path = "temp_loc_current.jpg"
        obs_mask = None
        
        while True:
            cv2.imwrite(temp_img_path, current_frame_bgr)
            _, obs_mask = create_wall_mask(temp_img_path)
            
            if obs_mask is not None and np.count_nonzero(obs_mask) > 500:
                break
                
            print("\n[WARNING] No walls identified in the current view!")
            ans = input("Adjust camera and press ENTER to retry, or type 'skip' to force Tier 3: ")
            
            if ans.lower().strip() == 'skip':
                obs_mask = None
                break
                
            if frame_retriever:
                new_frame = frame_retriever()
                if new_frame is not None:
                    current_frame_bgr = new_frame
        
        if obs_mask is not None:
            frame_gray = cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2GRAY)
            node_scores = []
            
            print(f"Testing current walls against map walls in range {window_nodes[0]} to {window_nodes[-1]}...")
            for node in window_nodes:
                map_img_path = os.path.join(self.img_dir, self.graph[node]["image"])
                inliers, _ = self._get_masked_ransac_inliers(frame_gray, obs_mask, map_img_path)
                
                if inliers >= 15: 
                    node_scores.append({"node": node, "inliers": inliers})
            
            if node_scores:
                ranked_nodes = sorted(node_scores, key=lambda x: x["inliers"], reverse=True)
                print("[TIER 2] Walls matched with sufficient inliers at the following nodes:")
                for n in ranked_nodes:
                    print(f"  -> Node {n['node']}: {n['inliers']} inliers")
                    
                best_match = ranked_nodes[0]
                print(f"[SUCCESS] Localized to Node {best_match['node']} (Most inliers).")
                
                if os.path.exists(temp_img_path): os.remove(temp_img_path)
                return best_match['node']
            else:
                print("[TIER 2] No walls matched sufficiently in the expected route window.")
                
        if os.path.exists(temp_img_path): os.remove(temp_img_path)

        # ==========================================
        # TIER 3: Global ResNet Search (>85%)
        # ==========================================
        print(f"\n[TIER 3] Fallback: Searching ALL nodes for ResNet similarity > {global_sim_thresh}...")
        
        ordered_global_nodes = self._get_all_nodes_ordered_by_distance(guessed_node)
        
        for node in ordered_global_nodes:
            if similarities[node] >= global_sim_thresh:
                print(f"[SUCCESS] Tier 3 Localized to Node {node} (Sim: {similarities[node]:.3f})")
                return node

        print("[ERROR] Lost: Exhausted all tiers. No valid matches found.")
        return None

    def _extract_feature(self, frame_rgb):
        img = Image.fromarray(frame_rgb)
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.flatten(feat, 1)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            
        return feat

class LocalizerTestPlayer(Player):
    def __init__(self, localizer, start_node=0, direction="forward"):
        super().__init__()
        self.localizer = localizer
        self.last_known_node = start_node
        self.direction = direction
        
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.q_pressed_last_frame = False

    def reset(self):
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        pygame.init()
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        pygame.display.set_caption(f"Localizer Test (Press 'Q') | Dir: {self.direction}")

        if self._state and self._state[1] == Phase.NAVIGATION:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_q] and not self.q_pressed_last_frame:
                print(f"\n[LOCALIZER] Triggered manually. Last known: {self.last_known_node}, Dir: {self.direction}")
                
                def get_current_frame():
                    return self.fpv
                
                current_node = self.localizer.localize(
                    current_frame_bgr=self.fpv, 
                    last_known_node=self.last_known_node, 
                    frame_retriever=get_current_frame
                )
                
                if current_node is not None:
                    print(f"[LOCALIZER] Localized to Node {current_node}")
                    self.last_known_node = current_node
                
                self.q_pressed_last_frame = True
            elif not keys[pygame.K_q]:
                self.q_pressed_last_frame = False

        rgb = fpv[:, :, ::-1]
        shape = rgb.shape[1::-1]
        pygame_image = pygame.image.frombuffer(rgb.tobytes(), shape, 'RGB')
        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Visual Localizer")
    parser.add_argument("--mode", choices=["game", "image"], required=True, 
                        help="Run in 'game' mode (Pygame) or 'image' mode (static image testing).")
    parser.add_argument("--image", type=str, 
                        help="Path to the test image (Required if mode is 'image')")
    parser.add_argument("--last_known", type=int, default=0, 
                        help="The last known node ID to start the search from")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward",
                        help="Simulate the direction of travel")
    parser.add_argument("--graph", type=str, default="topological_graph.json", 
                        help="Path to topological graph JSON")
    parser.add_argument("--embeddings", type=str, default="node_embeddings.pt", 
                        help="Path to precomputed embeddings tensor")
    parser.add_argument("--img_dir", type=str, default="data/exploration_data/images", 
                        help="Directory containing exploration images")
    
    args = parser.parse_args()

    if not (os.path.exists(args.graph) and os.path.exists(args.embeddings)):
        print(f"Error: Missing {args.graph} or {args.embeddings}. Run preprocessing.py first.")
        exit(1)

    print("Initializing Visual Localizer...")
    localizer = VisualLocalizer(args.graph, args.embeddings, args.img_dir)

    if args.mode == "image":
        if not args.image or not os.path.exists(args.image):
            print("Error: You must provide a valid --image path when running in 'image' mode.")
            exit(1)
            
        print(f"Testing localizer on static image: {args.image} (Direction: {args.direction})")
        test_frame = cv2.imread(args.image)
        
        if test_frame is not None:
            current_node = localizer.localize(test_frame, args.last_known)
            print(f"\nResult: Localized to Node {current_node}")

    elif args.mode == "game":
        if pygame is None:
            print("Error: vis_nav_game and pygame are required to run in 'game' mode.")
            exit(1)
            
        import vis_nav_game as vng
        print("Starting Pygame environment...")
        
        player = LocalizerTestPlayer(localizer, start_node=args.last_known, direction=args.direction)
        vng.play(the_player=player)