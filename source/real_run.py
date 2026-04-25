import os
import cv2
import json
import argparse
import numpy as np 
import pygame
import vis_nav_game
from vis_nav_game import Player, Action, Phase

# Import your localizer and planner
from visual_localizer import VisualLocalizer
from global_planner import GlobalPlanner

class LocalizerTestPlayer(Player):
    def __init__(self, localizer, planner, graph_data, target_node, img_dir, start_node=0, direction="forward"):
        super().__init__()
        self.localizer = localizer
        self.planner = planner
        self.graph_data = graph_data
        self.target_node = target_node
        self.img_dir = img_dir 
        self.last_known_node = start_node
        self.direction = direction
        
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None
        self.keymap = None
        self.q_pressed_last_frame = False
        self.current_path = [] # NEW: Retain the path state

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
        self.w_pressed_last_frame = False

    def pre_navigation(self):
        super().pre_navigation()
        self.last_act = Action.IDLE
        self.q_pressed_last_frame = False
        pygame.event.clear() 
        
        print("\n" + "="*50)
        print(" EXPLORATION SKIPPED/COMPLETE. STARTING NAVIGATION PHASE.")
        print(f" Target Node: {self.target_node}")
        print(" You have full control. Press 'Q' to localize and plan.")
        print("="*50 + "\n")

    def act(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    self.last_act |= self.keymap[event.key]
                else:
                    self._show_target_images()
                    
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    self.last_act &= ~self.keymap[event.key]
                    
        return self.last_act

    def _show_target_images(self):
        targets = self.get_target_images()
        if targets is None or len(targets) <= 0:
            return
            
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])
        cv2.imshow('Target Goals', concat_img)
        cv2.waitKey(1)

    def _get_node_image(self, node_id, label, target_size=(320, 240)):
        if node_id is None:
            # NEW: We are near the end of the path, load the exact target image!
            img = cv2.imread('target.jpg')
            if img is not None:
                h, w = img.shape[:2]
                img = img[:h // 2, :w // 2] # Exact front view from mosaic
                img = cv2.resize(img, target_size)
            else:
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(img, "Target Image Missing", (target_size[0]//4, target_size[1]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        else:
            node_key = str(node_id)
            img = None
            if node_key in self.graph_data:
                filename = self.graph_data[node_key].get("image")
                if filename:
                    filepath = os.path.join(self.img_dir, filename)
                    if os.path.exists(filepath):
                        img = cv2.imread(filepath)
            
            if img is None:
                # Blank image for missing file
                img = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
                cv2.putText(img, f"Missing File", (target_size[0]//4, target_size[1]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Resize for uniform concatenation
                img = cv2.resize(img, target_size)
                
        # Add the label banner at the top
        text = f"{label}: Node {node_id}" if node_id is not None else f"{label}: Target"
        cv2.rectangle(img, (0, 0), (target_size[0], 40), (0, 0, 0), -1)
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img

    def _show_path_preview(self, path):
        node_0 = path[0] if len(path) > 0 else None
        node_2 = path[2] if len(path) > 2 else None
        node_4 = path[4] if len(path) > 4 else None
        
        img0 = self._get_node_image(node_0, "Current")
        img2 = self._get_node_image(node_2, "")
        img4 = self._get_node_image(node_4, "Target Pose")
        
        concat_img = cv2.hconcat([img0, img2, img4])
        cv2.imshow('Path Preview', concat_img)
        cv2.waitKey(1)

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        if self._state:
            if self._state[1] == Phase.EXPLORATION:
                pygame.display.set_caption("EXPLORATION PLAYBACK - Please Wait...")
            elif self._state[1] == Phase.NAVIGATION:
                pygame.display.set_caption(f"NAV | Dir: {self.direction} | 'Q'=Localize | 'W'=Warp Next Node")

                keys = pygame.key.get_pressed()
                
                if keys[pygame.K_w] and not self.w_pressed_last_frame:
                    path, _ = self.planner.get_movement_strategy(
                        self.last_known_node, 
                        self.target_node, 
                        lookahead_size=2,
                        current_direction=self.direction 
                    )
                    if path and len(path) > 1:
                        next_node = path[1]
                        print(f"\n[SIMULATOR] Warped from Node {self.last_known_node} -> Node {next_node}")
                        self.last_known_node = next_node
                    else:
                        print(f"\n[SIMULATOR] Already at target or no path available.")
                    self.w_pressed_last_frame = True
                elif not keys[pygame.K_w]:
                    self.w_pressed_last_frame = False

                # --- LOCALIZER LOGIC ---
                if keys[pygame.K_q] and not self.q_pressed_last_frame:
                    print(f"\n[LOCALIZER] Triggered. Commencing topological search...")
                    
                    current_node = self.localizer.localize(
                        self.fpv, 
                        self.last_known_node,
                        expected_path=self.current_path # NEW: Pass the path to warp logic
                    )
                    
                    if current_node is not None:
                        print(f"[LOCALIZER] Success! Localized to Node {current_node}")
                        self.last_known_node = current_node
                        
                        print(f"[PLANNER] Calculating path to Target Node {self.target_node}...")
                        path, intents = self.planner.get_movement_strategy(
                            current_node, 
                            self.target_node, 
                            lookahead_size=10,
                            current_direction=self.direction 
                        )
                        
                        self.current_path = path # Save for the next Q warp
                        
                        if path:
                            self._show_path_preview(path)
                        
                        if path and len(path) > 1:
                            print("\n>>> UPCOMING SEQUENCE (Next 10 Nodes) <<<")
                            for i in range(min(len(intents), 10)):
                                print(f"    Step {i+1}: Node {path[i]} -> {path[i+1]} (Intent: {intents[i]})")
                            print("----------------------------------------")
                            
                            final_decision = self.planner.distill_directional_intent(intents)
                            print(f"\n>>> IMMEDIATE DIRECTIONAL INTENT VECTOR: {final_decision}\n")
                        
                    self.q_pressed_last_frame = True
                    
                elif not keys[pygame.K_q]:
                    self.q_pressed_last_frame = False

        rgb = fpv[:, :, ::-1]
        shape = rgb.shape[1::-1]
        pygame_image = pygame.image.frombuffer(rgb.tobytes(), shape, 'RGB')
        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Harness for visual_localizer.py and global_planner.py")
    parser.add_argument("--start_node", type=int, default=0, 
                        help="The initial node ID the robot assumes it is starting at (Default: 0)")
    parser.add_argument("--target_node", type=int, default=None, # Changed from required=True
                        help="The destination node ID the robot needs to reach (Defaults to preprocessing output)")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward",
                        help="Tells the localizer which way the robot is traveling")
    parser.add_argument("--graph", type=str, default="topological_graph.json", 
                        help="Path to topological graph JSON")
    parser.add_argument("--embeddings", type=str, default="node_embeddings.pt", 
                        help="Path to precomputed embeddings tensor")
    parser.add_argument("--img_dir", type=str, default="data/exploration_data/images", 
                        help="Directory containing exploration images")
    
    args = parser.parse_args()

    # NEW: Try to automatically fetch target node from preprocessing phase
    if args.target_node is None:
        if os.path.exists('target_info.json'):
            with open('target_info.json', 'r') as f:
                tinfo = json.load(f)
                args.target_node = tinfo.get("target_node")
                
        if args.target_node is None:
            print("Error: --target_node must be explicitly provided or 'target_info.json' must exist from preprocessing.")
            exit(1)

    if not (os.path.exists(args.graph) and os.path.exists(args.embeddings)):
        print(f"Error: Missing {args.graph} or {args.embeddings}. Run preprocessing.py first.")
        exit(1)

    print(f"Loading Topological Graph from {args.graph}...")
    with open(args.graph, 'r') as f:
        graph_data = json.load(f)

    print("Loading Visual Localizer (CNN weights & SIFT)...")
    localizer = VisualLocalizer(args.graph, args.embeddings, args.img_dir)
    print("Localizer loaded successfully!")
    
    print("\nLoading Global Planner...")
    planner = GlobalPlanner(args.graph)

    print("\nBooting vis_nav_game...")
    player = LocalizerTestPlayer(
        localizer=localizer, 
        planner=planner,
        graph_data=graph_data, 
        target_node=args.target_node,
        img_dir=args.img_dir, 
        start_node=args.start_node, 
        direction=args.direction
    )
    
    vis_nav_game.play(the_player=player)