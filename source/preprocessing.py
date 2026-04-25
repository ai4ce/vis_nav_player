import json
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from tqdm import tqdm

def load_and_subsample_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # 1. Filter out steps where the action is strictly ["IDLE"]
    active_steps = [entry for entry in data if "IDLE" not in entry.get('action', [])]
    
    # 2. Subsample: Take 1 out of every 8 remaining images
    subsampled_data = active_steps[::8]
    
    print(f"Original steps: {len(data)}")
    print(f"Active steps: {len(active_steps)}")
    print(f"Subsampled steps: {len(subsampled_data)}")
    
    return subsampled_data

def verify_match_with_ransac(img1_path, img2_path, min_inliers=25, min_width_ratio=0.25):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return False

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return False

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
        return False

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if mask is None:
        return False
        
    inlier_count = np.sum(mask)
    if inlier_count < min_inliers:
        return False

    # Horizontal Spread Check
    inlier_pts = dst_pts[mask.ravel() == 1]
    if len(inlier_pts) == 0:
        return False
        
    x_coords = inlier_pts[:, 0, 0]
    image_width = img2.shape[1]
    
    num_bins = 5
    min_active_bins = 3
    min_pts_per_bin = 3
    
    bins = np.linspace(0, image_width, num_bins + 1)
    bin_indices = np.digitize(x_coords, bins) - 1
    
    active_bins = 0
    for b in range(num_bins):
        pts_in_bin = np.sum(bin_indices == b)
        if pts_in_bin >= min_pts_per_bin:
            active_bins += 1
            
    if active_bins < min_active_bins:
        return False
        
    return True

def extract_features(data_list, img_dir, device):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1]).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    embeddings = []
    print("Extracting CNN features for topological nodes...")
    with torch.no_grad():
        for item in tqdm(data_list):
            img_path = os.path.join(img_dir, item['image'])
            img = Image.open(img_path).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            
            feat = model(tensor)
            feat = torch.flatten(feat, 1)
            feat = nn.functional.normalize(feat, p=2, dim=1)
            embeddings.append(feat)
            
    return torch.cat(embeddings, dim=0), model, transform

def build_topological_graph(data_list, embeddings, img_dir, time_margin=20):
    print("Computing similarity matrix for loop closures...")
    sim_matrix = torch.matmul(embeddings, embeddings.T) 
    
    graph = {}
    window_size = 1 
    
    print("Building topological graph with shortcuts...")
    for i in tqdm(range(len(data_list))):
        # Initialize node with standard sequential connection
        edges = []
        if i < len(data_list) - 1:
            edges.append(i + 1)
            
        # Mine for loop closures / shortcuts
        for j in range(len(data_list)):
            if abs(i - j) > time_margin:
                seq_match = True
                
                for w in range(-window_size, window_size + 1):
                    w_i, w_j = i + w, j + w
                    if 0 <= w_i < len(data_list) and 0 <= w_j < len(data_list):
                        if sim_matrix[w_i, w_j] < 0.85: 
                            seq_match = False
                            break
                    else:
                        seq_match = False
                        break
                        
                if seq_match and sim_matrix[i, j] > 0.88:
                    img1_path = os.path.join(img_dir, data_list[i]['image'])
                    img2_path = os.path.join(img_dir, data_list[j]['image'])
                    
                    if verify_match_with_ransac(img1_path, img2_path):
                        if j not in edges:
                            edges.append(j)
                            
        graph[i] = {
            "image": data_list[i]['image'],
            "action": data_list[i].get('action', []),
            "edges": edges
        }
        
    return graph

def localize_target(target_path, img_dir, data_list, embeddings, model, transform, device):
    print(f"Localizing target image: {target_path}")
    
    # 1. target.jpg is a 2x2 mosaic. Extract the top-left quadrant (front view).
    mosaic_bgr = cv2.imread(target_path)
    if mosaic_bgr is None:
        print(f"[ERROR] Could not load {target_path}")
        return 0
        
    h, w = mosaic_bgr.shape[:2]
    front_bgr = mosaic_bgr[:h // 2, :w // 2]
    front_rgb = cv2.cvtColor(front_bgr, cv2.COLOR_BGR2RGB)
    
    # Save the cropped version temporarily so RANSAC can load it from disk
    temp_target_path = "temp_front_target.jpg"
    cv2.imwrite(temp_target_path, front_bgr)
    
    # 2. Convert to PIL Image and extract CNN feature
    img = Image.fromarray(front_rgb)
    tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        target_feat = model(tensor)
        target_feat = torch.flatten(target_feat, 1)
        target_feat = nn.functional.normalize(target_feat, p=2, dim=1)
        
    # 3. Find highest cosine similarities
    similarities = torch.matmul(embeddings, target_feat.T).squeeze()
    top_k_indices = torch.topk(similarities, k=5).indices.cpu().numpy()
    
    # 4. Verify with RANSAC
    for idx in top_k_indices:
        candidate_path = os.path.join(img_dir, data_list[idx]['image'])
        # Pass the cropped temp image to RANSAC, not the original mosaic
        if verify_match_with_ransac(temp_target_path, candidate_path):
            print(f"Target matched verified via RANSAC at Node {idx} (Similarity: {similarities[idx]:.4f})")
            
            # Clean up temp file
            if os.path.exists(temp_target_path):
                os.remove(temp_target_path)
                
            return int(idx)
            
    # Clean up temp file if RANSAC fails
    if os.path.exists(temp_target_path):
        os.remove(temp_target_path)
        
    # Fallback if RANSAC fails on all top candidates
    best_idx = int(top_k_indices[0])
    print(f"Warning: RANSAC verification failed. Falling back to highest CNN similarity at Node {best_idx}.")
    return best_idx

if __name__ == "__main__":
    # Paths
    json_path = 'data/exploration_data/data_info.json'
    img_dir = 'data/exploration_data/images'
    target_img_path = 'target.jpg'
    
    output_graph_path = 'topological_graph.json'
    output_embeddings_path = 'node_embeddings.pt'
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Running preprocessing on {device}")

    # 1. Load and Subsample Data
    subsampled_data = load_and_subsample_data(json_path)

    # 2. Extract Features
    embeddings, model, transform = extract_features(subsampled_data, img_dir, device)
    
    # Save embeddings for instant access by visual_localizer.py and global_planner.py
    torch.save(embeddings, output_embeddings_path)
    print(f"Saved precomputed map features to {output_embeddings_path}")

    # 3. Build Topological Graph
    topological_graph = build_topological_graph(subsampled_data, embeddings, img_dir)
    
    # Save graph
    with open(output_graph_path, 'w') as f:
        json.dump(topological_graph, f, indent=4)
    print(f"Saved topological graph to {output_graph_path}")

    # 4. Localize Target Image
    if os.path.exists(target_img_path):
        target_node = localize_target(
            target_img_path, img_dir, subsampled_data, embeddings, model, transform, device
        )
        
        # Extract the original filename from the data list using the returned node ID
        matched_image_filename = subsampled_data[target_node]['image']
        
        print(f"\n[PIPELINE OUTPUT] Target Node ID: {target_node}")
        print(f"[PIPELINE OUTPUT] Matched Exploration Image: {matched_image_filename}")
        
        # NEW: Output a target info file so test_run doesn't need a manual argument
        with open('target_info.json', 'w') as f:
            json.dump({"target_node": target_node}, f)
        print("Saved target_node to target_info.json")
    else:
        print(f"\n[WARNING] Target image {target_img_path} not found. Skipping target localization.")