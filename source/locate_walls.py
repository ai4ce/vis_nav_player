import os
import random
import cv2
import numpy as np


def create_wall_mask(img_path, blur_size=5, close_size=7, open_size=3):
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        print(f"Failed to load {img_path}")
        return None, None

    # ==========================================
    # NEW: TUNABLE PRE-SMOOTHING
    # ==========================================
    # Applying a Gaussian blur before thresholding cleans up sensor noise 
    # and anti-aliasing artifacts, creating solid, contiguous masks.
    if blur_size > 0:
        # OpenCV requires the blur kernel size to be an odd number greater than 0
        k = blur_size if blur_size % 2 == 1 else blur_size + 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)

    height, width, _ = bgr.shape
    
    # ==========================================
    # EXACT COLOR MASKING
    # ==========================================
    white_bgr = np.array([239, 239, 239])
    blueish_bgr = np.array([224, 186, 163])
    sky = np.array([255, 255, 255])

    tolerance = 8

    lower_white = np.clip(white_bgr - tolerance, 0, 255).astype(np.uint8)
    upper_white = np.clip(white_bgr + tolerance, 0, 255).astype(np.uint8)

    lower_blue = np.clip(blueish_bgr - tolerance, 0, 255).astype(np.uint8)
    upper_blue = np.clip(blueish_bgr + tolerance, 0, 255).astype(np.uint8)

    lower_sky = np.clip(sky - tolerance, 0, 255).astype(np.uint8)
    upper_sky = np.clip(sky + tolerance, 0, 255).astype(np.uint8)
    
    mask_white = cv2.inRange(bgr, lower_white, upper_white)
    mask_blue = cv2.inRange(bgr, lower_blue, upper_blue)
    mask_sky = cv2.inRange(bgr, lower_sky, upper_sky)
    
    floor_mask = mask_white | mask_blue | mask_sky

    # ==========================================
    # TUNABLE MORPHOLOGICAL SMOOTHING
    # ==========================================
    if close_size > 0:
        close_kernel = np.ones((close_size, close_size), np.uint8)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_CLOSE, close_kernel)
    
    if open_size > 0:
        open_kernel = np.ones((open_size, open_size), np.uint8)
        floor_mask = cv2.morphologyEx(floor_mask, cv2.MORPH_OPEN, open_kernel)
    
    obstacle_mask = cv2.bitwise_not(floor_mask)

    return floor_mask, obstacle_mask




def test_color_lidar(img_path, num_rays=15, blur_size=5, close_size=7, open_size=3):
    # 1. Load Image
    bgr = cv2.imread(img_path)
    if bgr is None:
        return

    height, width, _ = bgr.shape
    
    # Pass the new tunable parameters into the mask creator
    floor_mask, obstacle_mask = create_wall_mask(img_path, blur_size, close_size, open_size)
    if floor_mask is None:
        return
    # ==========================================
    # 3. HIGH-DENSITY PSEUDO-LIDAR
    # ==========================================
    horizon_y = int(height * 0.4) # Don't look higher than the top 40%
    
    # Generate evenly spaced X coordinates across the screen width
    # We pad the edges slightly (e.g., 5% in) so we don't raycast the literal edge of the screen
    pad = int(width * 0.05)
    ray_x_coords = np.linspace(pad, width - pad, num_rays, dtype=int)
    
    left_free_space = 0
    right_free_space = 0
    hits = []
    
    center_index = num_rays // 2
    
    for i, x in enumerate(ray_x_coords):
        hit_y = horizon_y
        
        # Scan UP the column
        for y in range(height - 1, horizon_y, -1):
            if obstacle_mask[y, x] > 0: # We hit a non-floor pixel!
                hit_y = y
                break
                
        # Calculate how many pixels of floor we traversed before hitting the wall
        ray_distance = (height - 1) - hit_y
        hits.append((x, hit_y))
        
        # Accumulate space for steering
        if i < center_index:
            left_free_space += ray_distance
        elif i > center_index:
            right_free_space += ray_distance
        # Note: If num_rays is odd, we ignore the dead-center ray for the steering balance

    # ==========================================
    # 4. STEERING LOGIC
    # ==========================================
    total_space = left_free_space + right_free_space
    repel_turn = 0.0
    if total_space > 0:
        # Positive = Turn Right, Negative = Turn Left
        repel_turn = (right_free_space - left_free_space) / total_space


    # ==========================================
    # VISUALIZATION 
    # ==========================================
    vis = bgr.copy()
    
    # Draw horizon line
    cv2.line(vis, (0, horizon_y), (width, horizon_y), (0, 255, 255), 1)
    
    # Draw all the rays
    for x, hit_y in hits:
        # Green line for the free space (floor)
        cv2.line(vis, (x, height), (x, hit_y), (0, 255, 0), 1)
        # Red dot where it impacted an obstacle
        cv2.circle(vis, (x, hit_y), 4, (0, 0, 255), -1)
    
    # Draw a dividing line down the middle to show Left vs Right
    cv2.line(vis, (width//2, height), (width//2, horizon_y), (255, 0, 0), 1)
    
    # Print data
    direction = "CENTER"
    if repel_turn > 0.1: direction = "RIGHT"
    elif repel_turn < -0.1: direction = "LEFT"
    
    cv2.putText(vis, f"Turn: {repel_turn:.2f} ({direction})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    cv2.putText(vis, f"Rays: {num_rays}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

    # Show the pipeline
    cv2.imshow("Semantic LiDAR Array", vis)
    cv2.imshow("Floor/Sky Mask (White=Floor/Sky)", floor_mask)
    cv2.imshow("Obstacle Mask (White=Wall)", obstacle_mask)

    wall_segment_vis = draw_wall_segments(bgr, obstacle_mask)
    cv2.imshow("Wall Segments", wall_segment_vis)
    
    print(f"Image: {os.path.basename(img_path)} | Turn: {repel_turn:.2f}")

def find_wall_splits(obstacle_mask, slope_threshold=0.15, slope_window=15, smooth_window=11, jump_threshold=15, min_wall_width=30):
    """
    Traces the floor-wall boundary, detects hard depth discontinuities (jumps),
    smooths the profile piecewise, finds corners based on slope, and 
    enforces a minimum pixel width for all wall segments.
    """
    height, width = obstacle_mask.shape
    
    # 1. Trace the raw floor-wall boundary profile
    raw_y_coords = np.zeros(width, dtype=int)
    for x in range(width):
        hit_y = height - 1
        for y in range(height - 1, -1, -1):
            if obstacle_mask[y, x] > 0:
                hit_y = y
                break
        raw_y_coords[x] = hit_y
        
    # 2. Detect depth discontinuities (sudden jumps in raw data)
    jump_x_indices = [0]
    for x in range(1, width):
        if abs(raw_y_coords[x] - raw_y_coords[x-1]) > jump_threshold:
            # Prevent jumps from triggering simply because a steep wall bottom enters the frame
            if raw_y_coords[x] < height - 5 and raw_y_coords[x-1] < height - 5:
                jump_x_indices.append(x)
    jump_x_indices.append(width)
            
    # 3. Piecewise Smoothing & Slope-based Corner Detection
    smoothed_y = np.zeros(width, dtype=np.float64)
    split_points = []
    
    # Register the jump points as splits
    for j_x in jump_x_indices[1:-1]:
        split_points.append((j_x, raw_y_coords[j_x]))
        
    for i in range(len(jump_x_indices) - 1):
        start_x = jump_x_indices[i]
        end_x = jump_x_indices[i+1]
        
        if end_x - start_x == 0:
            continue
            
        segment = raw_y_coords[start_x:end_x]
        
        # --- Smoothing ---
        if len(segment) < smooth_window:
            smoothed_segment = segment.astype(np.float64)
        else:
            pad_size = smooth_window // 2
            padded = np.pad(segment, (pad_size, pad_size), mode='edge')
            window = np.ones(smooth_window) / smooth_window
            smoothed_segment = np.convolve(padded, window, mode='valid')
            
        smoothed_y[start_x:end_x] = smoothed_segment
        
        # --- Slope Analysis ---
        if len(segment) >= 2 * slope_window:
            slope_diffs = np.zeros(len(segment))
            
            for sx in range(slope_window, len(segment) - slope_window):
                y_left = smoothed_segment[sx - slope_window]
                y_center = smoothed_segment[sx]
                y_right = smoothed_segment[sx + slope_window]
                
                # FOV Clipping Exclusion
                if y_left >= height - 5 or y_center >= height - 5 or y_right >= height - 5:
                    continue
                
                # Rise over Run
                slope_left = (y_center - y_left) / slope_window
                slope_right = (y_right - y_center) / slope_window
                
                # The absolute change in trajectory
                slope_diffs[sx] = abs(slope_right - slope_left)
                
            # Find peaks (corners) using non-maximum suppression
            for sx in range(slope_window, len(segment) - slope_window):
                val = slope_diffs[sx]
                if val > slope_threshold:
                    l_bound = max(0, sx - slope_window // 2)
                    r_bound = min(len(segment), sx + slope_window // 2 + 1)
                    
                    if val == np.max(slope_diffs[l_bound:r_bound]):
                        global_x = start_x + sx
                        
                        is_duplicate = False
                        for existing_pt in split_points:
                            if abs(global_x - existing_pt[0]) < slope_window:
                                is_duplicate = True
                                break
                                
                        if not is_duplicate:
                            split_points.append((global_x, int(smoothed_segment[sx])))

    # 4. Format into boundary points for OpenCV
    boundary_points = [[x, int(smoothed_y[x])] for x in range(width)]
    boundary_pts = np.array(boundary_points, dtype=np.int32)
    
    split_points.sort(key=lambda p: p[0])
    
    # ==========================================
    # 5. NEW: ENFORCE MINIMUM WALL WIDTH
    # ==========================================
    final_splits = []
    last_x = 0  # The left edge of the screen
    
    for pt in split_points:
        x = pt[0]
        
        # 1. Does this split leave a large enough wall segment on its LEFT?
        if (x - last_x) >= min_wall_width:
            
            # 2. Does this split leave a large enough wall segment on its RIGHT?
            # (Prevents tiny slivers against the right edge of the screen)
            if (width - x) >= min_wall_width:
                final_splits.append(pt)
                last_x = x  # Update the anchor to the current valid split
                
    return final_splits, boundary_pts


def draw_wall_segments(bgr_img, obstacle_mask, slope_threshold=0.15, min_area_ratio=0.08, alpha=0.9): #tunable
    """
    Overlays each detected wall face with a distinct transparent colored polygon,
    filtering out walls that do not meet the minimum area requirement.
    """
    height, width = obstacle_mask.shape
    total_pixels = height * width
    min_pixels_required = total_pixels * min_area_ratio
    
    overlay = bgr_img.copy() 
    
    splits, boundary_pts = find_wall_splits(obstacle_mask, slope_threshold=slope_threshold)
    
    split_x = [s[0] for s in splits]
    x_boundaries = sorted(list(set([0] + split_x + [width - 1])))
    
    colors = [
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 128, 0),  # Light Blue
        (0, 128, 255)   # Orange
    ]
    
    # We use a separate counter for colors so we don't skip colors when a wall is ignored
    valid_walls_drawn = 0 
    
    for i in range(len(x_boundaries) - 1):
        start_x = x_boundaries[i]
        end_x = x_boundaries[i+1]
        
        if start_x == end_x:
            continue
            
        # ==========================================
        # AREA CHECK
        # ==========================================
        # Slice the obstacle mask to just this wall's column footprint
        wall_slice = obstacle_mask[:, start_x:end_x]
        
        # Count how many actual wall pixels are inside this boundary
        wall_area = np.count_nonzero(wall_slice)
        
        if wall_area < min_pixels_required:
            continue  # Skip highlighting this wall if it's too small
            
        # Build the polygon for this specific face
        poly_pts = []
        poly_pts.append([start_x, 0])
        poly_pts.append([end_x, 0])
        
        for x in range(end_x, start_x - 1, -1):
            hit_y = boundary_pts[x][1]
            poly_pts.append([x, hit_y])
            
        poly_pts = np.array(poly_pts, np.int32)
        color = colors[valid_walls_drawn % len(colors)]
        valid_walls_drawn += 1
        
        # Draw the filled polygon on the overlay layer
        cv2.fillPoly(overlay, [poly_pts], color)
        
    # 1. Create the globally blended image
    blended = cv2.addWeighted(overlay, alpha, bgr_img, 1 - alpha, 0)
    
    # 2. Restrict the blend to ONLY the walls using the obstacle_mask
    mask_3d = obstacle_mask[:, :, np.newaxis]
    vis_img = np.where(mask_3d > 0, blended, bgr_img)
                    
    return vis_img

if __name__ == "__main__":
    img_dir = 'data/exploration_data/images'
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    all_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
    
    if not all_images:
        print(f"No images found in {img_dir}")
        exit()
        
    print("CONTROLS:")
    print("  'w' / 's' : Increase/Decrease Rays")
    print("  'a' / 'd' : Increase/Decrease Image Blur Smoothing")
    print("  'SPACE'   : Next Image")
    print("  'q' / ESC : Quit")
    
    # Default parameters
    current_rays = 15
    current_blur = 5
    current_close = 7
    current_open = 3
    
    while True:
        random_img = random.choice(all_images)
        
        while True: # Inner loop to keep reloading the same image while tuning
            test_color_lidar(random_img, num_rays=current_rays, blur_size=current_blur, close_size=current_close, open_size=current_open)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord('q') or key == 27:
                cv2.destroyAllWindows()
                exit()
            elif key == ord('w'):
                current_rays += 2
                print(f"Rays: {current_rays}")
            elif key == ord('s'):
                current_rays = max(3, current_rays - 2)
                print(f"Rays: {current_rays}")
            elif key == ord('a'):
                current_blur += 2
                print(f"Blur Size: {current_blur}")
            elif key == ord('d'):
                current_blur = max(0, current_blur - 2)
                print(f"Blur Size: {current_blur}")
            else:
                # Any other key (like SPACE) breaks out of the tuning loop to fetch the next image
                break
            
    cv2.destroyAllWindows()