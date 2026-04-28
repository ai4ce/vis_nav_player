import os
import cv2
import numpy as np

class WallMatcher:
    def __init__(self, img_dir, graph_data):
        self.img_dir = img_dir
        self.graph_data = graph_data
        self.sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=20)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def _get_wall_mask(self, bgr_frame, blur_size=5):
        """
        Isolates walls using HSV color space and spatial cropping.
        """
        if blur_size > 0:
            k = blur_size if blur_size % 2 == 1 else blur_size + 1
            bgr_frame = cv2.GaussianBlur(bgr_frame, (k, k), 0)

        hsv_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([179, 40, 255])
        mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)

        lower_blue = np.array([100, 50, 50]) 
        upper_blue = np.array([135, 255, 255])
        mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        wall_mask = cv2.bitwise_or(mask_white, mask_blue)
        
        height, width = wall_mask.shape
        horizon_y = int(height * 0.4) 
        floor_y = int(height * 0.85)
        pad = int(width * 0.05)
        
        wall_mask[0:horizon_y, :] = 0      # Hide sky
        wall_mask[floor_y:height, :] = 0   # Hide floor
        wall_mask[:, 0:pad] = 0            # Hide left edge
        wall_mask[:, width-pad:width] = 0   # Hide right edge
        
        return wall_mask

    def find_best_wall_match(self, current_frame, expected_nodes, min_inliers=15):
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        current_mask = self._get_wall_mask(current_frame)
        
        kp_current, des_current = self.sift.detectAndCompute(current_gray, mask=current_mask)
        
        if des_current is None or len(kp_current) < 8:
            return None, None, None, None 

        best_node = None
        max_inliers = 0
        best_matches_data = (None, None, None) 

        for node_id in expected_nodes:
            node_info = self.graph_data.get(str(node_id)) or self.graph_data.get(int(node_id))
            if not node_info: continue
                
            node_img_path = os.path.join(self.img_dir, node_info['image'])
            node_frame = cv2.imread(node_img_path)
            if node_frame is None: continue
                
            node_gray = cv2.cvtColor(node_frame, cv2.COLOR_BGR2GRAY)
            node_mask = self._get_wall_mask(node_frame)
            kp_node, des_node = self.sift.detectAndCompute(node_gray, mask=node_mask)
            
            if des_node is None or len(kp_node) < 8: continue

            matches = self.flann.knnMatch(des_current, des_node, k=2)
            
            good_matches = [m for m, n in matches if m.distance < 0.8 * n.distance]

            if len(good_matches) > 8:
                pts1 = np.float32([kp_current[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                pts2 = np.float32([kp_node[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 5.0)
                
                if mask is not None:
                    inliers = np.sum(mask)
                    if inliers > max_inliers and inliers >= min_inliers:
                        max_inliers = inliers
                        best_node = node_id
                        best_matches_data = (good_matches, kp_current, kp_node)

        return (best_node, *best_matches_data)