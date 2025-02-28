# import necessary libraries and modules
from vis_nav_game import Player, Action, Phase
import pygame
import cv2

import numpy as np
import os
import pickle
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
from tqdm import tqdm
from natsort import natsorted

# Current Changes:
import networkx as nx
from scipy.spatial import distance
import joblib


import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Define a class for a player controlled by keyboard input using pygame
class KeyboardPlayerPyGame(Player):
    def __init__(self):
        # Initialize class variables
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        super(KeyboardPlayerPyGame, self).__init__()
        # self.tree = None
        # Variables for reading exploration data
        self.save_dir = "data/images_subsample/"
        if not os.path.exists(self.save_dir):
            print(f"Directory {self.save_dir} does not exist, please download exploration data.")

        # Initialize SIFT detector
        # SIFT stands for Scale-Invariant Feature Transform
        self.sift = cv2.SIFT_create()
        # Load pre-trained sift features and codebook
        self.sift_descriptors, self.codebook, self.database = None, None, None
        if os.path.exists("sift_descriptors.npy"):
            self.sift_descriptors = np.load("sift_descriptors.npy")
        if os.path.exists("codebook.pkl"):
            self.codebook = pickle.load(open("codebook.pkl", "rb"))
        if os.path.exists("VLAD_database.pkl"):
            self.database = pickle.load(open("VLAD_database.pkl", "rb"))
        
        # G - graph
        self.G = None
        # Load knn graph
        if os.path.exists("knn_graph.pkl"):
            print("Loading precomputed graph...")
            with open("knn_graph.pkl", "rb") as f:
                self.G = pickle.load(f)
        # Initialize goal location
        self.goal = None
        

    def reset(self):
        # Reset the player state
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Initialize pygame
        pygame.init()

        # Define key mappings for actions
        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT
        }

    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            #  Quit if user closes window or presses escape
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            # Check if a key has been pressed
            if event.type == pygame.KEYDOWN:
                # Check if the pressed key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise OR the current action with the new one
                    # This allows for multiple actions to be combined into a single action
                    self.last_act |= self.keymap[event.key]
                else:
                    # If a key is pressed that is not mapped to an action, then display target images
                    self.show_target_images()
            # Check if a key has been released
            if event.type == pygame.KEYUP:
                # Check if the released key is in the keymap
                if event.key in self.keymap:
                    # If yes, bitwise XOR the current action with the new one
                    # This allows for updating the accumulated actions to reflect the current sate of the keyboard inputs accurately
                    self.last_act ^= self.keymap[event.key]
        return self.last_act

    def show_target_images(self):
        """
        Display front, right, back, and left views of target location in 2x2 grid manner
        """
        targets = self.get_target_images()

        # Return if the target is not set yet
        if targets is None or len(targets) <= 0:
            return

        # Create a 2x2 grid of the 4 views of target location
        hor1 = cv2.hconcat(targets[:2])
        hor2 = cv2.hconcat(targets[2:])
        concat_img = cv2.vconcat([hor1, hor2])

        w, h = concat_img.shape[:2]
        
        color = (0, 0, 0)

        concat_img = cv2.line(concat_img, (int(h/2), 0), (int(h/2), w), color, 2)
        concat_img = cv2.line(concat_img, (0, int(w/2)), (h, int(w/2)), color, 2)

        w_offset = 25
        h_offset = 10
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = cv2.LINE_AA
        size = 0.75
        stroke = 1

        cv2.putText(concat_img, 'Front View', (h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Right View', (int(h/2) + h_offset, w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Back View', (h_offset, int(w/2) + w_offset), font, size, color, stroke, line)
        cv2.putText(concat_img, 'Left View', (int(h/2) + h_offset, int(w/2) + w_offset), font, size, color, stroke, line)

        cv2.imshow(f'KeyboardPlayer:target_images', concat_img)
        cv2.waitKey(1)

    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def display_img_from_id(self, id, window_name):
        """
        Display image from database based on its ID using OpenCV
        """
        path = self.save_dir + str(id) + ".jpg"
        if os.path.exists(path):
            img = cv2.imread(path)
            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        else:
            print(f"Image with ID {id} does not exist")

    def compute_sift_features(self):
        """
        Compute SIFT features for images in the data directory
        """
        files = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
        sift_descriptors = list()
        for img in tqdm(files, desc="Processing images"):
            img = cv2.imread(os.path.join(self.save_dir, img))
            # Pass the image to sift detector and get keypoints + descriptions
            # We only need the descriptors
            # These descriptors represent local features extracted from the image.
            _, des = self.sift.detectAndCompute(img, None)
            # Root-SIFT- Power normalization on SIFT decriptors to make it more robust to noise and equalize weak and strong features
            des = np.sqrt(des / np.linalg.norm(des, axis=1, keepdims=True))
            # Extend the sift_descriptors list with descriptors of the current image
            sift_descriptors.extend(des)
        return np.asarray(sift_descriptors)
    
    def get_VLAD(self, img):
        """
        Compute VLAD (Vector of Locally Aggregated Descriptors) descriptor for a given image
        """
        # We use a SIFT in combination with VLAD as a feature extractor as it offers several benefits
        # 1. SIFT features are invariant to scale and rotation changes in the image
        # 2. SIFT features are designed to capture local patterns which makes them more robust against noise
        # 3. VLAD aggregates local SIFT descriptors into a single compact representation for each image
        # 4. VLAD descriptors typically require less memory storage compared to storing the original set of SIFT
        # descriptors for each image. It is more practical for storing and retrieving large image databases efficicently.

        # Pass the image to sift detector and get keypoints + descriptions
        # Again we only need the descriptors
        _, des = self.sift.detectAndCompute(img, None)
        # Apply root normalization to make the descriptors more robust
        des = np.sqrt(des / np.linalg.norm(des, axis=1, keepdims=True))
        # We then predict the cluster labels using the pre-trained codebook
        # Each descriptor is assigned to a cluster, and the predicted cluster label is returned
        pred_labels = self.codebook.predict(des)
        # Get number of clusters that each descriptor belongs to
        centroids = self.codebook.cluster_centers_
        # Get the number of clusters from the codebook
        k = self.codebook.n_clusters
        VLAD_feature = np.zeros([k, des.shape[1]])

        # Loop over the clusters
        for i in range(k):
            # Check if the current descriptors are assigned to the [i]th cluster
            if np.sum(pred_labels == i) > 0:
                # Then, sum the residual vectors (difference between descriptors and cluster centroids)
                # for all the descriptors assigned to that clusters
                # axis=0 indicates summing along the rows (each row represents a descriptor)
                # This way we compute the VLAD vector for the current cluster i
                # This operation captures not only the presence of features but also their spatial distribution within the image
                VLAD_feature[i] = np.sum(des[pred_labels==i, :] - centroids[i], axis=0)
        
        VLAD_feature = VLAD_feature.flatten()
        # Apply power normalization to the VLAD feature vector
        # It takes the element-wise square root of the absolute values of the VLAD feature vector and then multiplies 
        # it by the element-wise sign of the VLAD feature vector
        # This makes the resulting descriptor robust to noise and variations in illumination which helps improve the 
        # robustness of VPR systems
        VLAD_feature = np.sign(VLAD_feature)*np.sqrt(np.abs(VLAD_feature))
        # Finally, the VLAD feature vector is normalized by dividing it by its L2 norm, ensuring that it has unit length
        VLAD_feature = VLAD_feature/np.linalg.norm(VLAD_feature)

        return VLAD_feature

    def get_neighbor(self, img):
        """
        Find the nearest neighbor in the database based on VLAD descriptor
        """
        # Get the VLAD feature of the image
        q_VLAD = self.get_VLAD(img).reshape(1, -1)
        # This function returns the index of the closest match of the provided VLAD feature from the database the tree was created
        # The '1' indicates the we want 1 nearest neighbor
        _, index = self.tree.query(q_VLAD, 1)
        return index[0][0]

    def pre_nav_compute(self):
        """
        Create a KNN graph for graph based path finding
        Build BallTree for nearest neighbor search and find the goal ID
        """
        # Compute sift features for images in the database
        if self.sift_descriptors is None:
            print("Computing SIFT features...")
            self.sift_descriptors = self.compute_sift_features()
            np.save("sift_descriptors.npy", self.sift_descriptors)
        else:
            print("Loaded SIFT features from sift_descriptors.npy")

        # KMeans clustering algorithm is used to create a visual vocabulary, also known as a codebook,
        # from the computed SIFT descriptors.
        # n_clusters = 128: Specifies the number of clusters (visual words) to be created in the codebook. In this case, 128 clusters are being used.
        # init='k-means++': This specifies the method for initializing centroids. 'k-means++' is a smart initialization technique that selects initial 
        # cluster centers in a way that speeds up convergence.
        # n_init=10: Specifies the number of times the KMeans algorithm will be run with different initial centroid seeds. The final result will be 
        # the best output of n_init consecutive runs in terms of inertia (sum of squared distances).
        # The fit() method of KMeans is then called with sift_descriptors as input data. 
        # This fits the KMeans model to the SIFT descriptors, clustering them into n_clusters clusters based on their feature vectors

        # TODO: try tuning the function parameters for better performance
        if self.codebook is None:
            print("Computing codebook...")
            self.codebook = KMeans(n_clusters=128, init='k-means++', n_init=5, verbose=1).fit(self.sift_descriptors)
            pickle.dump(self.codebook, open("codebook.pkl", "wb"))
        else:
            print("Loaded codebook from codebook.pkl")
        
        # get VLAD embedding for each image in the exploration phase
        if self.database is None:
            self.database = []
            print("Computing VLAD embeddings...")
            exploration_observation = natsorted([x for x in os.listdir(self.save_dir) if x.endswith('.jpg')])
            for img in tqdm(exploration_observation, desc="Processing images"):
                img = cv2.imread(os.path.join(self.save_dir, img))
                VLAD = self.get_VLAD(img)
                self.database.append(VLAD)
            with open("VLAD_database.pkl", "wb") as f:
                pickle.dump(self.database, f)
            print("VLAD database is saved")
        if self.G is None:
            # Initialize graph
            self.G = nx.Graph()
            # Adding nodes to the graph based on the VLADs
            for i in range(len(self.database)):
                self.G.add_node(i)
            
            # Computing distances 
            for i in tqdm(range(len(self.database)), desc="Computing distances..."):
                node_dist = []
                for j in range(len(self.database)):
                    if i!=j:
                        # Simply Take the Euclidean distance between the VLADS
                        # Short distance ~ More similarity
                        # TODO: Implement a better solution to find similarity between images and connect the edges based on that
                        d = distance.euclidean(self.database[i], self.database[j]) 
                        node_dist.append((j, d))

                # Select the K-nearest neighbours 
                # TODO: Implement a more robust method to find neighbours
                node_dist = sorted(node_dist, key=lambda x:x[1])[:10]

                # Adding weighted edges to the graph
                for neighbour, dist in node_dist:
                    self.G.add_edge(i, neighbour, weight=dist)
            
            # Save the graph to avoid recomputation during each run
            with open("knn_graph.pkl", "wb") as f:
                pickle.dump(self.G, f)
            print("Graph saved successfully!")
            
        # TODO: try tuning the leaf size for better performance
        print("Building BallTree...")
        tree = BallTree(self.database, leaf_size=64)
        self.tree = tree 


    def pre_navigation(self):
        """
        Computations to perform before entering navigation and after exiting exploration
        """
        super(KeyboardPlayerPyGame, self).pre_navigation()
        self.pre_nav_compute()
        
    def display_next_best_view(self):
        """
        Display the next best view based on the current first-person view
        """

        # TODO: could you write this function in a smarter way to not simply display the image that closely 
        # matches the current FPV but the image that can efficiently help you reach the target? 
        # Or you might not need this function at all...

        # Get the neighbor of current FPV
        # In other words, get the image from the database that closely matches current FPV
        current_index = self.get_neighbor(self.fpv)

        try:
            # Find the shortest path from the current position to the goal
            path = nx.astar_path(self.G, current_index, self.goal, weight="weight")
        except nx.NetworkXNoPath:
            print("No valid path found to the goal! Improve your algorithm!")
            return
        
        if len(path)>1:
            # Obtain adjacent nodes that are potential paths
            possible_views = list(self.G.neighbors(current_index))

            # Score thhe best view (Prioritize the ones that lead to the goal)
            next_best_view = min(possible_views, key=lambda view: nx.shortest_path_length(self.G, view, self.goal, weight="weight", method="dijkstra"))

            # Display the next best view
            self.display_img_from_id(next_best_view, "Next Best view")
            print(f'Next View ID: {current_index} || Goal ID: {self.goal}')


    def see(self, fpv):
        """
        Set the first-person view input
        """

        # Return if fpv is not available
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

        # If the pygame screen has not been initialized, initialize it with the size of the fpv image
        # This allows subsequent rendering of the first-person view image onto the pygame screen
        if self.screen is None:
            h, w, _ = fpv.shape
            self.screen = pygame.display.set_mode((w, h))

        def convert_opencv_img_to_pygame(opencv_image):
            """
            Convert OpenCV images for Pygame.

            see https://blanktar.jp/blog/2016/01/pygame-draw-opencv-image.html
            """
            opencv_image = opencv_image[:, :, ::-1]  # BGR->RGB
            shape = opencv_image.shape[1::-1]  # (height,width,Number of colors) -> (width, height)
            pygame_image = pygame.image.frombuffer(opencv_image.tobytes(), shape, 'RGB')

            return pygame_image

        pygame.display.set_caption("KeyboardPlayer:fpv")

        # If game has started
        if self._state:
            # If in exploration stage
            if self._state[1] == Phase.EXPLORATION:
                # TODO: could you employ any technique to strategically perform exploration instead of random exploration
                # to improve performance (reach target location faster)?
                
                # Nothing to do here since exploration data has been provided
                pass
            
            # If in navigation stage
            elif self._state[1] == Phase.NAVIGATION:
                # TODO: could you do something else, something smarter than simply getting the image closest to the current FPV?
                
                if self.goal is None:
                    # Get the neighbor nearest to the front view of the target image and set it as goal
                    targets = self.get_target_images()
                    index = self.get_neighbor(targets[0])
                    self.goal = index
                    print(f'Goal ID: {self.goal}')
                                
                # Key the state of the keys
                keys = pygame.key.get_pressed()
                # If 'q' key is pressed, then display the next best view based on the current FPV
                if keys[pygame.K_q]:
                    self.display_next_best_view()

        # Display the first-person view image on the pygame screen
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())
