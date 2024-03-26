from vis_nav_game import Player, Action, Phase
import pygame
import cv2
import pdb

from deadreckoning import Localizer
from image_storage import Storage_Bot
from place_recognition import (
    create_visual_dictionary,
    extract_sift_features,
    generate_feature_histograms,
    process_image_and_find_best_match,
)

class KeyboardPlayerPyGame(Player):
    def __init__(self):
        self.fpv = None  # First-person view image
        self.last_act = Action.IDLE  # Last action taken by the player
        self.screen = None  # Pygame screen
        self.keymap = None  # Mapping of keyboard keys to actions
        self.localizer = Localizer()  # Dead reckoning localizer
        super(KeyboardPlayerPyGame, self).__init__()
        self.storage = Storage_Bot()

        self.key_hold_state = {
            pygame.K_LEFT: False,
            pygame.K_RIGHT: False,
            pygame.K_UP: False,
            pygame.K_DOWN: False,
            pygame.K_LSHIFT: False,
        }
        self.is_navigation = False

    def reset(self) -> None:
        self.fpv = None
        self.last_act = Action.IDLE
        self.screen = None

        # Reset location
        self.localizer = Localizer()
        pygame.init()

        self.keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_SPACE: Action.CHECKIN,
            pygame.K_ESCAPE: Action.QUIT,
            pygame.K_p: 1,
            pygame.K_r: 1,
            pygame.K_t: 1,
            pygame.K_LSHIFT: 1,
        }
        breakpoint()
        
    def act(self):
        """
        Handle player actions based on keyboard input
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT

            if event.type == pygame.KEYDOWN:
                if event.key in self.keymap:
                    # if event.key == pygame.K_p:
                    #     self.pre_navigation_bypass()
                    # elif event.key == pygame.K_r:
                    #     self.player_position = (0, 0)
                    # elif event.key == pygame.K_t:
                    #     self.direction = 0
                    if event.key == pygame.K_LSHIFT:
                        self.key_hold_state[pygame.K_LSHIFT] = True
                    else:
                        # update action
                        self.key_hold_state[event.key] = True
                        self.last_act |= self.keymap[event.key]
                        if self.keymap[event.key] in [Action.LEFT, Action.RIGHT, Action.FORWARD, Action.BACKWARD]:
                            if not self.key_hold_state[pygame.K_LSHIFT]:
                                self.localizer.track(self.keymap[event.key], self.is_navigation)
                                pygame.event.set_blocked(pygame.KEYDOWN)
                                pygame.event.set_blocked(pygame.KEYUP)
                else:
                    if not self.is_navigation:
                        self.show_target_images()
            if event.type == pygame.KEYUP:
                if event.key in self.keymap:
                    # if (
                    #     event.key == pygame.K_p
                    #     or event.key == pygame.K_r
                    #     or event.key == pygame.K_t
                    # ):
                    #     pass
                    if event.key == pygame.K_LSHIFT:
                        self.key_hold_state[pygame.K_LSHIFT] = False
                    else:
                        self.key_hold_state[event.key] = False
                        self.last_act ^= self.keymap[event.key]
        # show the explored area and the current position
        self.localizer.map.update_minimap(self.localizer.current_x, self.localizer.current_y) 
        return self.last_act
    def post_exploration(self) -> None:
        #TODO: place reconigition
        self.is_navigation = True
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
        #TODO: Show top 10 matches
    
    def set_target_images(self, images):
        """
        Set target images
        """
        super(KeyboardPlayerPyGame, self).set_target_images(images)
        self.show_target_images()

    def pre_exploration(self):
        K = self.get_camera_intrinsic_matrix()
        print(f'K={K}')

    def pre_navigation(self) -> None:
        pass

    def see(self, fpv):
        if fpv is None or len(fpv.shape) < 3:
            return

        self.fpv = fpv

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
        if not self.is_navigation:
            self.storage.disk((self.localizer.current_x, self.localizer.current_y, self.localizer.heading), fpv)
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()

if __name__ == "__main__":
    import vis_nav_game
    # Start the game with the KeyboardPlayerPyGame player
    vis_nav_game.play(the_player=KeyboardPlayerPyGame())