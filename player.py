from interface import Player, Action
import pygame


class KeyboardPlayerPyGame(Player):
    def __init__(self):
        super(KeyboardPlayerPyGame, self).__init__()
        self.fpv = None
        self.last_act = Action.IDLE
        pygame.init()
        self.screen = None

    def act(self):
        keymap = {
            pygame.K_LEFT: Action.LEFT,
            pygame.K_RIGHT: Action.RIGHT,
            pygame.K_UP: Action.FORWARD,
            pygame.K_DOWN: Action.BACKWARD,
            pygame.K_ESCAPE: Action.QUIT
        }

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.last_act = Action.QUIT
                return Action.QUIT
            if event.type == pygame.KEYDOWN:
                if event.key in keymap:
                    self.last_act = keymap[event.key]
                return self.last_act
            if event.type == pygame.KEYUP:
                self.last_act = Action.IDLE
                return Action.IDLE
        return self.last_act

    def see(self, fpv):
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
        rgb = convert_opencv_img_to_pygame(fpv)
        self.screen.blit(rgb, (0, 0))
        pygame.display.update()


if __name__ == "__main__":
    import vis_nav
    vis_nav.play(the_player=KeyboardPlayerPyGame())
