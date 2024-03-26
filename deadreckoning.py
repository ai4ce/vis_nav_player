from vis_nav_game import Player, Action, Phase
import matplotlib.pyplot as plt

class Map():
    def __init__(self):
        self.x = []
        self.y = []
        self.MAP_DELAY = 20
        self.delay_counter = 0
        self.target = (None, None)
    def update_minimap(self, current_x : int, current_y : int) -> None:
        if self.delay_counter < self.MAP_DELAY:
            self.delay_counter += 1
            return
        self.delay_counter = 0
        plt.scatter(self.x, self.y, color='black')
        plt.scatter(current_x, current_y, color='blue')
        if self.target[0] is not None and self.target[1] is not None:
            plt.scatter(self.target[0], self.target[1], color='green')
        plt.axis('off')
        plt.show()

class Localizer():
    def __init__(self):
        self.current_x = 0
        self.current_y = 0
        self.heading = 0
        self.map = Map()
    def track(self, action) -> None:
        if action is Action.FORWARD:
            self._forward()
        elif action is Action.BACKWARD:
            self._backward()
        elif action is Action.LEFT:
            self.heading += 90
            if self.heading >= 360:
                self.heading = self.heading % 360
        elif action is Action.RIGHT:
            self.heading -= 90
            if self.heading < 0:
                self.heading = 360 + self.heading
    def _forward(self, navigation=False) -> None:
        if self.heading == 0:
            self.current_x += 1
        elif self.heading == 90:
            self.current_y += 1
        elif self.heading == 180:
            self.current_x -= 1
        elif self.heading == 270:
            self.current_y -= 1
        if not navigation:
            self.map.x.append(self.current_x)
            self.map.y.append(self.current_y)
    def _backward(self, navigation=False) -> None:
        if self.heading%360 == 0:
            self.current_x -= 1
        elif self.heading%360 == 90:
            self.current_y -= 1
        elif self.heading%360 == 180:
            self.current_x += 1
        elif self.heading%360 == 270:
            self.current_y += 1
        if not navigation:
            self.map.x.append(self.current_x)
            self.map.y.append(self.current_y)