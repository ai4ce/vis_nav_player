from enum import Flag
import numpy as np

"""
DO NOT MODIFY THIS FILE!
This file defines the interface between the server code and the player's code
"""


class Action(Flag):
    IDLE = 0
    FORWARD = 1
    BACKWARD = 2
    LEFT = 4
    RIGHT = 8
    QUIT = 16


class Player:
    """
    This is a base class. Inherit it like the KeyboardPlayerPyGame example below
    """
    def __init__(self):
        self._targets = None  # this is to be set by the simulation server when it is being constructed

    def act(self) -> Action:
        """
        This function is to be invoked by the simulation server in each step, right after invoking see(fpv)

        return an action
        """
        raise NotImplementedError('Your player class should at least implement '
                                  'this function to tell the robot what to do next after seeing the image.')

    def see(self, fpv: np.ndarray) -> None:
        """
        This function is to be invoked by the simulation server in each step
        :param fpv: an opencv image (BGR format)
        """
        raise NotImplementedError('Your player class should at least implement '
                                  'this function to receive a new observation.')

    def get_target_images(self) -> list[np.ndarray]:
        """
        This function is to be invoked by players
        :return: a reference to the internal list of target fpv images, i.e., self._targets
        """
        return self._targets

    def set_target_images(self, images: list[np.ndarray]) -> None:
        """
        This function is to be invoked by the simulation server when loading the data
        :param images: a list of images that represents the target
        :return: None
        """
        self._targets = images
