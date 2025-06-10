from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pygame

@dataclass
class GamepadConfig:
    dead_zone_left = 0.05
    dead_zone_right = 0.05
    dead_zone_trigger = -0.95

# @dataclass
# class GamepadInput:
#     left_axis: float
#     right_axis: float

@dataclass
class GamepadInput:
    button_a: bool
    button_b: bool
    button_x: bool
    button_y: bool
    left_trigger: float
    right_trigger: float
    left_stick_x_axis: float
    left_stick_y_axis: float
    right_stick_y_axis: float

class Gamepad:
    def __init__(
        self,
        config: GamepadConfig,
    ):
        pygame.init()
        if pygame.joystick.get_count() == 0:
            raise(Exception("Please connect a compatible joystick."))
        
        print(f"Using joystick: {pygame.joystick.Joystick(0).get_name()}")
        pygame.joystick.init()
        self.__joystick = pygame.joystick.Joystick(0)

        self.__config = config

    def get_input(self) -> GamepadInput:
        pygame.event.pump()

        button_a = self.__joystick.get_button(0)
        button_b = self.__joystick.get_button(1)
        button_y = self.__joystick.get_button(2)
        button_x = self.__joystick.get_button(3)
        left_trigger = (self.__joystick.get_axis(2) + 1)/2 # normalised 0-1
        right_trigger = (self.__joystick.get_axis(5) + 1)/2 # normalised 0-1
        left_stick_x_axis = self.__joystick.get_axis(0)
        left_stick_y_axis = -self.__joystick.get_axis(1)
        right_stick_y_axis = -self.__joystick.get_axis(4)

        if left_trigger < self.__config.dead_zone_trigger:
            left_trigger = 0
        if right_trigger < self.__config.dead_zone_trigger:
            right_trigger = 0
        if abs(left_stick_x_axis) < self.__config.dead_zone_left:
            left_stick_x_axis = 0
        if abs(left_stick_y_axis) < self.__config.dead_zone_left:
            left_stick_y_axis = 0
        if abs(right_stick_y_axis) < self.__config.dead_zone_right:
            right_stick_y_axis = 0
        
        return GamepadInput(
            button_a,
            button_b,
            button_x,
            button_y,
            left_trigger,
            right_trigger,
            left_stick_x_axis,
            left_stick_y_axis,
            right_stick_y_axis,
        )


    # def get_input(self) -> GamepadInput:
    #     pygame.event.pump()

    #     left_stick_y_axis = -self.__joystick.get_axis(1)
    #     right_stick_y_axis = -self.__joystick.get_axis(4)

    #     if abs(left_stick_y_axis) < self.__config.dead_zone_left:
    #         left_stick_y_axis = 0
    #     if abs(right_stick_y_axis) < self.__config.dead_zone_right:
    #         right_stick_y_axis = 0
        
    #     return GamepadInput(
    #         left_stick_y_axis,
    #         right_stick_y_axis,
    #     )

    # def get_user_input_racing_game(
    #     self,
    #     steering_scale: float = 0.3,
    # ) -> GamepadInput:
    #     pygame.event.pump()
        
    #     left_stick_x_axis = self.__joystick.get_axis(0)
    #     left_trigger = (self.__joystick.get_axis(2) + 1)/2 # normalised 0-1
    #     right_trigger = (self.__joystick.get_axis(5) + 1)/2 # normalised 0-1

    #     target_velocity_rad = 0
    #     if left_trigger > self.__config.dead_zone_trigger:
    #         target_velocity_rad -= left_trigger * (1 - steering_scale)
    #     if right_trigger > self.__config.dead_zone_trigger:
    #         target_velocity_rad += right_trigger * (1 - steering_scale)

    #     left_target_velocity_rad = target_velocity_rad
    #     right_target_velocity_rad = target_velocity_rad

    #     # steer the car with the left analog stick
    #     if abs(left_stick_x_axis) < self.__config.dead_zone_left:
    #         left_stick_x_axis = 0
        
    #     # invert controls when reversing
    #     steering_direction = 1
    #     if target_velocity_rad < 0:
    #         steering_direction = -1

    #     left_target_velocity_rad += steering_direction * left_stick_x_axis * steering_scale
    #     right_target_velocity_rad -= steering_direction * left_stick_x_axis * steering_scale
        
    #     return GamepadInput(
    #         left_target_velocity_rad,
    #         right_target_velocity_rad,
    #     )

    def convert_user_input_to_velocities(
        self,
        gamepad_input: GamepadInput,
        steering_scale: float = 0.3,
    ) -> Tuple[float, float]:
        pygame.event.pump()

        target_velocity_rad = 0
        target_velocity_rad -= gamepad_input.left_trigger * (1 - steering_scale)
        target_velocity_rad += gamepad_input.right_trigger * (1 - steering_scale)

        # invert controls when reversing
        steering_direction = 1
        if target_velocity_rad < 0:
            steering_direction = -1

        left_target_velocity_rad = target_velocity_rad
        right_target_velocity_rad = target_velocity_rad

        # steer the car with the left analog stick
        left_target_velocity_rad += steering_direction * gamepad_input.left_stick_x_axis * steering_scale
        right_target_velocity_rad -= steering_direction * gamepad_input.left_stick_x_axis * steering_scale
        
        return left_target_velocity_rad, right_target_velocity_rad
