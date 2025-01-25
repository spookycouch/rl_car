import gymnasium as gym
import numpy as np
import pybullet as p

from envs.gripper_one_object.sim import CarEnv as OneObjectCarEnv, get_new_distant_point

class CarEnv(OneObjectCarEnv):
    def __init__(
        self,
        real_time: bool,
    ):
        super().__init__(real_time)

    def reset(self, seed=None, options=None):
        np.random.seed(seed)
        
        if self._object is not None:
            p.removeBody(self._object)

        if np.random.random() > 0.5:
            collision_shape_id, visual_shape_id = self.__create_box()
        else:
            collision_shape_id, visual_shape_id = self.__create_cylinder()

        position, _orientation = p.getBasePositionAndOrientation(self._car)
        car_position_xy = np.array(position[:2])
        object_position_xy = get_new_distant_point(
            points=[car_position_xy],
            low=0,
            high=1,
        )
        object_position = list(object_position_xy) + [0.05]

        self._object = p.createMultiBody(
            baseMass=np.random.uniform(0.01, 0.5),
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=(0,0,0),
            baseOrientation=(0,0,0,1)
        )
        return super().reset()

    
    def __create_box(self):
        half_x = np.random.uniform(0.02, 0.07)
        half_y = np.random.uniform(0.02, 0.07)
        half_z = np.random.uniform(min(half_x, half_y), min(half_x, half_y) * 2)

        collision_shape_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z],
        )
        visual_shape_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[half_x, half_y, half_z],
            rgbaColor=[0, 1, 1, 1],
        )

        return collision_shape_id, visual_shape_id

    def __create_cylinder(self):
        radius = np.random.uniform(0.02, 0.07)
        height = np.random.uniform(0.04, radius * 2)

        collision_shape_id = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius = radius,
            height = height,
        )
        visual_shape_id = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius = radius,
            length = height,
            rgbaColor=[0, 1, 1, 1],
        )

        return collision_shape_id, visual_shape_id
