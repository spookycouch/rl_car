import gymnasium as gym
import numpy as np
import pybullet as p

from envs.gripper_one_object.sim import CarEnv as OneObjectCarEnv, get_homogeneous_transformation_from_pose, get_new_distant_point, create_sphere

class CarEnv(OneObjectCarEnv):
    def __init__(
        self,
        real_time: bool,
    ):
        self.__closest_point_obs_error = np.zeros(2)
        super().__init__(real_time)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # self.__contact_vis = create_sphere()

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

        # for joint_index in range(p.getNumJoints(self._car)):
        #     print(joint_index, p.getJointInfo(
        #         self._car,
        #         joint_index
        #     )[12])

        self._object = p.createMultiBody(
            baseMass=np.random.uniform(0.01, 0.5),
            baseCollisionShapeIndex=collision_shape_id,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=(0,0,0),
            baseOrientation=(0,0,0,1)
        )

        self.__closest_point_obs_error = np.random.uniform(-0.03, 0.03, 2)


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

    def _get_observation(
        self,
        transforms: OneObjectCarEnv.Transforms
    ):
        closest_points = p.getClosestPoints(
            self._car,
            self._object,
            2.0, # 2 metres
            linkIndexA=5,
        )

        delta_closest_point = np.zeros(2)
        if len(closest_points) > 0:
            closest_point = closest_points[0][6]
            world_P_closest = list(closest_point) + [1]

            position, orientation = p.getBasePositionAndOrientation(self._car)
            world_T_car = get_homogeneous_transformation_from_pose(position, orientation)
            car_P_closest = np.linalg.inv(world_T_car) @ world_P_closest

            # # try to visualise it
            # try:
            #     p.resetBasePositionAndOrientation(
            #         self.__contact_vis,
            #         posObj=closest_point,
            #         ornObj=(0,0,0,1)
            #     )
            # except:
            #     pass

            delta_closest_point = np.array(car_P_closest[:2])
            delta_closest_point += self.__closest_point_obs_error
        
        obs = super()._get_observation(transforms)
        obs = np.concatenate([obs, delta_closest_point])
        return obs