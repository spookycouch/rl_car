from argparse import ArgumentParser
import cv2
import numpy as np
from sympy import Point
from sam2.build_sam import build_sam2_camera_predictor

import sys
sys.path.insert(0, "../../gui")
sys.path.insert(0, "../../utils")
from gui import PointSelector
from scipy.ndimage import center_of_mass
from oak_d_camera import OakDCamera

class Sam2Detector:
    def __init__(
        self,
        model_cfg,
        sam2_checkpoint,
    ):
        self.__predictor = build_sam2_camera_predictor(model_cfg, sam2_checkpoint)
        self.__is_initialised = False

    def detect(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self.__is_initialised:
            self.__predictor.load_first_frame(frame)
            x_pos, y_pos = PointSelector().detect(frame)[0] # for the target object
            x_neg, y_neg = PointSelector().detect(frame)[0] # for the gripper

            points = np.array([[x_pos,y_pos], [x_neg, y_neg]], dtype=np.float32)
            labels = np.array([1, 0], dtype=np.int32)
            
            _, out_obj_ids, out_mask_logits = self.__predictor.add_new_prompt(
                frame_idx=0,obj_id=1, points=points, labels=labels
            )
            self.__is_initialised = True
        else:
            out_obj_ids, out_mask_logits = self.__predictor.track(frame)
        
        for i, out_obj_id in enumerate(out_obj_ids):
            if out_obj_id != 1:
                continue

            out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                np.uint8
            )
            
            blended = cv2.addWeighted(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                1,
                cv2.cvtColor(out_mask * 255, cv2.COLOR_GRAY2BGR),
                0.5,
                0
            )
            cv2.imshow("blended", blended)
            cv2.waitKey(1)

            com = center_of_mass(out_mask)[:2]
            if not np.any(np.isnan(com)):
                return com

        return None




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("model_cfg")
    parser.add_argument("sam2_checkpoint")
    args = parser.parse_args()
    detector = Sam2Detector(args.model_cfg, args.sam2_checkpoint)

    camera = OakDCamera()
    while True:
        frame = camera.get_frame().frame

        point = detector.detect(frame)
        if point is not None:
            y, x = [int(index) for index in point]
            cv2.circle(frame, (x,y), 5, (0,255,0), -1)

        cv2.imshow("frame", frame)
        cv2.waitKey(1)
