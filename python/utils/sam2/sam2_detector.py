import cv2
import numpy as np
from sam2.build_sam import build_sam2_camera_predictor

import sys
sys.path.insert(0, "../../gui")
from gui import PointSelector
from scipy.ndimage import center_of_mass

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
            point_selector = PointSelector()
            x, y = point_selector.detect(frame)[0]

            points = np.array([[x,y]], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)
            
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

            return center_of_mass(out_mask)[:2]

        return None
