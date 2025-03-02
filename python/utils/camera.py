
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import cv2

@dataclass
class CameraFrame:
    frame: np.ndarray
    depth: Optional[np.ndarray]
    intrinsic_matrix: np.ndarray
    distortion_coeffs: np.ndarray

class OpenCVCameraWrapper:
    def __init__(
        self,
        video_source: Union[int, str],
        intrinsic_matrix: np.ndarray,
        distortion_coeffs: np.ndarray,
    ):
        self.__cap = cv2.VideoCapture(video_source)
        self.__cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.__cap.set(cv2.CAP_PROP_EXPOSURE, 100)

        self.__intrinsic_matrix = intrinsic_matrix
        self.__distortion_coeffs = distortion_coeffs
    
    def __del__(self):
        self.__cap.release()
    
    def get_frame(self) -> CameraFrame:
        _, frame = self.__cap.read()
        return CameraFrame(
            frame,
            self.__intrinsic_matrix,
            self.__distortion_coeffs
        )