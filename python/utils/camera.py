
from dataclasses import dataclass
from typing import Union

import numpy as np
import cv2

@dataclass
class CameraFrame:
    frame: np.ndarray
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
        self.__intrinsic_matrix = intrinsic_matrix
        self.__distortion_coeffs = distortion_coeffs
    
    def get_frame(self) -> CameraFrame:
        _, frame = self.__cap.read()
        return CameraFrame(
            frame,
            self.__intrinsic_matrix,
            self.__distortion_coeffs
        )