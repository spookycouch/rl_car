
#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from datetime import timedelta

from .camera import CameraFrame

FPS = 20
MONO_RESOLUTION = dai.MonoCameraProperties.SensorResolution.THE_400_P
XOUT_DEPTH_KEY = "XOUT_DEPTH_KEY"
XOUT_RGB_KEY = "XOUT_RGB_KEY"
MAXIMUM_Z_MILLIMETRES = 2500
RGB_SIZE = (712, 400)
# PREVIEW_SIZE = (1920, 1080)
# PREVIEW_SIZE = (960, 540)
PREVIEW_SIZE = (640, 360)

class OakDCamera:
    def __init__(
        self,
        warm_up_frames: int = 0
    ):
        # create pipeline
        pipeline = dai.Pipeline()
        self.__device = dai.Device()

        camRgb = pipeline.create(dai.node.Camera)
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        sync = pipeline.create(dai.node.Sync)
        xoutGrp = pipeline.create(dai.node.XLinkOut)

        xoutGrp.setStreamName("xout")

        # setup RGB camera
        rgbCamSocket = dai.CameraBoardSocket.CAM_A
        camRgb.setBoardSocket(rgbCamSocket)
        camRgb.setSize(*RGB_SIZE)
        camRgb.setFps(FPS)
        camRgb.setPreviewSize(PREVIEW_SIZE)
        # For now, RGB needs fixed focus to properly align with depth.
        # This value was used during calibration
        try:
            calibData = self.__device.readCalibration2()
            lensPosition = calibData.getLensPosition(rgbCamSocket)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
            camRgb.initialControl.setAutoExposureLimit(10000)
        except:
            raise

        # setup left and right mono cameras
        left.setResolution(MONO_RESOLUTION)
        left.setCamera("left")
        left.setFps(FPS)
        right.setResolution(MONO_RESOLUTION)
        right.setCamera("right")
        right.setFps(FPS)

        # setup stereo depth
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
        stereo.setDepthAlign(rgbCamSocket)
        stereo.setOutputSize(*PREVIEW_SIZE)

        # setup sync
        sync.setSyncThreshold(timedelta(milliseconds=50))

        # linking
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        stereo.depth.link(sync.inputs[XOUT_DEPTH_KEY])
        camRgb.preview.link(sync.inputs[XOUT_RGB_KEY]) # camRgb.video is too large
        sync.out.link(xoutGrp.input)

        self.__intrinsics_matrix = np.array(calibData.getCameraIntrinsics(rgbCamSocket, resizeHeight=PREVIEW_SIZE[1], resizeWidth=PREVIEW_SIZE[0]))

        self.__device.startPipeline(pipeline)
        self.__queue = self.__device.getOutputQueue("xout", 1, False)
        
        # warm up the camera
        for _ in range(warm_up_frames):
            self.get_frame()

    def __del__(self):
        self.__device.close()

    def get_frame(self) -> CameraFrame:
        msgGrp = self.__queue.get()
        frameRgb = None
        frameDepth = None
        for name, msg in msgGrp:
            frame: np.ndarray = msg.getCvFrame()
            if name == XOUT_DEPTH_KEY:
                frameDepth = frame.copy()
            elif name == XOUT_RGB_KEY:
                frameRgb = frame.copy()

        return CameraFrame(
            frameRgb,
            frameDepth,
            self.__intrinsics_matrix,
            np.empty(0)
        )

class OakDRGB:
    def __init__(
        self,
        warm_up_frames: int = 0,
        fps: int = FPS
    ):
        pipeline = dai.Pipeline()
        self.__device = dai.Device()
        camRgb = pipeline.create(dai.node.Camera)

        rgbCamSocket = dai.CameraBoardSocket.CAM_A
        camRgb.setBoardSocket(rgbCamSocket)
        camRgb.setFps(fps)
        camRgb.setPreviewSize(PREVIEW_SIZE)

        try:
            calibData = self.__device.readCalibration2()
            lensPosition = calibData.getLensPosition(rgbCamSocket)
            if lensPosition:
                camRgb.initialControl.setManualFocus(lensPosition)
            camRgb.initialControl.setAutoExposureLimit(10000)
        except:
            raise

        xoutGrp = pipeline.create(dai.node.XLinkOut)
        xoutGrp.setStreamName("xout")
        camRgb.preview.link(xoutGrp.input)

        self.__intrinsics_matrix = np.array(calibData.getCameraIntrinsics(rgbCamSocket, resizeHeight=PREVIEW_SIZE[1], resizeWidth=PREVIEW_SIZE[0]))

        self.__device.startPipeline(pipeline)
        self.__queue = self.__device.getOutputQueue("xout", 1, False)
        
        # warm up the camera
        for _ in range(warm_up_frames):
            self.get_frame()

    def __del__(self):
        self.__device.close()

    def get_frame(self) -> CameraFrame:
        msg = self.__queue.get()
        frameRgb = msg.getCvFrame().copy()

        return CameraFrame(
            frameRgb,
            None,
            self.__intrinsics_matrix,
            np.empty(0)
        )   

if __name__ == "__main__":
    camera = OakDCamera()
    while 1:
        frame = camera.get_frame()

        depth = (frame.depth * (255 / MAXIMUM_Z_MILLIMETRES)).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame.frame, 0.4, depth, 0.6, 0)

        cv2.imshow("rgb", frame.frame)
        cv2.imshow("blended", blended)
        cv2.waitKey(1)
