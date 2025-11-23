from typing import Optional
import cv2
import numpy as np

from task_4 import QuaternionRotate

from srccam.season_reader import SeasonReader
from srccam.load_calib import CalibReader
from srccam.calib import Calib
from srccam.camera import Camera
from srccam.point import Point3d as Point
from srccam.object3d import Object3d

ROAD_START_Y = 1.5
ROAD_ROTATION_ANGLE_DEG = 0.5
ROAD_LENGTH = 100.0
ROAD_WIDTH = 3.3
ROAD_SHIFT_X = 0.95

CUBE_SIZE = 2.2
CUBE_HEIGHT = 1.0

BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
LINE_WIDTH = 5


class WayEstimator:
    def __init__(self, camera: Camera):
        self.camera = camera

        x_left = -ROAD_WIDTH / 2.0
        x_right = ROAD_WIDTH / 2.0
        y_start = ROAD_START_Y
        y_end = ROAD_START_Y + ROAD_LENGTH

        angle_rad = np.deg2rad(ROAD_ROTATION_ANGLE_DEG)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        def rotate_point(x, y):
            new_x = x * cos_a - y * sin_a
            new_y = x * sin_a + y * cos_a
            return new_x, new_y

        # p1 - левая ближняя
        # p2 - левая дальняя
        # p3 - правая ближняя
        # p4 - правая дальняя

        p1_x, p1_y = rotate_point(x_left, y_start)
        p2_x, p2_y = rotate_point(x_left, y_end)
        p3_x, p3_y = rotate_point(x_right, y_start)
        p4_x, p4_y = rotate_point(x_right, y_end)

        self.left_3d_near = Point((p1_x + ROAD_SHIFT_X, p1_y, 0))
        self.left_3d_far = Point((p2_x + ROAD_SHIFT_X, p2_y, 0))
        self.right_3d_near = Point((p3_x + ROAD_SHIFT_X, p3_y, 0))
        self.right_3d_far = Point((p4_x + ROAD_SHIFT_X, p4_y, 0))

    def draw_way(self, img: np.array):
        left_2d_near = self.camera.project_point_3d_to_2d(self.left_3d_near)
        left_2d_far = self.camera.project_point_3d_to_2d(self.left_3d_far)
        right_2d_near = self.camera.project_point_3d_to_2d(self.right_3d_near)
        right_2d_far = self.camera.project_point_3d_to_2d(self.right_3d_far)
        
        cv2.line(img, pt1=right_2d_near, pt2=right_2d_far, color=BLACK, thickness=LINE_WIDTH)
        cv2.line(img, pt1=left_2d_near, pt2=left_2d_far, color=BLACK, thickness=LINE_WIDTH)
        return img


class Reader(SeasonReader):
    def on_init(self, _file_name: str = None) -> bool:
        par = ["K", "D", "r", "t"]
        calib_reader = CalibReader(file_name="../data/city/leftImage.yml", param=par)
        calib_dict = calib_reader.read()
        if not calib_dict:
            return False
        
        calib = Calib(calib_dict)
        self.camera = Camera(calib)
        self.way_estimator = WayEstimator(self.camera)
        
        z_pos = 0 - (CUBE_HEIGHT / 2.0)
        y_pos = ROAD_START_Y + 7.0
        x_pos = ROAD_SHIFT_X

        self.cube = Object3d(
            Point((x_pos, y_pos, z_pos)), 
            np.array([0, 0, 0]), 
            CUBE_SIZE, CUBE_HEIGHT, CUBE_SIZE, 
            GREEN, BLUE
        )

        self.cube_original_points = np.copy(self.cube.points)
        self.cube_total_rotation_y = 0.0

        return True

    def on_frame(self):
        self.cube_total_rotation_y += 0.05
        self.cube.points = np.copy(self.cube_original_points)
        QuaternionRotate.quat_rotate(self.cube, 0.0, self.cube_total_rotation_y, 0.0)
        
        self.way_estimator.draw_way(self.frame)
        self.cube.draw(self.frame, self.camera, drawVertex=True)

        return True

    def on_shot(self) -> bool: return True
    def on_gps_frame(self) -> bool: return True
    def on_imu_frame(self) -> bool: return True

if __name__ == "__main__":
    init_args = {"path_to_data_root": "../data/city/"}
    reader = Reader()
    reader.initialize(**init_args)
    reader.run()
    print("Done!")
