#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Safe, staged red-cylinder grasp with eye-out and eye-in RealSense cameras."""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from bsp.camera_bsp.cylinder_detect import RedCylinderDetector
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.grasp_bsp.grasp_claw import grasp_claw
from bsp.robot_bsp.UR_Robot import UR_Robot


BASE_DIR = Path(__file__).resolve().parent
ROBOT_IP = "192.168.1.35"
HO_SERIAL = "215122257404"
HI_SERIAL = "215222074676"

HO_CALIB_PATH = BASE_DIR / "camera_pose.txt"          # cam -> base, metres
HO_DEPTH_SCALE_PATH = BASE_DIR / "camera_depth_scale.txt"
HI_CAM2END_PATH = BASE_DIR / "cam2end_20260906.txt"   # cam -> end, millimetres
HI_CAMERA_INI_PATH = BASE_DIR / "camera_20260906.ini"

HO_K = np.array([
    [386.471, 0.0, 321.617],
    [0.0, 386.034, 237.200],
    [0.0, 0.0, 1.0],
], dtype=np.float64)

TOOL_ORIENTATION = np.array([3.141, 0.0, 0.0], dtype=np.float64)
WORKSPACE_LIMITS = np.array([
    [-0.50, 0.05],
    [-0.80, -0.45],
    [0.02, 0.60],
], dtype=np.float64)

OBSERVE_CLEARANCE_M = 0.10
PREGRASP_CLEARANCE_M = 0.05
LIFT_DISTANCE_M = 0.10
MIN_TCP_Z_M = 0.03
MAX_XY_STEP_M = 0.40
MOVE_SPEED = 0.05
MOVE_ACCELERATION = 0.05
STABLE_WINDOW = 8
MAX_CENTER_SPREAD_PX = 5.0
MAX_DEPTH_SPREAD_MM = 8.0
MAX_BASE_SPREAD_M = 0.008

GRIPPER_PORT = "COM10"
GRIPPER_OPEN_POSITION = 6000
GRIPPER_CLOSED_POSITION = 11000
GRIPPER_SPEED = 80
GRIPPER_FORCE = 60

# Must be measured by teaching before descent is enabled.  This is the vector
# from UR TCP to the actual centre of the fingers, expressed in base axes for
# the fixed TOOL_ORIENTATION used above.
TCP_TO_GRASP_CENTER_M = None


class Stage:
    WAIT_HO = "WAIT_HO"
    HO_LOCKED = "HO_LOCKED"
    AT_OBSERVE = "AT_OBSERVE"
    HI_LOCKED = "HI_LOCKED"
    AT_PREGRASP = "AT_PREGRASP"
    AT_GRASP = "AT_GRASP"
    GRIPPED = "GRIPPED"
    LIFTED = "LIFTED"


def parse_args():
    parser = argparse.ArgumentParser(description="Staged dual-camera red-cylinder grasp")
    parser.add_argument("--ho-z-offset", type=float, default=0.0,
                        help="Optional eye-out Z correction in metres (default: 0)")
    parser.add_argument("--gripper-port", default=GRIPPER_PORT)
    parser.add_argument("--camera-only", action="store_true",
                        help="Run detection without connecting robot or gripper")
    return parser.parse_args()


def in_workspace(position):
    p = np.asarray(position[:3], dtype=np.float64)
    return bool(np.all(p >= WORKSPACE_LIMITS[:, 0]) and np.all(p <= WORKSPACE_LIMITS[:, 1]))


def median_stable(samples):
    """Return median [x,y,z] only when a complete sample window is stable."""
    if len(samples) < STABLE_WINDOW:
        return None
    values = np.asarray(samples, dtype=np.float64)
    median = np.median(values, axis=0)
    spread = np.max(np.linalg.norm(values - median, axis=1))
    if spread > MAX_BASE_SPREAD_M:
        return None
    return median


def stable_detection(history):
    if len(history) < STABLE_WINDOW:
        return False
    values = np.asarray(history, dtype=np.float64)
    centers = values[:, :2]
    depths = values[:, 2]
    center_med = np.median(centers, axis=0)
    return (np.max(np.linalg.norm(centers - center_med, axis=1)) <= MAX_CENTER_SPREAD_PX
            and np.ptp(depths) <= MAX_DEPTH_SPREAD_MM)


def make_pose(position):
    return np.concatenate([np.asarray(position, dtype=np.float64), TOOL_ORIENTATION]).tolist()


def ho_pixel_to_base(u, v, raw_depth_value, depth_scale, transform):
    """Eye-out projection without the legacy module's coordinate clamping."""
    z = float(raw_depth_value) * depth_scale
    if not math.isfinite(z) or z <= 0.0:
        return None
    x = (u - HO_K[0, 2]) * z / HO_K[0, 0]
    y = (v - HO_K[1, 2]) * z / HO_K[1, 1]
    point = transform @ np.array([x, y, z, 1.0], dtype=np.float64)
    result = point[:3]
    return result if np.all(np.isfinite(result)) else None


def safe_move(robot, position, source):
    target = np.asarray(position, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("invalid target position")
    if target[2] < MIN_TCP_Z_M:
        raise ValueError("target TCP Z %.3f m is below safety limit %.3f m" %
                         (target[2], MIN_TCP_Z_M))
    if not in_workspace(target):
        raise ValueError("target is outside workspace: %s" % np.round(target, 4))
    current = np.asarray(robot.get_actual_tcp_pose(), dtype=np.float64)
    xy_step = float(np.linalg.norm(target[:2] - current[:2]))
    if xy_step > MAX_XY_STEP_M:
        raise ValueError("XY step %.3f m exceeds %.3f m limit" % (xy_step, MAX_XY_STEP_M))
    pose = make_pose(target)
    print("[MOVE:%s] current=%s target=%s xy_step=%.3fm" %
          (source, np.round(current, 4), np.round(pose, 4), xy_step))
    robot.moveL(pose, speed=MOVE_SPEED, acceleration=MOVE_ACCELERATION)


def calculate_tcp_grasp_position(top_center_base, cylinder_height_m):
    if TCP_TO_GRASP_CENTER_M is None:
        raise RuntimeError(
            "TCP_TO_GRASP_CENTER_M has not been measured; descent is disabled")
    top = np.asarray(top_center_base, dtype=np.float64)
    grasp_center = top.copy()
    grasp_center[2] -= cylinder_height_m / 2.0
    return grasp_center - np.asarray(TCP_TO_GRASP_CENTER_M, dtype=np.float64)


def main():
    args = parse_args()
    cylinder_height_m = None
    if TCP_TO_GRASP_CENTER_M is not None:
        text = input("Cylinder height in millimetres: ").strip()
        cylinder_height_m = float(text) / 1000.0
        if cylinder_height_m <= 0:
            raise ValueError("cylinder height must be positive")

    robot = None
    gripper = None
    ho_cam = None
    hi_camera_only = None
    try:
        if not args.camera_only:
            robot = UR_Robot(
                robot_ip=ROBOT_IP,
                is_use_robot=True,
                is_use_camera=True,
                camera_serial=HI_SERIAL,
                cam2end_path=str(HI_CAM2END_PATH),
                cam_ini_path=str(HI_CAMERA_INI_PATH),
            )
            try:
                gripper = grasp_claw(gripper_port=args.gripper_port)
                print("[OK] gripper connected, position=", gripper.read_position())
            except Exception as exc:
                gripper = None
                print("[WARN] gripper unavailable; grip stage disabled:", exc)

        else:
            hi_camera_only = Camera(serial=HI_SERIAL)

        ho_cam = Camera(serial=HO_SERIAL)
        ho_depth_scale = float(np.loadtxt(HO_DEPTH_SCALE_PATH))
        ho_transform = np.loadtxt(HO_CALIB_PATH)
        if ho_transform.shape != (4, 4):
            raise ValueError("eye-out calibration must be a 4x4 matrix")
        detector = RedCylinderDetector()
        stage = Stage.WAIT_HO
        ho_history = deque(maxlen=STABLE_WINDOW)
        hi_history = deque(maxlen=STABLE_WINDOW)
        ho_base_history = deque(maxlen=STABLE_WINDOW)
        hi_base_history = deque(maxlen=STABLE_WINDOW)
        ho_target = None
        hi_target = None
        grasp_tcp = None

        print("Keys: l lock HO | m move observe | i lock HI | p pregrasp | d descend")
        print("      o open | g close | u lift | r reset cycle | q quit")
        if TCP_TO_GRASP_CENTER_M is None:
            print("[SAFE] TCP offset is unset: p/d are intentionally disabled")

        while True:
            ho_color, ho_depth = ho_cam.get_data()
            ho_res = detector.detect(ho_color, ho_depth.astype(np.float64))
            ho_view = detector.draw(ho_color, ho_res)
            if ho_res is not None and ho_res["z_mm"] is not None:
                u, v = ho_res["center"]
                ho_history.append((u, v, ho_res["z_mm"]))
                pt = ho_pixel_to_base(
                    u, v, ho_res["z_mm"], ho_depth_scale, ho_transform)
                if pt is not None:
                    pt = np.asarray(pt, dtype=np.float64)
                    pt[2] += args.ho_z_offset
                    if in_workspace(pt):
                        ho_base_history.append(pt)
            else:
                ho_history.clear()
                ho_base_history.clear()

            hi_view = np.zeros_like(ho_view)
            hi_res = None
            if robot is not None:
                hi_color, hi_depth_raw = robot.get_camera_data()
                hi_depth_mm = hi_depth_raw.astype(np.float64) * robot.depth_scale * 1000.0
                hi_res = detector.detect(hi_color, hi_depth_mm)
                hi_view = detector.draw(hi_color, hi_res)
                if hi_res is not None and hi_res["z_mm"] is not None:
                    u, v = hi_res["center"]
                    hi_history.append((u, v, hi_res["z_mm"]))
                    _, pt_m = robot.pixel_to_base(u, v, hi_res["z_mm"])
                    if in_workspace(pt_m):
                        hi_base_history.append(np.asarray(pt_m, dtype=np.float64))
                else:
                    hi_history.clear()
                    hi_base_history.clear()
            elif hi_camera_only is not None:
                hi_color, hi_depth_raw = hi_camera_only.get_data()
                hi_depth_mm = (hi_depth_raw.astype(np.float64)
                               * hi_camera_only.scale * 1000.0)
                hi_res = detector.detect(hi_color, hi_depth_mm)
                hi_view = detector.draw(hi_color, hi_res)

            cv2.putText(ho_view, "stage=" + stage, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (0, 255, 0), 2)
            cv2.imshow("HO red cylinder", ho_view)
            cv2.imshow("HI red cylinder", hi_view)
            key = cv2.waitKey(1) & 0xFF

            try:
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    stage = Stage.WAIT_HO
                    ho_target = hi_target = grasp_tcp = None
                    print("[RESET] new cycle")
                elif key == ord("l"):
                    if stage != Stage.WAIT_HO:
                        print("[BLOCK] HO can only be locked at WAIT_HO")
                    elif not stable_detection(ho_history):
                        print("[BLOCK] HO detection is not stable")
                    else:
                        ho_target = median_stable(ho_base_history)
                        if ho_target is None:
                            print("[BLOCK] HO base coordinates are not stable")
                        else:
                            stage = Stage.HO_LOCKED
                            print("[LOCK:HO]", np.round(ho_target, 4))
                elif key == ord("m"):
                    if stage != Stage.HO_LOCKED or robot is None:
                        print("[BLOCK] lock HO and connect robot first")
                    else:
                        observe = ho_target.copy()
                        observe[2] += OBSERVE_CLEARANCE_M
                        safe_move(robot, observe, "HO-observe")
                        stage = Stage.AT_OBSERVE
                        hi_history.clear(); hi_base_history.clear()
                elif key == ord("i"):
                    if stage != Stage.AT_OBSERVE:
                        print("[BLOCK] move to observation pose first")
                    elif not stable_detection(hi_history):
                        print("[BLOCK] HI detection is not stable")
                    else:
                        hi_target = median_stable(hi_base_history)
                        if hi_target is None:
                            print("[BLOCK] HI base coordinates are not stable")
                        else:
                            stage = Stage.HI_LOCKED
                            print("[LOCK:HI]", np.round(hi_target, 4))
                elif key == ord("p"):
                    if stage != Stage.HI_LOCKED or robot is None:
                        print("[BLOCK] lock HI first")
                    elif TCP_TO_GRASP_CENTER_M is None:
                        print("[BLOCK] measure and set TCP_TO_GRASP_CENTER_M first")
                    else:
                        grasp_tcp = calculate_tcp_grasp_position(hi_target, cylinder_height_m)
                        pregrasp = grasp_tcp.copy(); pregrasp[2] += PREGRASP_CLEARANCE_M
                        safe_move(robot, pregrasp, "HI-pregrasp")
                        stage = Stage.AT_PREGRASP
                elif key == ord("d"):
                    if stage != Stage.AT_PREGRASP or robot is None:
                        print("[BLOCK] reach pregrasp first")
                    else:
                        safe_move(robot, grasp_tcp, "descend")
                        stage = Stage.AT_GRASP
                elif key == ord("o"):
                    if gripper is None:
                        print("[BLOCK] gripper unavailable")
                    else:
                        gripper.grip(GRIPPER_OPEN_POSITION, GRIPPER_SPEED, GRIPPER_FORCE)
                        print("[GRIPPER] opened")
                elif key == ord("g"):
                    if stage != Stage.AT_GRASP or gripper is None:
                        print("[BLOCK] descend and connect gripper first")
                    else:
                        pos = gripper.grip(GRIPPER_CLOSED_POSITION, GRIPPER_SPEED, GRIPPER_FORCE)
                        print("[GRIPPER] closed, position=", pos)
                        stage = Stage.GRIPPED
                elif key == ord("u"):
                    if stage != Stage.GRIPPED or robot is None:
                        print("[BLOCK] close gripper first")
                    else:
                        current = np.asarray(robot.get_actual_tcp_pose()[:3], dtype=np.float64)
                        current[2] += LIFT_DISTANCE_M
                        safe_move(robot, current, "lift")
                        stage = Stage.LIFTED
            except Exception as exc:
                print("[ERROR] action refused/failed:", exc)

    finally:
        cv2.destroyAllWindows()
        if ho_cam is not None:
            try:
                ho_cam.close()
            except Exception:
                pass
        if robot is not None and getattr(robot, "camera", None) is not None:
            try:
                robot.camera.close()
            except Exception:
                pass
        if hi_camera_only is not None:
            try:
                hi_camera_only.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
