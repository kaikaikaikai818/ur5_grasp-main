#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
抓取目标预演（只读！绝不 moveL / grip）。

按 grasp_cylinder.py 的 do_grasp 逻辑，计算它将会发送的 4 个 moveL 目标，
打印 + 叠加到手内画面，并做工作空间 / 下探深度合理性检查。
按键：q/ESC 退出。

用法：python preview_grasp.py [--cyl-h 0.03]
"""
import argparse
import time
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.camera_bsp.hand_out_eye_calibration import HandOutEyeCalibration
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector

ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"
HO_SERIAL = "215122257404"
CALIB_PATH = "camera_pose.txt"
DEPTH_SCALE_FILE = "camera_depth_scale.txt"
CAM_INI = "camera_20260906.ini"
HO_FX, HO_FY, HO_CX, HO_CY = 386.471, 386.034, 321.617, 237.200
TOOL_ORIENTATION = [3.141, 0.0, 0.0]
LIFT_ABOVE = 0.05
CYL_H = 0.03
GRASP_DEPTH_OFFSET = 0.025
LIFT_Z_OFFSET = 0.15
HO_Z_OFFSET = 0.026
HI_Z_OFFSET = 0.0
WORKSPACE_LIMITS = [[-0.5, 0.05], [-0.80, -0.45], [-0.2, 0.6]]


def pose_vector_to_matrix(pose):
    T = np.eye(4)
    rv = np.asarray(pose[3:6], dtype=np.float64)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        R = np.eye(3)
    else:
        k = rv / th
        Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        R = np.eye(3) + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(pose[0:3], dtype=np.float64) * 1000.0
    return T


def base_to_pixel(p_base_m, tcp_pose, T_cam2end, K):
    """基座系目标点 -> 手内相机像素。用于叠加显示抓取点落在画面哪里。"""
    p_base_mm = np.array(p_base_m[:3], dtype=np.float64) * 1000.0
    T_inv = np.linalg.inv(pose_vector_to_matrix(tcp_pose) @ T_cam2end)
    p_cam = (T_inv @ np.concatenate([p_base_mm, [1.0]]))[:3]
    if p_cam[2] <= 1e-6:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return (int(u), int(v))


def reachable(pt, limits):
    return all(limits[i][0] <= pt[i] <= limits[i][1] for i in range(3))


def main():
    ap = argparse.ArgumentParser(description="抓取目标预演(只读, 绝不 moveL)")
    ap.add_argument("--cyl-h", type=float, default=0.03, help="圆柱高度(m), 默认0.03")
    args = ap.parse_args()

    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=True, is_use_camera=True,
                     is_use_gripper=False, camera_serial=HI_SERIAL,
                     cam_ini_path=CAM_INI)
    ho_cam = Camera(serial=HO_SERIAL)
    K_ho = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    depth_scale = float(np.loadtxt(DEPTH_SCALE_FILE))

    class ParamHolder:
        cam_intrinsics = K_ho
        workspace_limits = WORKSPACE_LIMITS

    ho = HandOutEyeCalibration(robot=ParamHolder(), calib_path=CALIB_PATH,
                               cam_depth_scale=depth_scale)
    detector = RedCylinderDetector()
    T_cam2end = robot.T_cam2end
    K = robot.cam_intrinsics

    win = "preview"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("[只读预演] 只计算并显示抓取目标, 绝不 moveL/grip。q/ESC 退出。")
    print("圆柱高=%.3fm  GRASP_DEPTH_OFFSET=%.3f" % (args.cyl_h, GRASP_DEPTH_OFFSET))
    last = 0.0
    try:
        while True:
            ho_color, ho_depth = ho_cam.get_data()
            res_ho = detector.detect(ho_color, ho_depth.astype(np.float64))
            coarse = None
            if res_ho is not None:
                pt = ho.pixel_to_robot_coords(*res_ho["center"], ho_depth)
                if pt is not None:
                    coarse = np.asarray(pt) + np.array([0.0, 0.0, HO_Z_OFFSET])

            hi_color, hi_depth = robot.get_camera_data()
            hi_disp = (hi_color.copy() if hi_color is not None
                       else np.zeros((480, 640, 3), np.uint8))
            refined = None
            if hi_color is not None:
                res_hi = detector.detect(hi_color, hi_depth.astype(np.float64))
                if res_hi is not None and res_hi["z_mm"] is not None:
                    _, base_m = robot.pixel_to_base(*res_hi["center"], res_hi["z_mm"])
                    refined = np.asarray(base_m) + np.array([0.0, 0.0, HI_Z_OFFSET])

            now = time.time()
            if (coarse is not None or refined is not None) and now - last > 0.5:
                last = now
                print("---- frame ----")
                src = refined if refined is not None else coarse
                if src is None:
                    print("  未识别到圆柱(手外/手内都无), 无法预测。")
                else:
                    kind = "精(手内)" if refined is not None else "粗(手外)"
                    x, y, z = src
                    top_above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
                    z_low = max(WORKSPACE_LIMITS[2][0],
                                min(z - GRASP_DEPTH_OFFSET, WORKSPACE_LIMITS[2][1]))
                    floor_z = z - CYL_H
                    if z_low < floor_z:
                        z_low = floor_z
                        print("   [安全] 下探抬至底面 z=%.3f (GRASP_DEPTH_OFFSET=%.3f 对 %.3f 高圆柱太深)"
                              % (z_low, GRASP_DEPTH_OFFSET, CYL_H))
                    grasp_pt = [x, y, z_low] + TOOL_ORIENTATION
                    lift = [x, y, z + LIFT_Z_OFFSET] + TOOL_ORIENTATION
                    print("  使用%s坐标: base=(%.4f, %.4f, %.4f)" % (kind, x, y, z))
                    for name, pt in [("上方(+0.05)", top_above),
                                     ("下探(-%.3f)" % (z - z_low), grasp_pt),
                                     ("抬起(+0.15)", lift)]:
                        ok = reachable(pt[:3], WORKSPACE_LIMITS)
                        print("    %-12s moveL=%.4f,%.4f,%.4f reach=%s"
                              % (name, pt[0], pt[1], pt[2], "OK" if ok else "越界!"))
                    grip_z = z - GRASP_DEPTH_OFFSET
                    base_of_cyl = z - args.cyl_h
                    ideal_grip_z = z - args.cyl_h / 2.0
                    if grip_z < base_of_cyl - 0.005:
                        print("   [警告] 下探 z=%.3f 低于圆柱底面 z=%.3f (圆柱高%.3f), 会穿到底/过深!"
                              % (grip_z, base_of_cyl, args.cyl_h))
                    else:
                        print("   下探 z=%.3f  理想跨住 z≈%.3f" % (grip_z, ideal_grip_z))
                    for name, pt in [("TOP", top_above), ("GRASP", grasp_pt)]:
                        pp = base_to_pixel(pt[:3], robot.get_actual_tcp_pose(), T_cam2end, K)
                        if pp is not None:
                            cv2.circle(hi_disp, pp, 8, (0, 255, 255), 2)
                            cv2.putText(hi_disp, name, (pp[0] + 10, pp[1]),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow(win, hi_disp)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
        print("退出(未执行任何运动)。")


if __name__ == "__main__":
    main()
