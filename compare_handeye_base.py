#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
只读对照脚本：同一时刻打印 手外D455 与 手内D435I 两套基座坐标，绝不 moveL。

用法：python compare_handeye_base.py
按键：q / ESC 退出（无任何运动，不初始化夹爪）

目的：判断 "直接远离目标" 是手外粗定位还是手内精定位的问题。
      HO base（D455 cam->base，米）  vs  HI base（D435I cam->end->base，米）
      两套接近 -> 两套标定基本正常（问题可能在可达性/夹爪/下探深度）
      两套相差大 -> 以 HO 为准的手外标定(camera_pose.txt/硬编码内参/深度比例)需重标
"""
import argparse
import time
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.camera_bsp.hand_out_eye_calibration import HandOutEyeCalibration
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector

# 与 grasp_cylinder.py 保持一致
ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"        # 手内 D435I
HO_SERIAL = "215122257404"        # 手外 D455
CALIB_PATH = "camera_pose.txt"    # 手外: cam->base (米)
DEPTH_SCALE_FILE = "camera_depth_scale.txt"
CAM_INI = "camera_20260906.ini"   # 手内: 标定内参/畸变
HO_FX, HO_FY, HO_CX, HO_CY = 386.471, 386.034, 321.617, 237.200
HO_Z_OFFSET = 0.026
HI_Z_OFFSET = 0.0
WORKSPACE_LIMITS = [[-0.5, 0.05], [-0.80, -0.45], [-0.2, 0.6]]


def main():
    ap = argparse.ArgumentParser(description="打印 手外D455 / 手内D435I 两套基座坐标(只读, 绝不 moveL)")
    ap.add_argument("--no-clamp", action="store_true",
                    help="关闭手外工作空间钳制(用大范围), 打印手外真实未裁剪坐标")
    args = ap.parse_args()

    # 手内 D435I（连 RTDE 以读实时 TCP；不启用夹爪）
    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=True,
                     is_use_camera=True, is_use_gripper=False,
                     camera_serial=HI_SERIAL, cam_ini_path=CAM_INI)

    # 手外 D455
    ho_cam = Camera(serial=HO_SERIAL)
    K_ho = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    depth_scale = float(np.loadtxt(DEPTH_SCALE_FILE))

    ho_limits = ([[-2.0, 2.0], [-2.0, 2.0], [-2.0, 2.0]] if args.no_clamp
                 else WORKSPACE_LIMITS)

    class ParamHolder:
        cam_intrinsics = K_ho
        workspace_limits = ho_limits

    ho = HandOutEyeCalibration(robot=ParamHolder(), calib_path=CALIB_PATH,
                               cam_depth_scale=depth_scale)
    detector = RedCylinderDetector()

    win = "compare"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("[只读] 仅打印两套基座坐标，绝不移动机械臂。q/ESC 退出。")
    print("HO = 手外D455(cam->base: %s)   HI = 手内D435I(cam->end->base + 实时TCP)" % CALIB_PATH)
    print("钳制: %s" % ("关(--no-clamp, 打印真实未裁剪 x)" if args.no_clamp else "开(默认, x 会被裁剪到 0.05)"))
    last = 0.0

    try:
        while True:
            # ---- 手外 D455：粗定位 ----
            ho_color, ho_depth = ho_cam.get_data()
            ho_disp = ho_color.copy()
            res_ho = detector.detect(ho_color, ho_depth.astype(np.float64))
            ho_raw = ho_off = None
            if res_ho is not None:
                pt = ho.pixel_to_robot_coords(*res_ho["center"], ho_depth)
                if pt is not None:
                    ho_raw = np.asarray(pt, dtype=np.float64)
                    ho_off = ho_raw + np.array([0.0, 0.0, HO_Z_OFFSET])
                    cv2.circle(ho_disp, res_ho["center"], 6, (0, 0, 255), 2)

            # ---- 手内 D435I：精定位 ----
            hi_color, hi_depth = robot.get_camera_data()
            hi_disp = (hi_color.copy() if hi_color is not None
                       else np.zeros((480, 640, 3), np.uint8))
            hi_raw = hi_off = None
            hi_z = None
            if hi_color is not None:
                res_hi = detector.detect(hi_color, hi_depth.astype(np.float64))
                if res_hi is not None and res_hi["z_mm"] is not None:
                    hi_z = res_hi["z_mm"]
                    base_mm, base_m = robot.pixel_to_base(*res_hi["center"], hi_z)
                    hi_raw = np.asarray(base_m, dtype=np.float64)
                    hi_off = hi_raw + np.array([0.0, 0.0, HI_Z_OFFSET])
                    cv2.circle(hi_disp, res_hi["center"], 6, (0, 0, 255), 2)

            # 列表格输出（约每 0.5s 一次，避免刷屏）
            now = time.time()
            if (ho_raw is not None or hi_raw is not None) and now - last > 0.5:
                last = now
                print("---- frame ----")
                if ho_raw is not None:
                    print("  HO px=%s  raw(m)=%s  +offset=%s"
                          % (res_ho["center"], np.round(ho_raw, 4), np.round(ho_off, 4)))
                if hi_raw is not None:
                    print("  HI px=%s  z=%.1fmm  raw(m)=%s  +offset=%s"
                          % (res_hi["center"], hi_z, np.round(hi_raw, 4), np.round(hi_off, 4)))
                if ho_raw is not None and hi_raw is not None:
                    d = hi_off - ho_off
                    print("  DIFF(HI-HO)(m)= %s   |d|=%.3f"
                          % (np.round(d, 4), float(np.linalg.norm(d))))

            cv2.putText(ho_disp, "HO (D455)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(hi_disp, "HI (D435I)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, np.hstack([ho_disp, hi_disp]))

            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break
    finally:
        cv2.destroyAllWindows()
        print("退出（未执行任何运动）。")


if __name__ == "__main__":
    main()
