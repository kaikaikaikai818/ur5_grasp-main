#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
红色圆柱一键抓取（手外粗定位 -> 手内精定位 -> 识别抓取，含夹爪）

全流程（按 g 触发）：
  ① 粗定位  手外 D455 检测红色圆柱 -> pixel_to_robot_coords -> 基座坐标
            -> 机械臂粗移到目标上方
  ② 精定位  手内 D435I 检测 -> pixel_to_base -> 细化基座坐标（多帧重试）
            -> 精对齐到细化坐标上方；若多次失败则回退用粗定位坐标直接抓
  ③ 抓取   下降 -> 收爪 -> 抬起（只抓起+抬起，不含放置）

按键：
  g 执行① ② ③     o 夹爪张开     c 夹爪闭合     q 退出
自检：
  --check-calib  启动时读两台相机实时内参，与标定内参比对，超阈值拒绝执行
  --gripper-test 交互式校定夹爪开/合 position
"""
import argparse
import time
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot, load_camera_ini
from bsp.camera_bsp.realsenseD415 import Camera
from bsp.camera_bsp.hand_out_eye_calibration import HandOutEyeCalibration
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector

# -------------------------- 配置 --------------------------
ROBOT_IP = "192.168.1.35"
HI_SERIAL = "215222074676"     # 手内 D435I（机器人内置相机）
HO_SERIAL = "215122257404"     # 手外 D455（固定相机）
CALIB_PATH = "camera_pose.txt"              # 手外: 相机->基座 4x4 (米)
DEPTH_SCALE_FILE = "camera_depth_scale.txt" # 手外: 原始 z16 计数 -> 米
CAM_INI = "camera_20260906.ini"             # 手内 D435I 标定内参/畸变

# 手外 D455 内参（与 camera_pose.txt 标定时所用一致）
HO_FX = 386.471
HO_FY = 386.034
HO_CX = 321.617
HO_CY = 237.200
# 手外内参与标定值容差（像素）
CALIB_TOL = 3.0

TOOL_ORIENTATION = [3.141, 0.0, 0.0]   # 固定朝下 (RX, RY, RZ)
LIFT_ABOVE = 0.05             # 目标正上方 50mm
CYL_H = 0.03                  # 圆柱高(m)，用于下探深度下界（避免穿底）
GRASP_DEPTH_OFFSET = 0.025    # 低于顶面 z 的下探深度（让手指跨住圆柱中下部）
LIFT_Z_OFFSET = 0.15          # 抓起后抬起高度
HO_Z_OFFSET = 0.026           # 手外 D455 高度基准补偿（标定 z 整体偏低 0.026m）
HI_Z_OFFSET = 0.0             # 手内 z 补偿（若顶面 z 偏低可微调）

# 精定位：多次采样取有效值，若连续 REFINE_RETRY_N 次都失败则回退粗定位
REFINE_RETRY_N = 5

# 夹爪
GRIP_PORT = "COM10"
GRIP_OPEN_POS = 6000          # 张开
GRIP_CLOSE_POS = 11000        # 闭合（参考仓库值，可用 --gripper-test 校定）
GRIP_SPEED = 50
GRIP_FORCE = 50               # 力矩百分比(≤100)，过低压不扁
GRIP_OPEN_SPEED = 100
GRIP_OPEN_FORCE = 40
GRIP_TORQUE_MIN = 80          # 实时力矩(0x060C)阈值: 低于此且力矩未到达即判空抓

WORKSPACE_LIMITS = [[-0.5, 0.05], [-0.80, -0.45], [-0.2, 0.6]]
GRASP_HOME = [-0.4, -0.025, 0.14981] + TOOL_ORIENTATION


def main():
    args = parse_args()

    # 1. 机器人（手内相机 + 夹爪）
    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=True, is_use_camera=True,
                     is_use_gripper=True, gripper_port=GRIP_PORT)
    print("[OK] 机械臂已连接:", ROBOT_IP)

    # 2. 手外 D455
    ho_cam = Camera(serial=HO_SERIAL)
    print("[OK] HO(D455) 已连接:", HO_SERIAL)
    K_ho = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    depth_scale = float(np.loadtxt(DEPTH_SCALE_FILE))

    # 3. 自检（可选）：比对实时内参与标定内参
    if args.check_calib:
        check_calib(robot, ho_cam)

    class ParamHolder:
        cam_intrinsics = K_ho
        workspace_limits = WORKSPACE_LIMITS

    ho = HandOutEyeCalibration(robot=ParamHolder(), calib_path=CALIB_PATH,
                               cam_depth_scale=depth_scale)
    print("[OK] 手外标定加载完成 (camera_pose.txt, cam->base, 米)")

    detector = RedCylinderDetector()

    win_ho = "HO(D455)_eye_out"
    win_hi = "HI(D435I)_eye_in"
    cv2.namedWindow(win_ho, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_hi, cv2.WINDOW_NORMAL)

    state = {"ho_base": None, "hi_base": None}

    print("\n操作说明:")
    print("  红色圆柱会被自动检测（双视场）。")
    print("  g -> 一键抓取：手外粗定位 -> 机械臂移到上方 -> 手内精定位 -> 下降收爪 -> 抬起")
    print("  o -> 夹爪张开     c -> 夹爪闭合     q -> 退出")

    try:
        while True:
            # ---- 手外 D455：粗定位 ----
            ho_color, ho_depth = ho_cam.get_data()
            ho_disp = ho_color.copy()
            res_ho = detector.detect(ho_color, ho_depth.astype(np.float64))
            state["ho_base"] = None
            if res_ho is not None:
                pt = ho.pixel_to_robot_coords(*res_ho["center"], ho_depth)
                if pt is not None:
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2]) + HO_Z_OFFSET
                    state["ho_base"] = (x, y, z)
                    cv2.circle(ho_disp, res_ho["center"], 6, (0, 0, 255), 2)
                    cv2.putText(ho_disp, "HO base [%.3f, %.3f, %.3f]" % (x, y, z),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(ho_disp, "HO (D455) eye-out", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win_ho, ho_disp)

            # ---- 手内 D435I：精定位预览 ----
            hi_color, hi_depth = robot.get_camera_data()
            hi_disp = hi_color.copy() if hi_color is not None else np.zeros((480, 640, 3), np.uint8)
            state["hi_base"] = None
            if hi_color is not None:
                res_hi = detector.detect(hi_color, hi_depth.astype(np.float64))
                if res_hi is not None and res_hi["z_mm"] is not None:
                    base_mm, base_m = robot.pixel_to_base(*res_hi["center"], res_hi["z_mm"])
                    x, y, z = float(base_m[0]), float(base_m[1]), float(base_m[2]) + HI_Z_OFFSET
                    state["hi_base"] = (x, y, z)
                    cv2.circle(hi_disp, res_hi["center"], 6, (0, 0, 255), 2)
                    cv2.putText(hi_disp, "HI base [%.3f, %.3f, %.3f]" % (x, y, z),
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(hi_disp, "HI (D435I) eye-in", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win_hi, hi_disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('o'):
                robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
                print("[夹爪] 张开 pos=%d" % GRIP_OPEN_POS)
            elif key == ord('c'):
                robot.grip(GRIP_CLOSE_POS, GRIP_SPEED, GRIP_FORCE)
                print("[夹爪] 闭合 pos=%d" % GRIP_CLOSE_POS)
            elif key == ord('g'):
                do_grasp(robot, detector, state)
    finally:
        cv2.destroyAllWindows()
        print("退出。")


def check_calib(robot, ho_cam):
    """比对实时内参与标定内参，超阈值则抛错拒绝执行。"""
    ok = True

    # ---- 手外 D455 ----
    live = ho_cam.intrinsics
    exp = np.array([[HO_FX, 0, HO_CX], [0, HO_FY, HO_CY], [0, 0, 1]])
    dho = max(abs(live[0, 0] - exp[0, 0]), abs(live[1, 1] - exp[1, 1]),
              abs(live[0, 2] - exp[0, 2]), abs(live[1, 2] - exp[1, 2]))
    print("手外D455 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
          % (live[0, 0], live[1, 1], live[0, 2], live[1, 2],
             exp[0, 0], exp[1, 1], exp[0, 2], exp[1, 2], dho))
    if dho > CALIB_TOL:
        print("[自检失败] 手外D455 内参与标定值不符(%.3f>%.1f)！"
              % (dho, CALIB_TOL))
        ok = False

    # ---- 手内 D435I ----
    calib = load_camera_ini(CAM_INI)
    if calib is None:
        print("[警告] 未找到/无法解析 %s，手内内参无法校验" % CAM_INI)
    else:
        K_ini, _ = calib
        live_hi = robot.camera.intrinsics
        dhi = max(abs(live_hi[0, 0] - K_ini[0, 0]), abs(live_hi[1, 1] - K_ini[1, 1]),
                  abs(live_hi[0, 2] - K_ini[0, 2]), abs(live_hi[1, 2] - K_ini[1, 2]))
        print("手内D435I 实时 fx=%.3f fy=%.3f cx=%.3f cy=%.3f | 期望=%.3f/%.3f/%.3f/%.3f | Δmax=%.3f px"
              % (live_hi[0, 0], live_hi[1, 1], live_hi[0, 2], live_hi[1, 2],
                 K_ini[0, 0], K_ini[1, 1], K_ini[0, 2], K_ini[1, 2], dhi))
        if dhi > CALIB_TOL:
            print("[自检失败] 手内D435I 内参与标定值不符(%.3f>%.1f)！" % (dhi, CALIB_TOL))
            ok = False

    if not ok:
        raise RuntimeError("内参自检未通过，请检查相机/标定。不要继续执行抓取。")


def do_grasp(robot, detector, state):
    """一键全流程：① 粗定位 -> ② 精定位 -> ③ 下降收爪抬起。失败回退安全。"""
    coarse = state["ho_base"] or state["hi_base"]
    if coarse is None:
        print("[提示] 未检测到红色圆柱，无法抓取")
        return

    x, y, z = coarse
    # ③ 工作空间钳制：x/y 同样限制，防止目标点在可达范围外
    clamp = lambda v, lim: max(lim[0], min(v, lim[1]))
    x = clamp(x, WORKSPACE_LIMITS[0])
    y = clamp(y, WORKSPACE_LIMITS[1])
    z = clamp(z, WORKSPACE_LIMITS[2])
    try:
        # ① 粗定位：手外基座坐标 -> 移到目标上方
        print("[①粗定位] 手外 base = [%.3f, %.3f, %.3f]" % (x, y, z))
        robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
        above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
        print("[移动] 粗定位上方 moveL %s" % (["%.3f" % v for v in above],))
        robot.moveL(above, speed=0.05, acceleration=0.05)
        time.sleep(1)

        # ② 精定位：手内多帧重试；失败则回退粗定位坐标
        refined = refine_locate(robot, detector, REFINE_RETRY_N)
        if refined is not None:
            x, y, z = refined
            x = clamp(x, WORKSPACE_LIMITS[0])
            y = clamp(y, WORKSPACE_LIMITS[1])
            z = clamp(z, WORKSPACE_LIMITS[2])
            print("[②精定位] 手内 base = [%.3f, %.3f, %.3f]" % (x, y, z))
        else:
            print("[②精定位] 连续 %d 次失败，回退用粗定位坐标直接抓" % REFINE_RETRY_N)
            above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
            robot.moveL(above, speed=0.05, acceleration=0.05)

        # ③ 精对齐 -> 下降 -> 收爪 -> 抬起
        above = [x, y, z + LIFT_ABOVE] + TOOL_ORIENTATION
        print("[移动] 精定位上方 moveL %s" % (["%.3f" % v for v in above],))
        robot.moveL(above, speed=0.05, acceleration=0.05)

        clamp = lambda v, lim: max(lim[0], min(v, lim[1]))
        z_low = clamp(z - GRASP_DEPTH_OFFSET, WORKSPACE_LIMITS[2])
        floor_z = z - CYL_H               # 圆柱底面的 z，下探不得低于此
        if z_low < floor_z:
            print("[安全] 下探 z=%.3f 低于圆柱底面 z=%.3f，抬至底面高度" % (z_low, floor_z))
            z_low = floor_z
        grasp_pt = [x, y, z_low] + TOOL_ORIENTATION
        print("[移动] 下降抓取 moveL %s" % (["%.3f" % v for v in grasp_pt],))
        robot.moveL(grasp_pt, speed=0.05, acceleration=0.05)

        print("[夹爪] 收爪抓取 pos=%d" % GRIP_CLOSE_POS)
        robot.grip(GRIP_CLOSE_POS, GRIP_SPEED, GRIP_FORCE)
        torq_reached = robot.read_torque_reached()
        torq_cur = robot.read_torque_current()
        print("[夹爪] 力矩到达=%d  实时力矩=%d" % (torq_reached, torq_cur))
        if not (torq_reached == 1 or torq_cur >= GRIP_TORQUE_MIN):
            print("[抓取失败] 力矩未到达且实时力矩过低(疑似空抓)，松开并中止，不抬升")
            robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
            return
        time.sleep(1)

        lift = [x, y, z + LIFT_Z_OFFSET] + TOOL_ORIENTATION
        print("[移动] 抬起 moveL %s" % (["%.3f" % v for v in lift],))
        robot.moveL(lift, speed=0.05, acceleration=0.05)

        print("抓取完成（已抓起并抬起，未放置）。按 o 张开可放下，q 退出。")
    except Exception as e:
        print("[异常] 抓取流程出错：%s" % e)
        safe_return(robot)


def refine_locate(robot, detector, n_repeat):
    """手内精定位，重复 n_repeat 次取第一个有效值；全失败返回 None。"""
    for i in range(n_repeat):
        hi_color, hi_depth = robot.get_camera_data()
        if hi_color is not None:
            res = detector.detect(hi_color, hi_depth.astype(np.float64))
            if res is not None and res["z_mm"] is not None:
                base_mm, base_m = robot.pixel_to_base(*res["center"], res["z_mm"])
                return (float(base_m[0]), float(base_m[1]),
                        float(base_m[2]) + HI_Z_OFFSET)
        time.sleep(0.3)
    return None


def safe_return(robot):
    """异常时回 home 并张开爪。"""
    try:
        robot.grip(GRIP_OPEN_POS, GRIP_OPEN_SPEED, GRIP_OPEN_FORCE)
        robot.moveL(GRASP_HOME, speed=0.05, acceleration=0.05)
    except Exception as e:
        print("[提示] 复位失败，请手动检查机器人：%s" % e)


def parse_args():
    p = argparse.ArgumentParser(description="红色圆柱一键抓取（手外粗定位->手内精定位->抓取）")
    p.add_argument("--gripper-test", action="store_true",
                   help="仅测试夹爪：交互式发送 position 并回显，用于定开/合值")
    p.add_argument("--check-calib", action="store_true",
                   help="启动时读实时内参并比对标定内参，超阈值则拒绝执行")
    return p.parse_args()


def gripper_test():
    """交互式夹爪自检：输入 position -> 控制并回显当前 POS。"""
    robot = UR_Robot(robot_ip=ROBOT_IP, is_use_robot=True, is_use_camera=False,
                     is_use_gripper=True, gripper_port=GRIP_PORT)
    print("夹爪自检：输入 position（0~65535）控制，观察爪的开合。")
    print("  数值越小越张开、越大越闭合。q 退出。")
    while True:
        s = input("position(回车看当前值) / q 退出 > ").strip()
        if not s:
            print("当前POS =", robot.read_position())
            continue
        if s.lower() == 'q':
            break
        try:
            pos = int(s)
        except ValueError:
            print("请输入整数")
            continue
        robot.grip(pos, 100, 40)
        print("已发 position=%d，回读 POS=%d" % (pos, robot.read_position()))
    print("结束。把张开/闭合对应的 position 填到脚本的 GRIP_OPEN_POS/GRIP_CLOSE_POS。")


if __name__ == "__main__":
    args = parse_args()
    if args.gripper_test:
        gripper_test()
    else:
        main()
