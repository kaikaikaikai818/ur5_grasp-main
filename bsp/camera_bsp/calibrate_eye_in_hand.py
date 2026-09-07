#!/usr/bin/env python

import numpy as np
import time
import cv2
from UR_Robot import UR_Robot
from scipy import optimize  

# ---------------------- 眼在手上标定参数（修改部分） ----------------------
# 1. 固定棋盘格在基坐标系下的3D坐标（世界坐标系）
# 假设棋盘格为5x5角点，格距0.04m，原点在棋盘格左下角，z=0（放置在工作台上）
checkerboard_grid_size = 0.04  # 棋盘格格距（米）
checkerboard_corners = []
for i in range(5):
    for j in range(5):
        checkerboard_corners.append([i * checkerboard_grid_size, j * checkerboard_grid_size, 0.0])
checkerboard_corners = np.array(checkerboard_corners)  # 5x5=25个角点的3D坐标（基坐标系下）

# 2. 机器人运动范围（用于采集不同位姿，相机随末端运动）
workspace_limits = np.asarray([[-0.385, -0.34], [-0.34, -0.26], [0.12, 0.20]])  # 机器人末端运动范围
calib_grid_step = 0.04  # 运动步长
tool_orientation = [1.18, -1.2, 1.13]  # 末端姿态（保持不变，仅移动位置）
# --------------------------------------------------------------------------

# 生成机器人末端运动的网格点（相机随末端在这些点采集图像）
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], 
                          int(1 + (workspace_limits[0][1] - workspace_limits[0][0])/calib_grid_step))
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], 
                          int(1 + (workspace_limits[1][1] - workspace_limits[1][0])/calib_grid_step))
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], 
                          int(1 + (workspace_limits[2][1] - workspace_limits[2][0])/calib_grid_step))
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]

calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

# 存储数据：工具在基坐标系的位姿 + 棋盘格在相机坐标系的坐标
measured_tool_poses = []  # 工具坐标系在基坐标系下的4x4变换矩阵
observed_checkerboard_pts = []  # 棋盘格角点在相机坐标系下的3D坐标

# 连接机器人
print('Connecting to robot...')
robot = UR_Robot(robot_ip="192.168.1.35", gripper_port='COM6', gripper_baudrate=115200, 
                 gripper_address=1, workspace_limits=workspace_limits, 
                 is_use_robot=True, is_use_camera=True)

# 降低机器人速度
robot.joint_acc = 0.2
robot.joint_spd = 0.2

# 采集数据：移动末端（带相机）到不同位置，观测固定棋盘格
print('Collecting data...')
for calib_pt_idx in range(num_calib_grid_pts):
    # 移动机器人末端到目标位置
    tool_position = calib_grid_pts[calib_pt_idx, :]
    tool_config = [tool_position[0], tool_position[1], tool_position[2],
                   tool_orientation[0], tool_orientation[1], tool_orientation[2]]
    print(f"Moving to tool pose: {tool_config}")
    robot.moveL(tool_config)
    time.sleep(2)  # 等待稳定

    # 从相机获取图像并检测棋盘格
    camera_color_img, camera_depth_img = robot.get_camera_data()
    bgr_color_data = cv2.cvtColor(camera_color_img, cv2.COLOR_RGB2BGR)
    gray_data = cv2.cvtColor(bgr_color_data, cv2.COLOR_BGR2GRAY)
    checkerboard_size = (5, 5)
    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    checkerboard_found, corners = cv2.findChessboardCorners(gray_data, checkerboard_size, 
                                                           None, cv2.CALIB_CB_ADAPTIVE_THRESH)

    if checkerboard_found:
        # 亚像素级角点优化
        corners_refined = cv2.cornerSubPix(gray_data, corners, (5, 5), (-1, -1), refine_criteria)
        
        # 计算棋盘格角点在相机坐标系下的3D坐标
        cam_pts = []
        for corner in corners_refined:
            u, v = corner[0]
            z = camera_depth_img[int(v)][int(u)]  # 深度值（相机坐标系z）
            if z == 0:  # 忽略无效深度
                continue
            # 用相机内参将像素坐标转换为相机坐标系3D坐标
            x = (u - robot.cam_intrinsics[0][2]) * z / robot.cam_intrinsics[0][0]
            y = (v - robot.cam_intrinsics[1][2]) * z / robot.cam_intrinsics[1][1]
            cam_pts.append([x, y, z])
        
        if len(cam_pts) == len(checkerboard_corners):  # 确保角点数量匹配
            # 记录当前工具在基坐标系的位姿（4x4矩阵）
            tool_pose = robot.get_current_pose()  # 假设UR_Robot类有此方法，返回4x4变换矩阵
            measured_tool_poses.append(tool_pose)
            observed_checkerboard_pts.append(np.array(cam_pts))

            # 可视化并保存图像
            vis = cv2.drawChessboardCorners(bgr_color_data, checkerboard_size, corners_refined, checkerboard_found)
            cv2.imwrite(f'calib_{calib_pt_idx:06d}.png', vis)
            cv2.imshow('Calibration', vis)
            cv2.waitKey(500)

cv2.destroyAllWindows()

# 转换为numpy数组
measured_tool_poses = np.array(measured_tool_poses)
observed_checkerboard_pts = np.array(observed_checkerboard_pts)

# ---------------------- 核心：求解相机到工具的变换矩阵 T_tool_cam ----------------------
def get_cam_to_tool_transform_error(params):
    """误差函数：最小化坐标变换误差，求解T_tool_cam"""
    # params: 旋转向量（3参数）+ 平移向量（3参数）
    rvec = params[:3]
    tvec = params[3:]
    R_tool_cam, _ = cv2.Rodrigues(rvec)  # 旋转矩阵
    T_tool_cam = np.eye(4)
    T_tool_cam[:3, :3] = R_tool_cam
    T_tool_cam[:3, 3] = tvec

    total_error = 0.0
    for i in range(len(measured_tool_poses)):
        T_base_tool = measured_tool_poses[i]  # 工具在基坐标系的位姿
        P_cam = observed_checkerboard_pts[i]  # 棋盘格在相机坐标系的坐标
        P_base_gt = checkerboard_corners  # 棋盘格在基坐标系的真实坐标（固定）

        # 坐标变换链：P_base_pred = T_base_tool × T_tool_cam × P_cam
        P_cam_hom = np.hstack([P_cam, np.ones((len(P_cam), 1))])  # 齐次坐标 (N,4)
        P_tool_pred = (T_tool_cam @ P_cam_hom.T).T  # 相机→工具 (N,4)
        P_base_pred = (T_base_tool @ P_tool_pred.T).T  # 工具→基坐标系 (N,4)

        # 计算预测值与真实值的误差（忽略齐次项）
        error = np.sum((P_base_pred[:, :3] - P_base_gt) **2)
        total_error += error
    return total_error

# 优化求解T_tool_cam
print('Calibrating camera to tool transform...')
initial_params = np.zeros(6)  # 初始猜测：无旋转和平移
optim_result = optimize.minimize(get_cam_to_tool_transform_error, initial_params, method='Nelder-Mead')
best_params = optim_result.x

# 转换为4x4变换矩阵
rvec = best_params[:3]
tvec = best_params[3:]
R_tool_cam, _ = cv2.Rodrigues(rvec)
T_tool_cam = np.eye(4)
T_tool_cam[:3, :3] = R_tool_cam
T_tool_cam[:3, 3] = tvec

# 保存结果
np.savetxt('cam_to_tool_pose.txt', T_tool_cam, delimiter=' ')
print(f'Done! Camera to tool transform saved to cam_to_tool_pose.txt\n{T_tool_cam}')