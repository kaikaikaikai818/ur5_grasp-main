#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
眼在手外（hand-out-eye）标定与坐标转换模块
功能：处理固定相机的坐标解算，将像素坐标转换为机器人基系坐标
"""
import numpy as np
import cv2

class HandOutEyeCalibration:
    def __init__(self, robot,calib_path=None,cam_depth_scale = None):
        self.robot = robot
        # 相机内参（实际使用中需替换为标定值）
        self.cam_intrinsics = None
        if cam_depth_scale is not None:
            # 仅处理浮点数/整数（确保传入的是有效数值）
            if isinstance(cam_depth_scale, (int, float)):
                # 检查缩放比例是否为正数（深度缩放不可能≤0）
                if cam_depth_scale > 0:
                    self.cam_depth_scale = float(cam_depth_scale)  # 直接赋值浮点数
                else:
                    raise ValueError("深度缩放比例必须是正数")  # 无效值时报错
            else:
                raise TypeError("cam_depth_scale必须是整数或浮点数")  # 类型错误时报错
        else:
            # 如果未传入，根据需求选择：要么报错提醒，要么用默认值
            # 推荐报错，避免用错缩放比例导致坐标错误
            raise ValueError("必须传入cam_depth_scale（浮点数）")
        self.calib_path = calib_path
        # 相机到机器人基系的变换矩阵（手眼标定结果）
        self.camera2robot_pose = None
        # 工作空间限制
        self.workspace_limits = [
            [-0.5, 0.5],   # X轴范围
            [-0.5, 0.5],   # Y轴范围
            [0.02, 0.5]    # Z轴范围
        ]
        
        if calib_path is None:
            raise ValueError("必须给出 calib_path（4×4 相机→机器人位姿文件）")
        # 优先按 txt 读取
        
        self.camera2robot_pose = np.loadtxt(calib_path)
        if self.camera2robot_pose.shape != (4, 4):
            raise ValueError("标定矩阵不是 4×4")
        

        self._init_parameters()
        
    def _init_parameters(self):
        """从机器人对象初始化相机参数"""
        if hasattr(self.robot, 'cam_intrinsics'):
            self.cam_intrinsics = self.robot.cam_intrinsics

        if hasattr(self.robot, 'workspace_limits'):
            self.workspace_limits = self.robot.workspace_limits
    
    def pixel_to_robot_coords(self, x, y, depth_img):
        """
        将像素坐标转换为机器人基系坐标
        参数:
            x, y: 像素坐标
            depth_img: 深度图像
        返回:
            机器人基系下的三维坐标 [X, Y, Z]，无效时返回None
        """
        if self.cam_intrinsics is None or self.camera2robot_pose is None:
            print("相机参数未初始化，无法进行坐标转换")
            return None
            
        # 获取深度值
        click_z = float(depth_img[y][x]) * self.cam_depth_scale
    
    # 直接判断标量是否有效（无需转为数组）
        if click_z <= 0:
            
            return None
         
        # 相机内参提取
        cx, cy = self.cam_intrinsics[0][2], self.cam_intrinsics[1][2]
        fx, fy = self.cam_intrinsics[0][0], self.cam_intrinsics[1][1]
        
        # 计算相机坐标系下的坐标
        click_x = (x - cx) * click_z / fx
        click_y = (y - cy) * click_z / fy
        click_point_cam = np.asarray([click_x, click_y, click_z])
        click_point_cam.shape = (3, 1)  # 转为列向量
        
        # 转换到机器人基系
        target_position = np.dot(
            self.camera2robot_pose[0:3, 0:3], 
            click_point_cam
        ) + self.camera2robot_pose[0:3, 3:]
        target_position = target_position[0:3, 0]  # 提取X, Y, Z
        
        # 安全检查：限制在工作空间内
        x_min, x_max = self.workspace_limits[0]
        y_min, y_max = self.workspace_limits[1]
        z_min, z_max = self.workspace_limits[2]
        
        target_position[0] = max(x_min, min(target_position[0], x_max))
        target_position[1] = max(y_min, min(target_position[1], y_max))
        target_position[2] = max(z_min, min(target_position[2], z_max))
           
        return target_position