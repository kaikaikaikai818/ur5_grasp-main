import time
import os
import cv2
import configparser
import rtde_control
import rtde_receive
import minimalmodbus
import copy
import numpy as np
import threading
import math

from ..camera_bsp.realsenseD415 import Camera


def load_camera_ini(path):
    """读取标定生成的 camera_*.ini，返回 (K3x3, dist1x5)；失败返回 None。"""
    if not os.path.exists(path):
        print('未找到相机标定文件: %s (回退到 RealSense 出厂内参, 不去畸变)' % path)
        return None
    cfg = configparser.ConfigParser()
    cfg.read(path)
    try:
        fx = float(cfg.get('camera_parameters', 'fx'))
        fy = float(cfg.get('camera_parameters', 'fy'))
        cx = float(cfg.get('camera_parameters', 'cx'))
        cy = float(cfg.get('camera_parameters', 'cy'))
        k1 = float(cfg.get('distortion_parameters', 'k1'))
        k2 = float(cfg.get('distortion_parameters', 'k2'))
        p1 = float(cfg.get('distortion_parameters', 'p1', fallback=0.0))
        p2 = float(cfg.get('distortion_parameters', 'p2', fallback=0.0))
        k3 = float(cfg.get('distortion_parameters', 'k3', fallback=0.0))
    except Exception as e:
        print('解析相机标定文件失败: %s (%s)，回退到出厂内参' % (path, e))
        return None
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([[k1, k2, p1, p2, k3]], dtype=np.float64)
    return K, dist


lock = threading.Lock()

# 夹爪 Modbus 寄存器地址（与 bsp/grasp_bsp/grasp_claw.py 一致）
POSITION_HIGH_8 = 0x0102
POSITION_LOW_8 = 0x0103
SPEED = 0x0104
FORCE = 0x0105
MOTION_TRIGGER = 0x0108


def _rotation_vector_to_matrix(rvec):
    """Rodrigues: 旋转向量 -> 3x3 旋转矩阵"""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rvec / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


class UR_Robot:
    def __init__(self, robot_ip="192.168.1.35", workspace_limits=None, is_use_robot=True,
                 is_use_camera=True, connect_robot=True, cam2end_path="cam2end_20260906.txt",
                 camera_serial="215222074676", cam_ini_path="camera_20260906.ini",
                 undistort_img=False, is_use_gripper=True, gripper_port="COM10",
                 gripper_baudrate=115200, gripper_address=1):
        if workspace_limits is None:
            #workspace_limits = [[-0.450, -0.200], [-0.65, -0.47], [0.003, 0.45]]
            workspace_limits = [[-1, 1], [-1, 1], [0.003, 0.9]]
        self.workspace_limits = workspace_limits
        self.connect_robot = connect_robot
        self.is_use_robotiq85 = is_use_robot
        self.is_use_camera = is_use_camera
        if connect_robot:
            self.rtde_c = rtde_control.RTDEControlInterface(robot_ip)
            self.rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        else:
            self.rtde_c = None
            self.rtde_r = None


        self.joint_acc = 0.2  
        self.joint_spd = 0.2  

        self.tool_acc = 0.1  
        self.tool_spd = 0.1 
        self.tool_pose_tolerance = [0.002, 0.002, 0.002, 0.01, 0.01, 0.01]
        self.joint_tolerance = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

        # self.home_joint_config = [-(0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
        #                      (0 / 360.0) * 2 * np.pi, -(90 / 360.0) * 2 * np.pi,
        #                      -(0 / 360.0) * 2 * np.pi, 0.0]
        self.initial_pose = [-0.4, -0.025, 0.14981, 0.000, 3.141, 0.000]
        self.home_joint_config = [0.0, -(90 / 360.0) * 2 * np.pi, 0.0,
                                  -(90 / 360.0) * 2 * np.pi, 0.0, 0.0]

        # -------------------------- 夹爪（Modbus RTU，单独初始化，失败不影响机器人） --------------------------
        self.is_use_gripper = is_use_gripper
        self.gripper_port = gripper_port
        self.gripper_baudrate = gripper_baudrate
        self.gripper_address = gripper_address
        self.instrument = None
        if self.is_use_gripper:
            self.init_gripper()

        if(self.is_use_camera):
            # Fetch RGB-D data from RealSense camera
            self.camera = Camera(serial=camera_serial)
            self.cam_intrinsics = self.camera.intrinsics
            self.depth_scale = self.camera.scale
            # 眼在手：相机到末端齐次变换 (cam->end，平移单位 mm)
            self.cam2end_path = cam2end_path
            self.T_cam2end = np.loadtxt(self.cam2end_path, delimiter=' ')
            print('T_cam2end (cam->end, translation in mm):')
            print(self.T_cam2end)
            # 标定内参/畸变：仅用于像素点级校正(undistort_points)，不改动整幅画面
            self.und_map1 = None
            self.und_map2 = None
            self.dist_calib = None
            calib = load_camera_ini(cam_ini_path)
            if calib is not None:
                K_calib, dist_calib = calib
                self.dist_calib = dist_calib
                self.cam_intrinsics = K_calib
                print('已加载标定内参:')
                print(self.cam_intrinsics)
                print('dist =', dist_calib.tolist())
                # 整幅图去畸变默认关闭(会改变画面观感)，仅当 undistort_img=True 时启用
                if undistort_img:
                    w, h = self.camera.im_width, self.camera.im_height
                    self.und_map1, self.und_map2 = cv2.initUndistortRectifyMap(
                        K_calib, dist_calib, None, K_calib, (w, h), cv2.CV_32FC1)
                    print('已启用整幅图去畸变(undistort_img=True)')
        
# Define the robot control class
    def moveL(self, target_pose, speed=0.05, acceleration=0.05):
        self.rtde_c.moveL(target_pose, speed, acceleration)
        actual_tool_positions = self.get_actual_tcp_pose()
        while not all([np.abs(actual_tool_positions[j] - target_pose[j]) < self.tool_pose_tolerance[j] for j in range(3)]):
            actual_tool_positions = self.get_actual_tcp_pose()
            time.sleep(0.01)
        time.sleep(1.5)  

    def moveJ(self, target_joint, speed=0.05, acceleration=0.05):
        self.rtde_c.moveJ(target_joint, speed, acceleration)
        actual_joint_positions = self.get_actual_joint_position()
        while not all([np.abs(actual_joint_positions[j] - target_joint[j]) < self.joint_tolerance[j] for j in range(len(target_joint))]):
            actual_joint_positions = self.get_actual_joint_position()
            time.sleep(0.01)
        time.sleep(1.5)

    def go_home(self):
        self.moveJ(self.home_joint_config)

    def get_actual_tcp_pose(self):
        return self.rtde_r.getActualTCPPose()

    def get_actual_joint_position(self):
        return self.rtde_r.getActualQ()

    def get_robot_status(self):
        return self.rtde_r.getRobotStatus()

    # -------------------------- 夹爪（Modbus RTU） --------------------------
    def init_gripper(self):
        """单独初始化夹爪，失败时仅提示，不阻断机器人"""
        if not self.is_use_gripper:
            print("未启用夹爪，跳过初始化")
            return
        try:
            self.instrument = minimalmodbus.Instrument(
                port=self.gripper_port, slaveaddress=self.gripper_address)
            self.instrument.serial.baudrate = self.gripper_baudrate
            self.instrument.serial.timeout = 1
            self.read_position()
            print("夹爪初始化成功（端口：%s）" % self.gripper_port)
        except Exception as e:
            print("夹爪初始化失败（不影响机器人）：%s" % e)
            self.instrument = None

    def write_position_high8(self, value):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_register(POSITION_HIGH_8, value, functioncode=6)

    def write_position_low8(self, value):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_register(POSITION_LOW_8, value, functioncode=6)

    def write_position(self, value):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_long(POSITION_HIGH_8, value)

    def write_speed(self, speed):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_register(SPEED, speed, functioncode=6)

    def write_force(self, force):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_register(FORCE, force, functioncode=6)

    def trigger_motion(self):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return
        with lock:
            self.instrument.write_register(MOTION_TRIGGER, 1, functioncode=6)

    def read_position(self):
        if self.instrument is None:
            print("夹爪未初始化，无法执行操作")
            return -1
        with lock:
            high = self.instrument.read_register(POSITION_HIGH_8, functioncode=3)
            low = self.instrument.read_register(POSITION_LOW_8, functioncode=3)
            return (high << 8) | low

    def read_torque_reached(self):
        """读 0x0601 力矩到达: 1=到达(夹到物体), 0=未到达; 失败返回 -1"""
        if self.instrument is None:
            print("夹爪未初始化，无法读取力矩")
            return -1
        try:
            with lock:
                return self.instrument.read_register(0x0601, functioncode=3)
        except Exception as e:
            print("读取力矩到达失败: %s" % e)
            return -1

    def read_torque_current(self):
        """读 0x060C 实时力矩/电流: 数值越大越夹紧; 失败返回 -1"""
        if self.instrument is None:
            print("夹爪未初始化，无法读取实时力矩")
            return -1
        try:
            with lock:
                return self.instrument.read_register(0x060C, functioncode=3)
        except Exception as e:
            print("读取实时力矩失败: %s" % e)
            return -1

    def grip(self, position, speed, force):
        if self.instrument is None:
            print("夹爪未初始化，无法执行抓取")
            return -1
        self.write_position(position)
        self.write_speed(speed)
        self.write_force(force)
        self.trigger_motion()
        time.sleep(2)
        return self.read_position()

    def pose_vector_to_matrix(self, pose):
        """[x,y,z,rx,ry,rz] (单位 m，旋转向量) -> 4x4 齐次矩阵 (平移单位 mm)"""
        T = np.eye(4)
        T[:3, :3] = _rotation_vector_to_matrix(pose[3:6])
        T[:3, 3] = np.asarray(pose[0:3], dtype=np.float64) * 1000.0
        return T

    def pixel_to_camera(self, u, v, z_mm):
        """像素 + 深度(mm) -> 相机系 3D 坐标 (mm)。
        若加载了标定畸变，先用 undistort_points 把像素校正为理想坐标再反投影
        (画面不做整幅去畸变，仅做点级校正)。"""
        K = self.cam_intrinsics
        if self.dist_calib is not None:
            pts = cv2.undistortPoints(
                np.array([[[float(u), float(v)]]], dtype=np.float32),
                K, self.dist_calib, P=K)
            u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        x = (u - cx) * z_mm / fx
        y = (v - cy) * z_mm / fy
        return np.array([x, y, z_mm], dtype=np.float64)

    def camera_to_base(self, p_cam_mm, tcp_pose=None):
        """相机系 3D 点(mm) -> 机器人基座坐标，返回 (mm, m)。
        P_base = T_end2base @ T_cam2end @ P_cam
        tcp_pose 缺省时实时读取当前位姿"""
        if tcp_pose is None:
            tcp_pose = self.get_actual_tcp_pose()
        T_end2base = self.pose_vector_to_matrix(tcp_pose)
        p_cam_h = np.concatenate([p_cam_mm, [1.0]])
        p_base_mm = (T_end2base @ self.T_cam2end @ p_cam_h)[:3]
        return p_base_mm, p_base_mm / 1000.0

    def pixel_to_base(self, u, v, z_mm, tcp_pose=None):
        """像素 + 深度(mm) -> 机器人基座坐标，返回 (mm, m)"""
        p_cam = self.pixel_to_camera(u, v, z_mm)
        return self.camera_to_base(p_cam, tcp_pose=tcp_pose)

    ## get camera data
    def get_camera_data(self):
        color_img, depth_img = self.camera.get_data()
        if self.und_map1 is not None:
            color_img = cv2.remap(color_img, self.und_map1, self.und_map2,
                                  cv2.INTER_LINEAR)
            depth_img = cv2.remap(depth_img, self.und_map1, self.und_map2,
                                  cv2.INTER_NEAREST)
        return color_img, depth_img
    
    def angle_to_cartesian(self,angle_degrees):
    # 将角度从度转换为弧度
        angle_radians = math.radians(angle_degrees)
    # 计算x和y坐标
        rx = math.cos(angle_radians)
        ry = math.sin(angle_radians)

        return rx, ry 




    def grasp(self, position,angle,close_position=5000, k_acc=0.1, k_vel=0.1, speed=100, force=40):

        open_position=0000

        rpy = [0,3.141,0]
        for i in range(3):
            position[i] = min(max(position[i], self.workspace_limits[i][0]), self.workspace_limits[i][1])
        # 判定抓取的角度RPY是否在规定范围内 [0.5*pi,1.5*pi]

        print('Executing: grasp at (%f, %f, %f)' \
              % (position[0], position[1], position[2]))


        # pre work
        grasp_home =[-0.4, -0.025, 0.15, 0.000, 3.141, 0.000]   #初始位置
        self.moveL(grasp_home, k_acc, k_vel)
        self.grip(open_position, speed, force)
        self.read_position()

        # Firstly, achieve pre-grasp position
        pre_position = copy.deepcopy(position)
        pre_position[2] = pre_position[2] + 0.05  # z axis + tcp
        print(pre_position)
        self.moveL(pre_position + rpy, k_acc, k_vel)

        # Second，achieve higher positiao
        air_joint = self.get_actual_joint_position()
        air_joint[5] = air_joint[5] + angle+1.57
        self.moveJ(air_joint,0.4,0.4)
        air_position  = self.get_actual_tcp_pose()

        #Third,grasp
        grasp_position = air_position
        grasp_position[2] = grasp_position[2]-0.096
        self.moveL(grasp_position, k_acc, k_vel)
        self.grip(close_position, speed, force)
        self.moveL(air_position , k_acc, k_vel)

        # Third,put the object into box
        box_position = [-0.4, -0.4, 0.03, 0, 3.141, 0]  # 末端位置
        self.moveL(box_position, k_acc, k_vel)
        # box_position[2] = 0.1  # down to the 10cm
        # self.moveL(box_position, k_acc, k_vel)
        self.grip(open_position, speed, force)
        box_position[2] = 0.1
        self.moveL(box_position, k_acc, k_vel)
        self.moveL(grasp_home, k_acc, k_vel)
        print("grasp success!")

