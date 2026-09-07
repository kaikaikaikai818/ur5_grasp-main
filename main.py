from bsp.robot_bsp.UR_Robot import UR_Robot

def main():
    # ① 创建机器人实例（连接机械臂）
    robot = UR_Robot(
        robot_ip="192.168.1.35",   # 机械臂的 IP，和类里默认值一致
        is_use_robot=True,         # 使用机械臂
        is_use_camera=False        # 使用相机（会初始化 RealSense）
    )

    

    # ② 读取当前位姿
    pose = robot.get_actual_tcp_pose()
    print("当前 TCP 位姿：", pose)
    pose[2]+=0.05
    robot.moveL(pose)
    print("移动后 TCP 位姿：", pose)

    
    # ③ 关节运动（在当前位置上微调第 1 个关节）
    joint = robot.get_actual_joint_position()
    print("当前 JOINT 位姿：", joint)
    
    joint[3] -= 0.10
    robot.moveJ(joint)
    joint = robot.get_actual_joint_position()
    print("当前 JOINT2 位姿：", joint)

    '''
    # ④ 直线运动到指定位置 [x, y, z, rx, ry, rz]
    target = [-0.4, -0.025, 0.15, 0.0, 3.141, 0.0]
    robot.moveL(target)
    '''
    # ⑤ 抓取
    #robot.grasp([0.0, 0.0, 0.1], angle=0.0)

    # ⑥ 相机取图
    #if robot.is_use_camera:
    #    color, depth = robot.get_camera_data()
    #    print("彩色图尺寸：", color.shape)

if __name__ == "__main__":
    main()
