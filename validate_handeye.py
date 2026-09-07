import argparse
import time

import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot, _rotation_vector_to_matrix
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='眼在手标定一致性验证: 固定红色圆柱, 手动示教换 6~8 个姿态, '
                    '按 Enter 记录, 用跨姿态换算出的基座坐标离散度判断 cam2end 是否可靠。'
                    '脚本不会主动移动机械臂。')
    parser.add_argument('--ip', default='192.168.1.35', help='机械臂 IP')
    parser.add_argument('--cam2end', default='cam2end_20260906.txt',
                        help='相机到末端标定矩阵路径 (cam->end, mm)')
    parser.add_argument('--cam-ini', default='camera_20260906.ini',
                        help='相机内参/畸变标定文件路径(.ini)；留空则用出厂内参且不去畸变')
    parser.add_argument('--save', default=None, help='采集结果保存路径(.npz), 默认自动命名')
    parser.add_argument('--load', default=None,
                        help='只分析已保存的 .npz, 不连接机器人与相机')
    parser.add_argument('--frames', type=int, default=5,
                        help='每个姿态 Enter 时采样的帧数(取中值), 默认 5')
    parser.add_argument('--min-pose', type=int, default=6,
                        help='分析建议所需最少姿态数, 默认 6')
    return parser.parse_args()


def pose_vector_to_matrix(pose):
    T = np.eye(4)
    T[:3, :3] = _rotation_vector_to_matrix(pose[3:6])
    T[:3, 3] = np.asarray(pose[0:3], dtype=np.float64) * 1000.0
    return T


def pixel_to_base(u, v, z_mm, tcp_pose, K, T_cam2end, dist=None):
    """像素+深度(mm)+tcp位姿 -> 基座坐标(mm)。与 UR_Robot.pixel_to_base 等价的离线复算。
    dist 非空时先对像素做去畸变点级校正(与 UR_Robot 一致)。"""
    if dist is not None:
        pts = cv2.undistortPoints(
            np.array([[[float(u), float(v)]]], dtype=np.float32),
            K, dist, P=K)
        u, v = float(pts[0, 0, 0]), float(pts[0, 0, 1])
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    x = (u - cx) * z_mm / fx
    y = (v - cy) * z_mm / fy
    p_cam = np.array([x, y, z_mm, 1.0])
    p_base = (pose_vector_to_matrix(tcp_pose) @ T_cam2end @ p_cam)[:3]
    return p_base


def capture_center(detector, robot, n_frames):
    us, vs, zs = [], [], []
    for _ in range(n_frames):
        color, depth = robot.get_camera_data()
        res = detector.detect(color, depth.astype(np.float64))
        if res is not None and res['z_mm'] is not None:
            u, v = res['center']
            us.append(u)
            vs.append(v)
            zs.append(res['z_mm'])
    if len(us) < 3:
        return None
    return int(np.median(us)), int(np.median(vs)), float(np.median(zs))


def analyze(tcp_poses, centers, zs, K, T_cam2end, min_pose, dist=None):
    n = len(tcp_poses)
    base_pts = np.array([pixel_to_base(centers[i][0], centers[i][1], zs[i],
                                       tcp_poses[i], K, T_cam2end, dist=dist)
                         for i in range(n)])
    centroid = base_pts.mean(axis=0)
    res = base_pts - centroid
    d3 = np.linalg.norm(res, axis=1)
    rms = float(np.sqrt(np.mean(d3 ** 2)))
    axis_rms = np.sqrt(np.mean(res ** 2, axis=0))

    print('\n================ 分析结果 ================')
    print('姿态数: %d' % n)
    print('所有姿态换算出的基座坐标均值(mm): %s' % np.round(centroid, 2))
    print('%-4s %-38s %-10s %s' % ('#', 'TCP 位姿 (m, 旋转向量)', '3D残差mm', '基座点 mm'))
    for i in range(n):
        print('%-4d %-38s %-10.2f %s' % (
            i + 1, np.round(np.asarray(tcp_poses[i]), 4), d3[i],
            np.round(base_pts[i], 2)))
    print('------------------------------------------')
    print('3D RMS = %.2f mm   最大偏差 = %.2f mm' % (rms, float(d3.max())))
    print('各轴 RMS (x, y, z) = %s mm' % np.round(axis_rms, 2))
    print('------------------------------------------')
    if n < min_pose:
        print('警告: 姿态数偏少(%d < %d), 以下结论仅供参考。'
              % (n, min_pose))
    if rms < 3.0:
        print('判定: 跨姿态一致性优良 (RMS<3mm), 建议可进入抓取测试。')
    elif rms <= 6.0:
        print('判定: 跨姿态一致性可接受 (3~6mm)。')
    else:
        print('判定: 跨姿态一致性差 (>6mm), 建议重做眼在手标定。')
    if n >= min_pose:
        print('提示: 若偏差随机械臂朝向(尤其绕竖直轴转动)明显变化,'
              '多为 T_cam2end 旋转项不准; 若恒定则多为平移项/深度比例问题。')
    print('提示: 本测试只证明内部一致性, 无法发现所有姿态共有的整体偏移。'
          '正式抓取前建议用探针/卷尺对点一次作绝对校验。')
    return {'rms': rms, 'max': float(d3.max()), 'centroid': centroid}


def main():
    args = parse_args()

    if args.load:
        d = np.load(args.load)
        print('载入采集文件:', args.load)
        dist = d['dist'] if 'dist' in d.files else None
        analyze(d['tcp'], d['uv'], d['z'], d['K'], d['T_cam2end'],
                args.min_pose, dist=dist)
        return

    robot = UR_Robot(robot_ip=args.ip, is_use_robot=True, is_use_camera=True,
                     cam2end_path=args.cam2end, cam_ini_path=args.cam_ini)
    detector = RedCylinderDetector()
    K = np.array(robot.cam_intrinsics, dtype=np.float64)
    T_cam2end = robot.T_cam2end
    dist = robot.dist_calib

    win = 'validate'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    print('验证采集开始: 固定红色圆柱不动, 手动示教机械臂换姿态。')
    print('  每个姿态按 Enter 记录 (画面内须识别到圆柱顶心, 深度有效)')
    print('  建议 6~8 个姿态, 其中 2 个绕竖直轴相差约 180°')
    print('  q/ESC 结束采集并分析。脚本不会移动机械臂。')

    tcp_list, uv_list, z_list = [], [], []
    try:
        while True:
            color, depth = robot.get_camera_data()
            res = detector.detect(color, depth.astype(np.float64))
            img = detector.draw(color, res)
            info = 'n=%d  Enter:记录  q:结束分析' % len(tcp_list)
            cv2.putText(img, info, (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
            cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (13, ord('r')):
                if res is None or res['z_mm'] is None:
                    print('当前未识别到圆柱(或深度无效), 未记录, 请调整后重试。')
                    continue
                tcp_now = np.array(robot.get_actual_tcp_pose(), dtype=np.float64)
                med = capture_center(detector, robot, args.frames)
                if med is None:
                    print('多次采样仍无法稳定识别, 未记录。')
                    continue
                u, v, z = med
                tcp_list.append(tcp_now)
                uv_list.append([u, v])
                z_list.append(z)
                print('[%d] 已记录: px=(%d,%d) z=%.1fmm  tcp=%s'
                      % (len(tcp_list), u, v, z, np.round(tcp_now, 4)))
    finally:
        cv2.destroyAllWindows()

    if len(tcp_list) == 0:
        print('未采集到任何姿态, 结束。')
        return

    save_path = args.save or ('handeye_poses_%s.npz'
                              % time.strftime('%Y%m%d_%H%M%S'))
    if dist is None:
        dist = np.zeros((1, 5), dtype=np.float64)
    np.savez(save_path, tcp=np.asarray(tcp_list), uv=np.asarray(uv_list),
             z=np.asarray(z_list), K=K, T_cam2end=T_cam2end, dist=dist)
    print('采集结果已保存: %s' % save_path)

    analyze(np.asarray(tcp_list), np.asarray(uv_list), np.asarray(z_list),
            K, T_cam2end, args.min_pose, dist=dist)


if __name__ == '__main__':
    main()
