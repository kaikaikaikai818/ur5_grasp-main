import argparse
import time
import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='D435 眼在手实时检测：识别红色圆柱顶面圆心并实时换算成机器人基座坐标')
    parser.add_argument('--ip', default='192.168.1.35', help='机械臂 IP')
    parser.add_argument('--cam2end', default='cam2end_20260906.txt',
                        help='相机到末端标定矩阵路径 (cam->end, mm)')
    parser.add_argument('--cam-ini', default='camera_20260906.ini',
                        help='相机内参/畸变标定文件路径(.ini)；留空则用出厂内参且不去畸变')
    parser.add_argument('--tcp-pose', default=None,
                        help='离线模式：用固定位姿 "x y z rx ry rz"(m) 换算，不连接机器人')
    parser.add_argument('--tune', action='store_true',
                        help='弹 HSV 滑条窗口实时调红色阈值')
    parser.add_argument('--save', default=None, help='按 s 保存图像时的路径前缀')
    return parser.parse_args()


def main():
    args = parse_args()

    offline_pose = None
    if args.tcp_pose:
        offline_pose = np.array([float(v) for v in args.tcp_pose.split()])
        print('离线位姿(m):', offline_pose)

    robot = UR_Robot(robot_ip=args.ip, is_use_robot=True, is_use_camera=True,
                     connect_robot=offline_pose is None, cam2end_path=args.cam2end,
                     cam_ini_path=args.cam_ini)
    detector = RedCylinderDetector()

    win = 'detect'
    ctrl = None
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    if args.tune:
        ctrl = 'control'
        cv2.namedWindow(ctrl, cv2.WINDOW_AUTOSIZE)
        detector.setup_trackbars(ctrl)

    print('实时检测中... (q/ESC 退出, c 打印当前基座坐标, s 保存图像)')
    last_print = 0.0
    while True:
        color, depth = robot.get_camera_data()
        depth_mm = depth.astype(np.float64)

        if args.tune:
            detector.ranges_from_trackbars(ctrl)

        res = detector.detect(color, depth_mm)
        img = detector.draw(color, res)

        if res is not None:
            u, v = res['center']
            z = res['z_mm']
            cv2.putText(img, 'px(%d,%d)' % (u, v), (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            if z is not None:
                base_mm, base_m = robot.pixel_to_base(u, v, z, tcp_pose=offline_pose)
                cv2.putText(img, 'base mm x=%.1f y=%.1f z=%.1f' % tuple(base_mm),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow(win, img)
        if args.tune:
            cv2.imshow('mask', detector._red_mask(color))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c') and res is not None and res['z_mm'] is not None:
            u, v = res['center']
            base_mm, base_m = robot.pixel_to_base(u, v, res['z_mm'], tcp_pose=offline_pose)
            now = time.time()
            if now - last_print > 0.3:
                print('px=(%d,%d)  z=%.1fmm  base_mm=%s  base_m=%s'
                      % (u, v, res['z_mm'], np.round(base_mm, 2), np.round(base_m, 4)))
                last_print = now
        elif key == ord('s') and res is not None:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            path = (args.save or 'snapshot') + '_' + stamp + '.png'
            cv2.imwrite(path, img)
            print('已保存:', path)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
