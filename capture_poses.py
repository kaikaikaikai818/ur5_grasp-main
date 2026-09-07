import argparse
import os
import time

import cv2
import numpy as np

from bsp.robot_bsp.UR_Robot import UR_Robot
from bsp.camera_bsp.cylinder_detect import RedCylinderDetector


def parse_args():
    parser = argparse.ArgumentParser(
        description='采集调试快照: 保存 color/depth/tcp 及当前检测结果, 用于分析斜视角顶面检测偏差。'
                    '脚本不会主动移动机械臂。')
    parser.add_argument('--ip', default='192.168.1.35', help='机械臂 IP')
    parser.add_argument('--cam2end', default='cam2end_20260906.txt')
    parser.add_argument('--cam-ini', default='camera_20260906.ini')
    parser.add_argument('--out', default=None,
                        help='输出文件夹名, 默认 oblique_poses_<时间戳>')
    return parser.parse_args()


def main():
    args = parse_args()
    robot = UR_Robot(robot_ip=args.ip, is_use_robot=True, is_use_camera=True,
                     cam2end_path=args.cam2end, cam_ini_path=args.cam_ini)
    detector = RedCylinderDetector()

    out = args.out or ('oblique_poses_%s' % time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    print('快照保存到: %s' % out)
    print('操作: 移动机械臂到姿态后按 Enter 保存该帧 (color/depth/tcp); q 退出')

    win = 'capture'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    idx = 0
    try:
        while True:
            color, depth = robot.get_camera_data()
            depth_mm = depth.astype(np.float64)
            res = detector.detect(color, depth_mm)
            img = detector.draw(color, res)
            info = 'n=%d  Enter:save  q:quit' % idx
            cv2.putText(img, info, (10, 470), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 1)
            cv2.imshow(win, img)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (13, ord('r')):
                idx += 1
                base = os.path.join(out, 'snap_%03d' % idx)
                cv2.imwrite(base + '_color.png', color)
                np.save(base + '_depth.npy', depth)
                tcp = np.array(robot.get_actual_tcp_pose(), dtype=np.float64)
                with open(os.path.join(out, 'poses.txt'), 'a') as f:
                    f.write('%s\n' % ' '.join('%.6f' % x for x in tcp))
                if res is not None:
                    print('[%d] saved px=%s z=%s tcp=%s' % (
                        idx, res['center'], res['z_mm'], np.round(tcp, 4)))
                else:
                    print('[%d] saved (未识别到圆柱) tcp=%s' % (idx, np.round(tcp, 4)))
    finally:
        cv2.destroyAllWindows()
    print('完成, 数据目录: %s' % out)


if __name__ == '__main__':
    main()
