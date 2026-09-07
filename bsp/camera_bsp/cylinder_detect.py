import cv2
import numpy as np


class RedCylinderDetector:
    def __init__(self, hsv_ranges=None, morph_kernel=(5, 5), min_area=500,
                 depth_tol_mm=15.0, depth_roi=5):
        # 红色在 HSV 色相上跨越 0° 与 180°，需要两个区间；S/V 阈值现场可用 tune() 调整
        if hsv_ranges is None:
            hsv_ranges = [
                ((0, 120, 60), (10, 255, 255)),
                ((170, 120, 60), (180, 255, 255)),
            ]
        self.hsv_ranges = hsv_ranges
        self.min_area = min_area
        self.depth_tol_mm = depth_tol_mm
        self.depth_roi = depth_roi
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel)

    def _red_mask(self, bgr):
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for low, high in self.hsv_ranges:
            lower = cv2.inRange(hsv, np.asarray(low), np.asarray(high))
            mask = cv2.bitwise_or(mask, lower)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        return mask

    def _query_depth(self, depth_mm, u, v, cap):
        r = self.depth_roi // 2
        h, w = depth_mm.shape
        x1, y1 = max(0, u - r), max(0, v - r)
        x2, y2 = min(w, u + r), min(h, v + r)
        patch = depth_mm[y1:y2, x1:x2]
        region = cap[y1:y2, x1:x2]
        vals = patch[region > 0]
        vals = vals[vals > 0]
        if vals.size < 3:
            return None
        return float(np.median(vals))

    def detect(self, bgr, depth_mm):
        """识别竖直立放红色圆柱的顶面圆心。

        竖直圆柱顶面是离相机最近的端面，故取红色连通域内深度接近最小值的像素子集，
        对其求质心即顶面圆心；纯正俯视时该子集即整个轮廓。
        返回 dict {center:(u,v), z_mm, area} 或 None。
        """
        mask = self._red_mask(bgr)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area < self.min_area:
            return None

        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, -1)

        cap = blob
        valid = depth_mm > 0
        if np.any(valid & (blob > 0)):
            d_min = float(np.percentile(depth_mm[valid & (blob > 0)], 5))
            sel = (blob > 0) & valid & (depth_mm <= d_min + self.depth_tol_mm)
            if np.count_nonzero(sel) >= 50:
                cap = np.zeros_like(blob)
                cap[sel] = 255

        m = cv2.moments(cap)
        if m['m00'] < 1e-6:
            x, y, w, h = cv2.boundingRect(contour)
            u, v = int(x + w / 2), int(y + h / 2)
        else:
            u, v = int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])

        z_mm = self._query_depth(depth_mm, u, v, cap)
        return {'center': (u, v), 'z_mm': z_mm, 'area': area}

    def draw(self, bgr, res, color=(0, 0, 255)):
        img = bgr.copy()
        if res is None:
            return img
        u, v = res['center']
        cv2.circle(img, (u, v), 6, color, 2)
        if res['z_mm'] is not None:
            txt = "z=%.1fmm" % res['z_mm']
            cv2.putText(img, txt, (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        return img

    def setup_trackbars(self, win):
        """在窗口 win 上创建 HSV 阈值滑条（基于当前 self.hsv_ranges）。"""
        r = self.hsv_ranges

        def _bar(name, init, high):
            cv2.createTrackbar(name, win, int(init), int(high), lambda x: None)

        _bar('h0_lo', r[0][0][0], 180)
        _bar('h0_hi', r[0][1][0], 180)
        _bar('h1_lo', r[1][0][0], 180)
        _bar('h1_hi', r[1][1][0], 180)
        _bar('s_lo', r[0][0][1], 255)
        _bar('s_hi', r[0][1][1], 255)
        _bar('v_lo', r[0][0][2], 255)
        _bar('v_hi', r[0][1][2], 255)

    def ranges_from_trackbars(self, win):
        """从窗口 win 的滑条读取并更新 self.hsv_ranges。"""
        def _get(name):
            return cv2.getTrackbarPos(name, win)

        s_lo, s_hi, v_lo, v_hi = _get('s_lo'), _get('s_hi'), _get('v_lo'), _get('v_hi')
        self.hsv_ranges = [
            ((_get('h0_lo'), s_lo, v_lo), (_get('h0_hi'), s_hi, v_hi)),
            ((_get('h1_lo'), s_lo, v_lo), (_get('h1_hi'), s_hi, v_hi)),
        ]

    def tune(self, bgr, depth_mm, win_color='red_tune', win_mask='red_mask'):
        """弹出 HSV 阈值滑条调参（静态图），按 q/ESC 结束并返回最后一次检测结果。"""
        cv2.namedWindow(win_color, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(win_mask, cv2.WINDOW_AUTOSIZE)
        self.setup_trackbars(win_color)

        res = None
        while True:
            self.ranges_from_trackbars(win_color)
            res = self.detect(bgr, depth_mm)
            cv2.imshow(win_color, self.draw(bgr, res))
            cv2.imshow(win_mask, self._red_mask(bgr))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
        cv2.destroyWindow(win_color)
        cv2.destroyWindow(win_mask)
        return res
