import numpy as np

from . import camera_calibration
from . import thresholding
from . import transform
from . import find_line


def process_img(raw_img: np.ndarray,
                trans_m: np.ndarray, trans_mint: np.ndarray,
                object_points: np.ndarray = np.empty(0), img_points: np.ndarray = np.empty(0),
                apply_calibrate_distort=True
                ) -> np.ndarray:
    """
    Mark the lane line on the single picture
    """
    if apply_calibrate_distort:
        binary_img = camera_calibration.calibrate_distort(raw_img, object_points, img_points)  # 矫正畸变
    else:
        binary_img = raw_img
    binary_img = thresholding.thresholding(binary_img)  # 阈值化
    binary_img = transform.transform_img(binary_img, trans_m)  # 转俯视图
    l_fit, r_fit, *_ = find_line.find_line(binary_img)  # 拟合车道线
    ret_img = find_line.draw_area(raw_img, binary_img, trans_mint, l_fit, r_fit)  # 给车道上色
    ret_img = find_line.draw_values(ret_img,
                                    *find_line.calculate_curv_and_pos(raw_img, l_fit, r_fit))  # 图片标注信息
    return ret_img
