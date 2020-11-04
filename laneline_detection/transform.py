import cv2
import numpy as np


def get_m_minv() -> [np.ndarray, np.ndarray]:
    """
    获取正向/逆向透视变换矩阵

    @return: (正变换矩阵, 逆变换矩阵)
    """
    # 定义对应的四边形顶点坐标
    src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])
    dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])
    m = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    return m, minv


def transform_img(img: np.ndarray, m: np.ndarray) -> np.ndarray:
    """
    应用透视变换
    """
    return cv2.warpPerspective(img, m, img.shape[1::-1])
