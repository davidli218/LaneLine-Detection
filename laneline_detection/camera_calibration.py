import cv2
import numpy as np


def camera_calibrate(images: [np.ndarray], grid):
    """Generate (Object points, Image points) by ChessboardCorners

    :param images: Chess board images. 10-20 imgs Recommended. Must be an 8-bit color image.
    :param grid: Number of inner corners per a chessboard row and column.
    :return: (Object points, Image points)
    """
    object_points = []
    img_points = []
    chessboard_corners_img = []  # save cv2.drawChessboardCorners()

    for img in images:
        '''Generate object points'''
        object_point = np.zeros((grid[0] * grid[1], 3), np.float32)
        object_point[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, grid, None)

        if ret:
            '''Renders the detected chessboard corners'''
            cv2.drawChessboardCorners(img, grid, corners, ret)
            chessboard_corners_img.append(img)

            object_points.append(object_point)
            img_points.append(corners)

    '''Display ChessboardCorners Images'''
    if True:
        for img in chessboard_corners_img:
            cv2.imshow('FindChessboardCorners', img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    return object_points, img_points


def calibrate_distort(img: np.ndarray, object_points, img_points) -> np.ndarray:
    """
    Calibrate the distorted picture (相机标定&校正畸变)

    :param img: Input (distorted) image.
    :param object_points: Object points
    :param img_points: Image points
    :return: Output (corrected) image that has the same size and type Input image
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, img_points, img.shape[1::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst


if __name__ == '__main__':
    import os
    import laneline_detection.utils as utils

    checkerboard_dir_ = '../Camera Calibration'
    test_images_dir_ = '../Test Images'

    # 获取 棋盘格图片
    cal_images_ = utils.get_images_by_dir(checkerboard_dir_)

    # 获取 测试图片
    test_images_ = utils.get_images_by_dir(test_images_dir_)

    # 计算 object_points, img_points
    object_points_, img_points_ = camera_calibrate(cal_images_, grid=(9, 6))

    # 校正 测试图片
    undistorted_ = []
    for img_ in test_images_:
        img_ = calibrate_distort(img_, object_points_, img_points_)
        undistorted_.append(img_)

    # 创建 储存文件夹 ${test_images_dir_}/After Calibrate
    if not os.path.exists(f'{test_images_dir_}/After Calibrate'):
        os.mkdir(f'{test_images_dir_}/After Calibrate')

    # 储存 校正后的图像
    for i_, img_ in enumerate(undistorted_):
        cv2.imwrite(f'{test_images_dir_}/After Calibrate/{utils.get_files_by_dir(test_images_dir_)[i_]}', img_)
