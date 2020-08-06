import cv2
import numpy as np


def abs_sobel_threshold(img, thresh_min=0, thresh_max=255):
    """Sobel算子 x轴方向梯度阈值过滤"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用cv2.Sobel()计算计算x方向的导数
    abs_sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0))

    # 阈值过滤
    ret, binary_output = cv2.threshold(abs_sobel_x, thresh_min, thresh_max, cv2.THRESH_BINARY)

    return binary_output


def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    """Sobel算子 梯度大小阈值过滤"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 使用cv2.Sobel()计算计算x方向和y方向的导数
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the gradient magnitude
    gradient_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Rescale to 8 bit
    scale_factor = np.max(gradient_mag) / 255
    gradient_mag = (gradient_mag / scale_factor).astype(np.uint8)

    ret, binary_output = cv2.threshold(gradient_mag, mag_thresh[0], mag_thresh[1], cv2.THRESH_BINARY)

    return binary_output


def hls_select(img, channel='s', thresh=(0, 255)):
    """使用HLS颜色空间的进行阈值过滤"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    channel = {'h': hls[:, :, 0], 'l': hls[:, :, 1], 's': hls[:, :, 2]}[channel]

    ret, binary_output = cv2.threshold(channel, thresh[0], thresh[1], cv2.THRESH_BINARY)

    return binary_output
    

def luv_select(img, thresh=(0, 255)):
    """使用LUV颜色空间的L(lightness亮度)通道进行阈值过滤"""
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    l_channel = luv[:, :, 0]

    # binary_output = np.zeros_like(l_channel)
    # binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255
    ret, binary_output = cv2.threshold(l_channel, thresh[0], thresh[1], cv2.THRESH_BINARY)

    return binary_output


def thresholding(img):
    """结合多种阈值过滤"""
    x_thresh = abs_sobel_threshold(img, thresh_min=90, thresh_max=255)  # x轴方向梯度过滤器
    mag_thresh = mag_threshold(img, sobel_kernel=3, mag_thresh=(50, 255))  # 梯度大小过滤器
    hls_thresh = hls_select(img, thresh=(160, 255))  # HLS颜色空间 S(Saturation饱和)通道过滤
    luv_thresh = luv_select(img, thresh=(235, 255))  # LUV色彩空间 L(lightness亮度)通道过滤

    # Thresholding combination
    combined = np.zeros_like(x_thresh)
    combined[(x_thresh == 255) | (hls_thresh == 255) | ((mag_thresh == 255) & (luv_thresh == 255))] = 255

    return combined


if __name__ == '__main__':
    import os
    import laneline_detection.utils as utils

    test_images_dir_ = '../Test Images/After Calibrate'
    save_result_dir = '../Test Images/After Thresholding'

    # 获取 测试图片
    test_images_ = utils.get_images_by_dir(test_images_dir_)

    # 阈值化 测试图片
    thresholded_ = []
    for img_ in test_images_:
        img_ = thresholding(img_)
        thresholded_.append(img_)

    # 创建 储存文件夹
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)

    # 储存 校正后的图像
    for i_, img_ in enumerate(thresholded_):
        cv2.imwrite(f'{save_result_dir}/{utils.get_files_by_dir(test_images_dir_)[i_]}', img_)
