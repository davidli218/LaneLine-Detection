import os
import timeit
import cv2
from laneline_detection import utils
from laneline_detection import transform
from laneline_detection import find_line as test_module

import numpy as np
import matplotlib.pyplot as plt

# 目录
raw_image_dir = '../testSource/0.RAW_INPUT/Test_Image'
test_images_dir = '../testSource/3.Thresholded_Aerial'
histogram_save_dir = '../testSource/4.Thresholded_Aerial_Histogram'
result_save_dir = '../testSource/5.Result'

# 创建 结果储存文件夹
if not os.path.exists(histogram_save_dir):
    os.mkdir(histogram_save_dir)
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)

# 获取 测试图片
test_images = utils.get_images_by_dir(test_images_dir)
raw_images = utils.get_images_by_dir(raw_image_dir)
test_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in test_images]  # 转灰度

M, Minv = transform.get_m_minv()

time_cost = []
# 处理&保存 测试图片
for i, img in enumerate(test_images):
    # 保存直方图
    hist = np.sum(img[img.shape[0] // 2:, :], axis=0)
    plt.plot(np.arange(len(hist)), hist)
    plt.savefig(f'{histogram_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', bbox_inches='tight')
    plt.close('all')

    # 处理
    start_time = timeit.default_timer()  # 计时_始

    l_fit, r_fit, *_ = test_module.find_line(img)
    new_img = test_module.draw_area(raw_images[i], img, Minv, l_fit, r_fit)  # 车道上色
    new_img = test_module.draw_values(new_img,
                                      *test_module.calculate_curv_and_pos(img, l_fit, r_fit))  # 标注信息

    time_cost.append(timeit.default_timer() - start_time)  # 计时_终

    # 储存
    cv2.imwrite(f'{result_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', new_img)

# 打印结果
print("==Time taken==")
for i, time in enumerate(time_cost):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost) / len(time_cost):.3}s")
print(f"Histogram saved in {histogram_save_dir}")
print(f"Result saved in {result_save_dir}")
