import os
import timeit
import cv2
from laneline_detection import utils
from laneline_detection import thresholding as test_module

# 目录
test_images_dir_ = '../testSource/1.Calibrated'
result_save_dir = '../testSource/2.Thresholded'

# 创建 结果储存文件夹
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)

# 获取 测试图片
test_images = utils.get_images_by_dir(test_images_dir_)

time_cost = []
# 阈值化&储存 测试图片
for i, img in enumerate(test_images):
    # 阈值化
    start_time = timeit.default_timer()  # 计时_始

    img = test_module.thresholding(img)

    time_cost.append(timeit.default_timer() - start_time)  # 计时_终
    # 储存
    cv2.imwrite(f'{result_save_dir}/{utils.get_files_by_dir(test_images_dir_)[i]}', img)

# 打印结果
print("==Time taken==")
for i, time in enumerate(time_cost):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost) / len(time_cost):.3}s")
print(f"Result saved in {result_save_dir}")
