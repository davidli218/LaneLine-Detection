import os
import timeit
import cv2
from laneline_detection import utils
from laneline_detection import camera_calibration as test_module

# 参数
checkerboard_size = (9, 6)  # 棋盘格尺寸 (width, height)
if_visualize = False

# 目录
checkerboard_dir = '../testSource/0.RAW_INPUT/Camera_Calibration'
test_images_dir = '../testSource/0.RAW_INPUT/Test_Image'
result_save_dir = '../testSource/1.Calibrated'

# 创建 结果储存文件夹
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)

# 获取 测试图片
cal_images = utils.get_images_by_dir(checkerboard_dir)
test_images = utils.get_images_by_dir(test_images_dir)

object_points, img_points = test_module.camera_calibrate(cal_images, checkerboard_size, if_visualize)

time_cost = []
# 校正&储存 测试图片
for i, img in enumerate(test_images):
    # 校正
    start_time = timeit.default_timer()  # 计时_始

    img = test_module.calibrate_distort(img, object_points, img_points)

    time_cost.append(timeit.default_timer() - start_time)  # 计时_终
    # 储存
    cv2.imwrite(f'{result_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', img)

# 打印结果
print("==Time taken==")
for i, time in enumerate(time_cost):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost) / len(time_cost):.3}s")
print(f"Result saved in {result_save_dir}")
