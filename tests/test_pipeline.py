import os
import timeit
import cv2
from laneline_detection import utils
from laneline_detection import pipeline as test_module

# 参数
checkerboard_size = (9, 6)  # 棋盘格尺寸 (width, height)

# 目录
checkerboard_dir = '../testSource/0.RAW_INPUT/Camera_Calibration'
test_images_dir = '../testSource/0.RAW_INPUT/Test_Image'
result_calibrated_save_dir = '../testSource/0.RAW_OUTPUT/Image_Calibrated'
result_uncalibrated_save_dir = '../testSource/0.RAW_OUTPUT/Image_Uncalibrated'

# 创建 结果储存文件夹
if not os.path.exists(result_calibrated_save_dir):
    os.mkdir(result_calibrated_save_dir)
if not os.path.exists(result_uncalibrated_save_dir):
    os.mkdir(result_uncalibrated_save_dir)

# 获取 测试图片
cal_images = utils.get_images_by_dir(checkerboard_dir)
test_images = utils.get_images_by_dir(test_images_dir)

object_points, img_points = test_module.camera_calibration.camera_calibrate(cal_images, checkerboard_size)
M, Mint = test_module.transform.get_m_minv()

time_cost_calibrated = []
# 处理&储存 测试图片 (矫正畸变)
for i, img in enumerate(test_images):
    # 处理
    start_time = timeit.default_timer()  # 计时_始

    img = test_module.process_img(img, M, Mint, object_points, img_points)

    time_cost_calibrated.append(timeit.default_timer() - start_time)  # 计时_终
    # 储存
    cv2.imwrite(f'{result_calibrated_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', img)

time_cost_uncalibrated = []
# 处理&储存 测试图片 (不矫正畸变)
for i, img in enumerate(test_images):
    # 处理
    start_time = timeit.default_timer()  # 计时_始

    img = test_module.process_img(img, M, Mint, apply_calibrate_distort=False)

    time_cost_uncalibrated.append(timeit.default_timer() - start_time)  # 计时_终
    # 储存
    cv2.imwrite(f'{result_uncalibrated_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', img)

# 打印结果
print("==Time taken (calibrated) ==")
for i, time in enumerate(time_cost_calibrated):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost_calibrated) / len(time_cost_calibrated):.3}s")
print(f"Result saved in {result_calibrated_save_dir}")

print("==Time taken (uncalibrated) ==")
for i, time in enumerate(time_cost_uncalibrated):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost_uncalibrated) / len(time_cost_uncalibrated):.3}s")
print(f"Result saved in {result_calibrated_save_dir}")
