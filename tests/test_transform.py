import os
import timeit
import cv2
from laneline_detection import utils
from laneline_detection import transform as test_module

# 参数
checkerboard_size = (9, 6)  # 棋盘格尺寸 (width, height)
if_visualize = False

# 目录
test_images_dir = '../testSource/2.Thresholded'
result_save_dir = '../testSource/3.Thresholded_Aerial'

# 创建 结果储存文件夹
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)

# 获取 测试图片
test_images = utils.get_images_by_dir(test_images_dir)
test_images = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in test_images]  # 转灰度

m, minv = test_module.get_m_minv()

time_cost = []
# 校正&储存 测试图片
for i, img in enumerate(test_images):
    # 校正
    start_time = timeit.default_timer()  # 计时_始

    img = test_module.transform_img(img, m)

    time_cost.append(timeit.default_timer() - start_time)  # 计时_终
    # 储存
    cv2.imwrite(f'{result_save_dir}/{utils.get_files_by_dir(test_images_dir)[i]}', img)

# 打印结果
print("==Time taken==")
for i, time in enumerate(time_cost):
    print(f"pict{i + 1}: {time:.3}s")
print(f"\nAverage: {sum(time_cost) / len(time_cost):.3}s")
print(f"Result saved in {result_save_dir}")
