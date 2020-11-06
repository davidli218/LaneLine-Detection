import os
import laneline_detection.utils as utils

result_save_dir = '../testSource/6.Effect_Viewing'

# 创建 结果储存文件夹
if not os.path.exists(result_save_dir):
    os.mkdir(result_save_dir)

compare_list = [
    {  # <原图>对比<已矫正畸变>
        'dir1': '../testSource/0.RAW_INPUT/Test_Image',
        'dir2': '../testSource/1.Calibrated',
        'result_name': '1.raw-calibrated.jpg'},
    {  # <已矫正畸变>对比<已阈值化>
        'dir1': '../testSource/1.Calibrated',
        'dir2': '../testSource/2.Thresholded',
        'result_name': '2.calibrated-thresholded.jpg'},
    {  # <已阈值化>对比<俯视图>
        'dir1': '../testSource/2.Thresholded',
        'dir2': '../testSource/3.Thresholded_Aerial',
        'result_name': '3.thresholded-transformed.jpg'},
    {  # <俯视图>对比<直方图>
        'dir1': '../testSource/3.Thresholded_Aerial',
        'dir2': '../testSource/4.Thresholded_Aerial_Histogram',
        'result_name': '4.transformed-histogram.jpg'},
    {  # <原图>对比<最终结果>
        'dir1': '../testSource/0.RAW_INPUT/Test_Image',
        'dir2': '../testSource/5.Result',
        'result_name': '5.raw-result.jpg'},
]

for i in compare_list:
    utils.compare_pairs(i['dir1'], i['dir2'], f"{result_save_dir}/{i['result_name']}")
