import os

import cv2
import numpy as np


def get_files_by_dir(dirname) -> [str]:
    """Returns all file names in this directory"""
    return [name for name in os.listdir(dirname) if os.path.isfile(f'{dirname}/{name}')]


def get_images_by_dir(dirname) -> [np.ndarray]:
    """
    get all image in the given directory.
    presume that this directory only contains image files.
    """
    img_names = get_files_by_dir(dirname)
    img_paths = [f'{dirname}/{name}' for name in img_names if os.path.isfile(f'{dirname}/{name}')]
    images = [cv2.imread(path) for path in img_paths]
    return images


def compare_pairs(images_src_dir, images_dst_dir, save_dir):
    """保存前后对比图"""
    images_src = get_images_by_dir(images_src_dir)
    images_dst = get_images_by_dir(images_dst_dir)

    h, w, _ = images_src[0].shape
    rate = round(480 / w, 1)
    img_src_resize = cv2.resize(images_src[0], (int(w * rate), int(h * rate)))
    img_dst_resize = cv2.resize(images_dst[0], (int(w * rate), int(h * rate)))
    result = np.hstack([img_src_resize, img_dst_resize])

    for i, img in enumerate(images_src):
        if i == 0:
            continue
        h, w, _ = img.shape
        rate = round(480 / w, 1)

        img_src_resize = cv2.resize(img, (int(w * rate), int(h * rate)))
        img_dst_resize = cv2.resize(images_dst[i], (int(w * rate), int(h * rate)))

        result = np.vstack((result, np.hstack([img_src_resize, img_dst_resize])))

    cv2.imwrite(save_dir, result)
