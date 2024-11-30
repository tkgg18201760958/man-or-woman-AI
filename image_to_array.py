import cv2
import numpy as np
from itertools import chain


def image_to_array(image_path):
    # 读取图片并转换为灰度图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图片是否加载成功
    if image is None:
        return "错误：无法加载图片。"
    else:
        # 将图片缩放到指定大小
        resized_image = cv2.resize(image, (1000, 1000))
        # 将灰度图像数据存入一个二维数组中
        grayscale_array = np.array(resized_image)
        normalized_array = np.interp(grayscale_array, (0, 255), (0, 1))
        # 输出
        return list(chain.from_iterable(normalized_array))
