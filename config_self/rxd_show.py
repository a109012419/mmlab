import math

import numpy as np
import rasterio
from matplotlib import pyplot as plt


def data_standardization(fnp, datarange):
    """
    数据归一化拉伸

    :param fnp: [H,W,C]数据矩阵
    :param datarange: 映射范围如[0,1]or[0,255]
    :return: 归一化后的[H,W,C]数组,0-1则为float32,0-255则为uint8
    """
    fnp = fnp.transpose((2, 0, 1)).astype(np.float32)  # 转为CHW方便操作,矩阵运算时转为float32(int型除法会出现误差)
    cnl, row, col = fnp.shape
    _, data_range = datarange

    for i in range(cnl):
        high_value = np.max(fnp[i])
        low_value = np.min(fnp[i])
        fnp[i] = (fnp[i] - low_value) / (high_value - low_value)  # 归一化到0-1之间

    fnp[fnp > 1] = 1
    if data_range == 255:
        fnp = (fnp * data_range).astype(np.uint8)  # 若data_range为255则乘255
    fnp = fnp.transpose((1, 2, 0))  # 转为HWC返回
    return fnp


def stretch(fnp, *band, threshold=2):
    """
    对[H,W,C]图像进行拉伸,默认为2%拉伸

    :param fnp:[H,W,C]格式的矩阵
    :param band:波段序列
    :param threshold:拉伸指数
    :return:2%拉伸后的HWC数组
    """
    row, col, cnl = fnp.shape
    if fnp.dtype != 'uint8':
        fnp = data_standardization(fnp, [0, 255])  # 如果不是uint8,就先归一化到0-255的uint8型
    fnp = fnp.transpose((2, 0, 1))
    for i in range(cnl):
        high_value = np.percentile(fnp[i], 100 - threshold)  # 取得98%直方图处对应灰度
        low_value = np.percentile(fnp[i], threshold)  # 取得2%直方图处对应灰度
        truncated_gray = np.clip(fnp[i], a_min=low_value, a_max=high_value)
        fnp[i] = (truncated_gray - low_value) / (high_value - low_value) * 255  # %2拉伸,映射到0-255
    fnp = fnp.transpose((1, 2, 0)).astype(np.uint8)  # 因为映射到0-255,所以改为uint8
    if band:
        band = band[0]
        band = [i - 1 for i in band]
    else:
        pass
    return fnp[:, :, band]


def stretch_display(fnp, *band, threshold=2):
    """
    利用plt展示[H,W,C]图像

    :param fnp:[H,W,C]格式的矩阵
    :param band:波段序列
    :param threshold:拉伸指数
    :return:None
    """
    row, col, cnl = fnp.shape
    if fnp.dtype != 'uint8':
        fnp = data_standardization(fnp, [0, 255])  # 如果不是uint8,就先归一化到0-255的uint8型
    fnp = fnp.transpose((2, 0, 1))
    for i in range(cnl):
        high_value = np.percentile(fnp[i], 100 - threshold)  # 取得98%直方图处对应灰度
        low_value = np.percentile(fnp[i], threshold)  # 取得2%直方图处对应灰度
        truncated_gray = np.clip(fnp[i], a_min=low_value, a_max=high_value)
        fnp[i] = (truncated_gray - low_value) / (high_value - low_value) * 255  # %2拉伸,映射到0-255
    fnp = fnp.transpose((1, 2, 0)).astype(np.uint8)  # 因为映射到0-255,所以改为uint8
    if band:
        band = band[0]
        band = [i - 1 for i in band]
        plt.imshow(fnp[:, :, band])  # 按波段序列展示图像
    else:
        plt.imshow(fnp)
    plt.show()


def imgs_display(*data):
    lens = len(data)
    rows = math.ceil(lens / 2)
    fig, axes = plt.subplots(rows, 2, sharex=True, sharey=True)

    if rows == 2:
        count = 0
        for i in range(rows):
            for j in [0, 1]:
                if count + 1 > lens: break
                axes[i, j].imshow(data[count])
                count += 1
    else:
        print(data[0].shape)
        axes[0].imshow(data[0])
        axes[1].imshow(data[1])

    plt.show()


if __name__ == '__main__':
    data = rasterio.open('wv2_20130916_clip2')

    all_band = data.read().transpose((1, 2, 0))
    stretch_display(all_band, [3, 4, 2])
