import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.measure import shannon_entropy
import numpy.fft as fft
matplotlib.use('TkAgg')


def calculate_spectral_norm(image):
    # 傅里叶变换
    f_transform = fft.fftshift(fft.fft2(image))
    # 计算频谱范数
    return np.linalg.norm(f_transform)


def calculate_spectrum_metric(image):
    # 对图像进行傅里叶变换
    f_transform = fft.fftshift(fft.fft2(image))
    # 计算频谱幅度
    magnitude_spectrum = np.abs(f_transform)
    # 频谱的平均值作为指标
    return np.mean(magnitude_spectrum)


def extract_number(s):
    return int(s.split('_')[1])  # 取第一个和第二个下划线之间的部分并转为整数


def evaluate_focus(image):
    # 在中心位置附近取出一定大小的窗口区域，计算其聚焦清晰度
    # image = image[1482:2508, 2664:3714]
    image = image[864:1900, 700:1700]

    # cv2.namedWindow('image', 0)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # # 计算该区域的清晰度（拉普拉斯方差或边缘锐度）
    # laplacian = cv2.Laplacian(image, cv2.CV_64F)
    # clarity = laplacian.var()

    # 强度起伏法，重建图像的平均强度不能突变，否则不能用
    # clarity = np.std(image)

    # # 使用Sobel算子计算x方向和y方向的梯度
    # sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    # sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    # # 计算梯度幅值
    # gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # # 返回梯度幅值的均值作为聚焦判据
    # clarity = np.mean(gradient_magnitude)

    # 熵值法，效果不错，聚焦在off_102
    # clarity = shannon_entropy(image)

    # 频谱指标
    # clarity = calculate_spectrum_metric(image)

    # 方差法 off_101
    # clarity = np.var(image)

    # 频谱范数 off_99 峰值不明显
    clarity = calculate_spectral_norm(image)

    return clarity


if __name__ == '__main__':
    input_dir = r'D:\Desktop\test\dof\boliqiu1'
    images = os.listdir(input_dir)
    # 按文件名排序，确保图片按顺序排列
    images = sorted(images, key=extract_number)
    x = []
    y = []

    for image in images:
        img = cv2.imread(os.path.join(input_dir, image), 0)
        clarity = evaluate_focus(img)
        print(image, clarity)

        x.append(images.index(image))
        y.append(clarity)

    print(np.argmin(y)+1)
    print(np.argmax(y)+1)

    # 绘制图像
    plt.plot(x, y)
    # 显示图像
    plt.show()
