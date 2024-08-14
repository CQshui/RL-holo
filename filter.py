"""
滤波器，滤出离轴全息图中的单个相，批量生成。
"""

import os
import cv2
import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from tqdm import tqdm


class Filter(object):
    def __init__(self, input_pth, output_pth):
        self.width = None
        self.height = None
        self.fft_img = None
        self.fft_width = None
        self.fft_height = None
        self.u0 = None
        self.cut_size = []
        self.input_pth = input_pth
        self.output_pth = output_pth
        self.fig_num = 0
        self.img_names = None
        self.img_name = None

    def start(self):
        self.img_names = os.listdir(self.input_pth)
        pbar = tqdm(total=len(self.img_names), desc='Filer')
        for img_name in self.img_names:
            self.fig_num += 1
            self.img_name = img_name
            img_pth = os.path.join(self.input_pth, img_name)

            # save_pth = os.path.join(self.output_pth, img_name)
            # if not os.path.exists(save_pth):
            #     os.makedirs(save_pth)

            filtered = self.filter(img_pth, self.input_pth)
            plt.imsave(os.path.join(self.output_pth, img_name), abs(filtered), cmap='gray')

            pbar.update(1)

    def filter(self, img_pth, save_pth):
        # 读取图片，生成灰度图
        img = Image.open(img_pth)
        self.width, self.height = img.size
        gray_image = img.convert("L")
        gray = np.asarray(gray_image)

        # FFT变换生成频谱图
        self.u0 = fftshift(fft2(gray))
        # 超出灰度阈值，降幂
        u1 = np.log(1 + np.abs(self.u0))
        self.fft_img = u1
        fft_pth = os.path.join(output_path, 'FFT.jpg')
        plt.imsave(fft_pth, u1, cmap="gray")

        # 裁剪频谱图，并移动到中心
        self.cut(fft_pth)
        self.move_to_center()

        # 从频域回到空间域
        self.u0 = ifft2(ifftshift(self.u0))

        return self.u0

    def cut(self, fft_pth):
        self.fft_img = cv2.imread(fft_pth, 0)
        if len(self.cut_size) == 0:
            self.fft_height, self.fft_width = self.fft_img.shape[:2]

            cv2.namedWindow('FFT', 0)
            cv2.setMouseCallback('FFT', self.on_mouse)
            cv2.imshow('FFT', self.fft_img)
            cv2.waitKey(0)

            # 滤出最中心的高亮像素块
            _, binary_image = cv2.threshold(self.fft_img, 180, 255, cv2.THRESH_BINARY)
            binary_image_blurred = cv2.GaussianBlur(binary_image, (1, 1), 50)
            contours, _ = cv2.findContours(binary_image_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            counter_x = []
            counter_y = []
            counter_w = []
            counter_h = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                counter_x.append(x)
                counter_y.append(y)
                counter_w.append(w)
                counter_h.append(h)
            # 找到最大宽度的矩形并画出
            max_index = counter_w.index(max(counter_w))
            x_center = counter_x[max_index]
            y_center = counter_y[max_index]
            w_center = counter_w[max_index]
            h_center = counter_h[max_index]
            cv2.rectangle(self.fft_img, (x_center, y_center), (x_center + w_center, y_center + h_center),
                          (0, 255, 0), 8)

            cv2.namedWindow('FFT with Bright Spots', 0)
            cv2.imshow("FFT with Bright Spots", self.fft_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            delta_x = int(0.5 * w_center + x_center - 0.5 * self.fft_width)
            delta_y = int(0.5 * h_center + y_center - 0.5 * self.fft_height)
            self.cut_size.extend([delta_x, delta_y])
            # 此时cut_size的结构为截取的矩形的[最小x坐标，最小y坐标，矩形宽，矩形高，x方向需要位移的距离，y方向需要位移的距离]

        else:
            pass

    def on_mouse(self, event, x, y, flags, param):
        global point1, point2
        fft_copy = self.fft_img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            point1 = (x, y)
            cv2.circle(fft_copy, point1, 10, (0, 255, 0), 3)
            cv2.imshow('FFT', fft_copy)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(fft_copy, point1, (x, y), (255, 0, 0), 3)
            cv2.imshow('FFT', fft_copy)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(fft_copy, point1, point2, (0, 0, 255), 8)
            cv2.imshow('FFT', fft_copy)

            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            rectangle_width = abs(point1[0] - point2[0])
            rectangle_height = abs(point1[1] - point2[1])

            self.fft_img[0:min_y, min_x:self.fft_width] = 0
            self.fft_img[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
            self.fft_img[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
            self.fft_img[0:min_y + rectangle_height, 0:min_x] = 0
            cv2.imshow('FFT', self.fft_img)

            self.cut_size = [min_x, min_y, rectangle_width, rectangle_height]

    def move_to_center(self):
        min_x, min_y, rectangle_width, rectangle_height, delta_x, delta_y = self.cut_size
        self.u0[0:min_y, min_x:self.fft_width] = 0
        self.u0[min_y:self.fft_height, min_x + rectangle_width:self.fft_width] = 0
        self.u0[min_y + rectangle_height:self.fft_height, 0:min_x + rectangle_width] = 0
        self.u0[0:min_y + rectangle_height, 0:min_x] = 0

        self.u0 = np.roll(self.u0, -delta_x, axis=1)
        self.u0 = np.roll(self.u0, -delta_y, axis=0)


if __name__ == '__main__':
    input_path = r'D:\Desktop\test\temp\input'  # FFT.jpg也会储存在这
    output_path = r'D:\Desktop\test\temp\output'
    fil = Filter(input_path, output_path)
    fil.start()
