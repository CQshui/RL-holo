# -*- coding: utf-8 -*-
"""
此程序将接收频谱滤波之后的全息干涉图像，使用菲涅尔变换。
"""
import os
import numpy as np
from PIL import Image
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
from tqdm import tqdm


class Fresnel(object):
    def __init__(self, lam, pix, z, input_pth, output_pth):
        self.width = None
        self.height = None
        self.u0 = None
        self.lam = lam
        self.pix = pix
        self.z = z
        self.input_pth = input_pth
        self.output_pth = output_pth
        self.fig_num = 0
        self.img_names = None

    def start(self):
        self.img_names = os.listdir(self.input_pth)
        pbar = tqdm(total=len(self.img_names), desc='Reconstruct')
        # for img_name in self.img_names:
        #     self.fig_num += 1
        #     self.img_name = img_name
        #     img_pth = os.path.join(self.input_pth, img_name)
        #     save_pth = os.path.join(self.output_pth, img_name)
        #
        #     if not os.path.exists(save_pth):
        #         os.makedirs(save_pth)
        #
        #     self.reconstruct(img_pth, save_pth)
        #     pbar.update(1)
        self.fig_num += 1
        img_name = self.img_names[self.fig_num - 1]
        img_pth = os.path.join(self.input_pth, img_name)
        save_pth = os.path.join(self.output_pth, img_name)

        if not os.path.exists(save_pth):
            os.makedirs(save_pth)

        self.reconstruct(img_pth, save_pth)
        pbar.update(1)

    def reconstruct(self, img_pth, save_pth):
        # 读取图片，生成灰度图
        img = Image.open(img_pth)
        self.width, self.height = img.size
        gray_image = img.convert("L")
        gray = np.asarray(gray_image)

        # 重建启动
        k = 2 * np.pi / self.lam
        x = np.linspace(-self.pix * self.width / 2, self.pix * self.width / 2, self.width)
        y = np.linspace(-self.pix * self.height / 2, self.pix * self.height / 2, self.height)
        x, y = np.meshgrid(x, y)
        r = np.sqrt(x ** 2 + y ** 2 + self.z ** 2)
        h = self.z / (1j * self.lam * r ** 2) * np.exp(1j * k * r)  # changed, h = 1/(1j*lam*r)*np.exp(1j*k*r)
        H = fft2(fftshift(h)) * self.pix ** 2
        u1 = fft2(fftshift(gray))
        u2 = u1 * H
        u3 = ifftshift(ifft2(u2))
        off_axis_dir = self.output_pth + '/{:}'.format(self.img_names[0])  # 要改
        off_img_path = off_axis_dir + '/off_{:.7f}.jpg'.format(self.z)
        if not os.path.exists(off_axis_dir):
            os.makedirs(off_axis_dir)
        plt.imsave(off_img_path, abs(u3), cmap="gray")


if __name__ == '__main__':
    lam = 532e-9
    pix = 0.223e-6
    z1 = 0.00008
    z_interval = 0.00001
    input_pth = r'D:\Desktop\Data\20240717\lvnishi\1'
    output_pth = r'D:\Desktop\temp\re'
    image = Fresnel(lam, pix, z1, input_pth, output_pth)
    image.start()
