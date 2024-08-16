"""
此程序用于创建数据集
"""

from datetime import datetime
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import cv2
import albumentations as A
import os
from PIL import Image, ImageDraw
from skimage import measure
from tensorflow.keras.utils import Sequence
import segmentation_models as sm
import matplotlib
from matplotlib.widgets import Button
import csv
import re

matplotlib.use('QTAgg')


class Dataset:
    def __init__(
                 self,
                 images_dir,
                 preprocessing=None
                 ):

        self.img_ids = os.listdir(images_dir)

        self.img_fps = [os.path.join(images_dir, id) for id in self.img_ids]

        self.preprocessing = preprocessing

        self.current_img = None

        self.cut_img = None

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_fps[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.current_img = img
        self.cut()
        try:
            shape_store.append(self.cut_img.shape[:2])
        except Exception as e:
            shape_store.append(None)
            self.cut_img = self.current_img
        self.cut_img = cv2.resize(self.cut_img, (512, 512))

        if self.preprocessing:
            # processed_sample = self.preprocessing(image=img, mask=img)
            # img = processed_sample['image']
            processed_sample = self.preprocessing(image=self.cut_img, mask=self.cut_img)
            self.cut_img = processed_sample['image']

        return self.cut_img

    def __len__(self):
        return len(self.img_ids)

    def cut(self):
        img_height, img_width = self.current_img.shape[:2]

        cv2.namedWindow('Image', 0)
        cv2.setMouseCallback('Image', self.on_mouse)
        cv2.imshow('Image', self.current_img)
        cv2.waitKey(0)

        return self.cut_img

    def on_mouse(self, event, x, y, flags, param):
        global point1, point2, flag, min_x, min_y
        img_copy = self.current_img.copy()
        if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
            flag = 1
            point1 = (x, y)
            cv2.circle(img_copy, point1, 10, (0, 255, 0), 3)
            cv2.imshow('Image', img_copy)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
            cv2.rectangle(img_copy, point1, (x, y), (255, 0, 0), 3)
            cv2.imshow('Image', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
            point2 = (x, y)
            cv2.rectangle(img_copy, point1, point2, (0, 0, 255), 8)
            cv2.imshow('Image', img_copy)

            min_x = min(point1[0], point2[0])
            min_y = min(point1[1], point2[1])
            rectangle_width = abs(point1[0] - point2[0])
            rectangle_height = abs(point1[1] - point2[1])

            self.cut_img = self.current_img[min_y:min_y+rectangle_height, min_x:min_x+rectangle_width]

            cv2.imshow('Image', self.cut_img)


def get_preprocessing(preprocessing_fn):
    _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
    return A.Compose(_transform)


def get_position(pr):
    # pr = cv2.cvtColor(pr, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(pr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cut_0_copy = cv2.cvtColor(np.squeeze(cut_0),cv2.COLOR_RGB2GRAY)

    # cv2.drawContours(cut_0, contours, -1, (0, 255, 0), 20)
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
    cv2.rectangle(cut_0_copy, (x_center, y_center), (x_center + w_center, y_center + h_center),
                  (0, 255, 0), 8)

    cv2.namedWindow('PR', 0)
    cv2.imshow("PR", cut_0_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return x_center, y_center, w_center, h_center


def save_as_csv(dicts_l, head):
    # 创建一个CSV
    fo = open("news.CSV", "w", newline='', encoding='utf_8_sig')
    # 写入表头
    writer = csv.DictWriter(fo, head)
    writer.writeheader()
    # 将上一步计算的字典列表写入 CSV 文件中
    writer.writerows(dicts_l)
    # 关闭文件对象
    fo.close()


if __name__ == "__main__":

    # 字典列表
    dicts_list = []
    # 图片名称
    image_name = []
    # 存储z
    z_state = []
    # 存储是否截图的标识符
    cut_flag = []
    # 存储起始/终止两点的坐标
    Start_0 = []
    Start_1 = []
    Finish_0 = []
    Finish_1 = []

    # 图片参数设置
    model_path = r'.\best_model.h5'      # TODO
    predict_images_dir = r'C:\Users\d1009\Desktop\temp\reconstruction\Image__2024-06-06__20-20-17.bmp'  # TODO
    save_dir = r'C:\Users\d1009\Desktop\test\offaxis\result'             # TODO
    shape_store = []

    # 模型参数设置
    sm.set_framework('tf.keras')
    BACKBONE = 'efficientnetb3'
    ACTIVATION = 'sigmoid'
    NUM_CLASSES = 1
    preprocess_input = sm.get_preprocessing(BACKBONE)
    model = sm.Unet(BACKBONE, classes=NUM_CLASSES, activation=ACTIVATION)
    # 模型加载
    model.load_weights(model_path)

    # 数据集加载
    record_dataset = Dataset(
        predict_images_dir,
        preprocessing=get_preprocessing(preprocess_input)
    )

    ids = [x for x in range(len(record_dataset))]
    for i, id in enumerate(ids):
        flag = 0
        min_x = min_y = 0
        cut_0 = record_dataset[id]
        cut = np.expand_dims(cut_0, axis=0)

        # 记录图片名
        image_id = record_dataset.img_ids[id]
        image_name.append(image_id)

        # 记录z
        z = re.search(r"_([^_]+)\.", image_id)
        z_state.append(float(z.group(1)))

        # 记录flag
        cut_flag.append(flag)

        if flag == 1:
            pr_mask = model.predict(cut).round()
            # pr_mask = cv2.bitwise_not(pr_mask)
            pr_mask = np.squeeze(pr_mask)*255

            # 大小回归
            pr_mask = cv2.resize(pr_mask, (shape_store[id][1], shape_store[id][0]))
            cut_0 = cv2.resize(cut_0, (shape_store[id][1], shape_store[id][0]))
            # cv2.imwrite(r'1.jpg', pr_mask)

            # 框选轮廓，返回左上和右下两个顶点坐标
            particle_pos = get_position(pr_mask.astype(np.uint8))
            left_top = (min_x+particle_pos[0], min_y+particle_pos[1])
            right_bottom = (min_x+particle_pos[0] + particle_pos[2], min_y+particle_pos[1] + particle_pos[3])

            # 记录坐标
            Start_0.append(left_top[0])
            Start_1.append(left_top[1])
            Finish_0.append(right_bottom[0])
            Finish_1.append(right_bottom[1])

        else:
            Start_0.append(None)
            Start_1.append(None)
            Finish_0.append(None)
            Finish_1.append(None)

    dicts = {'img': image_name, 'z_state': z_state, 'flag': cut_flag,
             'start_0': Start_0, 'start_1': Start_1, 'finish_0': Finish_0, 'finish_1': Finish_1}

    dicts_list.append(dicts)
    header = ['img', 'z_state', 'flag', 'start_0', 'start_1', 'finish_0', 'finish_1']

    # 写入CSV
    save_as_csv(dicts_list, header)

    # 思路：创建图像界面，鼠标大致截取颗粒图像，将这部分图像输入模型预测，输出其中最大contour的预测矩形框的两个顶点坐标。
    # CSV记录图像名称、z、是否截取颗粒、顶点坐标1和坐标2
