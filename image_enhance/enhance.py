import numpy as np
import os
from glob import glob
import pandas as pd
import argparse
import tqdm
import shutil
import cv2
from tqdm import tqdm
from config import opt
import random
from utils.no_change_bbox import *
from utils.with_change_bbox import *


class Augumentor(object):
    def __init__(self):
        self.ori_csv_path = opt.ori_csv_path
        self.ori_img_paths = glob(os.path.join(opt.ori_img_path, "*.%s" % (opt.img_format)))
        if opt.is_mix_up:
            self.total_mixup_back_img = os.listdir(opt.mix_up_back_img)

    def fit(self):
        total_boxes = {}
        # read box info for csv format
        annotations = pd.read_csv(self.ori_csv_path, header=0).values
        for annotation in annotations:
            key = annotation[0]
            value = np.array([annotation[1:]])
            if key in total_boxes.keys():
                total_boxes[key] = np.concatenate((total_boxes[key], value), axis=0)
            else:
                total_boxes[key] = value
        for image_path in tqdm(self.ori_img_paths):
            try:
                img = cv2.imread(image_path)
                filename = image_path.split(os.sep)[-1]
                ori_bbox = total_boxes[filename].tolist()

                if opt.is_brightness:
                    if random.random() < opt.p_brightness:
                        bright_img, bright_bbox = brightness(img, ori_bbox)
                        self.save_aug_csv("%s_%s" % ("bright", filename), bright_bbox)
                        self.save_aug_img("%s_%s" % ("bright", filename), bright_img)

                if opt.is_color:
                    if random.random() < opt.p_color:
                        color_img, color_bbox = color(img, ori_bbox)
                        self.save_aug_csv("%s_%s" % ("color", filename), color_bbox)
                        self.save_aug_img("%s_%s" % ("color", filename), color_img)

                if opt.is_blur:
                    if random.random() < opt.p_blur:
                        blur_img, blur_bbox = blur(img, ori_bbox)
                        self.save_aug_csv("%s_%s" % ("blur", filename), blur_bbox)
                        self.save_aug_img("%s_%s" % ("blur", filename), blur_img)

                if opt.is_noise:
                    if random.random() < opt.p_noise:
                        noise_img, noise_bbox = noise(img, ori_bbox)
                        self.save_aug_csv("%s_%s" % ("noise", filename), noise_bbox)
                        self.save_aug_img("%s_%s" % ("noise", filename), noise_img)

                if opt.is_contrast:
                    if random.random() < opt.p_contrast:
                        contrast_img, contrast_bbox = contrast(img, ori_bbox)
                        self.save_aug_csv("%s_%s" % ("contrast", filename), contrast_bbox)
                        self.save_aug_img("%s_%s" % ("contrast", filename), contrast_img)

                if opt.is_rotate:
                    if random.random() < opt.p_rotate:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        for angle in opt.rotate_angle:
                            trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                            rotated_image, rotated_box = rotate(img, trans_boxes, angle)
                            self.save_aug_csv("%s_%s_%s" % ("rotate", str(angle), filename), [aug_labels, rotated_box],
                                              original=False)
                            self.save_aug_img("%s_%s_%s" % ("rotate", str(angle), filename), rotated_image)

                if opt.is_horizontal_flip:
                    if random.random() < opt.p_horizontal_flip:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        hflip_image, hflip_box = horizontal_flip(img, trans_boxes)
                        self.save_aug_csv("%s_%s" % ("hflip", filename), [aug_labels, hflip_box],
                                          original=False)
                        self.save_aug_img("%s_%s" % ("hflip", filename), hflip_image)

                if opt.is_vertical_flip:
                    if random.random() < opt.p_vertical_flip:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        vflip_image, vflip_box = vertical_flip(img, trans_boxes)
                        self.save_aug_csv("%s_%s" % ("vflip", filename), [aug_labels, vflip_box],
                                          original=False)
                        self.save_aug_img("%s_%s" % ("vflip", filename), vflip_image)

                if opt.is_scale:
                    if random.random() < opt.p_scale:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        scale_image, scale_box = scale(img, trans_boxes, ratio=[0.2, 0.2])
                        self.save_aug_csv("%s_%s" % ("scale", filename), [aug_labels, scale_box], original=False)
                        self.save_aug_img("%s_%s" % ("scale", filename), scale_image)

                if opt.is_translation:
                    if random.random() < opt.p_translation:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        translation_image, translation_box = translation(img, trans_boxes)
                        self.save_aug_csv("%s_%s" % ("translation", filename), [aug_labels, translation_box],
                                          original=False)
                        self.save_aug_img("%s_%s" % ("translation", filename), translation_image)

                if opt.is_crop:
                    if random.random() < opt.p_crop:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        crop_image, crop_box = crop(img, trans_boxes)
                        self.save_aug_csv("%s_%s" % ("crop", filename), [aug_labels, crop_box], original=False)
                        self.save_aug_img("%s_%s" % ("crop", filename), crop_image)

                if opt.is_mix_up:
                    if random.random() < opt.p_mix_up:
                        aug_labels = np.array(ori_bbox)[:, -1]
                        trans_boxes = np.array(ori_bbox)[:, :-1].tolist()
                        back_img_name = random.sample(self.total_mixup_back_img, 1)[0]
                        back_img_path = os.path.join(opt.mix_up_back_img, back_img_name)
                        back_img = cv2.imread(back_img_path)
                        mixup_image, mixup_box = mix_up(img, back_img, trans_boxes)
                        self.save_aug_csv("%s_%s_%s" % (back_img_name.replace('.jpg', ''), "mixuped", filename),
                                          [aug_labels, mixup_box],
                                          original=False)
                        self.save_aug_img("%s_%s_%s" % (back_img_name.replace('.jpg', ''), "mixuped", filename),
                                          mixup_image)
            except Exception as e:
                print("{} labelimg type error".format(image_path))

    def save_aug_csv(self, filename, boxes, original=True):
        saved_file = open(opt.aug_csv_path, "a+")
        if original:
            for box in boxes:
                label = box[-1]
                saved_file.write(filename + "," + str(box[0]) + "," + str(box[1]) + "," + str(box[2]) + "," + str(
                    box[3]) + "," + str(label) + "\n")
        else:
            labels, new_boxes = boxes[0], boxes[1]
            for label, new_box in zip(labels, new_boxes):
                saved_file.write(
                    filename + "," + str(new_box[0]) + "," + str(new_box[1]) + "," + str(new_box[2]) + "," + str(
                        new_box[3]) + "," + str(label) + "\n")

    def save_aug_img(self, filename, img):
        cv2.imwrite(os.path.join(opt.aug_img_path, filename), img)


if __name__ == "__main__":
    os.makedirs(opt.aug_img_path) if not os.path.exists((opt.aug_img_path)) else None
    shutil.copy(opt.ori_csv_path, opt.aug_csv_path) if not os.path.exists((opt.aug_csv_path)) else None
    for img in tqdm(os.listdir(opt.ori_img_path)):
        shutil.copy(os.path.join(opt.ori_img_path, img), opt.aug_img_path)
    aug = Augumentor()
    aug.fit()
