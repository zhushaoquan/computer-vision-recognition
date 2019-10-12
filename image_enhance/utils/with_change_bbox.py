import cv2
import numpy as np
from .box_utils import *
from PIL import Image
import PIL
import random


def rotate(img, boxes, angle):
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    boxes = np.array(boxes).astype(np.float64)
    dst = img.copy()
    height, weight = img.shape[0], img.shape[1]
    cx, cy = weight // 2, height // 2
    corners = get_corners(boxes)
    dst = rotate_im(dst, angle)
    corners[:, :8] = rotate_box(corners[:, :8], angle, cx, cy, height, weight)
    new_bbox = get_enclosing_box(corners)
    scale_factor_x = dst.shape[1] / weight
    scale_factor_y = dst.shape[0] / height
    dst = cv2.resize(dst, (weight, height))
    new_bbox[:, :4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
    boxes = clip_box(new_bbox, [0, 0, weight, height], 0.25)
    return dst, boxes.astype(np.int64)


def horizontal_flip(img, boxes):
    """
    水平翻转
    input:
        image: PIL格式图像
        boxes: 原始标注框信息,list格式
    output:
        image: 水平翻转后的图像,PIL格式
        boxes: 翻转后的框
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    boxes = np.array(boxes).astype(np.float64)
    dst = img.copy()
    # get image center
    img_center = np.array(dst.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    # horizontal flip image
    dst = dst[:, ::-1, :]
    # change boxes for horizontal direction
    boxes[:, [0, 2]] += 2 * (img_center[[0, 2]] - boxes[:, [0, 2]])
    box_w = abs(boxes[:, 0] - boxes[:, 2])
    # finetune box
    boxes[:, 0] -= box_w
    boxes[:, 2] += box_w

    return dst, boxes.astype(np.int64)


def vertical_flip(img, boxes):
    """
    垂直翻转
    input:
        image: PIL格式图像
        boxes: 原始标注框信息,list格式
    output:
        image: 垂直翻转后的图像,PIL格式
        boxes: 翻转后的框
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    boxes = np.array(boxes).astype(np.float64)
    dst = img.copy()
    # get image center
    img_center = np.array(dst.shape[:2])[::-1] / 2
    img_center = np.hstack((img_center, img_center))
    # horizontal flip image
    dst = dst[::-1, :, :]
    # change boxes for horizontal direction
    boxes[:, [1, 3]] += 2 * (img_center[[1, 3]] - boxes[:, [1, 3]])
    box_h = abs(boxes[:, 1] - boxes[:, 3])
    # finetune box
    boxes[:, 1] -= box_h
    boxes[:, 3] += box_h

    return dst, boxes.astype(np.int64)


def scale(img, boxes, ratio=[0.2, 0.2]):
    """
    按给定尺度压缩图像
    input:
        image: PIL格式图像
        boxes: 原始标注框信息,list格式
        ratio: 压缩比例,[x,y] 两个方向
    output:
        image: 旋转后的图像,PIL格式
        boxes: 翻转后的框
    """
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    assert isinstance(ratio, list)
    assert ratio[0] < 1
    assert ratio[1] < 1
    boxes = np.array(boxes).astype(np.float64)
    scale_x, scale_y = ratio
    img_shape = img.shape
    # resize
    resize_scale_x = 1 - scale_x
    resize_scale_y = 1 - scale_y
    mask = cv2.resize(img, None, fx=resize_scale_x, fy=resize_scale_y)
    boxes[:, :4] *= [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
    dst = np.zeros(img_shape, dtype=np.uint8)

    y_lim = int(min(resize_scale_y, 1) * img_shape[0])
    x_lim = int(min(resize_scale_x, 1) * img_shape[1])

    dst[:y_lim, :x_lim, :] = mask[:y_lim, :x_lim, :]
    boxes = clip_box(boxes, [0, 0, 1 + img_shape[1], img_shape[0]], 0.25)

    return dst, boxes.astype(np.int64)


def translation(img, boxes):
    '''
    平移后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        shift_img:平移后的图像array
        shift_bboxes:平移后的bounding box的坐标list
    '''
    # ---------------------- 平移图像 ----------------------
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    height, width = img.shape[:2]
    x_min = width  # 平移后的包含所有目标框的最小的框
    x_max = 0
    y_min = height
    y_max = 0
    for bbox in boxes:
        x_min = min(x_min, int(bbox[0]))
        y_min = min(y_min, int(bbox[1]))
        x_max = max(x_max, int(bbox[2]))
        y_max = max(y_max, int(bbox[3]))

    d_to_left = x_min  # 包含所有目标框的最大左移动距离
    d_to_right = width - x_max  # 包含所有目标框的最大右移动距离
    d_to_top = y_min  # 包含所有目标框的最大上移动距离
    d_to_bottom = height - y_max  # 包含所有目标框的最大下移动距离

    x = int(random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3))
    y = int(random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3))

    M = np.float32([[1, 0, x], [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # ---------------------- 平移boundingbox ----------------------
    shift_bboxes = []
    for bbox in boxes:
        shift_bboxes.append([int(bbox[0]) + x, int(bbox[1]) + y, int(bbox[2]) + x, int(bbox[3]) + y])
    return shift_img, shift_bboxes


def crop(img, boxes):
    '''
    裁剪后的图片要包含所有的框
    输入:
        img:图像array
        bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
    输出:
        crop_img:裁剪后的图像array
        crop_bboxes:裁剪后的bounding box的坐标list
    '''
    # ---------------------- 裁剪图像 ----------------------
    assert isinstance(img, np.ndarray)
    assert isinstance(boxes, list)
    height, width = img.shape[:2]
    x_min = width  # 平移后的包含所有目标框的最小的框
    x_max = 0
    y_min = height
    y_max = 0
    for bbox in boxes:
        x_min = min(x_min, int(bbox[0]))
        y_min = min(y_min, int(bbox[1]))
        x_max = max(x_max, int(bbox[2]))
        y_max = max(y_max, int(bbox[3]))

    d_to_left = x_min  # 包含所有目标框的最大左移动距离
    d_to_right = width - x_max  # 包含所有目标框的最大右移动距离
    d_to_top = y_min  # 包含所有目标框的最大上移动距离
    d_to_bottom = height - y_max  # 包含所有目标框的最大下移动距离

    # 随机扩展这个最小框
    # crop_x_min = int(x_min - random.uniform(0, d_to_left))
    # crop_y_min = int(y_min - random.uniform(0, d_to_top))
    # crop_x_max = int(x_max + random.uniform(0, d_to_right))
    # crop_y_max = int(y_max + random.uniform(0, d_to_bottom))

    ##随机扩展这个最小框 , 防止别裁的太小
    crop_x_min = int(x_min - random.uniform(d_to_left // 2, d_to_left))
    crop_y_min = int(y_min - random.uniform(d_to_top // 2, d_to_top))
    crop_x_max = int(x_max + random.uniform(d_to_right // 2, d_to_right))
    crop_y_max = int(y_max + random.uniform(d_to_bottom // 2, d_to_bottom))

    # 确保不要越界
    crop_x_min = max(0, crop_x_min)
    crop_y_min = max(0, crop_y_min)
    crop_x_max = min(width, crop_x_max)
    crop_y_max = min(height, crop_y_max)

    crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

    # ---------------------- 裁剪boundingbox ----------------------
    # 裁剪后的boundingbox坐标计算
    crop_bboxes = list()
    for bbox in boxes:
        crop_bboxes.append(
            [int(bbox[0]) - crop_x_min, int(bbox[1]) - crop_y_min, int(bbox[2]) - crop_x_min,
             int(bbox[3]) - crop_y_min])

    return crop_img, crop_bboxes
