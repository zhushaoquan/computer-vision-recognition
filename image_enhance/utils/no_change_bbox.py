"""
不改变标注信息的数据扩充方式,通过python PIL.ImageEnhance 和opencv 来实现,
包括:

1. blur : 图像模糊操作(cv2),支持:均值模糊,中值模糊,高斯模糊
2. 图像对比度调节:PIL.ImageEnhance.Contrast()
3. 图像亮度调节:PIL.ImageEnhance.Brightness()
4. 图像加噪声:通过skimage.util.random_noise()实现,支持:高斯噪声、盐/椒噪声、泊松噪声、乘法噪声

"""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise
import cv2


def color(img, boxes, num=1.3):
    """
    func:   对图像进行色彩平衡调节
    input:
        image:  待增强原始图像的路径,np
        num:    亮度调节的程度,0表示黑白,1.0表示原始色彩,默认设置1.3
        boxes:  图像中待检测物体的标注框信息,以list格式传入
    output:
        dst:  np
        boxes_changed:  改变后的标注框,无改变时为原始框,list格式
    """
    assert isinstance(img, np.ndarray)
    dst = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
    enh_color = ImageEnhance.Color(dst)
    image_enhanced = enh_color.enhance(num)
    dst = cv2.cvtColor(np.asarray(image_enhanced), cv2.COLOR_RGB2BGR)
    return dst, boxes


def contrast(img, boxes, num=1.3):
    """
    func:   对图像进行对比度调节
    input:
        image:  待增强原始图像的路径,PIL格式
        num:    对比度的调节,0表示纯灰色,1.0表示原始对比,默认设置1.3
        boxes:  图像中待检测物体的标注框信息,以list格式传入
    output:
        image:  PIL格式的图像
        boxes_changed:  改变后的标注框,无改变时为原始框,list格式
    """
    assert isinstance(img, np.ndarray)
    dst = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
    enh_contrast = ImageEnhance.Contrast(dst)
    image_enhanced = enh_contrast.enhance(num)
    dst = cv2.cvtColor(np.asarray(image_enhanced), cv2.COLOR_RGB2BGR)
    return dst, boxes


def brightness(img, boxes, num=1.3):
    """
    func:   对图像进行亮度调节
    input:
        image:  待增强原始图像的路径,PIL格式
        num:    对比度的调节,0表示黑色,1.0表示原始亮度,默认设置1.3
        boxes:  图像中待检测物体的标注框信息,以list格式传入
    output:
        image:  PIL格式的图像
        boxes_changed:  改变后的标注框,无改变时为原始框,list格式
    """
    assert isinstance(img, np.ndarray)
    dst = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
    enh_brightness = ImageEnhance.Brightness(dst)
    image_enhanced = enh_brightness.enhance(num)
    dst = cv2.cvtColor(np.asarray(image_enhanced), cv2.COLOR_RGB2BGR)
    return dst, boxes


def noise(img, boxes, noise_type="gaussian"):
    """
    func:   对图像增加噪声
    input:
        image:  待增强原始图像的路径,PIL格式
        noise_type:    噪声类别,包括高斯/盐椒噪声/泊松噪声等
        boxes:  图像中待检测物体的标注框信息,以list格式传入
    output:
        image:  PIL格式的图像
        boxes_changed:  改变后的标注框,无改变时为原始框,list格式
    """
    assert isinstance(img, np.ndarray)
    dst = Image.fromarray(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB))
    dst = (random_noise(np.array(dst), mode=noise_type, seed=2020) * 255).astype(np.uint8)

    return dst, boxes


def blur(img, boxes, filter_type="gaussian", kernel=5):
    """
    func: 对图像进行模糊滤波
    input:
        image:  待增强原始图像,np
        filter_type:  模糊方式,包括"median"和"gaussian","mean"
        boxes:  图像中待检测物体的标注框信息,以list格式传入
    output:
        dst:  增强后图像,np
        boxes:  改变后的标注框,无改变时为原始框,list格式
    """
    assert isinstance(img, np.ndarray)
    dst = img.copy()
    if filter_type == "median":
        dst = cv2.medianBlur(dst, ksize=kernel)
    elif filter_type == "mean":
        dst = cv2.blur(dst, ksize=kernel)
    elif filter_type == "gaussian":
        dst = cv2.GaussianBlur(dst, (kernel, kernel), 1)
    return dst, boxes


def mix_up(img, back_img, boxes, alpha=0.5):
    assert isinstance(img, np.ndarray)
    assert isinstance(back_img, np.ndarray)
    if img.shape == back_img.shape:
        dst = cv2.addWeighted(img, alpha, back_img, 1 - alpha, 0)
    else:
        dst = None
    return dst, boxes
