class DefaultConfigs(object):
    ori_img_path = "./data/ori/images"
    ori_csv_path = "./data/ori/train_labels.csv"
    aug_img_path = "./data/aug/images"
    aug_csv_path = "./data/aug/train_labels.csv"
    img_format = "jpg"
    p_brightness = 0.8
    p_color = 0.8
    p_contrast = 0.8
    p_noise = 0.8
    p_blur = 0.8
    p_rotate = 0.8
    p_horizontal_flip = 0.8
    p_vertical_flip = 0.8
    p_scale = 0.8
    p_translation = 0.8
    p_crop = 0.8
    p_mix_up = 0.8
    # *****************************
    is_vertical_flip = True
    is_scale = True
    is_translation = True
    is_crop = True
    is_brightness = True
    is_mix_up = True
    is_horizontal_flip = True
    is_rotate = True
    is_blur = True
    is_color = True
    is_noise = True
    is_contrast = True

    rotate_angle = (90, 180, 270)
    mix_up_back_img = '/mnt/HD_2TB/code/tianchi_1/data/VOC2007/normal_Images'


opt = DefaultConfigs()
