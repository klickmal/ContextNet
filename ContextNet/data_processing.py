"""
Reference:
https://github.com/ooleksyuk/CarND-Semantic-Segmentation/blob/master/helper_cityscapes.py

"""

import random
import numpy as np
import os.path
import scipy.misc
import shutil
import time
import tensorflow as tf
from glob import glob
import matplotlib.pyplot as plt


from collections import namedtuple
from timeit import default_timer as timer

Label = namedtuple('Label', ['name', 'color'])
"""
# in case you would use cv2
def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])
"""
# cf https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

# num_classes 20
# background (unlabeled) + 19 classes as per official benchmark
# cf "The Cityscapes Dataset for Semantic Urban Scene Understanding"
label_defs = [
    Label('unlabeled',     (0,     0,   0)),
    Label('dynamic',       (111,  74,   0)),
    Label('ground',        ( 81,   0,  81)),
    Label('road',          (128,  64, 128)),
    Label('sidewalk',      (244,  35, 232)),
    Label('parking',       (250, 170, 160)),
    Label('rail track',    (230, 150, 140)),
    Label('building',      ( 70,  70,  70)),
    Label('wall',          (102, 102, 156)),
    Label('fence',         (190, 153, 153)),
    Label('guard rail',    (180, 165, 180)),
    Label('bridge',        (150, 100, 100)),
    Label('tunnel',        (150, 120,  90)),
    Label('pole',          (153, 153, 153)),
    Label('traffic light', (250, 170,  30)),
    Label('traffic sign',  (220, 220,   0)),
    Label('vegetation',    (107, 142,  35)),
    Label('terrain',       (152, 251, 152)),
    Label('sky',           ( 70, 130, 180)),
    Label('person',        (220,  20,  60)),
    Label('rider',         (255,   0,   0)),
    Label('car',           (  0,   0, 142)),
    Label('truck',         (  0,   0,  70)),
    Label('bus',           (  0,  60, 100)),
    Label('caravan',       (  0,   0,  90)),
    Label('trailer',       (  0,   0, 110)),
    Label('train',         (  0,  80, 100)),
    Label('motorcycle',    (  0,   0, 230)),
    Label('bicycle',       (119, 11, 32))]


def build_file_list(images_root, labels_root, sample_name):
    image_sample_root = images_root + '/' + sample_name
    image_root_len = len(image_sample_root)
    label_sample_root = labels_root + '/' + sample_name
    image_files = glob(image_sample_root + '/**/*png')
    file_list = []
    for f in image_files:
        f_relative = f[image_root_len:]
        f_dir = os.path.dirname(f_relative)
        f_base = os.path.basename(f_relative)
        f_base_gt = f_base.replace('leftImg8bit', 'gtFine_color')
        f_label = label_sample_root + f_dir + '/' + f_base_gt
        if os.path.exists(f_label):
            file_list.append((f, f_label))
    return file_list


def load_data(data_folder):

    images_root = os.path.join(data_folder, 'leftImg8bit')
    labels_root = os.path.join(data_folder, 'gtFine')

    train_images = build_file_list(images_root, labels_root, 'train')
    valid_images = build_file_list(images_root, labels_root, 'val')
    test_images = build_file_list(images_root, labels_root, 'test')
    num_classes = len(label_defs)
    label_colors = {i: np.array(l.color) for i, l in enumerate(label_defs)}
    #image_shape = (256, 512)

    return train_images, valid_images, test_images, num_classes, label_colors #, image_shape


def bc_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def gen_batch_function(image_paths, image_shape, test = False):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        random.shuffle(image_paths)

        while True:
            
            for batch_i in range(0, len(image_paths), batch_size):
                image_files = image_paths[batch_i:batch_i+batch_size]

                images = []
                images_lowR = []
                labels = []

                for f in image_files:
                    image_file = f[0]
                    gt_image_file = f[1]

                    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                    #gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file, mode='RGB'), image_shape)
                    gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file, mode='RGB'), (int(image_shape[0]/8), int(image_shape[1]/8)))

                    contrast = random.uniform(0.85, 1.15)  # Contrast augmentation
                    bright = random.randint(-45, 30)  # Brightness augmentation
                    image = bc_img(image, contrast, bright)

                    #label_bg = np.zeros([image_shape[0], image_shape[1]], dtype=bool)
                    label_bg = np.zeros([int(image_shape[0]/8), int(image_shape[1]/8)], dtype=bool)
                    label_list = []
                    for ldef in label_defs[1:]:
                        label_current = np.all(gt_image == np.array(ldef.color), axis=2)
                        label_bg |= label_current                   
                        label_list.append(label_current)

                    label_bg = ~label_bg
                    label_all = np.dstack([label_bg, *label_list])

                    label_all = label_all.astype(np.float32)
                    image_resize = scipy.misc.imresize(image, (int(image_shape[0]/4), int(image_shape[1]/4)))
                    
                    images.append(image)
                    images_lowR.append(image_resize)
                    labels.append(label_all)
                    
                    image_set = [np.array(images), np.array(images_lowR)]
                if not test:
                    yield image_set, np.array(labels)
                else:
                    yield image_set
    return get_batches_fn

