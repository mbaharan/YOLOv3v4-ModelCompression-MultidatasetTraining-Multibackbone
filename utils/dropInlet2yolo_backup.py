'''
    Thanks to: Tiago Miguel Rodrigues de Almeida (https://github.com/tmralmeida)
    The copy borrowed from:
        https://github.com/tmralmeida/bag-of-models
    and then modified.
'''

from math import sqrt
import os
import sys
import json
from pathlib import Path
import glob
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
import cv2
import numpy as np
from numpy.lib.function_base import select

_SMALL = 32*32
_LARGE = 96*96


def CheckForLess(list1, val):
    return(all(x <= val for x in list1))


def box2d_to_yolo(box2d, IMG_WIDTH=800, IMG_HEIGHT=600):
    # box2d = [x1 y1 x2 y2]
    x1 = box2d[0] / IMG_WIDTH
    x2 = box2d[2] / IMG_WIDTH
    y1 = box2d[1] / IMG_HEIGHT
    y2 = box2d[3] / IMG_HEIGHT

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = abs(x2 - x1)
    height = abs(y2 - y1)

    return cx, cy, width, height


class DropInlet2YOLO():
    def __init__(self, path):
        self.PATH = path

        self.imgs = {}
        self.lbls = {}
        self.labels_file = {}
        self.len_per_mode = {}

        for mode in ['train', 'val']:
            print("-> Opening the JSON file for {} mode.".format(mode))
            path_images = os.path.join(self.PATH, mode, 'images', mode)
            self.imgs[mode] = os.listdir(path_images)

            path_lbls = os.path.join(
                self.PATH, mode, 'annotations', 'instances_{}2017.json'.format(mode))
            self.lbls[mode] = open(path_lbls)
            self.labels_file[mode] = json.load(self.lbls[mode])
            # self.labels_file[mode]['annotations'][0]['id']
            # self.len_per_mode[mode] = len(self.labels_file[mode]["name"])

    def check_val_img_id(self, f):
        set_of_idx = dict()
        for idx, img_name in enumerate(self.labels_file['val']["name"]):
            x = f(img_name)
            if x in set_of_idx.keys():
                raise(ValueError(
                    "ID:{} for {} has already recored!!".format(x, img_name)))
            else:
                set_of_idx[x] = img_name
        assert(len(set_of_idx) == self.len_per_mode['val'])

    def convert_yolo_format(self, dst_folder, db_class_name_path, abs_adress=False, offset=0, augments_data=False, augmentation_per_image=10):

        bg_files = {}
        if augments_data:
            for mode in ['train', 'val']:
                bg_dir = os.path.join(self.PATH, 'bg', mode, '*.jpg')
                bg_files[mode] = glob.glob(bg_dir)

        if augments_data:
            dp_dir = os.path.join(self.PATH, 'dp', 'processed', '*.png')
            dp_files = glob.glob(dp_dir)

        for mode in ['val', 'train']:
            os.makedirs(os.path.join(dst_folder, mode,
                                     'images', mode), exist_ok=True)
            os.makedirs(os.path.join(dst_folder, mode,
                                     'labels', mode), exist_ok=True)

        for mode in ['val', 'train']:
            file_handler = open(os.path.join(
                dst_folder, mode, 'images', mode, '{}.txt'.format(mode)), 'w')
            for idx, img_data in enumerate(self.labels_file[mode]["images"]):
                img_name = img_data['file_name']
                img_id = img_data['id']

                lbl_file_row = list()
                for lbl in self.labels_file[mode]["annotations"]:
                    if lbl['image_id'] == img_id:
                        lbl_file_row.append(lbl)

                if len(lbl_file_row) == 0:
                    raise(ValueError(
                        "Could not find any label entery for image {}".format(img_name)))

                num_objs = len(lbl_file_row)

                if abs_adress:
                    file_name = "{}".format(
                        str(os.path.join(self.PATH, mode, 'images', mode, img_name)))
                else:
                    file_name = "{}".format(img_name)

                im = imageio.imread(file_name)  # cv2.imread(file_name)

                for j in range(augmentation_per_image):
                    data = ''
                    how_many_drop_in = random.randint(1, len(dp_files))
                    for i in range(num_objs):
                        dp_file = random.choice(dp_files)
                        dp = imageio.imread(dp_file)

                        selected_dp = True
                        if mode == 'val':
                            if random.random() < 0.5:
                                x1, y1, x2, y2 = lbl_file_row[i]['bbox']
                                selected_dp = False
                            else:
                                x1, y1, x2, y2 = 0, 0, dp.shape[1], dp.shape[0]
                        else:
                            x1, y1, x2, y2 = 0, 0, dp.shape[1], dp.shape[0]

                        # (lbl_file_row[i]['category_id'] - offset)
                        lbl_mapped = 0

                        if mode == 'val':
                            if not selected_dp:
                                if y2 > y1:
                                    drop_inlet = im[y1:y2, x1:x2]
                                else:
                                    drop_inlet = im[y2:y1, x1:x2]
                            else:
                                drop_inlet = dp
                        else:
                            drop_inlet = dp

                        bbs = BoundingBoxesOnImage([
                            BoundingBox(x1=0, x2=abs(x2-x1),
                                        y1=0, y2=abs(y2-y1))
                        ], shape=drop_inlet.shape[:2])

                        bg_file = random.choice(bg_files[mode])

                        bg = imageio.imread(bg_file)
                        h, w, _ = bg.shape

                        # 0.007 is scale of drop inlet area for bird-view
                        alpha = 0.009 - (random.random() * 0.001)
                        bb_area = abs(y2-y1)*abs(x2-x1)
                        scale_factor = sqrt((h*w*alpha)/bb_area)

                        aug = iaa.Sequential([
                            iaa.Affine(
                                scale=(scale_factor),
                                rotate=(random.randint(0, 90),
                                        random.randint(0, 90)),

                            ),
                            iaa.MotionBlur(k=random.randint(
                                5, 7), angle=[-45, 45])
                        ])

                        image_aug, bbs_aug = aug(
                            image=drop_inlet, bounding_boxes=bbs)

                        x1_filtered = int(
                            min(max(0, bbs_aug[0].x1), drop_inlet.shape[1]))
                        x2_filtered = int(
                            min(max(0, bbs_aug[0].x2), drop_inlet.shape[1]))
                        y1_filtered = int(
                            min(max(0, bbs_aug[0].y1), drop_inlet.shape[0]))
                        y2_filtered = int(
                            min(max(0, bbs_aug[0].y2), drop_inlet.shape[0]))

                        delta_x = (x2_filtered-x1_filtered)
                        delta_y = (y2_filtered-y1_filtered)

                        x_random = random.randint(
                            0, bg.shape[1]-delta_x)
                        y_random = random.randint(
                            0, bg.shape[0]-delta_y)

                        new_y2 = y_random+delta_y
                        new_x2 = x_random+delta_x

                        # bg[y_random:new_y2, x_random:new_x2] = image_aug

                        color = (0, 0, 0)
                        src2 = np.full(bg.shape, color, dtype=np.uint8)
                        #src2 = np.copy(bg)

                        # src2[y_random:new_y2, x_random:new_x2] = image_aug[]

                        src2[y_random:new_y2, x_random:new_x2] = image_aug[y1_filtered:y2_filtered,
                                                                           x1_filtered:x2_filtered]

                        dst = cv2.addWeighted(bg, 0.4, src2, 0.9, 0.0)
                        #dst = np.copy(src2)

                        
                        # Activate if you want to draw the bonding box
                        '''
                        bbs_final = BoundingBoxesOnImage([
                            BoundingBox(x1=x_random, x2=new_x2,
                                        y1=y_random, y2=new_y2)
                        ], shape=bg.shape)

                        test = bbs_final.draw_on_image(dst, size=2)
                        cv2.imwrite('test.jpg', test)
                        '''
                        
                        file_name_base = os.path.splitext(img_name)[0]
                        file_name_to_write = "{}".format(
                            str(os.path.join(dst_folder, mode, 'images', mode, '{}_{}.jpg'.format(file_name_base, j))))

                        cv2.imwrite(file_name_to_write, dst)
                        
                    # x1, y1, x2, y2 = (
                    #        coordinates['x1'], coordinates['y1'], coordinates['x2'], coordinates['y2'])

                        cx, cy, width, height = box2d_to_yolo(
                            (x_random, y_random, new_x2, new_y2), IMG_WIDTH=w, IMG_HEIGHT=h)

                        if not (CheckForLess([cx, cy, width, height], 1)):
                            raise(ValueError("Bonding box values <{},{},{},{}> are not correct for image size of <w={}, h={}>".format(
                                x_random, y_random, new_x2, new_y2, w, h)))

                        data = "{}{} {} {} {} {}\n".format(
                            data, lbl_mapped, cx, cy, width, height)

                    file_handler.writelines("{}\n".format(file_name_to_write))
                    label_file_name = (os.path.splitext(file_name_to_write)[
                        0]).replace('images', 'labels')
                    file_handler_label = open(
                        '{}.txt'.format(label_file_name), 'w')
                    file_handler_label.write(data)
                    file_handler_label.close()
                print("-> Creating {}.txt: {:.2f}%".format(mode,
                                                           ((idx + 1)*100/len(self.labels_file[mode]["images"]))), end='\r')
            print()
            file_handler.close()
        print('-> Writing dropInlet.name...')
        f = open(os.path.join(db_class_name_path, "dropInlet.name"), 'w')
        for k in self.labels_file["train"]["categories"]:
            f.write("{}\n".format(k["name"]))
        f.close()


def get_area(coordinates):
    return abs((coordinates['x2']-coordinates['x1']) * (coordinates['y2']-coordinates['y1']))


if __name__ == "__main__":
    path = "/mnt/AI_2TB/drop_inlet/"
    dst_path = '/mnt/AI_2TB/drop_inlet/augmented'
    file = Path(__file__).resolve()
    package_root_directory = file.parents[1]
    sys.path.append(str(package_root_directory))
    # from dataset import get_image_id
    obj = DropInlet2YOLO(path)
    # bdd100k.find_images(['city street', 'tunnel'], ['daytime', 'night'],
    #                    os.path.join(package_root_directory, 'val_samples'), 1000)
    # bdd100k.analyze_db(os.path.join(os.getcwd(), 'stats'))
    # bdd100k.check_val_img_id(get_image_id)
    obj.convert_yolo_format(dst_path, os.path.join(
        os.getcwd(), 'YOLO_V4', 'data'), abs_adress=True, offset=1, augments_data=True)
