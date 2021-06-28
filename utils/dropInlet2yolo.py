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
from shutil import copyfile
import time

_SMALL = 32*32
_LARGE = 96*96


def CheckForLess(list1, val):
    return(all(x <= val for x in list1))


classes = {
    'none-defective': 0,
    'defective': 1
}


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
    def __init__(self, path, val_ration=0.3, seed=1):

        if seed > 0:
            random.seed(seed)
        else:
            random.seed(int(time.time()))

        self.PATH = path

        self.imgs = {}
        self.lbls = {}
        self.label_files = {}
        self.len_per_mode = {}

        img_path = os.path.join(self.PATH, '*.jpg')
        self.imgs['all'] = glob.glob(img_path)

        label_dir = os.path.join(self.PATH, '*.json')
        self.label_files = glob.glob(label_dir)
        assert(len(self.label_files) > 0), "Couldn't find any JSON label files in {}.".format(
            label_dir)
        print('Starting analyzing files:')

        self.obj_type_num = {}
        self.obj_type_files = {}
        self.useless_files = []
        for k in classes:
            self.obj_type_num[k] = 0
            self.obj_type_files[k] = []

        self.idx = [0]
        self.img_in_label_files = []
        self.imgs_per_json = {}

        for json_f in self.label_files:

            lbl_file_handler = open(json_f)
            lbl = json.load(lbl_file_handler)
            json_key_name = Path(json_f).stem
            self.imgs_per_json[json_key_name] = []

            old_format = False
            if "images" in lbl:
                old_format = True

            if not old_format:
                self.process_new_format(lbl, json_key_name)
            else:
                self.process_old_format(lbl)

        self.total = 0
        for k in classes:
            self.total += self.obj_type_num[k]

        print('-> Number of defective drop inlets: {}. Suggested weight: {}'.format(
            self.obj_type_num['defective'], self.total/self.obj_type_num['defective']))
        print('-> Number of non-defective drop inlets: {}. Suggested weight: {}'.format(
            self.obj_type_num['none-defective'], self.total/self.obj_type_num['none-defective']))

        print('-> Balancing the distribution.')

        final_data = {}
        for k in classes:
            random.shuffle(self.obj_type_files[k])
            final_data[k] = list()

        if self.obj_type_num['none-defective'] > self.obj_type_num['defective']:
            final_data['defective'] = self.obj_type_files['defective']
            final_data['none-defective'] = self.obj_type_files['none-defective'][:len(
                self.obj_type_files['defective'])]
        else:
            final_data['none-defective'] = self.obj_type_files['none-defective']
            final_data['defective'] = self.obj_type_files['defective'][:
                                                                       len(self.obj_type_files['none-defective'])]

        self.imgs['val'] = []
        self.imgs['train'] = []
        for k in classes:
            val_index = int(val_ration*len(final_data[k]))
            self.imgs['val'].extend(final_data[k][:val_index])
            self.imgs['train'].extend(final_data[k][val_index:])
        assert(len(set(self.imgs['train']) & set(self.imgs['val'])) == 0)
        print()

    # -------------------------------------------------
    def process_old_format(self, lbl):
        for img_data in lbl["images"]:
            img_name = img_data['file_name']
            img_id = img_data['id']

            for lbl_r in lbl["annotations"]:
                if lbl_r['image_id'] == img_id:
                    key = 'none-defective'
                    if img_name not in self.obj_type_files[key]:
                        self.obj_type_num[key] = self.obj_type_num[key] + 1
                        self.obj_type_files[key].append(img_name)
                    else:
                        print('-> File {} has already seen.'.format(img_name))

            self.idx[0] += 1

    # -------------------------------------------------

    def process_new_format(self, lbl, json_key_name):
        for img_data in lbl:
            img_name = img_data['External ID']
            if img_name not in self.img_in_label_files:
                self.img_in_label_files.append(img_name)
                self.imgs_per_json[json_key_name].append(img_name)

                use_less = False
                img_file_path =  os.path.join(self.PATH, img_name)
                if len(img_data['Label']) == 0 or img_file_path not in self.imgs['all']:
                    self.useless_files.append(img_name)
                    use_less = True
                    for key in classes:
                        if img_name in self.obj_type_files[key]:
                            self.obj_type_files[key].remove(img_name)
                            self.obj_type_num[key] -= 1

                if not use_less:
                    # To make sure we are not duplicating images with more objects.
                    save_before = False
                    for obj in img_data['Label']['objects']:
                        lbl_mapped = obj['value']
                        key = lbl_mapped.split('_')[0]
                        if 'non' in key:
                            key = 'none-defective'
                        else:
                            key = 'defective'
                        self.obj_type_num[key] = self.obj_type_num[key] + 1
                        if not save_before:
                            self.obj_type_files[key].append(img_name)
                        save_before = True
            else:
                if len(img_data['Label']) == 0:
                    for k in classes:
                        if img_name in self.obj_type_files[key]:
                            self.obj_type_files[key].remove(img_name)
                            self.obj_type_num[key] -= 1

                for k, v in self.imgs_per_json.items():
                    if img_name in v:
                        print('!> While parsing {}, I found that the entry for file {} has already observed in `{}.json`. Ignoring the entry!'.format(
                            json_key_name, img_name, k))

            self.idx[0] += 1

            # print(
            #    "-> Processed({: .2f}%).".format(((self.idx[0])*100/len(self.imgs['all']))), end='\r')

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

    def convert_yolo_format(self, dst_folder, db_class_name_path, abs_address=False, do_resize=False, resize=(600, 800)):

        for mode in ['val', 'train']:
            os.makedirs(os.path.join(dst_folder, mode,
                                     'images', mode), exist_ok=True)
            os.makedirs(os.path.join(dst_folder, mode,
                                     'labels', mode), exist_ok=True)

        file_handler = {}
        for mode in ['val', 'train']:
            file_handler[mode] = open(os.path.join(
                dst_folder, mode, 'images', mode, '{}.txt'.format(mode)), 'w')
        idx = 0
        idx_img_valid = 0
        idx_img_train = 0
        for json_f in self.label_files:
            lbl_file_handler = open(json_f)
            lbl = json.load(lbl_file_handler)

            if not "images" in lbl:
                for _, img_data in enumerate(lbl):

                    data = ''

                    img_name = img_data['External ID']

                    mode = ''
                    if img_name in self.imgs['train']:
                        mode = 'train'
                        idx_img_train += 1
                    elif img_name in self.imgs['val']:
                        mode = 'val'
                        idx_img_valid += 1
                    # elif img_name in self.useless_files:
                    #    break
                    # else:
                    #    break
                        # raise(ValueError(
                        #    "File {} is not found in the image collections.". format(img_name)))

                    if mode is not '':
                        if abs_address:
                            file_name = "{}".format(
                                str(os.path.join(dst_folder, mode, 'images', mode, img_name)))
                        else:
                            file_name = "{}".format(img_name)

                        if not do_resize:
                            src_file = os.path.join(self.PATH, img_name)
                            copyfile(src_file, file_name)
                            # cv2.imread(file_name)
                            im = imageio.imread(file_name)
                            w = im.shape[1]
                            h = im.shape[0]
                        else:
                            im_tmp = imageio.imread(os.path.join(
                                self.PATH, img_name), as_gray=False, pilmode="RGB")  # cv2.imread(file_name)
                            aug = iaa.Resize(
                                {"height": resize[0], "width": resize[1]})
                            im = aug(image=im_tmp)
                            w = im.shape[1]
                            h = im.shape[0]
                            imageio.imwrite(file_name, im) #cv2.imwrite(file_name, im)

                        for obj in img_data['Label']['objects']:
                            idx += 1

                            x1, y1, wd, ht = obj['bbox']['left'], obj['bbox']['top'], obj['bbox']['width'], obj['bbox']['height']

                            lbl_str = obj['value']
                            key = lbl_str.split('_')[0]
                            if 'non' in key:
                                key = 'none-defective'
                            else:
                                key = 'defective'

                            lbl_mapped = classes[key]

                            if do_resize:
                                bbs_tmp = BoundingBoxesOnImage([
                                    BoundingBox(x1=x1, x2=wd+x1,
                                                y1=y1, y2=ht+y1)
                                ], shape=im_tmp.shape)
                                _, bbx = aug(
                                    image=im_tmp, bounding_boxes=bbs_tmp)
                                x1 = bbx[0].x1
                                x2 = bbx[0].x2
                                y1 = bbx[0].y1
                                y2 = bbx[0].y2
                            else:
                                x2 = x1+wd
                                y2 = y1+ht

                            '''
                            bbs_final = BoundingBoxesOnImage([
                                BoundingBox(x1=x1, x2=x2,
                                        y1=y1, y2=y2)
                            ], shape=im.shape)
                            test = bbs_final.draw_on_image(im, size=2)
                            imageio.imwrite('test.jpg', test)
                            '''

                    #file_name_base = os.path.splitext(img_name)[0]
                    # file_name_to_write = "{}".format(
                    #    str(os.path.join(dst_folder, mode, 'images', mode, '{}_{}.jpg'.format(file_name_base))))

                            cx, cy, width, height = box2d_to_yolo(
                                (x1, y1, x2, y2), IMG_WIDTH=w, IMG_HEIGHT=h)

                            if not (CheckForLess([cx, cy, width, height], 1)):
                                raise(ValueError("Bonding box values <{},{},{},{}> are not correct for image size of <w={}, h={}>".format(
                                    x1, y1, wd, ht, w, h)))

                            data = "{}{} {} {} {} {}\n".format(
                                data, lbl_mapped, cx, cy, width, height)

                    if mode is not '':
                        label_file_name = (os.path.splitext(file_name)[
                            0]).replace('images', 'labels')
                        file_handler_label = open(
                            '{}.txt'.format(label_file_name), 'w')
                        file_handler_label.write(data)
                        file_handler_label.close()
                        file_handler[mode].write('{}\n'.format(file_name))
                    print(
                        "-> Creating YoloV4 label files...", end='\r')

            else:
                for idx, img_data in enumerate(lbl["images"]):
                    img_name = img_data['file_name']
                    img_id = img_data['id']

                    mode = ''
                    if img_name in self.imgs['train']:
                        mode = 'train'
                        idx_img_train += 1
                    elif img_name in self.imgs['val']:
                        mode = 'val'
                        idx_img_valid += 1
 
                    if not mode == '':
                        lbl_file_row = list()
                        for lbl_r in lbl["annotations"]:
                            if lbl_r['image_id'] == img_id and img_name:
                                lbl_file_row.append(lbl_r)

                        if len(lbl_file_row) == 0:
                            raise(ValueError(
                                "Could not find any label entery for image {}".format(img_name)))

                        num_objs = len(lbl_file_row)

                        if abs_address:
                            file_name = "{}".format(
                                str(os.path.join(dst_folder, mode, 'images', mode, img_name)))
                        else:
                            file_name = "{}".format(img_name)

                        src_file = os.path.join(self.PATH, img_name)
                        copyfile(src_file, file_name)

                        im = imageio.imread(file_name)  # cv2.imread(file_name)
                        w = im.shape[1]
                        h = im.shape[0]
                        data = ''
                        for i in range(num_objs):
                            x1, y1, wd, ht = lbl_file_row[i]['bbox']
                            # old format files are all healthy
                            lbl_mapped = 0
                            '''
                            bbs_final = BoundingBoxesOnImage([
                                BoundingBox(x1=x1, x2=wd+x1,
                                    y1=y1, y2=ht+y1)
                            ], shape=im.shape)
                            test = bbs_final.draw_on_image(im, size=2)
                            cv2.imwrite('test.jpg', test)
                            '''
                            cx, cy, width, height = box2d_to_yolo(
                                (x1, y1, wd+x1, ht+y1), IMG_WIDTH=w, IMG_HEIGHT=h)

                            if not (CheckForLess([cx, cy, width, height], 1)):
                                raise(ValueError("Bonding box values <{},{},{},{}> are not correct for image size of <w={}, h={}>".format(
                                    x1, y1, wd, ht, w, h)))

                            data = "{}{} {} {} {} {}\n".format(
                                data, lbl_mapped, cx, cy, width, height)
                
                    if not mode == '':
                        label_file_name = (os.path.splitext(file_name)[
                            0]).replace('images', 'labels')
                        file_handler_label = open(
                            '{}.txt'.format(label_file_name), 'w')
                        file_handler_label.write(data)
                        file_handler_label.close()
                        file_handler[mode].write('{}\n'.format(file_name))
                        print(
                            "-> Creating YoloV4 label files...", end='\r')
        print()
        for mode in ['val', 'train']:
            file_handler[mode].close()
        print('-> Writing dropInlet.name...')
        f = open(os.path.join(db_class_name_path, "dropInlet.name"), 'w')
        for k in classes.keys():
            f.write("{}\n".format(k))
        f.close()


def get_area(coordinates):
    return abs((coordinates['x2']-coordinates['x1']) * (coordinates['y2']-coordinates['y1']))


if __name__ == "__main__":
    path = "/mnt/AI_2TB/drop_inlet/new_dataset/Correctly_Labeled_Images/all_files"
    dst_path = '/mnt/AI_2TB/drop_inlet/new_dataset/merged_new'
    file = Path(__file__).resolve()
    package_root_directory = file.parents[1]
    sys.path.append(str(package_root_directory))
    obj = DropInlet2YOLO(path)
    obj.convert_yolo_format(dst_path, os.path.join(
        os.getcwd(), 'YOLO_V4', 'data'), abs_address=True, do_resize=False)
