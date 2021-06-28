import imageio
import os
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def yolo_to_box2d(box2d, IMG_WIDTH=800, IMG_HEIGHT=600):
    # box2d = [x1 y1 x2 y2]
    cx = box2d[0]  # * IMG_WIDTH
    cy = box2d[1]  # * IMG_HEIGHT
    h = box2d[2]  # * IMG_HEIGHT
    w = box2d[3]  # * IMG_WIDTH

    x1, y1 = cx-w/2, cy-h/2
    x2, y2 = cx+w/2, cy+h/2

    return (x1*IMG_WIDTH), (y1*IMG_HEIGHT), (x2*IMG_WIDTH), (y2*IMG_HEIGHT)


file_name_base_name = '202103041451520961'
base_folder = '/mnt/AI_2TB/drop_inlet/new_dataset/merged/'
subfolder = 'val/labels/val'
predicted_image_folder = 'YOLO_V4/val_test/'

img_file = os.path.join(predicted_image_folder,
                        '{}.jpg'.format(file_name_base_name))

img = imageio.imread(img_file)

label_file_name = os.path.join(
    base_folder, subfolder, '{}.txt'.format(file_name_base_name))
file_handler = open(label_file_name, 'r')
_, xc, yc, h, w = file_handler.readline().split(' ')

x2, y2, x1, y1 = yolo_to_box2d(
    [float(xc), float(yc), float(h), float(w)], img.shape[1], img.shape[0])

bbs_final = BoundingBoxesOnImage([
    BoundingBox(x1=x1, x2=x2,
                y1=y1, y2=y2)
], shape=img.shape)

test = bbs_final.draw_on_image(img, size=2)
imageio.imwrite('{}_IoU.jpg'.format(file_name_base_name), test)
