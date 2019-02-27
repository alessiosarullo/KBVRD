from enum import Enum

from PIL import ImageOps


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class SquarePad:
    def __call__(self, img, pixel_mean):
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=(int(pixel_mean[0] * 256), int(pixel_mean[1] * 256), int(pixel_mean[2] * 256)))
        return img_padded
