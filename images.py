import os

import cv2  # type:ignore
import numpy as np


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def flatten(img):
    return np.reshape(img, (int(np.round(np.prod(img.shape[:-1]))), img.shape[-1]))


def sample_image(img, x, y, radius=5):
    # Samples a (2*radius, 2*radius) box of pixels surrounding the point (x,y) in the image. 0 < x, y < 1
    h, w, c = img.shape
    row, col = int(y * h), int(x * w)
    samples = img[row-radius:row+radius, col-radius:col+radius, :]
    avg = np.mean(flatten(samples), axis=0)
    return avg


def get_samples(img, patches=(6,4)):
    samples = np.zeros((patches[0], patches[1], 3))
    for patch_row in range(patches[0]):
        for patch_col in range(patches[1]):
            sample = sample_image(img, (patch_col*2+1) / (patches[1]*2), (patch_row*2+1) / (patches[0]*2))
            samples[patch_row, patch_col, :] = sample
    return samples