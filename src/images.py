import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2  # type:ignore
import numpy as np
from typing import Tuple, List, Union
import src.color_conversions as color_conversions


def open_image(image_fn: str) -> np.ndarray:
    print(f"Reading: {image_fn}")
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def flatten(img):
    return np.reshape(img, (int(np.round(np.prod(img.shape[:-1]))), img.shape[-1]))


def sample_image(img, x, y, radius_erode=0.5, patches=(6, 4)):
    # Samples a (2*radius, 2*radius) box of pixels surrounding the point (x,y) in the image. 0 < x, y < 1
    h, w, c = img.shape
    row_radius = int(radius_erode * (h / (2 * patches[0])))
    col_radius = int(radius_erode * (w / (2 * patches[1])))
    row, col = int(y * h), int(x * w)
    pos = [row - row_radius, col - col_radius, row + row_radius, col + col_radius]
    samples = img[pos[0] : pos[2], pos[1] : pos[3], :]
    avg = np.mean(flatten(samples), axis=0)
    return avg, pos


def show_image(img, title="image"):
    img = img.astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def get_samples(
    img, patches: Tuple[int, int] = (6, 4), flat: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    samples = np.zeros((patches[0], patches[1], 3))
    sample_positions = np.zeros((patches[0], patches[1], 4))
    for patch_row in range(patches[0]):
        for patch_col in range(patches[1]):
            sample, pos = sample_image(
                img,
                (patch_col * 2 + 1) / (patches[1] * 2),
                (patch_row * 2 + 1) / (patches[0] * 2),
                patches=patches,
            )
            samples[patch_row, patch_col, :] = sample
            sample_positions[patch_row, patch_col, :] = pos

    if patches[0] > patches[1]:
        # We have a sideways chart (Tall). Assume the bottom left is A1.
        samples = samples[::-1, :, :]
        sample_positions = sample_positions[::-1, :, :]
        pass
    else:
        # We have a normal, landscape chart. Assume top left is A1.
        samples = np.transpose(samples, [1, 0, 2])
        sample_positions = np.transpose(sample_positions, [1, 0, 2])
    if flat:
        samples = flatten(samples)
        sample_positions = flatten(sample_positions)
    return samples, sample_positions


def draw_samples(
    img: np.ndarray,
    source_chart: color_conversions.Chart,
    reference_chart: Union[color_conversions.ReferenceChart, color_conversions.RGBChart],
    sample_positions: np.ndarray,
    show: bool = True,
    title: str = "Image",
) -> np.ndarray:
    if isinstance(reference_chart, color_conversions.ReferenceChart):
        reference_colors = (
            reference_chart.convert_to_xyz(reference_chart.reference_white)
            .chromatic_adaptation(
                reference_chart.reference_white,
                color_conversions.GAMUT_REC709.white.convert_to_xyz(),
            )
            .convert_to_rgb(
                color_conversions.GAMUT_REC709.get_conversion_to_xyz().inverse()
            )
            .colors
        )
    elif isinstance(reference_chart, color_conversions.RGBChart):
        reference_colors = reference_chart.colors
    else:
        raise ValueError(f"Unexpected type for reference_chart: {type(reference_chart)}")
    source_colors = source_chart.colors
    sample_positions = flatten(sample_positions).astype(int)
    canvas = img.copy()
    for source_color, ref_color, (r1, c1, r2, c2) in zip(
        source_colors, reference_colors, sample_positions
    ):
        midpoint = [int(0.5 * (r1 + r2)), int(0.5 * (c1 + c2))]
        canvas[r1:r2, c1 : midpoint[1]] = source_color
        canvas[r1:r2, midpoint[1] : c2] = ref_color
        cv2.rectangle(canvas, (c1, r1), (c2, r2), (1.0, 1.0, 1.0))

    if show:
        show_image(np.maximum(canvas, 0.0) ** (1.0 / 2.4), title=title)
    return canvas
