from typing import Union
from enum import Enum
import numpy as np

class ImageState(str, Enum):
    RGB = "RGB"
    XYZ = "XYZ"
    xyY = "xyY"
    LAB = "LAB"


class Chart:
    colors: np.ndarray
    state: ImageState

    def __init__(self, colors: Union[np.ndarray, list]) -> None:
        self.colors = np.array(colors)
        self._check_shape()

    def set_colors(self, colors: np.ndarray) -> None:
        self.colors = colors
        self._check_shape()

    def get_colors(self) -> np.ndarray:
        return self.colors

    def _check_shape(self):
        assert len(self.colors.shape) == 2 and self.colors.shape[1] == 3


class RGBChart(Chart):
    state = ImageState.RGB

    def convert_to_xyz(self, mat: np.ndarray) -> 'XYZChart':
        assert mat.shape == (3, 3), f"Expected 3x3 matrix but found {mat.shape}"
        out_colors = self.colors @ mat
        out = XYZChart(out_colors)
        return out


class LABChart(Chart):
    state = ImageState.LAB

    def convert_to_xyz(self, white_xyz: 'XYZChart') -> 'XYZChart':
        assert white_xyz.state == ImageState.XYZ, f"convert_xyz_to_lab expected XYZ colors. Got {white_xyz.state}"
        assert white_xyz.colors.shape == (1, 3), f"Unexpected white_xyz shape: {white_xyz.colors.shape}"

        eps = 216 / 24389
        k = 24389 / 27

        fxyz = np.zeros_like(self.colors)
        fxyz[:, 1] = (self.colors[:, 0] + 16) / 116
        fxyz[:, 2] = fxyz[:, 1] - (self.colors[:, 2] / 200)
        fxyz[:, 0] = (self.colors[:, 1] / 500) + fxyz[:, 1]

        xyz_r = np.zeros_like(fxyz)
        mask = (fxyz**3) > eps
        xyz_r[mask] = fxyz[mask]**3
        xyz_r[~mask] = (116 * fxyz[~mask] - 16) / k

        xyz_colors = xyz_r * white_xyz.colors
        xyz = XYZChart(xyz_colors)
        return xyz

    def compute_delta_e(self, other: 'LABChart') -> float:
        return float(np.mean(np.sum((self.colors - other.colors)**2, axis=1)**0.5))


class XYZChart(Chart):
    state = ImageState.XYZ

    def convert_to_lab(self, white_xyz: 'XYZChart') -> LABChart:
        assert white_xyz.state == ImageState.XYZ, f"convert_xyz_to_lab expected XYZ colors. Got {white_xyz.state}"
        assert white_xyz.colors.shape == (1, 3), f"Unexpected white_xyz shape: {white_xyz.colors.shape}"

        xyz_r = self.colors / white_xyz.colors
        eps = 216 / 24389
        k = 24389 / 27

        fxyz = np.zeros_like(self.colors)
        mask = xyz_r > eps
        fxyz[mask] = xyz_r[mask]**(1.0/3.0)
        fxyz[~mask] = (k * xyz_r[~mask] + 16) / 116

        lab_colors = np.zeros_like(fxyz)
        lab_colors[:, 0] = 116 * fxyz[:, 1] - 16  # L
        lab_colors[:, 1] = 500 * (fxyz[:, 0] - fxyz[:, 1])  # a
        lab_colors[:, 2] = 200 * (fxyz[:, 1] - fxyz[:, 2])  # b

        lab = LABChart(lab_colors)
        return lab


    def convert_to_xyy(self, white_xyz: 'XYZChart') -> 'XYYChart':
        assert white_xyz.state == ImageState.XYZ, f"convert_xyz_to_lab expected XYZ colors. Got {white_xyz.state}"
        assert white_xyz.colors.shape == (1, 3), f"Unexpected white_xyz shape: {white_xyz.colors.shape}"

        white_xyy = [white_xyz.colors[0, 0] / np.sum(white_xyz.colors), white_xyz.colors[0, 2] / np.sum(white_xyz.colors)]

        xyy_colors = np.zeros_like(self.colors)
        zeros = (np.sum(self.colors, axis=1) == 0)
        xyy_colors[:, 0] = self.colors[:, 0] / np.sum(self.colors, axis=1)
        xyy_colors[:, 1] = self.colors[:, 1] / np.sum(self.colors, axis=1)
        xyy_colors[:, 2] = self.colors[:, 1]
        xyy_colors[zeros, :] = np.array([white_xyy[0], white_xyy[1], 0.0])

        xyy = XYYChart(xyy_colors)
        return xyy

class XYYChart(Chart):
    state = ImageState.xyY

    def convert_to_xyz(self) -> XYZChart:
        zeros = (self.colors[:, 1] == 0)

        xyz_colors = np.zeros_like(self.colors)
        xyz_colors[:, 0] = self.colors[:, 0] * self.colors[:, 2] / self.colors[:, 1]
        xyz_colors[:, 1] = self.colors[:, 2]
        xyz_colors[:, 2] = (1 - self.colors[:, 0] - self.colors[:, 1]) * self.colors[:, 2] / self.colors[:, 1]
        xyz_colors[zeros] = 0

        xyz = XYZChart(xyz_colors)
        return xyz

STD_A: XYZChart = XYYChart(colors=np.array([[0.34842, 0.35161, 1.0]])).convert_to_xyz()
STD_C: XYZChart = XYYChart(colors=np.array([[0.31006, 0.31616, 1.0]])).convert_to_xyz()
STD_D65: XYZChart = XYYChart(colors=np.array([[0.31271, 0.32902, 1.0]])).convert_to_xyz()
STD_E: XYZChart = XYZChart(colors=np.array([[1.0, 1.0, 1.0]]))
