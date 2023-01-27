from typing import Union
from enum import Enum
import numpy as np

class ImageState(str, Enum):
    RGB = "RGB"
    XYZ = "XYZ"
    xyY = "xyY"
    LAB = "LAB"

class ColorMatrix:
    mat: np.ndarray  # Assumed that we will multiply mat @ rgb, with column vector rgb
    from_state: ImageState
    to_state: ImageState

    def __init__(self, mat: np.ndarray, from_state: ImageState, to_state: ImageState) -> None:
        self.mat = mat
        self.from_state = from_state
        self.to_state = to_state
        assert self.mat.shape == (3, 3)

    def inverse(self) -> "ColorMatrix":
        m = ColorMatrix(np.linalg.pinv(self.mat), self.to_state, self.from_state)
        return m

    def composite(self, other: "ColorMatrix") -> "ColorMatrix":
        assert self.to_state == other.from_state
        m = ColorMatrix(other.mat @ self.mat, self.from_state, other.to_state)
        return m

    @staticmethod
    def get_chromatic_adaptation_matrix(input_white_point: "XYZChart", output_white_point: "XYZChart") -> "ColorMatrix":
        assert input_white_point.colors.shape == (1, 3)
        assert output_white_point.colors.shape == (1, 3)
        # CMCAT 2000 matrix.
        M: np.ndarray = np.array([
            [ 0.7982, 0.3389, -0.1371],
            [-0.5918, 1.5512,  0.0406],
            [ 0.0008, 0.0239,  0.9753]
        ])
        M_inv = np.linalg.pinv(M)
        result = M_inv @ np.diag((output_white_point.colors / input_white_point.colors)[0]) @ M
        return ColorMatrix(result, ImageState.XYZ, ImageState.XYZ)

class Gamut:
    red: "XYYChart"
    green: "XYYChart"
    blue: "XYYChart"
    white: "XYYChart"

    def __init__(self, red: "XYYChart", green: "XYYChart", blue: "XYYChart", white: "XYYChart") -> None:
        self.red = red.normalize()
        self.green = green.normalize()
        self.blue = blue.normalize()
        self.white = white.normalize()
        assert self.red.colors.shape == (1, 3)
        assert self.green.colors.shape == (1, 3)
        assert self.blue.colors.shape == (1, 3)
        assert self.white.colors.shape == (1, 3)
        assert self.red.colors[0, 2] == 1.0
        assert self.green.colors[0, 2] == 1.0
        assert self.blue.colors[0, 2] == 1.0
        assert self.white.colors[0, 2] == 1.0

    def get_conversion_to_xyz(self) -> ColorMatrix:
        xyzr = self.red.convert_to_xyz()
        xyzg = self.green.convert_to_xyz()
        xyzb = self.blue.convert_to_xyz()
        xyzw = self.white.convert_to_xyz()

        xyz = np.zeros((3, 3))
        xyz[:, [0]] = xyzr.colors.T
        xyz[:, [1]] = xyzg.colors.T
        xyz[:, [2]] = xyzb.colors.T
        s = xyzw.colors @ np.linalg.pinv(xyz).T  # shape (1, 3)
        m = xyz * s
        return ColorMatrix(m, ImageState.RGB, ImageState.XYZ)

    def get_conversion_to_gamut(self, other: "Gamut") -> ColorMatrix:
        m1 = self.get_conversion_to_xyz()
        cat = ColorMatrix.get_chromatic_adaptation_matrix(
            self.white.convert_to_xyz(),
            other.white.convert_to_xyz(),
        )
        m2 = other.get_conversion_to_xyz().inverse()
        m3 = m1.composite(cat).composite(m2)
        return m3

    @staticmethod
    def get_gamut_from_conversion_matrix(mat: ColorMatrix, target_gamut: "Gamut") -> "Gamut":
        """
        Given Mat, which converts from some unknown source_gamut to the specified
        target_gamut, return the source_gamut.
        """
        assert mat.from_state == ImageState.RGB and mat.to_state == ImageState.RGB
        source_to_xyz_mat: ColorMatrix = mat.composite(target_gamut.get_conversion_to_xyz())
        primaries_rgb: RGBChart = RGBChart(np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]))
        primaries_xyz: XYZChart = primaries_rgb.convert_to_xyz(source_to_xyz_mat)
        if any(primaries_xyz.colors[:, 1] == 0.0):
            raise ValueError("Encountered zero luminance in get_gamut_from_conversion_matrix.")
        primaries_xyy: XYYChart = primaries_xyz.convert_to_xyy(STD_E).normalize()
        source_gamut: "Gamut" = Gamut(
            red=XYYChart(primaries_xyy.colors[[0]]),
            green=XYYChart(primaries_xyy.colors[[1]]),
            blue=XYYChart(primaries_xyy.colors[[2]]),
            white=XYYChart(primaries_xyy.colors[[3]]),
        )
        return source_gamut


class Chart:
    colors: np.ndarray
    state: ImageState

    def __init__(self, colors: Union[np.ndarray, list]) -> None:
        self.colors = np.array(colors)
        self._check_shape()

    def _check_shape(self):
        assert len(self.colors.shape) == 2 and self.colors.shape[1] == 3

class RGBChart(Chart):
    state = ImageState.RGB

    def convert_to_xyz(self, mat: ColorMatrix) -> "XYZChart":
        assert mat.mat.shape == (3, 3), f"Expected 3x3 matrix but found {mat.mat.shape}"
        assert mat.from_state == ImageState.RGB and mat.to_state == ImageState.XYZ
        out_colors = self.colors @ mat.mat.T
        out = XYZChart(out_colors)
        return out

    def convert_to_rgb(self, mat: ColorMatrix) -> "RGBChart":
        assert mat.mat.shape == (3, 3), f"Expected 3x3 matrix but found {mat.mat.shape}"
        assert mat.from_state == ImageState.RGB and mat.to_state == ImageState.RGB
        out_colors = self.colors @ mat.mat.T
        out = RGBChart(out_colors)
        return out

    def scale(self, gain: float) -> "RGBChart":
        return RGBChart(self.colors * gain)


class LABChart(Chart):
    state = ImageState.LAB

    def convert_to_xyz(self, white_xyz: "XYZChart") -> "XYZChart":
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

    def compute_delta_e(self, other: "LABChart") -> float:
        return float(np.mean(np.sum((self.colors - other.colors)**2, axis=1)**0.5))


class XYZChart(Chart):
    state = ImageState.XYZ

    def convert_to_lab(self, white_xyz: "XYZChart") -> LABChart:
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

    def convert_to_xyy(self, white_xyz: "XYZChart") -> "XYYChart":
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

    def convert_to_rgb(self, mat: ColorMatrix) -> "RGBChart":
        assert mat.mat.shape == (3, 3), f"Expected 3x3 matrix but found {mat.mat.shape}"
        assert mat.from_state == ImageState.XYZ and mat.to_state == ImageState.RGB
        out_colors = self.colors @ mat.mat.T
        out = RGBChart(out_colors)
        return out

    def convert_to_xyz(self, mat: ColorMatrix) -> "XYZChart":
        assert mat.mat.shape == (3, 3), f"Expected 3x3 matrix but found {mat.mat.shape}"
        assert mat.from_state == ImageState.XYZ and mat.to_state == ImageState.XYZ
        out_colors = self.colors @ mat.mat.T
        out = XYZChart(out_colors)
        return out

    def chromatic_adaptation(self, from_whitepoint: "XYZChart", to_whitepoint: "XYZChart") -> "XYZChart":
        mat = ColorMatrix.get_chromatic_adaptation_matrix(from_whitepoint, to_whitepoint)
        xyz = self.convert_to_xyz(mat)
        return xyz

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

    def normalize(self) -> 'XYYChart':
        new_colors = self.colors.copy()
        new_colors[:, 2] = 1.0
        return XYYChart(new_colors)


class ReferenceChart(LABChart):
    reference_white: XYZChart
    def __init__(self, colors: Union[np.ndarray, list], reference_white: XYZChart):
        super(ReferenceChart, self).__init__(colors)
        self.reference_white = reference_white

STD_A: XYZChart = XYYChart(colors=np.array([[0.34842, 0.35161, 1.0]])).convert_to_xyz()
STD_C: XYZChart = XYYChart(colors=np.array([[0.31006, 0.31616, 1.0]])).convert_to_xyz()
STD_D65: XYZChart = XYYChart(colors=np.array([[0.3127, 0.3290, 1.0]])).convert_to_xyz()
STD_D50: XYZChart = XYYChart(colors=np.array([[0.34567, 0.35850, 1.0]])).convert_to_xyz()
STD_E: XYZChart = XYZChart(colors=np.array([[1.0, 1.0, 1.0]]))

GAMUT_DWG: Gamut = Gamut(
    red=XYYChart(np.array([[0.800, 0.3130, 1.0]])),
    green=XYYChart(np.array([[0.1682, 0.9877, 1.0]])),
    blue=XYYChart(np.array([[0.0790, -0.1155, 1.0]])),
    white=XYYChart(np.array([[0.3127, 0.3290, 1.0]])),
)
GAMUT_AP0: Gamut = Gamut(
    red=XYYChart(np.array([[0.7347, 0.2653, 1.0]])),
    green=XYYChart(np.array([[0.0, 1.0, 1.0]])),
    blue=XYYChart(np.array([[0.0001, -0.0770, 1.0]])),
    white=XYYChart(np.array([[0.32168, 0.33767, 1.0]])),
)
GAMUT_REC709: Gamut = Gamut(
    red=XYYChart(np.array([[0.64, 0.33, 1.0]])),
    green=XYYChart(np.array([[0.30, 0.60, 1.0]])),
    blue=XYYChart(np.array([[0.15, 0.06, 1.0]])),
    white=XYYChart(np.array([[0.3127, 0.3290, 1.0]])),
)
