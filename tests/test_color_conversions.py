import color_conversions
import numpy as np

def test_xyzchart() -> None:
    xyz = color_conversions.XYZChart([[0.0, 1.0, 0.0]])
    lab = xyz.convert_to_lab(color_conversions.STD_E)
    assert isinstance(lab, color_conversions.LABChart)
    print(lab.colors)
    assert np.sum(np.abs(lab.colors - np.array([[100.000, -431.034, 172.414]]))) < 0.001
    xyz2 = lab.convert_to_xyz(color_conversions.STD_E)
    assert np.sum(np.abs(xyz.colors - xyz2.colors)) < 0.001

    xyz3 = color_conversions.XYZChart([[0.3, 0.5, 0.2]])
    lab3 = xyz3.convert_to_lab(color_conversions.STD_D65)
    assert np.sum(np.abs(lab3.colors - np.array([[76.069, -56.418, 45.051]]))) < 0.1

    xyz = color_conversions.XYZChart([[.950470, 1.0, 1.088830]])
    xyy = xyz.convert_to_xyy(color_conversions.STD_E)
    assert np.sum(np.abs(xyy.colors - np.array([[0.3127, 0.329, 1.0]]))) < 1e-4

    xyz = color_conversions.XYZChart([[0.4, 1.0, 0.3]])
    xyy = xyz.convert_to_xyy(color_conversions.STD_D65)
    assert np.sum(np.abs(xyy.colors - np.array([[0.23529, 0.58824, 1.0]]))) < 1e-4

def test_xyychart() -> None:
    xyy = color_conversions.XYYChart([[0.3127, 0.3291, 1.0]])
    xyz = xyy.convert_to_xyz()
    xyy2 = xyz.convert_to_xyy(color_conversions.STD_E)
    assert np.sum(np.abs(xyy2.colors - xyy.colors)) < 1e-10
    assert np.sum(np.abs(xyz.colors - np.array([[.95016712, 1.0, 1.08842297]]))) < 1e-6

def test_labchart() -> None:
    lab1 = color_conversions.LABChart([[100.0, 0.0, 0.0]])
    xyz = lab1.convert_to_xyz(color_conversions.STD_E)
    assert np.sum(np.abs(xyz.colors - np.array([[1.0, 1.0, 1.0]]))) < 0.01

    lab2 = color_conversions.LABChart([[100.0, 10.0, 10.0]])
    xyz = lab2.convert_to_xyz(color_conversions.STD_E).convert_to_xyy(color_conversions.STD_E)
    assert np.sum(np.abs(xyz.colors - np.array([[0.36360, 0.34263, 1.0]]))) < 0.0001

    lab3 = color_conversions.LABChart([[100.0, -10.0, -10.0]])
    xyz = lab3.convert_to_xyz(color_conversions.STD_D65).convert_to_xyy(color_conversions.STD_D65)
    assert np.sum(np.abs(xyz.colors - np.array([[0.28354, 0.31695, 1.0]]))) < 0.0001

    lab4 = color_conversions.LABChart([[10, 20, 30]])
    lab5 = color_conversions.LABChart([[25, 10, 100]])
    assert np.abs(lab4.compute_delta_e(lab5) - (5225**0.5)) < 0.00001

def test_rgbchart() -> None:
    rgb = color_conversions.RGBChart([[1.0, 0.5, 0.25], [0.25, 1.0, 0.25]])
    xyz = rgb.convert_to_xyz(np.eye(3))
    assert np.sum(np.abs(xyz.colors - rgb.colors)) < 0.000001

    xyz2 = rgb.convert_to_xyz(np.array([[1.0, 0.1, 0.1], [0.2, 1.0, 0.2], [0.3, 0.3, 1.0]]))
    print(xyz2.colors)
    assert np.sum(np.abs(xyz2.colors - np.array([[1.175, 0.675, 0.45 ], [0.525, 1.1  , 0.475]]))) < 1e-4
