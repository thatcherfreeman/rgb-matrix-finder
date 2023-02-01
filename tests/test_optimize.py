import os

from src.optimizer import *
from src.color_conversions import *
from src.reference_charts import *
import numpy as np

src_gamut = GAMUT_REC709
target_gamut = GAMUT_DWG
ref_chart: ReferenceChart
ref_chart, _ = load_reference_chart(
    read_text_file(os.path.join("reference_charts", "ColorChecker24_After_Nov2014.txt"))
)
src_chart: RGBChart = (
    ref_chart.convert_to_xyz(ref_chart.reference_white)
    .convert_to_xyz(
        ColorMatrix.get_chromatic_adaptation_matrix(
            ref_chart.reference_white, src_gamut.white.convert_to_xyz()
        )
    )
    .convert_to_rgb(src_gamut.get_conversion_to_xyz().inverse())
)
src_chart.colors += np.random.default_rng(42069).normal(
    0.0, 0.0001, size=src_chart.colors.shape
)


def test_optimize() -> None:
    parameters: Parameters = optimize(src_chart, ref_chart, target_gamut, True)
    assert (
        np.sum(
            np.abs(
                parameters.matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.01
    )


def test_optimize_exp() -> None:
    exp = 1.5
    src_chart2 = src_chart.scale(exp)
    src_chart2.colors += np.random.normal(0.0, 0.0001, size=src_chart2.colors.shape)
    parameters: Parameters = optimize_exp(
        src_chart2, ref_chart, src_gamut, target_gamut, True
    )
    assert np.abs(parameters.exposure - (1.0 / exp)) < 0.0001
    assert (
        np.sum(
            np.abs(
                parameters.matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.0001
    )
    assert np.sum(np.abs(parameters.white_balance.mat - np.eye(3))) < 0.0001


def test_optimize_exp_wb() -> None:
    exp = 3.2
    wb_coeffs = np.diag([0.8, 1.0, 1.2])
    src_chart2 = src_chart.scale(exp).convert_to_rgb(
        ColorMatrix(wb_coeffs, ImageState.RGB, ImageState.RGB),
    )
    parameters: Parameters = optimize_exp_wb(
        src_chart2, ref_chart, src_gamut, target_gamut, True
    )
    assert (
        np.sum(
            np.abs(
                parameters.matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.001
    )
    assert np.abs(parameters.exposure - (1.0 / exp)) < 0.001
    assert np.abs(parameters.white_balance.mat[0, 0] - (1.0 / wb_coeffs[0, 0])) < 0.001
    assert np.abs(parameters.white_balance.mat[2, 2] - (1.0 / wb_coeffs[2, 2])) < 0.001


def test_optimize_nd_exp() -> None:
    exp1, exp2 = 2.5, 0.6
    src_chart1 = src_chart.scale(exp1)
    src_chart2 = src_chart.scale(exp2)

    parameters: List[Parameters] = optimize_nd_exp(
        [src_chart1, src_chart2], [ref_chart, ref_chart], src_gamut, target_gamut, True
    )
    assert np.abs(parameters[0].exposure - (1.0 / exp1)) < 0.001
    assert np.abs(parameters[1].exposure - (1.0 / exp2)) < 0.001
    assert (
        np.sum(
            np.abs(
                parameters[0].matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.0001
    )
    assert np.sum(np.abs(parameters[0].white_balance.mat - np.eye(3))) < 0.0001


def test_optimize_nd_exp_wb() -> None:
    exp1, exp2 = 2.5, 0.6
    wb_coeffs1 = np.diag([0.8, 1.0, 1.2])
    wb_coeffs2 = np.diag([0.93, 1.0, 0.9])
    src_chart1 = src_chart.scale(exp1).convert_to_rgb(
        ColorMatrix(wb_coeffs1, ImageState.RGB, ImageState.RGB),
    )
    src_chart2 = src_chart.scale(exp2).convert_to_rgb(
        ColorMatrix(wb_coeffs2, ImageState.RGB, ImageState.RGB),
    )
    parameters: List[Parameters] = optimize_nd_exp_wb(
        [src_chart1, src_chart2], [ref_chart, ref_chart], src_gamut, target_gamut, True
    )
    assert np.abs(parameters[0].exposure - (1.0 / exp1)) < 0.01
    assert np.abs(parameters[1].exposure - (1.0 / exp2)) < 0.01
    assert (
        np.sum(
            np.abs(
                parameters[0].matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.0001
    )
    assert (
        np.abs(parameters[0].white_balance.mat[0, 0] - (1.0 / wb_coeffs1[0, 0])) < 0.01
    )
    assert (
        np.abs(parameters[0].white_balance.mat[2, 2] - (1.0 / wb_coeffs1[2, 2])) < 0.01
    )
    assert (
        np.abs(parameters[1].white_balance.mat[0, 0] - (1.0 / wb_coeffs2[0, 0])) < 0.01
    )
    assert (
        np.abs(parameters[1].white_balance.mat[2, 2] - (1.0 / wb_coeffs2[2, 2])) < 0.01
    )


def test_optimize_nd() -> None:
    exp1, exp2 = 2.5, 0.6
    wb_coeffs1 = np.diag([0.8, 1.0, 1.2])
    wb_coeffs2 = np.diag([0.93, 1.0, 0.9])
    src_chart1 = src_chart.scale(exp1).convert_to_rgb(
        ColorMatrix(wb_coeffs1, ImageState.RGB, ImageState.RGB),
    )
    src_chart2 = src_chart.scale(exp2).convert_to_rgb(
        ColorMatrix(wb_coeffs2, ImageState.RGB, ImageState.RGB),
    )
    parameters: List[Parameters] = optimize_nd(
        [src_chart1, src_chart2],
        [ref_chart, ref_chart],
        target_gamut,
        True,
        [
            Parameters(),
            Parameters(ColorMatrix(np.eye(3), ImageState.RGB, ImageState.RGB)),
        ],
    )
    assert np.abs(parameters[0].exposure - (1.0 / exp1)) < 0.01
    assert np.abs(parameters[1].exposure - (1.0 / exp2)) < 0.01
    assert (
        np.sum(
            np.abs(
                parameters[0].matrix.mat
                - (src_gamut.get_conversion_to_gamut(target_gamut).mat)
            )
        )
        < 0.001
    )
    assert np.sum(np.abs(parameters[1].matrix.mat - (np.eye(3)))) < 0.001
    assert (
        np.abs(parameters[0].white_balance.mat[0, 0] - (1.0 / wb_coeffs1[0, 0])) < 0.01
    )
    assert (
        np.abs(parameters[0].white_balance.mat[2, 2] - (1.0 / wb_coeffs1[2, 2])) < 0.01
    )
    assert (
        np.abs(parameters[1].white_balance.mat[0, 0] - (1.0 / wb_coeffs2[0, 0])) < 0.01
    )
    assert (
        np.abs(parameters[1].white_balance.mat[2, 2] - (1.0 / wb_coeffs2[2, 2])) < 0.01
    )
