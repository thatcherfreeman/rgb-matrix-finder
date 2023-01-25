import argparse
from images import (
    get_samples,
    open_image,
    draw_samples,
)
import color_conversions
import reference_charts
import numpy as np
from scipy.optimize import minimize, OptimizeResult # type:ignore


def get_initial_parameters() -> np.ndarray:
    params = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    params = params.reshape((6,))
    params = np.concatenate([params, [1.0]])
    return params

def get_mat_from_params(params: np.ndarray) -> color_conversions.ColorMatrix:
    assert params.shape == (7,)
    arr = np.zeros((3, 3))
    params = params[:6].reshape((3, 2))
    arr[:, :2] = params
    arr[:, 2] = 1.0 - np.sum(params, axis=1)
    mat = color_conversions.ColorMatrix(arr, color_conversions.ImageState.RGB, color_conversions.ImageState.RGB)
    return mat

def get_exp_from_params(params: np.ndarray) -> float:
    assert params.shape == (7,)
    return params[6]

def cost_function(
    parameters: np.ndarray,
    source_chart: color_conversions.RGBChart,
    ref_chart: color_conversions.ReferenceChart,
    target_gamut: color_conversions.Gamut,
) -> float:
    mat = get_mat_from_params(parameters)
    exp = get_exp_from_params(parameters)
    source_lab: color_conversions.LABChart = source_chart \
        .scale(exp) \
        .convert_to_rgb(mat) \
        .convert_to_xyz(target_gamut.get_conversion_to_xyz()) \
        .chromatic_adaptation(target_gamut.white.convert_to_xyz(), ref_chart.reference_white) \
        .convert_to_lab(ref_chart.reference_white)
    de = ref_chart.compute_delta_e(source_lab)
    return de

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "camera_chart",
        type=str,
        help="File name of a color chart shot on a camera, with the image already in scene linear."
    )
    parser.add_argument(
        "reference_chart",
        type=str,
        help="File name of reference chart LAB txt description."
    )
    parser.add_argument(
        "--tall-chart",
        action="store_true",
        default=False,
        help="Set this flag to search for a portrait chart instead of a landscape one.",
    )
    parser.add_argument(
        "--target-gamut",
        default="DWG",
        const="DWG",
        nargs="?",
        choices=["DWG", "AP0"],
        help="Choose a target color space for the matrix (default: %(default)s)",
    )
    args = parser.parse_args()

    # Identify target gamut of IDT
    target_gamut: color_conversions.Gamut
    if (args.target_gamut == "DWG"):
        target_gamut = color_conversions.GAMUT_DWG
    elif args.target_gamut == "AP0":
        target_gamut = color_conversions.GAMUT_AP0
    else:
        raise ValueError(f"Unexpected target_gamut {args.target_gamut}")

    # Read Reference Chart
    reference_chart, patches = reference_charts.load_reference_chart(reference_charts.read_text_file(args.reference_chart))
    if args.tall_chart:
        patches = max(patches), min(patches)
    else:
        patches = min(patches), max(patches)
    print(f"Expecting a chart of shape {patches}")

    # Read Source image.
    source_image = open_image(args.camera_chart)
    source_samples, sample_positions = get_samples(source_image, patches=patches, flat=True)

    # Preprocess source chart
    preprocessed_source_samples = source_samples.copy()

    # 1. TODO: White balance (channel scaling)
    white_chip_mask = np.sum(np.abs(reference_chart.colors[:, 1:]), axis=1) < 3
    white_balance_factors = 1.0 / (np.prod(preprocessed_source_samples[white_chip_mask, :], axis=0, keepdims=True)**(1.0 / np.sum(white_chip_mask)))  # shape (1, 3)
    white_balance_factors /= white_balance_factors[0, 1] # scale green to 1.0
    print("White balance factors: ", white_balance_factors)
    preprocessed_source_samples *= white_balance_factors

    source_chart = color_conversions.RGBChart(preprocessed_source_samples)

    # Chart Alignment step.
    print("Make sure the selected area and the reference chips are correctly placed on the chart!")
    draw_samples(source_image, source_chart, reference_chart, sample_positions, show=True)

    # Optimize
    params = get_initial_parameters()
    print("Initial Delta-E: ", cost_function(params, source_chart, reference_chart, target_gamut))
    res: OptimizeResult = minimize(cost_function, params, args=(source_chart, reference_chart, target_gamut))
    optimized = res.x
    print("Final Delta-E: ", cost_function(optimized, source_chart, reference_chart, target_gamut))
    print(res.message)

    # Measure results.
    mat = get_mat_from_params(optimized)
    exp = get_exp_from_params(optimized)
    print("solved matrix: ", mat.mat)
    print("Corrected exposure: ", exp)
    draw_samples(source_image @ mat.mat.T, source_chart.scale(exp).convert_to_rgb(mat).convert_to_rgb(color_conversions.GAMUT_DWG.get_conversion_to_gamut(color_conversions.GAMUT_REC709)), reference_chart, sample_positions, show=True)


if __name__ == "__main__":
    main()
