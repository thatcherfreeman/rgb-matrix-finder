import argparse
import json
import os
from typing import Union
from src.images import (
    flatten,
    get_samples,
    open_image,
    draw_samples,
)
import src.color_conversions as color_conversions
import src.reference_charts as reference_charts
import src.optimizer as optimizer
import numpy as np
from scipy.optimize import minimize, OptimizeResult  # type:ignore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reference_chart",
        type=str,
        help="File name of reference chart LAB txt description.",
    )
    parser.add_argument(
        "camera_chart",
        type=str,
        help="File name of a color chart shot on a camera, with the image already in scene linear.",
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
    parser.add_argument(
        "--correct-wb",
        action="store_true",
        default=False,
        help="Set this flag to have the program automatically compute the ideal exposure adjustment and white balance for your camera chart. Without this flag, by default, this will only optimize the 3x3 matrix.",
    )
    args = parser.parse_args()

    # Identify target gamut of IDT
    target_gamut: color_conversions.Gamut
    if args.target_gamut == "DWG":
        target_gamut = color_conversions.GAMUT_DWG
    elif args.target_gamut == "AP0":
        target_gamut = color_conversions.GAMUT_AP0
    else:
        raise ValueError(f"Unexpected target_gamut {args.target_gamut}")

    # Read Reference Chart
    reference_chart, patches = reference_charts.load_reference_chart(
        reference_charts.read_text_file(args.reference_chart)
    )
    if args.tall_chart:
        patches = max(patches), min(patches)
    else:
        patches = min(patches), max(patches)
    print(f"Expecting a chart of shape {patches}")

    # Read Source image.
    source_image = open_image(args.camera_chart)
    source_samples, sample_positions = get_samples(
        source_image, patches=patches, flat=True
    )

    # Preprocess source chart
    preprocessed_source_samples = source_samples.copy()

    # 1. TODO: White balance (channel scaling)
    # white_chip_mask = np.sum(np.abs(reference_chart.colors[:, 1:]), axis=1) < 3
    # white_balance_factors = 1.0 / (np.prod(preprocessed_source_samples[white_chip_mask, :], axis=0, keepdims=True)**(1.0 / np.sum(white_chip_mask)))  # shape (1, 3)
    # white_balance_factors /= white_balance_factors[0, 1] # scale green to 1.0
    # print("White balance factors: ", white_balance_factors)
    # preprocessed_source_samples *= white_balance_factors

    source_chart = color_conversions.RGBChart(preprocessed_source_samples)

    # Chart Alignment step.
    print(
        "Make sure the selected area and the reference chips are correctly placed on the chart!"
    )
    draw_samples(
        source_image, source_chart, reference_chart, sample_positions, show=True
    )

    # Optimize
    if args.correct_wb:
        # Optimize all of mat, exp, wb.
        initial_parameters = optimizer.Parameters()
    else:
        # Just optimize mat and exp.
        initial_parameters = optimizer.Parameters(
            white_balance=color_conversions.ColorMatrix(
                np.diag([1.0, 1.0, 1.0]),
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            ),
        )
    parameters: optimizer.Parameters = optimizer.optimize(
        source_chart,
        reference_chart,
        target_gamut,
        verbose=True,
        parameters=initial_parameters,
    )
    mat, exp, wb = parameters.matrix, parameters.exposure, parameters.white_balance

    # Measure results.
    print("solved matrix: ", mat.mat)
    print("Corrected exposure: ", exp)
    print("optimized wb coefficients: ", [wb.mat[0, 0], wb.mat[1, 1], wb.mat[2, 2]])
    gamut = color_conversions.Gamut.get_gamut_from_conversion_matrix(mat, target_gamut)
    print(
        "Gamut Primaries: ",
        gamut.red.colors,
        gamut.green.colors,
        gamut.blue.colors,
        gamut.white.colors,
    )

    info = {
        "matrix": mat.mat.tolist(),
        "gain": exp,
        "wb_coefficients": wb.mat.tolist(),
        "gamut": {
            "red": gamut.red.colors.tolist(),
            "green": gamut.green.colors.tolist(),
            "blue": gamut.blue.colors.tolist(),
            "white": gamut.white.colors.tolist(),
        },
    }
    json_fn = os.path.splitext(args.camera_chart)[0] + ".json"
    with open(json_fn, "w", encoding="UTF-8") as json_file:
        json.dump(info, json_file)

    gamut_to_display = target_gamut.get_conversion_to_gamut(
        color_conversions.GAMUT_REC709
    )
    draw_samples(
        optimizer.image_pipeline(source_image, exp, mat, wb) @ gamut_to_display.mat.T,
        optimizer.chart_pipeline(source_chart, exp, mat, wb).convert_to_rgb(
            gamut_to_display
        ),
        reference_chart,
        sample_positions,
        show=True,
    )


if __name__ == "__main__":
    main()
