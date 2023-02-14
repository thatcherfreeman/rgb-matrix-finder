import argparse
import json
import os
from typing import List, Optional
from src.images import (
    get_samples,
    open_image,
    draw_samples,
)
import src.color_conversions as color_conversions
import src.reference_charts as reference_charts
import src.optimizer as optimizer

FN_SPLIT_CHAR = ","


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reference_chart",
        type=str,
        help="File name of reference chart LAB txt description. Can specify multiple if comma separated.",
    )
    parser.add_argument(
        "camera_chart",
        type=str,
        help="File name of a color chart shot on a camera, with the image already in scene linear. Can specify multiple if comma separated.",
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
    parser.add_argument(
        "--gamut-file",
        default=None,
        type=str,
        help="The path to a gamut file generated by gamut_solver.py. If None, assumes no transformation matrix and the camera_chart is already in the target gamut. Use this flag to specify a pre-solved matrix.",
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

    # Identify the source gamut that's assumed for the source image.
    source_gamut: Optional[color_conversions.Gamut]
    if args.gamut_file is None:
        source_gamut = None
    else:
        with open(args.gamut_file, "r", encoding="ascii") as json_file:
            items = json.load(json_file)
        source_gamut = color_conversions.Gamut(
            color_conversions.XYYChart(items["gamut"]["red"]),
            color_conversions.XYYChart(items["gamut"]["green"]),
            color_conversions.XYYChart(items["gamut"]["blue"]),
            color_conversions.XYYChart(items["gamut"]["white"]),
        )

    # Identify the files that we'll read as images.
    source_fns = [os.path.expanduser(x) for x in args.camera_chart.split(FN_SPLIT_CHAR)]

    # Read Reference Chart
    ref_charts = []
    ref_patches = []
    for i, fn in enumerate(args.reference_chart.split(FN_SPLIT_CHAR)):
        ref_chart, patches = reference_charts.load_reference_chart(
            reference_charts.read_text_file(fn)
        )
        if args.tall_chart:
            patches = max(patches), min(patches)
        else:
            patches = min(patches), max(patches)
        ref_charts.append(ref_chart)
        ref_patches.append(patches)
        print(f"Chart {i}: Expecting a chart of shape {patches}")

    # We can use one ref chart if there are multiple camera charts.
    if len(source_fns) > 1 and len(ref_charts) == 1:
        ref_patches = ref_patches * len(source_fns)
        ref_charts = ref_charts * len(source_fns)

    # Read Source image.
    source_images = []
    source_image_samples = []
    source_image_sample_positions = []
    for fn, ref_patch in zip(source_fns, ref_patches):
        source_image = open_image(fn)
        source_samples, sample_positions = get_samples(
            source_image, patches=ref_patch, flat=True
        )
        source_images.append(source_image)
        source_image_samples.append(source_samples)
        source_image_sample_positions.append(sample_positions)
    source_charts = [color_conversions.RGBChart(s) for s in source_image_samples]

    # Chart Alignment step.
    print(
        "Make sure the selected area and the reference chips are correctly placed on the chart!"
    )
    for i, (source_image, source_chart, ref_chart, sample_positions) in enumerate(
        zip(source_images, source_charts, ref_charts, source_image_sample_positions)
    ):
        draw_samples(
            source_image,
            source_chart,
            ref_chart,
            sample_positions,
            show=True,
            title=f"Input Image {i}",
        )

    # Optimize
    if args.correct_wb:
        # Optimize all of mat, exp, wb.
        initial_parameters = [optimizer.Parameters()]
        if len(source_charts) > 1:
            for _ in source_charts[1:]:
                initial_parameters.append(
                    optimizer.Parameters(matrix=color_conversions.MATRIX_RGB_IDENTITY)
                )
    else:
        # Just optimize mat and exp.
        initial_parameters = [
            optimizer.Parameters(white_balance=color_conversions.MATRIX_RGB_IDENTITY)
        ]
        if len(source_charts) > 1:
            for _ in source_charts[1:]:
                initial_parameters.append(
                    optimizer.Parameters(
                        matrix=color_conversions.MATRIX_RGB_IDENTITY,
                        white_balance=color_conversions.MATRIX_RGB_IDENTITY,
                    )
                )
    parameters: List[optimizer.Parameters]
    if source_gamut is None:
        parameters = optimizer.optimize_nd(
            source_charts,
            ref_charts,
            target_gamut,
            verbose=True,
            parameters=initial_parameters,
        )
    else:
        parameters = optimizer.optimize_nd_exp_wb(
            source_charts, ref_charts, source_gamut, target_gamut, True
        )

    # Measure results.
    mat = parameters[0].matrix
    for i in range(len(parameters)):
        exp, wb = (
            parameters[i].exposure,
            parameters[i].white_balance,
        )
        print(f"\nImage {i}")
        print("  Corrected exposure: ", exp)
        print(
            "  Optimized wb coefficients: ",
            [wb.mat[0, 0], wb.mat[1, 1], wb.mat[2, 2]],
        )
    print("\nsolved matrix: \n", mat.mat)
    gamut = color_conversions.Gamut.get_gamut_from_conversion_matrix(mat, target_gamut)
    print(
        "\nGamut Primaries: ",
        gamut.red.colors,
        gamut.green.colors,
        gamut.blue.colors,
        gamut.white.colors,
    )

    # Save a gamut file if the user did not specify one.
    if source_gamut is None:
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
        json_fn = os.path.splitext(source_fns[0])[0] + ".json"
        with open(json_fn, "w", encoding="UTF-8") as json_file:
            print(f"Writing gamut file to {json_fn}")
            json.dump(info, json_file, indent=4)

    # Draw corrected images on the screen.
    gamut_to_display = target_gamut.get_conversion_to_gamut(
        color_conversions.GAMUT_REC709
    )
    for i, (
        source_image,
        source_chart,
        ref_chart,
        sample_positions,
        parameter,
    ) in enumerate(
        zip(
            source_images,
            source_charts,
            ref_charts,
            source_image_sample_positions,
            parameters,
        )
    ):
        mat, exp, wb = parameter.matrix, parameter.exposure, parameter.white_balance
        draw_samples(
            optimizer.image_pipeline(source_image, exp, mat, wb)
            @ gamut_to_display.mat.T,
            optimizer.chart_pipeline(source_chart, exp, mat, wb).convert_to_rgb(
                gamut_to_display
            ),
            ref_chart,
            sample_positions,
            show=True,
            title=f"Corrected Image {i}",
        )


if __name__ == "__main__":
    main()
