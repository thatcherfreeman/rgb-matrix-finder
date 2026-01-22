from argparse import ArgumentParser
import src.color_conversions as color_conversions
import numpy as np

def main():
    parser = ArgumentParser(
        description="Computes the matrix to convert from one gamut to another. CAT02 Chromatic Adaptation is on by default."
    )

    gamuts = {
        "AP0": color_conversions.GAMUT_AP0,
        "AP1": color_conversions.GAMUT_AP1,
        "AWG3": color_conversions.GAMUT_AWG3,
        "AWG4": color_conversions.GAMUT_AWG4,
        "DWG": color_conversions.GAMUT_DWG,
        "KINEFINITY": color_conversions.GAMUT_KINEFINITY_WIDE_GAMUT,
        "REC601": color_conversions.GAMUT_REC601,
        "REC709": color_conversions.GAMUT_REC709,
        "REC2020": color_conversions.GAMUT_REC2020,
        "XYZ": color_conversions.GAMUT_XYZ,
    }

    parser.add_argument(
        "source_gamut",
        choices=gamuts.keys(),
        help="Choose a source color space for the matrix",
    )
    parser.add_argument(
        "target_gamut",
        nargs="?",
        choices=gamuts.keys(),
        help="Choose a target color space for the matrix",
    )
    parser.add_argument(
        "--no-cat",
        action="store_true",
        help="Disable Chromatic Adaptation Transform (CAT02) when computing the conversion matrix",
    )
    args = parser.parse_args()

    source_gamut = gamuts[args.source_gamut.upper()]
    target_gamut = gamuts[args.target_gamut.upper()]
    chromatic_adaptation = not args.no_cat

    print(
        f"Computing conversion matrix from {args.source_gamut} to {args.target_gamut}..."
    )

    matrix: color_conversions.ColorMatrix = source_gamut.get_conversion_to_gamut(
        target_gamut, chromatic_adaptation=chromatic_adaptation
    )

    print("Conversion Matrix:")
    print("{")
    for row in matrix.mat:
        formatted_row = ", ".join(f"{v:.10f}" for v in row)
        print(f"    {{{formatted_row}}},")
    print("}")


if __name__ == "__main__":
    main()
