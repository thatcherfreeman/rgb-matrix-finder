import argparse
import re
from typing import Optional, List, Union, Tuple

from images import (
    get_samples,
    open_image,
    draw_samples,
)
import color_conversions
import reference_charts
import numpy as np
import torch

if __name__ == "__main__":
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
    args = parser.parse_args()

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
    source_chart = color_conversions.RGBChart(source_samples)
    print("Make sure the selected area and the reference chips are correctly placed on the chart!")
    draw_samples(source_image, source_chart, reference_chart, sample_positions, show=True)

