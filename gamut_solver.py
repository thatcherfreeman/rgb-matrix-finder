import argparse
import re
from typing import Optional, List, Union

from images import (
    flatten,
    get_samples,
    open_image,
    sample_image,
)
import color_conversions

import numpy as np
import torch


def read_text_file(fn: str) -> List[str]:
    with open(fn, 'r') as f:
        return f.readlines()

def load_reference_chart(lines: List[str]) -> color_conversions.ReferenceChart:
    reference_white: Optional[color_conversions.XYZChart] = None
    patches: List[List[float]] = []
    verified_lab = False
    for l in lines:
        matches = re.match(r"\"MeasurementCondition=(\S*)", l)
        if matches is not None:
            groups = matches.groups()
            if groups[0] == "M0":
                reference_white = color_conversions.STD_A
            elif groups[0] == "M1":
                reference_white = color_conversions.STD_D50
            else:
                raise ValueError(f"Unexpected MeasurementCondition {matches[0]}")

        matches = re.match(r"NUMBER_OF_SETS (\d*)", l)
        if matches is not None:
            groups = matches.groups()
            exp_num_patches: int = int(groups[0])

        matches = re.match(r"([\S\d]*)\s+(-?[\d,]+)\s+(-?[\d,]+)\s+(-?[\d,]+)", l)
        if matches is not None:
            groups = matches.groups()
            groups_reformatted = [x.replace(",", ".") for x in groups]
            patches.append([float(groups_reformatted[1]), float(groups_reformatted[2]), float(groups_reformatted[3])])

        matches = re.match(r"SAMPLE_NAME\s+([\S\_]+)\s+([\S\_]+)\s+([\S\_]+)", l)
        if matches is not None:
            groups = matches.groups()
            assert [x.lower() for x in groups] == ["lab_l", "lab_a", "lab_b"]
            verified_lab = True

    assert verified_lab, "File did not indicate LAB color space. Color space unknown!"
    assert exp_num_patches == len(patches), f"Incorrect number of patches found in file. expected {exp_num_patches} but found {len(patches)}"
    assert reference_white is not None, "Could not parse reference white from file."

    reference_chart = color_conversions.ReferenceChart(colors=patches, reference_white=reference_white)
    return reference_chart


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "reference_chart",
        type=str,
        help="File name of reference chart LAB txt description."
    )
    args = parser.parse_args()

    chart = load_reference_chart(read_text_file(args.reference_chart))
    print(chart.colors)
    print(chart.reference_white.convert_to_xyy(color_conversions.STD_E).colors)
