import re
from typing import List, Tuple, Optional
import src.color_conversions as color_conversions


def read_file(fn: str) -> List[str]:
    with open(fn, "r", encoding="UTF-8") as f:
        return f.readlines()


def load_reference_chart_csv(
    lines: List[str],
) -> Tuple[color_conversions.ReferenceChart, Tuple[int, int]]:
    lines = [line.strip() for line in lines if len(line.strip()) > 0]
    cols = {col_name: idx for idx, col_name in enumerate(lines[0].split(","))}
    assert all(
        [
            expected_col in cols
            for expected_col in ["patch_number", "lab_l", "lab_a", "lab_b", "white"]
        ]
    )
    patches: List[List[float]] = []
    reference_white: color_conversions.XYZChart = color_conversions.STD_D65  # default
    max_row = 0
    for line in lines[1:]:
        parts: list[str] = line.split(",")
        if len(parts[cols["white"]].strip()) > 0:
            white_str = parts[cols["white"]]
            if white_str == "D65":
                reference_white = color_conversions.STD_D65
            elif white_str == "D50":
                reference_white = color_conversions.STD_D50
            elif white_str == "A":
                reference_white = color_conversions.STD_A
            else:
                raise ValueError(f"Unsupported white point: {white_str}")

        lab = [
            float(parts[cols["lab_l"]]),
            float(parts[cols["lab_a"]]),
            float(parts[cols["lab_b"]]),
        ]
        patch_idx = parts[cols["patch_number"]]
        patch_idx_matches = re.match(r"([A-Z]*)(\d*)", patch_idx)
        if patch_idx_matches is not None:
            patch_idx_groups = patch_idx_matches.groups()
            col = patch_idx_groups[0]
            row = int(patch_idx_groups[1])
            max_row = max(max_row, row)
            patches.append(lab)

    num_chips = len(patches)
    num_rows = max_row
    num_cols = num_chips // max_row
    estimated_dimensions = (num_rows, num_cols)

    reference_chart = color_conversions.ReferenceChart(
        colors=patches, reference_white=reference_white
    )
    return reference_chart, estimated_dimensions


def load_reference_chart_txt(
    lines: List[str],
) -> Tuple[color_conversions.ReferenceChart, Tuple[int, int]]:
    # Parses the X-Rite color chart official specification
    reference_white: color_conversions.XYZChart = color_conversions.STD_A  # default
    patches: List[List[float]] = []
    verified_lab = False
    highest_patch_coord = ("A", 0)
    lines = [line for line in lines if len(line.strip()) > 0]
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

        matches = re.match(
            r"([A-Z]*)(\d*)\s+(-?[\d,\.]+)\s+(-?[\d,\.]+)\s+(-?[\d,\.]+)", l
        )
        if matches is not None:
            groups = matches.groups()
            groups_reformatted = [x.replace(",", ".") for x in groups]
            patches.append(
                [
                    float(groups_reformatted[2]),
                    float(groups_reformatted[3]),
                    float(groups_reformatted[4]),
                ]
            )
            highest_patch_coord = max(
                highest_patch_coord[0], groups_reformatted[0]
            ), max(highest_patch_coord[1], int(groups_reformatted[1]))

        matches = re.match(
            r"sample_name\s+([\S\_]+)\s+([\S\_]+)\s+([\S\_]+)", l.lower()
        )
        if matches is not None:
            groups = matches.groups()
            assert [x.lower() for x in groups] == ["lab_l", "lab_a", "lab_b"]
            verified_lab = True

    estimated_dimensions = (
        highest_patch_coord[1],
        exp_num_patches // highest_patch_coord[1],
    )
    assert verified_lab, "File did not indicate LAB color space. Color space unknown!"
    assert exp_num_patches == len(
        patches
    ), f"Incorrect number of patches found in file. expected {exp_num_patches} but found {len(patches)}"
    assert reference_white is not None, "Could not parse reference white from file."

    reference_chart = color_conversions.ReferenceChart(
        colors=patches, reference_white=reference_white
    )
    return reference_chart, estimated_dimensions
