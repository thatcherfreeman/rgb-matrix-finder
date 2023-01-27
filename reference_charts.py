import re
from typing import List, Tuple, Optional
import color_conversions

def read_text_file(fn: str) -> List[str]:
    with open(fn, 'r', encoding='UTF-8') as f:
        return f.readlines()

def load_reference_chart(lines: List[str]) -> Tuple[color_conversions.ReferenceChart, Tuple[int, int]]:
    # Parses the X-Rite color chart official specification
    reference_white: Optional[color_conversions.XYZChart] = None
    patches: List[List[float]] = []
    verified_lab = False
    highest_patch_coord = ("A", 0)
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

        matches = re.match(r"([A-Z]*)(\d*)\s+(-?[\d,]+)\s+(-?[\d,]+)\s+(-?[\d,]+)", l)
        if matches is not None:
            groups = matches.groups()
            groups_reformatted = [x.replace(",", ".") for x in groups]
            patches.append([float(groups_reformatted[2]), float(groups_reformatted[3]), float(groups_reformatted[4])])
            highest_patch_coord = max(highest_patch_coord[0], groups_reformatted[0]), max(highest_patch_coord[1], int(groups_reformatted[1]))

        matches = re.match(r"SAMPLE_NAME\s+([\S\_]+)\s+([\S\_]+)\s+([\S\_]+)", l)
        if matches is not None:
            groups = matches.groups()
            assert [x.lower() for x in groups] == ["lab_l", "lab_a", "lab_b"]
            verified_lab = True

    estimated_dimensions = highest_patch_coord[1], exp_num_patches // highest_patch_coord[1]
    assert verified_lab, "File did not indicate LAB color space. Color space unknown!"
    assert exp_num_patches == len(patches), f"Incorrect number of patches found in file. expected {exp_num_patches} but found {len(patches)}"
    assert reference_white is not None, "Could not parse reference white from file."

    reference_chart = color_conversions.ReferenceChart(colors=patches, reference_white=reference_white)
    return reference_chart, estimated_dimensions
