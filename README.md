# rgb-matrix-finder

This script takes two **scene linear** images of a color chart and derives a best-fit RGB matrix that makes the "Source" image as similar to the "Target" image as possible.

## Running
Install the needed libraries with `pip install -r requirements.txt` and run with `python camera_matcher.py` and follow the instructions.

## Modeling

Given an RGB (3-dimensional) color patch `x` from the Source image and a RGB color patch `y` from the Target image, we suppose that `y ~ A(x + b)`, where `b` is either 1-dimensional or 3-dimensional, and `A` is a 3x3 matrix. The `b` term exists to handle any flare or reflections that was introduced or removed between the capture of the two images. We use gradient descent to identify the parameters `A` and `b`.

The Error modelling pipeline looks like this, starting with the code values extracted from the source image.

1. Source image code values for each patch
2. Exposure adjustment, which is a global gain that's automatically applied to minimize error relative to the reference chart
3. White balance coefficients (set to $(1, 1, 1)$ if `--skip-auto-wb` is set)
4. Fitted 3x3 matrix (shared across multiple source images) which is intended to go from the code values resulting from the previous step to the target gamut. This matrix preserves white by ensuring that the rows sum to 1.0, unless `--matrix-not-preserve-white` is set. If that flag is set, then we simply ensure that the sum of all the entries in this matrix is 3.0. This matrix is fitted to minimize error resulting from the end of this pipeline.
5. 3x3 matrix to go from the target gamut to XYZ (not fitted)
6. CAT02 chromatic adaptation matrix to go from the target gamut's white point to the white point of the reference chart (skipped if `--skip-chromatic-adaptation` is set).
7. The resulting XYZ values are converted to LAB, under the reference chart's specified lighting conditions.
8. Delta-E is computed in LAB space, as $\sqrt{(L_{src} - L_{ref})^2 + (A_{src} - A_{ref})^2 + (B_{src} - B_{ref})^2}$. The mean Delta-E across all patches is to be minimized via the exposure, white balance, and 3x3 matrix.

## Unit Tests
* Run linter with `mypy *.py src tests`
* Run unit tests with `pytest --cov tests`
* Reformat code with `black .`
