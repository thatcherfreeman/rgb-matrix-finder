import numpy as np
import color_conversions
from scipy.optimize import minimize, OptimizeResult  # type:ignore
from images import flatten
from typing import Tuple, Optional

class Parameters:
    _solve_matrix: bool
    matrix: color_conversions.ColorMatrix
    _solve_exposure: bool
    exposure: float
    _solve_white_balance: bool
    white_balance: color_conversions.ColorMatrix

    def __init__(
        self,
        matrix: Optional[color_conversions.ColorMatrix]=None,
        exposure: Optional[float]=None,
        white_balance: Optional[color_conversions.ColorMatrix]=None,
    ) -> None:
        """
        For each input, if None, then we will want to solve for this parameter.
        If provided, then we will just used the given value.
        """
        if matrix is None:
            self.matrix = color_conversions.ColorMatrix(
                np.identity(3),
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            )
            self._solve_matrix = True
        else:
            self.matrix = matrix
            self._solve_matrix = False

        if exposure is None:
            self.exposure = 1.0
            self._solve_exposure = True
        else:
            self.exposure = exposure
            self._solve_exposure = False

        if white_balance is None:
            self.white_balance = color_conversions.ColorMatrix(
                np.identity(3),
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            )
            self._solve_white_balance = True
        else:
            self.white_balance = white_balance
            self._solve_white_balance = False

    def to_numpy_parameters(self) -> np.ndarray:
        """
        For Scipy, we need to be able to convert this to a numpy array
        with a minimal length.
        """
        mat_params = self.matrix.mat[:, :2].reshape((6,))
        exposure_params = np.array(1.0)
        wb_params = np.array([self.white_balance.mat[0, 0], self.white_balance.mat[2, 2]])
        params: np.ndarray = np.array([])
        if self._solve_matrix:
            params = np.concatenate([params, mat_params])
        if self._solve_exposure:
            params = np.concatenate([params, [exposure_params]])
        if self._solve_white_balance:
            params = np.concatenate([params, wb_params])
        return params

    def update_from_numpy(self, params: np.ndarray) -> None:
        if self._solve_matrix:
            arr = np.zeros((3, 3))
            arr[:, :2] = params[:6].reshape((3, 2))
            arr[:, 2] = 1.0 - np.sum(arr[:, :2], axis=1)
            mat = color_conversions.ColorMatrix(
                arr,
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            )
            self.matrix = mat
            params = params[6:]
        if self._solve_exposure:
            self.exposure = params[0]
            params = params[1:]
        if self._solve_white_balance:
            self.white_balance = color_conversions.ColorMatrix(
                np.diag(np.array([params[0], 1.0, params[1]])),
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            )
            params = params[2:]

def chart_pipeline(
    source_colors: color_conversions.RGBChart,
    exp: float,
    gamut_transform: color_conversions.ColorMatrix,
    wb_factors: color_conversions.ColorMatrix,
) -> color_conversions.RGBChart:
    # Convert chart to the target gamut using the custom matrix.
    source_transformed = source_colors \
        .scale(exp) \
        .convert_to_rgb(wb_factors) \
        .convert_to_rgb(gamut_transform)
    return source_transformed

def image_pipeline(
    source_image: np.ndarray,
    exp: float,
    gamut_transform: color_conversions.ColorMatrix,
    wb_factors: color_conversions.ColorMatrix,
) -> np.ndarray:
    assert source_image.shape[-1] == 3
    source_image_flat = flatten(source_image)
    source_image_chart = color_conversions.RGBChart(source_image_flat)
    result_image_chart = chart_pipeline(
        source_image_chart, exp, gamut_transform, wb_factors
    )
    result_image = result_image_chart.colors.reshape(source_image.shape)
    return result_image

def cost_function(
    params: np.ndarray,
    parameters: Parameters,
    source_chart: color_conversions.RGBChart,
    ref_chart: color_conversions.ReferenceChart,
    target_gamut: color_conversions.Gamut,
) -> float:
    parameters.update_from_numpy(params)
    mat = parameters.matrix
    exp = parameters.exposure
    wb = parameters.white_balance
    source_lab: color_conversions.LABChart = chart_pipeline(source_chart, exp, mat, wb) \
        .convert_to_xyz(target_gamut.get_conversion_to_xyz()) \
        .chromatic_adaptation(target_gamut.white.convert_to_xyz(), ref_chart.reference_white) \
        .convert_to_lab(ref_chart.reference_white)
    de = ref_chart.compute_delta_e(source_lab)
    return de

def optimize(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    target_gamut: color_conversions.Gamut,
    verbose=False,
    parameters=Parameters(),
) -> Parameters:
    params = parameters.to_numpy_parameters()
    res: OptimizeResult = minimize(cost_function, params, args=(parameters, source_chart, reference_chart, target_gamut))
    optimized = res.x
    if verbose:
        print("Initial Delta-E: ", cost_function(params, parameters, source_chart, reference_chart, target_gamut))
        print("Final Delta-E: ", cost_function(optimized, parameters, source_chart, reference_chart, target_gamut))
        print(res.message)

    # Measure results.
    parameters.update_from_numpy(optimized)
    return parameters

def optimize_exp_wb(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool=False,
) -> Parameters:
    return optimize(
        source_chart,
        reference_chart,
        target_gamut,
        verbose=verbose,
        parameters=Parameters(
            matrix=source_gamut.get_conversion_to_gamut(target_gamut),
        ),
    )

def optimize_exp(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool=False,
) -> Parameters:
    return optimize(
        source_chart,
        reference_chart,
        target_gamut,
        verbose=verbose,
        parameters=Parameters(
            matrix=source_gamut.get_conversion_to_gamut(target_gamut),
            white_balance=color_conversions.ColorMatrix(
                np.diag([1.0, 1.0, 1.0]),
                color_conversions.ImageState.RGB,
                color_conversions.ImageState.RGB,
            ),
        ),
    )
