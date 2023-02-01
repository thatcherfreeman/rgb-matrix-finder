import numpy as np
import src.color_conversions as color_conversions
from scipy.optimize import minimize, OptimizeResult  # type:ignore
from src.images import flatten
from typing import Tuple, Optional, List


class Parameters:
    _solve_matrix: bool
    matrix: color_conversions.ColorMatrix
    _solve_exposure: bool
    exposure: float
    _solve_white_balance: bool
    white_balance: color_conversions.ColorMatrix

    def __init__(
        self,
        matrix: Optional[color_conversions.ColorMatrix] = None,
        exposure: Optional[float] = None,
        white_balance: Optional[color_conversions.ColorMatrix] = None,
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

    @property
    def solve_matrix(self) -> bool:
        return self._solve_matrix

    @property
    def solve_exposure(self) -> bool:
        return self._solve_exposure

    @property
    def solve_white_balance(self) -> bool:
        return self._solve_white_balance

    def to_numpy_parameters(self) -> np.ndarray:
        """
        For Scipy, we need to be able to convert this to a numpy array
        with a minimal length.
        """
        mat_params = self.matrix.mat[:, :2].reshape((6,))
        exposure_params = np.array(1.0)
        wb_params = np.array(
            [self.white_balance.mat[0, 0], self.white_balance.mat[2, 2]]
        )
        params: np.ndarray = np.array([])
        if self._solve_matrix:
            params = np.concatenate([params, mat_params])
        if self._solve_exposure:
            params = np.concatenate([params, [exposure_params]])
        if self._solve_white_balance:
            params = np.concatenate([params, wb_params])
        return params

    def update_from_numpy(self, params: np.ndarray) -> np.ndarray:
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
        return params

    @staticmethod
    def list_to_numpy(parameters_list: List["Parameters"]) -> np.ndarray:
        return np.concatenate([p.to_numpy_parameters() for p in parameters_list])

    @staticmethod
    def update_parameter_list_from_numpy(
        np_params: np.ndarray, parameters_list: List["Parameters"]
    ) -> List["Parameters"]:
        output = []
        for parameter in parameters_list:
            np_params = parameter.update_from_numpy(np_params)
            output.append(parameter)
        return output


def chart_pipeline(
    source_colors: color_conversions.RGBChart,
    exp: float,
    gamut_transform: color_conversions.ColorMatrix,
    wb_factors: color_conversions.ColorMatrix,
) -> color_conversions.RGBChart:
    """
    Converts the chart `source_colors` to an RGB output by applying
    exp, wb_factors, and gamut_transform in that order.
    """
    # Convert chart to the target gamut using the custom matrix.
    source_transformed = (
        source_colors.scale(exp)
        .convert_to_rgb(wb_factors)
        .convert_to_rgb(gamut_transform)
    )
    return source_transformed


def image_pipeline(
    source_image: np.ndarray,
    exp: float,
    gamut_transform: color_conversions.ColorMatrix,
    wb_factors: color_conversions.ColorMatrix,
) -> np.ndarray:
    """
    Converts the image `source_image` to an RGB output by applying
    exp, wb_factors, and gamut_transform in that order.
    """
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
    de = agg_cost_function(
        params, [parameters], [source_chart], [ref_chart], target_gamut
    )
    return de


def agg_cost_function(
    params: np.ndarray,
    parameters: List[Parameters],
    source_charts: List[color_conversions.RGBChart],
    ref_charts: List[color_conversions.ReferenceChart],
    target_gamut: color_conversions.Gamut,
) -> float:
    assert (
        len(source_charts) == len(ref_charts) == len(parameters)
    ), "source_charts, ref_charts, and parameters were different lengths!"
    avg_de = 0.0
    parameters = Parameters.update_parameter_list_from_numpy(params, parameters)
    for i, (parameter, source_chart, ref_chart) in enumerate(
        zip(parameters, source_charts, ref_charts)
    ):
        if i > 0:
            assert parameters[i].solve_matrix == False
        mat = parameters[0].matrix  # all charts share the same matrix.
        exp = parameter.exposure
        wb = parameter.white_balance
        source_lab: color_conversions.LABChart = (
            chart_pipeline(source_chart, exp, mat, wb)
            .convert_to_xyz(target_gamut.get_conversion_to_xyz())
            .chromatic_adaptation(
                target_gamut.white.convert_to_xyz(), ref_chart.reference_white
            )
            .convert_to_lab(ref_chart.reference_white)
        )
        de = ref_chart.compute_delta_e(source_lab)
        avg_de += de
    avg_de /= len(parameters)
    return avg_de


def optimize(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    target_gamut: color_conversions.Gamut,
    verbose=False,
    parameters=Parameters(),
) -> Parameters:
    parameters_list = optimize_nd(
        [source_chart], [reference_chart], target_gamut, verbose, [parameters]
    )
    return parameters_list[0]


def optimize_nd(
    source_charts: List[color_conversions.RGBChart],
    reference_charts: List[color_conversions.ReferenceChart],
    target_gamut: color_conversions.Gamut,
    verbose: bool,
    parameters: List[Parameters],
) -> List[Parameters]:
    params = Parameters.list_to_numpy(parameters)
    res: OptimizeResult = minimize(
        agg_cost_function,
        params,
        (parameters, source_charts, reference_charts, target_gamut),
    )
    optimized = res.x
    if verbose:
        parameters = Parameters.update_parameter_list_from_numpy(params, parameters)
        for i, parameter in enumerate(parameters):
            print(f"Image {i}")
            print(
                "  Initial Delta-E: ",
                cost_function(
                    parameter.to_numpy_parameters(),
                    parameter,
                    source_charts[i],
                    reference_charts[i],
                    target_gamut,
                ),
            )

        parameters = Parameters.update_parameter_list_from_numpy(optimized, parameters)
        for i, parameter in enumerate(parameters):
            print(f"Image {i}")
            print(
                "  Final Delta-E: ",
                cost_function(
                    parameter.to_numpy_parameters(),
                    parameter,
                    source_charts[i],
                    reference_charts[i],
                    target_gamut,
                ),
            )
        print(res.message)
    parameters = Parameters.update_parameter_list_from_numpy(optimized, parameters)
    return parameters


def optimize_exp_wb(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool = False,
) -> Parameters:
    return optimize_nd_exp_wb(
        [source_chart],
        [reference_chart],
        source_gamut,
        target_gamut,
        verbose=verbose,
    )[0]


def optimize_exp(
    source_chart: color_conversions.RGBChart,
    reference_chart: color_conversions.ReferenceChart,
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool = False,
) -> Parameters:
    return optimize_nd_exp(
        [source_chart],
        [reference_chart],
        source_gamut,
        target_gamut,
        verbose=verbose,
    )[0]


def optimize_nd_exp_wb(
    source_charts: List[color_conversions.RGBChart],
    reference_charts: List[color_conversions.ReferenceChart],
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool = False,
) -> List[Parameters]:
    return optimize_nd(
        source_charts,
        reference_charts,
        target_gamut,
        verbose=verbose,
        parameters=[
            Parameters(
                matrix=source_gamut.get_conversion_to_gamut(target_gamut),
            )
            for _ in range(len(source_charts))
        ],
    )


def optimize_nd_exp(
    source_charts: List[color_conversions.RGBChart],
    reference_charts: List[color_conversions.ReferenceChart],
    source_gamut: color_conversions.Gamut,
    target_gamut: color_conversions.Gamut,
    verbose: bool = False,
) -> List[Parameters]:
    return optimize_nd(
        source_charts,
        reference_charts,
        target_gamut,
        verbose=verbose,
        parameters=[
            Parameters(
                matrix=source_gamut.get_conversion_to_gamut(target_gamut),
                white_balance=color_conversions.ColorMatrix(
                    np.diag([1.0, 1.0, 1.0]),
                    color_conversions.ImageState.RGB,
                    color_conversions.ImageState.RGB,
                ),
            )
            for _ in range(len(source_charts))
        ],
    )
