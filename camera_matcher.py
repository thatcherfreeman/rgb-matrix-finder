from enum import Enum
import itertools
import numpy as np
import matplotlib.pyplot as plt  # type:ignore
from scipy.optimize import minimize, OptimizeResult  # type:ignore
from argparse import ArgumentParser, Namespace
from typing import Callable, Tuple, Any

from src.images import (
    flatten,
    get_samples,
    open_image,
    draw_samples,
    show_image,
)
from src.color_conversions import RGBChart
import src.color_conversions as color_conversions

# Intended to help you match one Scene Linear image of a color chart
# from one camera to another.


class Bias(str, Enum):
    BIAS_1D = "1d"
    BIAS_3D = "3d"
    BIAS_NONE = "none"


def fit_colors_ls(input_rgb, output_rgb, args):
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb)  # (24, 3)
    if args.bias == "3d":
        input_rgb_flat = np.concatenate(
            [input_rgb_flat, np.ones((input_rgb_flat.shape[0], 1))], axis=1
        )  # (24,4)
        print("Applying 3d bias, model is: bias + (A @ source) = target")
    else:
        print("Applying no bias, model is: A @ source = target")
    mat = np.linalg.lstsq(input_rgb_flat, output_rgb_flat, rcond=None)[0]
    if args.bias == "3d":
        # Last row represents bias term.
        bias = mat[[3], :]
        mat = mat[:-1, :]
    else:
        bias = np.zeros((1, 3))

    if args.enforce_whitepoint:
        col_sum = np.sum(mat, axis=0, keepdims=True)
        mat = mat / col_sum
    return (mat.T, bias), np.linalg.pinv(mat.T), lambda x: x @ mat + bias


def fit_colors_wppls(input_rgb, output_rgb, args):
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb)  # (24, 3)
    print("Applying no bias, model is: A @ source = target")

    if args.enforce_whitepoint:
        NT = np.eye(3)
        MT = np.eye(3)
    else:
        white_patch_idx = np.argmax(np.mean(output_rgb_flat, axis=1))
        print(
            f"white patch index: {white_patch_idx}",
            f"with color: {output_rgb_flat[white_patch_idx]}",
        )
        NT = np.diag(1 / input_rgb_flat[white_patch_idx, :])
        MT = np.diag(1 / output_rgb_flat[white_patch_idx, :])
    N = input_rgb_flat @ NT
    M = output_rgb_flat @ MT
    u = np.ones((3, 1))
    mat = np.zeros((3, 3))
    for i in range(mat.shape[1]):
        v = M[:, [i]]
        ntn = N.T @ N
        ntn_inv = np.linalg.pinv(ntn)
        ntn_inv_u = ntn_inv @ u
        c = ntn_inv @ N.T @ v
        c += (1 - v.T @ N @ ntn_inv_u) / (u.T @ ntn_inv_u) * ntn_inv_u
        assert c.shape == (3, 1)
        mat[:, [i]] = c

    mat2 = NT @ mat @ np.linalg.pinv(MT)
    return mat2.T, np.linalg.pinv(mat2.T), lambda x: x @ mat2


def fit_colors_rp(input_rgb, output_rgb, args):
    """Root polynomial method for color matching"""
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb)
    degree = args.degree

    def set_degree(input_rgb, degree):
        # input_rgb of shape (N, 3)
        result = input_rgb.copy()
        example_rgb = np.array([2.0, 3.0, 5.0])  # three relatively prime numbers
        seen_examples = [2.0, 3.0, 5.0]
        rgb_idxs = [0, 1, 2]
        accepted_combos = [[0], [1], [2]]
        for d in range(2, degree + 1):
            # identify all 3*d combinations of d elements
            all_combos = [
                list(x)
                for x in list(itertools.combinations_with_replacement(rgb_idxs, d))
            ]
            for combo in all_combos:
                # combo is d-tuple of indexes
                curr_example = np.prod(example_rgb[combo]) ** (1.0 / d)
                if not any([abs(curr_example - x) < 0.0001 for x in seen_examples]):
                    seen_examples.append(curr_example)
                    result = np.concatenate(
                        [
                            result,
                            np.prod(input_rgb[:, combo], axis=1, keepdims=True)
                            ** (1.0 / d),
                        ],
                        axis=1,
                    )
                    accepted_combos.append(combo)
        return result, accepted_combos

    expanded_input_rgb_flat, combos = set_degree(input_rgb_flat, degree)
    col_names = ["r", "g", "b"]
    print(
        "Columns: ",
        [
            f'({"*".join([col_names[x] for x in combo])})^(1/{len(combo)})'
            for combo in combos
        ],
    )
    mat = np.linalg.lstsq(expanded_input_rgb_flat, output_rgb_flat)[0].T
    return mat, np.linalg.pinv(mat), lambda x: set_degree(x, degree)[0] @ mat.T


class log_mat_model:
    optimize_vec: np.ndarray
    args: Namespace

    def __init__(self, optimize_vec, args):
        assert optimize_vec is not None or args is not None
        self.optimize_vec = optimize_vec
        assert self.optimize_vec.shape == (log_mat_model.get_num_args(args),)
        self.args = args

    @staticmethod
    def get_num_args(args) -> int:
        num_args = 0
        if args.enforce_whitepoint:
            # 3x3 matrix, but rows sum to 1
            num_args = 6
        else:
            # 3x3 matrix
            num_args = 9
        if args.bias == Bias.BIAS_1D.value:
            num_args += 1
        elif args.bias == Bias.BIAS_3D.value:
            num_args += 3
        elif args.bias == Bias.BIAS_NONE.value:
            num_args += 0
        return num_args

    def get_model(self):
        runner = 0
        if args.enforce_whitepoint:
            mat_vec = self.optimize_vec[runner : runner + 6]
            runner += 6
            mat = np.array(
                [
                    [1.0 - mat_vec[0] - mat_vec[1], mat_vec[0], mat_vec[1]],
                    [mat_vec[2], 1.0 - mat_vec[2] - mat_vec[3], mat_vec[3]],
                    [mat_vec[4], mat_vec[5], 1.0 - mat_vec[4] - mat_vec[5]],
                ]
            )
        else:
            mat = np.array(self.optimize_vec[runner : runner + 9]).reshape((3, 3))
            runner += 9
        bias_shape = (1, 3)
        if args.bias == Bias.BIAS_1D.value:
            bias_vec = np.full(bias_shape, self.optimize_vec[runner])
            runner += 1
        elif args.bias == Bias.BIAS_3D.value:
            bias_vec = np.array(self.optimize_vec[runner : runner + 3]).reshape(
                bias_shape
            )
            runner += 3
        elif args.bias == Bias.BIAS_NONE.value:
            bias_vec = np.zeros(bias_shape)
            runner += 0
        return mat, bias_vec

    def forward(self, input_rgb: np.ndarray):
        """
        input_rgb of shape (n, 3), returns colors of shape (n, 3)
        """
        mat, bias = self.get_model()
        return input_rgb @ mat.T + bias

    def loss(self, input_rgb, output_rgb):
        pixel_loss = np.log2(self.forward(input_rgb)) - np.log2(output_rgb)
        # replace nans and -infs with 0
        mask = np.isinf(pixel_loss) | np.isnan(pixel_loss)
        pixel_loss[mask] = 0.0
        return np.mean(pixel_loss**2)


def fit_colors_log_mat(input_rgb, output_rgb, args):
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb)

    optimize_vec = np.zeros((log_mat_model.get_num_args(args),))
    if not args.enforce_whitepoint:
        optimize_vec[:9] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    def optim_func(x, input_rgb, output_rgb, args):
        model = log_mat_model(x, args)
        return model.loss(input_rgb, output_rgb)

    res: OptimizeResult = minimize(
        optim_func,
        optimize_vec,
        (input_rgb_flat, output_rgb_flat, args),
        options={"disp": True},
    )

    optimized_vec = res.x
    print("Model: bias + (mat @ img)")
    model = log_mat_model(optimized_vec, args)
    parameters = model.get_model()
    return parameters, np.linalg.pinv(parameters[0]), lambda x: model.forward(x)


def plot_samples(samples, labels):
    # samples of shape (n, 3) or (r, c, 3)
    if len(samples.shape) == 2:
        n = samples.shape[0]
        num_cols = int(n**0.5) + 1
        num_rows = n // num_cols + 1
    else:
        num_cols = samples.shape[1]
        num_rows = samples.shape[0]
    f, axarr = plt.subplots(num_rows, num_cols)
    f.set_size_inches(16, 9)
    # scale = 1 / np.max(samples)
    scale = 1.0
    for i, (color, title) in enumerate(zip(flatten(samples), flatten(labels))):
        r, c = i // num_cols, i % num_cols
        axarr[r, c].imshow(np.ones((100, 100, 3)) * color * scale)
        axarr[r, c].set_title(title[0], fontsize=5)
        axarr[r, c].set_xticks([])
        axarr[r, c].set_yticks([])
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Original image to apply rgb matrix to.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Reference color chart that we're going to match to.",
    )
    parser.add_argument(
        "--bias",
        type=str,
        default="none",
        help="Optionally set to {1d, 3d} if you'd like to add a bias term, otherwise assumes bias of `none`",
    )
    parser.add_argument(
        "--method",
        default="lm",
        const="lm",
        nargs="?",
        choices=["ls", "wp", "lm", "rp"],
        help="Specify the method to match the two sets of colors. Default is: %(default)s",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=2,
        help="Degree of root polynomial method",
    )
    parser.add_argument(
        "--enforce-whitepoint",
        action="store_true",
        default=False,
        help="Include this flag if you want to enforce that the matrix maps (1,1,1) to (1,1,1), skipping the white point adjustment step for the wp method.",
    )
    parser.add_argument(
        "--tall-chart",
        action="store_true",
        default=False,
        help="Set this flag to search for a 6x4 chart instead of a 4x6 chart.",
    )
    parser.add_argument(
        "--chart-layout",
        type=str,
        help="Specify two integers separated by a comma to indicate (num_patch_rows) x (num_patch_cols)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        default=False,
        help="Just do pixel per pixel match.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        default=False,
        help="Skip showing you the images.",
    )
    args = parser.parse_args()
    print(args)

    # Want to find transformation that converts src to ref.
    ref = input("target image file path: ") if args.target is None else args.target
    src = input("source image file path: ") if args.source is None else args.source
    method = input("method {ls, wp}: ") if args.method is None else args.method

    ref_img = open_image(ref)
    src_img = open_image(src)

    chart_shape = (6, 4) if args.tall_chart else (4, 6)
    if args.chart_layout is not None:
        chart_shape = tuple([int(x) for x in args.chart_layout.split(",")])
        assert len(chart_shape) == 2 and all(
            [x > 0 for x in chart_shape]
        ), f"Invalid chart_shape: {chart_shape}"
    if args.no_chart:
        ref_samples = ref_img
        src_samples = src_img
        assert ref_img.shape == src_img.shape
        h, w, c = ref_img.shape
        chart_shape = (h, w)
        scaled_src_samples = src_samples
    else:
        ref_samples, ref_positions = get_samples(ref_img, chart_shape, flat=True)
        src_samples, src_positions = get_samples(src_img, chart_shape, flat=True)
        premultiply_amt = np.mean(ref_samples / src_samples)
        print(f"Scaling source samples by {premultiply_amt} before fitting.")
        scaled_src_samples = src_samples * premultiply_amt

    fit_colors: Callable[
        [np.ndarray, np.ndarray, Namespace],
        Tuple[Any, Any, Callable[[np.ndarray], np.ndarray]],
    ]
    if method == "ls":
        fit_colors = fit_colors_ls
    elif method == "wp":
        fit_colors = fit_colors_wppls
    elif method == "lm":
        fit_colors = fit_colors_log_mat
    elif method == "rp":
        fit_colors = fit_colors_rp

    parameters, inv_parameters, model_func = fit_colors(
        scaled_src_samples, ref_samples, args
    )

    estimated_ref_samples = model_func(flatten(scaled_src_samples)).reshape(
        scaled_src_samples.shape
    )
    print("Initial mean ABS error: ", np.mean(np.abs(scaled_src_samples - ref_samples)))
    print(
        "Final mean ABS error: ", np.mean(np.abs(estimated_ref_samples - ref_samples))
    )
    print("Initial MSE error: ", np.mean((scaled_src_samples - ref_samples) ** 2))
    print("Final MSE error: ", np.mean((estimated_ref_samples - ref_samples) ** 2))
    print("Forward matrix: ", repr(parameters))
    print("Inverse: ", repr(inv_parameters))

    src_img_shape = src_img.shape
    if not args.no_ui:
        if not args.no_chart:
            draw_samples(
                src_img * premultiply_amt,
                RGBChart(scaled_src_samples),
                RGBChart(ref_samples),
                src_positions,
                title="Source Samples",
            )
            draw_samples(
                ref_img,
                RGBChart(scaled_src_samples),
                RGBChart(ref_samples),
                ref_positions,
                title="Target Samples",
            )
            draw_samples(
                model_func(flatten(src_img) * premultiply_amt).reshape(src_img_shape),
                RGBChart(estimated_ref_samples),
                RGBChart(ref_samples),
                src_positions,
                title="Corrected source samples",
            )
        else:
            show_image(src_img ** (1.0 / 2.4), "Source")
            show_image(ref_img ** (1.0 / 2.4), "Reference")
            show_image(
                model_func(flatten(src_img)).reshape(src_img_shape) ** (1.0 / 2.4),
                "Converted Source",
            )
