import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt  # type:ignore
from tqdm import tqdm  # type:ignore
from argparse import ArgumentParser, Namespace
from typing import Callable, Tuple, Any
from itertools import product
from src.images import (
    flatten,
    get_samples,
    open_image,
    draw_samples,
    show_image,
)
from src.color_conversions import RGBChart

# Intended to help you match one Scene Linear image of a color chart
# from one camera to another.


class glass(nn.Module):
    # Model the change from src_img to ref_img as ref = A@src + b
    # A and b unknown, A \in R^3x3 and b \in R^3

    def __init__(self, bias="1d"):
        super().__init__()
        self.mat = nn.Linear(3, 3, bias=False)
        if bias == "1d":
            self.bias = nn.parameter.Parameter(torch.tensor(0.0))
            print("Applying 1d bias, model is: A @ (source + bias) = target")
        elif bias == "3d":
            self.bias = nn.parameter.Parameter(
                torch.tensor(
                    [
                        0.0,
                        0.0,
                        0.0,
                    ]
                )
            )
            print("Applying 3d bias, model is: A @ (source + bias) = target")
        elif bias == None:
            self.bias = torch.tensor(0.0)
            print("Applying no bias, model is: A @ (source) = target")
        else:
            assert False, f"Bias type {bias} not supported."
        self.init_weights()

    def init_weights(self):
        self.mat.weight.data = torch.eye(3)

    def forward(self, x):
        return self.mat(x + self.bias)

    def print_weights(self):
        print("Pre-bias: ", self.bias.data.detach().cpu().numpy())
        print("matrix: ", self.mat.weight.detach().cpu().numpy())


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


def fit_colors_gd(input_rgb, output_rgb, args):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    ds = TensorDataset(
        torch.tensor(flatten(input_rgb), device=device, dtype=torch.float32),
        torch.tensor(flatten(output_rgb), device=device, dtype=torch.float32),
    )

    dl = DataLoader(ds, batch_size=min(100, len(ds)))

    model = glass(bias=args.bias).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def relu_log(x: torch.Tensor) -> torch.Tensor:
        # Log x, but transition to y = x - 1 at x <= 1
        mask = x > 1
        x[mask] = torch.log(x[mask])
        x[~mask] -= 1
        return x

    epochs = int(max(1, 200000 / len(ds)))
    loss_fn = lambda y, y_pred: torch.mean(
        (relu_log(1 + y_pred) - relu_log(1 + y)) ** 2
    )

    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            for x, y in dl:
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y)
                loss.backward()
                optimizer.step()

                err = torch.mean(torch.abs(output - y)).detach().cpu().numpy()
                pbar.set_postfix(error=err)
            pbar.update(1)

    model.eval()
    return (
        (
            model.mat.weight.data.detach().cpu().numpy(),
            model.bias.detach().cpu().numpy(),
        ),
        np.linalg.pinv(model.mat.weight.data.detach().cpu().numpy()),
        lambda x: model(torch.tensor(x, device=device, dtype=torch.float32))
        .detach()
        .cpu()
        .numpy(),
    )


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
        default=None,
        help="Optionally set to {1d, 3d} if you'd like to add a bias term, otherwise assumes bias of 0",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Specify the method to match the two sets of colors. Options are: {gd, ls, wp}, ls is default",
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
        "--no-chart",
        action="store_true",
        default=False,
        help="Just do pixel per pixel match.",
    )
    args = parser.parse_args()

    # Want to find transformation that converts src to ref.
    ref = input("target image file path: ") if args.target is None else args.target
    src = input("source image file path: ") if args.source is None else args.source
    method = input("method {gd, ls, wp}: ") if args.method is None else args.method

    ref_img = open_image(ref)
    src_img = open_image(src)

    chart_shape = (6, 4) if args.tall_chart else (4, 6)
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
    elif method == "gd":
        fit_colors = fit_colors_gd
    elif method == "wp":
        fit_colors = fit_colors_wppls

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
    if args.no_chart is False:
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
