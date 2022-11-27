import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
import os


def open_image(image_fn: str) -> np.ndarray:
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    img: np.ndarray = cv2.imread(image_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    print(f"Read image data type of {img.dtype}")
    if img.dtype == np.uint8 or img.dtype == np.uint16:
        img = img.astype(np.float32) / np.iinfo(img.dtype).max
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def flatten(img):
    return np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))


def sample_image(img, x, y, radius=5):
    # Samples a (2*radius, 2*radius) box of pixels surrounding the point (x,y) in the image. 0 < x, y < 1
    h, w, c = img.shape
    row, col = int(y * h), int(x * w)
    samples = img[row-radius:row+radius, col-radius:col+radius, :]
    avg = np.mean(flatten(samples), axis=0)
    return avg


def get_samples(img, patches=(6,4)):
    samples = np.zeros((patches[0], patches[1], 3))
    for patch_row in range(patches[0]):
        for patch_col in range(patches[1]):
            sample = sample_image(img, (patch_col*2+1) / (patches[1]*2), (patch_row*2+1) / (patches[0]*2))
            samples[patch_row, patch_col, :] = sample
    return samples


class glass(nn.Module):
    # Model the change from src_img to ref_img as ref = A@src + b
    # A and b unknown, A \in R^3x3 and b \in R^3

    def __init__(self, bias_1d=True):
        super().__init__()
        self.mat = nn.Linear(3, 3, bias=False)
        if bias_1d:
            self.bias = nn.parameter.Parameter(torch.tensor(0.))
        else:
            self.bias = nn.parameter.Parameter(torch.tensor([0., 0., 0.,]))
        self.init_weights()

    def init_weights(self):
        self.mat.weight.data = torch.eye(3)

    def forward(self, x):
        return self.mat(x + self.bias)

    def print_weights(self):
        print("Pre-bias: ", self.bias.data.detach().cpu().numpy())
        print("matrix: ", self.mat.weight.detach().cpu().numpy())


def fit_colors_ls(input_rgb, output_rgb, weights, args):
    # TODO: Support a bias term.
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb) # (24, 3)

    mat = np.linalg.lstsq(input_rgb_flat, output_rgb_flat)[0]
    print(mat.T)
    print("initial error: ", np.mean(np.abs(input_rgb_flat - output_rgb_flat)))
    print("error: ", np.mean(np.abs((input_rgb_flat @ mat) - output_rgb_flat)))


def fit_colors_wppls(input_rgb, output_rgb, weights, args):
    input_rgb_flat = flatten(input_rgb)
    output_rgb_flat = flatten(output_rgb) # (24, 3)

    # TODO: figure out if we need this white patch normalizing step.
    white_patch_idx = np.argmax(np.mean(output_rgb_flat, axis=1))
    print(f"white patch index: {white_patch_idx}", output_rgb_flat[white_patch_idx])
    u = np.ones((3, 1))
    N = input_rgb_flat / input_rgb_flat[[white_patch_idx], :]
    M = output_rgb_flat / output_rgb_flat[[white_patch_idx], :]
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

    print(mat.T)
    print("initial error: ", np.mean(np.abs(input_rgb_flat - output_rgb_flat)))
    print("error: ", np.mean(np.abs((input_rgb_flat @ mat) - output_rgb_flat)))
    print("initial error: ", np.mean(np.abs(N, M)))
    print("error: ", np.mean(np.abs((N @ mat) - M)))
    print(np.sum(mat.T, axis=1))

def fit_colors_gd(input_rgb, output_rgb, weights, args):
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    ds = TensorDataset(
        torch.tensor(flatten(input_rgb), device=device, dtype=torch.float32),
        torch.tensor(flatten(output_rgb), device=device, dtype=torch.float32),
        torch.tensor(flatten(weights), device=device, dtype=torch.float32),
    )

    dl = DataLoader(ds, batch_size=min(100, len(ds)))

    model = glass(bias_1d=not args.bias3d).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    def relu_log(x: torch.Tensor) -> torch.Tensor:
        # Log x, but transition to y = x - 1 at x <= 1
        mask = x > 1
        x[mask] = torch.log(x[mask])
        x[~mask] -= 1
        return x

    epochs = int(max(1, 200000 / len(ds)))
    loss_fn = lambda y, y_pred, weights: torch.mean(weights * (relu_log(1 + y_pred) - relu_log(1 + y))**2)

    with tqdm(total=epochs) as pbar:
        for e in range(epochs):
            for x, y, weight in dl:
                x = x.to(device)
                y = y.to(device)
                weight = weight.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = loss_fn(output, y, weight)
                loss.backward()
                optimizer.step()

                err = torch.mean(torch.abs(output - y)).detach().cpu().numpy()
                pbar.set_postfix(error=err)
            pbar.update(1)

    model.print_weights()
    model.eval()
    with torch.no_grad():
        transformed_src_img = model(torch.tensor(flatten(input_rgb), device=device, dtype=torch.float32)).detach().cpu().numpy()
        print("Final mean ABS error: ", np.mean(np.abs(flatten(output_rgb) - transformed_src_img)))


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
        "--bias3d",
        action="store_true",
        default=False,
        help="Include this flag if you want to apply a colored offset to the result, after the matrix."
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        help="Specify the method to match the two sets of colors. Options are: {gd, ls, wp}, ls is default"
    )
    args = parser.parse_args()

    # Want to find transformation that converts src to ref.
    ref = input("target image file path: ") if args.target is None else args.target
    src = input("source image file path: ") if args.source is None else args.source
    method = input("method {gd, ls, wp}: ") if args.method is None else args.method

    ref_img = open_image(ref)
    src_img = open_image(src)

    ref_samples = get_samples(ref_img)
    src_samples = get_samples(src_img)

    sample_weights = np.ones_like(ref_samples)

    # Eliminate the impact of specific samples like so:
    # sample_weights[0, 3, :] *= 0

    # Compute initial error
    print("Initial mean ABS error: ", np.mean(np.abs(flatten(ref_samples) - flatten(src_samples))))

    if method == "ls":
        fit_colors = fit_colors_ls
    elif method == "gd":
        fit_colors = fit_colors_gd
    elif method == "wp":
        fit_colors = fit_colors_wppls

    fit_colors(src_samples, ref_samples, sample_weights, args)
