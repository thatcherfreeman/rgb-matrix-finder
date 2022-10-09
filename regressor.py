import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import OpenEXR
import cv2
import matplotlib.pyplot as plt
import Imath
import array
from tqdm import tqdm


def read_exr(filepath):
    file = OpenEXR.InputFile(filepath)

    # Compute the size
    dw = file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    sz = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # Read the three color channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    img = np.stack([array.array('f', file.channel(Chan, FLOAT)).tolist() for Chan in ("R", "G", "B")], axis=1)
    img = np.reshape(img, (height, width, 3), order='C')
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

    def forward(self, x):
        return self.mat(x + self.bias)

    def print_weights(self):
        print("Pre-bias: ", self.bias.data.detach().cpu().numpy())
        print("matrix: ", self.mat.weight.detach().cpu().numpy())



# Want to find transformation that converts src to ref.
ref = input("target image file path: ")
src = input("source image file path: ")

ref_img = read_exr(ref)
src_img = read_exr(src)

ref_samples = get_samples(ref_img)
src_samples = get_samples(src_img)

sample_weights = np.ones_like(ref_samples)

# Eliminate the impact of specific samples like so:
# sample_weights[0, 3, :] *= 0

# Compute initial error
print("Initial mean ABS error: ", np.mean(np.abs(flatten(ref_img) - flatten(src_img))))

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

ds = TensorDataset(
    torch.tensor(flatten(src_samples), device=device, dtype=torch.float32),
    torch.tensor(flatten(ref_samples), device=device, dtype=torch.float32),
    torch.tensor(flatten(sample_weights), device=device, dtype=torch.float32),
)

dl = DataLoader(ds, batch_size=min(100, len(ds)))

model = glass().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

def relu_log(x: torch.Tensor) -> torch.Tensor:
    # Log x, but transition to y = x - 1 at x <= 1
    mask = x > 1
    x[mask] = torch.log(x[mask])
    x[~mask] -= 1
    return x

epochs = int(max(1, 500000 / len(ds)))
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
    transformed_src_img = model(torch.tensor(flatten(src_img), device=device, dtype=torch.float32)).detach().cpu().numpy()
    print("Final mean ABS error: ", np.mean(np.abs(flatten(ref_img) - transformed_src_img)))