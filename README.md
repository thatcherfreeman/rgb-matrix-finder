# rgb-matrix-finder

This script takes two **scene linear** images of a color chart and derives a best-fit RGB matrix that makes the "Source" image as similar to the "Target" image as possible. Requires OpenEXR, which you can download at one of these two links:

https://www.lfd.uci.edu/~gohlke/pythonlibs/

https://www.excamera.com/sphinx/articles-openexr.html

Also requires `pytorch`.

## Running
Run with `python regressor.py` and follow the instructions.

## Modeling

Given an RGB (3-dimensional) color patch `x` from the Source image and a RGB color patch `y` from the Target image, we suppose that `y ~ A(x + b)`, where `b` is either 1-dimensional or 3-dimensional, and `A` is a 3x3 matrix. The `b` term exists to handle any flare or reflections that was introduced or removed between the capture of the two images. We use gradient descent to identify the parameters `A` and `b`.

