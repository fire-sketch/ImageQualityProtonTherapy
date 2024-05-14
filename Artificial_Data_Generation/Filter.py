import numpy as np
from scipy import ndimage


def gaussian_filter_3d(inp, sigmas, size, mode):
    sigma = sigmas / size
    return ndimage.gaussian_filter(inp, sigma, mode=mode)


def __triangle_filter_1d(inp, width, spacing, axis, mode):
    x_lim = np.ceil(width / spacing) * spacing
    x = np.arange(-x_lim, x_lim + spacing, spacing)
    tri_1d = (lambda val: (1 - 1 / width * np.abs(val)))(x)
    tri_1d[tri_1d < 0] = 0
    tri_1d = tri_1d / tri_1d.sum()
    return ndimage.convolve1d(inp, tri_1d, axis=axis, mode=mode)


def triangle_filter_3d(inp, sigma, spacings, mode):
    out = None
    axes = list(range(inp.ndim))
    for axis in axes:
        out = __triangle_filter_1d(inp, sigma, spacings[axis], axis, mode)
        inp = out
    return out


def __rectangle_filter_1d(inp, width, spacing, axis, mode):
    x_lim = np.ceil(width / (spacing * 2.0)) * spacing
    x = np.arange(-x_lim, x_lim + spacing, spacing)
    rec_1d = np.zeros_like(x)
    rec_1d[(np.abs(x) <= width / 2.0) & ((np.abs(x) + spacing / 2.0) <= width / 2.0)] = 1
    for i, v in enumerate(x):
        if (np.abs(v) <= width / 2.0) and ((np.abs(v) + spacing / 2.0) > width / 2.0):
            rec_1d[i] = (width / 2.0 - np.abs(v) + spacing / 2.0) / spacing
        elif (np.abs(v) > width / 2.0) & ((np.abs(v) - spacing / 2.0) <= width / 2.0):
            rec_1d[i] = (spacing / 2 - (np.abs(v) - width / 2.0)) / spacing
    rec_1d = rec_1d / np.sum(rec_1d)
    return ndimage.convolve1d(inp, rec_1d, axis=axis, mode=mode)


def rectangle_filter_3d(inp, sigma, spacings, mode):
    out = None
    axes = list(range(inp.ndim))
    for axis in axes:
        out = __rectangle_filter_1d(inp, sigma, spacings[axis], axis, mode)
        inp = out
    return out
