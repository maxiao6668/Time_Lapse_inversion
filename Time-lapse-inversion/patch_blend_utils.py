import numpy as np


def make_weight_window(d, h, window="hann", sigma=0.125):
    """
    Create a 2D weight window with higher center weights and tapered edges.
    Returns shape (d, h) with values in [0, 1].
    """
    if window == "hann":
        wd = np.hanning(d)
        wh = np.hanning(h)
    elif window == "hamming":
        wd = np.hamming(d)
        wh = np.hamming(h)
    elif window == "gaussian":
        xd = np.linspace(-1.0, 1.0, d)
        xh = np.linspace(-1.0, 1.0, h)
        wd = np.exp(-0.5 * (xd / sigma) ** 2)
        wh = np.exp(-0.5 * (xh / sigma) ** 2)
    elif window == "flat":
        wd = np.ones(d, dtype=np.float32)
        wh = np.ones(h, dtype=np.float32)
    else:
        raise ValueError(f"unknown window: {window}")

    w = np.outer(wd, wh).astype(np.float32)
    w /= np.max(w)
    return w


def stitch_weighted_3d(
    out_patch,
    accum,
    weight_sum,
    d0,
    h0,
    weight_win,
):
    """
    Accumulate a patch with weights into accum/weight_sum.
    out_patch: (d, h, w)
    accum/weight_sum: full volume (D, H, W)
    weight_win: (pd, ph) window; clipped internally for boundary patches.
    """
    d_len, h_len, _ = out_patch.shape
    w = weight_win[:d_len, :h_len][:, :, None]

    d1 = d0 + d_len
    h1 = h0 + h_len
    accum[d0:d1, h0:h1, :] += out_patch * w
    weight_sum[d0:d1, h0:h1, :] += w


def finalize_weighted(accum, weight_sum, eps=1e-6):
    return accum / np.maximum(weight_sum, eps)
