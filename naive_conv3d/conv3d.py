import numpy as np
import math as m


def calc_out_shape(in_shape, k_shape, pads=[0, 0, 0], strides=[1, 1, 1]):
    """
    Calculates the output shape based on the input shape, kernel size,
    paddings and strides.
    Input:
        - in_shape: A list with input size for each dimensions.
        - k_shape: A list with kernel size for each dimensions.
        - pads: A list with padding size for each dimensions.
        - strides: A list with stride size for each dimensions.
    Returns a list of shape (D_out, H_out, W_outt)
    """
    (D, H, W) = in_shape
    (DD, HH, WW) = k_shape

    D_out = 1 + (D + 2 * pads[0] - DD) // strides[0]
    H_out = 1 + (H + 2 * pads[1] - HH) // strides[1]
    W_out = 1 + (W + 2 * pads[2] - WW) // strides[2]

    return [D_out, H_out, W_out]


def n_conv3d(x, w, pads=[0, 0, 0], strides=[1, 1, 1]):
    """
    A naive (very naive) implementation of convolution.
    The input consists of N data points, each with C channels, depth D,
    height H and width W. We convolve each input with F different filters,
    where each filter spans all C channels and has depth DD, height HH
    and width WW.
    Input:
        - x: Input data of shape (N, C, D, H, W)
        - w: Filter weights of shape (F, C, DD, HH, WW)
        - pads: A list with padding size for each dimensions.
    Returns a ndarray of shape (N, F, D_out, H_out, W_out)
    """

    (N, C, D, H, W) = x.shape
    (F, C, DD, HH, WW) = w.shape

    D_out = m.ceil(1 + (D + 2 * pads[0] - DD) / strides[0])
    H_out = m.ceil(1 + (H + 2 * pads[1] - HH) / strides[1])
    W_out = m.ceil(1 + (W + 2 * pads[2] - WW) / strides[2])
    out = np.zeros((N, F, D_out, H_out, W_out))

    for n in range(N):
        x_pad = np.pad(x[n], ((0, 0), ) + tuple(zip(pads, pads)), 'constant')
        for f in range(F):
            for d_out in range(D_out):
                for h_out in range(H_out):
                    for w_out in range(W_out):
                        d1 = d_out * strides[0]
                        d2 = d_out * strides[0] + DD
                        h1 = h_out * strides[1]
                        h2 = h_out * strides[1] + HH
                        w1 = w_out * strides[2]
                        w2 = w_out * strides[2] + WW
                        window = x_pad[:, d1:d2, h1:h2, w1:w2]
                        out[n, f, d_out, h_out, w_out] = \
                            np.sum(window * w[f])

    return out


def vol2col(x, w_shape, pads=[0, 0, 0], strides=[1, 1, 1]):
    """
    A naive implementation of making marix of volume's 3d blocks.
    The input consists of N data points, each with C channels, height H,
    width W and depth D.
    Input:
        - x: Input data of shape (N, C, D, H, W)
        - w: A list with sizes of block's dimensions.
        - pads: A list with padding size for each dimensions.
        - strides: A list with stride size for each dimensions.
    Returns a ndarray of shape (N, C*DD*HH*WW, D_out*H_out*W_out)
    """

    (N, C, D, H, W) = x.shape
    (DD, HH, WW) = w_shape

    D_out, H_out, W_out = calc_out_shape((D, H, W), (DD, HH, WW),
                                         pads, strides)

    outs = []
    for n in range(N):
        x_pad = np.pad(x[n], ((0, 0), ) + tuple(zip(pads, pads)), 'constant')
        windows = []
        for d_out in range(D_out):
            for h_out in range(H_out):
                for w_out in range(W_out):
                    d1 = d_out * strides[0]
                    d2 = d_out * strides[0] + DD
                    h1 = h_out * strides[1]
                    h2 = h_out * strides[1] + HH
                    w1 = w_out * strides[2]
                    w2 = w_out * strides[2] + WW
                    window = x_pad[:, d1:d2, h1:h2, w1:w2].flatten()
                    windows.append(window)

        outs.append(windows)

    out = np.array(outs).transpose([0, 2, 1])
    return out


def m2m(w, c):
    """
    A naive implementation of matrix product of two arrays.
    Input:
        - w: ndarray of shape (F, DD, HH, WW).
        - c: ndarray of shape (N, C, D, H, W).
        - pads: A list with padding size for each dimensions.
        - strides: A list with stride size for each dimensions.
    Returns a ndarray of shape (N, C*DD*HH*WW, D_out*H_out*W_out)
    """
    out = np.matmul(w.reshape((w.shape[0], -1)), c)
    return out


def conv3d(x, w, pads=[0, 0, 0], strides=[1, 1, 1]):
    """
    A naive implementation of convolution.
    The input consists of N data points, each with C channels, height H,
    width W and depth D. We convolve each input with F different filters,
    where each filter spans all C channels and has height HH, width HH
    and depth DD.
    Input:
        - x: Input data of shape (N, C, D, H, W)
        - w: Filter weights of shape (F, C, DD, HH, WW)
        - pads: A list with padding size for each dimensions.
    Returns a ndarray of shape (N, F, D_out, H_out, W_out)
    """
    w_shape = w.shape
    cols = vol2col(x, w_shape[2:], pads, strides)

    out = m2m(w, cols)

    (N, C, D, H, W) = x.shape
    (DD, HH, WW) = w_shape[2:]

    D_out, H_out, W_out = calc_out_shape((D, H, W), (DD, HH, WW), pads, strides)
    return out.reshape((N, -1, D_out, H_out, W_out))
