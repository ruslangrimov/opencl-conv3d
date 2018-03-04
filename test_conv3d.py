import pytest
from numpy.testing import assert_allclose
import numpy as np
from naive_conv3d.conv3d import conv3d as n_conv3d
from naive_conv3d.conv3d import n_conv3d as n_n_conv3d
from naive_conv3d.conv3d import vol2col as n_vol2col
from naive_conv3d.conv3d import m2m as n_m2m

from pyopencl_conv3d.conv3d import conv3d as cl_conv3d
from pyopencl_conv3d.conv3d import init as cl_init
from pyopencl_conv3d.conv3d import vol2col as cl_vol2col
from naive_conv3d.conv3d import m2m as cl_m2m

ctx, queue, program = cl_init(0, 0)

@pytest.mark.parametrize('ch', [4, 11])
@pytest.mark.parametrize('f_count', [5, 9])
@pytest.mark.parametrize('inp_size', [(8, 8, 8), (11, 8, 7)])
@pytest.mark.parametrize('k_size', [(3, 3, 3), (4, 3, 5)])
@pytest.mark.parametrize('pads', [[0, 0, 0], [0, 2, 2]])
@pytest.mark.parametrize('strides', [[1, 1, 1], [2, 2, 2]])
@pytest.mark.parametrize('transpose', [False, True])
def test_conv3d(ch, f_count, inp_size, k_size, pads, strides, transpose):
    b_size = 2

    x_shape = (b_size, ch,) + inp_size
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    x = x / x.max()

    w_shape = (f_count, ch,) + k_size
    w = np.arange(np.prod(w_shape), dtype=np.float32).reshape(w_shape)
    w = w / w.max()

    n_res = n_conv3d(x, w, pads, strides)
    cl_res = cl_conv3d(ctx, queue, program, x, w, pads, strides, transpose)
    assert_allclose(n_res, cl_res, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize('ch', [4, 11])
@pytest.mark.parametrize('f_count', [5, 9])
@pytest.mark.parametrize('inp_size', [(8, 8, 8), (11, 8, 7)])
@pytest.mark.parametrize('k_size', [(3, 3, 3), (4, 3, 5)])
@pytest.mark.parametrize('pads', [[0, 0, 0], [0, 2, 2]])
@pytest.mark.parametrize('strides', [[1, 1, 1], [2, 2, 2]])
def test_vol2col(ch, f_count, inp_size, k_size, pads, strides):
    b_size = 2

    x_shape = (b_size, ch,) + inp_size
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    x = x / x.max()

    w_shape = (f_count, ch,) + k_size
    w = np.arange(np.prod(w_shape), dtype=np.float32).reshape(w_shape)
    w = w / w.max()

    n_cols = n_vol2col(x, w_shape[2:], pads, strides)
    cl_cols = cl_vol2col(ctx, queue, program, x, w_shape[2:], pads, strides)
    assert_allclose(cl_cols, n_cols, rtol=1e-05, atol=1e-08)


@pytest.mark.parametrize('ch', [4, 11])
@pytest.mark.parametrize('f_count', [5, 9])
@pytest.mark.parametrize('inp_size', [(8, 8, 8), (11, 8, 7)])
@pytest.mark.parametrize('k_size', [(3, 3, 3), (4, 3, 5)])
@pytest.mark.parametrize('transpose', [False, True])
def test_m2m(ch, f_count, inp_size, k_size, transpose):
    b_size = 2

    x_shape = (b_size, ch,) + inp_size
    x = np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
    x = x / x.max()

    w_shape = (f_count, ch,) + k_size
    w = np.arange(np.prod(w_shape), dtype=np.float32).reshape(w_shape)
    w = w / w.max()

    n_cols = n_vol2col(x, w_shape[2:], [0, 0, 0], [1, 1, 1])

    n_res = n_m2m(w, n_cols)
    cl_res = cl_m2m(w, n_cols)
    assert_allclose(cl_res, n_res, rtol=1e-05, atol=1e-08)
