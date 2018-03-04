import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import numpy as np
import pyopencl as cl
from pyopencl import mem_flags as mf

f_size = 4

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


def init(platform=0, device=0):
    device = cl.get_platforms()[platform].get_devices()[device]
    ctx = cl.Context(devices=[device], dev_type=None)
    queue = cl.CommandQueue(ctx)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, 'conv3d_v1.cl')) as f:
        cl_code = f.read()

    program = cl.Program(ctx, cl_code).build()

    return ctx, queue, program


def k_vol2col(ctx, queue, program,
               n, data_vol, vol_offset,
               channels, depth, height, width,
               kernel_d, kernel_h, kernel_w,
               pad_d, pad_h, pad_w,
               stride_d, stride_h, stride_w,
               depth_col, height_col, width_col,
               data_col, col_offset):

    kernel = program.vol2col

    kernel.set_scalar_arg_dtypes([np.int32,  # n
                                  None,  # data_vol
                                  np.int32,  # vol_offset
                                  np.int32,  # channels
                                  np.int32,  # depth
                                  np.int32,  # height
                                  np.int32,  # width
                                  np.int32,  # kernel_d
                                  np.int32,  # kernel_h
                                  np.int32,  # kernel_w
                                  np.int32,  # pad_d
                                  np.int32,  # pad_h
                                  np.int32,  # pad_w
                                  np.int32,  # stride_d
                                  np.int32,  # stride_h
                                  np.int32,  # stride_w
                                  np.int32,  # depth_col
                                  np.int32,  # height_col
                                  np.int32,  # width_col
                                  None,  # data_col
                                  np.int32])  # col_offset

    launch = kernel(queue, [n], None,
                    n,
                    data_vol,
                    vol_offset,
                    channels, depth, height, width,
                    kernel_d, kernel_h, kernel_w,
                    pad_d, pad_h, pad_w,
                    stride_d, stride_h, stride_w,
                    depth_col, height_col, width_col,
                    data_col,
                    col_offset)

    launch.wait()


def vol2col(ctx, queue, program, x, w_shape, pads=[0, 0, 0], strides=[1, 1, 1]):
    '''
    Used only for test
    '''
    b_size, channels, depth, height, width = x.shape
    kernel_d, kernel_h, kernel_w = w_shape

    pad_d, pad_h, pad_w = pads

    stride_d, stride_h, stride_w = strides

    depth_col, height_col, width_col = \
        calc_out_shape((depth, height, width),
                       (kernel_d, kernel_h, kernel_w),
                       pads, strides)

    n = channels * depth_col*height_col*width_col

    col_shape = (b_size, kernel_w*kernel_d*kernel_h*channels,
                 depth_col*height_col*width_col)

    im_shape = (b_size, channels, depth, height, width)
    data_vol = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    #col = np.full(col_shape, 2, dtype=np.float32)
    #data_col = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=col)
    data_col = cl.Buffer(ctx, mf.WRITE_ONLY, int(np.prod(col_shape)*f_size))

    vol_offset = 0
    col_offset = 0

    for i in range(b_size):
        # print(vol_offset, col_offset)
        k_vol2col(ctx, queue, program,
                   n, data_vol, vol_offset,
                   channels, depth, height, width,
                   kernel_d, kernel_h, kernel_w,
                   pad_d, pad_h, pad_w,
                   stride_d, stride_h, stride_w,
                   depth_col, height_col, width_col,
                   data_col, col_offset)

        col_offset += int(np.prod(col_shape[1:]))
        vol_offset += int(np.prod(im_shape[1:]))

    #col = np.ones(col_shape, dtype=np.float32)
    col = np.zeros(col_shape, dtype=np.float32)
    cl.enqueue_copy(queue, col, data_col).wait()

    return col


def k_m2m(ctx, queue, program,
          p, r,
          data_m1,
          data_m2,
          m2_offset,
          q,
          data_res,
          res_offset):

    kernel = program.m2m

    kernel.set_scalar_arg_dtypes([None, None, np.int32, np.int32,
                                  None, np.int32])

    launch = kernel(queue, [p, r], None,
                    data_m1,
                    data_m2,
                    m2_offset,
                    q,
                    data_res,
                    res_offset)

    launch.wait()


def k_m2tm(ctx, queue, program,
          p, r,
          data_m1,
          data_m2,
          m2_offset,
          q,
          data_res,
          res_offset):

    kernel = program.m2tm

    kernel.set_scalar_arg_dtypes([None, None, np.int32, np.int32,
                                  None, np.int32])

    launch = kernel(queue, [p, r], None,
                    data_m1,
                    data_m2,
                    m2_offset,
                    q,
                    data_res,
                    res_offset)

    launch.wait()


def k_transpose(ctx, queue, program,
                height, width,
                data_src, src_offset,
                data_dst, dst_offset):

    kernel = program.transpose

    kernel.set_scalar_arg_dtypes([None, np.int32, None, np.int32])
    launch = kernel(queue, [height, width], None,
                    data_src, src_offset, data_dst, dst_offset)

    launch.wait()


def m2m(ctx, queue, program, w, cols, transpose=False):
    '''
    Used only for test
    '''
    col_shape = cols.shape
    b_size, q, r = col_shape
    p = w.shape[0]

    data_col = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cols)
    data_w = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    res_shape = (b_size, p, r)
    data_res = cl.Buffer(ctx, mf.WRITE_ONLY, int(np.prod(res_shape)*f_size))

    if transpose:
        data_col_r = cl.Buffer(ctx, mf.WRITE_ONLY,
                               int(np.prod(col_shape[:0:-1])*f_size))

    col_offset = 0
    res_offset = 0

    for i in range(b_size):
        if transpose:
            k_transpose(ctx, queue, program,
                        col_shape[1], col_shape[2],
                        data_col,
                        col_offset,
                        data_col_r,
                        0)

            k_m2tm(ctx, queue, program,
                  p, r,
                  data_w,
                  data_col_r,
                  0,
                  q,
                  data_res,
                  res_offset)
        else:
            k_m2m(ctx, queue, program,
                  p, r,
                  data_w,
                  data_col,
                  col_offset,
                  q,
                  data_res,
                  res_offset)

        col_offset += int(np.prod(col_shape[1:]))
        res_offset += int(np.prod(res_shape[1:]))

    res = np.zeros(res_shape, dtype=np.float32)
    cl.enqueue_copy(queue, res, data_res).wait()

    return res


def conv3d(ctx, queue, program, x, w,
           pads=[0, 0, 0], strides=[1, 1, 1], transpose=False):
    """
    An OpenCL implementation of 3d convolution.
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

    b_size, channels, depth, height, width = x.shape
    (p, _, kernel_d, kernel_h, kernel_w) = w.shape

    pad_d, pad_h, pad_w = pads

    stride_d, stride_h, stride_w = strides

    depth_col, height_col, width_col = \
        calc_out_shape((depth, height, width),
                       (kernel_d, kernel_h, kernel_w),
                       pads, strides)

    n = channels * depth_col*height_col*width_col
    q = kernel_w*kernel_d*kernel_h*channels
    r = depth_col*height_col*width_col

    col_shape = (b_size, q, r)

    im_shape = (b_size, channels, depth, height, width)
    data_vol = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=x)
    data_col = cl.Buffer(ctx, mf.WRITE_ONLY, int(np.prod(col_shape)*f_size))

    vol_offset = 0
    col_offset = 0

    for i in range(b_size):
        # print(vol_offset, col_offset)
        k_vol2col(ctx, queue, program,
                   n, data_vol, vol_offset,
                   channels, depth, height, width,
                   kernel_d, kernel_h, kernel_w,
                   pad_d, pad_h, pad_w,
                   stride_d, stride_h, stride_w,
                   depth_col, height_col, width_col,
                   data_col, col_offset)

        col_offset += int(np.prod(col_shape[1:]))
        vol_offset += int(np.prod(im_shape[1:]))

    data_w = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w)
    res_shape = (b_size, p, depth_col, height_col, width_col)
    data_res = cl.Buffer(ctx, mf.WRITE_ONLY, int(np.prod(res_shape)*f_size))

    if transpose:
        data_col_r = cl.Buffer(ctx, mf.WRITE_ONLY,
                               int(np.prod(col_shape[:0:-1])*f_size))

    col_offset = 0
    res_offset = 0

    for i in range(b_size):
        if transpose:
            k_transpose(ctx, queue, program,
                        col_shape[1], col_shape[2],
                        data_col,
                        col_offset,
                        data_col_r,
                        0)

            k_m2tm(ctx, queue, program,
                  p, r,
                  data_w,
                  data_col_r,
                  0,
                  q,
                  data_res,
                  res_offset)
        else:
            k_m2m(ctx, queue, program,
                  p, r,
                  data_w,
                  data_col,
                  col_offset,
                  q,
                  data_res,
                  res_offset)

        col_offset += int(np.prod(col_shape[1:]))
        res_offset += int(np.prod(res_shape[1:]))

    res = np.zeros(res_shape, dtype=np.float32)
    cl.enqueue_copy(queue, res, data_res).wait()

    return res

