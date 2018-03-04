# opencl-conv3d
OpenCL 3d convolution

## ��������� �������

```naive_conv3d``` - ������� ���������� �� numpy

```pyopencl_conv3d``` - ���������� �� OpenCL

```pyopencl_conv3d/conv3d_v1.cl``` - ��� OpenCL

```pyopencl_conv3d/conv3d.py``` - python ������ ��� ������ OpenCL

```test_conv3d.py``` - ����� ��� pytest

## ������ �������������

```python
import numpy as np

from naive_conv3d.conv3d import conv3d as n_conv3d
from pyopencl_conv3d.conv3d import conv3d as cl_conv3d
from pyopencl_conv3d.conv3d import init as cl_init

# �������� �� �������, ������ � ������
pads = [0, 1, 1]

# ����
strides = [1, 2, 2]

# ������ �����
b_size = 4

# ���������� �������
ch = 8
# ������� ������, 5d ������ (batch_size, channels, depth, height, width)
x_shape = (b_size, ch, 8, 32, 32)
x = np.random.random(np.prod(x_shape)).astype(np.float32).reshape(x_shape)

# ���������� ��������
f_count = 16
# ���� ��������
w_shape = (f_count, ch, 3, 3, 3)
w = np.random.random(np.prod(w_shape)).astype(np.float32).reshape(w_shape)

# �������������� OpenCL ��������� � ����������
ctx, queue, program = cl_init(0, 0)

# �������� ������
cl_res = cl_conv3d(ctx, queue, program, x, w, pads, strides)

# �������� �������� �� � ������� �����������
n_res = n_conv3d(x, w, pads, strides)

print("Everything is OK" if np.isclose(n_res, cl_res).all() else "There is an error")


```
