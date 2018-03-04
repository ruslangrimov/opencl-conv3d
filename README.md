# opencl-conv3d
OpenCL 3d convolution

## Структура проекта

```naive_conv3d``` - наивная реализация на numpy

```pyopencl_conv3d``` - реализация на OpenCL

```pyopencl_conv3d/conv3d_v1.cl``` - код OpenCL

```pyopencl_conv3d/conv3d.py``` - python обёртка над ядрами OpenCL

```test_conv3d.py``` - тесты для pytest

## Пример использования

```python
import numpy as np

from naive_conv3d.conv3d import conv3d as n_conv3d
from pyopencl_conv3d.conv3d import conv3d as cl_conv3d
from pyopencl_conv3d.conv3d import init as cl_init

# Паддинги по глубине, высоте и ширине
pads = [0, 1, 1]

# Шаги
strides = [1, 2, 2]

# Размер батча
b_size = 4

# Количество каналов
ch = 8
# Входные данные, 5d тензор (batch_size, channels, depth, height, width)
x_shape = (b_size, ch, 8, 32, 32)
x = np.random.random(np.prod(x_shape)).astype(np.float32).reshape(x_shape)

# Количество фильтров
f_count = 16
# Веса фильтров
w_shape = (f_count, ch, 3, 3, 3)
w = np.random.random(np.prod(w_shape)).astype(np.float32).reshape(w_shape)

# Инициализируем OpenCL платформу и устройство
ctx, queue, program = cl_init(0, 0)

# Вычислим свёртку
cl_res = cl_conv3d(ctx, queue, program, x, w, pads, strides)

# Проверим сходится ли с наивной реализацией
n_res = n_conv3d(x, w, pads, strides)

print("Everything is OK" if np.isclose(n_res, cl_res).all() else "There is an error")


```
