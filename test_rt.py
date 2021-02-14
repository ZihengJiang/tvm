import tvm
from tvm import nd

ctx = tvm.opencl(0)
print('create array')
arr = nd.empty((3, 4), "float32", ctx, "texture")
print('print array')
print(arr)
print(arr.ctx)
