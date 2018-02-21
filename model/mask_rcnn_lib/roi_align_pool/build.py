import os
import torch
from torch.utils.ffi import create_extension


if not torch.cuda.is_available():
    raise ValueError('this version is for cuda only! cuda not found')

curr_dir = os.path.dirname(os.path.realpath(__file__))
print('curr_dir is:',curr_dir)


print('Including CUDA code.')
sources = ['src/roi_align_cuda.c']
headers = ['src/roi_align_cuda.h']
defines = [('WITH_CUDA', None)]
with_cuda = True
extra_objects = ['src/roi_align_kernel.cu.o']
extra_objects = [os.path.join(curr_dir, fname) for fname in extra_objects]

#  http://pytorch.org/docs/master/ffi.html
#  https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension
ffi = create_extension(
    'extension',
    headers       = headers,
    sources       = sources,
    define_macros = defines,
    relative_to   = __file__,
    with_cuda     = with_cuda,
    extra_objects = extra_objects
)

if __name__ == '__main__':
    ffi.build()