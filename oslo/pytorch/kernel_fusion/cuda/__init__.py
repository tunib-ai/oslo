from oslo.pytorch._C import CUDABinder

CUDA = None

if CUDA is None:
    CUDA = CUDABinder().bind()
