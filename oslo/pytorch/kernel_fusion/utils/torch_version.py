import torch

TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])


def higher_than(major, minor):
    if (TORCH_MAJOR > major) or (TORCH_MAJOR == major and TORCH_MINOR >= minor):
        return True
    return False
