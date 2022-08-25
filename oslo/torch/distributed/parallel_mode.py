from enum import Enum


class ParallelMode(Enum):
    """Enum class about parallelization mode."""

    # global parallel groups
    GLOBAL = "global"

    # data parallel groups
    DATA = "data"

    # model parallel groups - containing tensor and pipeline parallel groups
    # this is added to facilitate amp and grad clipping in hybrid parallel
    MODEL = "model"

    # pipeline parallel groups
    PIPELINE = "pipe"

    # tensor parallel groups - containing all ranks in tensor parallel
    TENSOR = "tensor"

    # sequence parallel groups
    SEQUENCE = "sequence"
    SEQUENCE_DP = "sequence_dp"

    # 1D tensor parallel groups
    TENSOR_1D = "tensor_1d"

    # 2D tensor parallel groups
    TENSOR_2D = "tensor_2d"
    TENSOR_2D_ROW = "tensor_2d_row"
    TENSOR_2D_COL = "tensor_2d_col"

    # 2.5D tensor parallel groups
    TENSOR_2P5D = "tensor_2p5d"
    TENSOR_2P5D_ROW = "2p5d_row"
    TENSOR_2P5D_COL = "2p5d_col"
    TENSOR_2P5D_DEP = "2p5d_dep"
    TENSOR_2P5D_XZ = "2p5d_xz"

    # 3D tensor parallel groups
    TENSOR_3D = "tensor_3d"
    TENSOR_3D_INPUT = "tensor_3d_input"
    TENSOR_3D_WEIGHT = "tensor_3d_weight"
    TENSOR_3D_OUTPUT = "tensor_3d_output"

    # Expert parallel groups
    EXPERT = "expert"
