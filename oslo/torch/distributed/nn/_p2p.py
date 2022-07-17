from typing import Any

import torch
import torch.distributed as dist

from oslo.torch.distributed.parallel_mode import ParallelMode
from oslo.torch.distributed.parallel_context import ParallelContext

NoneType = type(None)

TORCH_ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]

TORCH_DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(TORCH_ID_TO_DTYPE)}


ID_TO_DTYPE = [
    bool,
    int,
    float,
    complex,
    str,
    type,
    list,
    tuple,
    set,
    dict,
    NoneType,
    torch.Size,
    torch.Tensor,
]

DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(ID_TO_DTYPE)}


def _device():
    return torch.device(torch.cuda.current_device())


class _P2P(object):
    def __init__(self):
        self.INSTRUCTIONS = {
            bool: {"send": self._send_bool, "recv": self._recv_bool},
            int: {"send": self._send_int, "recv": self._recv_int},
            float: {"send": self._send_float, "recv": self._recv_float},
            complex: {"send": self._send_complex, "recv": self._recv_complex},
            str: {"send": self._send_str, "recv": self._recv_str},
            type: {"send": self._send_type, "recv": self._recv_type},
            list: {"send": self._send_list, "recv": self._recv_list},
            tuple: {"send": self._send_tuple, "recv": self._recv_tuple},
            set: {"send": self._send_set, "recv": self._recv_set},
            dict: {"send": self._send_dict, "recv": self._recv_dict},
            NoneType: {"send": self._send_none, "recv": self._recv_none},
            torch.Size: {"send": self._send_size, "recv": self._recv_size},
            torch.Tensor: {"send": self._send_tensor, "recv": self._recv_tensor},
        }

    @staticmethod
    def _send_type(
        data: type,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, type), f"wrong type: {data} must be {type} type."
        assert send_type is False, "to send `type`, you don't need to send type."

        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([DTYPE_TO_ID[data]], dtype=torch.long, device=_device())
        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_type(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(data, src=src_rank, group=group)
        return ID_TO_DTYPE[data.item()]

    def _send_none(
        self,
        data: NoneType,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(
            data, NoneType
        ), f"wrong type: {data} must be {NoneType} type."

        if send_type is True:
            self._send_type(
                NoneType, parallel_context=parallel_context, dst_rank=dst_rank
            )

    def _recv_none(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        return None

    def _send_str(
        self,
        data: str,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, str), f"wrong type: {data} must be {str} type."

        if send_type is True:
            self._send_type(str, parallel_context=parallel_context, dst_rank=dst_rank)

        group = parallel_context.get_group(parallel_mode)

        length = torch.tensor([len(data)], dtype=torch.long, device=_device())
        dist.send(length, dst=dst_rank, group=group)

        data = torch.tensor([ord(s) for s in data], dtype=torch.long, device=_device())
        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_str(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)

        length = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(length, src=src_rank, group=group)

        data = torch.tensor([0] * length, dtype=torch.long, device=_device())
        dist.recv(data, src=src_rank, group=group)

        return "".join([chr(s) for s in data])

    def _send_bool(
        self,
        data: bool,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, bool), f"wrong type: {data} must be {bool} type."

        if send_type is True:
            self._send_type(bool, parallel_context=parallel_context, dst_rank=dst_rank)

        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([1 if data else 0], dtype=torch.long, device=_device())

        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_bool(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=_device())

        dist.recv(data, src=src_rank, group=group)

        if data == 0:
            return False
        elif data == 1:
            return True
        else:
            raise ValueError(
                f"Wrong value for boolean. only 0 or 1 can be supported. "
                f"but your input is {data}."
            )

    def _send_int(
        self,
        data: int,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, int), f"wrong type: {data} must be {int} type."

        if send_type is True:
            self._send_type(int, parallel_context=parallel_context, dst_rank=dst_rank)

        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([data], dtype=torch.long, device=_device())
        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_int(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(data, src=src_rank, group=group)
        return data.item()

    def _send_float(
        self,
        data: float,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, float), f"wrong type: {data} must be {float} type."

        if send_type is True:
            self._send_type(float, parallel_context=parallel_context, dst_rank=dst_rank)

        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([data], dtype=torch.float, device=_device())
        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_float(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)
        data = torch.tensor([0.0], dtype=torch.float, device=_device())
        dist.recv(data, src=src_rank, group=group)
        return data.item()

    def _send_complex(
        self,
        data: complex,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, complex), f"wrong type: {data} must be {complex} type."

        if send_type is True:
            self._send_type(
                complex, parallel_context=parallel_context, dst_rank=dst_rank
            )

        group = parallel_context.get_group(parallel_mode)
        real = torch.tensor([data.real], dtype=torch.float, device=_device())
        dist.send(real, dst=dst_rank, group=group)

        iamg = torch.tensor([data.imag], dtype=torch.float, device=_device())
        dist.send(iamg, dst=dst_rank, group=group)

    @staticmethod
    def _recv_complex(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)
        real = torch.tensor([0.0], dtype=torch.float, device=_device())
        dist.recv(real, src=src_rank, group=group)

        imag = torch.tensor([0.0], dtype=torch.float, device=_device())
        dist.recv(imag, src=src_rank, group=group)

        return complex(real=real.item(), imag=imag.item())

    def _send_tensor(
        self,
        data: torch.Tensor,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(
            data, torch.Tensor
        ), f"wrong type: {data} must be {torch.Tensor} type."

        if send_type is True:
            self._send_type(
                torch.Tensor, parallel_context=parallel_context, dst_rank=dst_rank
            )

        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor(
            TORCH_DTYPE_TO_ID[data.dtype], dtype=torch.long, device=_device()
        )
        dist.send(dtype, dst=dst_rank, group=group)

        requires_grad = torch.tensor(
            1 if data.requires_grad else 0, dtype=torch.long, device=_device()
        )
        dist.send(requires_grad, dst=dst_rank, group=group)

        dims = torch.tensor(len(data.size()), dtype=torch.long, device=_device())
        dist.send(dims, dst=dst_rank, group=group)

        shape = torch.tensor(list(data.size()), dtype=torch.long, device=_device())
        dist.send(shape, dst=dst_rank, group=group)

        if not data.is_contiguous():
            data = data.contiguous()

        dist.send(data, dst=dst_rank, group=group)

    @staticmethod
    def _recv_tensor(
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        group = parallel_context.get_group(parallel_mode)

        dtype = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(dtype, src=src_rank, group=group)
        dtype = TORCH_ID_TO_DTYPE[dtype]

        requires_grad = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(requires_grad, src=src_rank, group=group)
        requires_grad = True if requires_grad.item() == 1 else False

        dims = torch.tensor([0], dtype=torch.long, device=_device())
        dist.recv(dims, src=src_rank, group=group)
        dims = dims.item()

        shape = torch.tensor([0] * dims, dtype=torch.long, device=_device())
        dist.recv(shape, src=src_rank, group=group)
        shape = tuple(shape.tolist())

        data = torch.zeros(size=shape, dtype=dtype, device=_device())
        data.requires_grad = requires_grad and data.is_floating_point()
        dist.recv(data, src=src_rank, group=group)

        return data

    def _send_list(
        self,
        data: list,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, list), f"wrong type: {data} must be {list} type."

        if send_type is True:
            self._send_type(list, parallel_context=parallel_context, dst_rank=dst_rank)

        len_list = len(data)

        self._send_int(
            len_list,
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        for item in data:
            _type = type(item)
            assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"
            self.INSTRUCTIONS[_type]["send"](
                data=item,
                dst_rank=dst_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
                send_type=True,
            )

    def _recv_list(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        output_list = []

        len_list = self._recv_int(
            src_rank=src_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        for _ in range(len_list):
            _type = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )

            assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"

            _item = self.INSTRUCTIONS[_type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
            output_list.append(_item)

        return output_list

    def _send_set(
        self,
        data: set,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, set), f"wrong type: {data} must be {set} type."

        if send_type is True:
            self._send_type(set, parallel_context=parallel_context, dst_rank=dst_rank)

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_set(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        return set(
            self._recv_list(
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
        )

    def _send_tuple(
        self,
        data: tuple,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, tuple), f"wrong type: {data} must be {tuple} type."

        if send_type is True:
            self._send_type(tuple, parallel_context=parallel_context, dst_rank=dst_rank)

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_tuple(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        return tuple(
            self._recv_list(
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
        )

    def _send_size(
        self,
        data: torch.Size,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(
            data, torch.Size
        ), f"wrong type: {data} must be {torch.Size} type."

        if send_type is True:
            self._send_type(
                torch.Size, parallel_context=parallel_context, dst_rank=dst_rank
            )

        self._send_list(
            list(data),
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
            send_type=False,
        )

    def _recv_size(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        return torch.Size(
            self._recv_list(
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
        )

    def _send_dict(
        self,
        data: dict,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
        send_type: bool = False,
    ):
        assert isinstance(data, dict), f"wrong type: {data} must be {dict} type."

        if send_type is True:
            self._send_type(dict, parallel_context=parallel_context, dst_rank=dst_rank)

        len_dict = len(data)

        self._send_int(
            len_dict,
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        for key, val in data.items():
            _type_key, _type_val = type(key), type(val)
            assert _type_key in ID_TO_DTYPE, f"unsupported type: {_type_key}"
            assert _type_val in ID_TO_DTYPE, f"unsupported type: {_type_val}"
            self.INSTRUCTIONS[_type_key]["send"](
                key,
                dst_rank=dst_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
                send_type=True,
            )
            self.INSTRUCTIONS[_type_val]["send"](
                val,
                dst_rank=dst_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
                send_type=True,
            )

    def _recv_dict(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        len_dict = self._recv_int(
            src_rank=src_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )
        output_dict = {}

        for _ in range(len_dict):
            _key_type = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
            _key_recv = self.INSTRUCTIONS[_key_type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
            _val_type = self.INSTRUCTIONS[type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
            _val_recv = self.INSTRUCTIONS[_val_type]["recv"](
                src_rank=src_rank,
                parallel_context=parallel_context,
                parallel_mode=parallel_mode,
            )
            output_dict[_key_recv] = _val_recv

        return output_dict

    def send(
        self,
        data: Any,
        dst_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        _type = type(data)
        assert _type in ID_TO_DTYPE, f"unsupported type: {_type}"

        self.INSTRUCTIONS[_type]["send"](
            data,
            dst_rank=dst_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
            send_type=True,
        )

    def recv(
        self,
        src_rank: int,
        parallel_context: ParallelContext,
        parallel_mode: ParallelMode = ParallelMode.PIPELINE,
    ):
        _type = self.INSTRUCTIONS[type]["recv"](
            src_rank=src_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )

        return self.INSTRUCTIONS[_type]["recv"](
            src_rank=src_rank,
            parallel_context=parallel_context,
            parallel_mode=parallel_mode,
        )
