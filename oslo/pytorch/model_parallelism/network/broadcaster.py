import torch
from torch import Tensor
import torch.distributed as dist


NoneType = type(None)


class Broadcaster(object):
    """
    Broadcaster hat can communicate various data types.
    """

    def __init__(self):
        self.device = torch.device(torch.cuda.current_device())
        self.INSTRUCTIONS = {
            bool: {"send": self.send_bool, "recv": self.recv_bool},
            int: {"send": self.send_int, "recv": self.recv_int},
            float: {"send": self.send_float, "recv": self.recv_float},
            complex: {"send": self.send_complex, "recv": self.recv_complex},
            str: {"send": self.send_str, "recv": self.recv_str},
            type: {"send": self.send_type, "recv": self.recv_type},
            list: {"send": self.send_list, "recv": self.recv_list},
            tuple: {"send": self.send_tuple, "recv": self.recv_tuple},
            set: {"send": self.send_set, "recv": self.recv_set},
            dict: {"send": self.send_dict, "recv": self.recv_dict},
            NoneType: {"send": self.send_none, "recv": self.recv_none},
            torch.Size: {"send": self.send_size, "recv": self.recv_size},
            torch.Tensor: {"send": self.send_tensor, "recv": self.recv_tensor},
        }

        self.TORCH_ID_TO_DTYPE = [
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

        self.ID_TO_DTYPE = list(self.INSTRUCTIONS.keys())
        self.DTYPE_TO_ID = {dtype: idx for idx, dtype in enumerate(self.ID_TO_DTYPE)}
        self.TORCH_DTYPE_TO_ID = {
            dtype: idx for idx, dtype in enumerate(self.TORCH_ID_TO_DTYPE)
        }

    def send(self, value, group, src=0):
        _type = type(value)
        assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"
        return self.INSTRUCTIONS[_type]["send"](
            value, group=group, src=src, send_type=True
        )

    def recv(self, group, src=0):
        _type = self.INSTRUCTIONS[type]["recv"](group, src=src)
        return self.INSTRUCTIONS[_type]["recv"](group, src=src)

    def send_type(self, _type, group, src=0, send_type=False):
        assert send_type is False, "to send ``type``, we don't need to send type."
        send_type = torch.tensor(
            [self.DTYPE_TO_ID[_type]],
            dtype=torch.long,
            device=self.device,
        )
        dist.broadcast(send_type, src=src, group=group)

    def recv_type(self, group, src=0):
        recv_type = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(recv_type, group=group, src=src)
        return self.ID_TO_DTYPE[recv_type.item()]

    def send_none(self, none, group, src=0, send_type=False):
        assert none is None, f"wrong type: {none} must be {NoneType} type"
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](NoneType, group, src=src)

    def recv_none(self, group, src=0):
        return None

    def send_str(self, _str, group, src=0, send_type=False):
        assert isinstance(_str, str), f"wrong type: {_str} must be {str} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](str, group, src=src)

        send_len_string = torch.tensor(
            [len(_str)],
            dtype=torch.long,
            device=self.device,
        )

        dist.broadcast(send_len_string, group=group, src=src)
        send_string = torch.tensor(
            [ord(s) for s in _str], dtype=torch.long, device=self.device
        )
        dist.broadcast(send_string, group=group, src=src)

    def recv_str(self, group, src=0):
        recv_len_string = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(recv_len_string, group=group, src=src)
        recv_len_string = recv_len_string.item()

        recv_string = torch.tensor(
            [0] * recv_len_string,
            dtype=torch.long,
            device=self.device,
        )
        dist.broadcast(recv_string, group=group, src=src)
        return "".join([chr(s) for s in recv_string])

    def send_bool(self, _bool, group, src=0, send_type=False):
        assert isinstance(_bool, bool), f"wrong type: {_bool} must be {bool} type."
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](bool, group=group, src=src)
        send_boolean = torch.tensor(
            [1 if _bool else 0], dtype=torch.long, device=self.device
        )
        dist.broadcast(send_boolean, group=group, src=src)

    def recv_bool(self, group, src=0):
        recv_boolean = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(recv_boolean, group=group, src=src)
        recv_boolean = recv_boolean.item()

        if recv_boolean == 0:
            return False
        elif recv_boolean == 1:
            return True
        else:
            raise ValueError(
                f"Wrong value for boolean. only 0 or 1 can be supported. "
                f"but your input is {recv_boolean}"
            )

    def send_int(self, _int, group, src=0, send_type=False):
        assert isinstance(_int, int), f"wrong type: {_int} must be {int} type."
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](int, group, src=src)
        send_int = torch.tensor([_int], dtype=torch.long, device=self.device)
        dist.broadcast(send_int, group=group, src=src)

    def recv_int(self, group, src=0):
        recv_int = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(recv_int, group=group, src=src)
        return recv_int.item()

    def send_float(self, _float, group, src=0, send_type=False):
        assert isinstance(_float, float), f"wrong type: {_float} must be {float} type."
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](float, group, src=src)
        send_float = torch.tensor([_float], dtype=torch.float, device=self.device)
        dist.broadcast(send_float, group=group, src=src)

    def recv_float(self, group, src=0):
        recv_float = torch.tensor([0.0], dtype=torch.float, device=self.device)
        dist.broadcast(recv_float, group=group, src=src)
        return recv_float.item()

    def send_complex(self, _complex, group, src=0, send_type=False):
        assert isinstance(
            _complex, complex
        ), f"wrong type: {_complex} must be {complex} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](complex, group, src=src)

        send_real = torch.tensor([_complex.real], dtype=torch.float, device=self.device)
        dist.broadcast(send_real, group=group, src=src)
        send_imag = torch.tensor([_complex.imag], dtype=torch.float, device=self.device)
        dist.broadcast(send_imag, group=group, src=src)

    def recv_complex(self, group, src=0):
        recv_real = torch.tensor([0.0], dtype=torch.float, device=self.device)
        dist.broadcast(recv_real, group=group, src=src)
        recv_imag = torch.tensor([0.0], dtype=torch.float, device=self.device)
        dist.broadcast(recv_imag, group=group, src=src)
        return complex(recv_real.item(), recv_imag.item())

    def send_tensor(self, _tensor, group, src=0, send_type=False):
        assert isinstance(
            _tensor, Tensor
        ), f"wrong type: {_tensor} must be {Tensor} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](Tensor, group, src=src)

        # dtype is float32 or float16, ... (type of element)
        _dtype = self.TORCH_DTYPE_TO_ID[_tensor.dtype]
        _dtype = torch.tensor(_dtype, dtype=torch.long, device=self.device)
        dist.broadcast(_dtype, group=group, src=src)

        _requires_grad = torch.tensor(
            1 if _tensor.requires_grad else 0,
            dtype=torch.long,
            device=self.device,
        )
        dist.broadcast(_requires_grad, group=group, src=src)

        _ndims = len(_tensor.size())
        _ndims = torch.tensor(_ndims, dtype=torch.long, device=self.device)
        dist.broadcast(_ndims, group=group, src=src)

        _shape = list(_tensor.size())
        _shape = torch.tensor(_shape, dtype=torch.long, device=self.device)
        dist.broadcast(_shape, group=group, src=src)

        if _tensor.dtype == torch.bool:
            _tensor = _tensor.half()

        if not _tensor.is_contiguous():
            _tensor = _tensor.contiguous()

        dist.broadcast(_tensor, group=group, src=src)

    def recv_tensor(self, group, src=0):
        _dtype = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(_dtype, group=group, src=src)
        _dtype = self.TORCH_ID_TO_DTYPE[_dtype.item()]

        _requires_grad = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(_requires_grad, group=group, src=src)
        _requires_grad = True if _requires_grad.item() == 1 else False

        _ndims = torch.tensor([0], dtype=torch.long, device=self.device)
        dist.broadcast(_ndims, group=group, src=src)

        _shape = torch.tensor([0] * _ndims.item(), dtype=torch.long, device=self.device)
        dist.broadcast(_shape, group=group, src=src)
        _shape = tuple(_shape.tolist())

        if _dtype == torch.bool:
            __dtype = torch.half
        else:
            __dtype = _dtype

        recv_tensor = torch.zeros(size=_shape, dtype=__dtype, device=self.device)
        recv_tensor.requires_grad = _requires_grad and recv_tensor.is_floating_point()
        dist.broadcast(recv_tensor, group=group, src=src)

        if _dtype == torch.bool:
            recv_tensor = recv_tensor.bool()

        return recv_tensor

    def send_list(self, _list, group, src=0, send_type=False):
        assert isinstance(_list, list), f"wrong type: {_list} must be {list} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](list, group, src=src)

        list_len = len(_list)
        self.send_int(list_len, group, src=src)

        for item in _list:
            _type = type(item)
            assert _type in self.ID_TO_DTYPE, f"unsupported type: {_type}"
            self.INSTRUCTIONS[_type]["send"](item, group, src=src, send_type=True)

    def recv_list(self, group, src=0):
        len_list = self.recv_int(group, src=src)
        output_list = []

        for _ in range(len_list):
            _type = self.INSTRUCTIONS[type]["recv"](group, src=src)
            _recv = self.INSTRUCTIONS[_type]["recv"](group, src=src)
            output_list.append(_recv)

        return output_list

    def send_set(self, _set, group, src=0, send_type=False):
        assert isinstance(_set, set), f"wrong type: {_set} must be {set} type."
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](set, group, src=src)
        self.send_list(list(_set), group, src=src, send_type=False)

    def recv_set(self, group, src=0):
        return set(self.recv_list(group, src=src))

    def send_tuple(self, _tuple, group, src=0, send_type=False):
        assert isinstance(_tuple, tuple), f"wrong type: {_tuple} must be {tuple} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](tuple, group, src=src)

        self.send_list(list(_tuple), group, src=src, send_type=False)

    def recv_tuple(self, group, src=0):
        return tuple(self.recv_list(group, src=src))

    def send_size(self, _size, group, src=0, send_type=False):
        assert isinstance(
            _size, torch.Size
        ), f"wrong type: {_size} must be {torch.Size} type."

        # type is ``torch.Tensor``
        if send_type is True:
            self.INSTRUCTIONS[type]["send"](torch.Size, group, src=src)

        self.send_list(list(_size), group, src=src, send_type=False)

    def recv_size(self, group, src=0):
        return torch.Size(self.recv_list(group, src=src))

    def send_dict(self, _dict, group, src=0, send_type=False):
        assert isinstance(_dict, dict), f"wrong type: {_dict} must be {dict} type."

        if send_type is True:
            self.INSTRUCTIONS[type]["send"](dict, group, src=src)

        dict_len = len(_dict)
        self.send_int(dict_len, group, src=src)

        for key, val in _dict.items():
            _type_key, _type_val = type(key), type(val)
            assert _type_key in self.ID_TO_DTYPE, f"unsupported type: {_type_key}"
            assert _type_val in self.ID_TO_DTYPE, f"unsupported type: {_type_val}"
            self.INSTRUCTIONS[_type_key]["send"](key, group, src=src, send_type=True)
            self.INSTRUCTIONS[_type_val]["send"](val, group, src=src, send_type=True)

    def recv_dict(self, group, src=0):
        len_dict = self.recv_int(group, src=src)
        output_dict = {}

        for _ in range(len_dict):
            _key_type = self.INSTRUCTIONS[type]["recv"](group, src=src)
            _key_recv = self.INSTRUCTIONS[_key_type]["recv"](group, src=src)
            _val_type = self.INSTRUCTIONS[type]["recv"](group, src=src)
            _val_recv = self.INSTRUCTIONS[_val_type]["recv"](group, src=src)
            output_dict[_key_recv] = _val_recv

        return output_dict
