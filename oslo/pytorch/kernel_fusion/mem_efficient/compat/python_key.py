import torch


def _make_wrapper_subclass(cls, elem, device):
    # From Torch 1.11
    if hasattr(torch.Tensor, "_make_wrapper_subclass"):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            requires_grad=elem.requires_grad,
            device=(elem.device if device is None else device),
        )
    else:
        meta = elem.new_empty((0,))
        meta.set_(meta.storage(), 0, elem.size(), elem.stride())
        return torch.Tensor._make_subclass(cls, meta, elem.requires_grad)
