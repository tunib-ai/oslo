import torch


def is_activation_to_checkpoint(item, world_size):
    """
    Check is activation to checkpoint or not

    Args:
        item (Any): any object
        world_size (int): model parallel world size

    Returns:
        bool: is activation to checkpoint or not
    """
    return (
        torch.is_tensor(item)
        and item.is_floating_point()
        and item.numel() >= world_size
        and item.requires_grad
    )


def copy_to_device(item, device, criterion):
    """
    Copy item to specific device when criterion is satisfied.

    Args:
        item (Any): any object
        device (device): torch device
        criterion (function): criterion function

    Returns:
        Union[Tensor, List, Tuple, Dict]: items that is copied to specific device
    """
    if torch.is_tensor(item) and criterion(item):
        return item.to(device)
    elif isinstance(item, list):
        return [copy_to_device(i, device, criterion) for i in item]
    elif isinstance(item, tuple):
        return tuple([copy_to_device(i, device, criterion) for i in item])
    elif isinstance(item, dict):
        return {k: copy_to_device(v, device, criterion) for k, v in item.items()}


def detach(args, device=None):
    """
    Detach all the tensors in the tuple of data.

    Args:
        args (Tuple[Any]): tuple of data
        device (torch.device): device object

    Returns:
        Tuple[Any]: tuple of detached data.
    """
    if isinstance(args, tuple):
        output = []
        for arg in args:
            if not isinstance(arg, torch.Tensor):
                output.append(arg)
                continue

            requires_grad = arg.requires_grad

            if device is not None:
                x = arg.to(device=device)
            else:
                x = arg

            x = x.detach()
            x.requires_grad = requires_grad
            output.append(x)

        return tuple(output)
    else:
        raise RuntimeError(
            "Only tuple of tensors is supported. Got Unsupported input type: ",
            type(args).__name__,
        )


def extract_tensors(all_objects):
    """
    Separate objects in list/tuple into tensors and non-tensors and create a mapping to enable re-aggregation.
    The order of tensors and non-tensors is preserved in their respective output groups.

    Args:
        all_objects (list/tuple): Objects containing tensors and non-tensors to be split.

    Returns:
        tuple: Containing tensors, non-tensors, and bools of whether each position in original list/tuple was a tensor.
    """
    tensor_objects = [v for v in all_objects if torch.is_tensor(v)]
    non_tensor_objects = [v for v in all_objects if not torch.is_tensor(v)]
    tensor_flags = [torch.is_tensor(v) for v in all_objects]

    if type(all_objects) is tuple:
        return tuple(tensor_objects), tuple(non_tensor_objects), tuple(tensor_flags)

    return tensor_objects, non_tensor_objects, tensor_flags
