from functools import partial

import torch

from oslo.pytorch.activation_checkpointing.utils.utils import (
    detach,
    extract_tensors,
    is_activation_to_checkpoint,
    copy_to_device,
)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, options, all_outputs, *args):
        def save_args_for_backward(*all_args):
            tensor_args, non_tensor_args, tensor_flags = extract_tensors(all_args)
            ctx._saved_tensors = tensor_args
            ctx.non_tensor_args = non_tensor_args
            ctx.tensor_flags = tensor_flags

        # 1. save some variables for backward.
        ctx.rng_tracker = options.get("rng_tracker", None)
        ctx.partitioner = options.get("partitioner", None)
        ctx.run_function = run_function

        # 2. partitioning or moving to cpu if user want.
        if ctx.partitioner.partitioned_checkpointing:
            inputs = ctx.partitioner.make_partitioned_activations(args)
        elif ctx.partitioner.cpu_checkpointing:
            inputs = copy_to_device(
                args,
                torch.device("cpu"),
                partial(
                    is_activation_to_checkpoint,
                    world_size=ctx.partitioner.mp_size,
                ),
            )

        inputs_cuda = copy_to_device(
            args,
            torch.cuda.current_device(),
            partial(
                is_activation_to_checkpoint,
                world_size=ctx.partitioner.mp_size,
            ),
        )

        # 3. save rng states to mimic in backward.
        ctx.fwd_cpu_rng_state = torch.get_rng_state()
        ctx.fwd_cuda_rng_state = torch.cuda.get_rng_state()
        ctx.fwd_cuda_rng_state_tracker = ctx.rng_tracker.get_states()

        # 4. run forward without gradients
        with torch.no_grad():
            outputs = run_function(*inputs_cuda)
        del inputs_cuda

        # 5. save inputs of partition_checkpoint or cpu_checkpoint for backward stage.
        if ctx.partitioner.partitioned_checkpointing:
            new_args = ctx.partitioner.get_partitioned_activations_for_backward(
                args, inputs
            )
            assert len(new_args) % 2 == 0, (
                f"save_for_backward must call with even number of args. "
                f"but currently called with odd number of args, num_args={len(new_args)}"
            )
            save_args_for_backward(*new_args)
        elif ctx.partitioner.cpu_checkpointing:
            new_args = ctx.partitioner.get_cpu_activations_for_backward(args, inputs)
            save_args_for_backward(*new_args)
        else:
            save_args_for_backward(*args)

        # 6. make the non-fp tensors non-differential
        if torch.is_tensor(outputs):
            non_grad_outputs = [outputs] if not outputs.is_floating_point() else []
        else:
            non_grad_outputs = [
                o for o in outputs if torch.is_tensor(o) and not o.is_floating_point()
            ]
        ctx.mark_non_differentiable(*non_grad_outputs)

        # 7. return the results of forward stage.
        if torch.is_tensor(outputs):
            all_outputs += [outputs]
            return outputs
        else:
            all_outputs += outputs
            tensor_outputs, non_tensor_outputs, _ = extract_tensors(all_objects=outputs)
            return tuple(outputs)

    @staticmethod
    def backward(ctx, *grads):
        # 1. frees up all the pointers if user want contiguous checkpointing.
        if ctx.partitioner.contiguous_checkpointing:
            ctx.partitioner.contiguous_data_buffers = []
            ctx.partitioner.contiguous_size_buffers = []
            ctx.partitioner.data_offsets = []
            ctx.partitioner.size_offsets = []

        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad(), "
                "please use .backward() if possible"
            )

        # 2. gathering or moving to gpu if user want and detach all the tensors.
        if ctx.partitioner.partitioned_checkpointing:
            inputs = ctx.partitioner.gather_partitioned_activations(
                ctx._saved_tensors,
                device=torch.cuda.current_device()
                if ctx.partitioner.cpu_checkpointing
                else None,
            )
            detached_inputs = detach(inputs)
            ctx.tensor_flags = ctx.tensor_flags[0::2]
            ctx.non_tensor_args = ctx.non_tensor_args[0::2]
        elif ctx.partitioner.cpu_checkpointing:
            inputs = copy_to_device(
                ctx._saved_tensors,
                torch.cuda.current_device(),
                partial(
                    is_activation_to_checkpoint,
                    world_size=ctx.partitioner.mp_size,
                ),
            )
            detached_inputs = detach(inputs)
        else:
            inputs = ctx._saved_tensors
            detached_inputs = detach(inputs)

        # 3. merge_tensors if tensors are partitioned.
        detached_inputs = ctx.partitioner.merge_tensors(
            tensor_objects=detached_inputs,
            non_tensor_objects=ctx.non_tensor_args,
            tensor_flags=ctx.tensor_flags,
        )

        # 4. restore rng states to mimic in backward.
        bwd_cpu_rng_state = torch.get_rng_state()
        bwd_cuda_rng_state = torch.cuda.get_rng_state()
        bwd_cuda_rng_state_tracker = ctx.rng_tracker.get_states()

        # 5. set the states to what it used to be before the forward pass.
        torch.set_rng_state(ctx.fwd_cpu_rng_state)
        ctx.rng_tracker.set_cuda_rng_state(ctx.fwd_cuda_rng_state)
        ctx.rng_tracker.set_states(ctx.fwd_cuda_rng_state_tracker)

        # 6. run forward with gradients
        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # 7. set the states back to what it was at the start of this function.
        torch.set_rng_state(bwd_cpu_rng_state)
        ctx.rng_tracker.set_cuda_rng_state(bwd_cuda_rng_state)
        ctx.rng_tracker.set_states(bwd_cuda_rng_state_tracker)

        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        # 8. filter out non tensor outputs.
        outputs, _, _ = extract_tensors(all_objects=outputs)

        # 9. run backward stage.
        output_tensors = []
        grad_tensors = []
        for out, grad in zip(outputs, grads):
            if out.requires_grad:
                output_tensors.append(out)
                grad_tensors.append(grad)
        torch.autograd.backward(output_tensors, grad_tensors)

        # 10. force clear our stashed tensors to prevent a memory leak in certain scenarios.
        ctx._saved_tensors = None
        ctx.non_tensor_args = None
        ctx.tensor_flags = None

        # 11. return gradients
        ret_list = [None, None, None]
        for inp in detached_inputs:
            if torch.is_tensor(inp):
                ret_list.append(inp.grad)
            else:
                ret_list.append(None)

        return tuple(ret_list)
