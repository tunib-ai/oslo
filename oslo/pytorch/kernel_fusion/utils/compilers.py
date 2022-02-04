import torch

from oslo.pytorch.kernel_fusion.utils.decompositions import decomposition_table


def ts_compile(fx_g, _):
    for node in fx_g.graph.nodes:
        if node.target == torch.ops.aten.new_zeros:
            if node.args[1] == []:
                args = list(node.args)
                args[1] = [1]
                node.args = tuple(args)

    for node in fx_g.graph.nodes:
        new_kwargs = {}
        for k, v in node.kwargs.items():
            if isinstance(v, torch.device):
                v = v.type
            new_kwargs[k] = v
        node.kwargs = new_kwargs

    fx_g.graph.lint()

    for i in range(1000):
        attr = f"_tensor_constant{i}"
        if hasattr(fx_g, attr):
            setattr(fx_g, attr, getattr(fx_g, attr).cuda())
        else:
            break

    fx_g.recompile()
    f = torch.jit.script(fx_g)
    torch._C._jit_pass_remove_mutation(f.graph)
    f = torch.jit.freeze(f.eval())
    f = torch.jit.optimize_for_inference(f)
    return f


aten = torch.ops.aten
default_decompositions = set(
    [
        aten.detach,
        aten.gelu_backward,
        aten._log_softmax_backward_data,
        aten.leaky_relu_backward,
        aten.sigmoid_backward,
        aten.threshold_backward,
        aten.hardtanh_backward,
        aten.hardsigmoid_backward,
        aten.hardswish_backward,
        aten.tanh_backward,
        aten.silu_backward,
    ]
)
default_decompositions = {
    k: v for k, v in decomposition_table.items() if k in default_decompositions
}
