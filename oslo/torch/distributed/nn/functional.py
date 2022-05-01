from oslo.torch.distributed.nn.ring import _RingQK, _RingAV


def ring_qk(sub_q, sub_k, parallel_context):
    return _RingQK.apply(sub_q, sub_k, parallel_context)


def ring_av(sub_attn, sub_v, parallel_context):
    return _RingAV.apply(sub_attn, sub_v, parallel_context)
