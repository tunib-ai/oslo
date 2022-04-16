from oslo.torch.distributed.nn.ring import _RingQK, _RingAV


def ring_qk(sub_q, sub_k, gpc):
    return _RingQK.apply(sub_q, sub_k, gpc)


def ring_av(sub_attn, sub_v, gpc):
    return _RingAV.apply(sub_attn, sub_v, gpc)
