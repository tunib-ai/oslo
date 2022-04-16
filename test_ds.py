from oslo.torch._C import FusedLayerNormBinder

binder = FusedLayerNormBinder().bind()
print(binder.scaled_masked_softmax_forward)
