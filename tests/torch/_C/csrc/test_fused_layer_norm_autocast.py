# from oslo.torch._C import FusedLayerNormBinder
import torch
import unittest
import itertools

import oslo.torch.nn as onn


def _prep_layers(normalized_shape, elementwise_affine, dtype):
    native = torch.nn.LayerNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    ).to(device="cuda", dtype=dtype)
    fused = onn.FusedLayerNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    ).cuda()
    return native, fused


def _prep_rms_layers(normalized_shape, elementwise_affine, dtype):
    native = onn.FusedRMSNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    )
    fused = onn.FusedRMSNorm(
        normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
    ).cuda()
    return native, fused


def _prep_inputs(batch_size, normalized_shape, dtype):
    shape = (batch_size, *normalized_shape)
    fused = torch.randn(shape).cuda().requires_grad_(True)
    with torch.no_grad():
        native = fused.clone().to(dtype).requires_grad_(True)
    return native, fused


autocast_dtypes = (
    (torch.half, torch.bfloat16) if torch.cuda.is_bf16_supported() else (torch.half,)
)


class TestAutocastFusedLayerNorm(unittest.TestCase):
    bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def setUp(self):
        self.batch_size = 16
        self.normalized_shape = [32, 16]

    def _run_test(self, dtype, elementwise_affine):
        native, fused = _prep_layers(self.normalized_shape, elementwise_affine, dtype)
        native_x, fused_x = _prep_inputs(self.batch_size, self.normalized_shape, dtype)

        expected = native(native_x)
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = (
            {"rtol": None, "atol": None}
            if dtype == torch.half
            else TestAutocastFusedLayerNorm.bf16_fwd_thresholds
        )
        torch.testing.assert_allclose(actual, expected, **tols)

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.clone()
        expected.backward(g_native)
        actual.backward(g_fused)

        tols = (
            {"rtol": None, "atol": None}
            if dtype == torch.half
            else TestAutocastFusedLayerNorm.bf16_bwd_thresholds
        )
        torch.testing.assert_allclose(native_x.grad, fused_x.grad, **tols)

    def test_autocast(self):
        for (dtype, elementwise_affine) in itertools.product(
            autocast_dtypes, (True, False)
        ):
            with self.subTest(f"{dtype}-{elementwise_affine}"):
                self._run_test(dtype, elementwise_affine)


class TestAutocastFusedRMSNorm(unittest.TestCase):
    bf16_fwd_thresholds = dict(rtol=1.6e-2, atol=3e-4)
    bf16_bwd_thresholds = dict(rtol=1.6e-2, atol=3e-3)

    def setUp(self):
        self.batch_size = 16
        self.normalized_shape = [32, 16]

    def _run_test(self, dtype, elementwise_affine):
        native, fused = _prep_rms_layers(
            self.normalized_shape, elementwise_affine, dtype
        )
        native_x, fused_x = _prep_inputs(self.batch_size, self.normalized_shape, dtype)

        expected = native(native_x.cpu())
        with torch.cuda.amp.autocast(dtype=dtype):
            actual = fused(fused_x)
        tols = (
            {"rtol": None, "atol": None}
            if dtype == torch.half
            else TestAutocastFusedRMSNorm.bf16_fwd_thresholds
        )
        torch.testing.assert_allclose(actual, expected.detach().clone().cuda(), **tols)

        g_native = torch.rand_like(expected)
        with torch.no_grad():
            g_fused = g_native.detach().clone().cuda()
        expected.backward(g_native)
        actual.backward(g_fused)

        tols = (
            {"rtol": None, "atol": None}
            if dtype == torch.half
            else TestAutocastFusedRMSNorm.bf16_bwd_thresholds
        )
        torch.testing.assert_allclose(native_x.grad.cuda(), fused_x.grad, **tols)

    def test_autocast(self):
        for (dtype, elementwise_affine) in itertools.product(
            autocast_dtypes, (True, False)
        ):
            with self.subTest(f"{dtype}-{elementwise_affine}"):
                self._run_test(dtype, elementwise_affine)


if __name__ == "__main__":
    unittest.main(verbosity=2)
