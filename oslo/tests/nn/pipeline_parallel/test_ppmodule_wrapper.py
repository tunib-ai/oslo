import unittest

from oslo.torch.nn.parallel.distributed.pipeline_parallel.p2p import (
    PPModuleWrapper, wrap_nn_modules, check_wrap_nn_modules
)

from torch import nn


class CustomModule(nn.Module):
    def __init__(self, ndim=8):
        super().__init__()
        self.rank = 0
        self.rank_parent = 0

        ### module 1
        self.module1 = nn.Linear(ndim, ndim)

        setattr(self.module1, "rank", 0)
        setattr(self.module1, "rank_parent", 0)

        ### module 2
        self.module2 = nn.Sequential(
            nn.Linear(ndim, ndim), # rank 0
            nn.Sequential(
                nn.Linear(ndim, ndim), # rank 1
                nn.Linear(ndim, ndim), # rank 0
            ),
            nn.Linear(ndim, ndim), # rank 1
        )
        setattr(self.module2, "rank", 0)
        setattr(self.module2, "rank_parent", 0)

        setattr(self.module2[0], "rank", 0)
        setattr(self.module2[0], "rank_parent", 0)

        setattr(self.module2[1], "rank", 0)
        setattr(self.module2[1], "rank_parent", 0)

        setattr(self.module2[1][0], "rank", 1)
        setattr(self.module2[1][0], "rank_parent", 0)

        setattr(self.module2[1][1], "rank", 0)
        setattr(self.module2[1][1], "rank_parent", 0)

        setattr(self.module2[2], "rank", 1)
        setattr(self.module2[2], "rank_parent", 0)


    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        return x


class TestPPModuleWrapper(unittest.TestCase):
    def setUp(self):
        self.model_original = CustomModule()
        model_wrapped = CustomModule()
        wrap_nn_modules(model_wrapped)
        self.model_wrapped = model_wrapped

    def test_check_wrap_nn_modules_negt_test(self):
        with self.assertRaises(AssertionError):
            check_wrap_nn_modules(self.model_original)

    def test_check_wrap_nn_modules_pos_test(self):
        self.assertTrue(check_wrap_nn_modules(self.model_wrapped))


if __name__ == '__main__':
    unittest.main()
