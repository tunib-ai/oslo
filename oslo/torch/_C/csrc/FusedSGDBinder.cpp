#include <torch/extension.h>

void multi_tensor_sgd_cuda(int chunk_size, at::Tensor noop_flag,
                           std::vector<std::vector<at::Tensor>> tensor_lists,
                           float wd, float momentum, float dampening, float lr,
                           bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_sgd", &multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors");
}
