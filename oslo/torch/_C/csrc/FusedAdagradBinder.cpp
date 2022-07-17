#include <torch/extension.h>

void multi_tensor_adagrad_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, const float lr,
    const float epsilon, const int mode, const float weight_decay);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_adagrad", &multi_tensor_adagrad_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
}
