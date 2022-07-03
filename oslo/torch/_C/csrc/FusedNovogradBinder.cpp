#include <torch/extension.h>

void multi_tensor_novograd_cuda(
    int chunk_size, at::Tensor noop_flag,
    std::vector<std::vector<at::Tensor>> tensor_lists, at::Tensor grad_norms,
    const float lr, const float beta1, const float beta2, const float epsilon,
    const int step, const int bias_correction, const float weight_decay,
    const int grad_averaging, const int mode, const int norm_type);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_novograd", &multi_tensor_novograd_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer");
}
