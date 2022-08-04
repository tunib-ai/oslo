#include <torch/extension.h>

std::tuple<at::Tensor, at::Tensor>
multi_tensor_l2norm_cuda(int chunk_size, at::Tensor noop_flag,
                         std::vector<std::vector<at::Tensor>> tensor_lists,
                         at::optional<bool> per_tensor_python);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("multi_tensor_l2norm", &multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors");
}
