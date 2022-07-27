#include "custom_cuda_layers.h"
#include <stdio.h>


using namespace std;

// to do list


// 0. host and device split for torch _C binding
// 0.0 CUDA side error casting
// 0.1 CUDA 
// 1. Single GPU basd grid,thread,WARP setting
// 2. Selected Single GPU based,grid,thread,WARP setting
// 3. Multi GPU based grid,thread,WARP setting
// 4. selected Multi GPU based grid,thread,WARP setting


// 0727 kernel ma
__global__ std::vector<int> kernel_query(int deviceid)
{
  cudaGetDevice(&deviceId)
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
  std::vector<int> current_cuda_props;
  current_cuda_props.pushback(props.major);
  current_cuda_props.pushback(props.minor);
  current_cuda_props.pushback(props.multiProcessorCount);
  current_cuda_props.pushback.(props.warpSize);
  current_cuda_props.pushback.(props.maxThreadsPerBlock);
  current_cuda_props.pushback.(props.maxThreadsDim[0]);
  current_cuda_props.pushback.(props.maxThreadsDim[1]);
  current_cuda_props.pushback.(props.maxThreadsDim[2]);
  current_cuda_props.pushback.(props.maxGridSize[0]);
  current_cuda_props.pushback.(props.maxGridSize[1]);
  current_cuda_props.pushback.(props.maxGridSize[2]);

  // printf("Device ID: %d\nNumber of SMs: 
  // %d\nCompute Capability Major: %d\n
  // Compute Capability Minor: %d\n
  // Warp Size: %d\n"
  // , deviceId, multiProcessorCount, computeCapabilityMajor, computeCapabilityMinor, warpSize);


  return current_cuda_props;
}


__global__ void param_update_kernel(const float *input, __half *output,
                                    int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < size) {
    output[id] = (__half)input[id];
  }
}

void launch_param_update(const float *input, __half *output, int size,
                         cudaStream_t stream) {
  int threads = 1024;

  dim3 grid_dim((size - 1) / threads + 1);
  dim3 block_dim(threads);

  param_update_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, size);
}

__global__ void param_update_kernel_half(const float *input, __half *output,
                                         int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  __half2 *output_cast = reinterpret_cast<__half2 *>(output);
  if (id < size) {
    float input_f = input[id];
    __half2 *input_h = reinterpret_cast<__half2 *>(&input_f);
    output_cast[id] = *input_h;
  }
}

void launch_param_update_half(const float *input, __half *output, int size,
                              cudaStream_t stream) {
  int threads = 1024;
  size /= 2;
  dim3 grid_dim((size - 1) / threads + 1);
  dim3 block_dim(threads);

  param_update_kernel_half<<<grid_dim, block_dim, 0, stream>>>(input, output,
                                                               size);
}
