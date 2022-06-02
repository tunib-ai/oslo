#pragma once

#include "cuda_fp16.h"

void launch_param_update(const float *input, __half *output, int size,
                         cudaStream_t stream);
void launch_param_update_half(const float *input, __half *output, int size,
                              cudaStream_t stream);
