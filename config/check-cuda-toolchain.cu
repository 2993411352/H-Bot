// Compile-only check: no GPU work is launched.
#include <cuda_runtime.h>
__global__ void hbot_toolchain_probe(float *value) { value[0] = 1.0f; }
