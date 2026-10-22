#pragma once

#شامل <cuda_runtime.h>
#شامل <cuda.h>
#شامل <cublas_v2.h>
#شامل <cuda_bf16.h>
#شامل <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#شامل <nccl.h>
#endif // GGML_USE_NCCL

#اگر CUDART_VERSION >= 11080
#شامل <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#اگر CUDART_VERSION >= 12080
#شامل <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#اگر CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
