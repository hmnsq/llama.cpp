#pragma once

#અન્તર્ભાવ <cuda_runtime.h>
#અન્તર્ભાવ <cuda.h>
#અન્તર્ભાવ <cublas_v2.h>
#અન્તર્ભાવ <cuda_bf16.h>
#અન્તર્ભાવ <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#અન્તર્ભાવ <nccl.h>
#endif // GGML_USE_NCCL

#જો CUDART_VERSION >= 11080
#અન્તર્ભાવ <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#જો CUDART_VERSION >= 12080
#અન્તર્ભાવ <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#જો CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
