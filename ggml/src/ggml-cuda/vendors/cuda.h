#pragma once

#ଅନ୍ତର୍ଭାଵ <cuda_runtime.h>
#ଅନ୍ତର୍ଭାଵ <cuda.h>
#ଅନ୍ତର୍ଭାଵ <cublas_v2.h>
#ଅନ୍ତର୍ଭାଵ <cuda_bf16.h>
#ଅନ୍ତର୍ଭାଵ <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#ଅନ୍ତର୍ଭାଵ <nccl.h>
#endif // GGML_USE_NCCL

#ଯଦି CUDART_VERSION >= 11080
#ଅନ୍ତର୍ଭାଵ <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#ଯଦି CUDART_VERSION >= 12080
#ଅନ୍ତର୍ଭାଵ <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#ଯଦି CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
