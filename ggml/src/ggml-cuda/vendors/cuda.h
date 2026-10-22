#pragma once

#ಅನ್ತರ್ಭಾವ <cuda_runtime.h>
#ಅನ್ತರ್ಭಾವ <cuda.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cuda_bf16.h>
#ಅನ್ತರ್ಭಾವ <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#ಅನ್ತರ್ಭಾವ <nccl.h>
#endif // GGML_USE_NCCL

#ಯದಿ CUDART_VERSION >= 11080
#ಅನ್ತರ್ಭಾವ <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#ಯದಿ CUDART_VERSION >= 12080
#ಅನ್ತರ್ಭಾವ <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#ಯದಿ CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
