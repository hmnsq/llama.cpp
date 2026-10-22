#pragma once

#অন্তর্ভাব <cuda_runtime.h>
#অন্তর্ভাব <cuda.h>
#অন্তর্ভাব <cublas_v2.h>
#অন্তর্ভাব <cuda_bf16.h>
#অন্তর্ভাব <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#অন্তর্ভাব <nccl.h>
#endif // GGML_USE_NCCL

#যদি CUDART_VERSION >= 11080
#অন্তর্ভাব <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#যদি CUDART_VERSION >= 12080
#অন্তর্ভাব <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#যদি CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cuda_তথ্য_Type_t
#endif // CUDART_VERSION < 11020
