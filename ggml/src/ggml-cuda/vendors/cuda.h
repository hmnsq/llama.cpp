#pragma once

#অন্তর্ভাৱ <cuda_runtime.h>
#অন্তর্ভাৱ <cuda.h>
#অন্তর্ভাৱ <cublas_v2.h>
#অন্তর্ভাৱ <cuda_bf16.h>
#অন্তর্ভাৱ <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#অন্তর্ভাৱ <nccl.h>
#endif // GGML_USE_NCCL

#যদি CUDART_VERSION >= 11080
#অন্তর্ভাৱ <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#যদি CUDART_VERSION >= 12080
#অন্তর্ভাৱ <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#যদি CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
