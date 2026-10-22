#pragma once

#அந்தர்பாவ <cuda_runtime.h>
#அந்தர்பாவ <cuda.h>
#அந்தர்பாவ <cublas_v2.h>
#அந்தர்பாவ <cuda_bf16.h>
#அந்தர்பாவ <cuda_fp16.h>

#ifdef GGML_USE_NCCL
#அந்தர்பாவ <nccl.h>
#endif // GGML_USE_NCCL

#எனில் CUDART_VERSION >= 11080
#அந்தர்பாவ <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#எனில் CUDART_VERSION >= 12080
#அந்தர்பாவ <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#எனில் CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION < 11020
