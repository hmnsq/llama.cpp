#आशय once

#समावेश <cuda_runtime.h>
#समावेश <cuda.h>
#समावेश <cublas_v2.h>
#समावेश <cuda_bf16.h>
#समावेश <cuda_fp16.h>

#अगर_घोषित GGML_USE_NCCL
#समावेश <nccl.h>
#पूर्ण_अगर // GGML_USE_NCCL

#अगर CUDART_VERSION >= 11080
#समावेश <cuda_fp8.h>
#घोषणा FP8_AVAILABLE
#पूर्ण_अगर // CUDART_VERSION >= 11080

#अगर CUDART_VERSION >= 12080
#समावेश <cuda_fp4.h>
#पूर्ण_अगर // CUDART_VERSION >= 12080

#अगर CUDART_VERSION < 11020
#घोषणा CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#घोषणा CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#घोषणा CUBLAS_COMPUTE_16F CUDA_R_16F
#घोषणा CUBLAS_COMPUTE_32F CUDA_R_32F
#घोषणा cublasComputeType_t cuda_तथ्य_Type_t
#पूर्ण_अगर // CUDART_VERSION < 11020
