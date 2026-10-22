#هدايت once

#ضمن <cuda_runtime.h>
#ضمن <cuda.h>
#ضمن <cublas_v2.h>
#ضمن <cuda_bf16.h>
#ضمن <cuda_fp16.h>

#جيڪڏهن_وصف_ٿيل GGML_USE_NCCL
#ضمن <nccl.h>
#جيڪڏهن_پورو // GGML_USE_NCCL

#إذا CUDART_نسخو >= 11080
#ضمن <cuda_fp8.h>
#وصف_ڪريو FP8_موجود
#جيڪڏهن_پورو // CUDART_VERSION >= 11080

#إذا CUDART_نسخو >= 12080
#ضمن <cuda_fp4.h>
#جيڪڏهن_پورو // CUDART_VERSION >= 12080

#إذا CUDART_نسخو < 11020
#وصف_ڪريو CU_DEVICE_ATTRIBUTE_VIRTUAL_يادگيري_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_پتو_MANAGEMENT_SUPPORTED
#وصف_ڪريو CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#وصف_ڪريو CUBLAS_COMPUTE_16F CUDA_R_16F
#وصف_ڪريو CUBLAS_COMPUTE_32F CUDA_R_32F
#وصف_ڪريو cublasComputeType_t cuda_معلومات_Type_t
#جيڪڏهن_پورو // CUDART_VERSION < 11020
