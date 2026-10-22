#لارښوونه once

#ضمن <cuda_runtime.h>
#ضمن <cuda.h>
#ضمن <cublas_v2.h>
#ضمن <cuda_bf16.h>
#ضمن <cuda_fp16.h>

#که_تعریف_شوی GGML_USE_NCCL
#ضمن <nccl.h>
#د_که_پای // GGML_USE_NCCL

#إذا CUDART_نسخه >= 11080
#ضمن <cuda_fp8.h>
#تعریف_کړه FP8_شته
#د_که_پای // CUDART_VERSION >= 11080

#إذا CUDART_نسخه >= 12080
#ضمن <cuda_fp4.h>
#د_که_پای // CUDART_VERSION >= 12080

#إذا CUDART_نسخه < 11020
#تعریف_کړه CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظه_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_پته_MANAGEMENT_SUPPORTED
#تعریف_کړه CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#تعریف_کړه CUBLAS_COMPUTE_16F CUDA_R_16F
#تعریف_کړه CUBLAS_COMPUTE_32F CUDA_R_32F
#تعریف_کړه cublasComputeType_t cuda_معلومات_Type_t
#د_که_پای // CUDART_VERSION < 11020
