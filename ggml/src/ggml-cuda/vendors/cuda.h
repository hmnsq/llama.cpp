#رهنمود once

#ضمن <cuda_runtime.h>
#ضمن <cuda.h>
#ضمن <cublas_v2.h>
#ضمن <cuda_bf16.h>
#ضمن <cuda_fp16.h>

#اگر_تعریف_شده GGML_USE_NCCL
#ضمن <nccl.h>
#پایان_اگر // GGML_USE_NCCL

#إذا CUDART_نسخه >= 11080
#ضمن <cuda_fp8.h>
#تعریف_کن FP8_موجود
#پایان_اگر // CUDART_VERSION >= 11080

#إذا CUDART_نسخه >= 12080
#ضمن <cuda_fp4.h>
#پایان_اگر // CUDART_VERSION >= 12080

#إذا CUDART_نسخه < 11020
#تعریف_کن CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظه_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_آدرس_MANAGEMENT_SUPPORTED
#تعریف_کن CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#تعریف_کن CUBLAS_COMPUTE_16F CUDA_R_16F
#تعریف_کن CUBLAS_COMPUTE_32F CUDA_R_32F
#تعریف_کن cublasComputeType_t cuda_داده_Type_t
#پایان_اگر // CUDART_VERSION < 11020
