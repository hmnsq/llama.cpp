#ہدایت once

#ضمن <cuda_runtime.h>
#ضمن <cuda.h>
#ضمن <cublas_v2.h>
#ضمن <cuda_bf16.h>
#ضمن <cuda_fp16.h>

#جے_تعریف_شدہ GGML_USE_NCCL
#ضمن <nccl.h>
#جے_انت // GGML_USE_NCCL

#إذا CUDART_نسخہ >= 11080
#ضمن <cuda_fp8.h>
#تعریف_کرو FP8_دستیاب
#جے_انت // CUDART_VERSION >= 11080

#إذا CUDART_نسخہ >= 12080
#ضمن <cuda_fp4.h>
#جے_انت // CUDART_VERSION >= 12080

#إذا CUDART_نسخہ < 11020
#تعریف_کرو CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظہ_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_پتہ_MANAGEMENT_SUPPORTED
#تعریف_کرو CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#تعریف_کرو CUBLAS_COMPUTE_16F CUDA_R_16F
#تعریف_کرو CUBLAS_COMPUTE_32F CUDA_R_32F
#تعریف_کرو cublasComputeType_t cuda_انکڑے_Type_t
#جے_انت // CUDART_VERSION < 11020
