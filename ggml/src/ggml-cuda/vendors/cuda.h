#ہدایت once

#شامل <cuda_runtime.h>
#شامل <cuda.h>
#شامل <cublas_v2.h>
#شامل <cuda_bf16.h>
#شامل <cuda_fp16.h>

#اگر_تعریف_شدہ GGML_USE_NCCL
#شامل <nccl.h>
#اگر_ختم // GGML_USE_NCCL

#اگر CUDART_نسخہ >= 11080
#شامل <cuda_fp8.h>
#تعریف_کرو FP8_دستیاب
#اگر_ختم // CUDART_VERSION >= 11080

#اگر CUDART_نسخہ >= 12080
#شامل <cuda_fp4.h>
#اگر_ختم // CUDART_VERSION >= 12080

#اگر CUDART_نسخہ < 11020
#تعریف_کرو CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظہ_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_پتہ_MANAGEMENT_SUPPORTED
#تعریف_کرو CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#تعریف_کرو CUBLAS_COMPUTE_16F CUDA_R_16F
#تعریف_کرو CUBLAS_COMPUTE_32F CUDA_R_32F
#تعریف_کرو cublasComputeType_t cuda_ڈیٹا_Type_t
#اگر_ختم // CUDART_VERSION < 11020
