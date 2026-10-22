#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_runtime.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cublas_v2.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_bf16.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_fp16.h>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_NCCL
#ਅਨ੍ਤਰ੍ਭਾਵ <nccl.h>
#ਜੇ_ਅੰਤ // GGML_USE_NCCL

#ਜੇ CUDART_ਸੰਸਕਰਣ >= 11080
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_fp8.h>
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FP8_ਉਪਲਬਧ
#ਜੇ_ਅੰਤ // CUDART_VERSION >= 11080

#ਜੇ CUDART_ਸੰਸਕਰਣ >= 12080
#ਅਨ੍ਤਰ੍ਭਾਵ <cuda_fp4.h>
#ਜੇ_ਅੰਤ // CUDART_VERSION >= 12080

#ਜੇ CUDART_ਸੰਸਕਰਣ < 11020
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_DEVICE_ATTRIBUTE_VIRTUAL_ਸਮ੍ਰਿਤੀ_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ਪਤਾ_MANAGEMENT_SUPPORTED
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_16F CUDA_R_16F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_32F CUDA_R_32F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasComputeType_t cuda_ਅੰਕੜੇ_Type_t
#ਜੇ_ਅੰਤ // CUDART_VERSION < 11020
