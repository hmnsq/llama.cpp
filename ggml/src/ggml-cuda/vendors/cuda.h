#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <cuda_runtime.h>
#ಅನ್ತರ್ಭಾವ <cuda.h>
#ಅನ್ತರ್ಭಾವ <cublas_v2.h>
#ಅನ್ತರ್ಭಾವ <cuda_bf16.h>
#ಅನ್ತರ್ಭಾವ <cuda_fp16.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_NCCL
#ಅನ್ತರ್ಭಾವ <nccl.h>
#ಆದರೆ_ಅಂತ್ಯ // GGML_USE_NCCL

#ಆದರೆ CUDART_ಆವೃತ್ತಿ >= 11080
#ಅನ್ತರ್ಭಾವ <cuda_fp8.h>
#ವ್ಯಾಖ್ಯಾನಿಸು FP8_ಲಭ್ಯ
#ಆದರೆ_ಅಂತ್ಯ // CUDART_VERSION >= 11080

#ಆದರೆ CUDART_ಆವೃತ್ತಿ >= 12080
#ಅನ್ತರ್ಭಾವ <cuda_fp4.h>
#ಆದರೆ_ಅಂತ್ಯ // CUDART_VERSION >= 12080

#ಆದರೆ CUDART_ಆವೃತ್ತಿ < 11020
#ವ್ಯಾಖ್ಯಾನಿಸು CU_DEVICE_ATTRIBUTE_VIRTUAL_ಸ್ಮರಣೆ_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ವಿಳಾಸ_MANAGEMENT_SUPPORTED
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_16F CUDA_R_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F CUDA_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು cublasComputeType_t cuda_ದತ್ತಾಂಶ_Type_t
#ಆದರೆ_ಅಂತ್ಯ // CUDART_VERSION < 11020
