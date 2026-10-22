#निर्देशः once

#अन्तर्भाव <cuda_runtime.h>
#अन्तर्भाव <cuda.h>
#अन्तर्भाव <cublas_v2.h>
#अन्तर्भाव <cuda_bf16.h>
#अन्तर्भाव <cuda_fp16.h>

#यदि_परिभाषितम् GGML_USE_NCCL
#अन्तर्भाव <nccl.h>
#यदि_अन्तः // GGML_USE_NCCL

#यदि CUDART_संस्करणम् >= 11080
#अन्तर्भाव <cuda_fp8.h>
#परिभाषय FP8_उपलब्धम्
#यदि_अन्तः // CUDART_VERSION >= 11080

#यदि CUDART_संस्करणम् >= 12080
#अन्तर्भाव <cuda_fp4.h>
#यदि_अन्तः // CUDART_VERSION >= 12080

#यदि CUDART_संस्करणम् < 11020
#परिभाषय CU_DEVICE_ATTRIBUTE_VIRTUAL_स्मृतिः_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_स्थानाङ्कः_MANAGEMENT_SUPPORTED
#परिभाषय CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#परिभाषय CUBLAS_COMPUTE_16F CUDA_R_16F
#परिभाषय CUBLAS_COMPUTE_32F CUDA_R_32F
#परिभाषय cublasComputeType_t cuda_दत्त_Type_t
#यदि_अन्तः // CUDART_VERSION < 11020
