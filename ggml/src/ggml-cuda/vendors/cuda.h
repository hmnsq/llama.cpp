#సూచన once

#అన్తర్భావ <cuda_runtime.h>
#అన్తర్భావ <cuda.h>
#అన్తర్భావ <cublas_v2.h>
#అన్తర్భావ <cuda_bf16.h>
#అన్తర్భావ <cuda_fp16.h>

#నిర్వచితమైతే GGML_USE_NCCL
#అన్తర్భావ <nccl.h>
#అయితే_ముగింపు // GGML_USE_NCCL

#ఒకవేళ CUDART_సంస్కరణ >= 11080
#అన్తర్భావ <cuda_fp8.h>
#నిర్వచించు FP8_అందుబాటు
#అయితే_ముగింపు // CUDART_VERSION >= 11080

#ఒకవేళ CUDART_సంస్కరణ >= 12080
#అన్తర్భావ <cuda_fp4.h>
#అయితే_ముగింపు // CUDART_VERSION >= 12080

#ఒకవేళ CUDART_సంస్కరణ < 11020
#నిర్వచించు CU_DEVICE_ATTRIBUTE_VIRTUAL_జ్ఞాపకం_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_చిరునామా_MANAGEMENT_SUPPORTED
#నిర్వచించు CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#నిర్వచించు CUBLAS_COMPUTE_16F CUDA_R_16F
#నిర్వచించు CUBLAS_COMPUTE_32F CUDA_R_32F
#నిర్వచించు cublasComputeType_t cuda_దత్తాంశం_Type_t
#అయితే_ముగింపు // CUDART_VERSION < 11020
