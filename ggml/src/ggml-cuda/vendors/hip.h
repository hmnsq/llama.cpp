#pragma once

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ضمن <hip/hip_runtime.h>
#ضمن <hipblas/hipblas.h>
#ضمن <hip/hip_fp16.h>
#ضمن <hip/hip_bf16.h>

#ifdef GGML_USE_NCCL
#ضمن <rccl/rccl.h>
#endif // GGML_USE_NCCL


#define CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N HIPBLAS_OP_N
#define CUBLAS_OP_T HIPBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUDA_R_16F  HIPBLAS_R_16F
#define CUDA_R_16BF HIPBLAS_R_16B
#define CUDA_R_32F  HIPBLAS_R_32F
#define CUBLAS_SIDE_ئوڭ HIPBLAS_SIDE_ئوڭ
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_CHECK(fn) {hipError_t err = fn; إذا(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#define __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#define __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#define __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#define __all_sync(mask, var) __all(var)
#define __any_sync(mask, var) __any(var)
#define cublasStrsmBatched hipblasStrsmBatched
#define cublasCreate hipblasCreate
#define cublasDestroy hipblasDestroy
#define cublasGemmEx hipblasGemmEx
#define cublasGemmBatchedEx hipblasGemmBatchedEx
#define cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#define cublasHandle_t hipblasHandle_t
#define cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#define cublasSetStream hipblasSetStream
#define cublasSgemm hipblasSgemm
#define cublasSgemmBatched hipblasSgemmBatched
#define cublasSgemmStridedBatched hipblasSgemmStridedBatched
#define cublasStatus_t hipblasStatus_t
#define cublasOperation_t hipblasOperation_t
#define cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaError_t hipError_t
#define cudaErrorMemoryAllocation hipErrorOutOfMemory
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaEventDestroy hipEventDestroy
#define cudaFree hipFree
#define cudaFreeHost hipHostFree
#define cudaGetDevice hipGetDevice
#define cuda_Get_Device_سانى hip_Get_Device_سانى
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaHostAlloc hipHostMalloc
#define cudaHostAllocPortable hipHostMallocPortable
#define cudaHostAllocMapped hipHostMallocMapped
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister hipHostRegister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaHostUnregister hipHostUnregister
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc hipMalloc
#define cudaMallocHost(ptr, size) hipHostMalloc(ptr, size, hipHostMallocDefault)
#define cudaMallocManaged hipMallocManaged
#define cudaMemAdvise hipMemAdvise
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize
#define cudaSetDevice hipSetDevice
#define cuDeviceGet hipDeviceGet
#define CUdevice hipDevice_t
#define CUdeviceptr hipDeviceptr_t
#define cuMemUnmap hipMemUnmap
#define CUmemAccessDesc hipMemAccessDesc
#define cuMemAddressFree hipMemAddressFree
#define cuMemRelease hipMemRelease
#define CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#define cuMemCreate hipMemCreate
#define cuMemAddressReserve hipMemAddressReserve
#define cuMemMap hipMemMap
#define cuMemSetAccess hipMemSetAccess
#define cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#define CUmemAllocationProp hipMemAllocationProp
#define cuDeviceGetAttribute hipDeviceGetAttribute
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamFireAndForget hipStreamFireAndForget
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamPerThread hipStreamPerThread
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaGraphExec_t hipGraphExec_t
#define cuda_Graph_تۈگۈن_t hip_Graph_تۈگۈن_t
#define cuda_Kernel_تۈگۈن_Params hip_Kernel_تۈگۈن_Params
#define cuda_Kernel_تۈگۈن_Params hip_Kernel_تۈگۈن_Params
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cudaGraphExecUpdateResult hipGraphExecUpdateResult
#define cuda_Graph_تۈگۈن_Type hip_Graph_تۈگۈن_Type
#define cuda_Graph_تۈگۈن_Type_Kernel hip_Graph_تۈگۈن_Type_Kernel
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cuda_Graph_Kernel_تۈگۈن_Set_Params hip_Graph_Kernel_تۈگۈن_Set_Params
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cuda_Graph_Kernel_تۈگۈن_Get_Params hip_Graph_Kernel_تۈگۈن_Get_Params
#define cuda_Graph_تۈگۈن_Get_Type hip_Graph_تۈگۈن_Get_Type
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaGraph_t hipGraph_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define __trap() افعل { abort(); __builtin_unreachable(); } طالما(0)
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_قىممەت HIPBLAS_STATUS_INVALID_قىممەت
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#إذا HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t
#define cudaDataType_t hipDataType
#وإلا
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t
#define cudaDataType_t hipblasDatatype_t
#endif // HIP_VERSION >= 6050000

#إذا !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)

#define __CUDA_ARCH__ 1300

#إذا defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#define GCN5
#endif // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#إذا defined(__gfx803__)
#define GCN4
#endif // defined(__gfx803__)

#إذا defined(GCN5) || defined(GCN4)
#define GCN
#endif // defined(GCN5) || defined(GCN4)

#إذا defined(__gfx950__)
#define CDNA4
#endif // defined(__gfx950__)

#إذا defined(__gfx942__)
#define CDNA3
#endif // defined(__gfx942__)

#إذا defined(__gfx90a__)
#define CDNA2
#endif // defined(__gfx90a__)

#إذا defined(__gfx908__)
#define CDNA1
#endif // defined(__gfx908__)

#إذا defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#define CDNA // For the entire family
#endif // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#إذا defined(__GFX12__)
#define RDNA4
#endif // defined(__GFX12__)

#إذا defined(__GFX11__)
#define RDNA3
#endif // defined(__GFX11__)

#إذا defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define RDNA3_5
#endif // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#إذا defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif // defined(RDNA3) && !defined(RDNA3_5)

#إذا defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#إذا defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif // defined(__gfx1010__) || defined(__gfx1012__)

#إذا defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#define RDNA // For the entire family
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

تعريفنوع __hip_bfloat16 nv_bfloat16;
تعريفنوع __hip_bfloat162 nv_bfloat162;

#إذا HIP_VERSION >= 60300000
#ضمن <hip/hip_fp8.h>
تعريفنوع __hip_fp8_e4m3 __nv_fp8_e4m3;
#define FP8_AVAILABLE
#endif // HIP_VERSION >= 60300000

تعريفنوع int8_t int8x4_t __attribute__((ext_vector_type(4)));
تعريفنوع uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
ساكن __device__ __forceinline__ صحيح __vsubss4(ثابت صحيح a, ثابت صحيح b) {
    ثابت int8x4_t va = reinterpret_cast<ثابت int8x4_t&>(a);
    ثابت int8x4_t vb = reinterpret_cast<ثابت int8x4_t&>(b);
#إذا __has_builtin(__builtin_elementwise_sub_sat)
    ثابت int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    أرجع reinterpret_cast<ثابت صحيح &>(c);
#وإلا
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    لكل (صحيح i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        إذا(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        إذا(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    أرجع reinterpret_cast<صحيح &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

ساكن __device__ __forceinline__ صحيح __vsub4(ثابت صحيح a, ثابت صحيح b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ثابت unsigned صحيح a_large = a | 0x80808080;
    ثابت unsigned صحيح b_small = b & 0x7f7f7f7f;
    ثابت unsigned صحيح result_تۆۋەن_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ثابت unsigned صحيح should_flip_يۇقىرى_1bit = (a ^ ~b) & 0x80808080;

    أرجع result_تۆۋەن_7bits ^ should_flip_يۇقىرى_1bit;
}

ساكن __device__ __forceinline__ unsigned صحيح __vcmpeq4(unsigned صحيح a, unsigned صحيح b) {
    ثابت uint8x4_t& va = reinterpret_cast<ثابت uint8x4_t&>(a);
    ثابت uint8x4_t& vb = reinterpret_cast<ثابت uint8x4_t&>(b);
    unsigned صحيح c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    لكل (صحيح i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    أرجع c;
}

ساكن __device__ __forceinline__ unsigned صحيح __vcmpne4(unsigned صحيح a, unsigned صحيح b) {
    ثابت unsigned صحيح x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ثابت unsigned صحيح ne_تۆۋەن_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ثابت unsigned صحيح ne_يۇقىرى_1bit = x & 0x80808080;
    ثابت unsigned صحيح ne_any_bit = ne_تۆۋەن_7bits | ne_يۇقىرى_1bit;

    أرجع (ne_any_bit >> 7) * 0xff;
}
