#pragma once

#define HIP_DISABLE_WARP_SYNC_BUILTINS 1
#অন্তর্ভাব <hip/hip_runtime.h>
#অন্তর্ভাব <hipblas/hipblas.h>
#অন্তর্ভাব <hip/hip_fp16.h>
#অন্তর্ভাব <hip/hip_bf16.h>

#ifdef GGML_USE_NCCL
#অন্তর্ভাব <rccl/rccl.h>
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
#define CUBLAS_SIDE_ডান HIPBLAS_SIDE_ডান
#define CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#define CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended
#define CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#define CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#define CU_CHECK(fn) {hipError_t err = fn; যদি(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
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
#define cuda_Get_Device_গুনতি hip_Get_Device_গুনতি
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
#define cudaMallocHost(ptr, আকার_2) hipHostMalloc(ptr, আকার_2, hipHostMallocDefault)
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
#define cuda_Occupancy_Max_Potential_Block_আকার hip_Occupancy_Max_Potential_Block_আকার
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
#define cuda_Graph_গ্রন্থি_t hip_Graph_গ্রন্থি_t
#define cuda_Kernel_গ্রন্থি_Params hip_Kernel_গ্রন্থি_Params
#define cuda_Kernel_গ্রন্থি_Params hip_Kernel_গ্রন্থি_Params
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphLaunch hipGraphLaunch
#define cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#define cuda_Graph_Exec_Update_ফলাফল hip_Graph_Exec_Update_ফলাফল
#define cuda_Graph_গ্রন্থি_Type hip_Graph_গ্রন্থি_Type
#define cuda_Graph_গ্রন্থি_Type_Kernel hip_Graph_গ্রন্থি_Type_Kernel
#define cudaGraphInstantiate hipGraphInstantiate
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaGraphDestroy hipGraphDestroy
#define cuda_Graph_Kernel_গ্রন্থি_Set_Params hip_Graph_Kernel_গ্রন্থি_Set_Params
#define cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#define cuda_Graph_Kernel_গ্রন্থি_Get_Params hip_Graph_Kernel_গ্রন্থি_Get_Params
#define cuda_Graph_গ্রন্থি_Get_Type hip_Graph_গ্রন্থি_Get_Type
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphExecUpdate hipGraphExecUpdate
#define cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaGraph_t hipGraph_t
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess
#define cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#define cudaFuncSetAttribute hipFuncSetAttribute
#define cuda_Func_Attribute_Max_Dynamic_Shared_Memory_আকার hip_Func_Attribute_Max_Dynamic_Shared_Memory_আকার
#define __trap() করো { abort(); __builtin_unreachable(); } যখন(0)
#define CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#define CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#define CUBLAS_STATUS_INVALID_মান HIPBLAS_STATUS_INVALID_মান
#define CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#define CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#define CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#define CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#define CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#যদি HIP_VERSION >= 60500000
#define CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#define cublasComputeType_t hipblasComputeType_t
#define cuda_তথ্য_Type_t hip_তথ্য_Type
#অন্যথা
#define CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#define CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#define cublasComputeType_t hipblasDatatype_t
#define cuda_তথ্য_Type_t hipblasDatatype_t
#endif // HIP_VERSION >= 6050000

#যদি !defined(__HIP_PLATFORM_AMD__)
#error "The HIP backend supports only AMD targets"
#endif // !defined(__HIP_PLATFORM_AMD__)

#define __CUDA_ARCH__ 1300

#যদি defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#define GCN5
#endif // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#যদি defined(__gfx803__)
#define GCN4
#endif // defined(__gfx803__)

#যদি defined(GCN5) || defined(GCN4)
#define GCN
#endif // defined(GCN5) || defined(GCN4)

#যদি defined(__gfx950__)
#define CDNA4
#endif // defined(__gfx950__)

#যদি defined(__gfx942__)
#define CDNA3
#endif // defined(__gfx942__)

#যদি defined(__gfx90a__)
#define CDNA2
#endif // defined(__gfx90a__)

#যদি defined(__gfx908__)
#define CDNA1
#endif // defined(__gfx908__)

#যদি defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#define CDNA // For the entire family
#endif // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#যদি defined(__GFX12__)
#define RDNA4
#endif // defined(__GFX12__)

#যদি defined(__GFX11__)
#define RDNA3
#endif // defined(__GFX11__)

#যদি defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#define RDNA3_5
#endif // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#যদি defined(RDNA3) && !defined(RDNA3_5)
#define RDNA3_0
#endif // defined(RDNA3) && !defined(RDNA3_5)

#যদি defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#define RDNA2
#endif

#যদি defined(__gfx1010__) || defined(__gfx1012__)
#define RDNA1
#endif // defined(__gfx1010__) || defined(__gfx1012__)

#যদি defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#define RDNA // For the entire family
#endif // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

প্রকারলেখ __hip_bfloat16 nv_bfloat16;
প্রকারলেখ __hip_bfloat162 nv_bfloat162;

#যদি HIP_VERSION >= 60300000
#অন্তর্ভাব <hip/hip_fp8.h>
প্রকারলেখ __hip_fp8_e4m3 __nv_fp8_e4m3;
#define FP8_AVAILABLE
#endif // HIP_VERSION >= 60300000

প্রকারলেখ int8_t int8x4_t __attribute__((ext_vector_type(4)));
প্রকারলেখ uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
স্থির __device__ __forceinline__ পূর্ণসংখ্যা __vsubss4(ধ্রুবক পূর্ণসংখ্যা a, ধ্রুবক পূর্ণসংখ্যা b) {
    ধ্রুবক int8x4_t va = reinterpret_cast<ধ্রুবক int8x4_t&>(a);
    ধ্রুবক int8x4_t vb = reinterpret_cast<ধ্রুবক int8x4_t&>(b);
#যদি __has_builtin(__builtin_elementwise_sub_sat)
    ধ্রুবক int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    ফেরত reinterpret_cast<ধ্রুবক পূর্ণসংখ্যা &>(c);
#অন্যথা
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        যদি(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        যদি(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    ফেরত reinterpret_cast<পূর্ণসংখ্যা &>(c);
#endif // __has_builtin(__builtin_elementwise_sub_sat)
}

স্থির __device__ __forceinline__ পূর্ণসংখ্যা __vsub4(ধ্রুবক পূর্ণসংখ্যা a, ধ্রুবক পূর্ণসংখ্যা b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ধ্রুবক unsigned পূর্ণসংখ্যা a_large = a | 0x80808080;
    ধ্রুবক unsigned পূর্ণসংখ্যা b_small = b & 0x7f7f7f7f;
    ধ্রুবক unsigned পূর্ণসংখ্যা ফলাফল_নিচের_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ধ্রুবক unsigned পূর্ণসংখ্যা should_flip_উপরের_1bit = (a ^ ~b) & 0x80808080;

    ফেরত ফলাফল_নিচের_7bits ^ should_flip_উপরের_1bit;
}

স্থির __device__ __forceinline__ unsigned পূর্ণসংখ্যা __vcmpeq4(unsigned পূর্ণসংখ্যা a, unsigned পূর্ণসংখ্যা b) {
    ধ্রুবক uint8x4_t& va = reinterpret_cast<ধ্রুবক uint8x4_t&>(a);
    ধ্রুবক uint8x4_t& vb = reinterpret_cast<ধ্রুবক uint8x4_t&>(b);
    unsigned পূর্ণসংখ্যা c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    জন্য (পূর্ণসংখ্যা i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    ফেরত c;
}

স্থির __device__ __forceinline__ unsigned পূর্ণসংখ্যা __vcmpne4(unsigned পূর্ণসংখ্যা a, unsigned পূর্ণসংখ্যা b) {
    ধ্রুবক unsigned পূর্ণসংখ্যা x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ধ্রুবক unsigned পূর্ণসংখ্যা ne_নিচের_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ধ্রুবক unsigned পূর্ণসংখ্যা ne_উপরের_1bit = x & 0x80808080;
    ধ্রুবক unsigned পূর্ণসংখ্যা ne_any_bit = ne_নিচের_7bits | ne_উপরের_1bit;

    ফেরত (ne_any_bit >> 7) * 0xff;
}
