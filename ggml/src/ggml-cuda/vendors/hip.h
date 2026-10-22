#आशय once

#घोषणा HIP_DISABLE_WARP_SYNC_BUILTINS 1
#समावेश <hip/hip_runtime.h>
#समावेश <hipblas/hipblas.h>
#समावेश <hip/hip_fp16.h>
#समावेश <hip/hip_bf16.h>

#अगर_घोषित GGML_USE_NCCL
#समावेश <rccl/rccl.h>
#पूर्ण_अगर // GGML_USE_NCCL


#घोषणा CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#घोषणा CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#घोषणा CUBLAS_OP_N HIPBLAS_OP_N
#घोषणा CUBLAS_OP_T HIPBLAS_OP_T
#घोषणा CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#घोषणा CUBLAS_TF32_TENSOR_OP_MATH 0
#घोषणा CUDA_R_16F  HIPBLAS_R_16F
#घोषणा CUDA_R_16BF HIPBLAS_R_16B
#घोषणा CUDA_R_32F  HIPBLAS_R_32F
#घोषणा CUBLAS_SIDE_दायाँ HIPBLAS_SIDE_दायाँ
#घोषणा CUBLAS_FILL_MODE_UPPER HIPBLAS_FILL_MODE_UPPER
#घोषणा CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#घोषणा CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED hipDeviceAttributeVirtualMemoryManagementSupported
#घोषणा CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hipMemAllocationGranularityRecommended
#घोषणा CU_MEM_ALLOCATION_TYPE_PINNED hipMemAllocationTypePinned
#घोषणा CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#घोषणा CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#घोषणा CU_CHECK(fn) {hipError_t err = fn; अगर(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#घोषणा __shfl_sync(mask, var, laneMask, width) __shfl(var, laneMask, width)
#घोषणा __shfl_up_sync(mask, var, laneMask, width) __shfl_up(var, laneMask, width)
#घोषणा __shfl_xor_sync(mask, var, laneMask, width) __shfl_xor(var, laneMask, width)
#घोषणा __all_sync(mask, var) __all(var)
#घोषणा __any_sync(mask, var) __any(var)
#घोषणा cublasStrsmBatched hipblasStrsmBatched
#घोषणा cublasCreate hipblasCreate
#घोषणा cublasDestroy hipblasDestroy
#घोषणा cublasGemmEx hipblasGemmEx
#घोषणा cublasGemmBatchedEx hipblasGemmBatchedEx
#घोषणा cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#घोषणा cublasHandle_t hipblasHandle_t
#घोषणा cublasSetMathMode(handle, mode) CUBLAS_STATUS_SUCCESS
#घोषणा cublasSetStream hipblasSetStream
#घोषणा cublasSgemm hipblasSgemm
#घोषणा cublasSgemmBatched hipblasSgemmBatched
#घोषणा cublasSgemmStridedBatched hipblasSgemmStridedBatched
#घोषणा cublasStatus_t hipblasStatus_t
#घोषणा cublasOperation_t hipblasOperation_t
#घोषणा cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#घोषणा cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#घोषणा cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#घोषणा cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#घोषणा cudaDeviceGetAttribute hipDeviceGetAttribute
#घोषणा cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#घोषणा cudaDeviceProp hipDeviceProp_t
#घोषणा cudaDeviceSynchronize hipDeviceSynchronize
#घोषणा cudaError_t hipError_t
#घोषणा cudaErrorMemoryAllocation hipErrorOutOfMemory
#घोषणा cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#घोषणा cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#घोषणा cudaEventCreateWithFlags hipEventCreateWithFlags
#घोषणा cudaEventDisableTiming hipEventDisableTiming
#घोषणा cudaEventRecord hipEventRecord
#घोषणा cudaEventSynchronize hipEventSynchronize
#घोषणा cudaEvent_t hipEvent_t
#घोषणा cudaEventDestroy hipEventDestroy
#घोषणा cudaFree hipFree
#घोषणा cudaFreeHost hipHostFree
#घोषणा cudaGetDevice hipGetDevice
#घोषणा cuda_Get_Device_गिनती hip_Get_Device_गिनती
#घोषणा cudaGetDeviceProperties hipGetDeviceProperties
#घोषणा cudaGetErrorString hipGetErrorString
#घोषणा cudaGetLastError hipGetLastError
#घोषणा cudaHostAlloc hipHostMalloc
#घोषणा cudaHostAllocPortable hipHostMallocPortable
#घोषणा cudaHostAllocMapped hipHostMallocMapped
#घोषणा cudaHostGetDevicePointer hipHostGetDevicePointer
#घोषणा cudaHostRegister hipHostRegister
#घोषणा cudaHostRegisterPortable hipHostRegisterPortable
#घोषणा cudaHostRegisterReadOnly hipHostRegisterReadOnly
#घोषणा cudaHostUnregister hipHostUnregister
#घोषणा cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#घोषणा cudaLaunchHostFunc hipLaunchHostFunc
#घोषणा cudaMalloc hipMalloc
#घोषणा cudaMallocHost(ptr, आकार) hipHostMalloc(ptr, आकार, hipHostMallocDefault)
#घोषणा cudaMallocManaged hipMallocManaged
#घोषणा cudaMemAdvise hipMemAdvise
#घोषणा cudaMemcpy hipMemcpy
#घोषणा cudaMemcpyAsync hipMemcpyAsync
#घोषणा cudaMemcpyPeerAsync hipMemcpyPeerAsync
#घोषणा cudaMemcpy2DAsync hipMemcpy2DAsync
#घोषणा cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#घोषणा cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#घोषणा cudaMemcpyHostToDevice hipMemcpyHostToDevice
#घोषणा cudaMemcpyKind hipMemcpyKind
#घोषणा cudaMemset hipMemset
#घोषणा cudaMemsetAsync hipMemsetAsync
#घोषणा cudaMemGetInfo hipMemGetInfo
#घोषणा cuda_Occupancy_Max_Potential_Block_आकार hip_Occupancy_Max_Potential_Block_आकार
#घोषणा cudaSetDevice hipSetDevice
#घोषणा cuDeviceGet hipDeviceGet
#घोषणा CUdevice hipDevice_t
#घोषणा CUdeviceptr hipDeviceptr_t
#घोषणा cuMemUnmap hipMemUnmap
#घोषणा CUmemAccessDesc hipMemAccessDesc
#घोषणा cuMemAddressFree hipMemAddressFree
#घोषणा cuMemRelease hipMemRelease
#घोषणा CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t
#घोषणा cuMemCreate hipMemCreate
#घोषणा cuMemAddressReserve hipMemAddressReserve
#घोषणा cuMemMap hipMemMap
#घोषणा cuMemSetAccess hipMemSetAccess
#घोषणा cuMemGetAllocationGranularity hipMemGetAllocationGranularity
#घोषणा CUmemAllocationProp hipMemAllocationProp
#घोषणा cuDeviceGetAttribute hipDeviceGetAttribute
#घोषणा cudaStreamCreateWithFlags hipStreamCreateWithFlags
#घोषणा cudaStreamDestroy hipStreamDestroy
#घोषणा cudaStreamFireAndForget hipStreamFireAndForget
#घोषणा cudaStreamNonBlocking hipStreamNonBlocking
#घोषणा cudaStreamPerThread hipStreamPerThread
#घोषणा cudaStreamSynchronize hipStreamSynchronize
#घोषणा cudaStreamWaitEvent hipStreamWaitEvent
#घोषणा cudaGraphExec_t hipGraphExec_t
#घोषणा cuda_Graph_गाँठ_t hip_Graph_गाँठ_t
#घोषणा cuda_Kernel_गाँठ_Params hip_Kernel_गाँठ_Params
#घोषणा cuda_Kernel_गाँठ_Params hip_Kernel_गाँठ_Params
#घोषणा cudaGraphExecDestroy hipGraphExecDestroy
#घोषणा cudaGraphLaunch hipGraphLaunch
#घोषणा cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#घोषणा cuda_Graph_Exec_Update_परिणाम hip_Graph_Exec_Update_परिणाम
#घोषणा cuda_Graph_गाँठ_Type hip_Graph_गाँठ_Type
#घोषणा cuda_Graph_गाँठ_Type_Kernel hip_Graph_गाँठ_Type_Kernel
#घोषणा cudaGraphInstantiate hipGraphInstantiate
#घोषणा cudaStreamEndCapture hipStreamEndCapture
#घोषणा cudaGraphDestroy hipGraphDestroy
#घोषणा cuda_Graph_Kernel_गाँठ_Set_Params hip_Graph_Kernel_गाँठ_Set_Params
#घोषणा cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#घोषणा cuda_Graph_Kernel_गाँठ_Get_Params hip_Graph_Kernel_गाँठ_Get_Params
#घोषणा cuda_Graph_गाँठ_Get_Type hip_Graph_गाँठ_Get_Type
#घोषणा cudaGraphGetNodes hipGraphGetNodes
#घोषणा cudaGraphExecUpdate hipGraphExecUpdate
#घोषणा cudaStreamCaptureModeRelaxed hipStreamCaptureModeRelaxed
#घोषणा cudaStreamBeginCapture hipStreamBeginCapture
#घोषणा cudaGraph_t hipGraph_t
#घोषणा cudaStream_t hipStream_t
#घोषणा cudaSuccess hipSuccess
#घोषणा cudaOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#घोषणा cudaFuncSetAttribute hipFuncSetAttribute
#घोषणा cuda_Func_Attribute_Max_Dynamic_Shared_Memory_आकार hip_Func_Attribute_Max_Dynamic_Shared_Memory_आकार
#घोषणा __trap() करो { पात(); __builtin_unreachable(); } जबतक(0)
#घोषणा CUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#घोषणा CUBLAS_STATUS_NOT_INITIALIZED HIPBLAS_STATUS_NOT_INITIALIZED
#घोषणा CUBLAS_STATUS_ALLOC_FAILED HIPBLAS_STATUS_ALLOC_FAILED
#घोषणा CUBLAS_STATUS_INVALID_मूल्य HIPBLAS_STATUS_INVALID_मूल्य
#घोषणा CUBLAS_STATUS_ARCH_MISMATCH HIPBLAS_STATUS_ARCH_MISMATCH
#घोषणा CUBLAS_STATUS_MAPPING_ERROR HIPBLAS_STATUS_MAPPING_ERROR
#घोषणा CUBLAS_STATUS_EXECUTION_FAILED HIPBLAS_STATUS_EXECUTION_FAILED
#घोषणा CUBLAS_STATUS_INTERNAL_ERROR HIPBLAS_STATUS_INTERNAL_ERROR
#घोषणा CUBLAS_STATUS_NOT_SUPPORTED HIPBLAS_STATUS_NOT_SUPPORTED

#अगर HIP_VERSION >= 60500000
#घोषणा CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#घोषणा CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#घोषणा CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#घोषणा cublasComputeType_t hipblasComputeType_t
#घोषणा cuda_तथ्य_Type_t hip_तथ्य_Type
#अन्यथा
#घोषणा CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#घोषणा CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#घोषणा CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#घोषणा cublasComputeType_t hipblasDatatype_t
#घोषणा cuda_तथ्य_Type_t hipblasDatatype_t
#पूर्ण_अगर // HIP_VERSION >= 6050000

#अगर !defined(__HIP_PLATFORM_AMD__)
#त्रुटि "The HIP backend supports only AMD targets"
#पूर्ण_अगर // !defined(__HIP_PLATFORM_AMD__)

#घोषणा __CUDA_ARCH__ 1300

#अगर defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#घोषणा GCN5
#पूर्ण_अगर // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#अगर defined(__gfx803__)
#घोषणा GCN4
#पूर्ण_अगर // defined(__gfx803__)

#अगर defined(GCN5) || defined(GCN4)
#घोषणा GCN
#पूर्ण_अगर // defined(GCN5) || defined(GCN4)

#अगर defined(__gfx950__)
#घोषणा CDNA4
#पूर्ण_अगर // defined(__gfx950__)

#अगर defined(__gfx942__)
#घोषणा CDNA3
#पूर्ण_अगर // defined(__gfx942__)

#अगर defined(__gfx90a__)
#घोषणा CDNA2
#पूर्ण_अगर // defined(__gfx90a__)

#अगर defined(__gfx908__)
#घोषणा CDNA1
#पूर्ण_अगर // defined(__gfx908__)

#अगर defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#घोषणा CDNA // For the entire family
#पूर्ण_अगर // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#अगर defined(__GFX12__)
#घोषणा RDNA4
#पूर्ण_अगर // defined(__GFX12__)

#अगर defined(__GFX11__)
#घोषणा RDNA3
#पूर्ण_अगर // defined(__GFX11__)

#अगर defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#घोषणा RDNA3_5
#पूर्ण_अगर // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#अगर defined(RDNA3) && !defined(RDNA3_5)
#घोषणा RDNA3_0
#पूर्ण_अगर // defined(RDNA3) && !defined(RDNA3_5)

#अगर defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#घोषणा RDNA2
#पूर्ण_अगर

#अगर defined(__gfx1010__) || defined(__gfx1012__)
#घोषणा RDNA1
#पूर्ण_अगर // defined(__gfx1010__) || defined(__gfx1012__)

#अगर defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#घोषणा RDNA // For the entire family
#पूर्ण_अगर // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#अगर_अघोषित __has_builtin
    #घोषणा __has_builtin(x) 0
#पूर्ण_अगर

प्रकार __hip_bfloat16 nv_bfloat16;
प्रकार __hip_bfloat162 nv_bfloat162;

#अगर HIP_VERSION >= 60300000
#समावेश <hip/hip_fp8.h>
प्रकार __hip_fp8_e4m3 __nv_fp8_e4m3;
#घोषणा FP8_AVAILABLE
#पूर्ण_अगर // HIP_VERSION >= 60300000

प्रकार int8_t int8x4_t __attribute__((ext_vector_type(4)));
प्रकार uint8_t uint8x4_t __attribute__((ext_vector_type(4)));
अटल __device__ __forceinline__ पूर्णांक __vsubss4(स्थिर पूर्णांक a, स्थिर पूर्णांक b) {
    स्थिर int8x4_t va = reinterpret_cast<स्थिर int8x4_t&>(a);
    स्थिर int8x4_t vb = reinterpret_cast<स्थिर int8x4_t&>(b);
#अगर __has_builtin(__builtin_elementwise_sub_sat)
    स्थिर int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    वापस reinterpret_cast<स्थिर पूर्णांक &>(c);
#अन्यथा
    int8x4_t c;
    int16_t tmp;
#आशय unroll
    क्रम (पूर्णांक i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        अगर(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        अगर(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    वापस reinterpret_cast<पूर्णांक &>(c);
#पूर्ण_अगर // __has_builtin(__builtin_elementwise_sub_sat)
}

अटल __device__ __forceinline__ पूर्णांक __vsub4(स्थिर पूर्णांक a, स्थिर पूर्णांक b) {
    // do some small modifications to a and b to make the subtraction not underflow
    स्थिर अचिन्हित पूर्णांक a_large = a | 0x80808080;
    स्थिर अचिन्हित पूर्णांक b_small = b & 0x7f7f7f7f;
    स्थिर अचिन्हित पूर्णांक परिणाम_निचला_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    स्थिर अचिन्हित पूर्णांक should_flip_ऊपरी_1bit = (a ^ ~b) & 0x80808080;

    वापस परिणाम_निचला_7bits ^ should_flip_ऊपरी_1bit;
}

अटल __device__ __forceinline__ अचिन्हित पूर्णांक __vcmpeq4(अचिन्हित पूर्णांक a, अचिन्हित पूर्णांक b) {
    स्थिर uint8x4_t& va = reinterpret_cast<स्थिर uint8x4_t&>(a);
    स्थिर uint8x4_t& vb = reinterpret_cast<स्थिर uint8x4_t&>(b);
    अचिन्हित पूर्णांक c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#आशय unroll
    क्रम (पूर्णांक i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    वापस c;
}

अटल __device__ __forceinline__ अचिन्हित पूर्णांक __vcmpne4(अचिन्हित पूर्णांक a, अचिन्हित पूर्णांक b) {
    स्थिर अचिन्हित पूर्णांक x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    स्थिर अचिन्हित पूर्णांक ne_निचला_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    स्थिर अचिन्हित पूर्णांक ne_ऊपरी_1bit = x & 0x80808080;
    स्थिर अचिन्हित पूर्णांक ne_any_bit = ne_निचला_7bits | ne_ऊपरी_1bit;

    वापस (ne_any_bit >> 7) * 0xff;
}
