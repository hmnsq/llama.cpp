#निर्देशः once

#परिभाषय HIP_DISABLE_WARP_SYNC_BUILTINS 1
#अन्तर्भाव <hip/hip_runtime.h>
#अन्तर्भाव <hipblas/hipblas.h>
#अन्तर्भाव <hip/hip_fp16.h>
#अन्तर्भाव <hip/hip_bf16.h>

#यदि_परिभाषितम् GGML_USE_NCCL
#अन्तर्भाव <rccl/rccl.h>
#यदि_अन्तः // GGML_USE_NCCL


#परिभाषय CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#परिभाषय CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#परिभाषय CUBLAS_OP_N HIPBLAS_OP_N
#परिभाषय CUBLAS_OP_T HIPBLAS_OP_T
#परिभाषय CUBLAS_स्थितिः_SUCCESS HIPBLAS_स्थितिः_SUCCESS
#परिभाषय CUBLAS_TF32_TENSOR_OP_MATH 0
#परिभाषय CUDA_R_16F  HIPBLAS_R_16F
#परिभाषय CUDA_R_16BF HIPBLAS_R_16B
#परिभाषय CUDA_R_32F  HIPBLAS_R_32F
#परिभाषय CUBLAS_SIDE_दक्षिणः HIPBLAS_SIDE_दक्षिणः
#परिभाषय CUBLAS_FILL_बहुलकम्_ऊर्ध्वस्थम् HIPBLAS_FILL_बहुलकम्_ऊर्ध्वस्थम्
#परिभाषय CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#परिभाषय CU_DEVICE_ATTRIBUTE_VIRTUAL_स्मृतिः_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_स्मृतिः_Management_Supported
#परिभाषय CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_आवण्टनम्_Granularity_Recommended
#परिभाषय CU_MEM_आवण्टनम्_TYPE_PINNED hip_Mem_आवण्टनम्_Type_Pinned
#परिभाषय CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#परिभाषय CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#परिभाषय CU_CHECK(fn) {hipError_t err = fn; यदि(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#परिभाषय __shfl_sync(आवरणम्, var, lane_आवरणम्, विस्तारः) __shfl(var, lane_आवरणम्, विस्तारः)
#परिभाषय __shfl_up_sync(आवरणम्, var, lane_आवरणम्, विस्तारः) __shfl_up(var, lane_आवरणम्, विस्तारः)
#परिभाषय __shfl_xor_sync(आवरणम्, var, lane_आवरणम्, विस्तारः) __shfl_xor(var, lane_आवरणम्, विस्तारः)
#परिभाषय __all_sync(आवरणम्, var) __all(var)
#परिभाषय __any_sync(आवरणम्, var) __any(var)
#परिभाषय cublasStrsmBatched hipblasStrsmBatched
#परिभाषय cublasCreate hipblasCreate
#परिभाषय cublasDestroy hipblasDestroy
#परिभाषय cublasGemmEx hipblasGemmEx
#परिभाषय cublasGemmBatchedEx hipblasGemmBatchedEx
#परिभाषय cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#परिभाषय cublasHandle_t hipblasHandle_t
#परिभाषय cublas_समुच्चयः_Math_बहुलकम्(handle, बहुलकम्) CUBLAS_स्थितिः_SUCCESS
#परिभाषय cublas_समुच्चयः_प्रवाहः hipblas_समुच्चयः_प्रवाहः
#परिभाषय cublasSgemm hipblasSgemm
#परिभाषय cublasSgemmBatched hipblasSgemmBatched
#परिभाषय cublasSgemmStridedBatched hipblasSgemmStridedBatched
#परिभाषय cublas_स्थितिः_t hipblas_स्थितिः_t
#परिभाषय cublasOperation_t hipblasOperation_t
#परिभाषय cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#परिभाषय cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#परिभाषय cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#परिभाषय cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#परिभाषय cudaDeviceGetAttribute hipDeviceGetAttribute
#परिभाषय cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#परिभाषय cudaDeviceProp hipDeviceProp_t
#परिभाषय cudaDeviceSynchronize hipDeviceSynchronize
#परिभाषय cudaError_t hipError_t
#परिभाषय cuda_Error_स्मृतिः_आवण्टनम् hip_Error_बहिर्गमनम्_सम्बन्धि_स्मृतिः
#परिभाषय cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#परिभाषय cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#परिभाषय cudaEventCreateWithFlags hipEventCreateWithFlags
#परिभाषय cudaEventDisableTiming hipEventDisableTiming
#परिभाषय cuda_Event_लेखः hip_Event_लेखः
#परिभाषय cudaEventSynchronize hipEventSynchronize
#परिभाषय cudaEvent_t hipEvent_t
#परिभाषय cudaEventDestroy hipEventDestroy
#परिभाषय cudaFree hipFree
#परिभाषय cudaFreeHost hipHostFree
#परिभाषय cudaGetDevice hipGetDevice
#परिभाषय cuda_Get_Device_गणनम् hip_Get_Device_गणनम्
#परिभाषय cudaGetDeviceProperties hipGetDeviceProperties
#परिभाषय cudaGetErrorString hipGetErrorString
#परिभाषय cuda_Get_चरमः_Error hip_Get_चरमः_Error
#परिभाषय cudaHostAlloc hipHostMalloc
#परिभाषय cudaHostAllocPortable hipHostMallocPortable
#परिभाषय cudaHostAllocMapped hipHostMallocMapped
#परिभाषय cudaHostGetDevicePointer hipHostGetDevicePointer
#परिभाषय cuda_Host_पञ्जिकाकोशः hip_Host_पञ्जिकाकोशः
#परिभाषय cuda_Host_पञ्जिकाकोशः_Portable hip_Host_पञ्जिकाकोशः_Portable
#परिभाषय cuda_Host_पञ्जिकाकोशः_Read_Only hip_Host_पञ्जिकाकोशः_Read_Only
#परिभाषय cudaHostUnregister hipHostUnregister
#परिभाषय cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#परिभाषय cudaLaunchHostFunc hipLaunchHostFunc
#परिभाषय cudaMalloc hipMalloc
#परिभाषय cudaMallocHost(ptr, आकार) hipHostMalloc(ptr, आकार, hipHostMallocDefault)
#परिभाषय cudaMallocManaged hipMallocManaged
#परिभाषय cudaMemAdvise hipMemAdvise
#परिभाषय cudaMemcpy hipMemcpy
#परिभाषय cudaMemcpyAsync hipMemcpyAsync
#परिभाषय cudaMemcpyPeerAsync hipMemcpyPeerAsync
#परिभाषय cudaMemcpy2DAsync hipMemcpy2DAsync
#परिभाषय cuda_Memcpy_Device_गन्तव्यम्_Device hip_Memcpy_Device_गन्तव्यम्_Device
#परिभाषय cuda_Memcpy_Device_गन्तव्यम्_Host hip_Memcpy_Device_गन्तव्यम्_Host
#परिभाषय cuda_Memcpy_Host_गन्तव्यम्_Device hip_Memcpy_Host_गन्तव्यम्_Device
#परिभाषय cuda_Memcpy_विधा hip_Memcpy_विधा
#परिभाषय cudaMemset hipMemset
#परिभाषय cudaMemsetAsync hipMemsetAsync
#परिभाषय cudaMemGetInfo hipMemGetInfo
#परिभाषय cuda_Occupancy_Max_Potential_खण्डम्_आकार hip_Occupancy_Max_Potential_खण्डम्_आकार
#परिभाषय cuda_समुच्चयः_Device hip_समुच्चयः_Device
#परिभाषय cuDeviceGet hipDeviceGet
#परिभाषय CUdevice hipDevice_t
#परिभाषय CUdeviceptr hipDeviceptr_t
#परिभाषय cuMemUnmap hipMemUnmap
#परिभाषय CUmemAccessDesc hipMemAccessDesc
#परिभाषय cu_Mem_स्थानाङ्कः_Free hip_Mem_स्थानाङ्कः_Free
#परिभाषय cuMemRelease hipMemRelease
#परिभाषय CUmem_Generic_आवण्टनम्_Handle hip_Mem_Generic_आवण्टनम्_Handle_t
#परिभाषय cuMemCreate hipMemCreate
#परिभाषय cu_Mem_स्थानाङ्कः_Reserve hip_Mem_स्थानाङ्कः_Reserve
#परिभाषय cuMemMap hipMemMap
#परिभाषय cu_Mem_समुच्चयः_Access hip_Mem_समुच्चयः_Access
#परिभाषय cu_Mem_Get_आवण्टनम्_Granularity hip_Mem_Get_आवण्टनम्_Granularity
#परिभाषय CUmem_आवण्टनम्_Prop hip_Mem_आवण्टनम्_Prop
#परिभाषय cuDeviceGetAttribute hipDeviceGetAttribute
#परिभाषय cuda_प्रवाहः_Create_With_Flags hip_प्रवाहः_Create_With_Flags
#परिभाषय cuda_प्रवाहः_Destroy hip_प्रवाहः_Destroy
#परिभाषय cuda_प्रवाहः_Fire_And_Forget hip_प्रवाहः_Fire_And_Forget
#परिभाषय cuda_प्रवाहः_Non_Blocking hip_प्रवाहः_Non_Blocking
#परिभाषय cuda_प्रवाहः_प्रत्येकम्_Thread hip_प्रवाहः_प्रत्येकम्_Thread
#परिभाषय cuda_प्रवाहः_Synchronize hip_प्रवाहः_Synchronize
#परिभाषय cuda_प्रवाहः_Wait_Event hip_प्रवाहः_Wait_Event
#परिभाषय cudaGraphExec_t hipGraphExec_t
#परिभाषय cuda_Graph_ग्रन्थिः_t hip_Graph_ग्रन्थिः_t
#परिभाषय cuda_Kernel_ग्रन्थिः_Params hip_Kernel_ग्रन्थिः_Params
#परिभाषय cuda_Kernel_ग्रन्थिः_Params hip_Kernel_ग्रन्थिः_Params
#परिभाषय cudaGraphExecDestroy hipGraphExecDestroy
#परिभाषय cudaGraphLaunch hipGraphLaunch
#परिभाषय cudaErrorGraphExecUpdateFailure hipErrorGraphExecUpdateFailure
#परिभाषय cuda_Graph_Exec_Update_परिणाम hip_Graph_Exec_Update_परिणाम
#परिभाषय cuda_Graph_ग्रन्थिः_Type hip_Graph_ग्रन्थिः_Type
#परिभाषय cuda_Graph_ग्रन्थिः_Type_Kernel hip_Graph_ग्रन्थिः_Type_Kernel
#परिभाषय cudaGraphInstantiate hipGraphInstantiate
#परिभाषय cuda_प्रवाहः_अन्तः_Capture hip_प्रवाहः_अन्तः_Capture
#परिभाषय cudaGraphDestroy hipGraphDestroy
#परिभाषय cuda_Graph_Kernel_ग्रन्थिः_समुच्चयः_Params hip_Graph_Kernel_ग्रन्थिः_समुच्चयः_Params
#परिभाषय cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#परिभाषय cuda_Graph_Kernel_ग्रन्थिः_Get_Params hip_Graph_Kernel_ग्रन्थिः_Get_Params
#परिभाषय cuda_Graph_ग्रन्थिः_Get_Type hip_Graph_ग्रन्थिः_Get_Type
#परिभाषय cudaGraphGetNodes hipGraphGetNodes
#परिभाषय cudaGraphExecUpdate hipGraphExecUpdate
#परिभाषय cuda_प्रवाहः_Capture_बहुलकम्_Relaxed hip_प्रवाहः_Capture_बहुलकम्_Relaxed
#परिभाषय cuda_प्रवाहः_Begin_Capture hip_प्रवाहः_Begin_Capture
#परिभाषय cudaGraph_t hipGraph_t
#परिभाषय cuda_प्रवाहः_t hip_प्रवाहः_t
#परिभाषय cudaSuccess hipSuccess
#परिभाषय cuda_Occupancy_Max_सक्रिय_खण्डानि_प्रत्येकम्_Multiprocessor hip_Occupancy_Max_सक्रिय_खण्डानि_प्रत्येकम्_Multiprocessor
#परिभाषय cuda_Func_समुच्चयः_Attribute hip_Func_समुच्चयः_Attribute
#परिभाषय cuda_Func_Attribute_Max_Dynamic_Shared_स्मृतिः_आकार hip_Func_Attribute_Max_Dynamic_Shared_स्मृतिः_आकार
#परिभाषय __trap() कुरु { abort(); __builtin_unreachable(); } यावत्(0)
#परिभाषय CUBLAS_स्थितिः_SUCCESS HIPBLAS_स्थितिः_SUCCESS
#परिभाषय CUBLAS_स्थितिः_NOT_INITIALIZED HIPBLAS_स्थितिः_NOT_INITIALIZED
#परिभाषय CUBLAS_स्थितिः_ALLOC_FAILED HIPBLAS_स्थितिः_ALLOC_FAILED
#परिभाषय CUBLAS_स्थितिः_INVALID_मानम् HIPBLAS_स्थितिः_INVALID_मानम्
#परिभाषय CUBLAS_स्थितिः_ARCH_MISMATCH HIPBLAS_स्थितिः_ARCH_MISMATCH
#परिभाषय CUBLAS_स्थितिः_MAPPING_ERROR HIPBLAS_स्थितिः_MAPPING_ERROR
#परिभाषय CUBLAS_स्थितिः_EXECUTION_FAILED HIPBLAS_स्थितिः_EXECUTION_FAILED
#परिभाषय CUBLAS_स्थितिः_आन्तरिकम्_ERROR HIPBLAS_स्थितिः_आन्तरिकम्_ERROR
#परिभाषय CUBLAS_स्थितिः_NOT_SUPPORTED HIPBLAS_स्थितिः_NOT_SUPPORTED

#यदि HIP_संस्करणम् >= 60500000
#परिभाषय CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#परिभाषय CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#परिभाषय CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#परिभाषय cublasComputeType_t hipblasComputeType_t
#परिभाषय cuda_दत्त_Type_t hip_दत्त_Type
#अन्यथा
#परिभाषय CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#परिभाषय CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#परिभाषय CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#परिभाषय cublasComputeType_t hipblasDatatype_t
#परिभाषय cuda_दत्त_Type_t hipblasDatatype_t
#यदि_अन्तः // HIP_VERSION >= 6050000

#यदि !defined(__HIP_PLATFORM_AMD__)
#दोषः "The HIP backend supports only AMD targets"
#यदि_अन्तः // !defined(__HIP_PLATFORM_AMD__)

#परिभाषय __CUDA_ARCH__ 1300

#यदि defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#परिभाषय GCN5
#यदि_अन्तः // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#यदि defined(__gfx803__)
#परिभाषय GCN4
#यदि_अन्तः // defined(__gfx803__)

#यदि defined(GCN5) || defined(GCN4)
#परिभाषय GCN
#यदि_अन्तः // defined(GCN5) || defined(GCN4)

#यदि defined(__gfx950__)
#परिभाषय CDNA4
#यदि_अन्तः // defined(__gfx950__)

#यदि defined(__gfx942__)
#परिभाषय CDNA3
#यदि_अन्तः // defined(__gfx942__)

#यदि defined(__gfx90a__)
#परिभाषय CDNA2
#यदि_अन्तः // defined(__gfx90a__)

#यदि defined(__gfx908__)
#परिभाषय CDNA1
#यदि_अन्तः // defined(__gfx908__)

#यदि defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#परिभाषय CDNA // For the entire family
#यदि_अन्तः // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#यदि defined(__GFX12__)
#परिभाषय RDNA4
#यदि_अन्तः // defined(__GFX12__)

#यदि defined(__GFX11__)
#परिभाषय RDNA3
#यदि_अन्तः // defined(__GFX11__)

#यदि defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#परिभाषय RDNA3_5
#यदि_अन्तः // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#यदि defined(RDNA3) && !defined(RDNA3_5)
#परिभाषय RDNA3_0
#यदि_अन्तः // defined(RDNA3) && !defined(RDNA3_5)

#यदि defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#परिभाषय RDNA2
#यदि_अन्तः

#यदि defined(__gfx1010__) || defined(__gfx1012__)
#परिभाषय RDNA1
#यदि_अन्तः // defined(__gfx1010__) || defined(__gfx1012__)

#यदि defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#परिभाषय RDNA // For the entire family
#यदि_अन्तः // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#यदि_अपरिभाषितम् __has_builtin
    #परिभाषय __has_builtin(x) 0
#यदि_अन्तः

प्रकारनिर्देश __hip_bfloat16 nv_bfloat16;
प्रकारनिर्देश __hip_bfloat162 nv_bfloat162;

#यदि HIP_संस्करणम् >= 60300000
#अन्तर्भाव <hip/hip_fp8.h>
प्रकारनिर्देश __hip_fp8_e4m3 __nv_fp8_e4m3;
#परिभाषय FP8_उपलब्धम्
#यदि_अन्तः // HIP_VERSION >= 60300000

प्रकारनिर्देश int8_t int8x4_t __attribute__((ext_सदिशम्_type(4)));
प्रकारनिर्देश uint8_t uint8x4_t __attribute__((ext_सदिशम्_type(4)));
स्थिर __device__ __forceinline__ पूर्णाङ्क __vsubss4(ध्रुव पूर्णाङ्क a, ध्रुव पूर्णाङ्क b) {
    ध्रुव int8x4_t va = reinterpret_cast<ध्रुव int8x4_t&>(a);
    ध्रुव int8x4_t vb = reinterpret_cast<ध्रुव int8x4_t&>(b);
#यदि __has_builtin(__builtin_elementwise_sub_sat)
    ध्रुव int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    प्रत्यागम reinterpret_cast<ध्रुव पूर्णाङ्क &>(c);
#अन्यथा
    int8x4_t c;
    int16_t tmp;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        यदि(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        यदि(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    प्रत्यागम reinterpret_cast<पूर्णाङ्क &>(c);
#यदि_अन्तः // __has_builtin(__builtin_elementwise_sub_sat)
}

स्थिर __device__ __forceinline__ पूर्णाङ्क __vsub4(ध्रुव पूर्णाङ्क a, ध्रुव पूर्णाङ्क b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ध्रुव अचिह्नित पूर्णाङ्क a_large = a | 0x80808080;
    ध्रुव अचिह्नित पूर्णाङ्क b_small = b & 0x7f7f7f7f;
    ध्रुव अचिह्नित पूर्णाङ्क परिणाम_नीचः_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ध्रुव अचिह्नित पूर्णाङ्क should_flip_उच्चः_1bit = (a ^ ~b) & 0x80808080;

    प्रत्यागम परिणाम_नीचः_7bits ^ should_flip_उच्चः_1bit;
}

स्थिर __device__ __forceinline__ अचिह्नित पूर्णाङ्क __vcmpeq4(अचिह्नित पूर्णाङ्क a, अचिह्नित पूर्णाङ्क b) {
    ध्रुव uint8x4_t& va = reinterpret_cast<ध्रुव uint8x4_t&>(a);
    ध्रुव uint8x4_t& vb = reinterpret_cast<ध्रुव uint8x4_t&>(b);
    अचिह्नित पूर्णाङ्क c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    प्रत्यागम c;
}

स्थिर __device__ __forceinline__ अचिह्नित पूर्णाङ्क __vcmpne4(अचिह्नित पूर्णाङ्क a, अचिह्नित पूर्णाङ्क b) {
    ध्रुव अचिह्नित पूर्णाङ्क x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ध्रुव अचिह्नित पूर्णाङ्क ne_नीचः_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ध्रुव अचिह्नित पूर्णाङ्क ne_उच्चः_1bit = x & 0x80808080;
    ध्रुव अचिह्नित पूर्णाङ्क ne_any_bit = ne_नीचः_7bits | ne_उच्चः_1bit;

    प्रत्यागम (ne_any_bit >> 7) * 0xff;
}
