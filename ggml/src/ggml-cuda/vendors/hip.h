#సూచన once

#నిర్వచించు HIP_DISABLE_WARP_SYNC_BUILTINS 1
#అన్తర్భావ <hip/hip_runtime.h>
#అన్తర్భావ <hipblas/hipblas.h>
#అన్తర్భావ <hip/hip_fp16.h>
#అన్తర్భావ <hip/hip_bf16.h>

#నిర్వచితమైతే GGML_USE_NCCL
#అన్తర్భావ <rccl/rccl.h>
#అయితే_ముగింపు // GGML_USE_NCCL


#నిర్వచించు CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#నిర్వచించు CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#నిర్వచించు CUBLAS_OP_N HIPBLAS_OP_N
#నిర్వచించు CUBLAS_OP_T HIPBLAS_OP_T
#నిర్వచించు CUBLAS_స్థితి_SUCCESS HIPBLAS_స్థితి_SUCCESS
#నిర్వచించు CUBLAS_TF32_TENSOR_OP_MATH 0
#నిర్వచించు CUDA_R_16F  HIPBLAS_R_16F
#నిర్వచించు CUDA_R_16BF HIPBLAS_R_16B
#నిర్వచించు CUDA_R_32F  HIPBLAS_R_32F
#నిర్వచించు CUBLAS_SIDE_కుడి HIPBLAS_SIDE_కుడి
#నిర్వచించు CUBLAS_FILL_బహుళకం_పై HIPBLAS_FILL_బహుళకం_పై
#నిర్వచించు CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#నిర్వచించు CU_DEVICE_ATTRIBUTE_VIRTUAL_జ్ఞాపకం_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_జ్ఞాపకం_Management_Supported
#నిర్వచించు CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_కేటాయింపు_Granularity_Recommended
#నిర్వచించు CU_MEM_కేటాయింపు_TYPE_PINNED hip_Mem_కేటాయింపు_Type_Pinned
#నిర్వచించు CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#నిర్వచించు CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#నిర్వచించు CU_CHECK(fn) {hipError_t err = fn; ఒకవేళ(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#నిర్వచించు __shfl_sync(ముసుగు, var, lane_ముసుగు, వెడల్పు) __shfl(var, lane_ముసుగు, వెడల్పు)
#నిర్వచించు __shfl_up_sync(ముసుగు, var, lane_ముసుగు, వెడల్పు) __shfl_up(var, lane_ముసుగు, వెడల్పు)
#నిర్వచించు __shfl_xor_sync(ముసుగు, var, lane_ముసుగు, వెడల్పు) __shfl_xor(var, lane_ముసుగు, వెడల్పు)
#నిర్వచించు __all_sync(ముసుగు, var) __all(var)
#నిర్వచించు __any_sync(ముసుగు, var) __any(var)
#నిర్వచించు cublasStrsmBatched hipblasStrsmBatched
#నిర్వచించు cublasCreate hipblasCreate
#నిర్వచించు cublasDestroy hipblasDestroy
#నిర్వచించు cublasGemmEx hipblasGemmEx
#నిర్వచించు cublasGemmBatchedEx hipblasGemmBatchedEx
#నిర్వచించు cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#నిర్వచించు cublasHandle_t hipblasHandle_t
#నిర్వచించు cublas_సమితి_Math_బహుళకం(handle, బహుళకం) CUBLAS_స్థితి_SUCCESS
#నిర్వచించు cublas_సమితి_ప్రవాహం hipblas_సమితి_ప్రవాహం
#నిర్వచించు cublasSgemm hipblasSgemm
#నిర్వచించు cublasSgemmBatched hipblasSgemmBatched
#నిర్వచించు cublasSgemmStridedBatched hipblasSgemmStridedBatched
#నిర్వచించు cublas_స్థితి_t hipblas_స్థితి_t
#నిర్వచించు cublasOperation_t hipblasOperation_t
#నిర్వచించు cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#నిర్వచించు cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#నిర్వచించు cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#నిర్వచించు cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#నిర్వచించు cudaDeviceGetAttribute hipDeviceGetAttribute
#నిర్వచించు cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#నిర్వచించు cudaDeviceProp hipDeviceProp_t
#నిర్వచించు cudaDeviceSynchronize hipDeviceSynchronize
#నిర్వచించు cudaError_t hipError_t
#నిర్వచించు cuda_Error_జ్ఞాపకం_కేటాయింపు hip_Error_బహిర్గతం_యొక్క_జ్ఞాపకం
#నిర్వచించు cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#నిర్వచించు cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#నిర్వచించు cudaEventCreateWithFlags hipEventCreateWithFlags
#నిర్వచించు cudaEventDisableTiming hipEventDisableTiming
#నిర్వచించు cuda_Event_రికార్డు hip_Event_రికార్డు
#నిర్వచించు cudaEventSynchronize hipEventSynchronize
#నిర్వచించు cudaEvent_t hipEvent_t
#నిర్వచించు cudaEventDestroy hipEventDestroy
#నిర్వచించు cudaFree hipFree
#నిర్వచించు cudaFreeHost hipHostFree
#నిర్వచించు cudaGetDevice hipGetDevice
#నిర్వచించు cuda_Get_Device_లెక్క hip_Get_Device_లెక్క
#నిర్వచించు cudaGetDeviceProperties hipGetDeviceProperties
#నిర్వచించు cudaGetErrorString hipGetErrorString
#నిర్వచించు cuda_Get_చివరి_Error hip_Get_చివరి_Error
#నిర్వచించు cudaHostAlloc hipHostMalloc
#నిర్వచించు cudaHostAllocPortable hipHostMallocPortable
#నిర్వచించు cudaHostAllocMapped hipHostMallocMapped
#నిర్వచించు cuda_Host_Get_Device_సూచిక hip_Host_Get_Device_సూచిక
#నిర్వచించు cuda_Host_రిజిస్టరు hip_Host_రిజిస్టరు
#నిర్వచించు cuda_Host_రిజిస్టరు_Portable hip_Host_రిజిస్టరు_Portable
#నిర్వచించు cuda_Host_రిజిస్టరు_Read_Only hip_Host_రిజిస్టరు_Read_Only
#నిర్వచించు cudaHostUnregister hipHostUnregister
#నిర్వచించు cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#నిర్వచించు cudaLaunchHostFunc hipLaunchHostFunc
#నిర్వచించు cudaMalloc hipMalloc
#నిర్వచించు cudaMallocHost(ptr, కొలత) hipHostMalloc(ptr, కొలత, hipHostMallocDefault)
#నిర్వచించు cudaMallocManaged hipMallocManaged
#నిర్వచించు cudaMemAdvise hipMemAdvise
#నిర్వచించు cudaMemcpy hipMemcpy
#నిర్వచించు cudaMemcpyAsync hipMemcpyAsync
#నిర్వచించు cudaMemcpyPeerAsync hipMemcpyPeerAsync
#నిర్వచించు cudaMemcpy2DAsync hipMemcpy2DAsync
#నిర్వచించు cuda_Memcpy_Device_గమ్యం_Device hip_Memcpy_Device_గమ్యం_Device
#నిర్వచించు cuda_Memcpy_Device_గమ్యం_Host hip_Memcpy_Device_గమ్యం_Host
#నిర్వచించు cuda_Memcpy_Host_గమ్యం_Device hip_Memcpy_Host_గమ్యం_Device
#నిర్వచించు cuda_Memcpy_రకం hip_Memcpy_రకం
#నిర్వచించు cudaMemset hipMemset
#నిర్వచించు cudaMemsetAsync hipMemsetAsync
#నిర్వచించు cudaMemGetInfo hipMemGetInfo
#నిర్వచించు cuda_Occupancy_Max_Potential_ఖండం_కొలత hip_Occupancy_Max_Potential_ఖండం_కొలత
#నిర్వచించు cuda_సమితి_Device hip_సమితి_Device
#నిర్వచించు cuDeviceGet hipDeviceGet
#నిర్వచించు CUdevice hipDevice_t
#నిర్వచించు CUdeviceptr hipDeviceptr_t
#నిర్వచించు cuMemUnmap hipMemUnmap
#నిర్వచించు CUmemAccessDesc hipMemAccessDesc
#నిర్వచించు cu_Mem_చిరునామా_Free hip_Mem_చిరునామా_Free
#నిర్వచించు cuMemRelease hipMemRelease
#నిర్వచించు CUmem_Generic_కేటాయింపు_Handle hip_Mem_Generic_కేటాయింపు_Handle_t
#నిర్వచించు cuMemCreate hipMemCreate
#నిర్వచించు cu_Mem_చిరునామా_Reserve hip_Mem_చిరునామా_Reserve
#నిర్వచించు cuMemMap hipMemMap
#నిర్వచించు cu_Mem_సమితి_Access hip_Mem_సమితి_Access
#నిర్వచించు cu_Mem_Get_కేటాయింపు_Granularity hip_Mem_Get_కేటాయింపు_Granularity
#నిర్వచించు CUmem_కేటాయింపు_Prop hip_Mem_కేటాయింపు_Prop
#నిర్వచించు cuDeviceGetAttribute hipDeviceGetAttribute
#నిర్వచించు cuda_ప్రవాహం_Create_With_Flags hip_ప్రవాహం_Create_With_Flags
#నిర్వచించు cuda_ప్రవాహం_Destroy hip_ప్రవాహం_Destroy
#నిర్వచించు cuda_ప్రవాహం_Fire_And_Forget hip_ప్రవాహం_Fire_And_Forget
#నిర్వచించు cuda_ప్రవాహం_Non_Blocking hip_ప్రవాహం_Non_Blocking
#నిర్వచించు cuda_ప్రవాహం_చొప్పున_Thread hip_ప్రవాహం_చొప్పున_Thread
#నిర్వచించు cuda_ప్రవాహం_Synchronize hip_ప్రవాహం_Synchronize
#నిర్వచించు cuda_ప్రవాహం_Wait_Event hip_ప్రవాహం_Wait_Event
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_t hip_లేఖాచిత్రం_Exec_t
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_t hip_లేఖాచిత్రం_కణుపు_t
#నిర్వచించు cuda_Kernel_కణుపు_Params hip_Kernel_కణుపు_Params
#నిర్వచించు cuda_Kernel_కణుపు_Params hip_Kernel_కణుపు_Params
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Destroy hip_లేఖాచిత్రం_Exec_Destroy
#నిర్వచించు cuda_లేఖాచిత్రం_Launch hip_లేఖాచిత్రం_Launch
#నిర్వచించు cuda_Error_లేఖాచిత్రం_Exec_Update_Failure hip_Error_లేఖాచిత్రం_Exec_Update_Failure
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Update_ఫలితం hip_లేఖాచిత్రం_Exec_Update_ఫలితం
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Type hip_లేఖాచిత్రం_కణుపు_Type
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Type_Kernel hip_లేఖాచిత్రం_కణుపు_Type_Kernel
#నిర్వచించు cuda_లేఖాచిత్రం_Instantiate hip_లేఖాచిత్రం_Instantiate
#నిర్వచించు cuda_ప్రవాహం_చివర_Capture hip_ప్రవాహం_చివర_Capture
#నిర్వచించు cuda_లేఖాచిత్రం_Destroy hip_లేఖాచిత్రం_Destroy
#నిర్వచించు cuda_లేఖాచిత్రం_Kernel_కణుపు_సమితి_Params hip_లేఖాచిత్రం_Kernel_కణుపు_సమితి_Params
#నిర్వచించు cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#నిర్వచించు cuda_లేఖాచిత్రం_Kernel_కణుపు_Get_Params hip_లేఖాచిత్రం_Kernel_కణుపు_Get_Params
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Get_Type hip_లేఖాచిత్రం_కణుపు_Get_Type
#నిర్వచించు cuda_లేఖాచిత్రం_Get_Nodes hip_లేఖాచిత్రం_Get_Nodes
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Update hip_లేఖాచిత్రం_Exec_Update
#నిర్వచించు cuda_ప్రవాహం_Capture_బహుళకం_Relaxed hip_ప్రవాహం_Capture_బహుళకం_Relaxed
#నిర్వచించు cuda_ప్రవాహం_Begin_Capture hip_ప్రవాహం_Begin_Capture
#నిర్వచించు cuda_లేఖాచిత్రం_t hip_లేఖాచిత్రం_t
#నిర్వచించు cuda_ప్రవాహం_t hip_ప్రవాహం_t
#నిర్వచించు cudaSuccess hipSuccess
#నిర్వచించు cuda_Occupancy_Max_చురుకైన_ఖండాలు_చొప్పున_Multiprocessor hip_Occupancy_Max_చురుకైన_ఖండాలు_చొప్పున_Multiprocessor
#నిర్వచించు cuda_Func_సమితి_Attribute hip_Func_సమితి_Attribute
#నిర్వచించు cuda_Func_Attribute_Max_Dynamic_Shared_జ్ఞాపకం_కొలత hip_Func_Attribute_Max_Dynamic_Shared_జ్ఞాపకం_కొలత
#నిర్వచించు __trap() చేయి { abort(); __builtin_unreachable(); } అయితే(0)
#నిర్వచించు CUBLAS_స్థితి_SUCCESS HIPBLAS_స్థితి_SUCCESS
#నిర్వచించు CUBLAS_స్థితి_NOT_INITIALIZED HIPBLAS_స్థితి_NOT_INITIALIZED
#నిర్వచించు CUBLAS_స్థితి_ALLOC_FAILED HIPBLAS_స్థితి_ALLOC_FAILED
#నిర్వచించు CUBLAS_స్థితి_INVALID_విలువ HIPBLAS_స్థితి_INVALID_విలువ
#నిర్వచించు CUBLAS_స్థితి_ARCH_MISMATCH HIPBLAS_స్థితి_ARCH_MISMATCH
#నిర్వచించు CUBLAS_స్థితి_MAPPING_ERROR HIPBLAS_స్థితి_MAPPING_ERROR
#నిర్వచించు CUBLAS_స్థితి_EXECUTION_FAILED HIPBLAS_స్థితి_EXECUTION_FAILED
#నిర్వచించు CUBLAS_స్థితి_అంతర్గత_ERROR HIPBLAS_స్థితి_అంతర్గత_ERROR
#నిర్వచించు CUBLAS_స్థితి_NOT_SUPPORTED HIPBLAS_స్థితి_NOT_SUPPORTED

#ఒకవేళ HIP_సంస్కరణ >= 60500000
#నిర్వచించు CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#నిర్వచించు CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#నిర్వచించు CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#నిర్వచించు cublasComputeType_t hipblasComputeType_t
#నిర్వచించు cuda_దత్తాంశం_Type_t hip_దత్తాంశం_Type
#లేకపోతే
#నిర్వచించు CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#నిర్వచించు CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#నిర్వచించు CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#నిర్వచించు cublasComputeType_t hipblasDatatype_t
#నిర్వచించు cuda_దత్తాంశం_Type_t hipblasDatatype_t
#అయితే_ముగింపు // HIP_VERSION >= 6050000

#ఒకవేళ !defined(__HIP_PLATFORM_AMD__)
#దోషం "The HIP backend supports only AMD targets"
#అయితే_ముగింపు // !defined(__HIP_PLATFORM_AMD__)

#నిర్వచించు __CUDA_ARCH__ 1300

#ఒకవేళ defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#నిర్వచించు GCN5
#అయితే_ముగింపు // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#ఒకవేళ defined(__gfx803__)
#నిర్వచించు GCN4
#అయితే_ముగింపు // defined(__gfx803__)

#ఒకవేళ defined(GCN5) || defined(GCN4)
#నిర్వచించు GCN
#అయితే_ముగింపు // defined(GCN5) || defined(GCN4)

#ఒకవేళ defined(__gfx950__)
#నిర్వచించు CDNA4
#అయితే_ముగింపు // defined(__gfx950__)

#ఒకవేళ defined(__gfx942__)
#నిర్వచించు CDNA3
#అయితే_ముగింపు // defined(__gfx942__)

#ఒకవేళ defined(__gfx90a__)
#నిర్వచించు CDNA2
#అయితే_ముగింపు // defined(__gfx90a__)

#ఒకవేళ defined(__gfx908__)
#నిర్వచించు CDNA1
#అయితే_ముగింపు // defined(__gfx908__)

#ఒకవేళ defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#నిర్వచించు CDNA // For the entire family
#అయితే_ముగింపు // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#ఒకవేళ defined(__GFX12__)
#నిర్వచించు RDNA4
#అయితే_ముగింపు // defined(__GFX12__)

#ఒకవేళ defined(__GFX11__)
#నిర్వచించు RDNA3
#అయితే_ముగింపు // defined(__GFX11__)

#ఒకవేళ defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#నిర్వచించు RDNA3_5
#అయితే_ముగింపు // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#ఒకవేళ defined(RDNA3) && !defined(RDNA3_5)
#నిర్వచించు RDNA3_0
#అయితే_ముగింపు // defined(RDNA3) && !defined(RDNA3_5)

#ఒకవేళ defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#నిర్వచించు RDNA2
#అయితే_ముగింపు

#ఒకవేళ defined(__gfx1010__) || defined(__gfx1012__)
#నిర్వచించు RDNA1
#అయితే_ముగింపు // defined(__gfx1010__) || defined(__gfx1012__)

#ఒకవేళ defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#నిర్వచించు RDNA // For the entire family
#అయితే_ముగింపు // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#నిర్వచితం_కాకపోతే __has_builtin
    #నిర్వచించు __has_builtin(x) 0
#అయితే_ముగింపు

రకనిర్వచనం __hip_bfloat16 nv_bfloat16;
రకనిర్వచనం __hip_bfloat162 nv_bfloat162;

#ఒకవేళ HIP_సంస్కరణ >= 60300000
#అన్తర్భావ <hip/hip_fp8.h>
రకనిర్వచనం __hip_fp8_e4m3 __nv_fp8_e4m3;
#నిర్వచించు FP8_అందుబాటు
#అయితే_ముగింపు // HIP_VERSION >= 60300000

రకనిర్వచనం int8_t int8x4_t __attribute__((ext_సదిశ_type(4)));
రకనిర్వచనం uint8_t uint8x4_t __attribute__((ext_సదిశ_type(4)));
స్థిర __device__ __forceinline__ పూర్ణాంకం __vsubss4(స్థిరరాశి పూర్ణాంకం a, స్థిరరాశి పూర్ణాంకం b) {
    స్థిరరాశి int8x4_t va = reinterpret_cast<స్థిరరాశి int8x4_t&>(a);
    స్థిరరాశి int8x4_t vb = reinterpret_cast<స్థిరరాశి int8x4_t&>(b);
#ఒకవేళ __has_builtin(__builtin_elementwise_sub_sat)
    స్థిరరాశి int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    తిరిగి reinterpret_cast<స్థిరరాశి పూర్ణాంకం &>(c);
#లేకపోతే
    int8x4_t c;
    int16_t tmp;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        ఒకవేళ(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        ఒకవేళ(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    తిరిగి reinterpret_cast<పూర్ణాంకం &>(c);
#అయితే_ముగింపు // __has_builtin(__builtin_elementwise_sub_sat)
}

స్థిర __device__ __forceinline__ పూర్ణాంకం __vsub4(స్థిరరాశి పూర్ణాంకం a, స్థిరరాశి పూర్ణాంకం b) {
    // do some small modifications to a and b to make the subtraction not underflow
    స్థిరరాశి గుర్తులేని పూర్ణాంకం a_large = a | 0x80808080;
    స్థిరరాశి గుర్తులేని పూర్ణాంకం b_small = b & 0x7f7f7f7f;
    స్థిరరాశి గుర్తులేని పూర్ణాంకం ఫలితం_దిగువ_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    స్థిరరాశి గుర్తులేని పూర్ణాంకం should_flip_ఎగువ_1bit = (a ^ ~b) & 0x80808080;

    తిరిగి ఫలితం_దిగువ_7bits ^ should_flip_ఎగువ_1bit;
}

స్థిర __device__ __forceinline__ గుర్తులేని పూర్ణాంకం __vcmpeq4(గుర్తులేని పూర్ణాంకం a, గుర్తులేని పూర్ణాంకం b) {
    స్థిరరాశి uint8x4_t& va = reinterpret_cast<స్థిరరాశి uint8x4_t&>(a);
    స్థిరరాశి uint8x4_t& vb = reinterpret_cast<స్థిరరాశి uint8x4_t&>(b);
    గుర్తులేని పూర్ణాంకం c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    తిరిగి c;
}

స్థిర __device__ __forceinline__ గుర్తులేని పూర్ణాంకం __vcmpne4(గుర్తులేని పూర్ణాంకం a, గుర్తులేని పూర్ణాంకం b) {
    స్థిరరాశి గుర్తులేని పూర్ణాంకం x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    స్థిరరాశి గుర్తులేని పూర్ణాంకం ne_దిగువ_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    స్థిరరాశి గుర్తులేని పూర్ణాంకం ne_ఎగువ_1bit = x & 0x80808080;
    స్థిరరాశి గుర్తులేని పూర్ణాంకం ne_any_బిట్టు = ne_దిగువ_7bits | ne_ఎగువ_1bit;

    తిరిగి (ne_any_బిట్టు >> 7) * 0xff;
}
