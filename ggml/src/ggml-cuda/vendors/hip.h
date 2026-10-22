#ਹਦਾਇਤ once

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ਅਨ੍ਤਰ੍ਭਾਵ <hip/hip_runtime.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hipblas/hipblas.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hip/hip_fp16.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hip/hip_bf16.h>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_NCCL
#ਅਨ੍ਤਰ੍ਭਾਵ <rccl/rccl.h>
#ਜੇ_ਅੰਤ // GGML_USE_NCCL


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_OP_N HIPBLAS_OP_N
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_OP_T HIPBLAS_OP_T
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_SUCCESS HIPBLAS_ਸਥਿਤੀ_SUCCESS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_TF32_TENSOR_OP_MATH 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDA_R_16F  HIPBLAS_R_16F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDA_R_16BF HIPBLAS_R_16B
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUDA_R_32F  HIPBLAS_R_32F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_SIDE_ਸੱਜਾ HIPBLAS_SIDE_ਸੱਜਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_FILL_ਬਹੁਲਕ_ਉੱਤਲਾ HIPBLAS_FILL_ਬਹੁਲਕ_ਉੱਤਲਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_DEVICE_ATTRIBUTE_VIRTUAL_ਸਮ੍ਰਿਤੀ_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_ਸਮ੍ਰਿਤੀ_Management_Supported
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_ਵੰਡ_Granularity_Recommended
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_MEM_ਵੰਡ_TYPE_PINNED hip_Mem_ਵੰਡ_Type_Pinned
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CU_CHECK(fn) {hipError_t err = fn; ਜੇ(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __shfl_sync(ਨਕਾਬ, var, lane_ਨਕਾਬ, ਚੌੜਾਈ) __shfl(var, lane_ਨਕਾਬ, ਚੌੜਾਈ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __shfl_up_sync(ਨਕਾਬ, var, lane_ਨਕਾਬ, ਚੌੜਾਈ) __shfl_up(var, lane_ਨਕਾਬ, ਚੌੜਾਈ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __shfl_xor_sync(ਨਕਾਬ, var, lane_ਨਕਾਬ, ਚੌੜਾਈ) __shfl_xor(var, lane_ਨਕਾਬ, ਚੌੜਾਈ)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __all_sync(ਨਕਾਬ, var) __all(var)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __any_sync(ਨਕਾਬ, var) __any(var)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasStrsmBatched hipblasStrsmBatched
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasCreate hipblasCreate
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasDestroy hipblasDestroy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasGemmEx hipblasGemmEx
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasGemmBatchedEx hipblasGemmBatchedEx
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasHandle_t hipblasHandle_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublas_ਸੈੱਟ_Math_ਬਹੁਲਕ(handle, ਬਹੁਲਕ) CUBLAS_ਸਥਿਤੀ_SUCCESS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublas_ਸੈੱਟ_ਧਾਰਾ hipblas_ਸੈੱਟ_ਧਾਰਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasSgemm hipblasSgemm
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasSgemmBatched hipblasSgemmBatched
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasSgemmStridedBatched hipblasSgemmStridedBatched
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublas_ਸਥਿਤੀ_t hipblas_ਸਥਿਤੀ_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasOperation_t hipblasOperation_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceGetAttribute hipDeviceGetAttribute
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceProp hipDeviceProp_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaDeviceSynchronize hipDeviceSynchronize
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaError_t hipError_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Error_ਸਮ੍ਰਿਤੀ_ਵੰਡ hip_Error_ਬਾਹਰਲਾ_ਦਾ_ਸਮ੍ਰਿਤੀ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaEventCreateWithFlags hipEventCreateWithFlags
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaEventDisableTiming hipEventDisableTiming
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Event_ਰਿਕਾਰਡ hip_Event_ਰਿਕਾਰਡ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaEventSynchronize hipEventSynchronize
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaEvent_t hipEvent_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaEventDestroy hipEventDestroy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaFree hipFree
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaFreeHost hipHostFree
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaGetDevice hipGetDevice
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Get_Device_ਗਿਣਤੀ hip_Get_Device_ਗਿਣਤੀ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaGetDeviceProperties hipGetDeviceProperties
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaGetErrorString hipGetErrorString
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Get_ਆਖਰੀ_Error hip_Get_ਆਖਰੀ_Error
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaHostAlloc hipHostMalloc
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaHostAllocPortable hipHostMallocPortable
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaHostAllocMapped hipHostMallocMapped
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Host_Get_Device_ਸੂਚਕ hip_Host_Get_Device_ਸੂਚਕ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Host_ਰਜਿਸਟਰ hip_Host_ਰਜਿਸਟਰ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Host_ਰਜਿਸਟਰ_Portable hip_Host_ਰਜਿਸਟਰ_Portable
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Host_ਰਜਿਸਟਰ_Read_Only hip_Host_ਰਜਿਸਟਰ_Read_Only
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaHostUnregister hipHostUnregister
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaLaunchHostFunc hipLaunchHostFunc
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMalloc hipMalloc
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMallocHost(ptr, ਮਾਪ) hipHostMalloc(ptr, ਮਾਪ, hipHostMallocDefault)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMallocManaged hipMallocManaged
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemAdvise hipMemAdvise
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemcpy hipMemcpy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemcpyAsync hipMemcpyAsync
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemcpyPeerAsync hipMemcpyPeerAsync
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemcpy2DAsync hipMemcpy2DAsync
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Memcpy_Device_ਮੰਜ਼ਿਲ_Device hip_Memcpy_Device_ਮੰਜ਼ਿਲ_Device
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Memcpy_Device_ਮੰਜ਼ਿਲ_Host hip_Memcpy_Device_ਮੰਜ਼ਿਲ_Host
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Memcpy_Host_ਮੰਜ਼ਿਲ_Device hip_Memcpy_Host_ਮੰਜ਼ਿਲ_Device
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Memcpy_ਕਿਸਮ hip_Memcpy_ਕਿਸਮ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemset hipMemset
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemsetAsync hipMemsetAsync
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaMemGetInfo hipMemGetInfo
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Occupancy_Max_Potential_ਖੰਡ_ਮਾਪ hip_Occupancy_Max_Potential_ਖੰਡ_ਮਾਪ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਸੈੱਟ_Device hip_ਸੈੱਟ_Device
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuDeviceGet hipDeviceGet
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUdevice hipDevice_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUdeviceptr hipDeviceptr_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuMemUnmap hipMemUnmap
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUmemAccessDesc hipMemAccessDesc
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cu_Mem_ਪਤਾ_Free hip_Mem_ਪਤਾ_Free
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuMemRelease hipMemRelease
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUmem_Generic_ਵੰਡ_Handle hip_Mem_Generic_ਵੰਡ_Handle_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuMemCreate hipMemCreate
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cu_Mem_ਪਤਾ_Reserve hip_Mem_ਪਤਾ_Reserve
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuMemMap hipMemMap
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cu_Mem_ਸੈੱਟ_Access hip_Mem_ਸੈੱਟ_Access
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cu_Mem_Get_ਵੰਡ_Granularity hip_Mem_Get_ਵੰਡ_Granularity
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUmem_ਵੰਡ_Prop hip_Mem_ਵੰਡ_Prop
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuDeviceGetAttribute hipDeviceGetAttribute
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Create_With_Flags hip_ਧਾਰਾ_Create_With_Flags
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Destroy hip_ਧਾਰਾ_Destroy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Fire_And_Forget hip_ਧਾਰਾ_Fire_And_Forget
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Non_Blocking hip_ਧਾਰਾ_Non_Blocking
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_ਪ੍ਰਤੀ_Thread hip_ਧਾਰਾ_ਪ੍ਰਤੀ_Thread
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Synchronize hip_ਧਾਰਾ_Synchronize
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Wait_Event hip_ਧਾਰਾ_Wait_Event
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Exec_t hip_ਗ੍ਰਾਫ_Exec_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_ਗੰਢ_t hip_ਗ੍ਰਾਫ_ਗੰਢ_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Kernel_ਗੰਢ_Params hip_Kernel_ਗੰਢ_Params
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Kernel_ਗੰਢ_Params hip_Kernel_ਗੰਢ_Params
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Exec_Destroy hip_ਗ੍ਰਾਫ_Exec_Destroy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Launch hip_ਗ੍ਰਾਫ_Launch
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Error_ਗ੍ਰਾਫ_Exec_Update_Failure hip_Error_ਗ੍ਰਾਫ_Exec_Update_Failure
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Exec_Update_ਨਤੀਜਾ hip_ਗ੍ਰਾਫ_Exec_Update_ਨਤੀਜਾ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_ਗੰਢ_Type hip_ਗ੍ਰਾਫ_ਗੰਢ_Type
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_ਗੰਢ_Type_Kernel hip_ਗ੍ਰਾਫ_ਗੰਢ_Type_Kernel
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Instantiate hip_ਗ੍ਰਾਫ_Instantiate
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_ਸਿਰਾ_Capture hip_ਧਾਰਾ_ਸਿਰਾ_Capture
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Destroy hip_ਗ੍ਰਾਫ_Destroy
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Kernel_ਗੰਢ_ਸੈੱਟ_Params hip_ਗ੍ਰਾਫ_Kernel_ਗੰਢ_ਸੈੱਟ_Params
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Kernel_ਗੰਢ_Get_Params hip_ਗ੍ਰਾਫ_Kernel_ਗੰਢ_Get_Params
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_ਗੰਢ_Get_Type hip_ਗ੍ਰਾਫ_ਗੰਢ_Get_Type
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Get_Nodes hip_ਗ੍ਰਾਫ_Get_Nodes
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_Exec_Update hip_ਗ੍ਰਾਫ_Exec_Update
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Capture_ਬਹੁਲਕ_Relaxed hip_ਧਾਰਾ_Capture_ਬਹੁਲਕ_Relaxed
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_Begin_Capture hip_ਧਾਰਾ_Begin_Capture
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਗ੍ਰਾਫ_t hip_ਗ੍ਰਾਫ_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਧਾਰਾ_t hip_ਧਾਰਾ_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cudaSuccess hipSuccess
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Occupancy_Max_ਸਰਗਰਮ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_Multiprocessor hip_Occupancy_Max_ਸਰਗਰਮ_ਖੰਡਾਂ_ਪ੍ਰਤੀ_Multiprocessor
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Func_ਸੈੱਟ_Attribute hip_Func_ਸੈੱਟ_Attribute
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_Func_Attribute_Max_Dynamic_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ hip_Func_Attribute_Max_Dynamic_Shared_ਸਮ੍ਰਿਤੀ_ਮਾਪ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __trap() ਕਰੋ { abort(); __builtin_unreachable(); } ਜਦੋਂ(0)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_SUCCESS HIPBLAS_ਸਥਿਤੀ_SUCCESS
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_NOT_INITIALIZED HIPBLAS_ਸਥਿਤੀ_NOT_INITIALIZED
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_ALLOC_FAILED HIPBLAS_ਸਥਿਤੀ_ALLOC_FAILED
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_INVALID_ਮੁੱਲ HIPBLAS_ਸਥਿਤੀ_INVALID_ਮੁੱਲ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_ARCH_MISMATCH HIPBLAS_ਸਥਿਤੀ_ARCH_MISMATCH
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_MAPPING_ERROR HIPBLAS_ਸਥਿਤੀ_MAPPING_ERROR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_EXECUTION_FAILED HIPBLAS_ਸਥਿਤੀ_EXECUTION_FAILED
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERROR HIPBLAS_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERROR
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_ਸਥਿਤੀ_NOT_SUPPORTED HIPBLAS_ਸਥਿਤੀ_NOT_SUPPORTED

#ਜੇ HIP_ਸੰਸਕਰਣ >= 60500000
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasComputeType_t hipblasComputeType_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਅੰਕੜੇ_Type_t hip_ਅੰਕੜੇ_Type
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cublasComputeType_t hipblasDatatype_t
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ cuda_ਅੰਕੜੇ_Type_t hipblasDatatype_t
#ਜੇ_ਅੰਤ // HIP_VERSION >= 6050000

#ਜੇ !defined(__HIP_PLATFORM_AMD__)
#ਗਲਤੀ "The HIP backend supports only AMD targets"
#ਜੇ_ਅੰਤ // !defined(__HIP_PLATFORM_AMD__)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __CUDA_ARCH__ 1300

#ਜੇ defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GCN5
#ਜੇ_ਅੰਤ // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#ਜੇ defined(__gfx803__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GCN4
#ਜੇ_ਅੰਤ // defined(__gfx803__)

#ਜੇ defined(GCN5) || defined(GCN4)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GCN
#ਜੇ_ਅੰਤ // defined(GCN5) || defined(GCN4)

#ਜੇ defined(__gfx950__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CDNA4
#ਜੇ_ਅੰਤ // defined(__gfx950__)

#ਜੇ defined(__gfx942__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CDNA3
#ਜੇ_ਅੰਤ // defined(__gfx942__)

#ਜੇ defined(__gfx90a__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CDNA2
#ਜੇ_ਅੰਤ // defined(__gfx90a__)

#ਜੇ defined(__gfx908__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CDNA1
#ਜੇ_ਅੰਤ // defined(__gfx908__)

#ਜੇ defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CDNA // For the entire family
#ਜੇ_ਅੰਤ // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#ਜੇ defined(__GFX12__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA4
#ਜੇ_ਅੰਤ // defined(__GFX12__)

#ਜੇ defined(__GFX11__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA3
#ਜੇ_ਅੰਤ // defined(__GFX11__)

#ਜੇ defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA3_5
#ਜੇ_ਅੰਤ // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#ਜੇ defined(RDNA3) && !defined(RDNA3_5)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA3_0
#ਜੇ_ਅੰਤ // defined(RDNA3) && !defined(RDNA3_5)

#ਜੇ defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA2
#ਜੇ_ਅੰਤ

#ਜੇ defined(__gfx1010__) || defined(__gfx1012__)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA1
#ਜੇ_ਅੰਤ // defined(__gfx1010__) || defined(__gfx1012__)

#ਜੇ defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ RDNA // For the entire family
#ਜੇ_ਅੰਤ // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ __has_builtin
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __has_builtin(x) 0
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ __hip_bfloat16 nv_bfloat16;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ __hip_bfloat162 nv_bfloat162;

#ਜੇ HIP_ਸੰਸਕਰਣ >= 60300000
#ਅਨ੍ਤਰ੍ਭਾਵ <hip/hip_fp8.h>
ਕਿਸਮਪਰਿਭਾਸ਼ਾ __hip_fp8_e4m3 __nv_fp8_e4m3;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FP8_ਉਪਲਬਧ
#ਜੇ_ਅੰਤ // HIP_VERSION >= 60300000

ਕਿਸਮਪਰਿਭਾਸ਼ਾ int8_t int8x4_t __attribute__((ext_ਸਦਿਸ਼_type(4)));
ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint8_t uint8x4_t __attribute__((ext_ਸਦਿਸ਼_type(4)));
ਸਥਿਰ __device__ __forceinline__ ਪੂਰਨ __vsubss4(ਅਚਲ ਪੂਰਨ a, ਅਚਲ ਪੂਰਨ b) {
    ਅਚਲ int8x4_t va = reinterpret_cast<ਅਚਲ int8x4_t&>(a);
    ਅਚਲ int8x4_t vb = reinterpret_cast<ਅਚਲ int8x4_t&>(b);
#ਜੇ __has_builtin(__builtin_elementwise_sub_sat)
    ਅਚਲ int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    ਵਾਪਸ reinterpret_cast<ਅਚਲ ਪੂਰਨ &>(c);
#ਨਹੀਂਤਾਂ
    int8x4_t c;
    int16_t tmp;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        ਜੇ(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        ਜੇ(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    ਵਾਪਸ reinterpret_cast<ਪੂਰਨ &>(c);
#ਜੇ_ਅੰਤ // __has_builtin(__builtin_elementwise_sub_sat)
}

ਸਥਿਰ __device__ __forceinline__ ਪੂਰਨ __vsub4(ਅਚਲ ਪੂਰਨ a, ਅਚਲ ਪੂਰਨ b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ a_large = a | 0x80808080;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ b_small = b & 0x7f7f7f7f;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਨਤੀਜਾ_ਨੀਵਾਂ_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ should_flip_ਉੱਚਾ_1bit = (a ^ ~b) & 0x80808080;

    ਵਾਪਸ ਨਤੀਜਾ_ਨੀਵਾਂ_7bits ^ should_flip_ਉੱਚਾ_1bit;
}

ਸਥਿਰ __device__ __forceinline__ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ __vcmpeq4(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ a, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ b) {
    ਅਚਲ uint8x4_t& va = reinterpret_cast<ਅਚਲ uint8x4_t&>(a);
    ਅਚਲ uint8x4_t& vb = reinterpret_cast<ਅਚਲ uint8x4_t&>(b);
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    ਵਾਪਸ c;
}

ਸਥਿਰ __device__ __forceinline__ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ __vcmpne4(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ a, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ b) {
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ne_ਨੀਵਾਂ_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ne_ਉੱਚਾ_1bit = x & 0x80808080;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ne_any_ਦੋਆਧਾਰੀ_ਅੰਕ = ne_ਨੀਵਾਂ_7bits | ne_ਉੱਚਾ_1bit;

    ਵਾਪਸ (ne_any_ਦੋਆਧਾਰੀ_ਅੰਕ >> 7) * 0xff;
}
