#ಸೂಚನೆ once

#ವ್ಯಾಖ್ಯಾನಿಸು HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ಅನ್ತರ್ಭಾವ <hip/hip_runtime.h>
#ಅನ್ತರ್ಭಾವ <hipblas/hipblas.h>
#ಅನ್ತರ್ಭಾವ <hip/hip_fp16.h>
#ಅನ್ತರ್ಭಾವ <hip/hip_bf16.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_NCCL
#ಅನ್ತರ್ಭಾವ <rccl/rccl.h>
#ಆದರೆ_ಅಂತ್ಯ // GGML_USE_NCCL


#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_OP_N HIPBLAS_OP_N
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_OP_T HIPBLAS_OP_T
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_SUCCESS HIPBLAS_ಸ್ಥಿತಿ_SUCCESS
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_TF32_TENSOR_OP_MATH 0
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_16F  HIPBLAS_R_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_16BF HIPBLAS_R_16B
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_32F  HIPBLAS_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_SIDE_ಬಲ HIPBLAS_SIDE_ಬಲ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_FILL_ಬಹುಲಕ_ಊರ್ಧ್ವ HIPBLAS_FILL_ಬಹುಲಕ_ಊರ್ಧ್ವ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#ವ್ಯಾಖ್ಯಾನಿಸು CU_DEVICE_ATTRIBUTE_VIRTUAL_ಸ್ಮರಣೆ_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_ಸ್ಮರಣೆ_Management_Supported
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_ಹಂಚಿಕೆ_Granularity_Recommended
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ಹಂಚಿಕೆ_TYPE_PINNED hip_Mem_ಹಂಚಿಕೆ_Type_Pinned
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#ವ್ಯಾಖ್ಯಾನಿಸು CU_CHECK(fn) {hipError_t err = fn; ಆದರೆ(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#ವ್ಯಾಖ್ಯಾನಿಸು __shfl_sync(ಮುಖವಾಡ, var, lane_ಮುಖವಾಡ, ಅಗಲ) __shfl(var, lane_ಮುಖವಾಡ, ಅಗಲ)
#ವ್ಯಾಖ್ಯಾನಿಸು __shfl_up_sync(ಮುಖವಾಡ, var, lane_ಮುಖವಾಡ, ಅಗಲ) __shfl_up(var, lane_ಮುಖವಾಡ, ಅಗಲ)
#ವ್ಯಾಖ್ಯಾನಿಸು __shfl_xor_sync(ಮುಖವಾಡ, var, lane_ಮುಖವಾಡ, ಅಗಲ) __shfl_xor(var, lane_ಮುಖವಾಡ, ಅಗಲ)
#ವ್ಯಾಖ್ಯಾನಿಸು __all_sync(ಮುಖವಾಡ, var) __all(var)
#ವ್ಯಾಖ್ಯಾನಿಸು __any_sync(ಮುಖವಾಡ, var) __any(var)
#ವ್ಯಾಖ್ಯಾನಿಸು cublasStrsmBatched hipblasStrsmBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublasCreate hipblasCreate
#ವ್ಯಾಖ್ಯಾನಿಸು cublasDestroy hipblasDestroy
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmEx hipblasGemmEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmBatchedEx hipblasGemmBatchedEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasHandle_t hipblasHandle_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಗಣ_Math_ಬಹುಲಕ(handle, ಬಹುಲಕ) CUBLAS_ಸ್ಥಿತಿ_SUCCESS
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಗಣ_ಹರಿವು hipblas_ಗಣ_ಹರಿವು
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemm hipblasSgemm
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemmBatched hipblasSgemmBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemmStridedBatched hipblasSgemmStridedBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಸ್ಥಿತಿ_t hipblas_ಸ್ಥಿತಿ_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublasOperation_t hipblasOperation_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceGetAttribute hipDeviceGetAttribute
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceProp hipDeviceProp_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceSynchronize hipDeviceSynchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cudaError_t hipError_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Error_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ hip_Error_ಹೊರಹೋಗು_ದ_ಸ್ಮರಣೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventCreateWithFlags hipEventCreateWithFlags
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventDisableTiming hipEventDisableTiming
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Event_ದಾಖಲೆ hip_Event_ದಾಖಲೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventSynchronize hipEventSynchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEvent_t hipEvent_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventDestroy hipEventDestroy
#ವ್ಯಾಖ್ಯಾನಿಸು cudaFree hipFree
#ವ್ಯಾಖ್ಯಾನಿಸು cudaFreeHost hipHostFree
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetDevice hipGetDevice
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Get_Device_ಎಣಿಕೆ hip_Get_Device_ಎಣಿಕೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetDeviceProperties hipGetDeviceProperties
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetErrorString hipGetErrorString
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Get_ಕೊನೆಯ_Error hip_Get_ಕೊನೆಯ_Error
#ವ್ಯಾಖ್ಯಾನಿಸು cudaHostAlloc hipHostMalloc
#ವ್ಯಾಖ್ಯಾನಿಸು cudaHostAllocPortable hipHostMallocPortable
#ವ್ಯಾಖ್ಯಾನಿಸು cudaHostAllocMapped hipHostMallocMapped
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_Get_Device_ಸೂಚಕ hip_Host_Get_Device_ಸೂಚಕ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್ hip_Host_ರಿಜಿಸ್ಟರ್
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್_Portable hip_Host_ರಿಜಿಸ್ಟರ್_Portable
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್_Read_Only hip_Host_ರಿಜಿಸ್ಟರ್_Read_Only
#ವ್ಯಾಖ್ಯಾನಿಸು cudaHostUnregister hipHostUnregister
#ವ್ಯಾಖ್ಯಾನಿಸು cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#ವ್ಯಾಖ್ಯಾನಿಸು cudaLaunchHostFunc hipLaunchHostFunc
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMalloc hipMalloc
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMallocHost(ptr, ಗಾತ್ರ_2) hipHostMalloc(ptr, ಗಾತ್ರ_2, hipHostMallocDefault)
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMallocManaged hipMallocManaged
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemAdvise hipMemAdvise
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpy hipMemcpy
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpyAsync hipMemcpyAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpyPeerAsync hipMemcpyPeerAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpy2DAsync hipMemcpy2DAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Device_ಗಮ್ಯ_Device hip_Memcpy_Device_ಗಮ್ಯ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Device_ಗಮ್ಯ_Host hip_Memcpy_Device_ಗಮ್ಯ_Host
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Host_ಗಮ್ಯ_Device hip_Memcpy_Host_ಗಮ್ಯ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_ಬಗೆ hip_Memcpy_ಬಗೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemset hipMemset
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemsetAsync hipMemsetAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemGetInfo hipMemGetInfo
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Occupancy_Max_Potential_ಖಂಡ_ಗಾತ್ರ hip_Occupancy_Max_Potential_ಖಂಡ_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಗಣ_Device hip_ಗಣ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuDeviceGet hipDeviceGet
#ವ್ಯಾಖ್ಯಾನಿಸು CUdevice hipDevice_t
#ವ್ಯಾಖ್ಯಾನಿಸು CUdeviceptr hipDeviceptr_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemUnmap hipMemUnmap
#ವ್ಯಾಖ್ಯಾನಿಸು CUmemAccessDesc hipMemAccessDesc
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ವಿಳಾಸ_Free hip_Mem_ವಿಳಾಸ_Free
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemRelease hipMemRelease
#ವ್ಯಾಖ್ಯಾನಿಸು CUmem_Generic_ಹಂಚಿಕೆ_Handle hip_Mem_Generic_ಹಂಚಿಕೆ_Handle_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemCreate hipMemCreate
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ವಿಳಾಸ_Reserve hip_Mem_ವಿಳಾಸ_Reserve
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemMap hipMemMap
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ಗಣ_Access hip_Mem_ಗಣ_Access
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_Get_ಹಂಚಿಕೆ_Granularity hip_Mem_Get_ಹಂಚಿಕೆ_Granularity
#ವ್ಯಾಖ್ಯಾನಿಸು CUmem_ಹಂಚಿಕೆ_Prop hip_Mem_ಹಂಚಿಕೆ_Prop
#ವ್ಯಾಖ್ಯಾನಿಸು cuDeviceGetAttribute hipDeviceGetAttribute
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Create_With_Flags hip_ಹರಿವು_Create_With_Flags
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Destroy hip_ಹರಿವು_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Fire_And_Forget hip_ಹರಿವು_Fire_And_Forget
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Non_Blocking hip_ಹರಿವು_Non_Blocking
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_ಪ್ರತಿ_Thread hip_ಹರಿವು_ಪ್ರತಿ_Thread
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Synchronize hip_ಹರಿವು_Synchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Wait_Event hip_ಹರಿವು_Wait_Event
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_t hip_ನಕ್ಷೆ_Exec_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_t hip_ನಕ್ಷೆ_ಗಂಟು_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Kernel_ಗಂಟು_Params hip_Kernel_ಗಂಟು_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Kernel_ಗಂಟು_Params hip_Kernel_ಗಂಟು_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Destroy hip_ನಕ್ಷೆ_Exec_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Launch hip_ನಕ್ಷೆ_Launch
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Error_ನಕ್ಷೆ_Exec_Update_Failure hip_Error_ನಕ್ಷೆ_Exec_Update_Failure
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Update_ಫಲಿತಾಂಶ hip_ನಕ್ಷೆ_Exec_Update_ಫಲಿತಾಂಶ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Type hip_ನಕ್ಷೆ_ಗಂಟು_Type
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Type_Kernel hip_ನಕ್ಷೆ_ಗಂಟು_Type_Kernel
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Instantiate hip_ನಕ್ಷೆ_Instantiate
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_ಕೊನೆ_Capture hip_ಹರಿವು_ಕೊನೆ_Capture
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Destroy hip_ನಕ್ಷೆ_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Kernel_ಗಂಟು_ಗಣ_Params hip_ನಕ್ಷೆ_Kernel_ಗಂಟು_ಗಣ_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Kernel_ಗಂಟು_Get_Params hip_ನಕ್ಷೆ_Kernel_ಗಂಟು_Get_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Get_Type hip_ನಕ್ಷೆ_ಗಂಟು_Get_Type
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Get_Nodes hip_ನಕ್ಷೆ_Get_Nodes
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Update hip_ನಕ್ಷೆ_Exec_Update
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Capture_ಬಹುಲಕ_Relaxed hip_ಹರಿವು_Capture_ಬಹುಲಕ_Relaxed
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Begin_Capture hip_ಹರಿವು_Begin_Capture
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_t hip_ನಕ್ಷೆ_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_t hip_ಹರಿವು_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaSuccess hipSuccess
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Occupancy_Max_ಸಕ್ರಿಯ_ಖಂಡಗಳು_ಪ್ರತಿ_Multiprocessor hip_Occupancy_Max_ಸಕ್ರಿಯ_ಖಂಡಗಳು_ಪ್ರತಿ_Multiprocessor
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Func_ಗಣ_Attribute hip_Func_ಗಣ_Attribute
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Func_Attribute_Max_Dynamic_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ hip_Func_Attribute_Max_Dynamic_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು __trap() ಮಾಡು { abort(); __builtin_unreachable(); } ಆಗ(0)
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_SUCCESS HIPBLAS_ಸ್ಥಿತಿ_SUCCESS
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_NOT_INITIALIZED HIPBLAS_ಸ್ಥಿತಿ_NOT_INITIALIZED
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_ALLOC_FAILED HIPBLAS_ಸ್ಥಿತಿ_ALLOC_FAILED
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_INVALID_ಮೌಲ್ಯ HIPBLAS_ಸ್ಥಿತಿ_INVALID_ಮೌಲ್ಯ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_ARCH_MISMATCH HIPBLAS_ಸ್ಥಿತಿ_ARCH_MISMATCH
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_MAPPING_ERROR HIPBLAS_ಸ್ಥಿತಿ_MAPPING_ERROR
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_EXECUTION_FAILED HIPBLAS_ಸ್ಥಿತಿ_EXECUTION_FAILED
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERROR HIPBLAS_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERROR
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_NOT_SUPPORTED HIPBLAS_ಸ್ಥಿತಿ_NOT_SUPPORTED

#ಆದರೆ HIP_ಆವೃತ್ತಿ >= 60500000
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#ವ್ಯಾಖ್ಯಾನಿಸು cublasComputeType_t hipblasComputeType_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ದತ್ತಾಂಶ_Type_t hip_ದತ್ತಾಂಶ_Type
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು cublasComputeType_t hipblasDatatype_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ದತ್ತಾಂಶ_Type_t hipblasDatatype_t
#ಆದರೆ_ಅಂತ್ಯ // HIP_VERSION >= 6050000

#ಆದರೆ !defined(__HIP_PLATFORM_AMD__)
#ದೋಷ "The HIP backend supports only AMD targets"
#ಆದರೆ_ಅಂತ್ಯ // !defined(__HIP_PLATFORM_AMD__)

#ವ್ಯಾಖ್ಯಾನಿಸು __CUDA_ARCH__ 1300

#ಆದರೆ defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#ವ್ಯಾಖ್ಯಾನಿಸು GCN5
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#ಆದರೆ defined(__gfx803__)
#ವ್ಯಾಖ್ಯಾನಿಸು GCN4
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx803__)

#ಆದರೆ defined(GCN5) || defined(GCN4)
#ವ್ಯಾಖ್ಯಾನಿಸು GCN
#ಆದರೆ_ಅಂತ್ಯ // defined(GCN5) || defined(GCN4)

#ಆದರೆ defined(__gfx950__)
#ವ್ಯಾಖ್ಯಾನಿಸು CDNA4
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx950__)

#ಆದರೆ defined(__gfx942__)
#ವ್ಯಾಖ್ಯಾನಿಸು CDNA3
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx942__)

#ಆದರೆ defined(__gfx90a__)
#ವ್ಯಾಖ್ಯಾನಿಸು CDNA2
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx90a__)

#ಆದರೆ defined(__gfx908__)
#ವ್ಯಾಖ್ಯಾನಿಸು CDNA1
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx908__)

#ಆದರೆ defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#ವ್ಯಾಖ್ಯಾನಿಸು CDNA // For the entire family
#ಆದರೆ_ಅಂತ್ಯ // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#ಆದರೆ defined(__GFX12__)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA4
#ಆದರೆ_ಅಂತ್ಯ // defined(__GFX12__)

#ಆದರೆ defined(__GFX11__)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA3
#ಆದರೆ_ಅಂತ್ಯ // defined(__GFX11__)

#ಆದರೆ defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA3_5
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#ಆದರೆ defined(RDNA3) && !defined(RDNA3_5)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA3_0
#ಆದರೆ_ಅಂತ್ಯ // defined(RDNA3) && !defined(RDNA3_5)

#ಆದರೆ defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA2
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__gfx1010__) || defined(__gfx1012__)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA1
#ಆದರೆ_ಅಂತ್ಯ // defined(__gfx1010__) || defined(__gfx1012__)

#ಆದರೆ defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#ವ್ಯಾಖ್ಯಾನಿಸು RDNA // For the entire family
#ಆದರೆ_ಅಂತ್ಯ // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ __has_builtin
    #ವ್ಯಾಖ್ಯಾನಿಸು __has_builtin(x) 0
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __hip_bfloat16 nv_bfloat16;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __hip_bfloat162 nv_bfloat162;

#ಆದರೆ HIP_ಆವೃತ್ತಿ >= 60300000
#ಅನ್ತರ್ಭಾವ <hip/hip_fp8.h>
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __hip_fp8_e4m3 __nv_fp8_e4m3;
#ವ್ಯಾಖ್ಯಾನಿಸು FP8_ಲಭ್ಯ
#ಆದರೆ_ಅಂತ್ಯ // HIP_VERSION >= 60300000

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ int8_t int8x4_t __attribute__((ext_ಸದಿಶ_type(4)));
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint8_t uint8x4_t __attribute__((ext_ಸದಿಶ_type(4)));
ಸ್ಥಿರ __device__ __forceinline__ ಪೂರ್ಣಾಂಕ __vsubss4(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b) {
    ಸ್ಥಿರಾಂಕ int8x4_t va = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8x4_t&>(a);
    ಸ್ಥಿರಾಂಕ int8x4_t vb = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8x4_t&>(b);
#ಆದರೆ __has_builtin(__builtin_elementwise_sub_sat)
    ಸ್ಥಿರಾಂಕ int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    ಹಿಂತಿರುಗಿ reinterpret_cast<ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &>(c);
#ಇಲ್ಲದಿದ್ದರೆ
    int8x4_t c;
    int16_t tmp;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        ಆದರೆ(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        ಆದರೆ(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    ಹಿಂತಿರುಗಿ reinterpret_cast<ಪೂರ್ಣಾಂಕ &>(c);
#ಆದರೆ_ಅಂತ್ಯ // __has_builtin(__builtin_elementwise_sub_sat)
}

ಸ್ಥಿರ __device__ __forceinline__ ಪೂರ್ಣಾಂಕ __vsub4(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ a_large = a | 0x80808080;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ b_small = b & 0x7f7f7f7f;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಫಲಿತಾಂಶ_ಕೆಳಗಿನ_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ should_flip_ಮೇಲಿನ_1bit = (a ^ ~b) & 0x80808080;

    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ_ಕೆಳಗಿನ_7bits ^ should_flip_ಮೇಲಿನ_1bit;
}

ಸ್ಥಿರ __device__ __forceinline__ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ __vcmpeq4(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ a, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ b) {
    ಸ್ಥಿರಾಂಕ uint8x4_t& va = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8x4_t&>(a);
    ಸ್ಥಿರಾಂಕ uint8x4_t& vb = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8x4_t&>(b);
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    ಹಿಂತಿರುಗಿ c;
}

ಸ್ಥಿರ __device__ __forceinline__ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ __vcmpne4(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ a, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ b) {
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ne_ಕೆಳಗಿನ_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ne_ಮೇಲಿನ_1bit = x & 0x80808080;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ne_any_ಬಿಟ್ = ne_ಕೆಳಗಿನ_7bits | ne_ಮೇಲಿನ_1bit;

    ಹಿಂತಿರುಗಿ (ne_any_ಬಿಟ್ >> 7) * 0xff;
}
