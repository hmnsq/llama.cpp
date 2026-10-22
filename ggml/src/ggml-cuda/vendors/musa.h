#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <musa_runtime.h>
#ಅನ್ತರ್ಭಾವ <musa.h>
#ಅನ್ತರ್ಭಾವ <mublas.h>
#ಅನ್ತರ್ಭಾವ <musa_bf16.h>
#ಅನ್ತರ್ಭಾವ <musa_fp16.h>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __MUSA_ARCH__
#ವ್ಯಾಖ್ಯಾನಿಸು __CUDA_ARCH__ 1300 // GGML_CUDA_CC_RUBIN
#ಆದರೆ_ಅಂತ್ಯ // __MUSA_ARCH__

#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_16F CUDA_R_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F CUDA_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_OP_N MUBLAS_OP_N
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_OP_T MUBLAS_OP_T
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_SIDE_ಬಲ MUBLAS_SIDE_ಬಲ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_FILL_ಬಹುಲಕ_ಊರ್ಧ್ವ MUBLAS_FILL_ಬಹುಲಕ_ಊರ್ಧ್ವ
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_DIAG_NON_UNIT MUBLAS_DIAG_NON_UNIT
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_ಸ್ಥಿತಿ_SUCCESS MUBLAS_ಸ್ಥಿತಿ_SUCCESS
#ವ್ಯಾಖ್ಯಾನಿಸು CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_16F  MUSA_R_16F
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_16BF MUSA_R_16BF
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_R_32F  MUSA_R_32F
#ವ್ಯಾಖ್ಯಾನಿಸು cublasStrsmBatched mublasStrsmBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublasComputeType_t cuda_ದತ್ತಾಂಶ_Type_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublasCreate mublasCreate
#ವ್ಯಾಖ್ಯಾನಿಸು cublasDestroy mublasDestroy
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmEx mublasGemmEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmBatchedEx mublasGemmBatchedEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#ವ್ಯಾಖ್ಯಾನಿಸು cublasHandle_t mublasHandle_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಗಣ_Math_ಬಹುಲಕ mublas_ಗಣ_Math_ಬಹುಲಕ
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಗಣ_ಹರಿವು mublas_ಗಣ_ಹರಿವು
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemm mublasSgemm
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemmBatched mublasSgemmBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublasSgemmStridedBatched mublasSgemmStridedBatched
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_ಸ್ಥಿತಿ_t mublas_ಸ್ಥಿತಿ_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublasOperation_t mublasOperation_t
#ವ್ಯಾಖ್ಯಾನಿಸು cublas_Get_ಸ್ಥಿತಿ_String mublas_Get_ಸ್ಥಿತಿ_String
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ದತ್ತಾಂಶ_Type_t musa_ದತ್ತಾಂಶ_Type_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceProp musaDeviceProp
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceSynchronize musaDeviceSynchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDeviceGetAttribute musaDeviceGetAttribute
#ವ್ಯಾಖ್ಯಾನಿಸು cudaError_t musaError_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Error_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ musa_Error_ಸ್ಮರಣೆ_ಹಂಚಿಕೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventCreateWithFlags musaEventCreateWithFlags
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventDisableTiming musaEventDisableTiming
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Event_ದಾಖಲೆ musa_Event_ದಾಖಲೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventSynchronize musaEventSynchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEvent_t musaEvent_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaEventDestroy musaEventDestroy
#ವ್ಯಾಖ್ಯಾನಿಸು cudaFree musaFree
#ವ್ಯಾಖ್ಯಾನಿಸು cudaFreeHost musaFreeHost
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetDevice musaGetDevice
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Get_Device_ಎಣಿಕೆ musa_Get_Device_ಎಣಿಕೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetDeviceProperties musaGetDeviceProperties
#ವ್ಯಾಖ್ಯಾನಿಸು cudaGetErrorString musaGetErrorString
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Get_ಕೊನೆಯ_Error musa_Get_ಕೊನೆಯ_Error
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್ musa_Host_ರಿಜಿಸ್ಟರ್
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್_Portable musa_Host_ರಿಜಿಸ್ಟರ್_Portable
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Host_ರಿಜಿಸ್ಟರ್_Read_Only musa_Host_ರಿಜಿಸ್ಟರ್_Read_Only
#ವ್ಯಾಖ್ಯಾನಿಸು cudaHostUnregister musaHostUnregister
#ವ್ಯಾಖ್ಯಾನಿಸು cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#ವ್ಯಾಖ್ಯಾನಿಸು cudaLaunchHostFunc musaLaunchHostFunc
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMalloc musaMalloc
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMallocHost musaMallocHost
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMallocManaged musaMallocManaged
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpy musaMemcpy
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpyAsync musaMemcpyAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpyPeerAsync musaMemcpyPeerAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpy2DAsync musaMemcpy2DAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Device_ಗಮ್ಯ_Device musa_Memcpy_Device_ಗಮ್ಯ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Device_ಗಮ್ಯ_Host musa_Memcpy_Device_ಗಮ್ಯ_Host
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_Host_ಗಮ್ಯ_Device musa_Memcpy_Host_ಗಮ್ಯ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Memcpy_ಬಗೆ musa_Memcpy_ಬಗೆ
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemset musaMemset
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemsetAsync musaMemsetAsync
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemGetInfo musaMemGetInfo
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Occupancy_Max_Potential_ಖಂಡ_ಗಾತ್ರ musa_Occupancy_Max_Potential_ಖಂಡ_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಗಣ_Device musa_ಗಣ_Device
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Create_With_Flags musa_ಹರಿವು_Create_With_Flags
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Destroy musa_ಹರಿವು_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Fire_And_Forget musa_ಹರಿವು_Fire_And_Forget
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Non_Blocking musa_ಹರಿವು_Non_Blocking
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_ಪ್ರತಿ_Thread musa_ಹರಿವು_ಪ್ರತಿ_Thread
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Synchronize musa_ಹರಿವು_Synchronize
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Wait_Event musa_ಹರಿವು_Wait_Event
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_t musa_ಹರಿವು_t
#ವ್ಯಾಖ್ಯಾನಿಸು cudaSuccess musaSuccess

// Additional mappings for MUSA virtual memory pool
#ವ್ಯಾಖ್ಯಾನಿಸು CU_DEVICE_ATTRIBUTE_VIRTUAL_ಸ್ಮರಣೆ_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_ವಿಳಾಸ_MANAGEMENT_SUPPORTED
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_ಹಂಚಿಕೆ_TYPE_PINNED MU_MEM_ಹಂಚಿಕೆ_TYPE_PINNED
#ವ್ಯಾಖ್ಯಾನಿಸು CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#ವ್ಯಾಖ್ಯಾನಿಸು CUdevice MUdevice
#ವ್ಯಾಖ್ಯಾನಿಸು CUdeviceptr MUdeviceptr
#ವ್ಯಾಖ್ಯಾನಿಸು CUmemAccessDesc MUmemAccessDesc
#ವ್ಯಾಖ್ಯಾನಿಸು CUmem_ಹಂಚಿಕೆ_Prop MUmem_ಹಂಚಿಕೆ_Prop
#ವ್ಯಾಖ್ಯಾನಿಸು CUmem_Generic_ಹಂಚಿಕೆ_Handle MUmem_Generic_ಹಂಚಿಕೆ_Handle
#ವ್ಯಾಖ್ಯಾನಿಸು cuDeviceGet muDeviceGet
#ವ್ಯಾಖ್ಯಾನಿಸು cuDeviceGetAttribute muDeviceGetAttribute
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ವಿಳಾಸ_Free mu_Mem_ವಿಳಾಸ_Free
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ವಿಳಾಸ_Reserve mu_Mem_ವಿಳಾಸ_Reserve
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemCreate muMemCreate
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_Get_ಹಂಚಿಕೆ_Granularity mu_Mem_Get_ಹಂಚಿಕೆ_Granularity
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemMap muMemMap
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemRelease muMemRelease
#ವ್ಯಾಖ್ಯಾನಿಸು cu_Mem_ಗಣ_Access mu_Mem_ಗಣ_Access
#ವ್ಯಾಖ್ಯಾನಿಸು cuMemUnmap muMemUnmap
#ವ್ಯಾಖ್ಯಾನಿಸು cudaDevAttrCooperativeLaunch musaDevAttrCooperativeLaunch
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Func_Attribute_Max_Dynamic_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ musa_Func_Attribute_Max_Dynamic_Shared_ಸ್ಮರಣೆ_ಗಾತ್ರ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Func_ಗಣ_Attribute musa_Func_ಗಣ_Attribute
#ವ್ಯಾಖ್ಯಾನಿಸು cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
#ವ್ಯಾಖ್ಯಾನಿಸು make_cudaExtent make_musaExtent
#ವ್ಯಾಖ್ಯಾನಿಸು make_cudaPitchedPtr make_musaPitchedPtr

// Additional mappings for MUSA graphs
#ವ್ಯಾಖ್ಯಾನಿಸು CUDA_SUCCESS MUSA_SUCCESS
#ವ್ಯಾಖ್ಯಾನಿಸು CUresult MUresult
#ವ್ಯಾಖ್ಯಾನಿಸು cuGetErrorString muGetErrorString
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Error_ನಕ್ಷೆ_Exec_Update_Failure musa_Error_ನಕ್ಷೆ_Exec_Update_Failure
#ವ್ಯಾಖ್ಯಾನಿಸು cudaErrorInvalidDeviceFunction musaErrorInvalidDeviceFunction
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Destroy musa_ನಕ್ಷೆ_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Destroy musa_ನಕ್ಷೆ_Exec_Destroy
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_t musa_ನಕ್ಷೆ_Exec_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Update musa_ನಕ್ಷೆ_Exec_Update
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Exec_Update_ಫಲಿತಾಂಶ musa_ನಕ್ಷೆ_Exec_Update_ಫಲಿತಾಂಶ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Get_Nodes musa_ನಕ್ಷೆ_Get_Nodes
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Instantiate musa_ನಕ್ಷೆ_Instantiate
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Kernel_ಗಂಟು_Get_Params musa_ನಕ್ಷೆ_Kernel_ಗಂಟು_Get_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Kernel_ಗಂಟು_ಗಣ_Params musa_ನಕ್ಷೆ_Kernel_ಗಂಟು_ಗಣ_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_Launch musa_ನಕ್ಷೆ_Launch
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Get_Type musa_ನಕ್ಷೆ_ಗಂಟು_Get_Type
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_t musa_ನಕ್ಷೆ_ಗಂಟು_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Type musa_ನಕ್ಷೆ_ಗಂಟು_Type
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_ಗಂಟು_Type_Kernel musa_ನಕ್ಷೆ_ಗಂಟು_Type_Kernel
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ನಕ್ಷೆ_t musa_ನಕ್ಷೆ_t
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Kernel_ಗಂಟು_Params musa_Kernel_ಗಂಟು_Params
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Capture_ಬಹುಲಕ_Relaxed musa_ಹರಿವು_Capture_ಬಹುಲಕ_Relaxed
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Begin_Capture musa_ಹರಿವು_Begin_Capture
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_ಕೊನೆ_Capture musa_ಹರಿವು_ಕೊನೆ_Capture
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Capture_ಸ್ಥಿತಿ musa_ಹರಿವು_Capture_ಸ್ಥಿತಿ
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Capture_ಸ್ಥಿತಿ_None musa_ಹರಿವು_Capture_ಸ್ಥಿತಿ_None
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_ಹರಿವು_Is_Capturing musa_ಹರಿವು_Is_Capturing
#ವ್ಯಾಖ್ಯಾನಿಸು cuda_Occupancy_Max_ಸಕ್ರಿಯ_ಖಂಡಗಳು_ಪ್ರತಿ_Multiprocessor musa_Occupancy_Max_ಸಕ್ರಿಯ_ಖಂಡಗಳು_ಪ್ರತಿ_Multiprocessor

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __mt_bfloat16 nv_bfloat16;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ __mt_bfloat162 nv_bfloat162;
