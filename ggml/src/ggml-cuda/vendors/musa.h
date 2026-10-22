#సూచన once

#అన్తర్భావ <musa_runtime.h>
#అన్తర్భావ <musa.h>
#అన్తర్భావ <mublas.h>
#అన్తర్భావ <musa_bf16.h>
#అన్తర్భావ <musa_fp16.h>

#నిర్వచితమైతే __MUSA_ARCH__
#నిర్వచించు __CUDA_ARCH__ 1300 // GGML_CUDA_CC_RUBIN
#అయితే_ముగింపు // __MUSA_ARCH__

#నిర్వచించు CUBLAS_COMPUTE_16F CUDA_R_16F
#నిర్వచించు CUBLAS_COMPUTE_32F CUDA_R_32F
#నిర్వచించు CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#నిర్వచించు CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#నిర్వచించు CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#నిర్వచించు CUBLAS_OP_N MUBLAS_OP_N
#నిర్వచించు CUBLAS_OP_T MUBLAS_OP_T
#నిర్వచించు CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#నిర్వచించు CUBLAS_SIDE_కుడి MUBLAS_SIDE_కుడి
#నిర్వచించు CUBLAS_FILL_బహుళకం_పై MUBLAS_FILL_బహుళకం_పై
#నిర్వచించు CUBLAS_DIAG_NON_UNIT MUBLAS_DIAG_NON_UNIT
#నిర్వచించు CUBLAS_స్థితి_SUCCESS MUBLAS_స్థితి_SUCCESS
#నిర్వచించు CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#నిర్వచించు CUDA_R_16F  MUSA_R_16F
#నిర్వచించు CUDA_R_16BF MUSA_R_16BF
#నిర్వచించు CUDA_R_32F  MUSA_R_32F
#నిర్వచించు cublasStrsmBatched mublasStrsmBatched
#నిర్వచించు cublasComputeType_t cuda_దత్తాంశం_Type_t
#నిర్వచించు cublasCreate mublasCreate
#నిర్వచించు cublasDestroy mublasDestroy
#నిర్వచించు cublasGemmEx mublasGemmEx
#నిర్వచించు cublasGemmBatchedEx mublasGemmBatchedEx
#నిర్వచించు cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#నిర్వచించు cublasHandle_t mublasHandle_t
#నిర్వచించు cublas_సమితి_Math_బహుళకం mublas_సమితి_Math_బహుళకం
#నిర్వచించు cublas_సమితి_ప్రవాహం mublas_సమితి_ప్రవాహం
#నిర్వచించు cublasSgemm mublasSgemm
#నిర్వచించు cublasSgemmBatched mublasSgemmBatched
#నిర్వచించు cublasSgemmStridedBatched mublasSgemmStridedBatched
#నిర్వచించు cublas_స్థితి_t mublas_స్థితి_t
#నిర్వచించు cublasOperation_t mublasOperation_t
#నిర్వచించు cublas_Get_స్థితి_String mublas_Get_స్థితి_String
#నిర్వచించు cuda_దత్తాంశం_Type_t musa_దత్తాంశం_Type_t
#నిర్వచించు cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#నిర్వచించు cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#నిర్వచించు cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#నిర్వచించు cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#నిర్వచించు cudaDeviceProp musaDeviceProp
#నిర్వచించు cudaDeviceSynchronize musaDeviceSynchronize
#నిర్వచించు cudaDeviceGetAttribute musaDeviceGetAttribute
#నిర్వచించు cudaError_t musaError_t
#నిర్వచించు cuda_Error_జ్ఞాపకం_కేటాయింపు musa_Error_జ్ఞాపకం_కేటాయింపు
#నిర్వచించు cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#నిర్వచించు cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled
#నిర్వచించు cudaEventCreateWithFlags musaEventCreateWithFlags
#నిర్వచించు cudaEventDisableTiming musaEventDisableTiming
#నిర్వచించు cuda_Event_రికార్డు musa_Event_రికార్డు
#నిర్వచించు cudaEventSynchronize musaEventSynchronize
#నిర్వచించు cudaEvent_t musaEvent_t
#నిర్వచించు cudaEventDestroy musaEventDestroy
#నిర్వచించు cudaFree musaFree
#నిర్వచించు cudaFreeHost musaFreeHost
#నిర్వచించు cudaGetDevice musaGetDevice
#నిర్వచించు cuda_Get_Device_లెక్క musa_Get_Device_లెక్క
#నిర్వచించు cudaGetDeviceProperties musaGetDeviceProperties
#నిర్వచించు cudaGetErrorString musaGetErrorString
#నిర్వచించు cuda_Get_చివరి_Error musa_Get_చివరి_Error
#నిర్వచించు cuda_Host_రిజిస్టరు musa_Host_రిజిస్టరు
#నిర్వచించు cuda_Host_రిజిస్టరు_Portable musa_Host_రిజిస్టరు_Portable
#నిర్వచించు cuda_Host_రిజిస్టరు_Read_Only musa_Host_రిజిస్టరు_Read_Only
#నిర్వచించు cudaHostUnregister musaHostUnregister
#నిర్వచించు cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#నిర్వచించు cudaLaunchHostFunc musaLaunchHostFunc
#నిర్వచించు cudaMalloc musaMalloc
#నిర్వచించు cudaMallocHost musaMallocHost
#నిర్వచించు cudaMallocManaged musaMallocManaged
#నిర్వచించు cudaMemcpy musaMemcpy
#నిర్వచించు cudaMemcpyAsync musaMemcpyAsync
#నిర్వచించు cudaMemcpyPeerAsync musaMemcpyPeerAsync
#నిర్వచించు cudaMemcpy2DAsync musaMemcpy2DAsync
#నిర్వచించు cuda_Memcpy_Device_గమ్యం_Device musa_Memcpy_Device_గమ్యం_Device
#నిర్వచించు cuda_Memcpy_Device_గమ్యం_Host musa_Memcpy_Device_గమ్యం_Host
#నిర్వచించు cuda_Memcpy_Host_గమ్యం_Device musa_Memcpy_Host_గమ్యం_Device
#నిర్వచించు cuda_Memcpy_రకం musa_Memcpy_రకం
#నిర్వచించు cudaMemset musaMemset
#నిర్వచించు cudaMemsetAsync musaMemsetAsync
#నిర్వచించు cudaMemGetInfo musaMemGetInfo
#నిర్వచించు cuda_Occupancy_Max_Potential_ఖండం_కొలత musa_Occupancy_Max_Potential_ఖండం_కొలత
#నిర్వచించు cuda_సమితి_Device musa_సమితి_Device
#నిర్వచించు cuda_ప్రవాహం_Create_With_Flags musa_ప్రవాహం_Create_With_Flags
#నిర్వచించు cuda_ప్రవాహం_Destroy musa_ప్రవాహం_Destroy
#నిర్వచించు cuda_ప్రవాహం_Fire_And_Forget musa_ప్రవాహం_Fire_And_Forget
#నిర్వచించు cuda_ప్రవాహం_Non_Blocking musa_ప్రవాహం_Non_Blocking
#నిర్వచించు cuda_ప్రవాహం_చొప్పున_Thread musa_ప్రవాహం_చొప్పున_Thread
#నిర్వచించు cuda_ప్రవాహం_Synchronize musa_ప్రవాహం_Synchronize
#నిర్వచించు cuda_ప్రవాహం_Wait_Event musa_ప్రవాహం_Wait_Event
#నిర్వచించు cuda_ప్రవాహం_t musa_ప్రవాహం_t
#నిర్వచించు cudaSuccess musaSuccess

// Additional mappings for MUSA virtual memory pool
#నిర్వచించు CU_DEVICE_ATTRIBUTE_VIRTUAL_జ్ఞాపకం_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_చిరునామా_MANAGEMENT_SUPPORTED
#నిర్వచించు CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#నిర్వచించు CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#నిర్వచించు CU_MEM_కేటాయింపు_TYPE_PINNED MU_MEM_కేటాయింపు_TYPE_PINNED
#నిర్వచించు CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#నిర్వచించు CUdevice MUdevice
#నిర్వచించు CUdeviceptr MUdeviceptr
#నిర్వచించు CUmemAccessDesc MUmemAccessDesc
#నిర్వచించు CUmem_కేటాయింపు_Prop MUmem_కేటాయింపు_Prop
#నిర్వచించు CUmem_Generic_కేటాయింపు_Handle MUmem_Generic_కేటాయింపు_Handle
#నిర్వచించు cuDeviceGet muDeviceGet
#నిర్వచించు cuDeviceGetAttribute muDeviceGetAttribute
#నిర్వచించు cu_Mem_చిరునామా_Free mu_Mem_చిరునామా_Free
#నిర్వచించు cu_Mem_చిరునామా_Reserve mu_Mem_చిరునామా_Reserve
#నిర్వచించు cuMemCreate muMemCreate
#నిర్వచించు cu_Mem_Get_కేటాయింపు_Granularity mu_Mem_Get_కేటాయింపు_Granularity
#నిర్వచించు cuMemMap muMemMap
#నిర్వచించు cuMemRelease muMemRelease
#నిర్వచించు cu_Mem_సమితి_Access mu_Mem_సమితి_Access
#నిర్వచించు cuMemUnmap muMemUnmap
#నిర్వచించు cudaDevAttrCooperativeLaunch musaDevAttrCooperativeLaunch
#నిర్వచించు cuda_Func_Attribute_Max_Dynamic_Shared_జ్ఞాపకం_కొలత musa_Func_Attribute_Max_Dynamic_Shared_జ్ఞాపకం_కొలత
#నిర్వచించు cuda_Func_సమితి_Attribute musa_Func_సమితి_Attribute
#నిర్వచించు cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
#నిర్వచించు make_cudaExtent make_musaExtent
#నిర్వచించు make_cudaPitchedPtr make_musaPitchedPtr

// Additional mappings for MUSA graphs
#నిర్వచించు CUDA_SUCCESS MUSA_SUCCESS
#నిర్వచించు CUresult MUresult
#నిర్వచించు cuGetErrorString muGetErrorString
#నిర్వచించు cuda_Error_లేఖాచిత్రం_Exec_Update_Failure musa_Error_లేఖాచిత్రం_Exec_Update_Failure
#నిర్వచించు cudaErrorInvalidDeviceFunction musaErrorInvalidDeviceFunction
#నిర్వచించు cuda_లేఖాచిత్రం_Destroy musa_లేఖాచిత్రం_Destroy
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Destroy musa_లేఖాచిత్రం_Exec_Destroy
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_t musa_లేఖాచిత్రం_Exec_t
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Update musa_లేఖాచిత్రం_Exec_Update
#నిర్వచించు cuda_లేఖాచిత్రం_Exec_Update_ఫలితం musa_లేఖాచిత్రం_Exec_Update_ఫలితం
#నిర్వచించు cuda_లేఖాచిత్రం_Get_Nodes musa_లేఖాచిత్రం_Get_Nodes
#నిర్వచించు cuda_లేఖాచిత్రం_Instantiate musa_లేఖాచిత్రం_Instantiate
#నిర్వచించు cuda_లేఖాచిత్రం_Kernel_కణుపు_Get_Params musa_లేఖాచిత్రం_Kernel_కణుపు_Get_Params
#నిర్వచించు cuda_లేఖాచిత్రం_Kernel_కణుపు_సమితి_Params musa_లేఖాచిత్రం_Kernel_కణుపు_సమితి_Params
#నిర్వచించు cuda_లేఖాచిత్రం_Launch musa_లేఖాచిత్రం_Launch
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Get_Type musa_లేఖాచిత్రం_కణుపు_Get_Type
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_t musa_లేఖాచిత్రం_కణుపు_t
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Type musa_లేఖాచిత్రం_కణుపు_Type
#నిర్వచించు cuda_లేఖాచిత్రం_కణుపు_Type_Kernel musa_లేఖాచిత్రం_కణుపు_Type_Kernel
#నిర్వచించు cuda_లేఖాచిత్రం_t musa_లేఖాచిత్రం_t
#నిర్వచించు cuda_Kernel_కణుపు_Params musa_Kernel_కణుపు_Params
#నిర్వచించు cuda_ప్రవాహం_Capture_బహుళకం_Relaxed musa_ప్రవాహం_Capture_బహుళకం_Relaxed
#నిర్వచించు cuda_ప్రవాహం_Begin_Capture musa_ప్రవాహం_Begin_Capture
#నిర్వచించు cuda_ప్రవాహం_చివర_Capture musa_ప్రవాహం_చివర_Capture
#నిర్వచించు cuda_ప్రవాహం_Capture_స్థితి musa_ప్రవాహం_Capture_స్థితి
#నిర్వచించు cuda_ప్రవాహం_Capture_స్థితి_None musa_ప్రవాహం_Capture_స్థితి_None
#నిర్వచించు cuda_ప్రవాహం_Is_Capturing musa_ప్రవాహం_Is_Capturing
#నిర్వచించు cuda_Occupancy_Max_చురుకైన_ఖండాలు_చొప్పున_Multiprocessor musa_Occupancy_Max_చురుకైన_ఖండాలు_చొప్పున_Multiprocessor

రకనిర్వచనం __mt_bfloat16 nv_bfloat16;
రకనిర్వచనం __mt_bfloat162 nv_bfloat162;
