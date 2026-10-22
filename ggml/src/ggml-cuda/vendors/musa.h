#निर्देशः once

#अन्तर्भाव <musa_runtime.h>
#अन्तर्भाव <musa.h>
#अन्तर्भाव <mublas.h>
#अन्तर्भाव <musa_bf16.h>
#अन्तर्भाव <musa_fp16.h>

#यदि_परिभाषितम् __MUSA_ARCH__
#परिभाषय __CUDA_ARCH__ 1300 // GGML_CUDA_CC_RUBIN
#यदि_अन्तः // __MUSA_ARCH__

#परिभाषय CUBLAS_COMPUTE_16F CUDA_R_16F
#परिभाषय CUBLAS_COMPUTE_32F CUDA_R_32F
#परिभाषय CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#परिभाषय CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#परिभाषय CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#परिभाषय CUBLAS_OP_N MUBLAS_OP_N
#परिभाषय CUBLAS_OP_T MUBLAS_OP_T
#परिभाषय CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#परिभाषय CUBLAS_SIDE_दक्षिणः MUBLAS_SIDE_दक्षिणः
#परिभाषय CUBLAS_FILL_बहुलकम्_ऊर्ध्वस्थम् MUBLAS_FILL_बहुलकम्_ऊर्ध्वस्थम्
#परिभाषय CUBLAS_DIAG_NON_UNIT MUBLAS_DIAG_NON_UNIT
#परिभाषय CUBLAS_स्थितिः_SUCCESS MUBLAS_स्थितिः_SUCCESS
#परिभाषय CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#परिभाषय CUDA_R_16F  MUSA_R_16F
#परिभाषय CUDA_R_16BF MUSA_R_16BF
#परिभाषय CUDA_R_32F  MUSA_R_32F
#परिभाषय cublasStrsmBatched mublasStrsmBatched
#परिभाषय cublasComputeType_t cuda_दत्त_Type_t
#परिभाषय cublasCreate mublasCreate
#परिभाषय cublasDestroy mublasDestroy
#परिभाषय cublasGemmEx mublasGemmEx
#परिभाषय cublasGemmBatchedEx mublasGemmBatchedEx
#परिभाषय cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#परिभाषय cublasHandle_t mublasHandle_t
#परिभाषय cublas_समुच्चयः_Math_बहुलकम् mublas_समुच्चयः_Math_बहुलकम्
#परिभाषय cublas_समुच्चयः_प्रवाहः mublas_समुच्चयः_प्रवाहः
#परिभाषय cublasSgemm mublasSgemm
#परिभाषय cublasSgemmBatched mublasSgemmBatched
#परिभाषय cublasSgemmStridedBatched mublasSgemmStridedBatched
#परिभाषय cublas_स्थितिः_t mublas_स्थितिः_t
#परिभाषय cublasOperation_t mublasOperation_t
#परिभाषय cublas_Get_स्थितिः_String mublas_Get_स्थितिः_String
#परिभाषय cuda_दत्त_Type_t musa_दत्त_Type_t
#परिभाषय cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#परिभाषय cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#परिभाषय cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#परिभाषय cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#परिभाषय cudaDeviceProp musaDeviceProp
#परिभाषय cudaDeviceSynchronize musaDeviceSynchronize
#परिभाषय cudaDeviceGetAttribute musaDeviceGetAttribute
#परिभाषय cudaError_t musaError_t
#परिभाषय cuda_Error_स्मृतिः_आवण्टनम् musa_Error_स्मृतिः_आवण्टनम्
#परिभाषय cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#परिभाषय cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled
#परिभाषय cudaEventCreateWithFlags musaEventCreateWithFlags
#परिभाषय cudaEventDisableTiming musaEventDisableTiming
#परिभाषय cuda_Event_लेखः musa_Event_लेखः
#परिभाषय cudaEventSynchronize musaEventSynchronize
#परिभाषय cudaEvent_t musaEvent_t
#परिभाषय cudaEventDestroy musaEventDestroy
#परिभाषय cudaFree musaFree
#परिभाषय cudaFreeHost musaFreeHost
#परिभाषय cudaGetDevice musaGetDevice
#परिभाषय cuda_Get_Device_गणनम् musa_Get_Device_गणनम्
#परिभाषय cudaGetDeviceProperties musaGetDeviceProperties
#परिभाषय cudaGetErrorString musaGetErrorString
#परिभाषय cuda_Get_चरमः_Error musa_Get_चरमः_Error
#परिभाषय cuda_Host_पञ्जिकाकोशः musa_Host_पञ्जिकाकोशः
#परिभाषय cuda_Host_पञ्जिकाकोशः_Portable musa_Host_पञ्जिकाकोशः_Portable
#परिभाषय cuda_Host_पञ्जिकाकोशः_Read_Only musa_Host_पञ्जिकाकोशः_Read_Only
#परिभाषय cudaHostUnregister musaHostUnregister
#परिभाषय cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#परिभाषय cudaLaunchHostFunc musaLaunchHostFunc
#परिभाषय cudaMalloc musaMalloc
#परिभाषय cudaMallocHost musaMallocHost
#परिभाषय cudaMallocManaged musaMallocManaged
#परिभाषय cudaMemcpy musaMemcpy
#परिभाषय cudaMemcpyAsync musaMemcpyAsync
#परिभाषय cudaMemcpyPeerAsync musaMemcpyPeerAsync
#परिभाषय cudaMemcpy2DAsync musaMemcpy2DAsync
#परिभाषय cuda_Memcpy_Device_गन्तव्यम्_Device musa_Memcpy_Device_गन्तव्यम्_Device
#परिभाषय cuda_Memcpy_Device_गन्तव्यम्_Host musa_Memcpy_Device_गन्तव्यम्_Host
#परिभाषय cuda_Memcpy_Host_गन्तव्यम्_Device musa_Memcpy_Host_गन्तव्यम्_Device
#परिभाषय cuda_Memcpy_विधा musa_Memcpy_विधा
#परिभाषय cudaMemset musaMemset
#परिभाषय cudaMemsetAsync musaMemsetAsync
#परिभाषय cudaMemGetInfo musaMemGetInfo
#परिभाषय cuda_Occupancy_Max_Potential_खण्डम्_आकार musa_Occupancy_Max_Potential_खण्डम्_आकार
#परिभाषय cuda_समुच्चयः_Device musa_समुच्चयः_Device
#परिभाषय cuda_प्रवाहः_Create_With_Flags musa_प्रवाहः_Create_With_Flags
#परिभाषय cuda_प्रवाहः_Destroy musa_प्रवाहः_Destroy
#परिभाषय cuda_प्रवाहः_Fire_And_Forget musa_प्रवाहः_Fire_And_Forget
#परिभाषय cuda_प्रवाहः_Non_Blocking musa_प्रवाहः_Non_Blocking
#परिभाषय cuda_प्रवाहः_प्रत्येकम्_Thread musa_प्रवाहः_प्रत्येकम्_Thread
#परिभाषय cuda_प्रवाहः_Synchronize musa_प्रवाहः_Synchronize
#परिभाषय cuda_प्रवाहः_Wait_Event musa_प्रवाहः_Wait_Event
#परिभाषय cuda_प्रवाहः_t musa_प्रवाहः_t
#परिभाषय cudaSuccess musaSuccess

// Additional mappings for MUSA virtual memory pool
#परिभाषय CU_DEVICE_ATTRIBUTE_VIRTUAL_स्मृतिः_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_स्थानाङ्कः_MANAGEMENT_SUPPORTED
#परिभाषय CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#परिभाषय CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#परिभाषय CU_MEM_आवण्टनम्_TYPE_PINNED MU_MEM_आवण्टनम्_TYPE_PINNED
#परिभाषय CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#परिभाषय CUdevice MUdevice
#परिभाषय CUdeviceptr MUdeviceptr
#परिभाषय CUmemAccessDesc MUmemAccessDesc
#परिभाषय CUmem_आवण्टनम्_Prop MUmem_आवण्टनम्_Prop
#परिभाषय CUmem_Generic_आवण्टनम्_Handle MUmem_Generic_आवण्टनम्_Handle
#परिभाषय cuDeviceGet muDeviceGet
#परिभाषय cuDeviceGetAttribute muDeviceGetAttribute
#परिभाषय cu_Mem_स्थानाङ्कः_Free mu_Mem_स्थानाङ्कः_Free
#परिभाषय cu_Mem_स्थानाङ्कः_Reserve mu_Mem_स्थानाङ्कः_Reserve
#परिभाषय cuMemCreate muMemCreate
#परिभाषय cu_Mem_Get_आवण्टनम्_Granularity mu_Mem_Get_आवण्टनम्_Granularity
#परिभाषय cuMemMap muMemMap
#परिभाषय cuMemRelease muMemRelease
#परिभाषय cu_Mem_समुच्चयः_Access mu_Mem_समुच्चयः_Access
#परिभाषय cuMemUnmap muMemUnmap
#परिभाषय cudaDevAttrCooperativeLaunch musaDevAttrCooperativeLaunch
#परिभाषय cuda_Func_Attribute_Max_Dynamic_Shared_स्मृतिः_आकार musa_Func_Attribute_Max_Dynamic_Shared_स्मृतिः_आकार
#परिभाषय cuda_Func_समुच्चयः_Attribute musa_Func_समुच्चयः_Attribute
#परिभाषय cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
#परिभाषय make_cudaExtent make_musaExtent
#परिभाषय make_cudaPitchedPtr make_musaPitchedPtr

// Additional mappings for MUSA graphs
#परिभाषय CUDA_SUCCESS MUSA_SUCCESS
#परिभाषय CUresult MUresult
#परिभाषय cuGetErrorString muGetErrorString
#परिभाषय cudaErrorGraphExecUpdateFailure musaErrorGraphExecUpdateFailure
#परिभाषय cudaErrorInvalidDeviceFunction musaErrorInvalidDeviceFunction
#परिभाषय cudaGraphDestroy musaGraphDestroy
#परिभाषय cudaGraphExecDestroy musaGraphExecDestroy
#परिभाषय cudaGraphExec_t musaGraphExec_t
#परिभाषय cudaGraphExecUpdate musaGraphExecUpdate
#परिभाषय cuda_Graph_Exec_Update_परिणाम musa_Graph_Exec_Update_परिणाम
#परिभाषय cudaGraphGetNodes musaGraphGetNodes
#परिभाषय cudaGraphInstantiate musaGraphInstantiate
#परिभाषय cuda_Graph_Kernel_ग्रन्थिः_Get_Params musa_Graph_Kernel_ग्रन्थिः_Get_Params
#परिभाषय cuda_Graph_Kernel_ग्रन्थिः_समुच्चयः_Params musa_Graph_Kernel_ग्रन्थिः_समुच्चयः_Params
#परिभाषय cudaGraphLaunch musaGraphLaunch
#परिभाषय cuda_Graph_ग्रन्थिः_Get_Type musa_Graph_ग्रन्थिः_Get_Type
#परिभाषय cuda_Graph_ग्रन्थिः_t musa_Graph_ग्रन्थिः_t
#परिभाषय cuda_Graph_ग्रन्थिः_Type musa_Graph_ग्रन्थिः_Type
#परिभाषय cuda_Graph_ग्रन्थिः_Type_Kernel musa_Graph_ग्रन्थिः_Type_Kernel
#परिभाषय cudaGraph_t musaGraph_t
#परिभाषय cuda_Kernel_ग्रन्थिः_Params musa_Kernel_ग्रन्थिः_Params
#परिभाषय cuda_प्रवाहः_Capture_बहुलकम्_Relaxed musa_प्रवाहः_Capture_बहुलकम्_Relaxed
#परिभाषय cuda_प्रवाहः_Begin_Capture musa_प्रवाहः_Begin_Capture
#परिभाषय cuda_प्रवाहः_अन्तः_Capture musa_प्रवाहः_अन्तः_Capture
#परिभाषय cuda_प्रवाहः_Capture_स्थितिः musa_प्रवाहः_Capture_स्थितिः
#परिभाषय cuda_प्रवाहः_Capture_स्थितिः_None musa_प्रवाहः_Capture_स्थितिः_None
#परिभाषय cuda_प्रवाहः_Is_Capturing musa_प्रवाहः_Is_Capturing
#परिभाषय cuda_Occupancy_Max_सक्रिय_खण्डानि_प्रत्येकम्_Multiprocessor musa_Occupancy_Max_सक्रिय_खण्डानि_प्रत्येकम्_Multiprocessor

प्रकारनिर्देश __mt_bfloat16 nv_bfloat16;
प्रकारनिर्देश __mt_bfloat162 nv_bfloat162;
