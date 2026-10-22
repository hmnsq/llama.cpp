#आशय once

#समावेश <musa_runtime.h>
#समावेश <musa.h>
#समावेश <mublas.h>
#समावेश <musa_bf16.h>
#समावेश <musa_fp16.h>

#अगर_घोषित __MUSA_ARCH__
#घोषणा __CUDA_ARCH__ 1300 // GGML_CUDA_CC_RUBIN
#पूर्ण_अगर // __MUSA_ARCH__

#घोषणा CUBLAS_COMPUTE_16F CUDA_R_16F
#घोषणा CUBLAS_COMPUTE_32F CUDA_R_32F
#घोषणा CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#घोषणा CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#घोषणा CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#घोषणा CUBLAS_OP_N MUBLAS_OP_N
#घोषणा CUBLAS_OP_T MUBLAS_OP_T
#घोषणा CUBLAS_DEFAULT_MATH MUBLAS_DEFAULT_MATH
#घोषणा CUBLAS_SIDE_दायाँ MUBLAS_SIDE_दायाँ
#घोषणा CUBLAS_FILL_MODE_UPPER MUBLAS_FILL_MODE_UPPER
#घोषणा CUBLAS_DIAG_NON_UNIT MUBLAS_DIAG_NON_UNIT
#घोषणा CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#घोषणा CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_TENSOR_OP_MATH
#घोषणा CUDA_R_16F  MUSA_R_16F
#घोषणा CUDA_R_16BF MUSA_R_16BF
#घोषणा CUDA_R_32F  MUSA_R_32F
#घोषणा cublasStrsmBatched mublasStrsmBatched
#घोषणा cublasComputeType_t cuda_तथ्य_Type_t
#घोषणा cublasCreate mublasCreate
#घोषणा cublasDestroy mublasDestroy
#घोषणा cublasGemmEx mublasGemmEx
#घोषणा cublasGemmBatchedEx mublasGemmBatchedEx
#घोषणा cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#घोषणा cublasHandle_t mublasHandle_t
#घोषणा cublasSetMathMode mublasSetMathMode
#घोषणा cublasSetStream mublasSetStream
#घोषणा cublasSgemm mublasSgemm
#घोषणा cublasSgemmBatched mublasSgemmBatched
#घोषणा cublasSgemmStridedBatched mublasSgemmStridedBatched
#घोषणा cublasStatus_t mublasStatus_t
#घोषणा cublasOperation_t mublasOperation_t
#घोषणा cublasGetStatusString mublasGetStatusString
#घोषणा cuda_तथ्य_Type_t musa_तथ्य_Type_t
#घोषणा cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#घोषणा cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#घोषणा cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#घोषणा cudaDeviceGetPCIBusId musaDeviceGetPCIBusId
#घोषणा cudaDeviceProp musaDeviceProp
#घोषणा cudaDeviceSynchronize musaDeviceSynchronize
#घोषणा cudaDeviceGetAttribute musaDeviceGetAttribute
#घोषणा cudaError_t musaError_t
#घोषणा cudaErrorMemoryAllocation musaErrorMemoryAllocation
#घोषणा cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#घोषणा cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled
#घोषणा cudaEventCreateWithFlags musaEventCreateWithFlags
#घोषणा cudaEventDisableTiming musaEventDisableTiming
#घोषणा cudaEventRecord musaEventRecord
#घोषणा cudaEventSynchronize musaEventSynchronize
#घोषणा cudaEvent_t musaEvent_t
#घोषणा cudaEventDestroy musaEventDestroy
#घोषणा cudaFree musaFree
#घोषणा cudaFreeHost musaFreeHost
#घोषणा cudaGetDevice musaGetDevice
#घोषणा cuda_Get_Device_गिनती musa_Get_Device_गिनती
#घोषणा cudaGetDeviceProperties musaGetDeviceProperties
#घोषणा cudaGetErrorString musaGetErrorString
#घोषणा cudaGetLastError musaGetLastError
#घोषणा cudaHostRegister musaHostRegister
#घोषणा cudaHostRegisterPortable musaHostRegisterPortable
#घोषणा cudaHostRegisterReadOnly musaHostRegisterReadOnly
#घोषणा cudaHostUnregister musaHostUnregister
#घोषणा cudaLaunchCooperativeKernel musaLaunchCooperativeKernel
#घोषणा cudaLaunchHostFunc musaLaunchHostFunc
#घोषणा cudaMalloc musaMalloc
#घोषणा cudaMallocHost musaMallocHost
#घोषणा cudaMallocManaged musaMallocManaged
#घोषणा cudaMemcpy musaMemcpy
#घोषणा cudaMemcpyAsync musaMemcpyAsync
#घोषणा cudaMemcpyPeerAsync musaMemcpyPeerAsync
#घोषणा cudaMemcpy2DAsync musaMemcpy2DAsync
#घोषणा cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#घोषणा cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#घोषणा cudaMemcpyHostToDevice musaMemcpyHostToDevice
#घोषणा cudaMemcpyKind musaMemcpyKind
#घोषणा cudaMemset musaMemset
#घोषणा cudaMemsetAsync musaMemsetAsync
#घोषणा cudaMemGetInfo musaMemGetInfo
#घोषणा cuda_Occupancy_Max_Potential_Block_आकार musa_Occupancy_Max_Potential_Block_आकार
#घोषणा cudaSetDevice musaSetDevice
#घोषणा cudaStreamCreateWithFlags musaStreamCreateWithFlags
#घोषणा cudaStreamDestroy musaStreamDestroy
#घोषणा cudaStreamFireAndForget musaStreamFireAndForget
#घोषणा cudaStreamNonBlocking musaStreamNonBlocking
#घोषणा cudaStreamPerThread musaStreamPerThread
#घोषणा cudaStreamSynchronize musaStreamSynchronize
#घोषणा cudaStreamWaitEvent musaStreamWaitEvent
#घोषणा cudaStream_t musaStream_t
#घोषणा cudaSuccess musaSuccess

// Additional mappings for MUSA virtual memory pool
#घोषणा CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#घोषणा CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#घोषणा CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#घोषणा CU_MEM_ALLOCATION_TYPE_PINNED MU_MEM_ALLOCATION_TYPE_PINNED
#घोषणा CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#घोषणा CUdevice MUdevice
#घोषणा CUdeviceptr MUdeviceptr
#घोषणा CUmemAccessDesc MUmemAccessDesc
#घोषणा CUmemAllocationProp MUmemAllocationProp
#घोषणा CUmemGenericAllocationHandle MUmemGenericAllocationHandle
#घोषणा cuDeviceGet muDeviceGet
#घोषणा cuDeviceGetAttribute muDeviceGetAttribute
#घोषणा cuMemAddressFree muMemAddressFree
#घोषणा cuMemAddressReserve muMemAddressReserve
#घोषणा cuMemCreate muMemCreate
#घोषणा cuMemGetAllocationGranularity muMemGetAllocationGranularity
#घोषणा cuMemMap muMemMap
#घोषणा cuMemRelease muMemRelease
#घोषणा cuMemSetAccess muMemSetAccess
#घोषणा cuMemUnmap muMemUnmap
#घोषणा cudaDevAttrCooperativeLaunch musaDevAttrCooperativeLaunch
#घोषणा cuda_Func_Attribute_Max_Dynamic_Shared_Memory_आकार musa_Func_Attribute_Max_Dynamic_Shared_Memory_आकार
#घोषणा cudaFuncSetAttribute musaFuncSetAttribute
#घोषणा cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
#घोषणा make_cudaExtent make_musaExtent
#घोषणा make_cudaPitchedPtr make_musaPitchedPtr

// Additional mappings for MUSA graphs
#घोषणा CUDA_SUCCESS MUSA_SUCCESS
#घोषणा CUresult MUresult
#घोषणा cuGetErrorString muGetErrorString
#घोषणा cudaErrorGraphExecUpdateFailure musaErrorGraphExecUpdateFailure
#घोषणा cudaErrorInvalidDeviceFunction musaErrorInvalidDeviceFunction
#घोषणा cudaGraphDestroy musaGraphDestroy
#घोषणा cudaGraphExecDestroy musaGraphExecDestroy
#घोषणा cudaGraphExec_t musaGraphExec_t
#घोषणा cudaGraphExecUpdate musaGraphExecUpdate
#घोषणा cuda_Graph_Exec_Update_परिणाम musa_Graph_Exec_Update_परिणाम
#घोषणा cudaGraphGetNodes musaGraphGetNodes
#घोषणा cudaGraphInstantiate musaGraphInstantiate
#घोषणा cuda_Graph_Kernel_गाँठ_Get_Params musa_Graph_Kernel_गाँठ_Get_Params
#घोषणा cuda_Graph_Kernel_गाँठ_Set_Params musa_Graph_Kernel_गाँठ_Set_Params
#घोषणा cudaGraphLaunch musaGraphLaunch
#घोषणा cuda_Graph_गाँठ_Get_Type musa_Graph_गाँठ_Get_Type
#घोषणा cuda_Graph_गाँठ_t musa_Graph_गाँठ_t
#घोषणा cuda_Graph_गाँठ_Type musa_Graph_गाँठ_Type
#घोषणा cuda_Graph_गाँठ_Type_Kernel musa_Graph_गाँठ_Type_Kernel
#घोषणा cudaGraph_t musaGraph_t
#घोषणा cuda_Kernel_गाँठ_Params musa_Kernel_गाँठ_Params
#घोषणा cudaStreamCaptureModeRelaxed musaStreamCaptureModeRelaxed
#घोषणा cudaStreamBeginCapture musaStreamBeginCapture
#घोषणा cudaStreamEndCapture musaStreamEndCapture
#घोषणा cudaStreamCaptureStatus musaStreamCaptureStatus
#घोषणा cudaStreamCaptureStatusNone musaStreamCaptureStatusNone
#घोषणा cudaStreamIsCapturing musaStreamIsCapturing
#घोषणा cudaOccupancyMaxActiveBlocksPerMultiprocessor musaOccupancyMaxActiveBlocksPerMultiprocessor

प्रकार __mt_bfloat16 nv_bfloat16;
प्रकार __mt_bfloat162 nv_bfloat162;
