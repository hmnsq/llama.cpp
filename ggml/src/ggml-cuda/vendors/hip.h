#لارښوونه once

#تعریف_کړه HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ضمن <hip/hip_runtime.h>
#ضمن <hipblas/hipblas.h>
#ضمن <hip/hip_fp16.h>
#ضمن <hip/hip_bf16.h>

#که_تعریف_شوی GGML_USE_NCCL
#ضمن <rccl/rccl.h>
#د_که_پای // GGML_USE_NCCL


#تعریف_کړه CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#تعریف_کړه CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#تعریف_کړه CUBLAS_OP_N HIPBLAS_OP_N
#تعریف_کړه CUBLAS_OP_T HIPBLAS_OP_T
#تعریف_کړه CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#تعریف_کړه CUBLAS_TF32_TENSOR_OP_MATH 0
#تعریف_کړه CUDA_R_16F  HIPBLAS_R_16F
#تعریف_کړه CUDA_R_16BF HIPBLAS_R_16B
#تعریف_کړه CUDA_R_32F  HIPBLAS_R_32F
#تعریف_کړه CUBLAS_SIDE_ښی HIPBLAS_SIDE_ښی
#تعریف_کړه CUBLAS_FILL_ډېرپېښ_پاسنی HIPBLAS_FILL_ډېرپېښ_پاسنی
#تعریف_کړه CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#تعریف_کړه CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظه_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_حافظه_Management_Supported
#تعریف_کړه CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_ویش_Granularity_Recommended
#تعریف_کړه CU_MEM_ویش_TYPE_PINNED hip_Mem_ویش_Type_Pinned
#تعریف_کړه CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#تعریف_کړه CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#تعریف_کړه CU_CHECK(fn) {hipError_t err = fn; إذا(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#تعریف_کړه __shfl_sync(ماسک, var, lane_ماسک, پلنوالی) __shfl(var, lane_ماسک, پلنوالی)
#تعریف_کړه __shfl_up_sync(ماسک, var, lane_ماسک, پلنوالی) __shfl_up(var, lane_ماسک, پلنوالی)
#تعریف_کړه __shfl_xor_sync(ماسک, var, lane_ماسک, پلنوالی) __shfl_xor(var, lane_ماسک, پلنوالی)
#تعریف_کړه __all_sync(ماسک, var) __all(var)
#تعریف_کړه __any_sync(ماسک, var) __any(var)
#تعریف_کړه cublasStrsmBatched hipblasStrsmBatched
#تعریف_کړه cublasCreate hipblasCreate
#تعریف_کړه cublasDestroy hipblasDestroy
#تعریف_کړه cublasGemmEx hipblasGemmEx
#تعریف_کړه cublasGemmBatchedEx hipblasGemmBatchedEx
#تعریف_کړه cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#تعریف_کړه cublasHandle_t hipblasHandle_t
#تعریف_کړه cublas_سېټ_Math_ډېرپېښ(handle, ډېرپېښ) CUBLAS_حالت_SUCCESS
#تعریف_کړه cublas_سېټ_بهیر hipblas_سېټ_بهیر
#تعریف_کړه cublasSgemm hipblasSgemm
#تعریف_کړه cublasSgemmBatched hipblasSgemmBatched
#تعریف_کړه cublasSgemmStridedBatched hipblasSgemmStridedBatched
#تعریف_کړه cublas_حالت_t hipblas_حالت_t
#تعریف_کړه cublasOperation_t hipblasOperation_t
#تعریف_کړه cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#تعریف_کړه cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#تعریف_کړه cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#تعریف_کړه cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#تعریف_کړه cudaDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کړه cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#تعریف_کړه cudaDeviceProp hipDeviceProp_t
#تعریف_کړه cudaDeviceSynchronize hipDeviceSynchronize
#تعریف_کړه cudaError_t hipError_t
#تعریف_کړه cuda_Error_حافظه_ویش hip_Error_بهر_د_حافظه
#تعریف_کړه cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#تعریف_کړه cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#تعریف_کړه cudaEventCreateWithFlags hipEventCreateWithFlags
#تعریف_کړه cudaEventDisableTiming hipEventDisableTiming
#تعریف_کړه cuda_Event_ریکارډ hip_Event_ریکارډ
#تعریف_کړه cudaEventSynchronize hipEventSynchronize
#تعریف_کړه cudaEvent_t hipEvent_t
#تعریف_کړه cudaEventDestroy hipEventDestroy
#تعریف_کړه cudaFree hipFree
#تعریف_کړه cudaFreeHost hipHostFree
#تعریف_کړه cudaGetDevice hipGetDevice
#تعریف_کړه cuda_Get_Device_شمېر hip_Get_Device_شمېر
#تعریف_کړه cudaGetDeviceProperties hipGetDeviceProperties
#تعریف_کړه cudaGetErrorString hipGetErrorString
#تعریف_کړه cuda_Get_وروستی_Error hip_Get_وروستی_Error
#تعریف_کړه cudaHostAlloc hipHostMalloc
#تعریف_کړه cudaHostAllocPortable hipHostMallocPortable
#تعریف_کړه cudaHostAllocMapped hipHostMallocMapped
#تعریف_کړه cuda_Host_Get_Device_اشاره_ګر hip_Host_Get_Device_اشاره_ګر
#تعریف_کړه cuda_Host_ثبتوونکی hip_Host_ثبتوونکی
#تعریف_کړه cuda_Host_ثبتوونکی_Portable hip_Host_ثبتوونکی_Portable
#تعریف_کړه cuda_Host_ثبتوونکی_Read_Only hip_Host_ثبتوونکی_Read_Only
#تعریف_کړه cudaHostUnregister hipHostUnregister
#تعریف_کړه cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#تعریف_کړه cudaLaunchHostFunc hipLaunchHostFunc
#تعریف_کړه cudaMalloc hipMalloc
#تعریف_کړه cudaMallocHost(ptr, اندازه) hipHostMalloc(ptr, اندازه, hipHostMallocDefault)
#تعریف_کړه cudaMallocManaged hipMallocManaged
#تعریف_کړه cudaMemAdvise hipMemAdvise
#تعریف_کړه cudaMemcpy hipMemcpy
#تعریف_کړه cudaMemcpyAsync hipMemcpyAsync
#تعریف_کړه cudaMemcpyPeerAsync hipMemcpyPeerAsync
#تعریف_کړه cudaMemcpy2DAsync hipMemcpy2DAsync
#تعریف_کړه cuda_Memcpy_Device_منزل_Device hip_Memcpy_Device_منزل_Device
#تعریف_کړه cuda_Memcpy_Device_منزل_Host hip_Memcpy_Device_منزل_Host
#تعریف_کړه cuda_Memcpy_Host_منزل_Device hip_Memcpy_Host_منزل_Device
#تعریف_کړه cuda_Memcpy_ډول hip_Memcpy_ډول
#تعریف_کړه cudaMemset hipMemset
#تعریف_کړه cudaMemsetAsync hipMemsetAsync
#تعریف_کړه cudaMemGetInfo hipMemGetInfo
#تعریف_کړه cuda_Occupancy_Max_Potential_بلاک_اندازه hip_Occupancy_Max_Potential_بلاک_اندازه
#تعریف_کړه cuda_سېټ_Device hip_سېټ_Device
#تعریف_کړه cuDeviceGet hipDeviceGet
#تعریف_کړه CUdevice hipDevice_t
#تعریف_کړه CUdeviceptr hipDeviceptr_t
#تعریف_کړه cuMemUnmap hipMemUnmap
#تعریف_کړه CUmemAccessDesc hipMemAccessDesc
#تعریف_کړه cu_Mem_پته_Free hip_Mem_پته_Free
#تعریف_کړه cuMemRelease hipMemRelease
#تعریف_کړه CUmem_Generic_ویش_Handle hip_Mem_Generic_ویش_Handle_t
#تعریف_کړه cuMemCreate hipMemCreate
#تعریف_کړه cu_Mem_پته_Reserve hip_Mem_پته_Reserve
#تعریف_کړه cuMemMap hipMemMap
#تعریف_کړه cu_Mem_سېټ_Access hip_Mem_سېټ_Access
#تعریف_کړه cu_Mem_Get_ویش_Granularity hip_Mem_Get_ویش_Granularity
#تعریف_کړه CUmem_ویش_Prop hip_Mem_ویش_Prop
#تعریف_کړه cuDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کړه cuda_بهیر_Create_With_Flags hip_بهیر_Create_With_Flags
#تعریف_کړه cuda_بهیر_Destroy hip_بهیر_Destroy
#تعریف_کړه cuda_بهیر_Fire_And_Forget hip_بهیر_Fire_And_Forget
#تعریف_کړه cuda_بهیر_Non_Blocking hip_بهیر_Non_Blocking
#تعریف_کړه cuda_بهیر_فی_Thread hip_بهیر_فی_Thread
#تعریف_کړه cuda_بهیر_Synchronize hip_بهیر_Synchronize
#تعریف_کړه cuda_بهیر_Wait_Event hip_بهیر_Wait_Event
#تعریف_کړه cuda_ګراف_Exec_t hip_ګراف_Exec_t
#تعریف_کړه cuda_ګراف_غوټه_t hip_ګراف_غوټه_t
#تعریف_کړه cuda_Kernel_غوټه_Params hip_Kernel_غوټه_Params
#تعریف_کړه cuda_Kernel_غوټه_Params hip_Kernel_غوټه_Params
#تعریف_کړه cuda_ګراف_Exec_Destroy hip_ګراف_Exec_Destroy
#تعریف_کړه cuda_ګراف_Launch hip_ګراف_Launch
#تعریف_کړه cuda_Error_ګراف_Exec_Update_Failure hip_Error_ګراف_Exec_Update_Failure
#تعریف_کړه cuda_ګراف_Exec_Update_پایله hip_ګراف_Exec_Update_پایله
#تعریف_کړه cuda_ګراف_غوټه_Type hip_ګراف_غوټه_Type
#تعریف_کړه cuda_ګراف_غوټه_Type_Kernel hip_ګراف_غوټه_Type_Kernel
#تعریف_کړه cuda_ګراف_Instantiate hip_ګراف_Instantiate
#تعریف_کړه cuda_بهیر_پای_ټکی_Capture hip_بهیر_پای_ټکی_Capture
#تعریف_کړه cuda_ګراف_Destroy hip_ګراف_Destroy
#تعریف_کړه cuda_ګراف_Kernel_غوټه_سېټ_Params hip_ګراف_Kernel_غوټه_سېټ_Params
#تعریف_کړه cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#تعریف_کړه cuda_ګراف_Kernel_غوټه_Get_Params hip_ګراف_Kernel_غوټه_Get_Params
#تعریف_کړه cuda_ګراف_غوټه_Get_Type hip_ګراف_غوټه_Get_Type
#تعریف_کړه cuda_ګراف_Get_Nodes hip_ګراف_Get_Nodes
#تعریف_کړه cuda_ګراف_Exec_Update hip_ګراف_Exec_Update
#تعریف_کړه cuda_بهیر_Capture_ډېرپېښ_Relaxed hip_بهیر_Capture_ډېرپېښ_Relaxed
#تعریف_کړه cuda_بهیر_Begin_Capture hip_بهیر_Begin_Capture
#تعریف_کړه cuda_ګراف_t hip_ګراف_t
#تعریف_کړه cuda_بهیر_t hip_بهیر_t
#تعریف_کړه cudaSuccess hipSuccess
#تعریف_کړه cuda_Occupancy_Max_فعال_بلاکونه_فی_Multiprocessor hip_Occupancy_Max_فعال_بلاکونه_فی_Multiprocessor
#تعریف_کړه cuda_Func_سېټ_Attribute hip_Func_سېټ_Attribute
#تعریف_کړه cuda_Func_Attribute_Max_Dynamic_Shared_حافظه_اندازه hip_Func_Attribute_Max_Dynamic_Shared_حافظه_اندازه
#تعریف_کړه __trap() افعل { abort(); __builtin_unreachable(); } طالما(0)
#تعریف_کړه CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#تعریف_کړه CUBLAS_حالت_NOT_INITIALIZED HIPBLAS_حالت_NOT_INITIALIZED
#تعریف_کړه CUBLAS_حالت_ALLOC_FAILED HIPBLAS_حالت_ALLOC_FAILED
#تعریف_کړه CUBLAS_حالت_INVALID_ارزښت HIPBLAS_حالت_INVALID_ارزښت
#تعریف_کړه CUBLAS_حالت_ARCH_MISMATCH HIPBLAS_حالت_ARCH_MISMATCH
#تعریف_کړه CUBLAS_حالت_MAPPING_ERROR HIPBLAS_حالت_MAPPING_ERROR
#تعریف_کړه CUBLAS_حالت_EXECUTION_FAILED HIPBLAS_حالت_EXECUTION_FAILED
#تعریف_کړه CUBLAS_حالت_دننی_ERROR HIPBLAS_حالت_دننی_ERROR
#تعریف_کړه CUBLAS_حالت_NOT_SUPPORTED HIPBLAS_حالت_NOT_SUPPORTED

#إذا HIP_نسخه >= 60500000
#تعریف_کړه CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#تعریف_کړه CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#تعریف_کړه CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#تعریف_کړه cublasComputeType_t hipblasComputeType_t
#تعریف_کړه cuda_معلومات_Type_t hip_معلومات_Type
#وإلا
#تعریف_کړه CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#تعریف_کړه CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#تعریف_کړه CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#تعریف_کړه cublasComputeType_t hipblasDatatype_t
#تعریف_کړه cuda_معلومات_Type_t hipblasDatatype_t
#د_که_پای // HIP_VERSION >= 6050000

#إذا !defined(__HIP_PLATFORM_AMD__)
#تېروتنه "The HIP backend supports only AMD targets"
#د_که_پای // !defined(__HIP_PLATFORM_AMD__)

#تعریف_کړه __CUDA_ARCH__ 1300

#إذا defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#تعریف_کړه GCN5
#د_که_پای // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#إذا defined(__gfx803__)
#تعریف_کړه GCN4
#د_که_پای // defined(__gfx803__)

#إذا defined(GCN5) || defined(GCN4)
#تعریف_کړه GCN
#د_که_پای // defined(GCN5) || defined(GCN4)

#إذا defined(__gfx950__)
#تعریف_کړه CDNA4
#د_که_پای // defined(__gfx950__)

#إذا defined(__gfx942__)
#تعریف_کړه CDNA3
#د_که_پای // defined(__gfx942__)

#إذا defined(__gfx90a__)
#تعریف_کړه CDNA2
#د_که_پای // defined(__gfx90a__)

#إذا defined(__gfx908__)
#تعریف_کړه CDNA1
#د_که_پای // defined(__gfx908__)

#إذا defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#تعریف_کړه CDNA // For the entire family
#د_که_پای // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#إذا defined(__GFX12__)
#تعریف_کړه RDNA4
#د_که_پای // defined(__GFX12__)

#إذا defined(__GFX11__)
#تعریف_کړه RDNA3
#د_که_پای // defined(__GFX11__)

#إذا defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#تعریف_کړه RDNA3_5
#د_که_پای // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#إذا defined(RDNA3) && !defined(RDNA3_5)
#تعریف_کړه RDNA3_0
#د_که_پای // defined(RDNA3) && !defined(RDNA3_5)

#إذا defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#تعریف_کړه RDNA2
#د_که_پای

#إذا defined(__gfx1010__) || defined(__gfx1012__)
#تعریف_کړه RDNA1
#د_که_پای // defined(__gfx1010__) || defined(__gfx1012__)

#إذا defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#تعریف_کړه RDNA // For the entire family
#د_که_پای // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#که_نه_تعریف_شوی __has_builtin
    #تعریف_کړه __has_builtin(x) 0
#د_که_پای

تعريفنوع __hip_bfloat16 nv_bfloat16;
تعريفنوع __hip_bfloat162 nv_bfloat162;

#إذا HIP_نسخه >= 60300000
#ضمن <hip/hip_fp8.h>
تعريفنوع __hip_fp8_e4m3 __nv_fp8_e4m3;
#تعریف_کړه FP8_شته
#د_که_پای // HIP_VERSION >= 60300000

تعريفنوع int8_t int8x4_t __attribute__((ext_ویکټور_type(4)));
تعريفنوع uint8_t uint8x4_t __attribute__((ext_ویکټور_type(4)));
ساكن __device__ __forceinline__ صحيح __vsubss4(ثابت صحيح a, ثابت صحيح b) {
    ثابت int8x4_t va = reinterpret_cast<ثابت int8x4_t&>(a);
    ثابت int8x4_t vb = reinterpret_cast<ثابت int8x4_t&>(b);
#إذا __has_builtin(__builtin_elementwise_sub_sat)
    ثابت int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    أرجع reinterpret_cast<ثابت صحيح &>(c);
#وإلا
    int8x4_t c;
    int16_t tmp;
#لارښوونه unroll
    لكل (صحيح i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        إذا(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        إذا(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    أرجع reinterpret_cast<صحيح &>(c);
#د_که_پای // __has_builtin(__builtin_elementwise_sub_sat)
}

ساكن __device__ __forceinline__ صحيح __vsub4(ثابت صحيح a, ثابت صحيح b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ثابت بینښه صحيح a_large = a | 0x80808080;
    ثابت بینښه صحيح b_small = b & 0x7f7f7f7f;
    ثابت بینښه صحيح پایله_ټیټ_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ثابت بینښه صحيح should_flip_لوړ_1bit = (a ^ ~b) & 0x80808080;

    أرجع پایله_ټیټ_7bits ^ should_flip_لوړ_1bit;
}

ساكن __device__ __forceinline__ بینښه صحيح __vcmpeq4(بینښه صحيح a, بینښه صحيح b) {
    ثابت uint8x4_t& va = reinterpret_cast<ثابت uint8x4_t&>(a);
    ثابت uint8x4_t& vb = reinterpret_cast<ثابت uint8x4_t&>(b);
    بینښه صحيح c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#لارښوونه unroll
    لكل (صحيح i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    أرجع c;
}

ساكن __device__ __forceinline__ بینښه صحيح __vcmpne4(بینښه صحيح a, بینښه صحيح b) {
    ثابت بینښه صحيح x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ثابت بینښه صحيح ne_ټیټ_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ثابت بینښه صحيح ne_لوړ_1bit = x & 0x80808080;
    ثابت بینښه صحيح ne_any_بټ = ne_ټیټ_7bits | ne_لوړ_1bit;

    أرجع (ne_any_بټ >> 7) * 0xff;
}
