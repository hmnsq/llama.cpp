#ہدایت once

#تعریف_کرو HIP_DISABLE_WARP_SYNC_BUILTINS 1
#شامل <hip/hip_runtime.h>
#شامل <hipblas/hipblas.h>
#شامل <hip/hip_fp16.h>
#شامل <hip/hip_bf16.h>

#اگر_تعریف_شدہ GGML_USE_NCCL
#شامل <rccl/rccl.h>
#اگر_ختم // GGML_USE_NCCL


#تعریف_کرو CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#تعریف_کرو CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#تعریف_کرو CUBLAS_OP_N HIPBLAS_OP_N
#تعریف_کرو CUBLAS_OP_T HIPBLAS_OP_T
#تعریف_کرو CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#تعریف_کرو CUBLAS_TF32_TENSOR_OP_MATH 0
#تعریف_کرو CUDA_R_16F  HIPBLAS_R_16F
#تعریف_کرو CUDA_R_16BF HIPBLAS_R_16B
#تعریف_کرو CUDA_R_32F  HIPBLAS_R_32F
#تعریف_کرو CUBLAS_SIDE_دایاں HIPBLAS_SIDE_دایاں
#تعریف_کرو CUBLAS_FILL_منوال_بالائی HIPBLAS_FILL_منوال_بالائی
#تعریف_کرو CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#تعریف_کرو CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظہ_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_حافظہ_Management_Supported
#تعریف_کرو CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_تخصیص_Granularity_Recommended
#تعریف_کرو CU_MEM_تخصیص_TYPE_PINNED hip_Mem_تخصیص_Type_Pinned
#تعریف_کرو CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#تعریف_کرو CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#تعریف_کرو CU_CHECK(fn) {hipError_t err = fn; اگر(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#تعریف_کرو __shfl_sync(نقاب, var, lane_نقاب, چوڑائی) __shfl(var, lane_نقاب, چوڑائی)
#تعریف_کرو __shfl_up_sync(نقاب, var, lane_نقاب, چوڑائی) __shfl_up(var, lane_نقاب, چوڑائی)
#تعریف_کرو __shfl_xor_sync(نقاب, var, lane_نقاب, چوڑائی) __shfl_xor(var, lane_نقاب, چوڑائی)
#تعریف_کرو __all_sync(نقاب, var) __all(var)
#تعریف_کرو __any_sync(نقاب, var) __any(var)
#تعریف_کرو cublasStrsmBatched hipblasStrsmBatched
#تعریف_کرو cublasCreate hipblasCreate
#تعریف_کرو cublasDestroy hipblasDestroy
#تعریف_کرو cublasGemmEx hipblasGemmEx
#تعریف_کرو cublasGemmBatchedEx hipblasGemmBatchedEx
#تعریف_کرو cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#تعریف_کرو cublasHandle_t hipblasHandle_t
#تعریف_کرو cublas_سیٹ_Math_منوال(handle, منوال) CUBLAS_حالت_SUCCESS
#تعریف_کرو cublas_سیٹ_دھارا hipblas_سیٹ_دھارا
#تعریف_کرو cublasSgemm hipblasSgemm
#تعریف_کرو cublasSgemmBatched hipblasSgemmBatched
#تعریف_کرو cublasSgemmStridedBatched hipblasSgemmStridedBatched
#تعریف_کرو cublas_حالت_t hipblas_حالت_t
#تعریف_کرو cublasOperation_t hipblasOperation_t
#تعریف_کرو cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#تعریف_کرو cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#تعریف_کرو cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#تعریف_کرو cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#تعریف_کرو cudaDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کرو cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#تعریف_کرو cudaDeviceProp hipDeviceProp_t
#تعریف_کرو cudaDeviceSynchronize hipDeviceSynchronize
#تعریف_کرو cudaError_t hipError_t
#تعریف_کرو cuda_Error_حافظہ_تخصیص hip_Error_برآمد_کا_حافظہ
#تعریف_کرو cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#تعریف_کرو cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#تعریف_کرو cudaEventCreateWithFlags hipEventCreateWithFlags
#تعریف_کرو cudaEventDisableTiming hipEventDisableTiming
#تعریف_کرو cuda_Event_ریکارڈ hip_Event_ریکارڈ
#تعریف_کرو cudaEventSynchronize hipEventSynchronize
#تعریف_کرو cudaEvent_t hipEvent_t
#تعریف_کرو cudaEventDestroy hipEventDestroy
#تعریف_کرو cudaFree hipFree
#تعریف_کرو cudaFreeHost hipHostFree
#تعریف_کرو cudaGetDevice hipGetDevice
#تعریف_کرو cuda_Get_Device_گنتی hip_Get_Device_گنتی
#تعریف_کرو cudaGetDeviceProperties hipGetDeviceProperties
#تعریف_کرو cudaGetErrorString hipGetErrorString
#تعریف_کرو cuda_Get_آخری_Error hip_Get_آخری_Error
#تعریف_کرو cudaHostAlloc hipHostMalloc
#تعریف_کرو cudaHostAllocPortable hipHostMallocPortable
#تعریف_کرو cudaHostAllocMapped hipHostMallocMapped
#تعریف_کرو cuda_Host_Get_Device_اشارہ_گر hip_Host_Get_Device_اشارہ_گر
#تعریف_کرو cuda_Host_رجسٹر hip_Host_رجسٹر
#تعریف_کرو cuda_Host_رجسٹر_Portable hip_Host_رجسٹر_Portable
#تعریف_کرو cuda_Host_رجسٹر_Read_Only hip_Host_رجسٹر_Read_Only
#تعریف_کرو cudaHostUnregister hipHostUnregister
#تعریف_کرو cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#تعریف_کرو cudaLaunchHostFunc hipLaunchHostFunc
#تعریف_کرو cudaMalloc hipMalloc
#تعریف_کرو cudaMallocHost(ptr, سائز) hipHostMalloc(ptr, سائز, hipHostMallocDefault)
#تعریف_کرو cudaMallocManaged hipMallocManaged
#تعریف_کرو cudaMemAdvise hipMemAdvise
#تعریف_کرو cudaMemcpy hipMemcpy
#تعریف_کرو cudaMemcpyAsync hipMemcpyAsync
#تعریف_کرو cudaMemcpyPeerAsync hipMemcpyPeerAsync
#تعریف_کرو cudaMemcpy2DAsync hipMemcpy2DAsync
#تعریف_کرو cuda_Memcpy_Device_مقصد_Device hip_Memcpy_Device_مقصد_Device
#تعریف_کرو cuda_Memcpy_Device_مقصد_Host hip_Memcpy_Device_مقصد_Host
#تعریف_کرو cuda_Memcpy_Host_مقصد_Device hip_Memcpy_Host_مقصد_Device
#تعریف_کرو cuda_Memcpy_قسم hip_Memcpy_قسم
#تعریف_کرو cudaMemset hipMemset
#تعریف_کرو cudaMemsetAsync hipMemsetAsync
#تعریف_کرو cudaMemGetInfo hipMemGetInfo
#تعریف_کرو cuda_Occupancy_Max_Potential_بلاک_سائز hip_Occupancy_Max_Potential_بلاک_سائز
#تعریف_کرو cuda_سیٹ_Device hip_سیٹ_Device
#تعریف_کرو cuDeviceGet hipDeviceGet
#تعریف_کرو CUdevice hipDevice_t
#تعریف_کرو CUdeviceptr hipDeviceptr_t
#تعریف_کرو cuMemUnmap hipMemUnmap
#تعریف_کرو CUmemAccessDesc hipMemAccessDesc
#تعریف_کرو cu_Mem_پتہ_Free hip_Mem_پتہ_Free
#تعریف_کرو cuMemRelease hipMemRelease
#تعریف_کرو CUmem_Generic_تخصیص_Handle hip_Mem_Generic_تخصیص_Handle_t
#تعریف_کرو cuMemCreate hipMemCreate
#تعریف_کرو cu_Mem_پتہ_Reserve hip_Mem_پتہ_Reserve
#تعریف_کرو cuMemMap hipMemMap
#تعریف_کرو cu_Mem_سیٹ_Access hip_Mem_سیٹ_Access
#تعریف_کرو cu_Mem_Get_تخصیص_Granularity hip_Mem_Get_تخصیص_Granularity
#تعریف_کرو CUmem_تخصیص_Prop hip_Mem_تخصیص_Prop
#تعریف_کرو cuDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کرو cuda_دھارا_Create_With_Flags hip_دھارا_Create_With_Flags
#تعریف_کرو cuda_دھارا_Destroy hip_دھارا_Destroy
#تعریف_کرو cuda_دھارا_Fire_And_Forget hip_دھارا_Fire_And_Forget
#تعریف_کرو cuda_دھارا_Non_Blocking hip_دھارا_Non_Blocking
#تعریف_کرو cuda_دھارا_فی_Thread hip_دھارا_فی_Thread
#تعریف_کرو cuda_دھارا_Synchronize hip_دھارا_Synchronize
#تعریف_کرو cuda_دھارا_Wait_Event hip_دھارا_Wait_Event
#تعریف_کرو cuda_گراف_Exec_t hip_گراف_Exec_t
#تعریف_کرو cuda_گراف_گرہ_t hip_گراف_گرہ_t
#تعریف_کرو cuda_Kernel_گرہ_Params hip_Kernel_گرہ_Params
#تعریف_کرو cuda_Kernel_گرہ_Params hip_Kernel_گرہ_Params
#تعریف_کرو cuda_گراف_Exec_Destroy hip_گراف_Exec_Destroy
#تعریف_کرو cuda_گراف_Launch hip_گراف_Launch
#تعریف_کرو cuda_Error_گراف_Exec_Update_Failure hip_Error_گراف_Exec_Update_Failure
#تعریف_کرو cuda_گراف_Exec_Update_نتیجہ hip_گراف_Exec_Update_نتیجہ
#تعریف_کرو cuda_گراف_گرہ_Type hip_گراف_گرہ_Type
#تعریف_کرو cuda_گراف_گرہ_Type_Kernel hip_گراف_گرہ_Type_Kernel
#تعریف_کرو cuda_گراف_Instantiate hip_گراف_Instantiate
#تعریف_کرو cuda_دھارا_انتہا_Capture hip_دھارا_انتہا_Capture
#تعریف_کرو cuda_گراف_Destroy hip_گراف_Destroy
#تعریف_کرو cuda_گراف_Kernel_گرہ_سیٹ_Params hip_گراف_Kernel_گرہ_سیٹ_Params
#تعریف_کرو cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#تعریف_کرو cuda_گراف_Kernel_گرہ_Get_Params hip_گراف_Kernel_گرہ_Get_Params
#تعریف_کرو cuda_گراف_گرہ_Get_Type hip_گراف_گرہ_Get_Type
#تعریف_کرو cuda_گراف_Get_Nodes hip_گراف_Get_Nodes
#تعریف_کرو cuda_گراف_Exec_Update hip_گراف_Exec_Update
#تعریف_کرو cuda_دھارا_Capture_منوال_Relaxed hip_دھارا_Capture_منوال_Relaxed
#تعریف_کرو cuda_دھارا_Begin_Capture hip_دھارا_Begin_Capture
#تعریف_کرو cuda_گراف_t hip_گراف_t
#تعریف_کرو cuda_دھارا_t hip_دھارا_t
#تعریف_کرو cudaSuccess hipSuccess
#تعریف_کرو cuda_Occupancy_Max_فعال_بلاکیں_فی_Multiprocessor hip_Occupancy_Max_فعال_بلاکیں_فی_Multiprocessor
#تعریف_کرو cuda_Func_سیٹ_Attribute hip_Func_سیٹ_Attribute
#تعریف_کرو cuda_Func_Attribute_Max_Dynamic_Shared_حافظہ_سائز hip_Func_Attribute_Max_Dynamic_Shared_حافظہ_سائز
#تعریف_کرو __trap() کرو { abort(); __builtin_unreachable(); } جب(0)
#تعریف_کرو CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#تعریف_کرو CUBLAS_حالت_NOT_INITIALIZED HIPBLAS_حالت_NOT_INITIALIZED
#تعریف_کرو CUBLAS_حالت_ALLOC_FAILED HIPBLAS_حالت_ALLOC_FAILED
#تعریف_کرو CUBLAS_حالت_INVALID_قدر HIPBLAS_حالت_INVALID_قدر
#تعریف_کرو CUBLAS_حالت_ARCH_MISMATCH HIPBLAS_حالت_ARCH_MISMATCH
#تعریف_کرو CUBLAS_حالت_MAPPING_ERROR HIPBLAS_حالت_MAPPING_ERROR
#تعریف_کرو CUBLAS_حالت_EXECUTION_FAILED HIPBLAS_حالت_EXECUTION_FAILED
#تعریف_کرو CUBLAS_حالت_اندرونی_ERROR HIPBLAS_حالت_اندرونی_ERROR
#تعریف_کرو CUBLAS_حالت_NOT_SUPPORTED HIPBLAS_حالت_NOT_SUPPORTED

#اگر HIP_نسخہ >= 60500000
#تعریف_کرو CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#تعریف_کرو CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#تعریف_کرو CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#تعریف_کرو cublasComputeType_t hipblasComputeType_t
#تعریف_کرو cuda_ڈیٹا_Type_t hip_ڈیٹا_Type
#ورنہ
#تعریف_کرو CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#تعریف_کرو CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#تعریف_کرو CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#تعریف_کرو cublasComputeType_t hipblasDatatype_t
#تعریف_کرو cuda_ڈیٹا_Type_t hipblasDatatype_t
#اگر_ختم // HIP_VERSION >= 6050000

#اگر !defined(__HIP_PLATFORM_AMD__)
#غلطی "The HIP backend supports only AMD targets"
#اگر_ختم // !defined(__HIP_PLATFORM_AMD__)

#تعریف_کرو __CUDA_ARCH__ 1300

#اگر defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#تعریف_کرو GCN5
#اگر_ختم // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#اگر defined(__gfx803__)
#تعریف_کرو GCN4
#اگر_ختم // defined(__gfx803__)

#اگر defined(GCN5) || defined(GCN4)
#تعریف_کرو GCN
#اگر_ختم // defined(GCN5) || defined(GCN4)

#اگر defined(__gfx950__)
#تعریف_کرو CDNA4
#اگر_ختم // defined(__gfx950__)

#اگر defined(__gfx942__)
#تعریف_کرو CDNA3
#اگر_ختم // defined(__gfx942__)

#اگر defined(__gfx90a__)
#تعریف_کرو CDNA2
#اگر_ختم // defined(__gfx90a__)

#اگر defined(__gfx908__)
#تعریف_کرو CDNA1
#اگر_ختم // defined(__gfx908__)

#اگر defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#تعریف_کرو CDNA // For the entire family
#اگر_ختم // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#اگر defined(__GFX12__)
#تعریف_کرو RDNA4
#اگر_ختم // defined(__GFX12__)

#اگر defined(__GFX11__)
#تعریف_کرو RDNA3
#اگر_ختم // defined(__GFX11__)

#اگر defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#تعریف_کرو RDNA3_5
#اگر_ختم // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#اگر defined(RDNA3) && !defined(RDNA3_5)
#تعریف_کرو RDNA3_0
#اگر_ختم // defined(RDNA3) && !defined(RDNA3_5)

#اگر defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#تعریف_کرو RDNA2
#اگر_ختم

#اگر defined(__gfx1010__) || defined(__gfx1012__)
#تعریف_کرو RDNA1
#اگر_ختم // defined(__gfx1010__) || defined(__gfx1012__)

#اگر defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#تعریف_کرو RDNA // For the entire family
#اگر_ختم // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#اگر_غیر_تعریف_شدہ __has_builtin
    #تعریف_کرو __has_builtin(x) 0
#اگر_ختم

قسمتعریف __hip_bfloat16 nv_bfloat16;
قسمتعریف __hip_bfloat162 nv_bfloat162;

#اگر HIP_نسخہ >= 60300000
#شامل <hip/hip_fp8.h>
قسمتعریف __hip_fp8_e4m3 __nv_fp8_e4m3;
#تعریف_کرو FP8_دستیاب
#اگر_ختم // HIP_VERSION >= 60300000

قسمتعریف int8_t int8x4_t __attribute__((ext_سمتیہ_type(4)));
قسمتعریف uint8_t uint8x4_t __attribute__((ext_سمتیہ_type(4)));
ساکن __device__ __forceinline__ صحیح __vsubss4(ثابت صحیح a, ثابت صحیح b) {
    ثابت int8x4_t va = reinterpret_cast<ثابت int8x4_t&>(a);
    ثابت int8x4_t vb = reinterpret_cast<ثابت int8x4_t&>(b);
#اگر __has_builtin(__builtin_elementwise_sub_sat)
    ثابت int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    واپس reinterpret_cast<ثابت صحیح &>(c);
#ورنہ
    int8x4_t c;
    int16_t tmp;
#ہدایت unroll
    برائے (صحیح i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        اگر(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        اگر(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    واپس reinterpret_cast<صحیح &>(c);
#اگر_ختم // __has_builtin(__builtin_elementwise_sub_sat)
}

ساکن __device__ __forceinline__ صحیح __vsub4(ثابت صحیح a, ثابت صحیح b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ثابت غیرعلامتی صحیح a_large = a | 0x80808080;
    ثابت غیرعلامتی صحیح b_small = b & 0x7f7f7f7f;
    ثابت غیرعلامتی صحیح نتیجہ_کم_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ثابت غیرعلامتی صحیح should_flip_بلند_1bit = (a ^ ~b) & 0x80808080;

    واپس نتیجہ_کم_7bits ^ should_flip_بلند_1bit;
}

ساکن __device__ __forceinline__ غیرعلامتی صحیح __vcmpeq4(غیرعلامتی صحیح a, غیرعلامتی صحیح b) {
    ثابت uint8x4_t& va = reinterpret_cast<ثابت uint8x4_t&>(a);
    ثابت uint8x4_t& vb = reinterpret_cast<ثابت uint8x4_t&>(b);
    غیرعلامتی صحیح c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#ہدایت unroll
    برائے (صحیح i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    واپس c;
}

ساکن __device__ __forceinline__ غیرعلامتی صحیح __vcmpne4(غیرعلامتی صحیح a, غیرعلامتی صحیح b) {
    ثابت غیرعلامتی صحیح x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ثابت غیرعلامتی صحیح ne_کم_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ثابت غیرعلامتی صحیح ne_بلند_1bit = x & 0x80808080;
    ثابت غیرعلامتی صحیح ne_any_ثنائی_ہندسہ = ne_کم_7bits | ne_بلند_1bit;

    واپس (ne_any_ثنائی_ہندسہ >> 7) * 0xff;
}
