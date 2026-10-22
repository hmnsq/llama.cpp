#هدايت once

#وصف_ڪريو HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ضمن <hip/hip_runtime.h>
#ضمن <hipblas/hipblas.h>
#ضمن <hip/hip_fp16.h>
#ضمن <hip/hip_bf16.h>

#جيڪڏهن_وصف_ٿيل GGML_USE_NCCL
#ضمن <rccl/rccl.h>
#جيڪڏهن_پورو // GGML_USE_NCCL


#وصف_ڪريو CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#وصف_ڪريو CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#وصف_ڪريو CUBLAS_OP_N HIPBLAS_OP_N
#وصف_ڪريو CUBLAS_OP_T HIPBLAS_OP_T
#وصف_ڪريو CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#وصف_ڪريو CUBLAS_TF32_TENSOR_OP_MATH 0
#وصف_ڪريو CUDA_R_16F  HIPBLAS_R_16F
#وصف_ڪريو CUDA_R_16BF HIPBLAS_R_16B
#وصف_ڪريو CUDA_R_32F  HIPBLAS_R_32F
#وصف_ڪريو CUBLAS_SIDE_ساڄو HIPBLAS_SIDE_ساڄو
#وصف_ڪريو CUBLAS_FILL_گهڻائي_مٿانهون HIPBLAS_FILL_گهڻائي_مٿانهون
#وصف_ڪريو CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#وصف_ڪريو CU_DEVICE_ATTRIBUTE_VIRTUAL_يادگيري_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_يادگيري_Management_Supported
#وصف_ڪريو CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_ورهاست_Granularity_Recommended
#وصف_ڪريو CU_MEM_ورهاست_TYPE_PINNED hip_Mem_ورهاست_Type_Pinned
#وصف_ڪريو CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#وصف_ڪريو CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#وصف_ڪريو CU_CHECK(fn) {hipError_t err = fn; إذا(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#وصف_ڪريو __shfl_sync(نقاب, var, lane_نقاب, ويڪر) __shfl(var, lane_نقاب, ويڪر)
#وصف_ڪريو __shfl_up_sync(نقاب, var, lane_نقاب, ويڪر) __shfl_up(var, lane_نقاب, ويڪر)
#وصف_ڪريو __shfl_xor_sync(نقاب, var, lane_نقاب, ويڪر) __shfl_xor(var, lane_نقاب, ويڪر)
#وصف_ڪريو __all_sync(نقاب, var) __all(var)
#وصف_ڪريو __any_sync(نقاب, var) __any(var)
#وصف_ڪريو cublasStrsmBatched hipblasStrsmBatched
#وصف_ڪريو cublasCreate hipblasCreate
#وصف_ڪريو cublasDestroy hipblasDestroy
#وصف_ڪريو cublasGemmEx hipblasGemmEx
#وصف_ڪريو cublasGemmBatchedEx hipblasGemmBatchedEx
#وصف_ڪريو cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#وصف_ڪريو cublasHandle_t hipblasHandle_t
#وصف_ڪريو cublas_سيٽ_Math_گهڻائي(handle, گهڻائي) CUBLAS_حالت_SUCCESS
#وصف_ڪريو cublas_سيٽ_وهڪرو hipblas_سيٽ_وهڪرو
#وصف_ڪريو cublasSgemm hipblasSgemm
#وصف_ڪريو cublasSgemmBatched hipblasSgemmBatched
#وصف_ڪريو cublasSgemmStridedBatched hipblasSgemmStridedBatched
#وصف_ڪريو cublas_حالت_t hipblas_حالت_t
#وصف_ڪريو cublasOperation_t hipblasOperation_t
#وصف_ڪريو cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#وصف_ڪريو cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#وصف_ڪريو cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#وصف_ڪريو cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#وصف_ڪريو cudaDeviceGetAttribute hipDeviceGetAttribute
#وصف_ڪريو cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#وصف_ڪريو cudaDeviceProp hipDeviceProp_t
#وصف_ڪريو cudaDeviceSynchronize hipDeviceSynchronize
#وصف_ڪريو cudaError_t hipError_t
#وصف_ڪريو cuda_Error_يادگيري_ورهاست hip_Error_ٻاهريون_جو_يادگيري
#وصف_ڪريو cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#وصف_ڪريو cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#وصف_ڪريو cudaEventCreateWithFlags hipEventCreateWithFlags
#وصف_ڪريو cudaEventDisableTiming hipEventDisableTiming
#وصف_ڪريو cuda_Event_رڪارڊ hip_Event_رڪارڊ
#وصف_ڪريو cudaEventSynchronize hipEventSynchronize
#وصف_ڪريو cudaEvent_t hipEvent_t
#وصف_ڪريو cudaEventDestroy hipEventDestroy
#وصف_ڪريو cudaFree hipFree
#وصف_ڪريو cudaFreeHost hipHostFree
#وصف_ڪريو cudaGetDevice hipGetDevice
#وصف_ڪريو cuda_Get_Device_ڳڻپ hip_Get_Device_ڳڻپ
#وصف_ڪريو cudaGetDeviceProperties hipGetDeviceProperties
#وصف_ڪريو cudaGetErrorString hipGetErrorString
#وصف_ڪريو cuda_Get_آخري_Error hip_Get_آخري_Error
#وصف_ڪريو cudaHostAlloc hipHostMalloc
#وصف_ڪريو cudaHostAllocPortable hipHostMallocPortable
#وصف_ڪريو cudaHostAllocMapped hipHostMallocMapped
#وصف_ڪريو cuda_Host_Get_Device_اشارو hip_Host_Get_Device_اشارو
#وصف_ڪريو cuda_Host_رجسٽر hip_Host_رجسٽر
#وصف_ڪريو cuda_Host_رجسٽر_Portable hip_Host_رجسٽر_Portable
#وصف_ڪريو cuda_Host_رجسٽر_Read_Only hip_Host_رجسٽر_Read_Only
#وصف_ڪريو cudaHostUnregister hipHostUnregister
#وصف_ڪريو cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#وصف_ڪريو cudaLaunchHostFunc hipLaunchHostFunc
#وصف_ڪريو cudaMalloc hipMalloc
#وصف_ڪريو cudaMallocHost(ptr, سائيز) hipHostMalloc(ptr, سائيز, hipHostMallocDefault)
#وصف_ڪريو cudaMallocManaged hipMallocManaged
#وصف_ڪريو cudaMemAdvise hipMemAdvise
#وصف_ڪريو cudaMemcpy hipMemcpy
#وصف_ڪريو cudaMemcpyAsync hipMemcpyAsync
#وصف_ڪريو cudaMemcpyPeerAsync hipMemcpyPeerAsync
#وصف_ڪريو cudaMemcpy2DAsync hipMemcpy2DAsync
#وصف_ڪريو cuda_Memcpy_Device_منزل_Device hip_Memcpy_Device_منزل_Device
#وصف_ڪريو cuda_Memcpy_Device_منزل_Host hip_Memcpy_Device_منزل_Host
#وصف_ڪريو cuda_Memcpy_Host_منزل_Device hip_Memcpy_Host_منزل_Device
#وصف_ڪريو cuda_Memcpy_قسم hip_Memcpy_قسم
#وصف_ڪريو cudaMemset hipMemset
#وصف_ڪريو cudaMemsetAsync hipMemsetAsync
#وصف_ڪريو cudaMemGetInfo hipMemGetInfo
#وصف_ڪريو cuda_Occupancy_Max_Potential_بلاڪ_سائيز hip_Occupancy_Max_Potential_بلاڪ_سائيز
#وصف_ڪريو cuda_سيٽ_Device hip_سيٽ_Device
#وصف_ڪريو cuDeviceGet hipDeviceGet
#وصف_ڪريو CUdevice hipDevice_t
#وصف_ڪريو CUdeviceptr hipDeviceptr_t
#وصف_ڪريو cuMemUnmap hipMemUnmap
#وصف_ڪريو CUmemAccessDesc hipMemAccessDesc
#وصف_ڪريو cu_Mem_پتو_Free hip_Mem_پتو_Free
#وصف_ڪريو cuMemRelease hipMemRelease
#وصف_ڪريو CUmem_Generic_ورهاست_Handle hip_Mem_Generic_ورهاست_Handle_t
#وصف_ڪريو cuMemCreate hipMemCreate
#وصف_ڪريو cu_Mem_پتو_Reserve hip_Mem_پتو_Reserve
#وصف_ڪريو cuMemMap hipMemMap
#وصف_ڪريو cu_Mem_سيٽ_Access hip_Mem_سيٽ_Access
#وصف_ڪريو cu_Mem_Get_ورهاست_Granularity hip_Mem_Get_ورهاست_Granularity
#وصف_ڪريو CUmem_ورهاست_Prop hip_Mem_ورهاست_Prop
#وصف_ڪريو cuDeviceGetAttribute hipDeviceGetAttribute
#وصف_ڪريو cuda_وهڪرو_Create_With_Flags hip_وهڪرو_Create_With_Flags
#وصف_ڪريو cuda_وهڪرو_Destroy hip_وهڪرو_Destroy
#وصف_ڪريو cuda_وهڪرو_Fire_And_Forget hip_وهڪرو_Fire_And_Forget
#وصف_ڪريو cuda_وهڪرو_Non_Blocking hip_وهڪرو_Non_Blocking
#وصف_ڪريو cuda_وهڪرو_في_Thread hip_وهڪرو_في_Thread
#وصف_ڪريو cuda_وهڪرو_Synchronize hip_وهڪرو_Synchronize
#وصف_ڪريو cuda_وهڪرو_Wait_Event hip_وهڪرو_Wait_Event
#وصف_ڪريو cuda_گراف_Exec_t hip_گراف_Exec_t
#وصف_ڪريو cuda_گراف_ڳنڍ_t hip_گراف_ڳنڍ_t
#وصف_ڪريو cuda_Kernel_ڳنڍ_Params hip_Kernel_ڳنڍ_Params
#وصف_ڪريو cuda_Kernel_ڳنڍ_Params hip_Kernel_ڳنڍ_Params
#وصف_ڪريو cuda_گراف_Exec_Destroy hip_گراف_Exec_Destroy
#وصف_ڪريو cuda_گراف_Launch hip_گراف_Launch
#وصف_ڪريو cuda_Error_گراف_Exec_Update_Failure hip_Error_گراف_Exec_Update_Failure
#وصف_ڪريو cuda_گراف_Exec_Update_نتيجو hip_گراف_Exec_Update_نتيجو
#وصف_ڪريو cuda_گراف_ڳنڍ_Type hip_گراف_ڳنڍ_Type
#وصف_ڪريو cuda_گراف_ڳنڍ_Type_Kernel hip_گراف_ڳنڍ_Type_Kernel
#وصف_ڪريو cuda_گراف_Instantiate hip_گراف_Instantiate
#وصف_ڪريو cuda_وهڪرو_پڇاڙي_Capture hip_وهڪرو_پڇاڙي_Capture
#وصف_ڪريو cuda_گراف_Destroy hip_گراف_Destroy
#وصف_ڪريو cuda_گراف_Kernel_ڳنڍ_سيٽ_Params hip_گراف_Kernel_ڳنڍ_سيٽ_Params
#وصف_ڪريو cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#وصف_ڪريو cuda_گراف_Kernel_ڳنڍ_Get_Params hip_گراف_Kernel_ڳنڍ_Get_Params
#وصف_ڪريو cuda_گراف_ڳنڍ_Get_Type hip_گراف_ڳنڍ_Get_Type
#وصف_ڪريو cuda_گراف_Get_Nodes hip_گراف_Get_Nodes
#وصف_ڪريو cuda_گراف_Exec_Update hip_گراف_Exec_Update
#وصف_ڪريو cuda_وهڪرو_Capture_گهڻائي_Relaxed hip_وهڪرو_Capture_گهڻائي_Relaxed
#وصف_ڪريو cuda_وهڪرو_Begin_Capture hip_وهڪرو_Begin_Capture
#وصف_ڪريو cuda_گراف_t hip_گراف_t
#وصف_ڪريو cuda_وهڪرو_t hip_وهڪرو_t
#وصف_ڪريو cudaSuccess hipSuccess
#وصف_ڪريو cuda_Occupancy_Max_سرگرم_بلاڪون_في_Multiprocessor hip_Occupancy_Max_سرگرم_بلاڪون_في_Multiprocessor
#وصف_ڪريو cuda_Func_سيٽ_Attribute hip_Func_سيٽ_Attribute
#وصف_ڪريو cuda_Func_Attribute_Max_Dynamic_Shared_يادگيري_سائيز hip_Func_Attribute_Max_Dynamic_Shared_يادگيري_سائيز
#وصف_ڪريو __trap() افعل { abort(); __builtin_unreachable(); } طالما(0)
#وصف_ڪريو CUBLAS_حالت_SUCCESS HIPBLAS_حالت_SUCCESS
#وصف_ڪريو CUBLAS_حالت_NOT_INITIALIZED HIPBLAS_حالت_NOT_INITIALIZED
#وصف_ڪريو CUBLAS_حالت_ALLOC_FAILED HIPBLAS_حالت_ALLOC_FAILED
#وصف_ڪريو CUBLAS_حالت_INVALID_قدر HIPBLAS_حالت_INVALID_قدر
#وصف_ڪريو CUBLAS_حالت_ARCH_MISMATCH HIPBLAS_حالت_ARCH_MISMATCH
#وصف_ڪريو CUBLAS_حالت_MAPPING_ERROR HIPBLAS_حالت_MAPPING_ERROR
#وصف_ڪريو CUBLAS_حالت_EXECUTION_FAILED HIPBLAS_حالت_EXECUTION_FAILED
#وصف_ڪريو CUBLAS_حالت_اندروني_ERROR HIPBLAS_حالت_اندروني_ERROR
#وصف_ڪريو CUBLAS_حالت_NOT_SUPPORTED HIPBLAS_حالت_NOT_SUPPORTED

#إذا HIP_نسخو >= 60500000
#وصف_ڪريو CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#وصف_ڪريو CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#وصف_ڪريو CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#وصف_ڪريو cublasComputeType_t hipblasComputeType_t
#وصف_ڪريو cuda_معلومات_Type_t hip_معلومات_Type
#وإلا
#وصف_ڪريو CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#وصف_ڪريو CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#وصف_ڪريو CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#وصف_ڪريو cublasComputeType_t hipblasDatatype_t
#وصف_ڪريو cuda_معلومات_Type_t hipblasDatatype_t
#جيڪڏهن_پورو // HIP_VERSION >= 6050000

#إذا !defined(__HIP_PLATFORM_AMD__)
#غلطي "The HIP backend supports only AMD targets"
#جيڪڏهن_پورو // !defined(__HIP_PLATFORM_AMD__)

#وصف_ڪريو __CUDA_ARCH__ 1300

#إذا defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#وصف_ڪريو GCN5
#جيڪڏهن_پورو // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#إذا defined(__gfx803__)
#وصف_ڪريو GCN4
#جيڪڏهن_پورو // defined(__gfx803__)

#إذا defined(GCN5) || defined(GCN4)
#وصف_ڪريو GCN
#جيڪڏهن_پورو // defined(GCN5) || defined(GCN4)

#إذا defined(__gfx950__)
#وصف_ڪريو CDNA4
#جيڪڏهن_پورو // defined(__gfx950__)

#إذا defined(__gfx942__)
#وصف_ڪريو CDNA3
#جيڪڏهن_پورو // defined(__gfx942__)

#إذا defined(__gfx90a__)
#وصف_ڪريو CDNA2
#جيڪڏهن_پورو // defined(__gfx90a__)

#إذا defined(__gfx908__)
#وصف_ڪريو CDNA1
#جيڪڏهن_پورو // defined(__gfx908__)

#إذا defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#وصف_ڪريو CDNA // For the entire family
#جيڪڏهن_پورو // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#إذا defined(__GFX12__)
#وصف_ڪريو RDNA4
#جيڪڏهن_پورو // defined(__GFX12__)

#إذا defined(__GFX11__)
#وصف_ڪريو RDNA3
#جيڪڏهن_پورو // defined(__GFX11__)

#إذا defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#وصف_ڪريو RDNA3_5
#جيڪڏهن_پورو // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#إذا defined(RDNA3) && !defined(RDNA3_5)
#وصف_ڪريو RDNA3_0
#جيڪڏهن_پورو // defined(RDNA3) && !defined(RDNA3_5)

#إذا defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#وصف_ڪريو RDNA2
#جيڪڏهن_پورو

#إذا defined(__gfx1010__) || defined(__gfx1012__)
#وصف_ڪريو RDNA1
#جيڪڏهن_پورو // defined(__gfx1010__) || defined(__gfx1012__)

#إذا defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#وصف_ڪريو RDNA // For the entire family
#جيڪڏهن_پورو // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#جيڪڏهن_وصف_نه_ٿيل __has_builtin
    #وصف_ڪريو __has_builtin(x) 0
#جيڪڏهن_پورو

تعريفنوع __hip_bfloat16 nv_bfloat16;
تعريفنوع __hip_bfloat162 nv_bfloat162;

#إذا HIP_نسخو >= 60300000
#ضمن <hip/hip_fp8.h>
تعريفنوع __hip_fp8_e4m3 __nv_fp8_e4m3;
#وصف_ڪريو FP8_موجود
#جيڪڏهن_پورو // HIP_VERSION >= 60300000

تعريفنوع int8_t int8x4_t __attribute__((ext_ويڪٽر_type(4)));
تعريفنوع uint8_t uint8x4_t __attribute__((ext_ويڪٽر_type(4)));
ساكن __device__ __forceinline__ صحيح __vsubss4(ثابت صحيح a, ثابت صحيح b) {
    ثابت int8x4_t va = reinterpret_cast<ثابت int8x4_t&>(a);
    ثابت int8x4_t vb = reinterpret_cast<ثابت int8x4_t&>(b);
#إذا __has_builtin(__builtin_elementwise_sub_sat)
    ثابت int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    أرجع reinterpret_cast<ثابت صحيح &>(c);
#وإلا
    int8x4_t c;
    int16_t tmp;
#هدايت unroll
    لكل (صحيح i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        إذا(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        إذا(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    أرجع reinterpret_cast<صحيح &>(c);
#جيڪڏهن_پورو // __has_builtin(__builtin_elementwise_sub_sat)
}

ساكن __device__ __forceinline__ صحيح __vsub4(ثابت صحيح a, ثابت صحيح b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ثابت بنانشان صحيح a_large = a | 0x80808080;
    ثابت بنانشان صحيح b_small = b & 0x7f7f7f7f;
    ثابت بنانشان صحيح نتيجو_هيٺيون_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ثابت بنانشان صحيح should_flip_مٿاهون_1bit = (a ^ ~b) & 0x80808080;

    أرجع نتيجو_هيٺيون_7bits ^ should_flip_مٿاهون_1bit;
}

ساكن __device__ __forceinline__ بنانشان صحيح __vcmpeq4(بنانشان صحيح a, بنانشان صحيح b) {
    ثابت uint8x4_t& va = reinterpret_cast<ثابت uint8x4_t&>(a);
    ثابت uint8x4_t& vb = reinterpret_cast<ثابت uint8x4_t&>(b);
    بنانشان صحيح c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#هدايت unroll
    لكل (صحيح i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    أرجع c;
}

ساكن __device__ __forceinline__ بنانشان صحيح __vcmpne4(بنانشان صحيح a, بنانشان صحيح b) {
    ثابت بنانشان صحيح x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ثابت بنانشان صحيح ne_هيٺيون_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ثابت بنانشان صحيح ne_مٿاهون_1bit = x & 0x80808080;
    ثابت بنانشان صحيح ne_any_ٻٽو_انگ = ne_هيٺيون_7bits | ne_مٿاهون_1bit;

    أرجع (ne_any_ٻٽو_انگ >> 7) * 0xff;
}
