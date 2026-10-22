#رهنمود once

#تعریف_کن HIP_DISABLE_WARP_SYNC_BUILTINS 1
#ضمن <hip/hip_runtime.h>
#ضمن <hipblas/hipblas.h>
#ضمن <hip/hip_fp16.h>
#ضمن <hip/hip_bf16.h>

#اگر_تعریف_شده GGML_USE_NCCL
#ضمن <rccl/rccl.h>
#پایان_اگر // GGML_USE_NCCL


#تعریف_کن CUBLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#تعریف_کن CUBLAS_GEMM_DEFAULT_TENSOR_OP HIPBLAS_GEMM_DEFAULT
#تعریف_کن CUBLAS_OP_N HIPBLAS_OP_N
#تعریف_کن CUBLAS_OP_T HIPBLAS_OP_T
#تعریف_کن CUBLAS_وضعیت_SUCCESS HIPBLAS_وضعیت_SUCCESS
#تعریف_کن CUBLAS_TF32_TENSOR_OP_MATH 0
#تعریف_کن CUDA_R_16F  HIPBLAS_R_16F
#تعریف_کن CUDA_R_16BF HIPBLAS_R_16B
#تعریف_کن CUDA_R_32F  HIPBLAS_R_32F
#تعریف_کن CUBLAS_SIDE_راست HIPBLAS_SIDE_راست
#تعریف_کن CUBLAS_FILL_نما_فوقانی HIPBLAS_FILL_نما_فوقانی
#تعریف_کن CUBLAS_DIAG_NON_UNIT HIPBLAS_DIAG_NON_UNIT
#تعریف_کن CU_DEVICE_ATTRIBUTE_VIRTUAL_حافظه_MANAGEMENT_SUPPORTED hip_Device_Attribute_Virtual_حافظه_Management_Supported
#تعریف_کن CU_MEM_ALLOC_GRANULARITY_RECOMMENDED hip_Mem_تخصیص_Granularity_Recommended
#تعریف_کن CU_MEM_تخصیص_TYPE_PINNED hip_Mem_تخصیص_Type_Pinned
#تعریف_کن CU_MEM_LOCATION_TYPE_DEVICE hipMemLocationTypeDevice
#تعریف_کن CU_MEM_ACCESS_FLAGS_PROT_READWRITE hipMemAccessFlagsProtReadWrite
#تعریف_کن CU_CHECK(fn) {hipError_t err = fn; إذا(err != hipSuccess) { GGML_ABORT("HipVMM Failure: %s\n", hipGetErrorString(err)); }}
#تعریف_کن __shfl_sync(ماسک, var, lane_ماسک, عرض) __shfl(var, lane_ماسک, عرض)
#تعریف_کن __shfl_up_sync(ماسک, var, lane_ماسک, عرض) __shfl_up(var, lane_ماسک, عرض)
#تعریف_کن __shfl_xor_sync(ماسک, var, lane_ماسک, عرض) __shfl_xor(var, lane_ماسک, عرض)
#تعریف_کن __all_sync(ماسک, var) __all(var)
#تعریف_کن __any_sync(ماسک, var) __any(var)
#تعریف_کن cublasStrsmBatched hipblasStrsmBatched
#تعریف_کن cublasCreate hipblasCreate
#تعریف_کن cublasDestroy hipblasDestroy
#تعریف_کن cublasGemmEx hipblasGemmEx
#تعریف_کن cublasGemmBatchedEx hipblasGemmBatchedEx
#تعریف_کن cublasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#تعریف_کن cublasHandle_t hipblasHandle_t
#تعریف_کن cublas_مجموعه_Math_نما(handle, نما) CUBLAS_وضعیت_SUCCESS
#تعریف_کن cublas_مجموعه_جریان hipblas_مجموعه_جریان
#تعریف_کن cublasSgemm hipblasSgemm
#تعریف_کن cublasSgemmBatched hipblasSgemmBatched
#تعریف_کن cublasSgemmStridedBatched hipblasSgemmStridedBatched
#تعریف_کن cublas_وضعیت_t hipblas_وضعیت_t
#تعریف_کن cublasOperation_t hipblasOperation_t
#تعریف_کن cudaDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#تعریف_کن cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#تعریف_کن cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#تعریف_کن cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#تعریف_کن cudaDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کن cudaDeviceGetPCIBusId hipDeviceGetPCIBusId
#تعریف_کن cudaDeviceProp hipDeviceProp_t
#تعریف_کن cudaDeviceSynchronize hipDeviceSynchronize
#تعریف_کن cudaError_t hipError_t
#تعریف_کن cuda_Error_حافظه_تخصیص hip_Error_بیرون_از_حافظه
#تعریف_کن cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#تعریف_کن cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#تعریف_کن cudaEventCreateWithFlags hipEventCreateWithFlags
#تعریف_کن cudaEventDisableTiming hipEventDisableTiming
#تعریف_کن cuda_Event_رکورد hip_Event_رکورد
#تعریف_کن cudaEventSynchronize hipEventSynchronize
#تعریف_کن cudaEvent_t hipEvent_t
#تعریف_کن cudaEventDestroy hipEventDestroy
#تعریف_کن cudaFree hipFree
#تعریف_کن cudaFreeHost hipHostFree
#تعریف_کن cudaGetDevice hipGetDevice
#تعریف_کن cuda_Get_Device_شمار hip_Get_Device_شمار
#تعریف_کن cudaGetDeviceProperties hipGetDeviceProperties
#تعریف_کن cudaGetErrorString hipGetErrorString
#تعریف_کن cuda_Get_آخرین_Error hip_Get_آخرین_Error
#تعریف_کن cudaHostAlloc hipHostMalloc
#تعریف_کن cudaHostAllocPortable hipHostMallocPortable
#تعریف_کن cudaHostAllocMapped hipHostMallocMapped
#تعریف_کن cuda_Host_Get_Device_اشاره_گر hip_Host_Get_Device_اشاره_گر
#تعریف_کن cuda_Host_ثبات hip_Host_ثبات
#تعریف_کن cuda_Host_ثبات_Portable hip_Host_ثبات_Portable
#تعریف_کن cuda_Host_ثبات_Read_Only hip_Host_ثبات_Read_Only
#تعریف_کن cudaHostUnregister hipHostUnregister
#تعریف_کن cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#تعریف_کن cudaLaunchHostFunc hipLaunchHostFunc
#تعریف_کن cudaMalloc hipMalloc
#تعریف_کن cudaMallocHost(ptr, اندازه) hipHostMalloc(ptr, اندازه, hipHostMallocDefault)
#تعریف_کن cudaMallocManaged hipMallocManaged
#تعریف_کن cudaMemAdvise hipMemAdvise
#تعریف_کن cudaMemcpy hipMemcpy
#تعریف_کن cudaMemcpyAsync hipMemcpyAsync
#تعریف_کن cudaMemcpyPeerAsync hipMemcpyPeerAsync
#تعریف_کن cudaMemcpy2DAsync hipMemcpy2DAsync
#تعریف_کن cuda_Memcpy_Device_مقصد_Device hip_Memcpy_Device_مقصد_Device
#تعریف_کن cuda_Memcpy_Device_مقصد_Host hip_Memcpy_Device_مقصد_Host
#تعریف_کن cuda_Memcpy_Host_مقصد_Device hip_Memcpy_Host_مقصد_Device
#تعریف_کن cuda_Memcpy_نوع hip_Memcpy_نوع
#تعریف_کن cudaMemset hipMemset
#تعریف_کن cudaMemsetAsync hipMemsetAsync
#تعریف_کن cudaMemGetInfo hipMemGetInfo
#تعریف_کن cuda_Occupancy_Max_Potential_بلوک_اندازه hip_Occupancy_Max_Potential_بلوک_اندازه
#تعریف_کن cuda_مجموعه_Device hip_مجموعه_Device
#تعریف_کن cuDeviceGet hipDeviceGet
#تعریف_کن CUdevice hipDevice_t
#تعریف_کن CUdeviceptr hipDeviceptr_t
#تعریف_کن cuMemUnmap hipMemUnmap
#تعریف_کن CUmemAccessDesc hipMemAccessDesc
#تعریف_کن cu_Mem_آدرس_Free hip_Mem_آدرس_Free
#تعریف_کن cuMemRelease hipMemRelease
#تعریف_کن CUmem_Generic_تخصیص_Handle hip_Mem_Generic_تخصیص_Handle_t
#تعریف_کن cuMemCreate hipMemCreate
#تعریف_کن cu_Mem_آدرس_Reserve hip_Mem_آدرس_Reserve
#تعریف_کن cuMemMap hipMemMap
#تعریف_کن cu_Mem_مجموعه_Access hip_Mem_مجموعه_Access
#تعریف_کن cu_Mem_Get_تخصیص_Granularity hip_Mem_Get_تخصیص_Granularity
#تعریف_کن CUmem_تخصیص_Prop hip_Mem_تخصیص_Prop
#تعریف_کن cuDeviceGetAttribute hipDeviceGetAttribute
#تعریف_کن cuda_جریان_Create_With_Flags hip_جریان_Create_With_Flags
#تعریف_کن cuda_جریان_Destroy hip_جریان_Destroy
#تعریف_کن cuda_جریان_Fire_And_Forget hip_جریان_Fire_And_Forget
#تعریف_کن cuda_جریان_Non_Blocking hip_جریان_Non_Blocking
#تعریف_کن cuda_جریان_در_هر_Thread hip_جریان_در_هر_Thread
#تعریف_کن cuda_جریان_Synchronize hip_جریان_Synchronize
#تعریف_کن cuda_جریان_Wait_Event hip_جریان_Wait_Event
#تعریف_کن cuda_گراف_Exec_t hip_گراف_Exec_t
#تعریف_کن cuda_گراف_گره_t hip_گراف_گره_t
#تعریف_کن cuda_Kernel_گره_Params hip_Kernel_گره_Params
#تعریف_کن cuda_Kernel_گره_Params hip_Kernel_گره_Params
#تعریف_کن cuda_گراف_Exec_Destroy hip_گراف_Exec_Destroy
#تعریف_کن cuda_گراف_Launch hip_گراف_Launch
#تعریف_کن cuda_Error_گراف_Exec_Update_Failure hip_Error_گراف_Exec_Update_Failure
#تعریف_کن cuda_گراف_Exec_Update_نتیجه hip_گراف_Exec_Update_نتیجه
#تعریف_کن cuda_گراف_گره_Type hip_گراف_گره_Type
#تعریف_کن cuda_گراف_گره_Type_Kernel hip_گراف_گره_Type_Kernel
#تعریف_کن cuda_گراف_Instantiate hip_گراف_Instantiate
#تعریف_کن cuda_جریان_انتها_Capture hip_جریان_انتها_Capture
#تعریف_کن cuda_گراف_Destroy hip_گراف_Destroy
#تعریف_کن cuda_گراف_Kernel_گره_مجموعه_Params hip_گراف_Kernel_گره_مجموعه_Params
#تعریف_کن cudaErrorInvalidDeviceFunction hipErrorInvalidDeviceFunction
#تعریف_کن cuda_گراف_Kernel_گره_Get_Params hip_گراف_Kernel_گره_Get_Params
#تعریف_کن cuda_گراف_گره_Get_Type hip_گراف_گره_Get_Type
#تعریف_کن cuda_گراف_Get_Nodes hip_گراف_Get_Nodes
#تعریف_کن cuda_گراف_Exec_Update hip_گراف_Exec_Update
#تعریف_کن cuda_جریان_Capture_نما_Relaxed hip_جریان_Capture_نما_Relaxed
#تعریف_کن cuda_جریان_Begin_Capture hip_جریان_Begin_Capture
#تعریف_کن cuda_گراف_t hip_گراف_t
#تعریف_کن cuda_جریان_t hip_جریان_t
#تعریف_کن cudaSuccess hipSuccess
#تعریف_کن cuda_Occupancy_Max_فعال_بلوکها_در_هر_Multiprocessor hip_Occupancy_Max_فعال_بلوکها_در_هر_Multiprocessor
#تعریف_کن cuda_Func_مجموعه_Attribute hip_Func_مجموعه_Attribute
#تعریف_کن cuda_Func_Attribute_Max_Dynamic_Shared_حافظه_اندازه hip_Func_Attribute_Max_Dynamic_Shared_حافظه_اندازه
#تعریف_کن __trap() افعل { abort(); __builtin_unreachable(); } طالما(0)
#تعریف_کن CUBLAS_وضعیت_SUCCESS HIPBLAS_وضعیت_SUCCESS
#تعریف_کن CUBLAS_وضعیت_NOT_INITIALIZED HIPBLAS_وضعیت_NOT_INITIALIZED
#تعریف_کن CUBLAS_وضعیت_ALLOC_FAILED HIPBLAS_وضعیت_ALLOC_FAILED
#تعریف_کن CUBLAS_وضعیت_INVALID_مقدار HIPBLAS_وضعیت_INVALID_مقدار
#تعریف_کن CUBLAS_وضعیت_ARCH_MISMATCH HIPBLAS_وضعیت_ARCH_MISMATCH
#تعریف_کن CUBLAS_وضعیت_MAPPING_ERROR HIPBLAS_وضعیت_MAPPING_ERROR
#تعریف_کن CUBLAS_وضعیت_EXECUTION_FAILED HIPBLAS_وضعیت_EXECUTION_FAILED
#تعریف_کن CUBLAS_وضعیت_داخلی_ERROR HIPBLAS_وضعیت_داخلی_ERROR
#تعریف_کن CUBLAS_وضعیت_NOT_SUPPORTED HIPBLAS_وضعیت_NOT_SUPPORTED

#إذا HIP_نسخه >= 60500000
#تعریف_کن CUBLAS_COMPUTE_16F HIPBLAS_COMPUTE_16F
#تعریف_کن CUBLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#تعریف_کن CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_COMPUTE_32F_FAST_16F
#تعریف_کن cublasComputeType_t hipblasComputeType_t
#تعریف_کن cuda_داده_Type_t hip_داده_Type
#وإلا
#تعریف_کن CUBLAS_COMPUTE_16F HIPBLAS_R_16F
#تعریف_کن CUBLAS_COMPUTE_32F HIPBLAS_R_32F
#تعریف_کن CUBLAS_COMPUTE_32F_FAST_16F HIPBLAS_R_32F
#تعریف_کن cublasComputeType_t hipblasDatatype_t
#تعریف_کن cuda_داده_Type_t hipblasDatatype_t
#پایان_اگر // HIP_VERSION >= 6050000

#إذا !defined(__HIP_PLATFORM_AMD__)
#خطا "The HIP backend supports only AMD targets"
#پایان_اگر // !defined(__HIP_PLATFORM_AMD__)

#تعریف_کن __CUDA_ARCH__ 1300

#إذا defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)
#تعریف_کن GCN5
#پایان_اگر // defined(__gfx900__) || defined(__gfx906__) || defined(__gfx909__) || defined(__gfx90c__)

#إذا defined(__gfx803__)
#تعریف_کن GCN4
#پایان_اگر // defined(__gfx803__)

#إذا defined(GCN5) || defined(GCN4)
#تعریف_کن GCN
#پایان_اگر // defined(GCN5) || defined(GCN4)

#إذا defined(__gfx950__)
#تعریف_کن CDNA4
#پایان_اگر // defined(__gfx950__)

#إذا defined(__gfx942__)
#تعریف_کن CDNA3
#پایان_اگر // defined(__gfx942__)

#إذا defined(__gfx90a__)
#تعریف_کن CDNA2
#پایان_اگر // defined(__gfx90a__)

#إذا defined(__gfx908__)
#تعریف_کن CDNA1
#پایان_اگر // defined(__gfx908__)

#إذا defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)
#تعریف_کن CDNA // For the entire family
#پایان_اگر // defined(CDNA4) || defined(CDNA3) || defined(CDNA2) || defined(CDNA1)

#إذا defined(__GFX12__)
#تعریف_کن RDNA4
#پایان_اگر // defined(__GFX12__)

#إذا defined(__GFX11__)
#تعریف_کن RDNA3
#پایان_اگر // defined(__GFX11__)

#إذا defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)
#تعریف_کن RDNA3_5
#پایان_اگر // defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__)

#إذا defined(RDNA3) && !defined(RDNA3_5)
#تعریف_کن RDNA3_0
#پایان_اگر // defined(RDNA3) && !defined(RDNA3_5)

#إذا defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || \
    defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__) || defined(__gfx1037__)
#تعریف_کن RDNA2
#پایان_اگر

#إذا defined(__gfx1010__) || defined(__gfx1012__)
#تعریف_کن RDNA1
#پایان_اگر // defined(__gfx1010__) || defined(__gfx1012__)

#إذا defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)
#تعریف_کن RDNA // For the entire family
#پایان_اگر // defined(RDNA4) || defined(RDNA3) || defined(RDNA2) || defined(RDNA1)

#اگر_تعریف_نشده __has_builtin
    #تعریف_کن __has_builtin(x) 0
#پایان_اگر

تعريفنوع __hip_bfloat16 nv_bfloat16;
تعريفنوع __hip_bfloat162 nv_bfloat162;

#إذا HIP_نسخه >= 60300000
#ضمن <hip/hip_fp8.h>
تعريفنوع __hip_fp8_e4m3 __nv_fp8_e4m3;
#تعریف_کن FP8_موجود
#پایان_اگر // HIP_VERSION >= 60300000

تعريفنوع int8_t int8x4_t __attribute__((ext_وکتور_type(4)));
تعريفنوع uint8_t uint8x4_t __attribute__((ext_وکتور_type(4)));
ساكن __device__ __forceinline__ صحيح __vsubss4(ثابت صحيح a, ثابت صحيح b) {
    ثابت int8x4_t va = reinterpret_cast<ثابت int8x4_t&>(a);
    ثابت int8x4_t vb = reinterpret_cast<ثابت int8x4_t&>(b);
#إذا __has_builtin(__builtin_elementwise_sub_sat)
    ثابت int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    أرجع reinterpret_cast<ثابت صحيح &>(c);
#وإلا
    int8x4_t c;
    int16_t tmp;
#رهنمود unroll
    لكل (صحيح i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        إذا(tmp > std::numeric_limits<int8_t>::max()) tmp = std::numeric_limits<int8_t>::max();
        إذا(tmp < std::numeric_limits<int8_t>::min()) tmp = std::numeric_limits<int8_t>::min();
        c[i] = tmp;
    }
    أرجع reinterpret_cast<صحيح &>(c);
#پایان_اگر // __has_builtin(__builtin_elementwise_sub_sat)
}

ساكن __device__ __forceinline__ صحيح __vsub4(ثابت صحيح a, ثابت صحيح b) {
    // do some small modifications to a and b to make the subtraction not underflow
    ثابت بیعلامت صحيح a_large = a | 0x80808080;
    ثابت بیعلامت صحيح b_small = b & 0x7f7f7f7f;
    ثابت بیعلامت صحيح نتیجه_کم_7bits = a_large - b_small;

    // if two ops share the same high bit, we should flip the high bit of the result
    ثابت بیعلامت صحيح should_flip_بالا_1bit = (a ^ ~b) & 0x80808080;

    أرجع نتیجه_کم_7bits ^ should_flip_بالا_1bit;
}

ساكن __device__ __forceinline__ بیعلامت صحيح __vcmpeq4(بیعلامت صحيح a, بیعلامت صحيح b) {
    ثابت uint8x4_t& va = reinterpret_cast<ثابت uint8x4_t&>(a);
    ثابت uint8x4_t& vb = reinterpret_cast<ثابت uint8x4_t&>(b);
    بیعلامت صحيح c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#رهنمود unroll
    لكل (صحيح i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    أرجع c;
}

ساكن __device__ __forceinline__ بیعلامت صحيح __vcmpne4(بیعلامت صحيح a, بیعلامت صحيح b) {
    ثابت بیعلامت صحيح x = a ^ b;

    // any non-equal bit in a byte will set the high bit of that byte here
    // the addition will not overflow in the byte as op1 and op2 are both less than 0x80
    ثابت بیعلامت صحيح ne_کم_7bits = ((x & 0x7f7f7f7f) + 0x7f7f7f7f) & 0x80808080;
    ثابت بیعلامت صحيح ne_بالا_1bit = x & 0x80808080;
    ثابت بیعلامت صحيح ne_any_بیت = ne_کم_7bits | ne_بالا_1bit;

    أرجع (ne_any_بیت >> 7) * 0xff;
}
