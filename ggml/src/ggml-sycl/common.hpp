//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <string>

#ಅನ್ತರ್ಭಾವ "base.hpp"
#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-sycl.h"
#ಅನ್ತರ್ಭಾವ "presets.hpp"
#ಅನ್ತರ್ಭಾವ "type.hpp"
#ಅನ್ತರ್ಭಾವ "sycl_hw.hpp"
#ಅನ್ತರ್ಭಾವ "memtrace.hpp"

ನಾಮಾಕಾಶ syclexp = sycl::ext::oneapi::experimental;

#ಯದಿ defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#ಯದಿ defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#ಯದಿ GGML_SYCL_DNNL
#ಅನ್ತರ್ಭಾವ "dnnl.hpp"
#ಅನ್ತರ್ಭಾವ "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic ತಳ್ಳು
#pragma clang diagnostic ignored "-Wnested-anon-types"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#pragma clang diagnostic ತೆಗೆ
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

ಶೂನ್ಯ* ggml_sycl_host_malloc(size_t size);
ಶೂನ್ಯ ggml_sycl_host_free(ಶೂನ್ಯ* ptr);


ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_debug;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_optimize;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_fusion;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_esimd;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_mmvq_wide;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ಗಣನಾ ggml_sycl_xmx_gather_type {
    GGML_SYCL_XMX_GATHER_IQ4_NL   = 1 << 0,
    GGML_SYCL_XMX_GATHER_IQ3_S    = 1 << 1,
    GGML_SYCL_XMX_GATHER_IQ4_XS   = 1 << 2,
    GGML_SYCL_XMX_GATHER_IQ3_XXS  = 1 << 3,
    GGML_SYCL_XMX_GATHER_IQ2_XXS  = 1 << 4,
    GGML_SYCL_XMX_GATHER_IQ2_XS   = 1 << 5,
    GGML_SYCL_XMX_GATHER_IQ2_S    = 1 << 6,
    GGML_SYCL_XMX_GATHER_IQ1_S    = 1 << 7,
    GGML_SYCL_XMX_GATHER_IQ1_M    = 1 << 8,
    GGML_SYCL_XMX_GATHER_Q8_0     = 1 << 9,
    GGML_SYCL_XMX_GATHER_Q4_K     = 1 << 10,
    GGML_SYCL_XMX_GATHER_Q5_K     = 1 << 11,
    GGML_SYCL_XMX_GATHER_Q6_K     = 1 << 12,
};
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ಗಣನಾ ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#ಅನ್ಯಥಾ
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int32_t ggml_sycl_src1_prec(ಧ್ರುವ ggml_tensor * dst) {
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_sycl_src1_f16_ok(ಧ್ರುವ ggml_tensor * dst) {
    ಧ್ರುವ int32_t src1_prec = ggml_sycl_src1_prec(dst);
    ಪ್ರತ್ಯಾಗಮ g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_flash_attention;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_dev2dev_memcpy;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_fa_onednn;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_fa_onednn_max_kv;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_enable_mkl_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_memtrace;
ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(ಅಭಿವ್ಯಕ್ತಿ)                                            \
  [&]() {                                                                \
    ಯತ್ನ {                                                                \
      ಅಭಿವ್ಯಕ್ತಿ;                                                              \
      ಪ್ರತ್ಯಾಗಮ dpct::success;                                              \
    } ಗೃಹಾಣ (std::exception ಧ್ರುವ& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      ಪ್ರತ್ಯಾಗಮ dpct::default_error;                                        \
    }                                                                    \
  }()


#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardware optimize.
#define VER_GEN9 700 // todo for hardware optimize.
#define VER_GEN12 1000000 // todo for hardware optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#ಯದಿ !defined(GGML_SYCL_FORCE_MMQ)
    #define SYCL_USE_XMX
#endif

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

ಪ್ರಕಾರನಿರ್ದೇಶ sycl::queue *queue_ptr;

ಗಣನಾ ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

ಗಣನಾ ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(ಪರಿಮಾಣ(sycl::half) == ಪರಿಮಾಣ(ggml_fp16_t), "wrong fp16 size");

ಸ್ಥಿರ ಶೂನ್ಯ crash() {
  ಪೂರ್ಣಾಙ್ಕ* ptr = ರಿಕ್ತ;
  *ptr = 0;
}

[[noreturn]] ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_error(
    ಧ್ರುವ ವರ್ಣ* stmt,
    ಧ್ರುವ ವರ್ಣ* func,
    ಧ್ರುವ ವರ್ಣ* file,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಸಾಲು,
    ಧ್ರುವ ವರ್ಣ* ಸಂದೇಶ) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, ಸಂದೇಶ);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, ಸಾಲು);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    ಕುರು {                                                                                                   \
        auto err_ = (err);                                                                                 \
        ಯದಿ (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } ಯಾವತ್ (0)

#ಯದಿ DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#ಅನ್ಯಥಾ
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
ಪ್ರಕಾರನಿರ್ದೇಶ sycl::half dfloat; // dequantize float
ಪ್ರಕಾರನಿರ್ದೇಶ sycl::half2 dfloat2;
#ಅನ್ಯಥಾ
ಪ್ರಕಾರನಿರ್ದೇಶ ಭಿನ್ನ dfloat; // dequantize float
ಪ್ರಕಾರನಿರ್ದೇಶ sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ g_all_sycl_device_ಎಣಿಕೆ = -1;
ಸ್ಥಿರ ತಾರ್ಕಿಕ g_ggml_backend_sycl_buffer_type_initialized = ಅಸತ್ಯ;

ಸ್ಥಿರ ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

ಸ್ಥಿರ ಶೂನ್ಯ* g_scratch_buffer = nullptr;
ಸ್ಥಿರ size_t g_scratch_size = 0; // disabled by default
ಸ್ಥಿರ size_t g_scratch_offset = 0;

[[noreturn]] ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bad_arch(ಧ್ರುವ sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (ಶೂನ್ಯ)bad_arch; // suppress unused function warning
}

ಪೂರ್ಣಾಙ್ಕ get_current_device_id();

ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ ggml_sycl_get_device() {
    ಪ್ರತ್ಯಾಗಮ get_current_device_id();
}

ಅನ್ತಃಸ್ಥ dpct::err0 ggml_sycl_set_device(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ device) ಯತ್ನ {
  ಪೂರ್ಣಾಙ್ಕ current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  ಯದಿ (device == current_device_id) {
    ಪ್ರತ್ಯಾಗಮ 0;
  }

  ಪ್ರತ್ಯಾಗಮ CHECK_TRY_ERROR(dpct::select_device(device));
} ಗೃಹಾಣ (sycl::exception ಧ್ರುವ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
ಸಂರಚನಾ optimize_feature {
    ತಾರ್ಕಿಕ reorder=ಅಸತ್ಯ;
    ತಾರ್ಕಿಕ onednn_optimized_gemm=ಅಸತ್ಯ;
};

ಸಂರಚನಾ sycl_device_info {
    ಪೂರ್ಣಾಙ್ಕ cc;  // compute capability
    ಪೂರ್ಣಾಙ್ಕ nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    ಪೂರ್ಣಾಙ್ಕ warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    ಪೂರ್ಣಾಙ್ಕ max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    ತಾರ್ಕಿಕ    vmm;                // virtual memory support
    ತಾರ್ಕಿಕ    l0_device_type_valid;
    ತಾರ್ಕಿಕ    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    ತಾರ್ಕಿಕ    usm_system_support; // support for USM system allocations
};


ಸಂರಚನಾ ggml_sycl_device_info {
    ಪೂರ್ಣಾಙ್ಕ device_ಎಣಿಕೆ;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ಭಿನ್ನ, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    ಪೂರ್ಣಾಙ್ಕ max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    ತಾರ್ಕಿಕ ext_oneapi_level_zero = ಸತ್ಯ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ಧ್ರುವ ggml_sycl_device_info & ggml_sycl_info();

ಸ್ಥಿರ constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

ಸಂರಚನಾ ggml_sycl_pool {
    ಆಭಾಸ ~ggml_sycl_pool() = ಸ್ವಭಾವ;

    ಆಭಾಸ ಶೂನ್ಯ * alloc(size_t size, size_t * actual_size) = 0;
    ಆಭಾಸ ಶೂನ್ಯ ಮುಕ್ತ(ಶೂನ್ಯ * ptr, size_t size) = 0;
};

template<typename T>
ಸಂರಚನಾ ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        ಯದಿ (ptr != nullptr) {
            pool->ಮುಕ್ತ(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        ಯದಿ (ptr)
            pool->ಮುಕ್ತ(ptr, actual_size);
        ptr = (T *) pool->alloc(size * ಪರಿಮಾಣ(T), &ಏತತ್->actual_size);
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * ಪರಿಮಾಣ(T), &ಏತತ್->actual_size);
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ಏತತ್->pool = &pool;
        ಪ್ರತ್ಯಾಗಮ alloc(size);
    }

    T * get() {
        ಪ್ರತ್ಯಾಗಮ ptr;
    }

    ggml_sycl_pool_alloc() = ಸ್ವಭಾವ;
    ggml_sycl_pool_alloc(ಧ್ರುವ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(ಧ್ರುವ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

ಸಂರಚನಾ ggml_tensor_extra_gpu {
  ಶೂನ್ಯ* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

ಬಾಹ್ಯ ಪೂರ್ಣಾಙ್ಕ g_ggml_sycl_use_level_zero_api;
ಶೂನ್ಯ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
ಶೂನ್ಯ ggml_sycl_free_device(ಶೂನ್ಯ *ptr, sycl::queue &q);

ಶೂನ್ಯ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

ಸಂರಚನಾ mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ಸಂರಚನಾ ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

ನಾಮಾಕಾಶ sycl_ex = sycl::ext::oneapi::experimental;
ಸಂರಚನಾ ggml_backend_sycl_context {
    ಪೂರ್ಣಾಙ್ಕ device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(ಪೂರ್ಣಾಙ್ಕ device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(ಪೂರ್ಣಾಙ್ಕ device, ಪೂರ್ಣಾಙ್ಕ stream) {
        ಯದಿ (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        ಪ್ರತ್ಯಾಗಮ qptrs[device][stream];
    }

    queue_ptr stream() {
        ಪ್ರತ್ಯಾಗಮ stream(device, 0);
    }

#ಯದಿ GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ಧ್ರುವ dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        ಪ್ರತ್ಯಾಗಮ eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(ಪೂರ್ಣಾಙ್ಕ device, ಪೂರ್ಣಾಙ್ಕ _stream) {
        auto q = stream(device, _stream);
        ಪ್ರತ್ಯಾಗಮ stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        ಯದಿ (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            ಪ್ರತ್ಯಾಗಮ eng;
        }
        ಅನ್ಯಥಾ
        {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        ಯದಿ (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            ಪ್ರತ್ಯಾಗಮ stream;
        }
        ಅನ್ಯಥಾ
        {
            ಪ್ರತ್ಯಾಗಮ it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        ಪ್ರತ್ಯಾಗಮ stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    ಸ್ಥಿರ std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, ಪೂರ್ಣಾಙ್ಕ device);

    ಸ್ಥಿರ std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, ಪೂರ್ಣಾಙ್ಕ device);

    ggml_sycl_pool & pool(ಪೂರ್ಣಾಙ್ಕ device) {
        ಯದಿ (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        ಪ್ರತ್ಯಾಗಮ *pools[device];
    }

    ggml_sycl_pool & pool() {
        ಪ್ರತ್ಯಾಗಮ pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(ಪೂರ್ಣಾಙ್ಕ device) {
        ಯದಿ (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        ಪ್ರತ್ಯಾಗಮ *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { ಪ್ರತ್ಯಾಗಮ host_pool(device); }
};

// common device functions

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_reduce_sum(ಭಿನ್ನ x,
    ಧ್ರುವ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    ಪ್ರತ್ಯಾಗಮ x;
}

ಸ್ಥಿರ __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, ಧ್ರುವ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    ಪ್ರತ್ಯಾಗಮ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ warp_reduce_sum(ಪೂರ್ಣಾಙ್ಕ x) {
  ಪ್ರತ್ಯಾಗಮ sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_reduce_sum(ಭಿನ್ನ x) {
#pragma unroll
  ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  ಪ್ರತ್ಯಾಗಮ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_reduce_sum(ಭಿನ್ನ x, ಧ್ರುವ sycl::nd_item<3>& item_ct1) {
#pragma unroll
  ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  ಪ್ರತ್ಯಾಗಮ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  ಪ್ರತ್ಯಾಗಮ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  ಪ್ರತ್ಯಾಗಮ a;
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  ಪ್ರತ್ಯಾಗಮ WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ warp_reduce_all(ಪೂರ್ಣಾಙ್ಕ x) {
    ಯದಿ (width == ggml_sycl_get_physical_warp_size()) {
        ಪ್ರತ್ಯಾಗಮ sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } ಅನ್ಯಥಾ {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        ಪ್ರತ್ಯಾಗಮ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ warp_reduce_any(ಪೂರ್ಣಾಙ್ಕ x) {
    ಯದಿ (width == ggml_sycl_get_physical_warp_size()) {
        ಪ್ರತ್ಯಾಗಮ sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } ಅನ್ಯಥಾ {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        ಪ್ರತ್ಯಾಗಮ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ಪೂರ್ಣಾಙ್ಕ width>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_reduce_max(ಭಿನ್ನ x) {
#pragma unroll
  ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  ಪ್ರತ್ಯಾಗಮ x;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ warp_reduce_max(ಭಿನ್ನ x,
    ಧ್ರುವ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    ಪ್ರತ್ಯಾಗಮ x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<ಪೂರ್ಣಾಙ್ಕ N>
__dpct_inline__ size_t calculate_offset(ಧ್ರುವ std::array<ಪೂರ್ಣಾಙ್ಕ, N> & strides, ಧ್ರುವ std::array<ಪೂರ್ಣಾಙ್ಕ, N> & indices) {
    size_t offset = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    ಪ್ರತ್ಯಾಗಮ offset;
}

// Helper for vec loading aligned data
template <typename Tp, ಪೂರ್ಣಾಙ್ಕ n>
ಅನ್ತಃಸ್ಥ sycl::vec<Tp, n> vec_aligned_load(ಧ್ರುವ Tp* aligned_ptr) {
    ಪ್ರತ್ಯಾಗಮ *reinterpret_cast<ಧ್ರುವ sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, ಪೂರ್ಣಾಙ್ಕ dim>
ಸ್ಥಿರ __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    ಪ್ರತ್ಯಾಗಮ acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_ಅಂಕೆ, int64_t block_size);

constexpr size_t ceil_div(ಧ್ರುವ size_t m, ಧ್ರುವ size_t n) {
    ಪ್ರತ್ಯಾಗಮ (m + n - 1) / n;
}

ತಾರ್ಕಿಕ gpu_has_xmx(sycl::device &dev);

#ಯದಿ GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
ಅನ್ತಃಸ್ಥ ತಾರ್ಕಿಕ ggml_sycl_dnnl_has_optimized_gemm(ಪೂರ್ಣಾಙ್ಕ device) {
    ಪ್ರತ್ಯಾಗಮ ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

ಪೂರ್ಣಾಙ್ಕ ggml_sycl_get_env(ಧ್ರುವ ವರ್ಣ *env_name, ಪೂರ್ಣಾಙ್ಕ default_val);

template <ಪೂರ್ಣಾಙ್ಕ N, ವರ್ಗ T> std::string debug_get_array_str(ಧ್ರುವ std::string & prefix, ಧ್ರುವ T array[N]) {
    ಯದಿ (LIKELY(!g_ggml_sycl_debug)) {
        ಪ್ರತ್ಯಾಗಮ "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    ಪ್ರತಿ (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    ಯದಿ constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    ಪ್ರತ್ಯಾಗಮ ss.str();
}

ಅನ್ತಃಸ್ಥ std::string debug_get_tensor_str(ಧ್ರುವ std::string &prefix,
        ಧ್ರುವ ggml_tensor *tensor, ಧ್ರುವ std::string &suffix = "") {
    std::stringstream ss;
    ಯದಿ (LIKELY(!g_ggml_sycl_debug)) { ಪ್ರತ್ಯಾಗಮ ss.str(); }
    ss << prefix.c_str() << "=";
    ಯದಿ (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        ಯದಿ (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        ಯದಿ (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } ಅನ್ಯಥಾ {
        ss << "nullptr";
    }
    ss << suffix;
    ಪ್ರತ್ಯಾಗಮ ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ಸಂರಚನಾ scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ಧ್ರುವ std::string_view & func, ಧ್ರುವ std::string_view & func_suffix, ಧ್ರುವ ggml_tensor * dst,
                         std::size_t ಅಂಕೆ_src, ಧ್ರುವ std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        ಯದಿ (LIKELY(!g_ggml_sycl_debug)) {
            ಪ್ರತ್ಯಾಗಮ;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        ಯದಿ (dst) {
            ಪ್ರತಿ (std::size_t i = 0; i < ಅಂಕೆ_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(ಧ್ರುವ std::string_view & func, ಧ್ರುವ ggml_tensor * dst, std::size_t ಅಂಕೆ_src,
                         ಧ್ರುವ std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, ಅಂಕೆ_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  ಗುಪ್ತ:
    std::string_view func;
    std::string_view func_suffix;
};

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ get_alibi_slope(ಧ್ರುವ ಭಿನ್ನ    max_bias,
                                             ಧ್ರುವ uint32_t h,
                                             ಧ್ರುವ uint32_t n_head_log2,
                                             ಧ್ರುವ ಭಿನ್ನ    m0,
                                             ಧ್ರುವ ಭಿನ್ನ    m1) {
    ಯದಿ (max_bias <= 0.0f) {
        ಪ್ರತ್ಯಾಗಮ 1.0f;
    }
    ಧ್ರುವ ಭಿನ್ನ base = h < n_head_log2 ? m0 : m1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    ಪ್ರತ್ಯಾಗಮ dpct::pow(base, exph);
}

ಸ್ಥಿರ ಧ್ರುವ sycl::uint3 init_fastdiv_ಮೌಲ್ಯಗಳು(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    ಯಾವತ್ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    ಪ್ರತ್ಯಾಗಮ sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ ggml_sycl_get_max_cpy_bytes() {
    ಪ್ರತ್ಯಾಗಮ 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <ಪೂರ್ಣಾಙ್ಕ nbytes, ಪೂರ್ಣಾಙ್ಕ alignment = 0>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_memcpy_1(ಶೂನ್ಯ * dst, ಧ್ರುವ ಶೂನ್ಯ * src) {
    ಯದಿ constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr ಪೂರ್ಣಾಙ್ಕ nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < nbytes/nb_per_cpy; ++i) {
        ಯದಿ constexpr (nb_per_cpy == 1) {
            ((ವರ್ಣ *) dst)[i] = ((ಧ್ರುವ ವರ್ಣ *) src)[i];
        } ಅನ್ಯಥಾ ಯದಿ constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((ಧ್ರುವ short *) src)[i];
        } ಅನ್ಯಥಾ ಯದಿ constexpr (nb_per_cpy == 4) {
            ((ಪೂರ್ಣಾಙ್ಕ *) dst)[i] = ((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) src)[i];
        } ಅನ್ಯಥಾ ಯದಿ constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ಧ್ರುವ sycl::int2 *) src)[i];
        } ಅನ್ಯಥಾ ಯದಿ constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ಧ್ರುವ sycl::int4 *) src)[i];
        } ಅನ್ಯಥಾ {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    ಪ್ರತ್ಯಾಗಮ res;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t fastdiv(uint32_t n, ಧ್ರುವ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    ಧ್ರುವ uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_ಮೌಲ್ಯಗಳು.x());
    ಪ್ರತ್ಯಾಗಮ (hi + n) >> fastdiv_ಮೌಲ್ಯಗಳು.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ಭಿನ್ನ>(x),static_cast<ಭಿನ್ನ>(y));
    ಪ್ರತ್ಯಾಗಮ res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_ಮೌಲ್ಯ(static_cast<ಭಿನ್ನ>(H.x()), static_cast<ಭಿನ್ನ>(H.y()));
    ಪ್ರತ್ಯಾಗಮ float2_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ಧ್ರುವ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    ಧ್ರುವ uint32_t div_val = fastdiv(n, fastdiv_ಮೌಲ್ಯಗಳು);
    ಧ್ರುವ uint32_t mod_val = n - div_val * fastdiv_ಮೌಲ್ಯಗಳು.z();
    ಪ್ರತ್ಯಾಗಮ sycl::uint2(div_val, mod_val);
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ ggml_sycl_dp4a(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ a, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b, ಪೂರ್ಣಾಙ್ಕ c) {
    ಪ್ರತ್ಯಾಗಮ dpct::dp4a(a, b, c);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    ಯದಿ (x == 0) {
        bits = 0x00400000;
    } ಅನ್ಯಥಾ {
        bits = (uint32_t) x << 23;
    }

    ಭಿನ್ನ result;
    memcpy(&result, &bits, ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತ್ಯಾಗಮ result;
}

sycl::float2 __dpct_inline__ __half22float2(ಧ್ರುವ sycl::half2 &H) {
    sycl::float2 float2_ಮೌಲ್ಯ(static_cast<ಭಿನ್ನ>(H.x()), static_cast<ಭಿನ್ನ>(H.y()));
    ಪ್ರತ್ಯಾಗಮ float2_ಮೌಲ್ಯ;
}

ಭಿನ್ನ __dpct_inline__ __half2float(sycl::half H) {
    ಪ್ರತ್ಯಾಗಮ static_cast<ಭಿನ್ನ>(H);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ಭಿನ್ನ & acc, ಧ್ರುವ ಭಿನ್ನ v, ಧ್ರುವ ಭಿನ್ನ u) {
    acc += v*u;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ಭಿನ್ನ & acc, ಧ್ರುವ sycl::float2 v, ಧ್ರುವ sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ಭಿನ್ನ & acc, ಧ್ರುವ sycl::half2 v, ಧ್ರುವ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    ಧ್ರುವ sycl::float2 tmp = (v * u).template convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#ಅನ್ಯಥಾ
    ಧ್ರುವ sycl::float2 tmpv = __half22float2(v);
    ಧ್ರುವ sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(sycl::half2 & acc, ಧ್ರುವ sycl::half2 v, ಧ್ರುವ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#ಅನ್ಯಥಾ
    ಧ್ರುವ sycl::float2 tmpv = __half22float2(v);
    ಧ್ರುವ sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <ಪೂರ್ಣಾಙ್ಕ n>
ಸಂರಚನಾ ggml_sycl_unroll {
    template <typename Func, typename... Args>
    ಶೂನ್ಯ operator()(ಧ್ರುವ Func & f, Args... args) ಧ್ರುವ {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
ಸಂರಚನಾ ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    ಶೂನ್ಯ operator()(ಧ್ರುವ Func & f, Args... args) ಧ್ರುವ {
        f(0, args...);
    }
};

ಸ್ಥಿರ __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ಧ್ರುವ sycl::half2 a, ಧ್ರುವ sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ಭಿನ್ನ, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ಭಿನ್ನ, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    ಪ್ರತ್ಯಾಗಮ ret;
}

ಸ್ಥಿರ __dpct_inline__ sycl::half ggml_sycl_hmax(ಧ್ರುವ sycl::half a, ಧ್ರುವ sycl::half b) {
    ಪ್ರತ್ಯಾಗಮ sycl::vec<ಭಿನ್ನ, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

ಸ್ಥಿರ __dpct_inline__ uint32_t __hgt2_mask(ಧ್ರುವ sycl::half2 a, ಧ್ರುವ sycl::half2 b) {
    ಧ್ರುವ uint32_t mask_ಕೆಳಗಿನ  = 0x0000FFFF * (ಭಿನ್ನ(a[0]) > ಭಿನ್ನ(b[0]));
    ಧ್ರುವ uint32_t mask_ಮೇಲಿನ = 0xFFFF0000 * (ಭಿನ್ನ(a[1]) > ಭಿನ್ನ(b[1]));
    ಪ್ರತ್ಯಾಗಮ mask_ಕೆಳಗಿನ | mask_ಮೇಲಿನ;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t fastmodulo(uint32_t n, ಧ್ರುವ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    ಪ್ರತ್ಯಾಗಮ n - fastdiv(n, fastdiv_ಮೌಲ್ಯಗಳು) * fastdiv_ಮೌಲ್ಯಗಳು.z();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ fast_fp16_available(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ cc) {
    GGML_UNUSED(cc);
    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;   //Intel GPUs always support FP16.
}

ಗಣನಾ ವರ್ಗ block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, ಪೂರ್ಣಾಙ್ಕ warp_size>
ಸಂರಚನಾ block_reduce_policy;

template <typename T, typename... Ts>
ಅನ್ತಃಸ್ಥ constexpr ತಾರ್ಕಿಕ is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
ಅನ್ತಃಸ್ಥ constexpr ತಾರ್ಕಿಕ ggml_sycl_dependent_false_v = ಅಸತ್ಯ;

#define WARP_32_SIZE 32

template <typename T, ಪೂರ್ಣಾಙ್ಕ warp_size> ಸಂರಚನಾ block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    ಸ್ಥಿರ T reduce(T val) {
        ಯದಿ constexpr (is_any<T, ಭಿನ್ನ, sycl::float2, sycl::half2, ಪೂರ್ಣಾಙ್ಕ>) {
            ಪ್ರತ್ಯಾಗಮ warp_reduce_sum<warp_size>(val);
        } ಅನ್ಯಥಾ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ಸ್ಥಿರ T sentinel() {
        ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
            ಪ್ರತ್ಯಾಗಮ 0.0f;
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<T, sycl::float2>) {
            ಪ್ರತ್ಯಾಗಮ sycl::float2(0.0f, 0.0f);
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<T, sycl::half2>) {
            ಪ್ರತ್ಯಾಗಮ sycl::half2(0.0f, 0.0f);
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<T, ಪೂರ್ಣಾಙ್ಕ>) {
            ಪ್ರತ್ಯಾಗಮ 0;
        } ಅನ್ಯಥಾ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, ಪೂರ್ಣಾಙ್ಕ warp_size> ಸಂರಚನಾ block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    ಸ್ಥಿರ T reduce(T val) {
        ಯದಿ constexpr (is_any<T, ಭಿನ್ನ, sycl::half2>) {
            ಪ್ರತ್ಯಾಗಮ warp_reduce_max<warp_size>(val);
        } ಅನ್ಯಥಾ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ಸ್ಥಿರ T sentinel() {
        ಯದಿ constexpr (std::is_same_v<T, ಭಿನ್ನ>) {
            ಪ್ರತ್ಯಾಗಮ -INFINITY;
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_same_v<T, sycl::half2>) {
            ಪ್ರತ್ಯಾಗಮ sycl::half2(-INFINITY, -INFINITY);
        } ಅನ್ಯಥಾ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, ಪೂರ್ಣಾಙ್ಕ warp_size, typename T>
ಸ್ಥಿರ T block_reduce(T val, T * shared_vals, ಪೂರ್ಣಾಙ್ಕ block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nthreads = item_ct1.get_local_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nwarps = nthreads / WARP_SIZE;

    ಯದಿ (block_size > warp_size) {
        ಪ್ರತಿಜ್ಞಾ((block_size <= 1024) && (block_size % warp_size) == 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ warp_id = item_ct1.get_local_id(2) / warp_size;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ lane_id = item_ct1.get_local_id(2) % warp_size;
        ಯದಿ (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        ಭಿನ್ನ tmp = 0.f;
        ಯದಿ (lane_id < (static_cast<ಪೂರ್ಣಾಙ್ಕ>(block_size) / warp_size)) {
            ಪ್ರತಿ (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        ಪ್ರತ್ಯಾಗಮ block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    ಪ್ರತ್ಯಾಗಮ val;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    ಯದಿ (x == 0 || x == 0x7F) {
        ಪ್ರತ್ಯಾಗಮ 0.0f;
    }
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ exp = (x >> 3) & 0xF;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ man = x & 0x7;
    ಭಿನ್ನ raw;
    ಯದಿ (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } ಅನ್ಯಥಾ {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (ಭಿನ್ನ) exp - 7.0f);
    }
    ಪ್ರತ್ಯಾಗಮ raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
