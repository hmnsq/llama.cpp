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

#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <string>

#ଅନ୍ତର୍ଭାଵ "base.hpp"
#ଅନ୍ତର୍ଭାଵ "dpct/helper.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl.h"
#ଅନ୍ତର୍ଭାଵ "presets.hpp"
#ଅନ୍ତର୍ଭାଵ "type.hpp"
#ଅନ୍ତର୍ଭାଵ "sycl_hw.hpp"
#ଅନ୍ତର୍ଭାଵ "memtrace.hpp"

ନାମାକାଶ syclexp = sycl::ext::oneapi::experimental;

#ଯଦି defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #ଅନ୍ତର୍ଭାଵ <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#ଯଦି defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#ଯଦି GGML_SYCL_DNNL
#ଅନ୍ତର୍ଭାଵ "dnnl.hpp"
#ଅନ୍ତର୍ଭାଵ "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic ଠେଲ
#pragma clang diagnostic ignored "-Wnested-anon-types"
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#pragma clang diagnostic କାଢ଼
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"

ଶୂନ୍ୟ* ggml_sycl_host_malloc(size_t size);
ଶୂନ୍ୟ ggml_sycl_host_free(ଶୂନ୍ୟ* ptr);


ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_debug;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_optimize;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_fusion;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_esimd;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_mmvq_wide;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ଗଣନା ggml_sycl_xmx_gather_type {
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
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ଗଣନା ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#ଅନ୍ୟଥା
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ସ୍ଥିର ଧାଡ଼ିଭିତରେ int32_t ggml_sycl_src1_prec(ଅଚଳ ggml_tensor * dst) {
    ଫେରନ୍ତୁ g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_src1_f16_ok(ଅଚଳ ggml_tensor * dst) {
    ଅଚଳ int32_t src1_prec = ggml_sycl_src1_prec(dst);
    ଫେରନ୍ତୁ g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_flash_attention;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dev2dev_memcpy;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_fa_onednn;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_fa_onednn_max_kv;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_mkl_fa;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_memtrace;
ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(ପଦସମଷ୍ଟି)                                            \
  [&]() {                                                                \
    ଚେଷ୍ଟା {                                                                \
      ପଦସମଷ୍ଟି;                                                              \
      ଫେରନ୍ତୁ dpct::success;                                              \
    } ଧର (std::exception ଅଚଳ& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      ଫେରନ୍ତୁ dpct::default_error;                                        \
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
#ଯଦି !defined(GGML_SYCL_FORCE_MMQ)
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

ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::queue *queue_ptr;

ଗଣନା ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

ଗଣନା ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(ଆକାର(sycl::half) == ଆକାର(ggml_fp16_t), "wrong fp16 size");

ସ୍ଥିର ଶୂନ୍ୟ crash() {
  ପୂର୍ଣ୍ଣସଂଖ୍ୟା* ptr = ଖାଲି;
  *ptr = 0;
}

[[noreturn]] ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_error(
    ଅଚଳ ଅକ୍ଷର* stmt,
    ଅଚଳ ଅକ୍ଷର* func,
    ଅଚଳ ଅକ୍ଷର* file,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଧାଡ଼ି,
    ଅଚଳ ଅକ୍ଷର* ସନ୍ଦେଶ) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, ସନ୍ଦେଶ);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, ଧାଡ଼ି);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    କର {                                                                                                   \
        auto err_ = (err);                                                                                 \
        ଯଦି (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } ଯେତେବେଳେ (0)

#ଯଦି DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#ଅନ୍ୟଥା
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::half dfloat; // dequantize float
ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::half2 dfloat2;
#ଅନ୍ୟଥା
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଦଶମିକ dfloat; // dequantize float
ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_all_sycl_device_ଗଣନା = -1;
ସ୍ଥିର ତାର୍କିକ g_ggml_backend_sycl_buffer_type_initialized = ମିଥ୍ୟା;

ସ୍ଥିର ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

ସ୍ଥିର ଶୂନ୍ୟ* g_scratch_buffer = nullptr;
ସ୍ଥିର size_t g_scratch_size = 0; // disabled by default
ସ୍ଥିର size_t g_scratch_offset = 0;

[[noreturn]] ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bad_arch(ଅଚଳ sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (ଶୂନ୍ୟ)bad_arch; // suppress unused function warning
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_current_device_id();

ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_device() {
    ଫେରନ୍ତୁ get_current_device_id();
}

ଧାଡ଼ିଭିତରେ dpct::err0 ggml_sycl_set_device(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) ଚେଷ୍ଟା {
  ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  ଯଦି (device == current_device_id) {
    ଫେରନ୍ତୁ 0;
  }

  ଫେରନ୍ତୁ CHECK_TRY_ERROR(dpct::select_device(device));
} ଧର (sycl::exception ଅଚଳ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
ସଂରଚନା optimize_feature {
    ତାର୍କିକ reorder=ମିଥ୍ୟା;
    ତାର୍କିକ onednn_optimized_gemm=ମିଥ୍ୟା;
};

ସଂରଚନା sycl_device_info {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା cc;  // compute capability
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    ତାର୍କିକ    vmm;                // virtual memory support
    ତାର୍କିକ    l0_device_type_valid;
    ତାର୍କିକ    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    ତାର୍କିକ    usm_system_support; // support for USM system allocations
};


ସଂରଚନା ggml_sycl_device_info {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_ଗଣନା;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    ତାର୍କିକ ext_oneapi_level_zero = ସତ୍ୟ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ଅଚଳ ggml_sycl_device_info & ggml_sycl_info();

ସ୍ଥିର constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

ସଂରଚନା ggml_sycl_pool {
    ଆଭାସୀ ~ggml_sycl_pool() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;

    ଆଭାସୀ ଶୂନ୍ୟ * alloc(size_t size, size_t * actual_size) = 0;
    ଆଭାସୀ ଶୂନ୍ୟ ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size) = 0;
};

template<typename T>
ସଂରଚନା ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        ଯଦି (ptr != nullptr) {
            pool->ମୁକ୍ତ(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        ଯଦି (ptr)
            pool->ମୁକ୍ତ(ptr, actual_size);
        ptr = (T *) pool->alloc(size * ଆକାର(T), &ଏହି->actual_size);
        ଫେରନ୍ତୁ ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * ଆକାର(T), &ଏହି->actual_size);
        ଫେରନ୍ତୁ ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ଏହି->pool = &pool;
        ଫେରନ୍ତୁ alloc(size);
    }

    T * get() {
        ଫେରନ୍ତୁ ptr;
    }

    ggml_sycl_pool_alloc() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;
    ggml_sycl_pool_alloc(ଅଚଳ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(ଅଚଳ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

ସଂରଚନା ggml_tensor_extra_gpu {
  ଶୂନ୍ୟ* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

ବାହ୍ୟ ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_use_level_zero_api;
ଶୂନ୍ୟ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
ଶୂନ୍ୟ ggml_sycl_free_device(ଶୂନ୍ୟ *ptr, sycl::queue &q);

ଶୂନ୍ୟ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

ସଂରଚନା mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ସଂରଚନା ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

ନାମାକାଶ sycl_ex = sycl::ext::oneapi::experimental;
ସଂରଚନା ggml_backend_sycl_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stream) {
        ଯଦି (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        ଫେରନ୍ତୁ qptrs[device][stream];
    }

    queue_ptr stream() {
        ଫେରନ୍ତୁ stream(device, 0);
    }

#ଯଦି GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ଅଚଳ dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        ଫେରନ୍ତୁ eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ପୂର୍ଣ୍ଣସଂଖ୍ୟା _stream) {
        auto q = stream(device, _stream);
        ଫେରନ୍ତୁ stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        ଯଦି (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            ଫେରନ୍ତୁ eng;
        }
        ଅନ୍ୟଥା
        {
            ଫେରନ୍ତୁ it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        ଯଦି (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            ଫେରନ୍ତୁ stream;
        }
        ଅନ୍ୟଥା
        {
            ଫେରନ୍ତୁ it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        ଫେରନ୍ତୁ stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    ସ୍ଥିର std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device);

    ସ୍ଥିର std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device);

    ggml_sycl_pool & pool(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
        ଯଦି (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        ଫେରନ୍ତୁ *pools[device];
    }

    ggml_sycl_pool & pool() {
        ଫେରନ୍ତୁ pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
        ଯଦି (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        ଫେରନ୍ତୁ *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { ଫେରନ୍ତୁ host_pool(device); }
};

// common device functions

ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_reduce_sum(ଦଶମିକ x,
    ଅଚଳ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    ଫେରନ୍ତୁ x;
}

ସ୍ଥିର __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, ଅଚଳ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    ଫେରନ୍ତୁ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_reduce_sum(ପୂର୍ଣ୍ଣସଂଖ୍ୟା x) {
  ଫେରନ୍ତୁ sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_reduce_sum(ଦଶମିକ x) {
#pragma unroll
  ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  ଫେରନ୍ତୁ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_reduce_sum(ଦଶମିକ x, ଅଚଳ sycl::nd_item<3>& item_ct1) {
#pragma unroll
  ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  ଫେରନ୍ତୁ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  ଫେରନ୍ତୁ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  ଫେରନ୍ତୁ a;
}

ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  ଫେରନ୍ତୁ WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_reduce_all(ପୂର୍ଣ୍ଣସଂଖ୍ୟା x) {
    ଯଦି (width == ggml_sycl_get_physical_warp_size()) {
        ଫେରନ୍ତୁ sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } ଅନ୍ୟଥା {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        ଫେରନ୍ତୁ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_reduce_any(ପୂର୍ଣ୍ଣସଂଖ୍ୟା x) {
    ଯଦି (width == ggml_sycl_get_physical_warp_size()) {
        ଫେରନ୍ତୁ sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } ଅନ୍ୟଥା {
#pragma unroll
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        ଫେରନ୍ତୁ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା width>
ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_reduce_max(ଦଶମିକ x) {
#pragma unroll
  ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  ଫେରନ୍ତୁ x;
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ warp_reduce_max(ଦଶମିକ x,
    ଅଚଳ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    ଫେରନ୍ତୁ x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<ପୂର୍ଣ୍ଣସଂଖ୍ୟା N>
__dpct_inline__ size_t calculate_offset(ଅଚଳ std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, N> & strides, ଅଚଳ std::array<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, N> & indices) {
    size_t offset = 0;
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    ଫେରନ୍ତୁ offset;
}

// Helper for vec loading aligned data
template <typename Tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n>
ଧାଡ଼ିଭିତରେ sycl::vec<Tp, n> vec_aligned_load(ଅଚଳ Tp* aligned_ptr) {
    ଫେରନ୍ତୁ *reinterpret_cast<ଅଚଳ sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dim>
ସ୍ଥିର __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    ଫେରନ୍ତୁ acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_ଅଙ୍କ, int64_t block_size);

constexpr size_t ceil_div(ଅଚଳ size_t m, ଅଚଳ size_t n) {
    ଫେରନ୍ତୁ (m + n - 1) / n;
}

ତାର୍କିକ gpu_has_xmx(sycl::device &dev);

#ଯଦି GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_dnnl_has_optimized_gemm(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଫେରନ୍ତୁ ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_env(ଅଚଳ ଅକ୍ଷର *env_name, ପୂର୍ଣ୍ଣସଂଖ୍ୟା default_val);

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ଶ୍ରେଣୀ T> std::string debug_get_array_str(ଅଚଳ std::string & prefix, ଅଚଳ T array[N]) {
    ଯଦି (LIKELY(!g_ggml_sycl_debug)) {
        ଫେରନ୍ତୁ "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    ପାଇଁ (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    ଯଦି constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    ଫେରନ୍ତୁ ss.str();
}

ଧାଡ଼ିଭିତରେ std::string debug_get_tensor_str(ଅଚଳ std::string &prefix,
        ଅଚଳ ggml_tensor *tensor, ଅଚଳ std::string &suffix = "") {
    std::stringstream ss;
    ଯଦି (LIKELY(!g_ggml_sycl_debug)) { ଫେରନ୍ତୁ ss.str(); }
    ss << prefix.c_str() << "=";
    ଯଦି (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        ଯଦି (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        ଯଦି (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } ଅନ୍ୟଥା {
        ss << "nullptr";
    }
    ss << suffix;
    ଫେରନ୍ତୁ ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ସଂରଚନା scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ଅଚଳ std::string_view & func, ଅଚଳ std::string_view & func_suffix, ଅଚଳ ggml_tensor * dst,
                         std::size_t ଅଙ୍କ_src, ଅଚଳ std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        ଯଦି (LIKELY(!g_ggml_sycl_debug)) {
            ଫେରନ୍ତୁ;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        ଯଦି (dst) {
            ପାଇଁ (std::size_t i = 0; i < ଅଙ୍କ_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(ଅଚଳ std::string_view & func, ଅଚଳ ggml_tensor * dst, std::size_t ଅଙ୍କ_src,
                         ଅଚଳ std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, ଅଙ୍କ_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  ବ୍ୟକ୍ତିଗତ:
    std::string_view func;
    std::string_view func_suffix;
};

ସ୍ଥିର __dpct_inline__ ଦଶମିକ get_alibi_slope(ଅଚଳ ଦଶମିକ    max_bias,
                                             ଅଚଳ uint32_t h,
                                             ଅଚଳ uint32_t n_head_log2,
                                             ଅଚଳ ଦଶମିକ    m0,
                                             ଅଚଳ ଦଶମିକ    m1) {
    ଯଦି (max_bias <= 0.0f) {
        ଫେରନ୍ତୁ 1.0f;
    }
    ଅଚଳ ଦଶମିକ base = h < n_head_log2 ? m0 : m1;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    ଫେରନ୍ତୁ dpct::pow(base, exph);
}

ସ୍ଥିର ଅଚଳ sycl::uint3 init_fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    ଯେତେବେଳେ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    ଫେରନ୍ତୁ sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_max_cpy_bytes() {
    ଫେରନ୍ତୁ 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା nbytes, ପୂର୍ଣ୍ଣସଂଖ୍ୟା alignment = 0>
ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ggml_sycl_memcpy_1(ଶୂନ୍ୟ * dst, ଅଚଳ ଶୂନ୍ୟ * src) {
    ଯଦି constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < nbytes/nb_per_cpy; ++i) {
        ଯଦି constexpr (nb_per_cpy == 1) {
            ((ଅକ୍ଷର *) dst)[i] = ((ଅଚଳ ଅକ୍ଷର *) src)[i];
        } ଅନ୍ୟଥା ଯଦି constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((ଅଚଳ short *) src)[i];
        } ଅନ୍ୟଥା ଯଦି constexpr (nb_per_cpy == 4) {
            ((ପୂର୍ଣ୍ଣସଂଖ୍ୟା *) dst)[i] = ((ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *) src)[i];
        } ଅନ୍ୟଥା ଯଦି constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ଅଚଳ sycl::int2 *) src)[i];
        } ଅନ୍ୟଥା ଯଦି constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ଅଚଳ sycl::int4 *) src)[i];
        } ଅନ୍ୟଥା {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    ଫେରନ୍ତୁ res;
}

ସ୍ଥିର __dpct_inline__ uint32_t fastdiv(uint32_t n, ଅଚଳ sycl::uint3 fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ) {
    ଅଚଳ uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ.x());
    ଫେରନ୍ତୁ (hi + n) >> fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ଦଶମିକ>(x),static_cast<ଦଶମିକ>(y));
    ଫେରନ୍ତୁ res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_ମୂଲ୍ୟ(static_cast<ଦଶମିକ>(H.x()), static_cast<ଦଶମିକ>(H.y()));
    ଫେରନ୍ତୁ float2_ମୂଲ୍ୟ;
}

ସ୍ଥିର __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ଅଚଳ sycl::uint3 fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ) {
    ଅଚଳ uint32_t div_val = fastdiv(n, fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ);
    ଅଚଳ uint32_t mod_val = n - div_val * fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ.z();
    ଫେରନ୍ତୁ sycl::uint2(div_val, mod_val);
}

ସ୍ଥିର __dpct_inline__ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_dp4a(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା a, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା c) {
    ଫେରନ୍ତୁ dpct::dp4a(a, b, c);
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    ଯଦି (x == 0) {
        bits = 0x00400000;
    } ଅନ୍ୟଥା {
        bits = (uint32_t) x << 23;
    }

    ଦଶମିକ result;
    memcpy(&result, &bits, ଆକାର(ଦଶମିକ));
    ଫେରନ୍ତୁ result;
}

sycl::float2 __dpct_inline__ __half22float2(ଅଚଳ sycl::half2 &H) {
    sycl::float2 float2_ମୂଲ୍ୟ(static_cast<ଦଶମିକ>(H.x()), static_cast<ଦଶମିକ>(H.y()));
    ଫେରନ୍ତୁ float2_ମୂଲ୍ୟ;
}

ଦଶମିକ __dpct_inline__ __half2float(sycl::half H) {
    ଫେରନ୍ତୁ static_cast<ଦଶମିକ>(H);
}

ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ggml_sycl_mad(ଦଶମିକ & acc, ଅଚଳ ଦଶମିକ v, ଅଚଳ ଦଶମିକ u) {
    acc += v*u;
}

ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ggml_sycl_mad(ଦଶମିକ & acc, ଅଚଳ sycl::float2 v, ଅଚଳ sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ggml_sycl_mad(ଦଶମିକ & acc, ଅଚଳ sycl::half2 v, ଅଚଳ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    ଅଚଳ sycl::float2 tmp = (v * u).template convert<ଦଶମିକ, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#ଅନ୍ୟଥା
    ଅଚଳ sycl::float2 tmpv = __half22float2(v);
    ଅଚଳ sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

ସ୍ଥିର __dpct_inline__ ଶୂନ୍ୟ ggml_sycl_mad(sycl::half2 & acc, ଅଚଳ sycl::half2 v, ଅଚଳ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#ଅନ୍ୟଥା
    ଅଚଳ sycl::float2 tmpv = __half22float2(v);
    ଅଚଳ sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n>
ସଂରଚନା ggml_sycl_unroll {
    template <typename Func, typename... Args>
    ଶୂନ୍ୟ operator()(ଅଚଳ Func & f, Args... args) ଅଚଳ {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
ସଂରଚନା ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    ଶୂନ୍ୟ operator()(ଅଚଳ Func & f, Args... args) ଅଚଳ {
        f(0, args...);
    }
};

ସ୍ଥିର __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ଅଚଳ sycl::half2 a, ଅଚଳ sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ଦଶମିକ, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ଦଶମିକ, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    ଫେରନ୍ତୁ ret;
}

ସ୍ଥିର __dpct_inline__ sycl::half ggml_sycl_hmax(ଅଚଳ sycl::half a, ଅଚଳ sycl::half b) {
    ଫେରନ୍ତୁ sycl::vec<ଦଶମିକ, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

ସ୍ଥିର __dpct_inline__ uint32_t __hgt2_mask(ଅଚଳ sycl::half2 a, ଅଚଳ sycl::half2 b) {
    ଅଚଳ uint32_t mask_ତଳ  = 0x0000FFFF * (ଦଶମିକ(a[0]) > ଦଶମିକ(b[0]));
    ଅଚଳ uint32_t mask_ଉପର = 0xFFFF0000 * (ଦଶମିକ(a[1]) > ଦଶମିକ(b[1]));
    ଫେରନ୍ତୁ mask_ତଳ | mask_ଉପର;
}

ସ୍ଥିର __dpct_inline__ uint32_t fastmodulo(uint32_t n, ଅଚଳ sycl::uint3 fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    ଫେରନ୍ତୁ n - fastdiv(n, fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ) * fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ.z();
}

ସ୍ଥିର ତାର୍କିକ fast_fp16_available(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା cc) {
    GGML_UNUSED(cc);
    ଫେରନ୍ତୁ ସତ୍ୟ;   //Intel GPUs always support FP16.
}

ଗଣନା ଶ୍ରେଣୀ block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size>
ସଂରଚନା block_reduce_policy;

template <typename T, typename... Ts>
ଧାଡ଼ିଭିତରେ constexpr ତାର୍କିକ is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
ଧାଡ଼ିଭିତରେ constexpr ତାର୍କିକ ggml_sycl_dependent_false_v = ମିଥ୍ୟା;

#define WARP_32_SIZE 32

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size> ସଂରଚନା block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    ସ୍ଥିର T reduce(T val) {
        ଯଦି constexpr (is_any<T, ଦଶମିକ, sycl::float2, sycl::half2, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>) {
            ଫେରନ୍ତୁ warp_reduce_sum<warp_size>(val);
        } ଅନ୍ୟଥା {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ସ୍ଥିର T sentinel() {
        ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
            ଫେରନ୍ତୁ 0.0f;
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, sycl::float2>) {
            ଫେରନ୍ତୁ sycl::float2(0.0f, 0.0f);
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, sycl::half2>) {
            ଫେରନ୍ତୁ sycl::half2(0.0f, 0.0f);
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>) {
            ଫେରନ୍ତୁ 0;
        } ଅନ୍ୟଥା {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size> ସଂରଚନା block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    ସ୍ଥିର T reduce(T val) {
        ଯଦି constexpr (is_any<T, ଦଶମିକ, sycl::half2>) {
            ଫେରନ୍ତୁ warp_reduce_max<warp_size>(val);
        } ଅନ୍ୟଥା {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ସ୍ଥିର T sentinel() {
        ଯଦି constexpr (std::is_same_v<T, ଦଶମିକ>) {
            ଫେରନ୍ତୁ -INFINITY;
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_same_v<T, sycl::half2>) {
            ଫେରନ୍ତୁ sycl::half2(-INFINITY, -INFINITY);
        } ଅନ୍ୟଥା {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_size, typename T>
ସ୍ଥିର T block_reduce(T val, T * shared_vals, ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nthreads = item_ct1.get_local_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nwarps = nthreads / WARP_SIZE;

    ଯଦି (block_size > warp_size) {
        ପ୍ରତିଜ୍ଞା((block_size <= 1024) && (block_size % warp_size) == 0);
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା warp_id = item_ct1.get_local_id(2) / warp_size;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane_id = item_ct1.get_local_id(2) % warp_size;
        ଯଦି (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        ଦଶମିକ tmp = 0.f;
        ଯଦି (lane_id < (static_cast<ପୂର୍ଣ୍ଣସଂଖ୍ୟା>(block_size) / warp_size)) {
            ପାଇଁ (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        ଫେରନ୍ତୁ block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    ଫେରନ୍ତୁ val;
}

ସ୍ଥିର __dpct_inline__ ଦଶମିକ ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    ଯଦି (x == 0 || x == 0x7F) {
        ଫେରନ୍ତୁ 0.0f;
    }
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା exp = (x >> 3) & 0xF;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା man = x & 0x7;
    ଦଶମିକ raw;
    ଯଦି (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } ଅନ୍ୟଥା {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (ଦଶମିକ) exp - 7.0f);
    }
    ଫେରନ୍ତୁ raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
