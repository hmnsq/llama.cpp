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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_COMMON_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_COMMON_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>

#ਅਨ੍ਤਰ੍ਭਾਵ "base.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "presets.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "type.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "sycl_hw.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "memtrace.hpp"

ਨਾਮਾਕਾਸ਼ syclexp = sycl::ext::oneapi::experimental;

#ਜੇ defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/bfloat16.hpp>
    #ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
        #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_HAS_BF16
    #ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ

#ਜੇ defined(__INTEL_LLVM_COMPILER)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DMMV_HAS_ESIMD
#ਜੇ_ਅੰਤ

#ਜੇ GGML_SYCL_DNNL
#ਅਨ੍ਤਰ੍ਭਾਵ "dnnl.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dnnl_sycl.hpp"
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_SYCL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_SYCL
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#ਹਦਾਇਤ clang diagnostic ਧੱਕੋ
#ਹਦਾਇਤ clang diagnostic ignored "-Wnested-anon-types"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਹਦਾਇਤ clang diagnostic ਕੱਢੋ
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"

ਖਾਲੀ* ggml_sycl_host_malloc(size_t size);
ਖਾਲੀ ggml_sycl_host_free(ਖਾਲੀ* ptr);


ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_debug;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_optimize;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_fusion;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_esimd;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_mmvq_wide;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ਗਿਣਤੀ ggml_sycl_xmx_gather_type {
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
ਸਥਿਰ constexpr ਪੂਰਨ GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ਸਥਿਰ constexpr ਪੂਰਨ GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ਗਿਣਤੀ ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
ਸਥਿਰ constexpr ਪੂਰਨ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#ਨਹੀਂਤਾਂ
ਸਥਿਰ constexpr ਪੂਰਨ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#ਜੇ_ਅੰਤ
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ਸਥਿਰ ਸਤਰਅੰਦਰ int32_t ggml_sycl_src1_prec(ਅਚਲ ggml_tensor * dst) {
    ਵਾਪਸ g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ਸਥਿਰ ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_src1_f16_ok(ਅਚਲ ggml_tensor * dst) {
    ਅਚਲ int32_t src1_prec = ggml_sycl_src1_prec(dst);
    ਵਾਪਸ g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_flash_attention;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_dev2dev_memcpy;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_fa_onednn;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_fa_onednn_max_kv;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_enable_mkl_fa;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_memtrace;
ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_memtrace_ਕਦਮ;


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CHECK_TRY_ERROR(ਵਿਅੰਜਕ)                                            \
  [&]() {                                                                \
    ਕੋਸ਼ਿਸ਼ {                                                                \
      ਵਿਅੰਜਕ;                                                              \
      ਵਾਪਸ dpct::success;                                              \
    } ਫੜੋ (std::exception ਅਚਲ& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::ਸਤਰ_ਅੰਤ;                                            \
      ਵਾਪਸ dpct::default_error;                                        \
    }                                                                    \
  }()


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SYCL_ARCH__ DPCT_COMPATIBILITY_ਆਰਜ਼ੀ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VER_4VEC 610 // todo for hardware optimize.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VER_GEN9 700 // todo for hardware optimize.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VER_GEN12 1000000 // todo for hardware optimize.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#ਜੇ !defined(GGML_SYCL_FORCE_MMQ)
    #ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_USE_XMX
#ਜੇ_ਅੰਤ

// max batch size to use MMQ kernels when tensor cores are available
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMQ_MAX_BATCH_ਮਾਪ 32

// dmmv = dequantize_mul_mat_vec
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_X
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DMMV_X 32
#ਜੇ_ਅੰਤ
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_MMV_Y
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_MMV_Y 1
#ਜੇ_ਅੰਤ

ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::queue *ਕਤਾਰ_ptr;

ਗਿਣਤੀ ggml_sycl_backend_gpu_ਬਹੁਲਕ {
  SYCL_UNSET_GPU_ਬਹੁਲਕ = -1,
  SYCL_SINGLE_GPU_ਬਹੁਲਕ = 0,
  SYCL_MUL_GPU_ਬਹੁਲਕ
};

ਗਿਣਤੀ ggml_sycl_dev2dev_memcpy_ਬਹੁਲਕ {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(ਆਕਾਰ(sycl::half) == ਆਕਾਰ(ggml_fp16_t), "wrong fp16 size");

ਸਥਿਰ ਖਾਲੀ crash() {
  ਪੂਰਨ* ptr = ਨਿਰਾ;
  *ptr = 0;
}

[[noreturn]] ਸਥਿਰ ਖਾਲੀ ggml_sycl_error(
    ਅਚਲ ਅੱਖਰ* stmt,
    ਅਚਲ ਅੱਖਰ* func,
    ਅਚਲ ਅੱਖਰ* file,
    ਅਚਲ ਪੂਰਨ ਸਤਰ,
    ਅਚਲ ਅੱਖਰ* ਸੁਨੇਹਾ_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, ਸੁਨੇਹਾ_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, ਸਤਰ);
  GGML_ABORT("SYCL error");
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_CHECK(err)                                                                                    \
    ਕਰੋ {                                                                                                   \
        ਸਵੈਚਾਲਿਤ err_ = (err);                                                                                 \
        ਜੇ (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } ਜਦੋਂ (0)

#ਜੇ DPCT_COMPAT_RT_ਸੰਸਕਰਣ >= 11100
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_ASSUME(x) __builtin_assume(x)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_ASSUME(x)
#ਜੇ_ਅੰਤ // DPCT_COMPAT_RT_VERSION >= 11100

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::half dfloat; // dequantize float
ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::half2 dfloat2;
#ਨਹੀਂਤਾਂ
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ dfloat; // dequantize float
ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::float2 dfloat2;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MMVQ_MAX_BATCH_ਮਾਪ  8

ਸਥਿਰ ਪੂਰਨ g_all_sycl_device_ਗਿਣਤੀ = -1;
ਸਥਿਰ ਤਾਰਕਿਕ g_ggml_backend_sycl_ਬਫਰ_type_initialized = ਝੂਠ;

ਸਥਿਰ ggml_sycl_backend_gpu_ਬਹੁਲਕ g_ggml_sycl_backend_gpu_ਬਹੁਲਕ =
    SYCL_UNSET_GPU_ਬਹੁਲਕ;

ਸਥਿਰ ਖਾਲੀ* g_scratch_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
ਸਥਿਰ size_t g_scratch_ਮਾਪ = 0; // disabled by default
ਸਥਿਰ size_t g_scratch_ਵਿੱਥ = 0;

[[noreturn]] ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bad_arch(ਅਚਲ sycl::stream& ਧਾਰਾ_ct1) {
  ਧਾਰਾ_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);

  (ਖਾਲੀ)bad_arch; // suppress unused function warning
}

ਪੂਰਨ get_current_device_id();

ਸਤਰਅੰਦਰ ਪੂਰਨ ggml_sycl_get_device() {
    ਵਾਪਸ get_current_device_id();
}

ਸਤਰਅੰਦਰ dpct::err0 ggml_sycl_ਸੈੱਟ_device(ਅਚਲ ਪੂਰਨ device) ਕੋਸ਼ਿਸ਼ {
  ਪੂਰਨ current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  ਜੇ (device == current_device_id) {
    ਵਾਪਸ 0;
  }

  ਵਾਪਸ CHECK_TRY_ERROR(dpct::select_device(device));
} ਫੜੋ (sycl::exception ਅਚਲ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  crash();
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

//////////////////////
ਬਣਤਰ optimize_feature {
    ਤਾਰਕਿਕ reorder=ਝੂਠ;
    ਤਾਰਕਿਕ onednn_optimized_gemm=ਝੂਠ;
};

ਬਣਤਰ sycl_device_info {
    ਪੂਰਨ cc;  // compute capability
    ਪੂਰਨ nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    ਪੂਰਨ warp_ਮਾਪ;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    ਪੂਰਨ max_wg_ਪ੍ਰਤੀ_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    ਤਾਰਕਿਕ    vmm;                // virtual memory support
    ਤਾਰਕਿਕ    l0_device_type_valid;
    ਤਾਰਕਿਕ    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  ਕੁੱਲ_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    ਤਾਰਕਿਕ    usm_system_support; // support for USM system allocations
};


ਬਣਤਰ ggml_sycl_device_info {
    ਪੂਰਨ device_ਗਿਣਤੀ;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> default_tensor_ਵੰਡੋ = {};

    ਪੂਰਨ max_work_ਸਮੂਹ_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    ਤਾਰਕਿਕ ext_oneapi_level_zero = ਸੱਚ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ਅਚਲ ggml_sycl_device_info & ggml_sycl_info();

ਸਥਿਰ constexpr size_t SYCL_ਬਫਰ_ALIGNMENT = 128;

ਬਣਤਰ ggml_sycl_pool {
    ਆਭਾਸੀ ~ggml_sycl_pool() = ਮੂਲ;

    ਆਭਾਸੀ ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) = 0;
    ਆਭਾਸੀ ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) = 0;
};

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਬਣਤਰ ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = ਖਾਲੀ_ਸੂਚਕ;
    T * ptr = ਖਾਲੀ_ਸੂਚਕ;
    size_t actual_ਮਾਪ = 0;

    ਸਪਸ਼ਟ ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        ਜੇ (ptr != ਖਾਲੀ_ਸੂਚਕ) {
            pool->ਮੁਕਤ(ptr, actual_ਮਾਪ);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (ptr)
            pool->ਮੁਕਤ(ptr, actual_ਮਾਪ);
        ptr = (T *) pool->alloc(size * ਆਕਾਰ(T), &ਇਹ->actual_ਮਾਪ);
        ਵਾਪਸ ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != ਖਾਲੀ_ਸੂਚਕ);
        GGML_ASSERT(ptr == ਖਾਲੀ_ਸੂਚਕ);
        ptr = (T *) pool->alloc(size * ਆਕਾਰ(T), &ਇਹ->actual_ਮਾਪ);
        ਵਾਪਸ ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ਇਹ->pool = &pool;
        ਵਾਪਸ alloc(size);
    }

    T * get() {
        ਵਾਪਸ ptr;
    }

    ggml_sycl_pool_alloc() = ਮੂਲ;
    ggml_sycl_pool_alloc(ਅਚਲ ggml_sycl_pool_alloc &) = ਮਿਟਾਓ;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = ਮਿਟਾਓ;
    ggml_sycl_pool_alloc& ਸੰਚਾਲਕ=(ਅਚਲ ggml_sycl_pool_alloc &) = ਮਿਟਾਓ;
    ggml_sycl_pool_alloc& ਸੰਚਾਲਕ=(ggml_sycl_pool_alloc &&) = ਮਿਟਾਓ;
};

// backend interface

ਬਣਤਰ ggml_tensor_extra_gpu {
  ਖਾਲੀ* ਅੰਕੜੇ_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

ਬਾਹਰੀ ਪੂਰਨ g_ggml_sycl_use_level_zero_api;
ਖਾਲੀ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_ਸਿੱਧਾ);
ਖਾਲੀ ggml_sycl_free_device(ਖਾਲੀ *ptr, sycl::queue &q);

ਖਾਲੀ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<ਕਤਾਰ_ptr> streams={});

ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
    int32_t i1;
    int32_t i2;
};

ਬਣਤਰ ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

ਨਾਮਾਕਾਸ਼ sycl_ex = sycl::ext::oneapi::experimental;
ਬਣਤਰ ggml_backend_sycl_context {
    ਪੂਰਨ device;
    std::string name;
    optimize_feature opt_feature;

    ਕਤਾਰ_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { ਖਾਲੀ_ਸੂਚਕ } };

    ਸਪਸ਼ਟ ggml_backend_sycl_context(ਪੂਰਨ device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    ਕਤਾਰ_ptr stream(ਪੂਰਨ device, ਪੂਰਨ stream) {
        ਜੇ (qptrs[device][stream] == ਖਾਲੀ_ਸੂਚਕ) {
            qptrs[device][stream] = &(dpct::get_device(device).default_ਕਤਾਰ());
        }
        ਵਾਪਸ qptrs[device][stream];
    }

    ਕਤਾਰ_ptr stream() {
        ਵਾਪਸ stream(device, 0);
    }

#ਜੇ GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ਅਚਲ dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        ਵਾਪਸ eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> ਧਾਰਾ_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream ਧਾਰਾ_dnnl(ਪੂਰਨ device, ਪੂਰਨ ਧਾਰਾ_2) {
        ਸਵੈਚਾਲਿਤ q = stream(device, ਧਾਰਾ_2);
        ਵਾਪਸ ਧਾਰਾ_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        ਸਵੈਚਾਲਿਤ it = engine_map.find(qptr);
        ਜੇ (it == engine_map.end()) {
            ਸਵੈਚਾਲਿਤ eng = make_engine(qptr);
            engine_map[qptr] = eng;
            ਵਾਪਸ eng;
        }
        ਨਹੀਂਤਾਂ
        {
            ਵਾਪਸ it->second;
        }
    }
    dnnl::stream ਧਾਰਾ_dnnl(sycl::queue* qptr) {
        ਸਵੈਚਾਲਿਤ it = ਧਾਰਾ_map.find(qptr);
        ਜੇ (it == ਧਾਰਾ_map.end()) {
            ਸਵੈਚਾਲਿਤ eng = engine_dnnl(qptr);
            ਸਵੈਚਾਲਿਤ stream = dnnl::sycl_interop::make_ਧਾਰਾ(eng, *qptr);
            ਧਾਰਾ_map[qptr] = stream;
            ਵਾਪਸ stream;
        }
        ਨਹੀਂਤਾਂ
        {
            ਵਾਪਸ it->second;
        }
    }
    dnnl::stream ਧਾਰਾ_dnnl() {
        ਵਾਪਸ ਧਾਰਾ_dnnl(device, 0);
    }
#ਜੇ_ਅੰਤ

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_ਪੰਕਤੀ_mapping> mmid_ਪੰਕਤੀ_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_ਸਮਾਂ_ਸਾਰਣੀ_host;

    ਸਥਿਰ std::unique_ptr<ggml_sycl_pool> new_pool_for_device(ਕਤਾਰ_ptr qptr, ਪੂਰਨ device);

    ਸਥਿਰ std::unique_ptr<ggml_sycl_pool> new_pool_for_host(ਕਤਾਰ_ptr qptr, ਪੂਰਨ device);

    ggml_sycl_pool & pool(ਪੂਰਨ device) {
        ਜੇ (pools[device] == ਖਾਲੀ_ਸੂਚਕ) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        ਵਾਪਸ *pools[device];
    }

    ggml_sycl_pool & pool() {
        ਵਾਪਸ pool(device);
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ਗ੍ਰਾਫ
    std::unique_ptr<sycl_ex::command_ਗ੍ਰਾਫ<sycl_ex::ਗ੍ਰਾਫ_state::executable>> exec_ਗ੍ਰਾਫ = ਖਾਲੀ_ਸੂਚਕ;
#ਜੇ_ਅੰਤ

    ggml_sycl_pool & host_pool(ਪੂਰਨ device) {
        ਜੇ (host_pools[device] == ਖਾਲੀ_ਸੂਚਕ) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        ਵਾਪਸ *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { ਵਾਪਸ host_pool(device); }
};

// common device functions

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_reduce_ਜੋੜ(ਦਸ਼ਮਲਵ x,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        x += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), x, ਨਕਾਬ);
    }
    ਵਾਪਸ x;
}

ਸਥਿਰ __dpct_inline__ sycl::float2
warp_reduce_ਜੋੜ(sycl::float2 a, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        a.x() += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), a.x(),
            ਨਕਾਬ);
        a.y() += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), a.y(),
            ਨਕਾਬ);
    }
    ਵਾਪਸ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਪੂਰਨ warp_reduce_ਜੋੜ(ਪੂਰਨ x) {
  ਵਾਪਸ sycl::reduce_over_ਸਮੂਹ(
      sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_reduce_ਜੋੜ(ਦਸ਼ਮਲਵ x) {
#ਹਦਾਇਤ unroll
  ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
    x += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
        sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), x, ਵਿੱਥ, ਚੌੜਾਈ);
  }
  ਵਾਪਸ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_reduce_ਜੋੜ(ਦਸ਼ਮਲਵ x, ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1) {
#ਹਦਾਇਤ unroll
  ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
    x += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
        ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), x, ਵਿੱਥ);
  }
  ਵਾਪਸ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ sycl::float2 warp_reduce_ਜੋੜ(sycl::float2 a) {
#ਹਦਾਇਤ unroll
  ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
    a.x() += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
        sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), a.x(), ਵਿੱਥ,
        ਚੌੜਾਈ);
    a.y() += dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
        sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), a.y(), ਵਿੱਥ,
        ਚੌੜਾਈ);
  }
  ਵਾਪਸ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ sycl::half2 warp_reduce_ਜੋੜ(sycl::half2 a) {
#ਹਦਾਇਤ unroll
  ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
    a = a + dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), a, ਵਿੱਥ,
                ਚੌੜਾਈ);
  }
  ਵਾਪਸ a;
}

ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_get_physical_warp_ਮਾਪ() {
  // todo: for old iGPU + dGPU case, need to be changed.
  ਵਾਪਸ WARP_ਮਾਪ;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਪੂਰਨ warp_reduce_all(ਪੂਰਨ x) {
    ਜੇ (ਚੌੜਾਈ == ggml_sycl_get_physical_warp_ਮਾਪ()) {
        ਵਾਪਸ sycl::all_ਦਾ_ਸਮੂਹ(
            sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ()
                         .get_local_ਰੇਖਿਕ_id())) ||
                x);
    } ਨਹੀਂਤਾਂ {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
            x = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                    sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), x,
                    ਵਿੱਥ, ਚੌੜਾਈ) &&
                x;
        }
        ਵਾਪਸ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਪੂਰਨ warp_reduce_any(ਪੂਰਨ x) {
    ਜੇ (ਚੌੜਾਈ == ggml_sycl_get_physical_warp_ਮਾਪ()) {
        ਵਾਪਸ sycl::any_ਦਾ_ਸਮੂਹ(
            sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ()
                         .get_local_ਰੇਖਿਕ_id())) &&
                x);
    } ਨਹੀਂਤਾਂ {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
            x = dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                    sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), x,
                    ਵਿੱਥ, ਚੌੜਾਈ) ||
                x;
        }
        ਵਾਪਸ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ਸਾਂਚਾ <ਪੂਰਨ ਚੌੜਾਈ>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_reduce_max(ਦਸ਼ਮਲਵ x) {
#ਹਦਾਇਤ unroll
  ਲਈ (ਪੂਰਨ ਵਿੱਥ = ਚੌੜਾਈ / 2; ਵਿੱਥ > 0; ਵਿੱਥ >>= 1) {
    x = sycl::fmax(x, dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
                          sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ(), x,
                          ਵਿੱਥ, ਚੌੜਾਈ));
  }
  ਵਾਪਸ x;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ warp_reduce_max(ਦਸ਼ਮਲਵ x,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3>& ਵਸਤੂ_ct1) {
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        x = sycl::fmax(x, dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(
            ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), x, ਨਕਾਬ));
    }
    ਵਾਪਸ x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
ਸਾਂਚਾ<ਪੂਰਨ N>
__dpct_inline__ size_t calculate_ਵਿੱਥ(ਅਚਲ std::array<ਪੂਰਨ, N> & strides, ਅਚਲ std::array<ਪੂਰਨ, N> & indices) {
    size_t ਵਿੱਥ = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < N; i++) {
        ਸਵੈਚਾਲਿਤ ਸੂਚਕਾਂਕ_i = indices[i];
        ਵਿੱਥ += strides[i] * ਸੂਚਕਾਂਕ_i;
    }
    ਵਾਪਸ ਵਿੱਥ;
}

// Helper for vec loading aligned data
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Tp, ਪੂਰਨ n>
ਸਤਰਅੰਦਰ sycl::vec<Tp, n> vec_aligned_ਲੱਦੋ(ਅਚਲ Tp* aligned_ptr) {
    ਵਾਪਸ *reinterpret_cast<ਅਚਲ sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Tp, ਪੂਰਨ dim>
ਸਥਿਰ __dpct_inline__ Tp* get_ਸੂਚਕ(sycl::local_accessor<Tp, dim> acc) {
    ਵਾਪਸ acc.ਸਾਂਚਾ get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_ਖੰਡ_ਅੰਕ, int64_t ਖੰਡ_ਮਾਪ_3);

constexpr size_t ceil_div(ਅਚਲ size_t m, ਅਚਲ size_t n) {
    ਵਾਪਸ (m + n - 1) / n;
}

ਤਾਰਕਿਕ gpu_has_xmx(sycl::device &dev);

#ਜੇ GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_dnnl_has_optimized_gemm(ਪੂਰਨ device) {
    ਵਾਪਸ ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#ਜੇ_ਅੰਤ

ਪੂਰਨ ggml_sycl_get_env(ਅਚਲ ਅੱਖਰ *env_name, ਪੂਰਨ default_val);

ਸਾਂਚਾ <ਪੂਰਨ N, ਵਰਗ T> std::string debug_get_array_str(ਅਚਲ std::string & prefix, ਅਚਲ T array[N]) {
    ਜੇ (LIKELY(!g_ggml_sycl_debug)) {
        ਵਾਪਸ "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    ਲਈ (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    ਜੇ constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    ਵਾਪਸ ss.str();
}

ਸਤਰਅੰਦਰ std::string debug_get_tensor_str(ਅਚਲ std::string &prefix,
        ਅਚਲ ggml_tensor *tensor, ਅਚਲ std::string &suffix = "") {
    std::stringstream ss;
    ਜੇ (LIKELY(!g_ggml_sycl_debug)) { ਵਾਪਸ ss.str(); }
    ss << prefix.c_str() << "=";
    ਜੇ (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        ਜੇ (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        ਜੇ (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } ਨਹੀਂਤਾਂ {
        ss << "nullptr";
    }
    ss << suffix;
    ਵਾਪਸ ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ਬਣਤਰ scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ਅਚਲ std::string_view & func, ਅਚਲ std::string_view & func_suffix, ਅਚਲ ggml_tensor * dst,
                         std::size_t ਅੰਕ_src, ਅਚਲ std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        ਜੇ (LIKELY(!g_ggml_sycl_debug)) {
            ਵਾਪਸ;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.ਅੰਕੜੇ(), func_suffix.ਅੰਕੜੇ());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        ਜੇ (dst) {
            ਲਈ (std::size_t i = 0; i < ਅੰਕ_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.ਅੰਕੜੇ());
    }

    scope_op_debug_print(ਅਚਲ std::string_view & func, ਅਚਲ ggml_tensor * dst, std::size_t ਅੰਕ_src,
                         ਅਚਲ std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, ਅੰਕ_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.ਅੰਕੜੇ(), func_suffix.ਅੰਕੜੇ()); }

  ਨਿਜੀ:
    std::string_view func;
    std::string_view func_suffix;
};

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ get_alibi_ਢਲਾਨ(ਅਚਲ ਦਸ਼ਮਲਵ    max_bias,
                                             ਅਚਲ uint32_t h,
                                             ਅਚਲ uint32_t n_ਸਿਰ_log2,
                                             ਅਚਲ ਦਸ਼ਮਲਵ    m0,
                                             ਅਚਲ ਦਸ਼ਮਲਵ    m1) {
    ਜੇ (max_bias <= 0.0f) {
        ਵਾਪਸ 1.0f;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ ਅਧਾਰ = h < n_ਸਿਰ_log2 ? m0 : m1;
    ਅਚਲ ਪੂਰਨ   exph = h < n_ਸਿਰ_log2 ? h + 1 : 2*(h - n_ਸਿਰ_log2) + 1;

    ਵਾਪਸ dpct::ਘਾਤ(ਅਧਾਰ, exph);
}

ਸਥਿਰ ਅਚਲ sycl::uint3 init_fastdiv_ਮੁੱਲਾਂ(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    ਜਦੋਂ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    ਵਾਪਸ sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ਸਥਿਰ constexpr ਪੂਰਨ ggml_sycl_get_max_cpy_ਬਾਈਟਾਂ() {
    ਵਾਪਸ 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
ਸਾਂਚਾ <ਪੂਰਨ nbytes, ਪੂਰਨ alignment = 0>
ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ggml_sycl_memcpy_1(ਖਾਲੀ * dst, ਅਚਲ ਖਾਲੀ * src) {
    ਜੇ constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr ਪੂਰਨ nb_ਪ੍ਰਤੀ_cpy = alignment == 0 ? nbytes : alignment;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < nbytes/nb_ਪ੍ਰਤੀ_cpy; ++i) {
        ਜੇ constexpr (nb_ਪ੍ਰਤੀ_cpy == 1) {
            ((ਅੱਖਰ *) dst)[i] = ((ਅਚਲ ਅੱਖਰ *) src)[i];
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (nb_ਪ੍ਰਤੀ_cpy == 2) {
            ((ਛੋਟਾ *) dst)[i] = ((ਅਚਲ ਛੋਟਾ *) src)[i];
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (nb_ਪ੍ਰਤੀ_cpy == 4) {
            ((ਪੂਰਨ *) dst)[i] = ((ਅਚਲ ਪੂਰਨ *) src)[i];
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (nb_ਪ੍ਰਤੀ_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ਅਚਲ sycl::int2 *) src)[i];
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (nb_ਪ੍ਰਤੀ_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ਅਚਲ sycl::int4 *) src)[i];
        } ਨਹੀਂਤਾਂ {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    ਵਾਪਸ res;
}

ਸਥਿਰ __dpct_inline__ uint32_t fastdiv(uint32_t n, ਅਚਲ sycl::uint3 fastdiv_ਮੁੱਲਾਂ) {
    ਅਚਲ uint32_t hi = sycl::mul_hi<ਬਿਨਾਨਿਸ਼ਾਨੀ>(n, fastdiv_ਮੁੱਲਾਂ.x());
    ਵਾਪਸ (hi + n) >> fastdiv_ਮੁੱਲਾਂ.y();
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ਦਸ਼ਮਲਵ>(x),static_cast<ਦਸ਼ਮਲਵ>(y));
    ਵਾਪਸ res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_ਮੁੱਲ(static_cast<ਦਸ਼ਮਲਵ>(H.x()), static_cast<ਦਸ਼ਮਲਵ>(H.y()));
    ਵਾਪਸ float2_ਮੁੱਲ;
}

ਸਥਿਰ __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ਅਚਲ sycl::uint3 fastdiv_ਮੁੱਲਾਂ) {
    ਅਚਲ uint32_t div_val = fastdiv(n, fastdiv_ਮੁੱਲਾਂ);
    ਅਚਲ uint32_t mod_val = n - div_val * fastdiv_ਮੁੱਲਾਂ.z();
    ਵਾਪਸ sycl::uint2(div_val, mod_val);
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ ggml_sycl_dp4a(ਅਚਲ ਪੂਰਨ a, ਅਚਲ ਪੂਰਨ b, ਪੂਰਨ c) {
    ਵਾਪਸ dpct::dp4a(a, b, c);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(uint8_t x) {
    uint32_t ਬਿੱਟ;
    ਜੇ (x == 0) {
        ਬਿੱਟ = 0x00400000;
    } ਨਹੀਂਤਾਂ {
        ਬਿੱਟ = (uint32_t) x << 23;
    }

    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&ਨਤੀਜਾ, &ਬਿੱਟ, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਵਾਪਸ ਨਤੀਜਾ;
}

sycl::float2 __dpct_inline__ __half22float2(ਅਚਲ sycl::half2 &H) {
    sycl::float2 float2_ਮੁੱਲ(static_cast<ਦਸ਼ਮਲਵ>(H.x()), static_cast<ਦਸ਼ਮਲਵ>(H.y()));
    ਵਾਪਸ float2_ਮੁੱਲ;
}

ਦਸ਼ਮਲਵ __dpct_inline__ __half2float(sycl::half H) {
    ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(H);
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ggml_sycl_mad(ਦਸ਼ਮਲਵ & acc, ਅਚਲ ਦਸ਼ਮਲਵ v, ਅਚਲ ਦਸ਼ਮਲਵ u) {
    acc += v*u;
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ggml_sycl_mad(ਦਸ਼ਮਲਵ & acc, ਅਚਲ sycl::float2 v, ਅਚਲ sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ggml_sycl_mad(ਦਸ਼ਮਲਵ & acc, ਅਚਲ sycl::half2 v, ਅਚਲ sycl::half2 u) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਅਚਲ sycl::float2 tmp = (v * u).ਸਾਂਚਾ convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    acc += tmp.x() + tmp.y();
#ਨਹੀਂਤਾਂ
    ਅਚਲ sycl::float2 tmpv = __half22float2(v);
    ਅਚਲ sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ ggml_sycl_mad(sycl::half2 & acc, ਅਚਲ sycl::half2 v, ਅਚਲ sycl::half2 u) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    acc += v*u;
#ਨਹੀਂਤਾਂ
    ਅਚਲ sycl::float2 tmpv = __half22float2(v);
    ਅਚਲ sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਾਂਚਾ <ਪੂਰਨ n>
ਬਣਤਰ ggml_sycl_unroll {
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Func, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ Func & f, ਦਲੀਲਾਂ... ਦਲੀਲਾਂ_2) ਅਚਲ {
        f(n - 1, ਦਲੀਲਾਂ_2...);
        ggml_sycl_unroll<n - 1>{}(f, ਦਲੀਲਾਂ_2...);
    }
};

ਸਾਂਚਾ <>
ਬਣਤਰ ggml_sycl_unroll<1> {
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Func, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ>
    ਖਾਲੀ ਸੰਚਾਲਕ()(ਅਚਲ Func & f, ਦਲੀਲਾਂ... ਦਲੀਲਾਂ_2) ਅਚਲ {
        f(0, ਦਲੀਲਾਂ_2...);
    }
};

ਸਥਿਰ __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ਅਚਲ sycl::half2 a, ਅਚਲ sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ਦਸ਼ਮਲਵ, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ਦਸ਼ਮਲਵ, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
    ਵਾਪਸ ret;
}

ਸਥਿਰ __dpct_inline__ sycl::half ggml_sycl_hmax(ਅਚਲ sycl::half a, ਅਚਲ sycl::half b) {
    ਵਾਪਸ sycl::vec<ਦਸ਼ਮਲਵ, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
}

ਸਥਿਰ __dpct_inline__ uint32_t __hgt2_mask(ਅਚਲ sycl::half2 a, ਅਚਲ sycl::half2 b) {
    ਅਚਲ uint32_t ਨਕਾਬ_ਨੀਵਾਂ  = 0x0000FFFF * (ਦਸ਼ਮਲਵ(a[0]) > ਦਸ਼ਮਲਵ(b[0]));
    ਅਚਲ uint32_t ਨਕਾਬ_ਉੱਚਾ = 0xFFFF0000 * (ਦਸ਼ਮਲਵ(a[1]) > ਦਸ਼ਮਲਵ(b[1]));
    ਵਾਪਸ ਨਕਾਬ_ਨੀਵਾਂ | ਨਕਾਬ_ਉੱਚਾ;
}

ਸਥਿਰ __dpct_inline__ uint32_t fastmodulo(uint32_t n, ਅਚਲ sycl::uint3 fastdiv_ਮੁੱਲਾਂ) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    ਵਾਪਸ n - fastdiv(n, fastdiv_ਮੁੱਲਾਂ) * fastdiv_ਮੁੱਲਾਂ.z();
}

ਸਥਿਰ ਤਾਰਕਿਕ fast_fp16_ਉਪਲਬਧ(ਅਚਲ ਪੂਰਨ cc) {
    GGML_ਅਣਵਰਤਿਆ(cc);
    ਵਾਪਸ ਸੱਚ;   //Intel GPUs always support FP16.
}

ਗਿਣਤੀ ਵਰਗ ਖੰਡ_reduce_ਵਿਧੀ {
    MAX,
    ਜੋੜ_3,
};

ਸਾਂਚਾ<ਖੰਡ_reduce_ਵਿਧੀ ਵਿਧੀ_t, ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ warp_ਮਾਪ>
ਬਣਤਰ ਖੰਡ_reduce_policy;

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ... Ts>
ਸਤਰਅੰਦਰ constexpr ਤਾਰਕਿਕ is_any = (std::is_same_v<T, Ts> || ...);

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ...>
ਸਤਰਅੰਦਰ constexpr ਤਾਰਕਿਕ ggml_sycl_dependent_false_v = ਝੂਠ;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ WARP_32_ਮਾਪ 32

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ warp_ਮਾਪ> ਬਣਤਰ ਖੰਡ_reduce_policy<ਖੰਡ_reduce_ਵਿਧੀ::ਜੋੜ_3, T, warp_ਮਾਪ> {
    ਸਥਿਰ T reduce(T val) {
        ਜੇ constexpr (is_any<T, ਦਸ਼ਮਲਵ, sycl::float2, sycl::half2, ਪੂਰਨ>) {
            ਵਾਪਸ warp_reduce_ਜੋੜ<warp_ਮਾਪ>(val);
        } ਨਹੀਂਤਾਂ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ਸਥਿਰ T sentinel() {
        ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
            ਵਾਪਸ 0.0f;
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, sycl::float2>) {
            ਵਾਪਸ sycl::float2(0.0f, 0.0f);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, sycl::half2>) {
            ਵਾਪਸ sycl::half2(0.0f, 0.0f);
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, ਪੂਰਨ>) {
            ਵਾਪਸ 0;
        } ਨਹੀਂਤਾਂ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ warp_ਮਾਪ> ਬਣਤਰ ਖੰਡ_reduce_policy<ਖੰਡ_reduce_ਵਿਧੀ::MAX, T, warp_ਮਾਪ> {
    ਸਥਿਰ T reduce(T val) {
        ਜੇ constexpr (is_any<T, ਦਸ਼ਮਲਵ, sycl::half2>) {
            ਵਾਪਸ warp_reduce_max<warp_ਮਾਪ>(val);
        } ਨਹੀਂਤਾਂ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ਸਥਿਰ T sentinel() {
        ਜੇ constexpr (std::is_same_v<T, ਦਸ਼ਮਲਵ>) {
            ਵਾਪਸ -INFINITY;
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_same_v<T, sycl::half2>) {
            ਵਾਪਸ sycl::half2(-INFINITY, -INFINITY);
        } ਨਹੀਂਤਾਂ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


ਸਾਂਚਾ <ਖੰਡ_reduce_ਵਿਧੀ reduce_ਵਿਧੀ_t, ਪੂਰਨ warp_ਮਾਪ, ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ T ਖੰਡ_reduce(T val, T * shared_vals, ਪੂਰਨ ਖੰਡ_ਮਾਪ_template) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1                 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    val                           = ਖੰਡ_reduce_policy<reduce_ਵਿਧੀ_t, T,warp_ਮਾਪ>::reduce(val);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = ਖੰਡ_ਮਾਪ_template == 0 ? ਵਸਤੂ_ct1.get_local_range(2) : ਖੰਡ_ਮਾਪ_template;
    ਅਚਲ ਪੂਰਨ nthreads = ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ ਪੂਰਨ nwarps = nthreads / WARP_ਮਾਪ;

    ਜੇ (ਖੰਡ_ਮਾਪ_3 > warp_ਮਾਪ) {
        ਪ੍ਰਤਿਜ੍ਞਾ((ਖੰਡ_ਮਾਪ_3 <= 1024) && (ਖੰਡ_ਮਾਪ_3 % warp_ਮਾਪ) == 0);
        ਅਚਲ ਪੂਰਨ warp_id = ਵਸਤੂ_ct1.get_local_id(2) / warp_ਮਾਪ;
        ਅਚਲ ਪੂਰਨ lane_id = ਵਸਤੂ_ct1.get_local_id(2) % warp_ਮਾਪ;
        ਜੇ (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_ਮਾਪ;
        ਦਸ਼ਮਲਵ tmp = 0.f;
        ਜੇ (lane_id < (static_cast<ਪੂਰਨ>(ਖੰਡ_ਮਾਪ_3) / warp_ਮਾਪ)) {
            ਲਈ (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_ਮਾਪ];
            }
        }
        ਵਾਪਸ ਖੰਡ_reduce_policy<reduce_ਵਿਧੀ_t, T, warp_ਮਾਪ>::reduce(tmp);
    }
    ਵਾਪਸ val;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ ggml_sycl_ue4m3_ਮੰਜ਼ਿਲ_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    ਜੇ (x == 0 || x == 0x7F) {
        ਵਾਪਸ 0.0f;
    }
    ਅਚਲ ਪੂਰਨ ਘਾਤਾਂਕ = (x >> 3) & 0xF;
    ਅਚਲ ਪੂਰਨ man = x & 0x7;
    ਦਸ਼ਮਲਵ raw;
    ਜੇ (ਘਾਤਾਂਕ == 0) {
        raw = man * (1.0f / 8.0f) * sycl::ਘਾਤ(2.0f, -6.0f);
    } ਨਹੀਂਤਾਂ {
        raw = (1.0f + man / 8.0f) * sycl::ਘਾਤ(2.0f, (ਦਸ਼ਮਲਵ) ਘਾਤਾਂਕ - 7.0f);
    }
    ਵਾਪਸ raw * 0.5f;
}

#ਜੇ_ਅੰਤ // GGML_SYCL_COMMON_HPP
