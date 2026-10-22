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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_COMMON_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_COMMON_HPP

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

#ಆದರೆ defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/bfloat16.hpp>
    #ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_HAS_BF16
        #ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_HAS_BF16
    #ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__INTEL_LLVM_COMPILER)
    #ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DMMV_HAS_ESIMD
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ GGML_SYCL_DNNL
#ಅನ್ತರ್ಭಾವ "dnnl.hpp"
#ಅನ್ತರ್ಭಾವ "dnnl_sycl.hpp"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_SYCL
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_SYCL
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#ಸೂಚನೆ clang diagnostic ತಳ್ಳು
#ಸೂಚನೆ clang diagnostic ignored "-Wnested-anon-types"
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಸೂಚನೆ clang diagnostic ತೆಗೆ
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"

ಶೂನ್ಯ* ggml_sycl_host_malloc(size_t size);
ಶೂನ್ಯ ggml_sycl_host_free(ಶೂನ್ಯ* ptr);


ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_debug;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_optimize;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_fusion;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_esimd;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_mmvq_wide;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ಎಣಿಕೆ ggml_sycl_xmx_gather_type {
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
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ಎಣಿಕೆ ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#ಇಲ್ಲದಿದ್ದರೆ
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#ಆದರೆ_ಅಂತ್ಯ
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ಸ್ಥಿರ ಸಾಲೊಳಗೆ int32_t ggml_sycl_src1_prec(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_src1_f16_ok(ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ int32_t src1_prec = ggml_sycl_src1_prec(dst);
    ಹಿಂತಿರುಗಿ g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_flash_attention;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_dev2dev_memcpy;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_fa_onednn;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_fa_onednn_max_kv;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_mkl_fa;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_memtrace;
ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ;


#ವ್ಯಾಖ್ಯಾನಿಸು CHECK_TRY_ERROR(ಅಭಿವ್ಯಕ್ತಿ)                                            \
  [&]() {                                                                \
    ಪ್ರಯತ್ನಿಸು {                                                                \
      ಅಭಿವ್ಯಕ್ತಿ;                                                              \
      ಹಿಂತಿರುಗಿ dpct::success;                                              \
    } ಹಿಡಿ (std::exception ಸ್ಥಿರಾಂಕ& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::ಸಾಲು_ಅಂತ್ಯ;                                            \
      ಹಿಂತಿರುಗಿ dpct::default_error;                                        \
    }                                                                    \
  }()


#ವ್ಯಾಖ್ಯಾನಿಸು __SYCL_ARCH__ DPCT_COMPATIBILITY_ತಾತ್ಕಾಲಿಕ
#ವ್ಯಾಖ್ಯಾನಿಸು VER_4VEC 610 // todo for hardware optimize.
#ವ್ಯಾಖ್ಯಾನಿಸು VER_GEN9 700 // todo for hardware optimize.
#ವ್ಯಾಖ್ಯಾನಿಸು VER_GEN12 1000000 // todo for hardware optimize.
#ವ್ಯಾಖ್ಯಾನಿಸು VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#ಆದರೆ !defined(GGML_SYCL_FORCE_MMQ)
    #ವ್ಯಾಖ್ಯಾನಿಸು SYCL_USE_XMX
#ಆದರೆ_ಅಂತ್ಯ

// max batch size to use MMQ kernels when tensor cores are available
#ವ್ಯಾಖ್ಯಾನಿಸು MMQ_MAX_BATCH_ಗಾತ್ರ 32

// dmmv = dequantize_mul_mat_vec
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_DMMV_X
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DMMV_X 32
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_MMV_Y
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_MMV_Y 1
#ಆದರೆ_ಅಂತ್ಯ

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::queue *ಸರತಿ_ptr;

ಎಣಿಕೆ ggml_sycl_backend_gpu_ಬಹುಲಕ {
  SYCL_UNSET_GPU_ಬಹುಲಕ = -1,
  SYCL_SINGLE_GPU_ಬಹುಲಕ = 0,
  SYCL_MUL_GPU_ಬಹುಲಕ
};

ಎಣಿಕೆ ggml_sycl_dev2dev_memcpy_ಬಹುಲಕ {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(ಗಾತ್ರ(sycl::half) == ಗಾತ್ರ(ggml_fp16_t), "wrong fp16 size");

ಸ್ಥಿರ ಶೂನ್ಯ crash() {
  ಪೂರ್ಣಾಂಕ* ptr = ಖಾಲಿ;
  *ptr = 0;
}

[[noreturn]] ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_error(
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* stmt,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* func,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* file,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸಾಲು,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* ಸಂದೇಶ_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, ಸಂದೇಶ_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, ಸಾಲು);
  GGML_ABORT("SYCL error");
}

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_CHECK(err)                                                                                    \
    ಮಾಡು {                                                                                                   \
        ಸ್ವಯಂಚಾಲಿತ err_ = (err);                                                                                 \
        ಆದರೆ (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } ಆಗ (0)

#ಆದರೆ DPCT_COMPAT_RT_ಆವೃತ್ತಿ >= 11100
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_ASSUME(x) __builtin_assume(x)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_ASSUME(x)
#ಆದರೆ_ಅಂತ್ಯ // DPCT_COMPAT_RT_VERSION >= 11100

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::half dfloat; // dequantize float
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::half2 dfloat2;
#ಇಲ್ಲದಿದ್ದರೆ
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ dfloat; // dequantize float
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::float2 dfloat2;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

#ವ್ಯಾಖ್ಯಾನಿಸು MMVQ_MAX_BATCH_ಗಾತ್ರ  8

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ g_all_sycl_device_ಎಣಿಕೆ = -1;
ಸ್ಥಿರ ತಾರ್ಕಿಕ g_ggml_backend_sycl_ಬಫರ್_type_initialized = ಸುಳ್ಳು;

ಸ್ಥಿರ ggml_sycl_backend_gpu_ಬಹುಲಕ g_ggml_sycl_backend_gpu_ಬಹುಲಕ =
    SYCL_UNSET_GPU_ಬಹುಲಕ;

ಸ್ಥಿರ ಶೂನ್ಯ* g_scratch_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
ಸ್ಥಿರ size_t g_scratch_ಗಾತ್ರ = 0; // disabled by default
ಸ್ಥಿರ size_t g_scratch_ಸ್ಥಾನಾಂತರ = 0;

[[noreturn]] ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bad_arch(ಸ್ಥಿರಾಂಕ sycl::stream& ಹರಿವು_ct1) {
  ಹರಿವು_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::ನಿರ್ಗಮಿಸು(1);

  (ಶೂನ್ಯ)bad_arch; // suppress unused function warning
}

ಪೂರ್ಣಾಂಕ get_current_device_id();

ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ggml_sycl_get_device() {
    ಹಿಂತಿರುಗಿ get_current_device_id();
}

ಸಾಲೊಳಗೆ dpct::err0 ggml_sycl_ಗಣ_device(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device) ಪ್ರಯತ್ನಿಸು {
  ಪೂರ್ಣಾಂಕ current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  ಆದರೆ (device == current_device_id) {
    ಹಿಂತಿರುಗಿ 0;
  }

  ಹಿಂತಿರುಗಿ CHECK_TRY_ERROR(dpct::select_device(device));
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  crash();
  std::ನಿರ್ಗಮಿಸು(1);
}

//////////////////////
ರಚನೆ optimize_feature {
    ತಾರ್ಕಿಕ reorder=ಸುಳ್ಳು;
    ತಾರ್ಕಿಕ onednn_optimized_gemm=ಸುಳ್ಳು;
};

ರಚನೆ sycl_device_info {
    ಪೂರ್ಣಾಂಕ cc;  // compute capability
    ಪೂರ್ಣಾಂಕ nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    ಪೂರ್ಣಾಂಕ max_wg_ಪ್ರತಿ_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    ತಾರ್ಕಿಕ    vmm;                // virtual memory support
    ತಾರ್ಕಿಕ    l0_device_type_valid;
    ತಾರ್ಕಿಕ    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  ಒಟ್ಟು_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    ತಾರ್ಕಿಕ    usm_system_support; // support for USM system allocations
};


ರಚನೆ ggml_sycl_device_info {
    ಪೂರ್ಣಾಂಕ device_ಎಣಿಕೆ;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> default_tensor_ವಿಭಜಿಸು = {};

    ಪೂರ್ಣಾಂಕ max_work_ಗುಂಪು_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    ತಾರ್ಕಿಕ ext_oneapi_level_zero = ನಿಜ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ಸ್ಥಿರಾಂಕ ggml_sycl_device_info & ggml_sycl_info();

ಸ್ಥಿರ constexpr size_t SYCL_ಬಫರ್_ALIGNMENT = 128;

ರಚನೆ ggml_sycl_pool {
    ವಾಸ್ತವಿಕ ~ggml_sycl_pool() = ಪೂರ್ವನಿಯೋಜಿತ;

    ವಾಸ್ತವಿಕ ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) = 0;
    ವಾಸ್ತವಿಕ ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) = 0;
};

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ರಚನೆ ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = ಶೂನ್ಯಸೂಚಕ;
    T * ptr = ಶೂನ್ಯಸೂಚಕ;
    size_t actual_ಗಾತ್ರ = 0;

    ಸ್ಪಷ್ಟ ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        ಆದರೆ (ptr != ಶೂನ್ಯಸೂಚಕ) {
            pool->ಬಿಡುಗಡೆ(ptr, actual_ಗಾತ್ರ);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (ptr)
            pool->ಬಿಡುಗಡೆ(ptr, actual_ಗಾತ್ರ);
        ptr = (T *) pool->alloc(size * ಗಾತ್ರ(T), &ಇದು->actual_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != ಶೂನ್ಯಸೂಚಕ);
        GGML_ASSERT(ptr == ಶೂನ್ಯಸೂಚಕ);
        ptr = (T *) pool->alloc(size * ಗಾತ್ರ(T), &ಇದು->actual_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ಇದು->pool = &pool;
        ಹಿಂತಿರುಗಿ alloc(size);
    }

    T * get() {
        ಹಿಂತಿರುಗಿ ptr;
    }

    ggml_sycl_pool_alloc() = ಪೂರ್ವನಿಯೋಜಿತ;
    ggml_sycl_pool_alloc(ಸ್ಥಿರಾಂಕ ggml_sycl_pool_alloc &) = ಅಳಿಸು;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = ಅಳಿಸು;
    ggml_sycl_pool_alloc& ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ ggml_sycl_pool_alloc &) = ಅಳಿಸು;
    ggml_sycl_pool_alloc& ಕ್ರಿಯಾಕಾರಕ=(ggml_sycl_pool_alloc &&) = ಅಳಿಸು;
};

// backend interface

ರಚನೆ ggml_tensor_extra_gpu {
  ಶೂನ್ಯ* ದತ್ತಾಂಶ_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

ಬಾಹ್ಯ ಪೂರ್ಣಾಂಕ g_ggml_sycl_use_level_zero_api;
ಶೂನ್ಯ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_ನೇರ);
ಶೂನ್ಯ ggml_sycl_free_device(ಶೂನ್ಯ *ptr, sycl::queue &q);

ಶೂನ್ಯ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<ಸರತಿ_ptr> streams={});

ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping {
    int32_t i1;
    int32_t i2;
};

ರಚನೆ ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

ನಾಮಾಕಾಶ sycl_ex = sycl::ext::oneapi::experimental;
ರಚನೆ ggml_backend_sycl_context {
    ಪೂರ್ಣಾಂಕ device;
    std::string name;
    optimize_feature opt_feature;

    ಸರತಿ_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { ಶೂನ್ಯಸೂಚಕ } };

    ಸ್ಪಷ್ಟ ggml_backend_sycl_context(ಪೂರ್ಣಾಂಕ device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    ಸರತಿ_ptr stream(ಪೂರ್ಣಾಂಕ device, ಪೂರ್ಣಾಂಕ stream) {
        ಆದರೆ (qptrs[device][stream] == ಶೂನ್ಯಸೂಚಕ) {
            qptrs[device][stream] = &(dpct::get_device(device).default_ಸರತಿ());
        }
        ಹಿಂತಿರುಗಿ qptrs[device][stream];
    }

    ಸರತಿ_ptr stream() {
        ಹಿಂತಿರುಗಿ stream(device, 0);
    }

#ಆದರೆ GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ಸ್ಥಿರಾಂಕ dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        ಹಿಂತಿರುಗಿ eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> ಹರಿವು_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream ಹರಿವು_dnnl(ಪೂರ್ಣಾಂಕ device, ಪೂರ್ಣಾಂಕ ಹರಿವು_2) {
        ಸ್ವಯಂಚಾಲಿತ q = stream(device, ಹರಿವು_2);
        ಹಿಂತಿರುಗಿ ಹರಿವು_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        ಸ್ವಯಂಚಾಲಿತ it = engine_map.find(qptr);
        ಆದರೆ (it == engine_map.end()) {
            ಸ್ವಯಂಚಾಲಿತ eng = make_engine(qptr);
            engine_map[qptr] = eng;
            ಹಿಂತಿರುಗಿ eng;
        }
        ಇಲ್ಲದಿದ್ದರೆ
        {
            ಹಿಂತಿರುಗಿ it->second;
        }
    }
    dnnl::stream ಹರಿವು_dnnl(sycl::queue* qptr) {
        ಸ್ವಯಂಚಾಲಿತ it = ಹರಿವು_map.find(qptr);
        ಆದರೆ (it == ಹರಿವು_map.end()) {
            ಸ್ವಯಂಚಾಲಿತ eng = engine_dnnl(qptr);
            ಸ್ವಯಂಚಾಲಿತ stream = dnnl::sycl_interop::make_ಹರಿವು(eng, *qptr);
            ಹರಿವು_map[qptr] = stream;
            ಹಿಂತಿರುಗಿ stream;
        }
        ಇಲ್ಲದಿದ್ದರೆ
        {
            ಹಿಂತಿರುಗಿ it->second;
        }
    }
    dnnl::stream ಹರಿವು_dnnl() {
        ಹಿಂತಿರುಗಿ ಹರಿವು_dnnl(device, 0);
    }
#ಆದರೆ_ಅಂತ್ಯ

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_ಅಡ್ಡಸಾಲು_mapping> mmid_ಅಡ್ಡಸಾಲು_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_ವೇಳಾಪಟ್ಟಿ_host;

    ಸ್ಥಿರ std::unique_ptr<ggml_sycl_pool> new_pool_for_device(ಸರತಿ_ptr qptr, ಪೂರ್ಣಾಂಕ device);

    ಸ್ಥಿರ std::unique_ptr<ggml_sycl_pool> new_pool_for_host(ಸರತಿ_ptr qptr, ಪೂರ್ಣಾಂಕ device);

    ggml_sycl_pool & pool(ಪೂರ್ಣಾಂಕ device) {
        ಆದರೆ (pools[device] == ಶೂನ್ಯಸೂಚಕ) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        ಹಿಂತಿರುಗಿ *pools[device];
    }

    ggml_sycl_pool & pool() {
        ಹಿಂತಿರುಗಿ pool(device);
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_ನಕ್ಷೆ
    std::unique_ptr<sycl_ex::command_ನಕ್ಷೆ<sycl_ex::ನಕ್ಷೆ_state::executable>> exec_ನಕ್ಷೆ = ಶೂನ್ಯಸೂಚಕ;
#ಆದರೆ_ಅಂತ್ಯ

    ggml_sycl_pool & host_pool(ಪೂರ್ಣಾಂಕ device) {
        ಆದರೆ (host_pools[device] == ಶೂನ್ಯಸೂಚಕ) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        ಹಿಂತಿರುಗಿ *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { ಹಿಂತಿರುಗಿ host_pool(device); }
};

// common device functions

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_reduce_ಮೊತ್ತ(ದಶಮಾಂಶ x,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        x += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), x, ಮುಖವಾಡ);
    }
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ __dpct_inline__ sycl::float2
warp_reduce_ಮೊತ್ತ(sycl::float2 a, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        a.x() += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), a.x(),
            ಮುಖವಾಡ);
        a.y() += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), a.y(),
            ಮುಖವಾಡ);
    }
    ಹಿಂತಿರುಗಿ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ warp_reduce_ಮೊತ್ತ(ಪೂರ್ಣಾಂಕ x) {
  ಹಿಂತಿರುಗಿ sycl::reduce_over_ಗುಂಪು(
      sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_reduce_ಮೊತ್ತ(ದಶಮಾಂಶ x) {
#ಸೂಚನೆ unroll
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
    x += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
        sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), x, ಸ್ಥಾನಾಂತರ, ಅಗಲ);
  }
  ಹಿಂತಿರುಗಿ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_reduce_ಮೊತ್ತ(ದಶಮಾಂಶ x, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1) {
#ಸೂಚನೆ unroll
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
    x += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
        ಅಂಶ_ct1.get_sub_ಗುಂಪು(), x, ಸ್ಥಾನಾಂತರ);
  }
  ಹಿಂತಿರುಗಿ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ sycl::float2 warp_reduce_ಮೊತ್ತ(sycl::float2 a) {
#ಸೂಚನೆ unroll
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
    a.x() += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
        sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), a.x(), ಸ್ಥಾನಾಂತರ,
        ಅಗಲ);
    a.y() += dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
        sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), a.y(), ಸ್ಥಾನಾಂತರ,
        ಅಗಲ);
  }
  ಹಿಂತಿರುಗಿ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ sycl::half2 warp_reduce_ಮೊತ್ತ(sycl::half2 a) {
#ಸೂಚನೆ unroll
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
    a = a + dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), a, ಸ್ಥಾನಾಂತರ,
                ಅಗಲ);
  }
  ಹಿಂತಿರುಗಿ a;
}

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_get_physical_warp_ಗಾತ್ರ() {
  // todo: for old iGPU + dGPU case, need to be changed.
  ಹಿಂತಿರುಗಿ WARP_ಗಾತ್ರ;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ warp_reduce_all(ಪೂರ್ಣಾಂಕ x) {
    ಆದರೆ (ಅಗಲ == ggml_sycl_get_physical_warp_ಗಾತ್ರ()) {
        ಹಿಂತಿರುಗಿ sycl::all_ದ_ಗುಂಪು(
            sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು()
                         .get_local_ರೇಖೀಯ_id())) ||
                x);
    } ಇಲ್ಲದಿದ್ದರೆ {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
            x = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                    sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), x,
                    ಸ್ಥಾನಾಂತರ, ಅಗಲ) &&
                x;
        }
        ಹಿಂತಿರುಗಿ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ warp_reduce_any(ಪೂರ್ಣಾಂಕ x) {
    ಆದರೆ (ಅಗಲ == ggml_sycl_get_physical_warp_ಗಾತ್ರ()) {
        ಹಿಂತಿರುಗಿ sycl::any_ದ_ಗುಂಪು(
            sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು()
                         .get_local_ರೇಖೀಯ_id())) &&
                x);
    } ಇಲ್ಲದಿದ್ದರೆ {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
            x = dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                    sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), x,
                    ಸ್ಥಾನಾಂತರ, ಅಗಲ) ||
                x;
        }
        ಹಿಂತಿರುಗಿ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ ಅಗಲ>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_reduce_max(ದಶಮಾಂಶ x) {
#ಸೂಚನೆ unroll
  ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ = ಅಗಲ / 2; ಸ್ಥಾನಾಂತರ > 0; ಸ್ಥಾನಾಂತರ >>= 1) {
    x = sycl::fmax(x, dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
                          sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು(), x,
                          ಸ್ಥಾನಾಂತರ, ಅಗಲ));
  }
  ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ warp_reduce_max(ದಶಮಾಂಶ x,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3>& ಅಂಶ_ct1) {
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        x = sycl::fmax(x, dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(
            ಅಂಶ_ct1.get_sub_ಗುಂಪು(), x, ಮುಖವಾಡ));
    }
    ಹಿಂತಿರುಗಿ x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
ಮಾದರಿ<ಪೂರ್ಣಾಂಕ N>
__dpct_inline__ size_t calculate_ಸ್ಥಾನಾಂತರ(ಸ್ಥಿರಾಂಕ std::array<ಪೂರ್ಣಾಂಕ, N> & strides, ಸ್ಥಿರಾಂಕ std::array<ಪೂರ್ಣಾಂಕ, N> & indices) {
    size_t ಸ್ಥಾನಾಂತರ = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < N; i++) {
        ಸ್ವಯಂಚಾಲಿತ ಸೂಚ್ಯಂಕ_i = indices[i];
        ಸ್ಥಾನಾಂತರ += strides[i] * ಸೂಚ್ಯಂಕ_i;
    }
    ಹಿಂತಿರುಗಿ ಸ್ಥಾನಾಂತರ;
}

// Helper for vec loading aligned data
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Tp, ಪೂರ್ಣಾಂಕ n>
ಸಾಲೊಳಗೆ sycl::vec<Tp, n> vec_aligned_ತುಂಬು(ಸ್ಥಿರಾಂಕ Tp* aligned_ptr) {
    ಹಿಂತಿರುಗಿ *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Tp, ಪೂರ್ಣಾಂಕ dim>
ಸ್ಥಿರ __dpct_inline__ Tp* get_ಸೂಚಕ(sycl::local_accessor<Tp, dim> acc) {
    ಹಿಂತಿರುಗಿ acc.ಮಾದರಿ get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_ಖಂಡ_ಅಂಕೆ, int64_t ಖಂಡ_ಗಾತ್ರ_3);

constexpr size_t ceil_div(ಸ್ಥಿರಾಂಕ size_t m, ಸ್ಥಿರಾಂಕ size_t n) {
    ಹಿಂತಿರುಗಿ (m + n - 1) / n;
}

ತಾರ್ಕಿಕ gpu_has_xmx(sycl::device &dev);

#ಆದರೆ GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_dnnl_has_optimized_gemm(ಪೂರ್ಣಾಂಕ device) {
    ಹಿಂತಿರುಗಿ ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#ಆದರೆ_ಅಂತ್ಯ

ಪೂರ್ಣಾಂಕ ggml_sycl_get_env(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *env_name, ಪೂರ್ಣಾಂಕ default_val);

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ N, ವರ್ಗ T> std::string debug_get_array_str(ಸ್ಥಿರಾಂಕ std::string & prefix, ಸ್ಥಿರಾಂಕ T array[N]) {
    ಆದರೆ (LIKELY(!g_ggml_sycl_debug)) {
        ಹಿಂತಿರುಗಿ "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    ಗಾಗಿ (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    ಆದರೆ constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    ಹಿಂತಿರುಗಿ ss.str();
}

ಸಾಲೊಳಗೆ std::string debug_get_tensor_str(ಸ್ಥಿರಾಂಕ std::string &prefix,
        ಸ್ಥಿರಾಂಕ ggml_tensor *tensor, ಸ್ಥಿರಾಂಕ std::string &suffix = "") {
    std::stringstream ss;
    ಆದರೆ (LIKELY(!g_ggml_sycl_debug)) { ಹಿಂತಿರುಗಿ ss.str(); }
    ss << prefix.c_str() << "=";
    ಆದರೆ (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        ಆದರೆ (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        ಆದರೆ (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ss << "nullptr";
    }
    ss << suffix;
    ಹಿಂತಿರುಗಿ ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ರಚನೆ scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ಸ್ಥಿರಾಂಕ std::string_view & func, ಸ್ಥಿರಾಂಕ std::string_view & func_suffix, ಸ್ಥಿರಾಂಕ ggml_tensor * dst,
                         std::size_t ಅಂಕೆ_src, ಸ್ಥಿರಾಂಕ std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        ಆದರೆ (LIKELY(!g_ggml_sycl_debug)) {
            ಹಿಂತಿರುಗಿ;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.ದತ್ತಾಂಶ(), func_suffix.ದತ್ತಾಂಶ());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        ಆದರೆ (dst) {
            ಗಾಗಿ (std::size_t i = 0; i < ಅಂಕೆ_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.ದತ್ತಾಂಶ());
    }

    scope_op_debug_print(ಸ್ಥಿರಾಂಕ std::string_view & func, ಸ್ಥಿರಾಂಕ ggml_tensor * dst, std::size_t ಅಂಕೆ_src,
                         ಸ್ಥಿರಾಂಕ std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, ಅಂಕೆ_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.ದತ್ತಾಂಶ(), func_suffix.ದತ್ತಾಂಶ()); }

  ಖಾಸಗಿ:
    std::string_view func;
    std::string_view func_suffix;
};

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ get_alibi_ಇಳಿಜಾರು(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    max_bias,
                                             ಸ್ಥಿರಾಂಕ uint32_t h,
                                             ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2,
                                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    m0,
                                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    m1) {
    ಆದರೆ (max_bias <= 0.0f) {
        ಹಿಂತಿರುಗಿ 1.0f;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ತಳ = h < n_ತಲೆ_log2 ? m0 : m1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ   exph = h < n_ತಲೆ_log2 ? h + 1 : 2*(h - n_ತಲೆ_log2) + 1;

    ಹಿಂತಿರುಗಿ dpct::ಘಾತ(ತಳ, exph);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ sycl::uint3 init_fastdiv_ಮೌಲ್ಯಗಳು(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    ಆಗ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    ಹಿಂತಿರುಗಿ sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ ggml_sycl_get_max_cpy_ಬೈಟುಗಳು() {
    ಹಿಂತಿರುಗಿ 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ nbytes, ಪೂರ್ಣಾಂಕ alignment = 0>
ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_memcpy_1(ಶೂನ್ಯ * dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src) {
    ಆದರೆ constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr ಪೂರ್ಣಾಂಕ nb_ಪ್ರತಿ_cpy = alignment == 0 ? nbytes : alignment;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nbytes/nb_ಪ್ರತಿ_cpy; ++i) {
        ಆದರೆ constexpr (nb_ಪ್ರತಿ_cpy == 1) {
            ((ಅಕ್ಷರ *) dst)[i] = ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src)[i];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (nb_ಪ್ರತಿ_cpy == 2) {
            ((ಗಿಡ್ಡ *) dst)[i] = ((ಸ್ಥಿರಾಂಕ ಗಿಡ್ಡ *) src)[i];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (nb_ಪ್ರತಿ_cpy == 4) {
            ((ಪೂರ್ಣಾಂಕ *) dst)[i] = ((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) src)[i];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (nb_ಪ್ರತಿ_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ಸ್ಥಿರಾಂಕ sycl::int2 *) src)[i];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (nb_ಪ್ರತಿ_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ಸ್ಥಿರಾಂಕ sycl::int4 *) src)[i];
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    ಹಿಂತಿರುಗಿ res;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t fastdiv(uint32_t n, ಸ್ಥಿರಾಂಕ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    ಸ್ಥಿರಾಂಕ uint32_t hi = sycl::mul_hi<ಚಿಹ್ನೆರಹಿತ>(n, fastdiv_ಮೌಲ್ಯಗಳು.x());
    ಹಿಂತಿರುಗಿ (hi + n) >> fastdiv_ಮೌಲ್ಯಗಳು.y();
}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ದಶಮಾಂಶ>(x),static_cast<ದಶಮಾಂಶ>(y));
    ಹಿಂತಿರುಗಿ res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_ಮೌಲ್ಯ(static_cast<ದಶಮಾಂಶ>(H.x()), static_cast<ದಶಮಾಂಶ>(H.y()));
    ಹಿಂತಿರುಗಿ float2_ಮೌಲ್ಯ;
}

ಸ್ಥಿರ __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ಸ್ಥಿರಾಂಕ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    ಸ್ಥಿರಾಂಕ uint32_t div_val = fastdiv(n, fastdiv_ಮೌಲ್ಯಗಳು);
    ಸ್ಥಿರಾಂಕ uint32_t mod_val = n - div_val * fastdiv_ಮೌಲ್ಯಗಳು.z();
    ಹಿಂತಿರುಗಿ sycl::uint2(div_val, mod_val);
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ ggml_sycl_dp4a(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b, ಪೂರ್ಣಾಂಕ c) {
    ಹಿಂತಿರುಗಿ dpct::dp4a(a, b, c);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ ggml_sycl_e8m0_ಗಮ್ಯ_fp32(uint8_t x) {
    uint32_t ಬಿಟ್ಟುಗಳು;
    ಆದರೆ (x == 0) {
        ಬಿಟ್ಟುಗಳು = 0x00400000;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಬಿಟ್ಟುಗಳು = (uint32_t) x << 23;
    }

    ದಶಮಾಂಶ ಫಲಿತಾಂಶ;
    ಸ್ಮರಣೆ_ನಕಲು(&ಫಲಿತಾಂಶ, &ಬಿಟ್ಟುಗಳು, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

sycl::float2 __dpct_inline__ __half22float2(ಸ್ಥಿರಾಂಕ sycl::half2 &H) {
    sycl::float2 float2_ಮೌಲ್ಯ(static_cast<ದಶಮಾಂಶ>(H.x()), static_cast<ದಶಮಾಂಶ>(H.y()));
    ಹಿಂತಿರುಗಿ float2_ಮೌಲ್ಯ;
}

ದಶಮಾಂಶ __dpct_inline__ __half2float(sycl::half H) {
    ಹಿಂತಿರುಗಿ static_cast<ದಶಮಾಂಶ>(H);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ದಶಮಾಂಶ & acc, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ u) {
    acc += v*u;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ದಶಮಾಂಶ & acc, ಸ್ಥಿರಾಂಕ sycl::float2 v, ಸ್ಥಿರಾಂಕ sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(ದಶಮಾಂಶ & acc, ಸ್ಥಿರಾಂಕ sycl::half2 v, ಸ್ಥಿರಾಂಕ sycl::half2 u) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಸ್ಥಿರಾಂಕ sycl::float2 tmp = (v * u).ಮಾದರಿ convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    acc += tmp.x() + tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ sycl::float2 tmpv = __half22float2(v);
    ಸ್ಥಿರಾಂಕ sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ ggml_sycl_mad(sycl::half2 & acc, ಸ್ಥಿರಾಂಕ sycl::half2 v, ಸ್ಥಿರಾಂಕ sycl::half2 u) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    acc += v*u;
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ sycl::float2 tmpv = __half22float2(v);
    ಸ್ಥಿರಾಂಕ sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n>
ರಚನೆ ggml_sycl_unroll {
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Func, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ Func & f, ವಾದಗಳು... ವಾದಗಳು_2) ಸ್ಥಿರಾಂಕ {
        f(n - 1, ವಾದಗಳು_2...);
        ggml_sycl_unroll<n - 1>{}(f, ವಾದಗಳು_2...);
    }
};

ಮಾದರಿ <>
ರಚನೆ ggml_sycl_unroll<1> {
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Func, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು>
    ಶೂನ್ಯ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ Func & f, ವಾದಗಳು... ವಾದಗಳು_2) ಸ್ಥಿರಾಂಕ {
        f(0, ವಾದಗಳು_2...);
    }
};

ಸ್ಥಿರ __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ಸ್ಥಿರಾಂಕ sycl::half2 a, ಸ್ಥಿರಾಂಕ sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ದಶಮಾಂಶ, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ದಶಮಾಂಶ, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
    ಹಿಂತಿರುಗಿ ret;
}

ಸ್ಥಿರ __dpct_inline__ sycl::half ggml_sycl_hmax(ಸ್ಥಿರಾಂಕ sycl::half a, ಸ್ಥಿರಾಂಕ sycl::half b) {
    ಹಿಂತಿರುಗಿ sycl::vec<ದಶಮಾಂಶ, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
}

ಸ್ಥಿರ __dpct_inline__ uint32_t __hgt2_mask(ಸ್ಥಿರಾಂಕ sycl::half2 a, ಸ್ಥಿರಾಂಕ sycl::half2 b) {
    ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ_ಕೆಳಗಿನ  = 0x0000FFFF * (ದಶಮಾಂಶ(a[0]) > ದಶಮಾಂಶ(b[0]));
    ಸ್ಥಿರಾಂಕ uint32_t ಮುಖವಾಡ_ಮೇಲಿನ = 0xFFFF0000 * (ದಶಮಾಂಶ(a[1]) > ದಶಮಾಂಶ(b[1]));
    ಹಿಂತಿರುಗಿ ಮುಖವಾಡ_ಕೆಳಗಿನ | ಮುಖವಾಡ_ಮೇಲಿನ;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t fastmodulo(uint32_t n, ಸ್ಥಿರಾಂಕ sycl::uint3 fastdiv_ಮೌಲ್ಯಗಳು) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    ಹಿಂತಿರುಗಿ n - fastdiv(n, fastdiv_ಮೌಲ್ಯಗಳು) * fastdiv_ಮೌಲ್ಯಗಳು.z();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ fast_fp16_ಲಭ್ಯ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ cc) {
    GGML_ಬಳಸದ(cc);
    ಹಿಂತಿರುಗಿ ನಿಜ;   //Intel GPUs always support FP16.
}

ಎಣಿಕೆ ವರ್ಗ ಖಂಡ_reduce_ವಿಧಾನ {
    MAX,
    ಮೊತ್ತ_3,
};

ಮಾದರಿ<ಖಂಡ_reduce_ವಿಧಾನ ವಿಧಾನ_t, ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ>
ರಚನೆ ಖಂಡ_reduce_policy;

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು... Ts>
ಸಾಲೊಳಗೆ constexpr ತಾರ್ಕಿಕ is_any = (std::is_same_v<T, Ts> || ...);

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು...>
ಸಾಲೊಳಗೆ constexpr ತಾರ್ಕಿಕ ggml_sycl_dependent_false_v = ಸುಳ್ಳು;

#ವ್ಯಾಖ್ಯಾನಿಸು WARP_32_ಗಾತ್ರ 32

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ> ರಚನೆ ಖಂಡ_reduce_policy<ಖಂಡ_reduce_ವಿಧಾನ::ಮೊತ್ತ_3, T, warp_ಗಾತ್ರ> {
    ಸ್ಥಿರ T reduce(T val) {
        ಆದರೆ constexpr (is_any<T, ದಶಮಾಂಶ, sycl::float2, sycl::half2, ಪೂರ್ಣಾಂಕ>) {
            ಹಿಂತಿರುಗಿ warp_reduce_ಮೊತ್ತ<warp_ಗಾತ್ರ>(val);
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ಸ್ಥಿರ T sentinel() {
        ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
            ಹಿಂತಿರುಗಿ 0.0f;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, sycl::float2>) {
            ಹಿಂತಿರುಗಿ sycl::float2(0.0f, 0.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, sycl::half2>) {
            ಹಿಂತಿರುಗಿ sycl::half2(0.0f, 0.0f);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, ಪೂರ್ಣಾಂಕ>) {
            ಹಿಂತಿರುಗಿ 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ> ರಚನೆ ಖಂಡ_reduce_policy<ಖಂಡ_reduce_ವಿಧಾನ::MAX, T, warp_ಗಾತ್ರ> {
    ಸ್ಥಿರ T reduce(T val) {
        ಆದರೆ constexpr (is_any<T, ದಶಮಾಂಶ, sycl::half2>) {
            ಹಿಂತಿರುಗಿ warp_reduce_max<warp_ಗಾತ್ರ>(val);
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ಸ್ಥಿರ T sentinel() {
        ಆದರೆ constexpr (std::is_same_v<T, ದಶಮಾಂಶ>) {
            ಹಿಂತಿರುಗಿ -INFINITY;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_same_v<T, sycl::half2>) {
            ಹಿಂತಿರುಗಿ sycl::half2(-INFINITY, -INFINITY);
        } ಇಲ್ಲದಿದ್ದರೆ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


ಮಾದರಿ <ಖಂಡ_reduce_ವಿಧಾನ reduce_ವಿಧಾನ_t, ಪೂರ್ಣಾಂಕ warp_ಗಾತ್ರ, ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ T ಖಂಡ_reduce(T val, T * shared_vals, ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_template) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1                 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    val                           = ಖಂಡ_reduce_policy<reduce_ವಿಧಾನ_t, T,warp_ಗಾತ್ರ>::reduce(val);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = ಖಂಡ_ಗಾತ್ರ_template == 0 ? ಅಂಶ_ct1.get_local_range(2) : ಖಂಡ_ಗಾತ್ರ_template;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nthreads = ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nwarps = nthreads / WARP_ಗಾತ್ರ;

    ಆದರೆ (ಖಂಡ_ಗಾತ್ರ_3 > warp_ಗಾತ್ರ) {
        ಪ್ರತಿಜ್ಞಾ((ಖಂಡ_ಗಾತ್ರ_3 <= 1024) && (ಖಂಡ_ಗಾತ್ರ_3 % warp_ಗಾತ್ರ) == 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ warp_id = ಅಂಶ_ct1.get_local_id(2) / warp_ಗಾತ್ರ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lane_id = ಅಂಶ_ct1.get_local_id(2) % warp_ಗಾತ್ರ;
        ಆದರೆ (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_ಗಾತ್ರ;
        ದಶಮಾಂಶ tmp = 0.f;
        ಆದರೆ (lane_id < (static_cast<ಪೂರ್ಣಾಂಕ>(ಖಂಡ_ಗಾತ್ರ_3) / warp_ಗಾತ್ರ)) {
            ಗಾಗಿ (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_ಗಾತ್ರ];
            }
        }
        ಹಿಂತಿರುಗಿ ಖಂಡ_reduce_policy<reduce_ವಿಧಾನ_t, T, warp_ಗಾತ್ರ>::reduce(tmp);
    }
    ಹಿಂತಿರುಗಿ val;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ ggml_sycl_ue4m3_ಗಮ್ಯ_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    ಆದರೆ (x == 0 || x == 0x7F) {
        ಹಿಂತಿರುಗಿ 0.0f;
    }
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಘಾತಾಂಕ = (x >> 3) & 0xF;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ man = x & 0x7;
    ದಶಮಾಂಶ raw;
    ಆದರೆ (ಘಾತಾಂಕ == 0) {
        raw = man * (1.0f / 8.0f) * sycl::ಘಾತ(2.0f, -6.0f);
    } ಇಲ್ಲದಿದ್ದರೆ {
        raw = (1.0f + man / 8.0f) * sycl::ಘಾತ(2.0f, (ದಶಮಾಂಶ) ಘಾತಾಂಕ - 7.0f);
    }
    ಹಿಂತಿರುಗಿ raw * 0.5f;
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_COMMON_HPP
