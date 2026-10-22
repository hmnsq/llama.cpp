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

#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <string>

#അന്തര്ഭാവ "base.hpp"
#അന്തര്ഭാവ "dpct/helper.hpp"
#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-sycl.h"
#അന്തര്ഭാവ "presets.hpp"
#അന്തര്ഭാവ "type.hpp"
#അന്തര്ഭാവ "sycl_hw.hpp"
#അന്തര്ഭാവ "memtrace.hpp"

നാമാകാശ syclexp = sycl::ext::oneapi::experimental;

#എങ്കിൽ defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #അന്തര്ഭാവ <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#എങ്കിൽ defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#എങ്കിൽ GGML_SYCL_DNNL
#അന്തര്ഭാവ "dnnl.hpp"
#അന്തര്ഭാവ "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic തള്ളുക
#pragma clang diagnostic ignored "-Wnested-anon-types"
#അന്തര്ഭാവ "ggml-common.h"
#pragma clang diagnostic എടുക്കുക
#അന്തര്ഭാവ "ggml-impl.h"

ശൂന്യം* ggml_sycl_host_malloc(size_t size);
ശൂന്യം ggml_sycl_host_free(ശൂന്യം* ptr);


ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_debug;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_optimize;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_fusion;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_esimd;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_mmvq_wide;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
എണ്ണൽ ggml_sycl_xmx_gather_type {
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
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
എണ്ണൽ ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#അല്ലെങ്കിൽ
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
സ്ഥിതം വരിയിൽ int32_t ggml_sycl_src1_prec(സ്ഥിരം ggml_tensor * dst) {
    തിരികെ g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
സ്ഥിതം വരിയിൽ തർക്കം ggml_sycl_src1_f16_ok(സ്ഥിരം ggml_tensor * dst) {
    സ്ഥിരം int32_t src1_prec = ggml_sycl_src1_prec(dst);
    തിരികെ g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_flash_attention;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_dev2dev_memcpy;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_fa_onednn;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_fa_onednn_max_kv;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_mkl_fa;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_memtrace;
ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(വ്യഞ്ജകം)                                            \
  [&]() {                                                                \
    ശ്രമിക്കുക {                                                                \
      വ്യഞ്ജകം;                                                              \
      തിരികെ dpct::success;                                              \
    } പിടിക്കുക (std::exception സ്ഥിരം& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      തിരികെ dpct::default_error;                                        \
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
#എങ്കിൽ !defined(GGML_SYCL_FORCE_MMQ)
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

തരനിർവചനം sycl::queue *queue_ptr;

എണ്ണൽ ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

എണ്ണൽ ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(വലുപ്പം(sycl::half) == വലുപ്പം(ggml_fp16_t), "wrong fp16 size");

സ്ഥിതം ശൂന്യം crash() {
  പൂർണ്ണസംഖ്യ* ptr = ശൂന്യ;
  *ptr = 0;
}

[[noreturn]] സ്ഥിതം ശൂന്യം ggml_sycl_error(
    സ്ഥിരം അക്ഷരം* stmt,
    സ്ഥിരം അക്ഷരം* func,
    സ്ഥിരം അക്ഷരം* file,
    സ്ഥിരം പൂർണ്ണസംഖ്യ വരി,
    സ്ഥിരം അക്ഷരം* സന്ദേശം) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, സന്ദേശം);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, വരി);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    ചെയ്യുക {                                                                                                   \
        auto err_ = (err);                                                                                 \
        എങ്കിൽ (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } ആകുമ്പോൾ (0)

#എങ്കിൽ DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#അല്ലെങ്കിൽ
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
തരനിർവചനം sycl::half dfloat; // dequantize float
തരനിർവചനം sycl::half2 dfloat2;
#അല്ലെങ്കിൽ
തരനിർവചനം ദശാംശം dfloat; // dequantize float
തരനിർവചനം sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

സ്ഥിതം പൂർണ്ണസംഖ്യ g_all_sycl_device_എണ്ണം = -1;
സ്ഥിതം തർക്കം g_ggml_backend_sycl_buffer_type_initialized = അസത്യം;

സ്ഥിതം ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

സ്ഥിതം ശൂന്യം* g_scratch_buffer = nullptr;
സ്ഥിതം size_t g_scratch_size = 0; // disabled by default
സ്ഥിതം size_t g_scratch_offset = 0;

[[noreturn]] സ്ഥിതം വരിയിൽ ശൂന്യം bad_arch(സ്ഥിരം sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (ശൂന്യം)bad_arch; // suppress unused function warning
}

പൂർണ്ണസംഖ്യ get_current_device_id();

വരിയിൽ പൂർണ്ണസംഖ്യ ggml_sycl_get_device() {
    തിരികെ get_current_device_id();
}

വരിയിൽ dpct::err0 ggml_sycl_set_device(സ്ഥിരം പൂർണ്ണസംഖ്യ device) ശ്രമിക്കുക {
  പൂർണ്ണസംഖ്യ current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  എങ്കിൽ (device == current_device_id) {
    തിരികെ 0;
  }

  തിരികെ CHECK_TRY_ERROR(dpct::select_device(device));
} പിടിക്കുക (sycl::exception സ്ഥിരം& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
ഘടന optimize_feature {
    തർക്കം reorder=അസത്യം;
    തർക്കം onednn_optimized_gemm=അസത്യം;
};

ഘടന sycl_device_info {
    പൂർണ്ണസംഖ്യ cc;  // compute capability
    പൂർണ്ണസംഖ്യ nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    പൂർണ്ണസംഖ്യ warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    പൂർണ്ണസംഖ്യ max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    തർക്കം    vmm;                // virtual memory support
    തർക്കം    l0_device_type_valid;
    തർക്കം    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    തർക്കം    usm_system_support; // support for USM system allocations
};


ഘടന ggml_sycl_device_info {
    പൂർണ്ണസംഖ്യ device_എണ്ണം;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    പൂർണ്ണസംഖ്യ max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    തർക്കം ext_oneapi_level_zero = സത്യം; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

സ്ഥിരം ggml_sycl_device_info & ggml_sycl_info();

സ്ഥിതം constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

ഘടന ggml_sycl_pool {
    മായാ ~ggml_sycl_pool() = സ്ഥിരസ്ഥിതി;

    മായാ ശൂന്യം * alloc(size_t size, size_t * actual_size) = 0;
    മായാ ശൂന്യം സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size) = 0;
};

template<typename T>
ഘടന ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        എങ്കിൽ (ptr != nullptr) {
            pool->സ്വതന്ത്രം(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        എങ്കിൽ (ptr)
            pool->സ്വതന്ത്രം(ptr, actual_size);
        ptr = (T *) pool->alloc(size * വലുപ്പം(T), &ഇത്->actual_size);
        തിരികെ ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * വലുപ്പം(T), &ഇത്->actual_size);
        തിരികെ ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ഇത്->pool = &pool;
        തിരികെ alloc(size);
    }

    T * get() {
        തിരികെ ptr;
    }

    ggml_sycl_pool_alloc() = സ്ഥിരസ്ഥിതി;
    ggml_sycl_pool_alloc(സ്ഥിരം ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(സ്ഥിരം ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

ഘടന ggml_tensor_extra_gpu {
  ശൂന്യം* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

ബാഹ്യ പൂർണ്ണസംഖ്യ g_ggml_sycl_use_level_zero_api;
ശൂന്യം * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
ശൂന്യം ggml_sycl_free_device(ശൂന്യം *ptr, sycl::queue &q);

ശൂന്യം release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

ഘടന mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

ഘടന ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

നാമാകാശ sycl_ex = sycl::ext::oneapi::experimental;
ഘടന ggml_backend_sycl_context {
    പൂർണ്ണസംഖ്യ device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(പൂർണ്ണസംഖ്യ device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(പൂർണ്ണസംഖ്യ device, പൂർണ്ണസംഖ്യ stream) {
        എങ്കിൽ (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        തിരികെ qptrs[device][stream];
    }

    queue_ptr stream() {
        തിരികെ stream(device, 0);
    }

#എങ്കിൽ GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        സ്ഥിരം dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        തിരികെ eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(പൂർണ്ണസംഖ്യ device, പൂർണ്ണസംഖ്യ _stream) {
        auto q = stream(device, _stream);
        തിരികെ stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        എങ്കിൽ (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            തിരികെ eng;
        }
        അല്ലെങ്കിൽ
        {
            തിരികെ it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        എങ്കിൽ (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            തിരികെ stream;
        }
        അല്ലെങ്കിൽ
        {
            തിരികെ it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        തിരികെ stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    സ്ഥിതം std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, പൂർണ്ണസംഖ്യ device);

    സ്ഥിതം std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, പൂർണ്ണസംഖ്യ device);

    ggml_sycl_pool & pool(പൂർണ്ണസംഖ്യ device) {
        എങ്കിൽ (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        തിരികെ *pools[device];
    }

    ggml_sycl_pool & pool() {
        തിരികെ pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(പൂർണ്ണസംഖ്യ device) {
        എങ്കിൽ (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        തിരികെ *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { തിരികെ host_pool(device); }
};

// common device functions

സ്ഥിതം __dpct_inline__ ദശാംശം warp_reduce_sum(ദശാംശം x,
    സ്ഥിരം sycl::nd_item<3>& item_ct1) {
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    തിരികെ x;
}

സ്ഥിതം __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, സ്ഥിരം sycl::nd_item<3>& item_ct1) {
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    തിരികെ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ പൂർണ്ണസംഖ്യ warp_reduce_sum(പൂർണ്ണസംഖ്യ x) {
  തിരികെ sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ ദശാംശം warp_reduce_sum(ദശാംശം x) {
#pragma unroll
  വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  തിരികെ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ ദശാംശം warp_reduce_sum(ദശാംശം x, സ്ഥിരം sycl::nd_item<3>& item_ct1) {
#pragma unroll
  വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  തിരികെ x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  തിരികെ a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  തിരികെ a;
}

സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  തിരികെ WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ പൂർണ്ണസംഖ്യ warp_reduce_all(പൂർണ്ണസംഖ്യ x) {
    എങ്കിൽ (width == ggml_sycl_get_physical_warp_size()) {
        തിരികെ sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } അല്ലെങ്കിൽ {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        തിരികെ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ പൂർണ്ണസംഖ്യ warp_reduce_any(പൂർണ്ണസംഖ്യ x) {
    എങ്കിൽ (width == ggml_sycl_get_physical_warp_size()) {
        തിരികെ sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } അല്ലെങ്കിൽ {
#pragma unroll
        വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        തിരികെ x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <പൂർണ്ണസംഖ്യ width>
സ്ഥിതം __dpct_inline__ ദശാംശം warp_reduce_max(ദശാംശം x) {
#pragma unroll
  വേണ്ടി (പൂർണ്ണസംഖ്യ offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  തിരികെ x;
}

സ്ഥിതം __dpct_inline__ ദശാംശം warp_reduce_max(ദശാംശം x,
    സ്ഥിരം sycl::nd_item<3>& item_ct1) {
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    തിരികെ x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<പൂർണ്ണസംഖ്യ N>
__dpct_inline__ size_t calculate_offset(സ്ഥിരം std::array<പൂർണ്ണസംഖ്യ, N> & strides, സ്ഥിരം std::array<പൂർണ്ണസംഖ്യ, N> & indices) {
    size_t offset = 0;
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    തിരികെ offset;
}

// Helper for vec loading aligned data
template <typename Tp, പൂർണ്ണസംഖ്യ n>
വരിയിൽ sycl::vec<Tp, n> vec_aligned_load(സ്ഥിരം Tp* aligned_ptr) {
    തിരികെ *reinterpret_cast<സ്ഥിരം sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, പൂർണ്ണസംഖ്യ dim>
സ്ഥിതം __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    തിരികെ acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_അക്കം, int64_t block_size);

constexpr size_t ceil_div(സ്ഥിരം size_t m, സ്ഥിരം size_t n) {
    തിരികെ (m + n - 1) / n;
}

തർക്കം gpu_has_xmx(sycl::device &dev);

#എങ്കിൽ GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
വരിയിൽ തർക്കം ggml_sycl_dnnl_has_optimized_gemm(പൂർണ്ണസംഖ്യ device) {
    തിരികെ ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

പൂർണ്ണസംഖ്യ ggml_sycl_get_env(സ്ഥിരം അക്ഷരം *env_name, പൂർണ്ണസംഖ്യ default_val);

template <പൂർണ്ണസംഖ്യ N, വർഗ്ഗം T> std::string debug_get_array_str(സ്ഥിരം std::string & prefix, സ്ഥിരം T array[N]) {
    എങ്കിൽ (LIKELY(!g_ggml_sycl_debug)) {
        തിരികെ "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    വേണ്ടി (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    എങ്കിൽ constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    തിരികെ ss.str();
}

വരിയിൽ std::string debug_get_tensor_str(സ്ഥിരം std::string &prefix,
        സ്ഥിരം ggml_tensor *tensor, സ്ഥിരം std::string &suffix = "") {
    std::stringstream ss;
    എങ്കിൽ (LIKELY(!g_ggml_sycl_debug)) { തിരികെ ss.str(); }
    ss << prefix.c_str() << "=";
    എങ്കിൽ (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        എങ്കിൽ (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        എങ്കിൽ (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } അല്ലെങ്കിൽ {
        ss << "nullptr";
    }
    ss << suffix;
    തിരികെ ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ഘടന scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(സ്ഥിരം std::string_view & func, സ്ഥിരം std::string_view & func_suffix, സ്ഥിരം ggml_tensor * dst,
                         std::size_t അക്കം_src, സ്ഥിരം std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        എങ്കിൽ (LIKELY(!g_ggml_sycl_debug)) {
            തിരികെ;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        എങ്കിൽ (dst) {
            വേണ്ടി (std::size_t i = 0; i < അക്കം_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(സ്ഥിരം std::string_view & func, സ്ഥിരം ggml_tensor * dst, std::size_t അക്കം_src,
                         സ്ഥിരം std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, അക്കം_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  സ്വകാര്യ:
    std::string_view func;
    std::string_view func_suffix;
};

സ്ഥിതം __dpct_inline__ ദശാംശം get_alibi_slope(സ്ഥിരം ദശാംശം    max_bias,
                                             സ്ഥിരം uint32_t h,
                                             സ്ഥിരം uint32_t n_head_log2,
                                             സ്ഥിരം ദശാംശം    m0,
                                             സ്ഥിരം ദശാംശം    m1) {
    എങ്കിൽ (max_bias <= 0.0f) {
        തിരികെ 1.0f;
    }
    സ്ഥിരം ദശാംശം base = h < n_head_log2 ? m0 : m1;
    സ്ഥിരം പൂർണ്ണസംഖ്യ   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    തിരികെ dpct::pow(base, exph);
}

സ്ഥിതം സ്ഥിരം sycl::uint3 init_fastdiv_മൂല്യങ്ങൾ(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    ആകുമ്പോൾ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    തിരികെ sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ ggml_sycl_get_max_cpy_bytes() {
    തിരികെ 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <പൂർണ്ണസംഖ്യ nbytes, പൂർണ്ണസംഖ്യ alignment = 0>
സ്ഥിതം __dpct_inline__ ശൂന്യം ggml_sycl_memcpy_1(ശൂന്യം * dst, സ്ഥിരം ശൂന്യം * src) {
    എങ്കിൽ constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr പൂർണ്ണസംഖ്യ nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < nbytes/nb_per_cpy; ++i) {
        എങ്കിൽ constexpr (nb_per_cpy == 1) {
            ((അക്ഷരം *) dst)[i] = ((സ്ഥിരം അക്ഷരം *) src)[i];
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((സ്ഥിരം short *) src)[i];
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (nb_per_cpy == 4) {
            ((പൂർണ്ണസംഖ്യ *) dst)[i] = ((സ്ഥിരം പൂർണ്ണസംഖ്യ *) src)[i];
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((സ്ഥിരം sycl::int2 *) src)[i];
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((സ്ഥിരം sycl::int4 *) src)[i];
        } അല്ലെങ്കിൽ {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    തിരികെ res;
}

സ്ഥിതം __dpct_inline__ uint32_t fastdiv(uint32_t n, സ്ഥിരം sycl::uint3 fastdiv_മൂല്യങ്ങൾ) {
    സ്ഥിരം uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_മൂല്യങ്ങൾ.x());
    തിരികെ (hi + n) >> fastdiv_മൂല്യങ്ങൾ.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ദശാംശം>(x),static_cast<ദശാംശം>(y));
    തിരികെ res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_മൂല്യം(static_cast<ദശാംശം>(H.x()), static_cast<ദശാംശം>(H.y()));
    തിരികെ float2_മൂല്യം;
}

സ്ഥിതം __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, സ്ഥിരം sycl::uint3 fastdiv_മൂല്യങ്ങൾ) {
    സ്ഥിരം uint32_t div_val = fastdiv(n, fastdiv_മൂല്യങ്ങൾ);
    സ്ഥിരം uint32_t mod_val = n - div_val * fastdiv_മൂല്യങ്ങൾ.z();
    തിരികെ sycl::uint2(div_val, mod_val);
}

സ്ഥിതം __dpct_inline__ പൂർണ്ണസംഖ്യ ggml_sycl_dp4a(സ്ഥിരം പൂർണ്ണസംഖ്യ a, സ്ഥിരം പൂർണ്ണസംഖ്യ b, പൂർണ്ണസംഖ്യ c) {
    തിരികെ dpct::dp4a(a, b, c);
}

സ്ഥിതം __dpct_inline__ ദശാംശം ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    എങ്കിൽ (x == 0) {
        bits = 0x00400000;
    } അല്ലെങ്കിൽ {
        bits = (uint32_t) x << 23;
    }

    ദശാംശം result;
    memcpy(&result, &bits, വലുപ്പം(ദശാംശം));
    തിരികെ result;
}

sycl::float2 __dpct_inline__ __half22float2(സ്ഥിരം sycl::half2 &H) {
    sycl::float2 float2_മൂല്യം(static_cast<ദശാംശം>(H.x()), static_cast<ദശാംശം>(H.y()));
    തിരികെ float2_മൂല്യം;
}

ദശാംശം __dpct_inline__ __half2float(sycl::half H) {
    തിരികെ static_cast<ദശാംശം>(H);
}

സ്ഥിതം __dpct_inline__ ശൂന്യം ggml_sycl_mad(ദശാംശം & acc, സ്ഥിരം ദശാംശം v, സ്ഥിരം ദശാംശം u) {
    acc += v*u;
}

സ്ഥിതം __dpct_inline__ ശൂന്യം ggml_sycl_mad(ദശാംശം & acc, സ്ഥിരം sycl::float2 v, സ്ഥിരം sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

സ്ഥിതം __dpct_inline__ ശൂന്യം ggml_sycl_mad(ദശാംശം & acc, സ്ഥിരം sycl::half2 v, സ്ഥിരം sycl::half2 u) {
#ifdef GGML_SYCL_F16
    സ്ഥിരം sycl::float2 tmp = (v * u).template convert<ദശാംശം, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#അല്ലെങ്കിൽ
    സ്ഥിരം sycl::float2 tmpv = __half22float2(v);
    സ്ഥിരം sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

സ്ഥിതം __dpct_inline__ ശൂന്യം ggml_sycl_mad(sycl::half2 & acc, സ്ഥിരം sycl::half2 v, സ്ഥിരം sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#അല്ലെങ്കിൽ
    സ്ഥിരം sycl::float2 tmpv = __half22float2(v);
    സ്ഥിരം sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <പൂർണ്ണസംഖ്യ n>
ഘടന ggml_sycl_unroll {
    template <typename Func, typename... Args>
    ശൂന്യം operator()(സ്ഥിരം Func & f, Args... args) സ്ഥിരം {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
ഘടന ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    ശൂന്യം operator()(സ്ഥിരം Func & f, Args... args) സ്ഥിരം {
        f(0, args...);
    }
};

സ്ഥിതം __dpct_inline__ sycl::half2 ggml_sycl_hmax2(സ്ഥിരം sycl::half2 a, സ്ഥിരം sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ദശാംശം, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ദശാംശം, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    തിരികെ ret;
}

സ്ഥിതം __dpct_inline__ sycl::half ggml_sycl_hmax(സ്ഥിരം sycl::half a, സ്ഥിരം sycl::half b) {
    തിരികെ sycl::vec<ദശാംശം, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ദശാംശം, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ദശാംശം, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

സ്ഥിതം __dpct_inline__ uint32_t __hgt2_mask(സ്ഥിരം sycl::half2 a, സ്ഥിരം sycl::half2 b) {
    സ്ഥിരം uint32_t mask_താഴെയുള്ള  = 0x0000FFFF * (ദശാംശം(a[0]) > ദശാംശം(b[0]));
    സ്ഥിരം uint32_t mask_മുകളിലെ = 0xFFFF0000 * (ദശാംശം(a[1]) > ദശാംശം(b[1]));
    തിരികെ mask_താഴെയുള്ള | mask_മുകളിലെ;
}

സ്ഥിതം __dpct_inline__ uint32_t fastmodulo(uint32_t n, സ്ഥിരം sycl::uint3 fastdiv_മൂല്യങ്ങൾ) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    തിരികെ n - fastdiv(n, fastdiv_മൂല്യങ്ങൾ) * fastdiv_മൂല്യങ്ങൾ.z();
}

സ്ഥിതം തർക്കം fast_fp16_available(സ്ഥിരം പൂർണ്ണസംഖ്യ cc) {
    GGML_UNUSED(cc);
    തിരികെ സത്യം;   //Intel GPUs always support FP16.
}

എണ്ണൽ വർഗ്ഗം block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, പൂർണ്ണസംഖ്യ warp_size>
ഘടന block_reduce_policy;

template <typename T, typename... Ts>
വരിയിൽ constexpr തർക്കം is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
വരിയിൽ constexpr തർക്കം ggml_sycl_dependent_false_v = അസത്യം;

#define WARP_32_SIZE 32

template <typename T, പൂർണ്ണസംഖ്യ warp_size> ഘടന block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    സ്ഥിതം T reduce(T val) {
        എങ്കിൽ constexpr (is_any<T, ദശാംശം, sycl::float2, sycl::half2, പൂർണ്ണസംഖ്യ>) {
            തിരികെ warp_reduce_sum<warp_size>(val);
        } അല്ലെങ്കിൽ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    സ്ഥിതം T sentinel() {
        എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
            തിരികെ 0.0f;
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, sycl::float2>) {
            തിരികെ sycl::float2(0.0f, 0.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, sycl::half2>) {
            തിരികെ sycl::half2(0.0f, 0.0f);
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, പൂർണ്ണസംഖ്യ>) {
            തിരികെ 0;
        } അല്ലെങ്കിൽ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, പൂർണ്ണസംഖ്യ warp_size> ഘടന block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    സ്ഥിതം T reduce(T val) {
        എങ്കിൽ constexpr (is_any<T, ദശാംശം, sycl::half2>) {
            തിരികെ warp_reduce_max<warp_size>(val);
        } അല്ലെങ്കിൽ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    സ്ഥിതം T sentinel() {
        എങ്കിൽ constexpr (std::is_same_v<T, ദശാംശം>) {
            തിരികെ -INFINITY;
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_same_v<T, sycl::half2>) {
            തിരികെ sycl::half2(-INFINITY, -INFINITY);
        } അല്ലെങ്കിൽ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, പൂർണ്ണസംഖ്യ warp_size, typename T>
സ്ഥിതം T block_reduce(T val, T * shared_vals, പൂർണ്ണസംഖ്യ block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nthreads = item_ct1.get_local_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ nwarps = nthreads / WARP_SIZE;

    എങ്കിൽ (block_size > warp_size) {
        പ്രതിജ്ഞാ((block_size <= 1024) && (block_size % warp_size) == 0);
        സ്ഥിരം പൂർണ്ണസംഖ്യ warp_id = item_ct1.get_local_id(2) / warp_size;
        സ്ഥിരം പൂർണ്ണസംഖ്യ lane_id = item_ct1.get_local_id(2) % warp_size;
        എങ്കിൽ (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        ദശാംശം tmp = 0.f;
        എങ്കിൽ (lane_id < (static_cast<പൂർണ്ണസംഖ്യ>(block_size) / warp_size)) {
            വേണ്ടി (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        തിരികെ block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    തിരികെ val;
}

സ്ഥിതം __dpct_inline__ ദശാംശം ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    എങ്കിൽ (x == 0 || x == 0x7F) {
        തിരികെ 0.0f;
    }
    സ്ഥിരം പൂർണ്ണസംഖ്യ exp = (x >> 3) & 0xF;
    സ്ഥിരം പൂർണ്ണസംഖ്യ man = x & 0x7;
    ദശാംശം raw;
    എങ്കിൽ (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } അല്ലെങ്കിൽ {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (ദശാംശം) exp - 7.0f);
    }
    തിരികെ raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
