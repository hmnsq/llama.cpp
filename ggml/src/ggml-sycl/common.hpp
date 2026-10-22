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

#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <string>

#અન્તર્ભાવ "base.hpp"
#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-sycl.h"
#અન્તર્ભાવ "presets.hpp"
#અન્તર્ભાવ "type.hpp"
#અન્તર્ભાવ "sycl_hw.hpp"
#અન્તર્ભાવ "memtrace.hpp"

નામાકાશ syclexp = sycl::ext::oneapi::experimental;

#જો defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #અન્તર્ભાવ <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#જો defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#જો GGML_SYCL_DNNL
#અન્તર્ભાવ "dnnl.hpp"
#અન્તર્ભાવ "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic ધકેલો
#pragma clang diagnostic ignored "-Wnested-anon-types"
#અન્તર્ભાવ "ggml-common.h"
#pragma clang diagnostic કાઢો
#અન્તર્ભાવ "ggml-impl.h"

શૂન્ય* ggml_sycl_host_malloc(size_t size);
શૂન્ય ggml_sycl_host_free(શૂન્ય* ptr);


બાહ્ય પૂર્ણાંક g_ggml_sycl_debug;
બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_optimize;
બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_fusion;
બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_esimd;
બાહ્ય પૂર્ણાંક g_ggml_sycl_mmvq_wide;
બાહ્ય પૂર્ણાંક g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ગણના ggml_sycl_xmx_gather_type {
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
સ્થિર constexpr પૂર્ણાંક GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
બાહ્ય પૂર્ણાંક g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
સ્થિર constexpr પૂર્ણાંક GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
બાહ્ય પૂર્ણાંક g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ગણના ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
સ્થિર constexpr પૂર્ણાંક GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#નહીંતર
સ્થિર constexpr પૂર્ણાંક GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
બાહ્ય પૂર્ણાંક g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
બાહ્ય પૂર્ણાંક g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
સ્થિર અંતઃરેખ int32_t ggml_sycl_src1_prec(અચલ ggml_tensor * dst) {
    પાછા g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
સ્થિર અંતઃરેખ તાર્કિક ggml_sycl_src1_f16_ok(અચલ ggml_tensor * dst) {
    અચલ int32_t src1_prec = ggml_sycl_src1_prec(dst);
    પાછા g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_flash_attention;
બાહ્ય પૂર્ણાંક g_ggml_sycl_dev2dev_memcpy;
બાહ્ય પૂર્ણાંક g_ggml_sycl_fa_onednn;
બાહ્ય પૂર્ણાંક g_ggml_sycl_fa_onednn_max_kv;
બાહ્ય પૂર્ણાંક g_ggml_sycl_enable_mkl_fa;
બાહ્ય પૂર્ણાંક g_ggml_sycl_memtrace;
બાહ્ય પૂર્ણાંક g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(પદાવલિ)                                            \
  [&]() {                                                                \
    પ્રયાસ {                                                                \
      પદાવલિ;                                                              \
      પાછા dpct::success;                                              \
    } પકડો (std::exception અચલ& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      પાછા dpct::default_error;                                        \
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
#જો !defined(GGML_SYCL_FORCE_MMQ)
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

પ્રકારવ્યાખ્યા sycl::queue *queue_ptr;

ગણના ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

ગણના ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(કદ(sycl::half) == કદ(ggml_fp16_t), "wrong fp16 size");

સ્થિર શૂન્ય crash() {
  પૂર્ણાંક* ptr = ખાલી;
  *ptr = 0;
}

[[noreturn]] સ્થિર શૂન્ય ggml_sycl_error(
    અચલ અક્ષર* stmt,
    અચલ અક્ષર* func,
    અચલ અક્ષર* file,
    અચલ પૂર્ણાંક લીટી,
    અચલ અક્ષર* સંદેશ) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, સંદેશ);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, લીટી);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    કરો {                                                                                                   \
        auto err_ = (err);                                                                                 \
        જો (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } જ્યારે (0)

#જો DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#નહીંતર
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
પ્રકારવ્યાખ્યા sycl::half dfloat; // dequantize float
પ્રકારવ્યાખ્યા sycl::half2 dfloat2;
#નહીંતર
પ્રકારવ્યાખ્યા દશાંશ dfloat; // dequantize float
પ્રકારવ્યાખ્યા sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

સ્થિર પૂર્ણાંક g_all_sycl_device_ગણતરી = -1;
સ્થિર તાર્કિક g_ggml_backend_sycl_buffer_type_initialized = ખોટું;

સ્થિર ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

સ્થિર શૂન્ય* g_scratch_buffer = nullptr;
સ્થિર size_t g_scratch_size = 0; // disabled by default
સ્થિર size_t g_scratch_offset = 0;

[[noreturn]] સ્થિર અંતઃરેખ શૂન્ય bad_arch(અચલ sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (શૂન્ય)bad_arch; // suppress unused function warning
}

પૂર્ણાંક get_current_device_id();

અંતઃરેખ પૂર્ણાંક ggml_sycl_get_device() {
    પાછા get_current_device_id();
}

અંતઃરેખ dpct::err0 ggml_sycl_set_device(અચલ પૂર્ણાંક device) પ્રયાસ {
  પૂર્ણાંક current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  જો (device == current_device_id) {
    પાછા 0;
  }

  પાછા CHECK_TRY_ERROR(dpct::select_device(device));
} પકડો (sycl::exception અચલ& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
સંરચના optimize_feature {
    તાર્કિક reorder=ખોટું;
    તાર્કિક onednn_optimized_gemm=ખોટું;
};

સંરચના sycl_device_info {
    પૂર્ણાંક cc;  // compute capability
    પૂર્ણાંક nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    પૂર્ણાંક warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    પૂર્ણાંક max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    તાર્કિક    vmm;                // virtual memory support
    તાર્કિક    l0_device_type_valid;
    તાર્કિક    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    તાર્કિક    usm_system_support; // support for USM system allocations
};


સંરચના ggml_sycl_device_info {
    પૂર્ણાંક device_ગણતરી;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    પૂર્ણાંક max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    તાર્કિક ext_oneapi_level_zero = સાચું; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

અચલ ggml_sycl_device_info & ggml_sycl_info();

સ્થિર constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

સંરચના ggml_sycl_pool {
    આભાસી ~ggml_sycl_pool() = મૂળભૂત;

    આભાસી શૂન્ય * alloc(size_t size, size_t * actual_size) = 0;
    આભાસી શૂન્ય મુક્ત(શૂન્ય * ptr, size_t size) = 0;
};

template<typename T>
સંરચના ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        જો (ptr != nullptr) {
            pool->મુક્ત(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        જો (ptr)
            pool->મુક્ત(ptr, actual_size);
        ptr = (T *) pool->alloc(size * કદ(T), &આ->actual_size);
        પાછા ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * કદ(T), &આ->actual_size);
        પાછા ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        આ->pool = &pool;
        પાછા alloc(size);
    }

    T * get() {
        પાછા ptr;
    }

    ggml_sycl_pool_alloc() = મૂળભૂત;
    ggml_sycl_pool_alloc(અચલ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(અચલ ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

સંરચના ggml_tensor_extra_gpu {
  શૂન્ય* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

બાહ્ય પૂર્ણાંક g_ggml_sycl_use_level_zero_api;
શૂન્ય * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
શૂન્ય ggml_sycl_free_device(શૂન્ય *ptr, sycl::queue &q);

શૂન્ય release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

સંરચના mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

સંરચના ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

નામાકાશ sycl_ex = sycl::ext::oneapi::experimental;
સંરચના ggml_backend_sycl_context {
    પૂર્ણાંક device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(પૂર્ણાંક device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(પૂર્ણાંક device, પૂર્ણાંક stream) {
        જો (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        પાછા qptrs[device][stream];
    }

    queue_ptr stream() {
        પાછા stream(device, 0);
    }

#જો GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        અચલ dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        પાછા eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(પૂર્ણાંક device, પૂર્ણાંક _stream) {
        auto q = stream(device, _stream);
        પાછા stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        જો (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            પાછા eng;
        }
        નહીંતર
        {
            પાછા it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        જો (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            પાછા stream;
        }
        નહીંતર
        {
            પાછા it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        પાછા stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    સ્થિર std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, પૂર્ણાંક device);

    સ્થિર std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, પૂર્ણાંક device);

    ggml_sycl_pool & pool(પૂર્ણાંક device) {
        જો (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        પાછા *pools[device];
    }

    ggml_sycl_pool & pool() {
        પાછા pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(પૂર્ણાંક device) {
        જો (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        પાછા *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { પાછા host_pool(device); }
};

// common device functions

સ્થિર __dpct_inline__ દશાંશ warp_reduce_sum(દશાંશ x,
    અચલ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    પાછા x;
}

સ્થિર __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, અચલ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    પાછા a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ પૂર્ણાંક warp_reduce_sum(પૂર્ણાંક x) {
  પાછા sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ દશાંશ warp_reduce_sum(દશાંશ x) {
#pragma unroll
  માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  પાછા x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ દશાંશ warp_reduce_sum(દશાંશ x, અચલ sycl::nd_item<3>& item_ct1) {
#pragma unroll
  માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  પાછા x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  પાછા a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  પાછા a;
}

સ્થિર constexpr પૂર્ણાંક ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  પાછા WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ પૂર્ણાંક warp_reduce_all(પૂર્ણાંક x) {
    જો (width == ggml_sycl_get_physical_warp_size()) {
        પાછા sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } નહીંતર {
#pragma unroll
        માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        પાછા x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ પૂર્ણાંક warp_reduce_any(પૂર્ણાંક x) {
    જો (width == ggml_sycl_get_physical_warp_size()) {
        પાછા sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } નહીંતર {
#pragma unroll
        માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        પાછા x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <પૂર્ણાંક width>
સ્થિર __dpct_inline__ દશાંશ warp_reduce_max(દશાંશ x) {
#pragma unroll
  માટે (પૂર્ણાંક offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  પાછા x;
}

સ્થિર __dpct_inline__ દશાંશ warp_reduce_max(દશાંશ x,
    અચલ sycl::nd_item<3>& item_ct1) {
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    પાછા x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<પૂર્ણાંક N>
__dpct_inline__ size_t calculate_offset(અચલ std::array<પૂર્ણાંક, N> & strides, અચલ std::array<પૂર્ણાંક, N> & indices) {
    size_t offset = 0;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    પાછા offset;
}

// Helper for vec loading aligned data
template <typename Tp, પૂર્ણાંક n>
અંતઃરેખ sycl::vec<Tp, n> vec_aligned_load(અચલ Tp* aligned_ptr) {
    પાછા *reinterpret_cast<અચલ sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, પૂર્ણાંક dim>
સ્થિર __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    પાછા acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_અંક, int64_t block_size);

constexpr size_t ceil_div(અચલ size_t m, અચલ size_t n) {
    પાછા (m + n - 1) / n;
}

તાર્કિક gpu_has_xmx(sycl::device &dev);

#જો GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
અંતઃરેખ તાર્કિક ggml_sycl_dnnl_has_optimized_gemm(પૂર્ણાંક device) {
    પાછા ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

પૂર્ણાંક ggml_sycl_get_env(અચલ અક્ષર *env_name, પૂર્ણાંક default_val);

template <પૂર્ણાંક N, વર્ગ T> std::string debug_get_array_str(અચલ std::string & prefix, અચલ T array[N]) {
    જો (LIKELY(!g_ggml_sycl_debug)) {
        પાછા "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    માટે (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    જો constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    પાછા ss.str();
}

અંતઃરેખ std::string debug_get_tensor_str(અચલ std::string &prefix,
        અચલ ggml_tensor *tensor, અચલ std::string &suffix = "") {
    std::stringstream ss;
    જો (LIKELY(!g_ggml_sycl_debug)) { પાછા ss.str(); }
    ss << prefix.c_str() << "=";
    જો (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        જો (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        જો (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } નહીંતર {
        ss << "nullptr";
    }
    ss << suffix;
    પાછા ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
સંરચના scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(અચલ std::string_view & func, અચલ std::string_view & func_suffix, અચલ ggml_tensor * dst,
                         std::size_t અંક_src, અચલ std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        જો (LIKELY(!g_ggml_sycl_debug)) {
            પાછા;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        જો (dst) {
            માટે (std::size_t i = 0; i < અંક_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(અચલ std::string_view & func, અચલ ggml_tensor * dst, std::size_t અંક_src,
                         અચલ std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, અંક_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  ખાનગી:
    std::string_view func;
    std::string_view func_suffix;
};

સ્થિર __dpct_inline__ દશાંશ get_alibi_slope(અચલ દશાંશ    max_bias,
                                             અચલ uint32_t h,
                                             અચલ uint32_t n_head_log2,
                                             અચલ દશાંશ    m0,
                                             અચલ દશાંશ    m1) {
    જો (max_bias <= 0.0f) {
        પાછા 1.0f;
    }
    અચલ દશાંશ base = h < n_head_log2 ? m0 : m1;
    અચલ પૂર્ણાંક   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    પાછા dpct::pow(base, exph);
}

સ્થિર અચલ sycl::uint3 init_fastdiv_કિંમતો(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    જ્યારે (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    પાછા sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
સ્થિર constexpr પૂર્ણાંક ggml_sycl_get_max_cpy_bytes() {
    પાછા 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <પૂર્ણાંક nbytes, પૂર્ણાંક alignment = 0>
સ્થિર __dpct_inline__ શૂન્ય ggml_sycl_memcpy_1(શૂન્ય * dst, અચલ શૂન્ય * src) {
    જો constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr પૂર્ણાંક nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < nbytes/nb_per_cpy; ++i) {
        જો constexpr (nb_per_cpy == 1) {
            ((અક્ષર *) dst)[i] = ((અચલ અક્ષર *) src)[i];
        } નહીંતર જો constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((અચલ short *) src)[i];
        } નહીંતર જો constexpr (nb_per_cpy == 4) {
            ((પૂર્ણાંક *) dst)[i] = ((અચલ પૂર્ણાંક *) src)[i];
        } નહીંતર જો constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((અચલ sycl::int2 *) src)[i];
        } નહીંતર જો constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((અચલ sycl::int4 *) src)[i];
        } નહીંતર {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    પાછા res;
}

સ્થિર __dpct_inline__ uint32_t fastdiv(uint32_t n, અચલ sycl::uint3 fastdiv_કિંમતો) {
    અચલ uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_કિંમતો.x());
    પાછા (hi + n) >> fastdiv_કિંમતો.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<દશાંશ>(x),static_cast<દશાંશ>(y));
    પાછા res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_કિંમત(static_cast<દશાંશ>(H.x()), static_cast<દશાંશ>(H.y()));
    પાછા float2_કિંમત;
}

સ્થિર __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, અચલ sycl::uint3 fastdiv_કિંમતો) {
    અચલ uint32_t div_val = fastdiv(n, fastdiv_કિંમતો);
    અચલ uint32_t mod_val = n - div_val * fastdiv_કિંમતો.z();
    પાછા sycl::uint2(div_val, mod_val);
}

સ્થિર __dpct_inline__ પૂર્ણાંક ggml_sycl_dp4a(અચલ પૂર્ણાંક a, અચલ પૂર્ણાંક b, પૂર્ણાંક c) {
    પાછા dpct::dp4a(a, b, c);
}

સ્થિર __dpct_inline__ દશાંશ ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    જો (x == 0) {
        bits = 0x00400000;
    } નહીંતર {
        bits = (uint32_t) x << 23;
    }

    દશાંશ result;
    memcpy(&result, &bits, કદ(દશાંશ));
    પાછા result;
}

sycl::float2 __dpct_inline__ __half22float2(અચલ sycl::half2 &H) {
    sycl::float2 float2_કિંમત(static_cast<દશાંશ>(H.x()), static_cast<દશાંશ>(H.y()));
    પાછા float2_કિંમત;
}

દશાંશ __dpct_inline__ __half2float(sycl::half H) {
    પાછા static_cast<દશાંશ>(H);
}

સ્થિર __dpct_inline__ શૂન્ય ggml_sycl_mad(દશાંશ & acc, અચલ દશાંશ v, અચલ દશાંશ u) {
    acc += v*u;
}

સ્થિર __dpct_inline__ શૂન્ય ggml_sycl_mad(દશાંશ & acc, અચલ sycl::float2 v, અચલ sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

સ્થિર __dpct_inline__ શૂન્ય ggml_sycl_mad(દશાંશ & acc, અચલ sycl::half2 v, અચલ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    અચલ sycl::float2 tmp = (v * u).template convert<દશાંશ, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#નહીંતર
    અચલ sycl::float2 tmpv = __half22float2(v);
    અચલ sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

સ્થિર __dpct_inline__ શૂન્ય ggml_sycl_mad(sycl::half2 & acc, અચલ sycl::half2 v, અચલ sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#નહીંતર
    અચલ sycl::float2 tmpv = __half22float2(v);
    અચલ sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <પૂર્ણાંક n>
સંરચના ggml_sycl_unroll {
    template <typename Func, typename... Args>
    શૂન્ય operator()(અચલ Func & f, Args... args) અચલ {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
સંરચના ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    શૂન્ય operator()(અચલ Func & f, Args... args) અચલ {
        f(0, args...);
    }
};

સ્થિર __dpct_inline__ sycl::half2 ggml_sycl_hmax2(અચલ sycl::half2 a, અચલ sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<દશાંશ, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<દશાંશ, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    પાછા ret;
}

સ્થિર __dpct_inline__ sycl::half ggml_sycl_hmax(અચલ sycl::half a, અચલ sycl::half b) {
    પાછા sycl::vec<દશાંશ, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<દશાંશ, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<દશાંશ, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

સ્થિર __dpct_inline__ uint32_t __hgt2_mask(અચલ sycl::half2 a, અચલ sycl::half2 b) {
    અચલ uint32_t mask_નીચલું  = 0x0000FFFF * (દશાંશ(a[0]) > દશાંશ(b[0]));
    અચલ uint32_t mask_ઉપલું = 0xFFFF0000 * (દશાંશ(a[1]) > દશાંશ(b[1]));
    પાછા mask_નીચલું | mask_ઉપલું;
}

સ્થિર __dpct_inline__ uint32_t fastmodulo(uint32_t n, અચલ sycl::uint3 fastdiv_કિંમતો) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    પાછા n - fastdiv(n, fastdiv_કિંમતો) * fastdiv_કિંમતો.z();
}

સ્થિર તાર્કિક fast_fp16_available(અચલ પૂર્ણાંક cc) {
    GGML_UNUSED(cc);
    પાછા સાચું;   //Intel GPUs always support FP16.
}

ગણના વર્ગ block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, પૂર્ણાંક warp_size>
સંરચના block_reduce_policy;

template <typename T, typename... Ts>
અંતઃરેખ constexpr તાર્કિક is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
અંતઃરેખ constexpr તાર્કિક ggml_sycl_dependent_false_v = ખોટું;

#define WARP_32_SIZE 32

template <typename T, પૂર્ણાંક warp_size> સંરચના block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    સ્થિર T reduce(T val) {
        જો constexpr (is_any<T, દશાંશ, sycl::float2, sycl::half2, પૂર્ણાંક>) {
            પાછા warp_reduce_sum<warp_size>(val);
        } નહીંતર {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    સ્થિર T sentinel() {
        જો constexpr (std::is_same_v<T, દશાંશ>) {
            પાછા 0.0f;
        } નહીંતર જો constexpr (std::is_same_v<T, sycl::float2>) {
            પાછા sycl::float2(0.0f, 0.0f);
        } નહીંતર જો constexpr (std::is_same_v<T, sycl::half2>) {
            પાછા sycl::half2(0.0f, 0.0f);
        } નહીંતર જો constexpr (std::is_same_v<T, પૂર્ણાંક>) {
            પાછા 0;
        } નહીંતર {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, પૂર્ણાંક warp_size> સંરચના block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    સ્થિર T reduce(T val) {
        જો constexpr (is_any<T, દશાંશ, sycl::half2>) {
            પાછા warp_reduce_max<warp_size>(val);
        } નહીંતર {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    સ્થિર T sentinel() {
        જો constexpr (std::is_same_v<T, દશાંશ>) {
            પાછા -INFINITY;
        } નહીંતર જો constexpr (std::is_same_v<T, sycl::half2>) {
            પાછા sycl::half2(-INFINITY, -INFINITY);
        } નહીંતર {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, પૂર્ણાંક warp_size, typename T>
સ્થિર T block_reduce(T val, T * shared_vals, પૂર્ણાંક block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    અચલ પૂર્ણાંક block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    અચલ પૂર્ણાંક nthreads = item_ct1.get_local_range(2);
    અચલ પૂર્ણાંક nwarps = nthreads / WARP_SIZE;

    જો (block_size > warp_size) {
        પ્રતિજ્ઞા((block_size <= 1024) && (block_size % warp_size) == 0);
        અચલ પૂર્ણાંક warp_id = item_ct1.get_local_id(2) / warp_size;
        અચલ પૂર્ણાંક lane_id = item_ct1.get_local_id(2) % warp_size;
        જો (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        દશાંશ tmp = 0.f;
        જો (lane_id < (static_cast<પૂર્ણાંક>(block_size) / warp_size)) {
            માટે (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        પાછા block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    પાછા val;
}

સ્થિર __dpct_inline__ દશાંશ ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    જો (x == 0 || x == 0x7F) {
        પાછા 0.0f;
    }
    અચલ પૂર્ણાંક exp = (x >> 3) & 0xF;
    અચલ પૂર્ણાંક man = x & 0x7;
    દશાંશ raw;
    જો (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } નહીંતર {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (દશાંશ) exp - 7.0f);
    }
    પાછા raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
