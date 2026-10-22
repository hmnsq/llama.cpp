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

#include <cstddef>
#include <fstream>
#include <iostream>
#include <string>

#include "base.hpp"
#include "dpct/helper.hpp"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "presets.hpp"
#include "type.hpp"
#include "sycl_hw.hpp"
#include "memtrace.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

#නම් defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#නම් defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#නම් GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"
#pragma clang diagnostic pop
#include "ggml-impl.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);


extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_debug;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_enable_optimize;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_enable_fusion;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_enable_esimd;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_mmvq_wide;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
ga_nanaa ggml_sycl_xmx_gather_type {
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
static constexpr පූර්ණ_සංඛ්‍යා GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
static constexpr පූර්ණ_සංඛ්‍යා GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
ga_nanaa ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
static constexpr පූර්ණ_සංඛ්‍යා GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#නැතහොත්
static constexpr පූර්ණ_සංඛ්‍යා GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
static anta_hstha int32_t ggml_sycl_src1_prec(const ggml_tensor * dst) {
    ආපසු g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
static anta_hstha bool ggml_sycl_src1_f16_ok(const ggml_tensor * dst) {
    const int32_t src1_prec = ggml_sycl_src1_prec(dst);
    ආපසු g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_enable_flash_attention;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_dev2dev_memcpy;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_fa_onednn;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_fa_onednn_max_kv;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_enable_mkl_fa;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_memtrace;
extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      ආපසු dpct::success;                                              \
    } gr_haa_na (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      ආපසු dpct::default_error;                                        \
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
#නම් !defined(GGML_SYCL_FORCE_MMQ)
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

typedef sycl::queue *queue_ptr;

ga_nanaa ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

ga_nanaa ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(parimaa_na(sycl::half) == parimaa_na(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  පූර්ණ_සංඛ්‍යා* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const var_na* stmt,
    const var_na* func,
    const var_na* file,
    const පූර්ණ_සංඛ්‍යා line,
    const var_na* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        නම් (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } අතරතුර (0)

#නම් DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#නැතහොත්
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#නැතහොත්
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

static පූර්ණ_සංඛ්‍යා g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static anta_hstha void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

පූර්ණ_සංඛ්‍යා get_current_device_id();

anta_hstha පූර්ණ_සංඛ්‍යා ggml_sycl_get_device() {
    ආපසු get_current_device_id();
}

anta_hstha dpct::err0 ggml_sycl_set_device(const පූර්ණ_සංඛ්‍යා device) try {
  පූර්ණ_සංඛ්‍යා current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  නම් (device == current_device_id) {
    ආපසු 0;
  }

  ආපසු CHECK_TRY_ERROR(dpct::select_device(device));
} gr_haa_na (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
sa_nracanaa optimize_feature {
    bool reorder=false;
    bool onednn_optimized_gemm=false;
};

sa_nracanaa sycl_device_info {
    පූර්ණ_සංඛ්‍යා cc;  // compute capability
    පූර්ණ_සංඛ්‍යා nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    පූර්ණ_සංඛ්‍යා warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    පූර්ණ_සංඛ්‍යා max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    bool    vmm;                // virtual memory support
    bool    l0_device_type_valid;
    bool    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    bool    usm_system_support; // support for USM system allocations
};


sa_nracanaa ggml_sycl_device_info {
    පූර්ණ_සංඛ්‍යා device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    පූර්ණ_සංඛ්‍යා max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    bool ext_oneapi_level_zero = true; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

const ggml_sycl_device_info & ggml_sycl_info();

static constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

sa_nracanaa ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
sa_nracanaa ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        නම් (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        නම් (ptr)
            pool->free(ptr, actual_size);
        ptr = (T *) pool->alloc(size * parimaa_na(T), &this->actual_size);
        ආපසු ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * parimaa_na(T), &this->actual_size);
        ආපසු ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        ආපසු alloc(size);
    }

    T * get() {
        ආපසු ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

sa_nracanaa ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

extern පූර්ණ_සංඛ්‍යා g_ggml_sycl_use_level_zero_api;
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
void ggml_sycl_free_device(void *ptr, sycl::queue &q);

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

sa_nracanaa mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

sa_nracanaa ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

namespace sycl_ex = sycl::ext::oneapi::experimental;
sa_nracanaa ggml_backend_sycl_context {
    පූර්ණ_සංඛ්‍යා device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(පූර්ණ_සංඛ්‍යා device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(පූර්ණ_සංඛ්‍යා device, පූර්ණ_සංඛ්‍යා stream) {
        නම් (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        ආපසු qptrs[device][stream];
    }

    queue_ptr stream() {
        ආපසු stream(device, 0);
    }

#නම් GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        ආපසු eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(පූර්ණ_සංඛ්‍යා device, පූර්ණ_සංඛ්‍යා _stream) {
        auto q = stream(device, _stream);
        ආපසු stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        නම් (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            ආපසු eng;
        }
        නැතහොත්
        {
            ආපසු it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        නම් (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            ආපසු stream;
        }
        නැතහොත්
        {
            ආපසු it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        ආපසු stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, පූර්ණ_සංඛ්‍යා device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, පූර්ණ_සංඛ්‍යා device);

    ggml_sycl_pool & pool(පූර්ණ_සංඛ්‍යා device) {
        නම් (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        ආපසු *pools[device];
    }

    ggml_sycl_pool & pool() {
        ආපසු pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(පූර්ණ_සංඛ්‍යා device) {
        නම් (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        ආපසු *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { ආපසු host_pool(device); }
};

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    සඳහා (පූර්ණ_සංඛ්‍යා mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    ආපසු x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    සඳහා (පූර්ණ_සංඛ්‍යා mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    ආපසු a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ පූර්ණ_සංඛ්‍යා warp_reduce_sum(පූර්ණ_සංඛ්‍යා x) {
  ආපසු sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ float warp_reduce_sum(float x) {
#pragma unroll
  සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  ආපසු x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
  සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  ආපසු x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  ආපසු a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  ආපසු a;
}

static constexpr පූර්ණ_සංඛ්‍යා ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  ආපසු WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ පූර්ණ_සංඛ්‍යා warp_reduce_all(පූර්ණ_සංඛ්‍යා x) {
    නම් (width == ggml_sycl_get_physical_warp_size()) {
        ආපසු sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } නැතහොත් {
#pragma unroll
        සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        ආපසු x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ පූර්ණ_සංඛ්‍යා warp_reduce_any(පූර්ණ_සංඛ්‍යා x) {
    නම් (width == ggml_sycl_get_physical_warp_size()) {
        ආපසු sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } නැතහොත් {
#pragma unroll
        සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        ආපසු x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <පූර්ණ_සංඛ්‍යා width>
static __dpct_inline__ float warp_reduce_max(float x) {
#pragma unroll
  සඳහා (පූර්ණ_සංඛ්‍යා offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  ආපසු x;
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    සඳහා (පූර්ණ_සංඛ්‍යා mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    ආපසු x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<පූර්ණ_සංඛ්‍යා N>
__dpct_inline__ size_t calculate_offset(const std::array<පූර්ණ_සංඛ්‍යා, N> & strides, const std::array<පූර්ණ_සංඛ්‍යා, N> & indices) {
    size_t offset = 0;
#pragma unroll
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    ආපසු offset;
}

// Helper for vec loading aligned data
template <typename Tp, පූර්ණ_සංඛ්‍යා n>
anta_hstha sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    ආපසු *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, පූර්ණ_සංඛ්‍යා dim>
static __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    ආපසු acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    ආපසු (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);

#නම් GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
anta_hstha bool ggml_sycl_dnnl_has_optimized_gemm(පූර්ණ_සංඛ්‍යා device) {
    ආපසු ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

පූර්ණ_සංඛ්‍යා ggml_sycl_get_env(const var_na *env_name, පූර්ණ_සංඛ්‍යා default_val);

template <පූර්ණ_සංඛ්‍යා N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {
    නම් (LIKELY(!g_ggml_sycl_debug)) {
        ආපසු "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    සඳහා (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    නම් constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    ආපසු ss.str();
}

anta_hstha std::string debug_get_tensor_str(const std::string &prefix,
        const ggml_tensor *tensor, const std::string &suffix = "") {
    std::stringstream ss;
    නම් (LIKELY(!g_ggml_sycl_debug)) { ආපසු ss.str(); }
    ss << prefix.c_str() << "=";
    නම් (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        නම් (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        නම් (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } නැතහොත් {
        ss << "nullptr";
    }
    ss << suffix;
    ආපසු ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
sa_nracanaa scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func, const std::string_view & func_suffix, const ggml_tensor * dst,
                         std::size_t num_src, const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        නම් (LIKELY(!g_ggml_sycl_debug)) {
            ආපසු;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        නම් (dst) {
            සඳහා (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func, const ggml_tensor * dst, std::size_t num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    නම් (max_bias <= 0.0f) {
        ආපසු 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const පූර්ණ_සංඛ්‍යා   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    ආපසු dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    අතරතුර (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    ආපසු sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
static constexpr පූර්ණ_සංඛ්‍යා ggml_sycl_get_max_cpy_bytes() {
    ආපසු 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <පූර්ණ_සංඛ්‍යා nbytes, පූර්ණ_සංඛ්‍යා alignment = 0>
static __dpct_inline__ void ggml_sycl_memcpy_1(void * dst, const void * src) {
    නම් constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr පූර්ණ_සංඛ්‍යා nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    සඳහා (පූර්ණ_සංඛ්‍යා i = 0; i < nbytes/nb_per_cpy; ++i) {
        නම් constexpr (nb_per_cpy == 1) {
            ((var_na *) dst)[i] = ((const var_na *) src)[i];
        } නැතහොත් නම් constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } නැතහොත් නම් constexpr (nb_per_cpy == 4) {
            ((පූර්ණ_සංඛ්‍යා *) dst)[i] = ((const පූර්ණ_සංඛ්‍යා *) src)[i];
        } නැතහොත් නම් constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((const sycl::int2 *) src)[i];
        } නැතහොත් නම් constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((const sycl::int4 *) src)[i];
        } නැතහොත් {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    ආපසු res;
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    ආපසු (hi + n) >> fastdiv_values.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<float>(x),static_cast<float>(y));
    ආපසු res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    ආපසු float2_value;
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    ආපසු sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ පූර්ණ_සංඛ්‍යා ggml_sycl_dp4a(const පූර්ණ_සංඛ්‍යා a, const පූර්ණ_සංඛ්‍යා b, පූර්ණ_සංඛ්‍යා c) {
    ආපසු dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    නම් (x == 0) {
        bits = 0x00400000;
    } නැතහොත් {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, parimaa_na(float));
    ආපසු result;
}

sycl::float2 __dpct_inline__ __half22float2(const sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    ආපසු float2_value;
}

float __dpct_inline__ __half2float(sycl::half H) {
    ආපසු static_cast<float>(H);
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::float2 v, const sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    const sycl::float2 tmp = (v * u).template convert<float, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#නැතහොත්
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void ggml_sycl_mad(sycl::half2 & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#නැතහොත්
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <පූර්ණ_සංඛ්‍යා n>
sa_nracanaa ggml_sycl_unroll {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
sa_nracanaa ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

static __dpct_inline__ sycl::half2 ggml_sycl_hmax2(const sycl::half2 a, const sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<float, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<float, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    ආපසු ret;
}

static __dpct_inline__ sycl::half ggml_sycl_hmax(const sycl::half a, const sycl::half b) {
    ආපසු sycl::vec<float, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<float, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<float, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static __dpct_inline__ uint32_t __hgt2_mask(const sycl::half2 a, const sycl::half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float(a[0]) > float(b[0]));
    const uint32_t mask_high = 0xFFFF0000 * (float(a[1]) > float(b[1]));
    ආපසු mask_low | mask_high;
}

static __dpct_inline__ uint32_t fastmodulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    ආපසු n - fastdiv(n, fastdiv_values) * fastdiv_values.z();
}

static bool fast_fp16_available(const පූර්ණ_සංඛ්‍යා cc) {
    GGML_UNUSED(cc);
    ආපසු true;   //Intel GPUs always support FP16.
}

ga_nanaa class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, පූර්ණ_සංඛ්‍යා warp_size>
sa_nracanaa block_reduce_policy;

template <typename T, typename... Ts>
anta_hstha constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
anta_hstha constexpr bool ggml_sycl_dependent_false_v = false;

#define WARP_32_SIZE 32

template <typename T, පූර්ණ_සංඛ්‍යා warp_size> sa_nracanaa block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    static T reduce(T val) {
        නම් constexpr (is_any<T, float, sycl::float2, sycl::half2, පූර්ණ_සංඛ්‍යා>) {
            ආපසු warp_reduce_sum<warp_size>(val);
        } නැතහොත් {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static T sentinel() {
        නම් constexpr (std::is_same_v<T, float>) {
            ආපසු 0.0f;
        } නැතහොත් නම් constexpr (std::is_same_v<T, sycl::float2>) {
            ආපසු sycl::float2(0.0f, 0.0f);
        } නැතහොත් නම් constexpr (std::is_same_v<T, sycl::half2>) {
            ආපසු sycl::half2(0.0f, 0.0f);
        } නැතහොත් නම් constexpr (std::is_same_v<T, පූර්ණ_සංඛ්‍යා>) {
            ආපසු 0;
        } නැතහොත් {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, පූර්ණ_සංඛ්‍යා warp_size> sa_nracanaa block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    static T reduce(T val) {
        නම් constexpr (is_any<T, float, sycl::half2>) {
            ආපසු warp_reduce_max<warp_size>(val);
        } නැතහොත් {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static T sentinel() {
        නම් constexpr (std::is_same_v<T, float>) {
            ආපසු -INFINITY;
        } නැතහොත් නම් constexpr (std::is_same_v<T, sycl::half2>) {
            ආපසු sycl::half2(-INFINITY, -INFINITY);
        } නැතහොත් {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, පූර්ණ_සංඛ්‍යා warp_size, typename T>
static T block_reduce(T val, T * shared_vals, පූර්ණ_සංඛ්‍යා block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    const පූර්ණ_සංඛ්‍යා block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    const පූර්ණ_සංඛ්‍යා nthreads = item_ct1.get_local_range(2);
    const පූර්ණ_සංඛ්‍යා nwarps = nthreads / WARP_SIZE;

    නම් (block_size > warp_size) {
        assert((block_size <= 1024) && (block_size % warp_size) == 0);
        const පූර්ණ_සංඛ්‍යා warp_id = item_ct1.get_local_id(2) / warp_size;
        const පූර්ණ_සංඛ්‍යා lane_id = item_ct1.get_local_id(2) % warp_size;
        නම් (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        float tmp = 0.f;
        නම් (lane_id < (static_cast<පූර්ණ_සංඛ්‍යා>(block_size) / warp_size)) {
            සඳහා (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        ආපසු block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    ආපසු val;
}

static __dpct_inline__ float ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    නම් (x == 0 || x == 0x7F) {
        ආපසු 0.0f;
    }
    const පූර්ණ_සංඛ්‍යා exp = (x >> 3) & 0xF;
    const පූර්ණ_සංඛ්‍යා man = x & 0x7;
    float raw;
    නම් (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } නැතහොත් {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (float) exp - 7.0f);
    }
    ආපසු raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
