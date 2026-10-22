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

#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <fstream>
#அந்தர்பாவ <iostream>
#அந்தர்பாவ <string>

#அந்தர்பாவ "base.hpp"
#அந்தர்பாவ "dpct/helper.hpp"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ggml-sycl.h"
#அந்தர்பாவ "presets.hpp"
#அந்தர்பாவ "type.hpp"
#அந்தர்பாவ "sycl_hw.hpp"
#அந்தர்பாவ "memtrace.hpp"

நாமாகாஶ syclexp = sycl::ext::oneapi::experimental;

#எனில் defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #அந்தர்பாவ <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#எனில் defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#எனில் GGML_SYCL_DNNL
#அந்தர்பாவ "dnnl.hpp"
#அந்தர்பாவ "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic தள்ளு
#pragma clang diagnostic ignored "-Wnested-anon-types"
#அந்தர்பாவ "ggml-common.h"
#pragma clang diagnostic எடு
#அந்தர்பாவ "ggml-impl.h"

வெறுமை* ggml_sycl_host_malloc(size_t size);
வெறுமை ggml_sycl_host_free(வெறுமை* ptr);


வெளி முழு g_ggml_sycl_debug;
வெளி முழு g_ggml_sycl_enable_optimize;
வெளி முழு g_ggml_sycl_enable_fusion;
வெளி முழு g_ggml_sycl_enable_esimd;
வெளி முழு g_ggml_sycl_mmvq_wide;
வெளி முழு g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
எண்ணிக்கை ggml_sycl_xmx_gather_type {
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
நிலையான constexpr முழு GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
வெளி முழு g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
நிலையான constexpr முழு GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
வெளி முழு g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
எண்ணிக்கை ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
நிலையான constexpr முழு GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#இல்லையெனில்
நிலையான constexpr முழு GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
வெளி முழு g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
வெளி முழு g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
நிலையான உள்வரி int32_t ggml_sycl_src1_prec(மாறிலி ggml_tensor * dst) {
    திரும்பு g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
நிலையான உள்வரி தருக்க ggml_sycl_src1_f16_ok(மாறிலி ggml_tensor * dst) {
    மாறிலி int32_t src1_prec = ggml_sycl_src1_prec(dst);
    திரும்பு g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
வெளி முழு g_ggml_sycl_enable_flash_attention;
வெளி முழு g_ggml_sycl_dev2dev_memcpy;
வெளி முழு g_ggml_sycl_fa_onednn;
வெளி முழு g_ggml_sycl_fa_onednn_max_kv;
வெளி முழு g_ggml_sycl_enable_mkl_fa;
வெளி முழு g_ggml_sycl_memtrace;
வெளி முழு g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(கோவை)                                            \
  [&]() {                                                                \
    முயல் {                                                                \
      கோவை;                                                              \
      திரும்பு dpct::success;                                              \
    } பிடி (std::exception மாறிலி& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      திரும்பு dpct::default_error;                                        \
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
#எனில் !defined(GGML_SYCL_FORCE_MMQ)
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

வகைவரை sycl::queue *queue_ptr;

எண்ணிக்கை ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

எண்ணிக்கை ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(அளவு(sycl::half) == அளவு(ggml_fp16_t), "wrong fp16 size");

நிலையான வெறுமை crash() {
  முழு* ptr = வெற்று;
  *ptr = 0;
}

[[noreturn]] நிலையான வெறுமை ggml_sycl_error(
    மாறிலி எழுத்து* stmt,
    மாறிலி எழுத்து* func,
    மாறிலி எழுத்து* file,
    மாறிலி முழு வரி,
    மாறிலி எழுத்து* செய்தி) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, செய்தி);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, வரி);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    செய் {                                                                                                   \
        auto err_ = (err);                                                                                 \
        எனில் (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } போது (0)

#எனில் DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#இல்லையெனில்
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
வகைவரை sycl::half dfloat; // dequantize float
வகைவரை sycl::half2 dfloat2;
#இல்லையெனில்
வகைவரை தசம dfloat; // dequantize float
வகைவரை sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

நிலையான முழு g_all_sycl_device_மொத்தம் = -1;
நிலையான தருக்க g_ggml_backend_sycl_buffer_type_initialized = பொய்;

நிலையான ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

நிலையான வெறுமை* g_scratch_buffer = nullptr;
நிலையான size_t g_scratch_size = 0; // disabled by default
நிலையான size_t g_scratch_offset = 0;

[[noreturn]] நிலையான உள்வரி வெறுமை bad_arch(மாறிலி sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (வெறுமை)bad_arch; // suppress unused function warning
}

முழு get_current_device_id();

உள்வரி முழு ggml_sycl_get_device() {
    திரும்பு get_current_device_id();
}

உள்வரி dpct::err0 ggml_sycl_set_device(மாறிலி முழு device) முயல் {
  முழு current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  எனில் (device == current_device_id) {
    திரும்பு 0;
  }

  திரும்பு CHECK_TRY_ERROR(dpct::select_device(device));
} பிடி (sycl::exception மாறிலி& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
கட்டமைப்பு optimize_feature {
    தருக்க reorder=பொய்;
    தருக்க onednn_optimized_gemm=பொய்;
};

கட்டமைப்பு sycl_device_info {
    முழு cc;  // compute capability
    முழு nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    முழு warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    முழு max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    தருக்க    vmm;                // virtual memory support
    தருக்க    l0_device_type_valid;
    தருக்க    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    தருக்க    usm_system_support; // support for USM system allocations
};


கட்டமைப்பு ggml_sycl_device_info {
    முழு device_மொத்தம்;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<தசம, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    முழு max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    தருக்க ext_oneapi_level_zero = உண்மை; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

மாறிலி ggml_sycl_device_info & ggml_sycl_info();

நிலையான constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

கட்டமைப்பு ggml_sycl_pool {
    மெய்நிகர் ~ggml_sycl_pool() = இயல்பு;

    மெய்நிகர் வெறுமை * alloc(size_t size, size_t * actual_size) = 0;
    மெய்நிகர் வெறுமை விடுவி(வெறுமை * ptr, size_t size) = 0;
};

template<typename T>
கட்டமைப்பு ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        எனில் (ptr != nullptr) {
            pool->விடுவி(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        எனில் (ptr)
            pool->விடுவி(ptr, actual_size);
        ptr = (T *) pool->alloc(size * அளவு(T), &இது->actual_size);
        திரும்பு ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * அளவு(T), &இது->actual_size);
        திரும்பு ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        இது->pool = &pool;
        திரும்பு alloc(size);
    }

    T * get() {
        திரும்பு ptr;
    }

    ggml_sycl_pool_alloc() = இயல்பு;
    ggml_sycl_pool_alloc(மாறிலி ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(மாறிலி ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

கட்டமைப்பு ggml_tensor_extra_gpu {
  வெறுமை* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

வெளி முழு g_ggml_sycl_use_level_zero_api;
வெறுமை * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
வெறுமை ggml_sycl_free_device(வெறுமை *ptr, sycl::queue &q);

வெறுமை release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

கட்டமைப்பு mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

கட்டமைப்பு ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

நாமாகாஶ sycl_ex = sycl::ext::oneapi::experimental;
கட்டமைப்பு ggml_backend_sycl_context {
    முழு device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(முழு device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(முழு device, முழு stream) {
        எனில் (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        திரும்பு qptrs[device][stream];
    }

    queue_ptr stream() {
        திரும்பு stream(device, 0);
    }

#எனில் GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        மாறிலி dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        திரும்பு eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(முழு device, முழு _stream) {
        auto q = stream(device, _stream);
        திரும்பு stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        எனில் (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            திரும்பு eng;
        }
        இல்லையெனில்
        {
            திரும்பு it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        எனில் (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            திரும்பு stream;
        }
        இல்லையெனில்
        {
            திரும்பு it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        திரும்பு stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    நிலையான std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, முழு device);

    நிலையான std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, முழு device);

    ggml_sycl_pool & pool(முழு device) {
        எனில் (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        திரும்பு *pools[device];
    }

    ggml_sycl_pool & pool() {
        திரும்பு pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(முழு device) {
        எனில் (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        திரும்பு *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { திரும்பு host_pool(device); }
};

// common device functions

நிலையான __dpct_inline__ தசம warp_reduce_sum(தசம x,
    மாறிலி sycl::nd_item<3>& item_ct1) {
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    திரும்பு x;
}

நிலையான __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, மாறிலி sycl::nd_item<3>& item_ct1) {
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    திரும்பு a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ முழு warp_reduce_sum(முழு x) {
  திரும்பு sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ தசம warp_reduce_sum(தசம x) {
#pragma unroll
  க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  திரும்பு x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ தசம warp_reduce_sum(தசம x, மாறிலி sycl::nd_item<3>& item_ct1) {
#pragma unroll
  க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  திரும்பு x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  திரும்பு a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  திரும்பு a;
}

நிலையான constexpr முழு ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  திரும்பு WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ முழு warp_reduce_all(முழு x) {
    எனில் (width == ggml_sycl_get_physical_warp_size()) {
        திரும்பு sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } இல்லையெனில் {
#pragma unroll
        க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        திரும்பு x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ முழு warp_reduce_any(முழு x) {
    எனில் (width == ggml_sycl_get_physical_warp_size()) {
        திரும்பு sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } இல்லையெனில் {
#pragma unroll
        க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        திரும்பு x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <முழு width>
நிலையான __dpct_inline__ தசம warp_reduce_max(தசம x) {
#pragma unroll
  க்கு (முழு offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  திரும்பு x;
}

நிலையான __dpct_inline__ தசம warp_reduce_max(தசம x,
    மாறிலி sycl::nd_item<3>& item_ct1) {
#pragma unroll
    க்கு (முழு mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    திரும்பு x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<முழு N>
__dpct_inline__ size_t calculate_offset(மாறிலி std::array<முழு, N> & strides, மாறிலி std::array<முழு, N> & indices) {
    size_t offset = 0;
#pragma unroll
    க்கு (முழு i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    திரும்பு offset;
}

// Helper for vec loading aligned data
template <typename Tp, முழு n>
உள்வரி sycl::vec<Tp, n> vec_aligned_load(மாறிலி Tp* aligned_ptr) {
    திரும்பு *reinterpret_cast<மாறிலி sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, முழு dim>
நிலையான __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    திரும்பு acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_இலக்கம், int64_t block_size);

constexpr size_t ceil_div(மாறிலி size_t m, மாறிலி size_t n) {
    திரும்பு (m + n - 1) / n;
}

தருக்க gpu_has_xmx(sycl::device &dev);

#எனில் GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
உள்வரி தருக்க ggml_sycl_dnnl_has_optimized_gemm(முழு device) {
    திரும்பு ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

முழு ggml_sycl_get_env(மாறிலி எழுத்து *env_name, முழு default_val);

template <முழு N, வகுப்பு T> std::string debug_get_array_str(மாறிலி std::string & prefix, மாறிலி T array[N]) {
    எனில் (LIKELY(!g_ggml_sycl_debug)) {
        திரும்பு "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    க்கு (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    எனில் constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    திரும்பு ss.str();
}

உள்வரி std::string debug_get_tensor_str(மாறிலி std::string &prefix,
        மாறிலி ggml_tensor *tensor, மாறிலி std::string &suffix = "") {
    std::stringstream ss;
    எனில் (LIKELY(!g_ggml_sycl_debug)) { திரும்பு ss.str(); }
    ss << prefix.c_str() << "=";
    எனில் (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        எனில் (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        எனில் (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } இல்லையெனில் {
        ss << "nullptr";
    }
    ss << suffix;
    திரும்பு ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
கட்டமைப்பு scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(மாறிலி std::string_view & func, மாறிலி std::string_view & func_suffix, மாறிலி ggml_tensor * dst,
                         std::size_t இலக்கம்_src, மாறிலி std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        எனில் (LIKELY(!g_ggml_sycl_debug)) {
            திரும்பு;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        எனில் (dst) {
            க்கு (std::size_t i = 0; i < இலக்கம்_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(மாறிலி std::string_view & func, மாறிலி ggml_tensor * dst, std::size_t இலக்கம்_src,
                         மாறிலி std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, இலக்கம்_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  தனிப்பட்ட:
    std::string_view func;
    std::string_view func_suffix;
};

நிலையான __dpct_inline__ தசம get_alibi_slope(மாறிலி தசம    max_bias,
                                             மாறிலி uint32_t h,
                                             மாறிலி uint32_t n_head_log2,
                                             மாறிலி தசம    m0,
                                             மாறிலி தசம    m1) {
    எனில் (max_bias <= 0.0f) {
        திரும்பு 1.0f;
    }
    மாறிலி தசம base = h < n_head_log2 ? m0 : m1;
    மாறிலி முழு   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    திரும்பு dpct::pow(base, exph);
}

நிலையான மாறிலி sycl::uint3 init_fastdiv_மதிப்புகள்(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    போது (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    திரும்பு sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
நிலையான constexpr முழு ggml_sycl_get_max_cpy_bytes() {
    திரும்பு 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <முழு nbytes, முழு alignment = 0>
நிலையான __dpct_inline__ வெறுமை ggml_sycl_memcpy_1(வெறுமை * dst, மாறிலி வெறுமை * src) {
    எனில் constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr முழு nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    க்கு (முழு i = 0; i < nbytes/nb_per_cpy; ++i) {
        எனில் constexpr (nb_per_cpy == 1) {
            ((எழுத்து *) dst)[i] = ((மாறிலி எழுத்து *) src)[i];
        } இல்லையெனில் எனில் constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((மாறிலி short *) src)[i];
        } இல்லையெனில் எனில் constexpr (nb_per_cpy == 4) {
            ((முழு *) dst)[i] = ((மாறிலி முழு *) src)[i];
        } இல்லையெனில் எனில் constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((மாறிலி sycl::int2 *) src)[i];
        } இல்லையெனில் எனில் constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((மாறிலி sycl::int4 *) src)[i];
        } இல்லையெனில் {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    திரும்பு res;
}

நிலையான __dpct_inline__ uint32_t fastdiv(uint32_t n, மாறிலி sycl::uint3 fastdiv_மதிப்புகள்) {
    மாறிலி uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_மதிப்புகள்.x());
    திரும்பு (hi + n) >> fastdiv_மதிப்புகள்.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<தசம>(x),static_cast<தசம>(y));
    திரும்பு res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_மதிப்பு(static_cast<தசம>(H.x()), static_cast<தசம>(H.y()));
    திரும்பு float2_மதிப்பு;
}

நிலையான __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, மாறிலி sycl::uint3 fastdiv_மதிப்புகள்) {
    மாறிலி uint32_t div_val = fastdiv(n, fastdiv_மதிப்புகள்);
    மாறிலி uint32_t mod_val = n - div_val * fastdiv_மதிப்புகள்.z();
    திரும்பு sycl::uint2(div_val, mod_val);
}

நிலையான __dpct_inline__ முழு ggml_sycl_dp4a(மாறிலி முழு a, மாறிலி முழு b, முழு c) {
    திரும்பு dpct::dp4a(a, b, c);
}

நிலையான __dpct_inline__ தசம ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    எனில் (x == 0) {
        bits = 0x00400000;
    } இல்லையெனில் {
        bits = (uint32_t) x << 23;
    }

    தசம result;
    memcpy(&result, &bits, அளவு(தசம));
    திரும்பு result;
}

sycl::float2 __dpct_inline__ __half22float2(மாறிலி sycl::half2 &H) {
    sycl::float2 float2_மதிப்பு(static_cast<தசம>(H.x()), static_cast<தசம>(H.y()));
    திரும்பு float2_மதிப்பு;
}

தசம __dpct_inline__ __half2float(sycl::half H) {
    திரும்பு static_cast<தசம>(H);
}

நிலையான __dpct_inline__ வெறுமை ggml_sycl_mad(தசம & acc, மாறிலி தசம v, மாறிலி தசம u) {
    acc += v*u;
}

நிலையான __dpct_inline__ வெறுமை ggml_sycl_mad(தசம & acc, மாறிலி sycl::float2 v, மாறிலி sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

நிலையான __dpct_inline__ வெறுமை ggml_sycl_mad(தசம & acc, மாறிலி sycl::half2 v, மாறிலி sycl::half2 u) {
#ifdef GGML_SYCL_F16
    மாறிலி sycl::float2 tmp = (v * u).template convert<தசம, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#இல்லையெனில்
    மாறிலி sycl::float2 tmpv = __half22float2(v);
    மாறிலி sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

நிலையான __dpct_inline__ வெறுமை ggml_sycl_mad(sycl::half2 & acc, மாறிலி sycl::half2 v, மாறிலி sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#இல்லையெனில்
    மாறிலி sycl::float2 tmpv = __half22float2(v);
    மாறிலி sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <முழு n>
கட்டமைப்பு ggml_sycl_unroll {
    template <typename Func, typename... Args>
    வெறுமை operator()(மாறிலி Func & f, Args... args) மாறிலி {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
கட்டமைப்பு ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    வெறுமை operator()(மாறிலி Func & f, Args... args) மாறிலி {
        f(0, args...);
    }
};

நிலையான __dpct_inline__ sycl::half2 ggml_sycl_hmax2(மாறிலி sycl::half2 a, மாறிலி sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<தசம, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<தசம, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    திரும்பு ret;
}

நிலையான __dpct_inline__ sycl::half ggml_sycl_hmax(மாறிலி sycl::half a, மாறிலி sycl::half b) {
    திரும்பு sycl::vec<தசம, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<தசம, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<தசம, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

நிலையான __dpct_inline__ uint32_t __hgt2_mask(மாறிலி sycl::half2 a, மாறிலி sycl::half2 b) {
    மாறிலி uint32_t mask_கீழ்  = 0x0000FFFF * (தசம(a[0]) > தசம(b[0]));
    மாறிலி uint32_t mask_உயர்வு = 0xFFFF0000 * (தசம(a[1]) > தசம(b[1]));
    திரும்பு mask_கீழ் | mask_உயர்வு;
}

நிலையான __dpct_inline__ uint32_t fastmodulo(uint32_t n, மாறிலி sycl::uint3 fastdiv_மதிப்புகள்) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    திரும்பு n - fastdiv(n, fastdiv_மதிப்புகள்) * fastdiv_மதிப்புகள்.z();
}

நிலையான தருக்க fast_fp16_available(மாறிலி முழு cc) {
    GGML_UNUSED(cc);
    திரும்பு உண்மை;   //Intel GPUs always support FP16.
}

எண்ணிக்கை வகுப்பு block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, முழு warp_size>
கட்டமைப்பு block_reduce_policy;

template <typename T, typename... Ts>
உள்வரி constexpr தருக்க is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
உள்வரி constexpr தருக்க ggml_sycl_dependent_false_v = பொய்;

#define WARP_32_SIZE 32

template <typename T, முழு warp_size> கட்டமைப்பு block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    நிலையான T reduce(T val) {
        எனில் constexpr (is_any<T, தசம, sycl::float2, sycl::half2, முழு>) {
            திரும்பு warp_reduce_sum<warp_size>(val);
        } இல்லையெனில் {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    நிலையான T sentinel() {
        எனில் constexpr (std::is_same_v<T, தசம>) {
            திரும்பு 0.0f;
        } இல்லையெனில் எனில் constexpr (std::is_same_v<T, sycl::float2>) {
            திரும்பு sycl::float2(0.0f, 0.0f);
        } இல்லையெனில் எனில் constexpr (std::is_same_v<T, sycl::half2>) {
            திரும்பு sycl::half2(0.0f, 0.0f);
        } இல்லையெனில் எனில் constexpr (std::is_same_v<T, முழு>) {
            திரும்பு 0;
        } இல்லையெனில் {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, முழு warp_size> கட்டமைப்பு block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    நிலையான T reduce(T val) {
        எனில் constexpr (is_any<T, தசம, sycl::half2>) {
            திரும்பு warp_reduce_max<warp_size>(val);
        } இல்லையெனில் {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    நிலையான T sentinel() {
        எனில் constexpr (std::is_same_v<T, தசம>) {
            திரும்பு -INFINITY;
        } இல்லையெனில் எனில் constexpr (std::is_same_v<T, sycl::half2>) {
            திரும்பு sycl::half2(-INFINITY, -INFINITY);
        } இல்லையெனில் {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, முழு warp_size, typename T>
நிலையான T block_reduce(T val, T * shared_vals, முழு block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    மாறிலி முழு block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    மாறிலி முழு nthreads = item_ct1.get_local_range(2);
    மாறிலி முழு nwarps = nthreads / WARP_SIZE;

    எனில் (block_size > warp_size) {
        ப்ரதிஜ்ஞா((block_size <= 1024) && (block_size % warp_size) == 0);
        மாறிலி முழு warp_id = item_ct1.get_local_id(2) / warp_size;
        மாறிலி முழு lane_id = item_ct1.get_local_id(2) % warp_size;
        எனில் (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        தசம tmp = 0.f;
        எனில் (lane_id < (static_cast<முழு>(block_size) / warp_size)) {
            க்கு (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        திரும்பு block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    திரும்பு val;
}

நிலையான __dpct_inline__ தசம ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    எனில் (x == 0 || x == 0x7F) {
        திரும்பு 0.0f;
    }
    மாறிலி முழு exp = (x >> 3) & 0xF;
    மாறிலி முழு man = x & 0x7;
    தசம raw;
    எனில் (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } இல்லையெனில் {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (தசம) exp - 7.0f);
    }
    திரும்பு raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
