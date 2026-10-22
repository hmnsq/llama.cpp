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

#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <fstream>
#অন্তর্ভাব <iostream>
#অন্তর্ভাব <string>

#অন্তর্ভাব "base.hpp"
#অন্তর্ভাব "dpct/helper.hpp"
#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "ggml-sycl.h"
#অন্তর্ভাব "presets.hpp"
#অন্তর্ভাব "type.hpp"
#অন্তর্ভাব "sycl_hw.hpp"
#অন্তর্ভাব "memtrace.hpp"

নামাকাশ syclexp = sycl::ext::oneapi::experimental;

#যদি defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #অন্তর্ভাব <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#যদি defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#যদি GGML_SYCL_DNNL
#অন্তর্ভাব "dnnl.hpp"
#অন্তর্ভাব "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic ঠেলো
#pragma clang diagnostic ignored "-Wnested-anon-types"
#অন্তর্ভাব "ggml-common.h"
#pragma clang diagnostic বের_করো
#অন্তর্ভাব "ggml-impl.h"

শূন্য* ggml_sycl_host_malloc(size_t size);
শূন্য ggml_sycl_host_free(শূন্য* ptr);


বাহ্য পূর্ণাঙ্ক g_ggml_sycl_debug;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_enable_optimize;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_enable_fusion;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_enable_esimd;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_mmvq_wide;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
গণনা ggml_sycl_xmx_gather_type {
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
স্থির constexpr পূর্ণাঙ্ক GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
স্থির constexpr পূর্ণাঙ্ক GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
গণনা ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
স্থির constexpr পূর্ণাঙ্ক GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#অন্যথা
স্থির constexpr পূর্ণাঙ্ক GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
স্থির অন্তঃস্থ int32_t ggml_sycl_src1_prec(ধ্রুব ggml_tensor * dst) {
    প্রত্যাগম g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
স্থির অন্তঃস্থ তার্কিক ggml_sycl_src1_f16_ok(ধ্রুব ggml_tensor * dst) {
    ধ্রুব int32_t src1_prec = ggml_sycl_src1_prec(dst);
    প্রত্যাগম g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_enable_flash_attention;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_dev2dev_memcpy;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_fa_onednn;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_fa_onednn_max_kv;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_enable_mkl_fa;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_memtrace;
বাহ্য পূর্ণাঙ্ক g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(রাশিমালা)                                            \
  [&]() {                                                                \
    যত্ন {                                                                \
      রাশিমালা;                                                              \
      প্রত্যাগম dpct::success;                                              \
    } গৃহাণ (std::exception ধ্রুব& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      প্রত্যাগম dpct::default_error;                                        \
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
#যদি !defined(GGML_SYCL_FORCE_MMQ)
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

প্রকারনির্দেশ sycl::queue *queue_ptr;

গণনা ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

গণনা ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(পরিমাণ(sycl::half) == পরিমাণ(ggml_fp16_t), "wrong fp16 size");

স্থির শূন্য crash() {
  পূর্ণাঙ্ক* ptr = রিক্ত;
  *ptr = 0;
}

[[noreturn]] স্থির শূন্য ggml_sycl_error(
    ধ্রুব বর্ণ* stmt,
    ধ্রুব বর্ণ* func,
    ধ্রুব বর্ণ* file,
    ধ্রুব পূর্ণাঙ্ক সারি,
    ধ্রুব বর্ণ* বার্তা) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, বার্তা);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, সারি);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    কুরু {                                                                                                   \
        auto err_ = (err);                                                                                 \
        যদি (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } যাবত্ (0)

#যদি DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#অন্যথা
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
প্রকারনির্দেশ sycl::half dfloat; // dequantize float
প্রকারনির্দেশ sycl::half2 dfloat2;
#অন্যথা
প্রকারনির্দেশ ভিন্ন dfloat; // dequantize float
প্রকারনির্দেশ sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

স্থির পূর্ণাঙ্ক g_all_sycl_device_গুনতি = -1;
স্থির তার্কিক g_ggml_backend_sycl_buffer_type_initialized = অসত্য;

স্থির ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

স্থির শূন্য* g_scratch_buffer = nullptr;
স্থির size_t g_scratch_size = 0; // disabled by default
স্থির size_t g_scratch_offset = 0;

[[noreturn]] স্থির অন্তঃস্থ শূন্য bad_arch(ধ্রুব sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (শূন্য)bad_arch; // suppress unused function warning
}

পূর্ণাঙ্ক get_current_device_id();

অন্তঃস্থ পূর্ণাঙ্ক ggml_sycl_get_device() {
    প্রত্যাগম get_current_device_id();
}

অন্তঃস্থ dpct::err0 ggml_sycl_set_device(ধ্রুব পূর্ণাঙ্ক device) যত্ন {
  পূর্ণাঙ্ক current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  যদি (device == current_device_id) {
    প্রত্যাগম 0;
  }

  প্রত্যাগম CHECK_TRY_ERROR(dpct::select_device(device));
} গৃহাণ (sycl::exception ধ্রুব& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
সংরচনা optimize_feature {
    তার্কিক reorder=অসত্য;
    তার্কিক onednn_optimized_gemm=অসত্য;
};

সংরচনা sycl_device_info {
    পূর্ণাঙ্ক cc;  // compute capability
    পূর্ণাঙ্ক nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    পূর্ণাঙ্ক warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    পূর্ণাঙ্ক max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    তার্কিক    vmm;                // virtual memory support
    তার্কিক    l0_device_type_valid;
    তার্কিক    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    তার্কিক    usm_system_support; // support for USM system allocations
};


সংরচনা ggml_sycl_device_info {
    পূর্ণাঙ্ক device_গুনতি;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<ভিন্ন, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    পূর্ণাঙ্ক max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    তার্কিক ext_oneapi_level_zero = সত্য; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ধ্রুব ggml_sycl_device_info & ggml_sycl_info();

স্থির constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

সংরচনা ggml_sycl_pool {
    আভাস ~ggml_sycl_pool() = স্বভাব;

    আভাস শূন্য * alloc(size_t size, size_t * actual_size) = 0;
    আভাস শূন্য মুক্ত(শূন্য * ptr, size_t size) = 0;
};

template<typename T>
সংরচনা ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        যদি (ptr != nullptr) {
            pool->মুক্ত(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        যদি (ptr)
            pool->মুক্ত(ptr, actual_size);
        ptr = (T *) pool->alloc(size * পরিমাণ(T), &এতত্->actual_size);
        প্রত্যাগম ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * পরিমাণ(T), &এতত্->actual_size);
        প্রত্যাগম ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        এতত্->pool = &pool;
        প্রত্যাগম alloc(size);
    }

    T * get() {
        প্রত্যাগম ptr;
    }

    ggml_sycl_pool_alloc() = স্বভাব;
    ggml_sycl_pool_alloc(ধ্রুব ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(ধ্রুব ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

সংরচনা ggml_tensor_extra_gpu {
  শূন্য* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

বাহ্য পূর্ণাঙ্ক g_ggml_sycl_use_level_zero_api;
শূন্য * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
শূন্য ggml_sycl_free_device(শূন্য *ptr, sycl::queue &q);

শূন্য release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

সংরচনা mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

সংরচনা ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

নামাকাশ sycl_ex = sycl::ext::oneapi::experimental;
সংরচনা ggml_backend_sycl_context {
    পূর্ণাঙ্ক device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(পূর্ণাঙ্ক device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(পূর্ণাঙ্ক device, পূর্ণাঙ্ক stream) {
        যদি (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        প্রত্যাগম qptrs[device][stream];
    }

    queue_ptr stream() {
        প্রত্যাগম stream(device, 0);
    }

#যদি GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ধ্রুব dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        প্রত্যাগম eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(পূর্ণাঙ্ক device, পূর্ণাঙ্ক _stream) {
        auto q = stream(device, _stream);
        প্রত্যাগম stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        যদি (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            প্রত্যাগম eng;
        }
        অন্যথা
        {
            প্রত্যাগম it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        যদি (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            প্রত্যাগম stream;
        }
        অন্যথা
        {
            প্রত্যাগম it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        প্রত্যাগম stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    স্থির std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, পূর্ণাঙ্ক device);

    স্থির std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, পূর্ণাঙ্ক device);

    ggml_sycl_pool & pool(পূর্ণাঙ্ক device) {
        যদি (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        প্রত্যাগম *pools[device];
    }

    ggml_sycl_pool & pool() {
        প্রত্যাগম pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(পূর্ণাঙ্ক device) {
        যদি (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        প্রত্যাগম *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { প্রত্যাগম host_pool(device); }
};

// common device functions

স্থির __dpct_inline__ ভিন্ন warp_reduce_sum(ভিন্ন x,
    ধ্রুব sycl::nd_item<3>& item_ct1) {
#pragma unroll
    প্রতি (পূর্ণাঙ্ক mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    প্রত্যাগম x;
}

স্থির __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, ধ্রুব sycl::nd_item<3>& item_ct1) {
#pragma unroll
    প্রতি (পূর্ণাঙ্ক mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    প্রত্যাগম a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ পূর্ণাঙ্ক warp_reduce_sum(পূর্ণাঙ্ক x) {
  প্রত্যাগম sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ ভিন্ন warp_reduce_sum(ভিন্ন x) {
#pragma unroll
  প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  প্রত্যাগম x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ ভিন্ন warp_reduce_sum(ভিন্ন x, ধ্রুব sycl::nd_item<3>& item_ct1) {
#pragma unroll
  প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  প্রত্যাগম x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  প্রত্যাগম a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  প্রত্যাগম a;
}

স্থির constexpr পূর্ণাঙ্ক ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  প্রত্যাগম WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ পূর্ণাঙ্ক warp_reduce_all(পূর্ণাঙ্ক x) {
    যদি (width == ggml_sycl_get_physical_warp_size()) {
        প্রত্যাগম sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } অন্যথা {
#pragma unroll
        প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        প্রত্যাগম x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ পূর্ণাঙ্ক warp_reduce_any(পূর্ণাঙ্ক x) {
    যদি (width == ggml_sycl_get_physical_warp_size()) {
        প্রত্যাগম sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } অন্যথা {
#pragma unroll
        প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        প্রত্যাগম x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <পূর্ণাঙ্ক width>
স্থির __dpct_inline__ ভিন্ন warp_reduce_max(ভিন্ন x) {
#pragma unroll
  প্রতি (পূর্ণাঙ্ক offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  প্রত্যাগম x;
}

স্থির __dpct_inline__ ভিন্ন warp_reduce_max(ভিন্ন x,
    ধ্রুব sycl::nd_item<3>& item_ct1) {
#pragma unroll
    প্রতি (পূর্ণাঙ্ক mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    প্রত্যাগম x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<পূর্ণাঙ্ক N>
__dpct_inline__ size_t calculate_offset(ধ্রুব std::array<পূর্ণাঙ্ক, N> & strides, ধ্রুব std::array<পূর্ণাঙ্ক, N> & indices) {
    size_t offset = 0;
#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    প্রত্যাগম offset;
}

// Helper for vec loading aligned data
template <typename Tp, পূর্ণাঙ্ক n>
অন্তঃস্থ sycl::vec<Tp, n> vec_aligned_load(ধ্রুব Tp* aligned_ptr) {
    প্রত্যাগম *reinterpret_cast<ধ্রুব sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, পূর্ণাঙ্ক dim>
স্থির __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    প্রত্যাগম acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_অঙ্ক, int64_t block_size);

constexpr size_t ceil_div(ধ্রুব size_t m, ধ্রুব size_t n) {
    প্রত্যাগম (m + n - 1) / n;
}

তার্কিক gpu_has_xmx(sycl::device &dev);

#যদি GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
অন্তঃস্থ তার্কিক ggml_sycl_dnnl_has_optimized_gemm(পূর্ণাঙ্ক device) {
    প্রত্যাগম ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

পূর্ণাঙ্ক ggml_sycl_get_env(ধ্রুব বর্ণ *env_name, পূর্ণাঙ্ক default_val);

template <পূর্ণাঙ্ক N, বর্গ T> std::string debug_get_array_str(ধ্রুব std::string & prefix, ধ্রুব T array[N]) {
    যদি (LIKELY(!g_ggml_sycl_debug)) {
        প্রত্যাগম "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    প্রতি (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    যদি constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    প্রত্যাগম ss.str();
}

অন্তঃস্থ std::string debug_get_tensor_str(ধ্রুব std::string &prefix,
        ধ্রুব ggml_tensor *tensor, ধ্রুব std::string &suffix = "") {
    std::stringstream ss;
    যদি (LIKELY(!g_ggml_sycl_debug)) { প্রত্যাগম ss.str(); }
    ss << prefix.c_str() << "=";
    যদি (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        যদি (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        যদি (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } অন্যথা {
        ss << "nullptr";
    }
    ss << suffix;
    প্রত্যাগম ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
সংরচনা scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ধ্রুব std::string_view & func, ধ্রুব std::string_view & func_suffix, ধ্রুব ggml_tensor * dst,
                         std::size_t অঙ্ক_src, ধ্রুব std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        যদি (LIKELY(!g_ggml_sycl_debug)) {
            প্রত্যাগম;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        যদি (dst) {
            প্রতি (std::size_t i = 0; i < অঙ্ক_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(ধ্রুব std::string_view & func, ধ্রুব ggml_tensor * dst, std::size_t অঙ্ক_src,
                         ধ্রুব std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, অঙ্ক_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  গুপ্ত:
    std::string_view func;
    std::string_view func_suffix;
};

স্থির __dpct_inline__ ভিন্ন get_alibi_slope(ধ্রুব ভিন্ন    max_bias,
                                             ধ্রুব uint32_t h,
                                             ধ্রুব uint32_t n_head_log2,
                                             ধ্রুব ভিন্ন    m0,
                                             ধ্রুব ভিন্ন    m1) {
    যদি (max_bias <= 0.0f) {
        প্রত্যাগম 1.0f;
    }
    ধ্রুব ভিন্ন base = h < n_head_log2 ? m0 : m1;
    ধ্রুব পূর্ণাঙ্ক   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    প্রত্যাগম dpct::pow(base, exph);
}

স্থির ধ্রুব sycl::uint3 init_fastdiv_মানসমূহ(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    যাবত্ (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    প্রত্যাগম sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
স্থির constexpr পূর্ণাঙ্ক ggml_sycl_get_max_cpy_bytes() {
    প্রত্যাগম 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <পূর্ণাঙ্ক nbytes, পূর্ণাঙ্ক alignment = 0>
স্থির __dpct_inline__ শূন্য ggml_sycl_memcpy_1(শূন্য * dst, ধ্রুব শূন্য * src) {
    যদি constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr পূর্ণাঙ্ক nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    প্রতি (পূর্ণাঙ্ক i = 0; i < nbytes/nb_per_cpy; ++i) {
        যদি constexpr (nb_per_cpy == 1) {
            ((বর্ণ *) dst)[i] = ((ধ্রুব বর্ণ *) src)[i];
        } অন্যথা যদি constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((ধ্রুব short *) src)[i];
        } অন্যথা যদি constexpr (nb_per_cpy == 4) {
            ((পূর্ণাঙ্ক *) dst)[i] = ((ধ্রুব পূর্ণাঙ্ক *) src)[i];
        } অন্যথা যদি constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ধ্রুব sycl::int2 *) src)[i];
        } অন্যথা যদি constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ধ্রুব sycl::int4 *) src)[i];
        } অন্যথা {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    প্রত্যাগম res;
}

স্থির __dpct_inline__ uint32_t fastdiv(uint32_t n, ধ্রুব sycl::uint3 fastdiv_মানসমূহ) {
    ধ্রুব uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_মানসমূহ.x());
    প্রত্যাগম (hi + n) >> fastdiv_মানসমূহ.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<ভিন্ন>(x),static_cast<ভিন্ন>(y));
    প্রত্যাগম res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_মান(static_cast<ভিন্ন>(H.x()), static_cast<ভিন্ন>(H.y()));
    প্রত্যাগম float2_মান;
}

স্থির __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ধ্রুব sycl::uint3 fastdiv_মানসমূহ) {
    ধ্রুব uint32_t div_val = fastdiv(n, fastdiv_মানসমূহ);
    ধ্রুব uint32_t mod_val = n - div_val * fastdiv_মানসমূহ.z();
    প্রত্যাগম sycl::uint2(div_val, mod_val);
}

স্থির __dpct_inline__ পূর্ণাঙ্ক ggml_sycl_dp4a(ধ্রুব পূর্ণাঙ্ক a, ধ্রুব পূর্ণাঙ্ক b, পূর্ণাঙ্ক c) {
    প্রত্যাগম dpct::dp4a(a, b, c);
}

স্থির __dpct_inline__ ভিন্ন ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    যদি (x == 0) {
        bits = 0x00400000;
    } অন্যথা {
        bits = (uint32_t) x << 23;
    }

    ভিন্ন result;
    memcpy(&result, &bits, পরিমাণ(ভিন্ন));
    প্রত্যাগম result;
}

sycl::float2 __dpct_inline__ __half22float2(ধ্রুব sycl::half2 &H) {
    sycl::float2 float2_মান(static_cast<ভিন্ন>(H.x()), static_cast<ভিন্ন>(H.y()));
    প্রত্যাগম float2_মান;
}

ভিন্ন __dpct_inline__ __half2float(sycl::half H) {
    প্রত্যাগম static_cast<ভিন্ন>(H);
}

স্থির __dpct_inline__ শূন্য ggml_sycl_mad(ভিন্ন & acc, ধ্রুব ভিন্ন v, ধ্রুব ভিন্ন u) {
    acc += v*u;
}

স্থির __dpct_inline__ শূন্য ggml_sycl_mad(ভিন্ন & acc, ধ্রুব sycl::float2 v, ধ্রুব sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

স্থির __dpct_inline__ শূন্য ggml_sycl_mad(ভিন্ন & acc, ধ্রুব sycl::half2 v, ধ্রুব sycl::half2 u) {
#ifdef GGML_SYCL_F16
    ধ্রুব sycl::float2 tmp = (v * u).template convert<ভিন্ন, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#অন্যথা
    ধ্রুব sycl::float2 tmpv = __half22float2(v);
    ধ্রুব sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

স্থির __dpct_inline__ শূন্য ggml_sycl_mad(sycl::half2 & acc, ধ্রুব sycl::half2 v, ধ্রুব sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#অন্যথা
    ধ্রুব sycl::float2 tmpv = __half22float2(v);
    ধ্রুব sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <পূর্ণাঙ্ক n>
সংরচনা ggml_sycl_unroll {
    template <typename Func, typename... Args>
    শূন্য operator()(ধ্রুব Func & f, Args... args) ধ্রুব {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
সংরচনা ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    শূন্য operator()(ধ্রুব Func & f, Args... args) ধ্রুব {
        f(0, args...);
    }
};

স্থির __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ধ্রুব sycl::half2 a, ধ্রুব sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<ভিন্ন, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<ভিন্ন, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    প্রত্যাগম ret;
}

স্থির __dpct_inline__ sycl::half ggml_sycl_hmax(ধ্রুব sycl::half a, ধ্রুব sycl::half b) {
    প্রত্যাগম sycl::vec<ভিন্ন, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<ভিন্ন, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<ভিন্ন, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

স্থির __dpct_inline__ uint32_t __hgt2_mask(ধ্রুব sycl::half2 a, ধ্রুব sycl::half2 b) {
    ধ্রুব uint32_t mask_নিচের  = 0x0000FFFF * (ভিন্ন(a[0]) > ভিন্ন(b[0]));
    ধ্রুব uint32_t mask_উপরের = 0xFFFF0000 * (ভিন্ন(a[1]) > ভিন্ন(b[1]));
    প্রত্যাগম mask_নিচের | mask_উপরের;
}

স্থির __dpct_inline__ uint32_t fastmodulo(uint32_t n, ধ্রুব sycl::uint3 fastdiv_মানসমূহ) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    প্রত্যাগম n - fastdiv(n, fastdiv_মানসমূহ) * fastdiv_মানসমূহ.z();
}

স্থির তার্কিক fast_fp16_available(ধ্রুব পূর্ণাঙ্ক cc) {
    GGML_UNUSED(cc);
    প্রত্যাগম সত্য;   //Intel GPUs always support FP16.
}

গণনা বর্গ block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, পূর্ণাঙ্ক warp_size>
সংরচনা block_reduce_policy;

template <typename T, typename... Ts>
অন্তঃস্থ constexpr তার্কিক is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
অন্তঃস্থ constexpr তার্কিক ggml_sycl_dependent_false_v = অসত্য;

#define WARP_32_SIZE 32

template <typename T, পূর্ণাঙ্ক warp_size> সংরচনা block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    স্থির T reduce(T val) {
        যদি constexpr (is_any<T, ভিন্ন, sycl::float2, sycl::half2, পূর্ণাঙ্ক>) {
            প্রত্যাগম warp_reduce_sum<warp_size>(val);
        } অন্যথা {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    স্থির T sentinel() {
        যদি constexpr (std::is_same_v<T, ভিন্ন>) {
            প্রত্যাগম 0.0f;
        } অন্যথা যদি constexpr (std::is_same_v<T, sycl::float2>) {
            প্রত্যাগম sycl::float2(0.0f, 0.0f);
        } অন্যথা যদি constexpr (std::is_same_v<T, sycl::half2>) {
            প্রত্যাগম sycl::half2(0.0f, 0.0f);
        } অন্যথা যদি constexpr (std::is_same_v<T, পূর্ণাঙ্ক>) {
            প্রত্যাগম 0;
        } অন্যথা {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, পূর্ণাঙ্ক warp_size> সংরচনা block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    স্থির T reduce(T val) {
        যদি constexpr (is_any<T, ভিন্ন, sycl::half2>) {
            প্রত্যাগম warp_reduce_max<warp_size>(val);
        } অন্যথা {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    স্থির T sentinel() {
        যদি constexpr (std::is_same_v<T, ভিন্ন>) {
            প্রত্যাগম -INFINITY;
        } অন্যথা যদি constexpr (std::is_same_v<T, sycl::half2>) {
            প্রত্যাগম sycl::half2(-INFINITY, -INFINITY);
        } অন্যথা {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, পূর্ণাঙ্ক warp_size, typename T>
স্থির T block_reduce(T val, T * shared_vals, পূর্ণাঙ্ক block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    ধ্রুব পূর্ণাঙ্ক block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    ধ্রুব পূর্ণাঙ্ক nthreads = item_ct1.get_local_range(2);
    ধ্রুব পূর্ণাঙ্ক nwarps = nthreads / WARP_SIZE;

    যদি (block_size > warp_size) {
        প্রতিজ্ঞা((block_size <= 1024) && (block_size % warp_size) == 0);
        ধ্রুব পূর্ণাঙ্ক warp_id = item_ct1.get_local_id(2) / warp_size;
        ধ্রুব পূর্ণাঙ্ক lane_id = item_ct1.get_local_id(2) % warp_size;
        যদি (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        ভিন্ন tmp = 0.f;
        যদি (lane_id < (static_cast<পূর্ণাঙ্ক>(block_size) / warp_size)) {
            প্রতি (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        প্রত্যাগম block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    প্রত্যাগম val;
}

স্থির __dpct_inline__ ভিন্ন ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    যদি (x == 0 || x == 0x7F) {
        প্রত্যাগম 0.0f;
    }
    ধ্রুব পূর্ণাঙ্ক exp = (x >> 3) & 0xF;
    ধ্রুব পূর্ণাঙ্ক man = x & 0x7;
    ভিন্ন raw;
    যদি (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } অন্যথা {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (ভিন্ন) exp - 7.0f);
    }
    প্রত্যাগম raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
