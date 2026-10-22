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

#अन्तर्भाव <cstddef>
#अन्तर्भाव <fstream>
#अन्तर्भाव <iostream>
#अन्तर्भाव <string>

#अन्तर्भाव "base.hpp"
#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-sycl.h"
#अन्तर्भाव "presets.hpp"
#अन्तर्भाव "type.hpp"
#अन्तर्भाव "sycl_hw.hpp"
#अन्तर्भाव "memtrace.hpp"

नामाकाश syclexp = sycl::ext::oneapi::experimental;

#यदि defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #अन्तर्भाव <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#यदि defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#यदि GGML_SYCL_DNNL
#अन्तर्भाव "dnnl.hpp"
#अन्तर्भाव "dnnl_sycl.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#define GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#pragma clang diagnostic धकेलो
#pragma clang diagnostic ignored "-Wnested-anon-types"
#अन्तर्भाव "ggml-common.h"
#pragma clang diagnostic निकालो
#अन्तर्भाव "ggml-impl.h"

शून्य* ggml_sycl_host_malloc(size_t size);
शून्य ggml_sycl_host_free(शून्य* ptr);


बाह्य पूर्णाङ्क g_ggml_sycl_debug;
बाह्य पूर्णाङ्क g_ggml_sycl_enable_optimize;
बाह्य पूर्णाङ्क g_ggml_sycl_enable_fusion;
बाह्य पूर्णाङ्क g_ggml_sycl_enable_esimd;
बाह्य पूर्णाङ्क g_ggml_sycl_mmvq_wide;
बाह्य पूर्णाङ्क g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
गणना ggml_sycl_xmx_gather_type {
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
स्थिर constexpr पूर्णाङ्क GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
बाह्य पूर्णाङ्क g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
स्थिर constexpr पूर्णाङ्क GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
बाह्य पूर्णाङ्क g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
गणना ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
स्थिर constexpr पूर्णाङ्क GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#अन्यथा
स्थिर constexpr पूर्णाङ्क GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
बाह्य पूर्णाङ्क g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
बाह्य पूर्णाङ्क g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
स्थिर अन्तःस्थ int32_t ggml_sycl_src1_prec(ध्रुव ggml_tensor * dst) {
    प्रत्यागम g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
स्थिर अन्तःस्थ तार्किक ggml_sycl_src1_f16_ok(ध्रुव ggml_tensor * dst) {
    ध्रुव int32_t src1_prec = ggml_sycl_src1_prec(dst);
    प्रत्यागम g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
बाह्य पूर्णाङ्क g_ggml_sycl_enable_flash_attention;
बाह्य पूर्णाङ्क g_ggml_sycl_dev2dev_memcpy;
बाह्य पूर्णाङ्क g_ggml_sycl_fa_onednn;
बाह्य पूर्णाङ्क g_ggml_sycl_fa_onednn_max_kv;
बाह्य पूर्णाङ्क g_ggml_sycl_enable_mkl_fa;
बाह्य पूर्णाङ्क g_ggml_sycl_memtrace;
बाह्य पूर्णाङ्क g_ggml_sycl_memtrace_step;


#define CHECK_TRY_ERROR(व्यंजक)                                            \
  [&]() {                                                                \
    यत्न {                                                                \
      व्यंजक;                                                              \
      प्रत्यागम dpct::success;                                              \
    } गृहाण (std::exception ध्रुव& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      प्रत्यागम dpct::default_error;                                        \
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
#यदि !defined(GGML_SYCL_FORCE_MMQ)
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

प्रकारनिर्देश sycl::queue *queue_ptr;

गणना ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

गणना ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(परिमाण(sycl::half) == परिमाण(ggml_fp16_t), "wrong fp16 size");

स्थिर शून्य crash() {
  पूर्णाङ्क* ptr = रिक्त;
  *ptr = 0;
}

[[noreturn]] स्थिर शून्य ggml_sycl_error(
    ध्रुव वर्ण* stmt,
    ध्रुव वर्ण* func,
    ध्रुव वर्ण* file,
    ध्रुव पूर्णाङ्क पंक्ति,
    ध्रुव वर्ण* संदेश) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, संदेश);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, पंक्ति);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    कुरु {                                                                                                   \
        auto err_ = (err);                                                                                 \
        यदि (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } यावत् (0)

#यदि DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#अन्यथा
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
प्रकारनिर्देश sycl::half dfloat; // dequantize float
प्रकारनिर्देश sycl::half2 dfloat2;
#अन्यथा
प्रकारनिर्देश भिन्न dfloat; // dequantize float
प्रकारनिर्देश sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8

स्थिर पूर्णाङ्क g_all_sycl_device_गिनती = -1;
स्थिर तार्किक g_ggml_backend_sycl_buffer_type_initialized = असत्य;

स्थिर ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

स्थिर शून्य* g_scratch_buffer = nullptr;
स्थिर size_t g_scratch_size = 0; // disabled by default
स्थिर size_t g_scratch_offset = 0;

[[noreturn]] स्थिर अन्तःस्थ शून्य bad_arch(ध्रुव sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (शून्य)bad_arch; // suppress unused function warning
}

पूर्णाङ्क get_current_device_id();

अन्तःस्थ पूर्णाङ्क ggml_sycl_get_device() {
    प्रत्यागम get_current_device_id();
}

अन्तःस्थ dpct::err0 ggml_sycl_set_device(ध्रुव पूर्णाङ्क device) यत्न {
  पूर्णाङ्क current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  यदि (device == current_device_id) {
    प्रत्यागम 0;
  }

  प्रत्यागम CHECK_TRY_ERROR(dpct::select_device(device));
} गृहाण (sycl::exception ध्रुव& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
संरचना optimize_feature {
    तार्किक reorder=असत्य;
    तार्किक onednn_optimized_gemm=असत्य;
};

संरचना sycl_device_info {
    पूर्णाङ्क cc;  // compute capability
    पूर्णाङ्क nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    पूर्णाङ्क warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    पूर्णाङ्क max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    तार्किक    vmm;                // virtual memory support
    तार्किक    l0_device_type_valid;
    तार्किक    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    तार्किक    usm_system_support; // support for USM system allocations
};


संरचना ggml_sycl_device_info {
    पूर्णाङ्क device_गिनती;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<भिन्न, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    पूर्णाङ्क max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    तार्किक ext_oneapi_level_zero = सत्य; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ध्रुव ggml_sycl_device_info & ggml_sycl_info();

स्थिर constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

संरचना ggml_sycl_pool {
    आभास ~ggml_sycl_pool() = स्वभाव;

    आभास शून्य * alloc(size_t size, size_t * actual_size) = 0;
    आभास शून्य मुक्त(शून्य * ptr, size_t size) = 0;
};

template<typename T>
संरचना ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        यदि (ptr != nullptr) {
            pool->मुक्त(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        यदि (ptr)
            pool->मुक्त(ptr, actual_size);
        ptr = (T *) pool->alloc(size * परिमाण(T), &एतत्->actual_size);
        प्रत्यागम ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * परिमाण(T), &एतत्->actual_size);
        प्रत्यागम ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        एतत्->pool = &pool;
        प्रत्यागम alloc(size);
    }

    T * get() {
        प्रत्यागम ptr;
    }

    ggml_sycl_pool_alloc() = स्वभाव;
    ggml_sycl_pool_alloc(ध्रुव ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(ध्रुव ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

संरचना ggml_tensor_extra_gpu {
  शून्य* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

बाह्य पूर्णाङ्क g_ggml_sycl_use_level_zero_api;
शून्य * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
शून्य ggml_sycl_free_device(शून्य *ptr, sycl::queue &q);

शून्य release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

संरचना mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

संरचना ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

नामाकाश sycl_ex = sycl::ext::oneapi::experimental;
संरचना ggml_backend_sycl_context {
    पूर्णाङ्क device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(पूर्णाङ्क device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(पूर्णाङ्क device, पूर्णाङ्क stream) {
        यदि (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        प्रत्यागम qptrs[device][stream];
    }

    queue_ptr stream() {
        प्रत्यागम stream(device, 0);
    }

#यदि GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ध्रुव dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        प्रत्यागम eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(पूर्णाङ्क device, पूर्णाङ्क _stream) {
        auto q = stream(device, _stream);
        प्रत्यागम stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        यदि (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            प्रत्यागम eng;
        }
        अन्यथा
        {
            प्रत्यागम it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        यदि (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            प्रत्यागम stream;
        }
        अन्यथा
        {
            प्रत्यागम it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        प्रत्यागम stream_dnnl(device, 0);
    }
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    स्थिर std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, पूर्णाङ्क device);

    स्थिर std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, पूर्णाङ्क device);

    ggml_sycl_pool & pool(पूर्णाङ्क device) {
        यदि (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        प्रत्यागम *pools[device];
    }

    ggml_sycl_pool & pool() {
        प्रत्यागम pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#endif

    ggml_sycl_pool & host_pool(पूर्णाङ्क device) {
        यदि (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        प्रत्यागम *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { प्रत्यागम host_pool(device); }
};

// common device functions

स्थिर __dpct_inline__ भिन्न warp_reduce_sum(भिन्न x,
    ध्रुव sycl::nd_item<3>& item_ct1) {
#pragma unroll
    प्रति (पूर्णाङ्क mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    प्रत्यागम x;
}

स्थिर __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, ध्रुव sycl::nd_item<3>& item_ct1) {
#pragma unroll
    प्रति (पूर्णाङ्क mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    प्रत्यागम a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ पूर्णाङ्क warp_reduce_sum(पूर्णाङ्क x) {
  प्रत्यागम sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ भिन्न warp_reduce_sum(भिन्न x) {
#pragma unroll
  प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  प्रत्यागम x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ भिन्न warp_reduce_sum(भिन्न x, ध्रुव sycl::nd_item<3>& item_ct1) {
#pragma unroll
  प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  प्रत्यागम x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  प्रत्यागम a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  प्रत्यागम a;
}

स्थिर constexpr पूर्णाङ्क ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  प्रत्यागम WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ पूर्णाङ्क warp_reduce_all(पूर्णाङ्क x) {
    यदि (width == ggml_sycl_get_physical_warp_size()) {
        प्रत्यागम sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } अन्यथा {
#pragma unroll
        प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        प्रत्यागम x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ पूर्णाङ्क warp_reduce_any(पूर्णाङ्क x) {
    यदि (width == ggml_sycl_get_physical_warp_size()) {
        प्रत्यागम sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } अन्यथा {
#pragma unroll
        प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        प्रत्यागम x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <पूर्णाङ्क width>
स्थिर __dpct_inline__ भिन्न warp_reduce_max(भिन्न x) {
#pragma unroll
  प्रति (पूर्णाङ्क offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  प्रत्यागम x;
}

स्थिर __dpct_inline__ भिन्न warp_reduce_max(भिन्न x,
    ध्रुव sycl::nd_item<3>& item_ct1) {
#pragma unroll
    प्रति (पूर्णाङ्क mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    प्रत्यागम x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<पूर्णाङ्क N>
__dpct_inline__ size_t calculate_offset(ध्रुव std::array<पूर्णाङ्क, N> & strides, ध्रुव std::array<पूर्णाङ्क, N> & indices) {
    size_t offset = 0;
#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    प्रत्यागम offset;
}

// Helper for vec loading aligned data
template <typename Tp, पूर्णाङ्क n>
अन्तःस्थ sycl::vec<Tp, n> vec_aligned_load(ध्रुव Tp* aligned_ptr) {
    प्रत्यागम *reinterpret_cast<ध्रुव sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, पूर्णाङ्क dim>
स्थिर __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    प्रत्यागम acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_अंक, int64_t block_size);

constexpr size_t ceil_div(ध्रुव size_t m, ध्रुव size_t n) {
    प्रत्यागम (m + n - 1) / n;
}

तार्किक gpu_has_xmx(sycl::device &dev);

#यदि GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
अन्तःस्थ तार्किक ggml_sycl_dnnl_has_optimized_gemm(पूर्णाङ्क device) {
    प्रत्यागम ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

पूर्णाङ्क ggml_sycl_get_env(ध्रुव वर्ण *env_name, पूर्णाङ्क default_val);

template <पूर्णाङ्क N, वर्ग T> std::string debug_get_array_str(ध्रुव std::string & prefix, ध्रुव T array[N]) {
    यदि (LIKELY(!g_ggml_sycl_debug)) {
        प्रत्यागम "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    प्रति (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    यदि constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    प्रत्यागम ss.str();
}

अन्तःस्थ std::string debug_get_tensor_str(ध्रुव std::string &prefix,
        ध्रुव ggml_tensor *tensor, ध्रुव std::string &suffix = "") {
    std::stringstream ss;
    यदि (LIKELY(!g_ggml_sycl_debug)) { प्रत्यागम ss.str(); }
    ss << prefix.c_str() << "=";
    यदि (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        यदि (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        यदि (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } अन्यथा {
        ss << "nullptr";
    }
    ss << suffix;
    प्रत्यागम ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
संरचना scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ध्रुव std::string_view & func, ध्रुव std::string_view & func_suffix, ध्रुव ggml_tensor * dst,
                         std::size_t अंक_src, ध्रुव std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        यदि (LIKELY(!g_ggml_sycl_debug)) {
            प्रत्यागम;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        यदि (dst) {
            प्रति (std::size_t i = 0; i < अंक_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(ध्रुव std::string_view & func, ध्रुव ggml_tensor * dst, std::size_t अंक_src,
                         ध्रुव std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, अंक_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  गुप्त:
    std::string_view func;
    std::string_view func_suffix;
};

स्थिर __dpct_inline__ भिन्न get_alibi_slope(ध्रुव भिन्न    max_bias,
                                             ध्रुव uint32_t h,
                                             ध्रुव uint32_t n_head_log2,
                                             ध्रुव भिन्न    m0,
                                             ध्रुव भिन्न    m1) {
    यदि (max_bias <= 0.0f) {
        प्रत्यागम 1.0f;
    }
    ध्रुव भिन्न base = h < n_head_log2 ? m0 : m1;
    ध्रुव पूर्णाङ्क   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    प्रत्यागम dpct::pow(base, exph);
}

स्थिर ध्रुव sycl::uint3 init_fastdiv_मूल्यावली(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    यावत् (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    प्रत्यागम sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
स्थिर constexpr पूर्णाङ्क ggml_sycl_get_max_cpy_bytes() {
    प्रत्यागम 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <पूर्णाङ्क nbytes, पूर्णाङ्क alignment = 0>
स्थिर __dpct_inline__ शून्य ggml_sycl_memcpy_1(शून्य * dst, ध्रुव शून्य * src) {
    यदि constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr पूर्णाङ्क nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    प्रति (पूर्णाङ्क i = 0; i < nbytes/nb_per_cpy; ++i) {
        यदि constexpr (nb_per_cpy == 1) {
            ((वर्ण *) dst)[i] = ((ध्रुव वर्ण *) src)[i];
        } अन्यथा यदि constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((ध्रुव short *) src)[i];
        } अन्यथा यदि constexpr (nb_per_cpy == 4) {
            ((पूर्णाङ्क *) dst)[i] = ((ध्रुव पूर्णाङ्क *) src)[i];
        } अन्यथा यदि constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ध्रुव sycl::int2 *) src)[i];
        } अन्यथा यदि constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ध्रुव sycl::int4 *) src)[i];
        } अन्यथा {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    प्रत्यागम res;
}

स्थिर __dpct_inline__ uint32_t fastdiv(uint32_t n, ध्रुव sycl::uint3 fastdiv_मूल्यावली) {
    ध्रुव uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_मूल्यावली.x());
    प्रत्यागम (hi + n) >> fastdiv_मूल्यावली.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<भिन्न>(x),static_cast<भिन्न>(y));
    प्रत्यागम res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_मूल्य(static_cast<भिन्न>(H.x()), static_cast<भिन्न>(H.y()));
    प्रत्यागम float2_मूल्य;
}

स्थिर __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ध्रुव sycl::uint3 fastdiv_मूल्यावली) {
    ध्रुव uint32_t div_val = fastdiv(n, fastdiv_मूल्यावली);
    ध्रुव uint32_t mod_val = n - div_val * fastdiv_मूल्यावली.z();
    प्रत्यागम sycl::uint2(div_val, mod_val);
}

स्थिर __dpct_inline__ पूर्णाङ्क ggml_sycl_dp4a(ध्रुव पूर्णाङ्क a, ध्रुव पूर्णाङ्क b, पूर्णाङ्क c) {
    प्रत्यागम dpct::dp4a(a, b, c);
}

स्थिर __dpct_inline__ भिन्न ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    यदि (x == 0) {
        bits = 0x00400000;
    } अन्यथा {
        bits = (uint32_t) x << 23;
    }

    भिन्न result;
    memcpy(&result, &bits, परिमाण(भिन्न));
    प्रत्यागम result;
}

sycl::float2 __dpct_inline__ __half22float2(ध्रुव sycl::half2 &H) {
    sycl::float2 float2_मूल्य(static_cast<भिन्न>(H.x()), static_cast<भिन्न>(H.y()));
    प्रत्यागम float2_मूल्य;
}

भिन्न __dpct_inline__ __half2float(sycl::half H) {
    प्रत्यागम static_cast<भिन्न>(H);
}

स्थिर __dpct_inline__ शून्य ggml_sycl_mad(भिन्न & acc, ध्रुव भिन्न v, ध्रुव भिन्न u) {
    acc += v*u;
}

स्थिर __dpct_inline__ शून्य ggml_sycl_mad(भिन्न & acc, ध्रुव sycl::float2 v, ध्रुव sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

स्थिर __dpct_inline__ शून्य ggml_sycl_mad(भिन्न & acc, ध्रुव sycl::half2 v, ध्रुव sycl::half2 u) {
#ifdef GGML_SYCL_F16
    ध्रुव sycl::float2 tmp = (v * u).template convert<भिन्न, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#अन्यथा
    ध्रुव sycl::float2 tmpv = __half22float2(v);
    ध्रुव sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य ggml_sycl_mad(sycl::half2 & acc, ध्रुव sycl::half2 v, ध्रुव sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#अन्यथा
    ध्रुव sycl::float2 tmpv = __half22float2(v);
    ध्रुव sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <पूर्णाङ्क n>
संरचना ggml_sycl_unroll {
    template <typename Func, typename... Args>
    शून्य operator()(ध्रुव Func & f, Args... args) ध्रुव {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
संरचना ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    शून्य operator()(ध्रुव Func & f, Args... args) ध्रुव {
        f(0, args...);
    }
};

स्थिर __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ध्रुव sycl::half2 a, ध्रुव sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<भिन्न, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<भिन्न, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    प्रत्यागम ret;
}

स्थिर __dpct_inline__ sycl::half ggml_sycl_hmax(ध्रुव sycl::half a, ध्रुव sycl::half b) {
    प्रत्यागम sycl::vec<भिन्न, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<भिन्न, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<भिन्न, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

स्थिर __dpct_inline__ uint32_t __hgt2_mask(ध्रुव sycl::half2 a, ध्रुव sycl::half2 b) {
    ध्रुव uint32_t mask_निचला  = 0x0000FFFF * (भिन्न(a[0]) > भिन्न(b[0]));
    ध्रुव uint32_t mask_ऊपरी = 0xFFFF0000 * (भिन्न(a[1]) > भिन्न(b[1]));
    प्रत्यागम mask_निचला | mask_ऊपरी;
}

स्थिर __dpct_inline__ uint32_t fastmodulo(uint32_t n, ध्रुव sycl::uint3 fastdiv_मूल्यावली) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    प्रत्यागम n - fastdiv(n, fastdiv_मूल्यावली) * fastdiv_मूल्यावली.z();
}

स्थिर तार्किक fast_fp16_available(ध्रुव पूर्णाङ्क cc) {
    GGML_UNUSED(cc);
    प्रत्यागम सत्य;   //Intel GPUs always support FP16.
}

गणना वर्ग block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, पूर्णाङ्क warp_size>
संरचना block_reduce_policy;

template <typename T, typename... Ts>
अन्तःस्थ constexpr तार्किक is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
अन्तःस्थ constexpr तार्किक ggml_sycl_dependent_false_v = असत्य;

#define WARP_32_SIZE 32

template <typename T, पूर्णाङ्क warp_size> संरचना block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    स्थिर T reduce(T val) {
        यदि constexpr (is_any<T, भिन्न, sycl::float2, sycl::half2, पूर्णाङ्क>) {
            प्रत्यागम warp_reduce_sum<warp_size>(val);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    स्थिर T sentinel() {
        यदि constexpr (std::is_same_v<T, भिन्न>) {
            प्रत्यागम 0.0f;
        } अन्यथा यदि constexpr (std::is_same_v<T, sycl::float2>) {
            प्रत्यागम sycl::float2(0.0f, 0.0f);
        } अन्यथा यदि constexpr (std::is_same_v<T, sycl::half2>) {
            प्रत्यागम sycl::half2(0.0f, 0.0f);
        } अन्यथा यदि constexpr (std::is_same_v<T, पूर्णाङ्क>) {
            प्रत्यागम 0;
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, पूर्णाङ्क warp_size> संरचना block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    स्थिर T reduce(T val) {
        यदि constexpr (is_any<T, भिन्न, sycl::half2>) {
            प्रत्यागम warp_reduce_max<warp_size>(val);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    स्थिर T sentinel() {
        यदि constexpr (std::is_same_v<T, भिन्न>) {
            प्रत्यागम -INFINITY;
        } अन्यथा यदि constexpr (std::is_same_v<T, sycl::half2>) {
            प्रत्यागम sycl::half2(-INFINITY, -INFINITY);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, पूर्णाङ्क warp_size, typename T>
स्थिर T block_reduce(T val, T * shared_vals, पूर्णाङ्क block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    ध्रुव पूर्णाङ्क block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    ध्रुव पूर्णाङ्क nthreads = item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_SIZE;

    यदि (block_size > warp_size) {
        प्रतिज्ञा((block_size <= 1024) && (block_size % warp_size) == 0);
        ध्रुव पूर्णाङ्क warp_id = item_ct1.get_local_id(2) / warp_size;
        ध्रुव पूर्णाङ्क lane_id = item_ct1.get_local_id(2) % warp_size;
        यदि (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        भिन्न tmp = 0.f;
        यदि (lane_id < (static_cast<पूर्णाङ्क>(block_size) / warp_size)) {
            प्रति (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        प्रत्यागम block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    प्रत्यागम val;
}

स्थिर __dpct_inline__ भिन्न ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    यदि (x == 0 || x == 0x7F) {
        प्रत्यागम 0.0f;
    }
    ध्रुव पूर्णाङ्क exp = (x >> 3) & 0xF;
    ध्रुव पूर्णाङ्क man = x & 0x7;
    भिन्न raw;
    यदि (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } अन्यथा {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (भिन्न) exp - 7.0f);
    }
    प्रत्यागम raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP
