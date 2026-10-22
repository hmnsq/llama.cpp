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

#अगर_अघोषित GGML_SYCL_COMMON_HPP
#घोषणा GGML_SYCL_COMMON_HPP

#समावेश <cstddef>
#समावेश <fstream>
#समावेश <iostream>
#समावेश <string>

#समावेश "base.hpp"
#समावेश "dpct/helper.hpp"
#समावेश "ggml.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-sycl.h"
#समावेश "presets.hpp"
#समावेश "type.hpp"
#समावेश "sycl_hw.hpp"
#समावेश "memtrace.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

#अगर defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #समावेश <sycl/ext/oneapi/bfloat16.hpp>
    #अगर_अघोषित GGML_SYCL_HAS_BF16
        #घोषणा GGML_SYCL_HAS_BF16
    #पूर्ण_अगर
#पूर्ण_अगर

#अगर defined(__INTEL_LLVM_COMPILER)
    #घोषणा GGML_SYCL_DMMV_HAS_ESIMD
#पूर्ण_अगर

#अगर GGML_SYCL_DNNL
#समावेश "dnnl.hpp"
#समावेश "dnnl_sycl.hpp"
#पूर्ण_अगर

#घोषणा GGML_COMMON_DECL_SYCL
#घोषणा GGML_COMMON_IMPL_SYCL
#घोषणा SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#घोषणा SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#घोषणा GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#आशय clang diagnostic धकेलो
#आशय clang diagnostic ignored "-Wnested-anon-types"
#समावेश "ggml-common.h"
#आशय clang diagnostic निकालो
#समावेश "ggml-impl.h"

व्योम* ggml_sycl_host_malloc(माप_प्रकार size);
व्योम ggml_sycl_host_free(व्योम* ptr);


बाह्य पूर्णांक g_ggml_sycl_debug;
बाह्य पूर्णांक g_ggml_sycl_enable_optimize;
बाह्य पूर्णांक g_ggml_sycl_enable_fusion;
बाह्य पूर्णांक g_ggml_sycl_enable_esimd;
बाह्य पूर्णांक g_ggml_sycl_mmvq_wide;
बाह्य पूर्णांक g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
क्रमागत ggml_sycl_xmx_gather_type {
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
अटल constexpr पूर्णांक GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
बाह्य पूर्णांक g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
अटल constexpr पूर्णांक GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
बाह्य पूर्णांक g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
क्रमागत ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#अगर_घोषित GGML_SYCL_F16
अटल constexpr पूर्णांक GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#अन्यथा
अटल constexpr पूर्णांक GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#पूर्ण_अगर
बाह्य पूर्णांक g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
बाह्य पूर्णांक g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
अटल अंतरभूत int32_t ggml_sycl_src1_prec(स्थिर ggml_tensor * dst) {
    वापस g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
अटल अंतरभूत bool ggml_sycl_src1_f16_ok(स्थिर ggml_tensor * dst) {
    स्थिर int32_t src1_prec = ggml_sycl_src1_prec(dst);
    वापस g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
बाह्य पूर्णांक g_ggml_sycl_enable_flash_attention;
बाह्य पूर्णांक g_ggml_sycl_dev2dev_memcpy;
बाह्य पूर्णांक g_ggml_sycl_fa_onednn;
बाह्य पूर्णांक g_ggml_sycl_fa_onednn_max_kv;
बाह्य पूर्णांक g_ggml_sycl_enable_mkl_fa;
बाह्य पूर्णांक g_ggml_sycl_memtrace;
बाह्य पूर्णांक g_ggml_sycl_memtrace_step;


#घोषणा CHECK_TRY_ERROR(व्यंजक)                                            \
  [&]() {                                                                \
    प्रयास {                                                                \
      व्यंजक;                                                              \
      वापस dpct::success;                                              \
    } पकड़ो (std::exception स्थिर& e) {                                  \
      std::अ_त्रुटि << e.what() << "\nException caught at file:" << __खाता__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::प_पूर्ण;                                            \
      वापस dpct::default_error;                                        \
    }                                                                    \
  }()


#घोषणा __SYCL_ARCH__ DPCT_COMPATIBILITY_अस्थायी
#घोषणा VER_4VEC 610 // todo for hardware optimize.
#घोषणा VER_GEN9 700 // todo for hardware optimize.
#घोषणा VER_GEN12 1000000 // todo for hardware optimize.
#घोषणा VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#घोषणा GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#अगर !defined(GGML_SYCL_FORCE_MMQ)
    #घोषणा SYCL_USE_XMX
#पूर्ण_अगर

// max batch size to use MMQ kernels when tensor cores are available
#घोषणा MMQ_MAX_BATCH_आकार 32

// dmmv = dequantize_mul_mat_vec
#अगर_अघोषित GGML_SYCL_DMMV_X
#घोषणा GGML_SYCL_DMMV_X 32
#पूर्ण_अगर
#अगर_अघोषित GGML_SYCL_MMV_Y
#घोषणा GGML_SYCL_MMV_Y 1
#पूर्ण_अगर

प्रकार sycl::queue *queue_ptr;

क्रमागत ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

क्रमागत ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(माप(sycl::half) == माप(ggml_fp16_t), "wrong fp16 size");

अटल व्योम crash() {
  पूर्णांक* ptr = शून्य;
  *ptr = 0;
}

[[noreturn]] अटल व्योम ggml_sycl_error(
    स्थिर अक्षर* stmt,
    स्थिर अक्षर* func,
    स्थिर अक्षर* file,
    स्थिर पूर्णांक पंक्ति,
    स्थिर अक्षर* संदेश) {
  ख_लिखो(मानक_त्रुटि, "SYCL error: %s: %s\n", stmt, संदेश);
  ख_लिखो(मानक_त्रुटि, "  in function %s at %s:%d\n", func, file, पंक्ति);
  GGML_ABORT("SYCL error");
}

#घोषणा SYCL_CHECK(err)                                                                                    \
    करो {                                                                                                   \
        स्वतः err_ = (err);                                                                                 \
        अगर (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __खाता__, __LINE__, "Exception caught in this line of code."); \
    } जबतक (0)

#अगर DPCT_COMPAT_RT_VERSION >= 11100
#घोषणा GGML_SYCL_ASSUME(x) __builtin_assume(x)
#अन्यथा
#घोषणा GGML_SYCL_ASSUME(x)
#पूर्ण_अगर // DPCT_COMPAT_RT_VERSION >= 11100

#अगर_घोषित GGML_SYCL_F16
प्रकार sycl::half dfloat; // dequantize float
प्रकार sycl::half2 dfloat2;
#अन्यथा
प्रकार भग्न dfloat; // dequantize float
प्रकार sycl::float2 dfloat2;
#पूर्ण_अगर // GGML_SYCL_F16

#घोषणा MMVQ_MAX_BATCH_आकार  8

अटल पूर्णांक g_all_sycl_device_गिनती = -1;
अटल bool g_ggml_backend_sycl_बफर_type_initialized = false;

अटल ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

अटल व्योम* g_scratch_बफर = nullptr;
अटल माप_प्रकार g_scratch_आकार = 0; // disabled by default
अटल माप_प्रकार g_scratch_offset = 0;

[[noreturn]] अटल अंतरभूत व्योम bad_arch(स्थिर sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::निकास(1);

  (व्योम)bad_arch; // suppress unused function warning
}

पूर्णांक get_current_device_id();

अंतरभूत पूर्णांक ggml_sycl_get_device() {
    वापस get_current_device_id();
}

अंतरभूत dpct::err0 ggml_sycl_set_device(स्थिर पूर्णांक device) प्रयास {
  पूर्णांक current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  अगर (device == current_device_id) {
    वापस 0;
  }

  वापस CHECK_TRY_ERROR(dpct::select_device(device));
} पकड़ो (sycl::exception स्थिर& exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  crash();
  std::निकास(1);
}

//////////////////////
काष्ठा optimize_feature {
    bool reorder=false;
    bool onednn_optimized_gemm=false;
};

काष्ठा sycl_device_info {
    पूर्णांक cc;  // compute capability
    पूर्णांक nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    माप_प्रकार  smpbo;              // max. shared memory per block (with opt-in)
    पूर्णांक warp_आकार;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    पूर्णांक max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    bool    vmm;                // virtual memory support
    bool    l0_device_type_valid;
    bool    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    माप_प्रकार  vmm_granularity;    // granularity of virtual memory
    माप_प्रकार  योग_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    bool    usm_system_support; // support for USM system allocations
};


काष्ठा ggml_sycl_device_info {
    पूर्णांक device_गिनती;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<भग्न, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    पूर्णांक max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    bool ext_oneapi_level_zero = true; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

स्थिर ggml_sycl_device_info & ggml_sycl_info();

अटल constexpr माप_प्रकार SYCL_बफर_ALIGNMENT = 128;

काष्ठा ggml_sycl_pool {
    भव ~ggml_sycl_pool() = शेष;

    भव व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) = 0;
    भव व्योम मुक्त(व्योम * ptr, माप_प्रकार size) = 0;
};

ढाँचा<typename T>
काष्ठा ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    माप_प्रकार actual_आकार = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, माप_प्रकार size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        अगर (ptr != nullptr) {
            pool->मुक्त(ptr, actual_आकार);
        }
    }

    T * पुनः_स्मृति(माप_प्रकार size) {
        GGML_ASSERT(pool != nullptr);
        अगर (ptr)
            pool->मुक्त(ptr, actual_आकार);
        ptr = (T *) pool->alloc(size * माप(T), &यह->actual_आकार);
        वापस ptr;
    }

    // size is in number of elements
    T * alloc(माप_प्रकार size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * माप(T), &यह->actual_आकार);
        वापस ptr;
    }

    T * alloc(ggml_sycl_pool & pool, माप_प्रकार size) {
        यह->pool = &pool;
        वापस alloc(size);
    }

    T * लो() {
        वापस ptr;
    }

    ggml_sycl_pool_alloc() = शेष;
    ggml_sycl_pool_alloc(स्थिर ggml_sycl_pool_alloc &) = मिटाओ;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = मिटाओ;
    ggml_sycl_pool_alloc& चालक=(स्थिर ggml_sycl_pool_alloc &) = मिटाओ;
    ggml_sycl_pool_alloc& चालक=(ggml_sycl_pool_alloc &&) = मिटाओ;
};

// backend interface

काष्ठा ggml_tensor_extra_gpu {
  व्योम* तथ्य_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

बाह्य पूर्णांक g_ggml_sycl_use_level_zero_api;
व्योम * ggml_sycl_malloc_device(माप_प्रकार size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
व्योम ggml_sycl_free_device(व्योम *ptr, sycl::queue &q);

व्योम release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

काष्ठा mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

काष्ठा ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

namespace sycl_ex = sycl::ext::oneapi::experimental;
काष्ठा ggml_backend_sycl_context {
    पूर्णांक device;
    std::string name;
    optimize_feature opt_feature;

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(पूर्णांक device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    queue_ptr stream(पूर्णांक device, पूर्णांक stream) {
        अगर (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        वापस qptrs[device][stream];
    }

    queue_ptr stream() {
        वापस stream(device, 0);
    }

#अगर GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        स्थिर dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        वापस eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(पूर्णांक device, पूर्णांक _stream) {
        स्वतः q = stream(device, _stream);
        वापस stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        स्वतः it = engine_map.find(qptr);
        अगर (it == engine_map.end()) {
            स्वतः eng = make_engine(qptr);
            engine_map[qptr] = eng;
            वापस eng;
        }
        अन्यथा
        {
            वापस it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        स्वतः it = stream_map.find(qptr);
        अगर (it == stream_map.end()) {
            स्वतः eng = engine_dnnl(qptr);
            स्वतः stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            वापस stream;
        }
        अन्यथा
        {
            वापस it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        वापस stream_dnnl(device, 0);
    }
#पूर्ण_अगर

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_schedule_host;

    अटल std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, पूर्णांक device);

    अटल std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, पूर्णांक device);

    ggml_sycl_pool & pool(पूर्णांक device) {
        अगर (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        वापस *pools[device];
    }

    ggml_sycl_pool & pool() {
        वापस pool(device);
    }

#अगर_घोषित GGML_SYCL_GRAPH
    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
#पूर्ण_अगर

    ggml_sycl_pool & host_pool(पूर्णांक device) {
        अगर (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        वापस *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { वापस host_pool(device); }
};

// common device functions

अटल __dpct_inline__ भग्न warp_reduce_योग(भग्न x,
    स्थिर sycl::nd_item<3>& item_ct1) {
#आशय unroll
    क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    वापस x;
}

अटल __dpct_inline__ sycl::float2
warp_reduce_योग(sycl::float2 a, स्थिर sycl::nd_item<3>& item_ct1) {
#आशय unroll
    क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    वापस a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ पूर्णांक warp_reduce_योग(पूर्णांक x) {
  वापस sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ भग्न warp_reduce_योग(भग्न x) {
#आशय unroll
  क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  वापस x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ भग्न warp_reduce_योग(भग्न x, स्थिर sycl::nd_item<3>& item_ct1) {
#आशय unroll
  क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  वापस x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ sycl::float2 warp_reduce_योग(sycl::float2 a) {
#आशय unroll
  क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  वापस a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ sycl::half2 warp_reduce_योग(sycl::half2 a) {
#आशय unroll
  क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  वापस a;
}

अटल constexpr पूर्णांक ggml_sycl_get_physical_warp_आकार() {
  // todo: for old iGPU + dGPU case, need to be changed.
  वापस WARP_आकार;
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ पूर्णांक warp_reduce_all(पूर्णांक x) {
    अगर (width == ggml_sycl_get_physical_warp_आकार()) {
        वापस sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } अन्यथा {
#आशय unroll
        क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        वापस x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ पूर्णांक warp_reduce_any(पूर्णांक x) {
    अगर (width == ggml_sycl_get_physical_warp_आकार()) {
        वापस sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } अन्यथा {
#आशय unroll
        क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        वापस x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
ढाँचा <पूर्णांक width>
अटल __dpct_inline__ भग्न warp_reduce_max(भग्न x) {
#आशय unroll
  क्रम (पूर्णांक offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  वापस x;
}

अटल __dpct_inline__ भग्न warp_reduce_max(भग्न x,
    स्थिर sycl::nd_item<3>& item_ct1) {
#आशय unroll
    क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    वापस x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
ढाँचा<पूर्णांक N>
__dpct_inline__ माप_प्रकार calculate_offset(स्थिर std::array<पूर्णांक, N> & strides, स्थिर std::array<पूर्णांक, N> & indices) {
    माप_प्रकार offset = 0;
#आशय unroll
    क्रम (पूर्णांक i = 0; i < N; i++) {
        स्वतः सूचकांक_i = indices[i];
        offset += strides[i] * सूचकांक_i;
    }
    वापस offset;
}

// Helper for vec loading aligned data
ढाँचा <typename Tp, पूर्णांक n>
अंतरभूत sycl::vec<Tp, n> vec_aligned_load(स्थिर Tp* aligned_ptr) {
    वापस *reinterpret_cast<स्थिर sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
ढाँचा <typename Tp, पूर्णांक dim>
अटल __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    वापस acc.ढाँचा get_multi_ptr<sycl::access::decorated::no>().लो();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_अंक, int64_t block_आकार_2);

constexpr माप_प्रकार ceil_div(स्थिर माप_प्रकार m, स्थिर माप_प्रकार n) {
    वापस (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);

#अगर GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
अंतरभूत bool ggml_sycl_dnnl_has_optimized_gemm(पूर्णांक device) {
    वापस ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#पूर्ण_अगर

पूर्णांक ggml_sycl_get_env(स्थिर अक्षर *env_name, पूर्णांक default_val);

ढाँचा <पूर्णांक N, श्रेणी T> std::string debug_get_array_str(स्थिर std::string & prefix, स्थिर T array[N]) {
    अगर (LIKELY(!g_ggml_sycl_debug)) {
        वापस "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    क्रम (std::माप_प्रकार i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    अगर constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    वापस ss.str();
}

अंतरभूत std::string debug_get_tensor_str(स्थिर std::string &prefix,
        स्थिर ggml_tensor *tensor, स्थिर std::string &suffix = "") {
    std::stringstream ss;
    अगर (LIKELY(!g_ggml_sycl_debug)) { वापस ss.str(); }
    ss << prefix.c_str() << "=";
    अगर (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        अगर (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        अगर (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } अन्यथा {
        ss << "nullptr";
    }
    ss << suffix;
    वापस ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
काष्ठा scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(स्थिर std::string_view & func, स्थिर std::string_view & func_suffix, स्थिर ggml_tensor * dst,
                         std::माप_प्रकार अंक_src, स्थिर std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        अगर (LIKELY(!g_ggml_sycl_debug)) {
            वापस;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.तथ्य(), func_suffix.तथ्य());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        अगर (dst) {
            क्रम (std::माप_प्रकार i = 0; i < अंक_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.तथ्य());
    }

    scope_op_debug_print(स्थिर std::string_view & func, स्थिर ggml_tensor * dst, std::माप_प्रकार अंक_src,
                         स्थिर std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, अंक_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.तथ्य(), func_suffix.तथ्य()); }

  निजी:
    std::string_view func;
    std::string_view func_suffix;
};

अटल __dpct_inline__ भग्न get_alibi_slope(स्थिर भग्न    max_bias,
                                             स्थिर uint32_t h,
                                             स्थिर uint32_t n_शीर्ष_log2,
                                             स्थिर भग्न    m0,
                                             स्थिर भग्न    m1) {
    अगर (max_bias <= 0.0f) {
        वापस 1.0f;
    }
    स्थिर भग्न base = h < n_शीर्ष_log2 ? m0 : m1;
    स्थिर पूर्णांक   exph = h < n_शीर्ष_log2 ? h + 1 : 2*(h - n_शीर्ष_log2) + 1;

    वापस dpct::घात(base, exph);
}

अटल स्थिर sycl::uint3 init_fastdiv_मूल्यावली(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    जबतक (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    वापस sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
अटल constexpr पूर्णांक ggml_sycl_get_max_cpy_bytes() {
    वापस 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
ढाँचा <पूर्णांक nbytes, पूर्णांक alignment = 0>
अटल __dpct_inline__ व्योम ggml_sycl_memcpy_1(व्योम * dst, स्थिर व्योम * src) {
    अगर constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr पूर्णांक nb_per_cpy = alignment == 0 ? nbytes : alignment;

#आशय unroll
    क्रम (पूर्णांक i = 0; i < nbytes/nb_per_cpy; ++i) {
        अगर constexpr (nb_per_cpy == 1) {
            ((अक्षर *) dst)[i] = ((स्थिर अक्षर *) src)[i];
        } अन्यथा अगर constexpr (nb_per_cpy == 2) {
            ((लघु *) dst)[i] = ((स्थिर लघु *) src)[i];
        } अन्यथा अगर constexpr (nb_per_cpy == 4) {
            ((पूर्णांक *) dst)[i] = ((स्थिर पूर्णांक *) src)[i];
        } अन्यथा अगर constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((स्थिर sycl::int2 *) src)[i];
        } अन्यथा अगर constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((स्थिर sycl::int4 *) src)[i];
        } अन्यथा {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
ढाँचा <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    वापस res;
}

अटल __dpct_inline__ uint32_t fastdiv(uint32_t n, स्थिर sycl::uint3 fastdiv_मूल्यावली) {
    स्थिर uint32_t hi = sycl::mul_hi<अचिन्हित>(n, fastdiv_मूल्यावली.x());
    वापस (hi + n) >> fastdiv_मूल्यावली.y();
}


ढाँचा <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<भग्न>(x),static_cast<भग्न>(y));
    वापस res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_मूल्य(static_cast<भग्न>(H.x()), static_cast<भग्न>(H.y()));
    वापस float2_मूल्य;
}

अटल __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, स्थिर sycl::uint3 fastdiv_मूल्यावली) {
    स्थिर uint32_t div_val = fastdiv(n, fastdiv_मूल्यावली);
    स्थिर uint32_t mod_val = n - div_val * fastdiv_मूल्यावली.z();
    वापस sycl::uint2(div_val, mod_val);
}

अटल __dpct_inline__ पूर्णांक ggml_sycl_dp4a(स्थिर पूर्णांक a, स्थिर पूर्णांक b, पूर्णांक c) {
    वापस dpct::dp4a(a, b, c);
}

अटल __dpct_inline__ भग्न ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    अगर (x == 0) {
        bits = 0x00400000;
    } अन्यथा {
        bits = (uint32_t) x << 23;
    }

    भग्न परिणाम;
    स_नकल(&परिणाम, &bits, माप(भग्न));
    वापस परिणाम;
}

sycl::float2 __dpct_inline__ __half22float2(स्थिर sycl::half2 &H) {
    sycl::float2 float2_मूल्य(static_cast<भग्न>(H.x()), static_cast<भग्न>(H.y()));
    वापस float2_मूल्य;
}

भग्न __dpct_inline__ __half2float(sycl::half H) {
    वापस static_cast<भग्न>(H);
}

अटल __dpct_inline__ व्योम ggml_sycl_mad(भग्न & acc, स्थिर भग्न v, स्थिर भग्न u) {
    acc += v*u;
}

अटल __dpct_inline__ व्योम ggml_sycl_mad(भग्न & acc, स्थिर sycl::float2 v, स्थिर sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

अटल __dpct_inline__ व्योम ggml_sycl_mad(भग्न & acc, स्थिर sycl::half2 v, स्थिर sycl::half2 u) {
#अगर_घोषित GGML_SYCL_F16
    स्थिर sycl::float2 tmp = (v * u).ढाँचा convert<भग्न, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#अन्यथा
    स्थिर sycl::float2 tmpv = __half22float2(v);
    स्थिर sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#पूर्ण_अगर // GGML_SYCL_F16
}

अटल __dpct_inline__ व्योम ggml_sycl_mad(sycl::half2 & acc, स्थिर sycl::half2 v, स्थिर sycl::half2 u) {
#अगर_घोषित GGML_SYCL_F16
    acc += v*u;
#अन्यथा
    स्थिर sycl::float2 tmpv = __half22float2(v);
    स्थिर sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#पूर्ण_अगर // GGML_SYCL_F16
}

ढाँचा <पूर्णांक n>
काष्ठा ggml_sycl_unroll {
    ढाँचा <typename Func, typename... Args>
    व्योम चालक()(स्थिर Func & f, Args... args) स्थिर {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

ढाँचा <>
काष्ठा ggml_sycl_unroll<1> {
    ढाँचा <typename Func, typename... Args>
    व्योम चालक()(स्थिर Func & f, Args... args) स्थिर {
        f(0, args...);
    }
};

अटल __dpct_inline__ sycl::half2 ggml_sycl_hmax2(स्थिर sycl::half2 a, स्थिर sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<भग्न, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<भग्न, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    वापस ret;
}

अटल __dpct_inline__ sycl::half ggml_sycl_hmax(स्थिर sycl::half a, स्थिर sycl::half b) {
    वापस sycl::vec<भग्न, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<भग्न, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<भग्न, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

अटल __dpct_inline__ uint32_t __hgt2_mask(स्थिर sycl::half2 a, स्थिर sycl::half2 b) {
    स्थिर uint32_t mask_निचला  = 0x0000FFFF * (भग्न(a[0]) > भग्न(b[0]));
    स्थिर uint32_t mask_ऊपरी = 0xFFFF0000 * (भग्न(a[1]) > भग्न(b[1]));
    वापस mask_निचला | mask_ऊपरी;
}

अटल __dpct_inline__ uint32_t fastmodulo(uint32_t n, स्थिर sycl::uint3 fastdiv_मूल्यावली) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    वापस n - fastdiv(n, fastdiv_मूल्यावली) * fastdiv_मूल्यावली.z();
}

अटल bool fast_fp16_available(स्थिर पूर्णांक cc) {
    GGML_UNUSED(cc);
    वापस true;   //Intel GPUs always support FP16.
}

क्रमागत श्रेणी block_reduce_method {
    MAX,
    योग_4,
};

ढाँचा<block_reduce_method method_t, typename T, पूर्णांक warp_आकार>
काष्ठा block_reduce_policy;

ढाँचा <typename T, typename... Ts>
अंतरभूत constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

ढाँचा<typename...>
अंतरभूत constexpr bool ggml_sycl_dependent_false_v = false;

#घोषणा WARP_32_आकार 32

ढाँचा <typename T, पूर्णांक warp_आकार> काष्ठा block_reduce_policy<block_reduce_method::योग_4, T, warp_आकार> {
    अटल T reduce(T val) {
        अगर constexpr (is_any<T, भग्न, sycl::float2, sycl::half2, पूर्णांक>) {
            वापस warp_reduce_योग<warp_आकार>(val);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    अटल T sentinel() {
        अगर constexpr (std::is_same_v<T, भग्न>) {
            वापस 0.0f;
        } अन्यथा अगर constexpr (std::is_same_v<T, sycl::float2>) {
            वापस sycl::float2(0.0f, 0.0f);
        } अन्यथा अगर constexpr (std::is_same_v<T, sycl::half2>) {
            वापस sycl::half2(0.0f, 0.0f);
        } अन्यथा अगर constexpr (std::is_same_v<T, पूर्णांक>) {
            वापस 0;
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

ढाँचा <typename T, पूर्णांक warp_आकार> काष्ठा block_reduce_policy<block_reduce_method::MAX, T, warp_आकार> {
    अटल T reduce(T val) {
        अगर constexpr (is_any<T, भग्न, sycl::half2>) {
            वापस warp_reduce_max<warp_आकार>(val);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    अटल T sentinel() {
        अगर constexpr (std::is_same_v<T, भग्न>) {
            वापस -अनन्त;
        } अन्यथा अगर constexpr (std::is_same_v<T, sycl::half2>) {
            वापस sycl::half2(-अनन्त, -अनन्त);
        } अन्यथा {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


ढाँचा <block_reduce_method reduce_method_t, पूर्णांक warp_आकार, typename T>
अटल T block_reduce(T val, T * shared_vals, पूर्णांक block_आकार_template) {
    स्वतः item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_आकार>::reduce(val);
    स्थिर पूर्णांक block_आकार_2 = block_आकार_template == 0 ? item_ct1.get_local_range(2) : block_आकार_template;
    स्थिर पूर्णांक nthreads = item_ct1.get_local_range(2);
    स्थिर पूर्णांक nwarps = nthreads / WARP_आकार;

    अगर (block_आकार_2 > warp_आकार) {
        निश्चित((block_आकार_2 <= 1024) && (block_आकार_2 % warp_आकार) == 0);
        स्थिर पूर्णांक warp_id = item_ct1.get_local_id(2) / warp_आकार;
        स्थिर पूर्णांक lane_id = item_ct1.get_local_id(2) % warp_आकार;
        अगर (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        माप_प्रकार nreduce = nwarps / WARP_आकार;
        भग्न tmp = 0.f;
        अगर (lane_id < (static_cast<पूर्णांक>(block_आकार_2) / warp_आकार)) {
            क्रम (माप_प्रकार i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_आकार];
            }
        }
        वापस block_reduce_policy<reduce_method_t, T, warp_आकार>::reduce(tmp);
    }
    वापस val;
}

अटल __dpct_inline__ भग्न ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    अगर (x == 0 || x == 0x7F) {
        वापस 0.0f;
    }
    स्थिर पूर्णांक exp = (x >> 3) & 0xF;
    स्थिर पूर्णांक man = x & 0x7;
    भग्न raw;
    अगर (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::घात(2.0f, -6.0f);
    } अन्यथा {
        raw = (1.0f + man / 8.0f) * sycl::घात(2.0f, (भग्न) exp - 7.0f);
    }
    वापस raw * 0.5f;
}

#पूर्ण_अगर // GGML_SYCL_COMMON_HPP
