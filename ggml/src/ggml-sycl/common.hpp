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

#که_نه_تعریف_شوی GGML_SYCL_COMMON_HPP
#تعریف_کړه GGML_SYCL_COMMON_HPP

#ضمن <cstddef>
#ضمن <fstream>
#ضمن <iostream>
#ضمن <string>

#ضمن "base.hpp"
#ضمن "dpct/helper.hpp"
#ضمن "ggml.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-sycl.h"
#ضمن "presets.hpp"
#ضمن "type.hpp"
#ضمن "sycl_hw.hpp"
#ضمن "memtrace.hpp"

نطاق syclexp = sycl::ext::oneapi::experimental;

#إذا defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #ضمن <sycl/ext/oneapi/bfloat16.hpp>
    #که_نه_تعریف_شوی GGML_SYCL_HAS_BF16
        #تعریف_کړه GGML_SYCL_HAS_BF16
    #د_که_پای
#د_که_پای

#إذا defined(__INTEL_LLVM_COMPILER)
    #تعریف_کړه GGML_SYCL_DMMV_HAS_ESIMD
#د_که_پای

#إذا GGML_SYCL_DNNL
#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"
#د_که_پای

#تعریف_کړه GGML_COMMON_DECL_SYCL
#تعریف_کړه GGML_COMMON_IMPL_SYCL
#تعریف_کړه SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#تعریف_کړه SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#تعریف_کړه GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#لارښوونه clang diagnostic ټېل
#لارښوونه clang diagnostic ignored "-Wnested-anon-types"
#ضمن "ggml-common.h"
#لارښوونه clang diagnostic راوباسه
#ضمن "ggml-impl.h"

فراغ* ggml_sycl_host_malloc(size_t size);
فراغ ggml_sycl_host_free(فراغ* ptr);


خارجي صحيح g_ggml_sycl_debug;
خارجي صحيح g_ggml_sycl_enable_optimize;
خارجي صحيح g_ggml_sycl_enable_fusion;
خارجي صحيح g_ggml_sycl_enable_esimd;
خارجي صحيح g_ggml_sycl_mmvq_wide;
خارجي صحيح g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
تعداد ggml_sycl_xmx_gather_type {
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
ساكن constexpr صحيح GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
خارجي صحيح g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ساكن constexpr صحيح GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
خارجي صحيح g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
تعداد ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#که_تعریف_شوی GGML_SYCL_F16
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#وإلا
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#د_که_پای
خارجي صحيح g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
خارجي صحيح g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ساكن مضمن int32_t ggml_sycl_src1_prec(ثابت ggml_tensor * dst) {
    أرجع g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ساكن مضمن منطقي ggml_sycl_src1_f16_ok(ثابت ggml_tensor * dst) {
    ثابت int32_t src1_prec = ggml_sycl_src1_prec(dst);
    أرجع g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
خارجي صحيح g_ggml_sycl_enable_flash_attention;
خارجي صحيح g_ggml_sycl_dev2dev_memcpy;
خارجي صحيح g_ggml_sycl_fa_onednn;
خارجي صحيح g_ggml_sycl_fa_onednn_max_kv;
خارجي صحيح g_ggml_sycl_enable_mkl_fa;
خارجي صحيح g_ggml_sycl_memtrace;
خارجي صحيح g_ggml_sycl_memtrace_ګام;


#تعریف_کړه CHECK_TRY_ERROR(بیان)                                            \
  [&]() {                                                                \
    حاول {                                                                \
      بیان;                                                              \
      أرجع dpct::success;                                              \
    } امسك (std::exception ثابت& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::کرښه_پای;                                            \
      أرجع dpct::default_error;                                        \
    }                                                                    \
  }()


#تعریف_کړه __SYCL_ARCH__ DPCT_COMPATIBILITY_لنډمهاله
#تعریف_کړه VER_4VEC 610 // todo for hardware optimize.
#تعریف_کړه VER_GEN9 700 // todo for hardware optimize.
#تعریف_کړه VER_GEN12 1000000 // todo for hardware optimize.
#تعریف_کړه VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#تعریف_کړه GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#إذا !defined(GGML_SYCL_FORCE_MMQ)
    #تعریف_کړه SYCL_USE_XMX
#د_که_پای

// max batch size to use MMQ kernels when tensor cores are available
#تعریف_کړه MMQ_MAX_BATCH_اندازه 32

// dmmv = dequantize_mul_mat_vec
#که_نه_تعریف_شوی GGML_SYCL_DMMV_X
#تعریف_کړه GGML_SYCL_DMMV_X 32
#د_که_پای
#که_نه_تعریف_شوی GGML_SYCL_MMV_Y
#تعریف_کړه GGML_SYCL_MMV_Y 1
#د_که_پای

تعريفنوع sycl::queue *کتار_ptr;

تعداد ggml_sycl_backend_gpu_ډېرپېښ {
  SYCL_UNSET_GPU_ډېرپېښ = -1,
  SYCL_SINGLE_GPU_ډېرپېښ = 0,
  SYCL_MUL_GPU_ډېرپېښ
};

تعداد ggml_sycl_dev2dev_memcpy_ډېرپېښ {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(حجم(sycl::half) == حجم(ggml_fp16_t), "wrong fp16 size");

ساكن فراغ crash() {
  صحيح* ptr = عدم;
  *ptr = 0;
}

[[noreturn]] ساكن فراغ ggml_sycl_error(
    ثابت حرف* stmt,
    ثابت حرف* func,
    ثابت حرف* file,
    ثابت صحيح کرښه,
    ثابت حرف* پیغام_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, پیغام_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, کرښه);
  GGML_ABORT("SYCL error");
}

#تعریف_کړه SYCL_CHECK(err)                                                                                    \
    افعل {                                                                                                   \
        خپلکاری err_ = (err);                                                                                 \
        إذا (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } طالما (0)

#إذا DPCT_COMPAT_RT_نسخه >= 11100
#تعریف_کړه GGML_SYCL_ASSUME(x) __builtin_assume(x)
#وإلا
#تعریف_کړه GGML_SYCL_ASSUME(x)
#د_که_پای // DPCT_COMPAT_RT_VERSION >= 11100

#که_تعریف_شوی GGML_SYCL_F16
تعريفنوع sycl::half dfloat; // dequantize float
تعريفنوع sycl::half2 dfloat2;
#وإلا
تعريفنوع عائم dfloat; // dequantize float
تعريفنوع sycl::float2 dfloat2;
#د_که_پای // GGML_SYCL_F16

#تعریف_کړه MMVQ_MAX_BATCH_اندازه  8

ساكن صحيح g_all_sycl_device_شمېر = -1;
ساكن منطقي g_ggml_backend_sycl_بفر_type_initialized = خطأ;

ساكن ggml_sycl_backend_gpu_ډېرپېښ g_ggml_sycl_backend_gpu_ډېرپېښ =
    SYCL_UNSET_GPU_ډېرپېښ;

ساكن فراغ* g_scratch_بفر = تش_اشاره;
ساكن size_t g_scratch_اندازه = 0; // disabled by default
ساكن size_t g_scratch_دننه_واټن = 0;

[[noreturn]] ساكن مضمن فراغ bad_arch(ثابت sycl::stream& بهیر_ct1) {
  بهیر_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::وتل(1);

  (فراغ)bad_arch; // suppress unused function warning
}

صحيح get_current_device_id();

مضمن صحيح ggml_sycl_get_device() {
    أرجع get_current_device_id();
}

مضمن dpct::err0 ggml_sycl_سېټ_device(ثابت صحيح device) حاول {
  صحيح current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  إذا (device == current_device_id) {
    أرجع 0;
  }

  أرجع CHECK_TRY_ERROR(dpct::select_device(device));
} امسك (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::کرښه_پای;
  crash();
  std::وتل(1);
}

//////////////////////
بنية optimize_feature {
    منطقي reorder=خطأ;
    منطقي onednn_optimized_gemm=خطأ;
};

بنية sycl_device_info {
    صحيح cc;  // compute capability
    صحيح nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    صحيح warp_اندازه;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    صحيح max_wg_فی_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    منطقي    vmm;                // virtual memory support
    منطقي    l0_device_type_valid;
    منطقي    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  ټول_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    منطقي    usm_system_support; // support for USM system allocations
};


بنية ggml_sycl_device_info {
    صحيح device_شمېر;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<عائم, GGML_SYCL_MAX_DEVICES> default_tensor_وېشل = {};

    صحيح max_work_ډله_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    منطقي ext_oneapi_level_zero = ریښتیا; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ثابت ggml_sycl_device_info & ggml_sycl_info();

ساكن constexpr size_t SYCL_بفر_ALIGNMENT = 128;

بنية ggml_sycl_pool {
    مجازي ~ggml_sycl_pool() = افتراضي;

    مجازي فراغ * alloc(size_t size, size_t * actual_اندازه) = 0;
    مجازي فراغ حرر(فراغ * ptr, size_t size) = 0;
};

کينډۍ<ډول_نوم T>
بنية ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = تش_اشاره;
    T * ptr = تش_اشاره;
    size_t actual_اندازه = 0;

    څرګند ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        إذا (ptr != تش_اشاره) {
            pool->حرر(ptr, actual_اندازه);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != تش_اشاره);
        إذا (ptr)
            pool->حرر(ptr, actual_اندازه);
        ptr = (T *) pool->alloc(size * حجم(T), &هذا->actual_اندازه);
        أرجع ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != تش_اشاره);
        GGML_ASSERT(ptr == تش_اشاره);
        ptr = (T *) pool->alloc(size * حجم(T), &هذا->actual_اندازه);
        أرجع ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        هذا->pool = &pool;
        أرجع alloc(size);
    }

    T * get() {
        أرجع ptr;
    }

    ggml_sycl_pool_alloc() = افتراضي;
    ggml_sycl_pool_alloc(ثابت ggml_sycl_pool_alloc &) = ړنګول;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = ړنګول;
    ggml_sycl_pool_alloc& چلوونکی=(ثابت ggml_sycl_pool_alloc &) = ړنګول;
    ggml_sycl_pool_alloc& چلوونکی=(ggml_sycl_pool_alloc &&) = ړنګول;
};

// backend interface

بنية ggml_tensor_extra_gpu {
  فراغ* معلومات_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

خارجي صحيح g_ggml_sycl_use_level_zero_api;
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_مستقیم);
فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q);

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<کتار_ptr> streams={});

بنية mmid_لیکه_mapping {
    int32_t i1;
    int32_t i2;
};

بنية ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

نطاق sycl_ex = sycl::ext::oneapi::experimental;
بنية ggml_backend_sycl_context {
    صحيح device;
    std::string name;
    optimize_feature opt_feature;

    کتار_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { تش_اشاره } };

    څرګند ggml_backend_sycl_context(صحيح device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    کتار_ptr stream(صحيح device, صحيح stream) {
        إذا (qptrs[device][stream] == تش_اشاره) {
            qptrs[device][stream] = &(dpct::get_device(device).default_کتار());
        }
        أرجع qptrs[device][stream];
    }

    کتار_ptr stream() {
        أرجع stream(device, 0);
    }

#إذا GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ثابت dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        أرجع eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> بهیر_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream بهیر_dnnl(صحيح device, صحيح بهیر_2) {
        خپلکاری q = stream(device, بهیر_2);
        أرجع بهیر_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        خپلکاری it = engine_map.find(qptr);
        إذا (it == engine_map.end()) {
            خپلکاری eng = make_engine(qptr);
            engine_map[qptr] = eng;
            أرجع eng;
        }
        وإلا
        {
            أرجع it->second;
        }
    }
    dnnl::stream بهیر_dnnl(sycl::queue* qptr) {
        خپلکاری it = بهیر_map.find(qptr);
        إذا (it == بهیر_map.end()) {
            خپلکاری eng = engine_dnnl(qptr);
            خپلکاری stream = dnnl::sycl_interop::make_بهیر(eng, *qptr);
            بهیر_map[qptr] = stream;
            أرجع stream;
        }
        وإلا
        {
            أرجع it->second;
        }
    }
    dnnl::stream بهیر_dnnl() {
        أرجع بهیر_dnnl(device, 0);
    }
#د_که_پای

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_لیکه_mapping> mmid_لیکه_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_مهالوېش_host;

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_device(کتار_ptr qptr, صحيح device);

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_host(کتار_ptr qptr, صحيح device);

    ggml_sycl_pool & pool(صحيح device) {
        إذا (pools[device] == تش_اشاره) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        أرجع *pools[device];
    }

    ggml_sycl_pool & pool() {
        أرجع pool(device);
    }

#که_تعریف_شوی GGML_SYCL_ګراف
    std::unique_ptr<sycl_ex::command_ګراف<sycl_ex::ګراف_state::executable>> exec_ګراف = تش_اشاره;
#د_که_پای

    ggml_sycl_pool & host_pool(صحيح device) {
        إذا (host_pools[device] == تش_اشاره) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        أرجع *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { أرجع host_pool(device); }
};

// common device functions

ساكن __dpct_inline__ عائم warp_reduce_مجموعه(عائم x,
    ثابت sycl::nd_توکی<3>& توکی_ct1) {
#لارښوونه unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        x += dpct::ترتیب_بدلول_sub_ډله_by_xor(توکی_ct1.get_sub_ډله(), x, ماسک);
    }
    أرجع x;
}

ساكن __dpct_inline__ sycl::float2
warp_reduce_مجموعه(sycl::float2 a, ثابت sycl::nd_توکی<3>& توکی_ct1) {
#لارښوونه unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        a.x() += dpct::ترتیب_بدلول_sub_ډله_by_xor(توکی_ct1.get_sub_ډله(), a.x(),
            ماسک);
        a.y() += dpct::ترتیب_بدلول_sub_ډله_by_xor(توکی_ct1.get_sub_ډله(), a.y(),
            ماسک);
    }
    أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ صحيح warp_reduce_مجموعه(صحيح x) {
  أرجع sycl::reduce_over_ډله(
      sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ عائم warp_reduce_مجموعه(عائم x) {
#لارښوونه unroll
  لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
    x += dpct::ترتیب_بدلول_sub_ډله_by_xor(
        sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), x, دننه_واټن, پلنوالی);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ عائم warp_reduce_مجموعه(عائم x, ثابت sycl::nd_توکی<3>& توکی_ct1) {
#لارښوونه unroll
  لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
    x += dpct::ترتیب_بدلول_sub_ډله_by_xor(
        توکی_ct1.get_sub_ډله(), x, دننه_واټن);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ sycl::float2 warp_reduce_مجموعه(sycl::float2 a) {
#لارښوونه unroll
  لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
    a.x() += dpct::ترتیب_بدلول_sub_ډله_by_xor(
        sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), a.x(), دننه_واټن,
        پلنوالی);
    a.y() += dpct::ترتیب_بدلول_sub_ډله_by_xor(
        sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), a.y(), دننه_واټن,
        پلنوالی);
  }
  أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ sycl::half2 warp_reduce_مجموعه(sycl::half2 a) {
#لارښوونه unroll
  لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
    a = a + dpct::ترتیب_بدلول_sub_ډله_by_xor(
                sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), a, دننه_واټن,
                پلنوالی);
  }
  أرجع a;
}

ساكن constexpr صحيح ggml_sycl_get_physical_warp_اندازه() {
  // todo: for old iGPU + dGPU case, need to be changed.
  أرجع WARP_اندازه;
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ صحيح warp_reduce_all(صحيح x) {
    إذا (پلنوالی == ggml_sycl_get_physical_warp_اندازه()) {
        أرجع sycl::all_د_ډله(
            sycl::ext::oneapi::this_work_توکی::get_sub_ډله(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_توکی::get_sub_ډله()
                         .get_local_خطي_id())) ||
                x);
    } وإلا {
#لارښوونه unroll
        لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
            x = dpct::ترتیب_بدلول_sub_ډله_by_xor(
                    sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), x,
                    دننه_واټن, پلنوالی) &&
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ صحيح warp_reduce_any(صحيح x) {
    إذا (پلنوالی == ggml_sycl_get_physical_warp_اندازه()) {
        أرجع sycl::any_د_ډله(
            sycl::ext::oneapi::this_work_توکی::get_sub_ډله(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_توکی::get_sub_ډله()
                         .get_local_خطي_id())) &&
                x);
    } وإلا {
#لارښوونه unroll
        لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
            x = dpct::ترتیب_بدلول_sub_ډله_by_xor(
                    sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), x,
                    دننه_واټن, پلنوالی) ||
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
کينډۍ <صحيح پلنوالی>
ساكن __dpct_inline__ عائم warp_reduce_max(عائم x) {
#لارښوونه unroll
  لكل (صحيح دننه_واټن = پلنوالی / 2; دننه_واټن > 0; دننه_واټن >>= 1) {
    x = sycl::fmax(x, dpct::ترتیب_بدلول_sub_ډله_by_xor(
                          sycl::ext::oneapi::this_work_توکی::get_sub_ډله(), x,
                          دننه_واټن, پلنوالی));
  }
  أرجع x;
}

ساكن __dpct_inline__ عائم warp_reduce_max(عائم x,
    ثابت sycl::nd_توکی<3>& توکی_ct1) {
#لارښوونه unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        x = sycl::fmax(x, dpct::ترتیب_بدلول_sub_ډله_by_xor(
            توکی_ct1.get_sub_ډله(), x, ماسک));
    }
    أرجع x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
کينډۍ<صحيح N>
__dpct_inline__ size_t calculate_دننه_واټن(ثابت std::array<صحيح, N> & strides, ثابت std::array<صحيح, N> & indices) {
    size_t دننه_واټن = 0;
#لارښوونه unroll
    لكل (صحيح i = 0; i < N; i++) {
        خپلکاری شاخص_i = indices[i];
        دننه_واټن += strides[i] * شاخص_i;
    }
    أرجع دننه_واټن;
}

// Helper for vec loading aligned data
کينډۍ <ډول_نوم Tp, صحيح n>
مضمن sycl::vec<Tp, n> vec_aligned_بارول(ثابت Tp* aligned_ptr) {
    أرجع *reinterpret_cast<ثابت sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
کينډۍ <ډول_نوم Tp, صحيح dim>
ساكن __dpct_inline__ Tp* get_اشاره_ګر(sycl::local_accessor<Tp, dim> acc) {
    أرجع acc.کينډۍ get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_اندازه_3);

constexpr size_t ceil_div(ثابت size_t m, ثابت size_t n) {
    أرجع (m + n - 1) / n;
}

منطقي gpu_has_xmx(sycl::device &dev);

#إذا GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
مضمن منطقي ggml_sycl_dnnl_has_optimized_gemm(صحيح device) {
    أرجع ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#د_که_پای

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val);

کينډۍ <صحيح N, صنف T> std::string debug_get_array_str(ثابت std::string & prefix, ثابت T array[N]) {
    إذا (LIKELY(!g_ggml_sycl_debug)) {
        أرجع "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    لكل (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    إذا constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    أرجع ss.str();
}

مضمن std::string debug_get_tensor_str(ثابت std::string &prefix,
        ثابت ggml_tensor *tensor, ثابت std::string &suffix = "") {
    std::stringstream ss;
    إذا (LIKELY(!g_ggml_sycl_debug)) { أرجع ss.str(); }
    ss << prefix.c_str() << "=";
    إذا (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        إذا (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        إذا (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } وإلا {
        ss << "nullptr";
    }
    ss << suffix;
    أرجع ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
بنية scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ثابت std::string_view & func, ثابت std::string_view & func_suffix, ثابت ggml_tensor * dst,
                         std::size_t عدد_src, ثابت std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        إذا (LIKELY(!g_ggml_sycl_debug)) {
            أرجع;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.معلومات(), func_suffix.معلومات());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        إذا (dst) {
            لكل (std::size_t i = 0; i < عدد_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.معلومات());
    }

    scope_op_debug_print(ثابت std::string_view & func, ثابت ggml_tensor * dst, std::size_t عدد_src,
                         ثابت std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, عدد_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.معلومات(), func_suffix.معلومات()); }

  خاص:
    std::string_view func;
    std::string_view func_suffix;
};

ساكن __dpct_inline__ عائم get_alibi_میلان(ثابت عائم    max_bias,
                                             ثابت uint32_t h,
                                             ثابت uint32_t n_سر_log2,
                                             ثابت عائم    m0,
                                             ثابت عائم    m1) {
    إذا (max_bias <= 0.0f) {
        أرجع 1.0f;
    }
    ثابت عائم بنسټ = h < n_سر_log2 ? m0 : m1;
    ثابت صحيح   exph = h < n_سر_log2 ? h + 1 : 2*(h - n_سر_log2) + 1;

    أرجع dpct::توان(بنسټ, exph);
}

ساكن ثابت sycl::uint3 init_fastdiv_ارزښتونه(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    طالما (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    أرجع sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ساكن constexpr صحيح ggml_sycl_get_max_cpy_بایټونه() {
    أرجع 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
کينډۍ <صحيح nbytes, صحيح alignment = 0>
ساكن __dpct_inline__ فراغ ggml_sycl_memcpy_1(فراغ * dst, ثابت فراغ * src) {
    إذا constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr صحيح nb_فی_cpy = alignment == 0 ? nbytes : alignment;

#لارښوونه unroll
    لكل (صحيح i = 0; i < nbytes/nb_فی_cpy; ++i) {
        إذا constexpr (nb_فی_cpy == 1) {
            ((حرف *) dst)[i] = ((ثابت حرف *) src)[i];
        } وإلا إذا constexpr (nb_فی_cpy == 2) {
            ((لنډ *) dst)[i] = ((ثابت لنډ *) src)[i];
        } وإلا إذا constexpr (nb_فی_cpy == 4) {
            ((صحيح *) dst)[i] = ((ثابت صحيح *) src)[i];
        } وإلا إذا constexpr (nb_فی_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ثابت sycl::int2 *) src)[i];
        } وإلا إذا constexpr (nb_فی_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ثابت sycl::int4 *) src)[i];
        } وإلا {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
کينډۍ <ډول_نوم T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    أرجع res;
}

ساكن __dpct_inline__ uint32_t fastdiv(uint32_t n, ثابت sycl::uint3 fastdiv_ارزښتونه) {
    ثابت uint32_t hi = sycl::mul_hi<بینښه>(n, fastdiv_ارزښتونه.x());
    أرجع (hi + n) >> fastdiv_ارزښتونه.y();
}


کينډۍ <ډول_نوم T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<عائم>(x),static_cast<عائم>(y));
    أرجع res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_ارزښت(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_ارزښت;
}

ساكن __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ثابت sycl::uint3 fastdiv_ارزښتونه) {
    ثابت uint32_t div_val = fastdiv(n, fastdiv_ارزښتونه);
    ثابت uint32_t mod_val = n - div_val * fastdiv_ارزښتونه.z();
    أرجع sycl::uint2(div_val, mod_val);
}

ساكن __dpct_inline__ صحيح ggml_sycl_dp4a(ثابت صحيح a, ثابت صحيح b, صحيح c) {
    أرجع dpct::dp4a(a, b, c);
}

ساكن __dpct_inline__ عائم ggml_sycl_e8m0_منزل_fp32(uint8_t x) {
    uint32_t بټونه;
    إذا (x == 0) {
        بټونه = 0x00400000;
    } وإلا {
        بټونه = (uint32_t) x << 23;
    }

    عائم پایله;
    حافظه_نقل(&پایله, &بټونه, حجم(عائم));
    أرجع پایله;
}

sycl::float2 __dpct_inline__ __half22float2(ثابت sycl::half2 &H) {
    sycl::float2 float2_ارزښت(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_ارزښت;
}

عائم __dpct_inline__ __half2float(sycl::half H) {
    أرجع static_cast<عائم>(H);
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(عائم & acc, ثابت عائم v, ثابت عائم u) {
    acc += v*u;
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(عائم & acc, ثابت sycl::float2 v, ثابت sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(عائم & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#که_تعریف_شوی GGML_SYCL_F16
    ثابت sycl::float2 tmp = (v * u).کينډۍ convert<عائم, sycl::rounding_ډېرپېښ::automatic>();
    acc += tmp.x() + tmp.y();
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#د_که_پای // GGML_SYCL_F16
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(sycl::half2 & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#که_تعریف_شوی GGML_SYCL_F16
    acc += v*u;
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#د_که_پای // GGML_SYCL_F16
}

کينډۍ <صحيح n>
بنية ggml_sycl_unroll {
    کينډۍ <ډول_نوم Func, ډول_نوم... دلیلونه>
    فراغ چلوونکی()(ثابت Func & f, دلیلونه... دلیلونه_2) ثابت {
        f(n - 1, دلیلونه_2...);
        ggml_sycl_unroll<n - 1>{}(f, دلیلونه_2...);
    }
};

کينډۍ <>
بنية ggml_sycl_unroll<1> {
    کينډۍ <ډول_نوم Func, ډول_نوم... دلیلونه>
    فراغ چلوونکی()(ثابت Func & f, دلیلونه... دلیلونه_2) ثابت {
        f(0, دلیلونه_2...);
    }
};

ساكن __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_ډېرپېښ::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_ډېرپېښ::automatic>()[0];
    أرجع ret;
}

ساكن __dpct_inline__ sycl::half ggml_sycl_hmax(ثابت sycl::half a, ثابت sycl::half b) {
    أرجع sycl::vec<عائم, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<عائم, sycl::rounding_ډېرپېښ::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<عائم, sycl::rounding_ډېرپېښ::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_ډېرپېښ::automatic>()[0];
}

ساكن __dpct_inline__ uint32_t __hgt2_mask(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    ثابت uint32_t ماسک_ټیټ  = 0x0000FFFF * (عائم(a[0]) > عائم(b[0]));
    ثابت uint32_t ماسک_لوړ = 0xFFFF0000 * (عائم(a[1]) > عائم(b[1]));
    أرجع ماسک_ټیټ | ماسک_لوړ;
}

ساكن __dpct_inline__ uint32_t fastmodulo(uint32_t n, ثابت sycl::uint3 fastdiv_ارزښتونه) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    أرجع n - fastdiv(n, fastdiv_ارزښتونه) * fastdiv_ارزښتونه.z();
}

ساكن منطقي fast_fp16_شته(ثابت صحيح cc) {
    GGML_ناکارول_شوی(cc);
    أرجع ریښتیا;   //Intel GPUs always support FP16.
}

تعداد صنف بلاک_reduce_لاره {
    MAX,
    مجموعه_3,
};

کينډۍ<بلاک_reduce_لاره لاره_t, ډول_نوم T, صحيح warp_اندازه>
بنية بلاک_reduce_policy;

کينډۍ <ډول_نوم T, ډول_نوم... Ts>
مضمن constexpr منطقي is_any = (std::is_same_v<T, Ts> || ...);

کينډۍ<ډول_نوم...>
مضمن constexpr منطقي ggml_sycl_dependent_false_v = خطأ;

#تعریف_کړه WARP_32_اندازه 32

کينډۍ <ډول_نوم T, صحيح warp_اندازه> بنية بلاک_reduce_policy<بلاک_reduce_لاره::مجموعه_3, T, warp_اندازه> {
    ساكن T reduce(T val) {
        إذا constexpr (is_any<T, عائم, sycl::float2, sycl::half2, صحيح>) {
            أرجع warp_reduce_مجموعه<warp_اندازه>(val);
        } وإلا {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ساكن T sentinel() {
        إذا constexpr (std::is_same_v<T, عائم>) {
            أرجع 0.0f;
        } وإلا إذا constexpr (std::is_same_v<T, sycl::float2>) {
            أرجع sycl::float2(0.0f, 0.0f);
        } وإلا إذا constexpr (std::is_same_v<T, sycl::half2>) {
            أرجع sycl::half2(0.0f, 0.0f);
        } وإلا إذا constexpr (std::is_same_v<T, صحيح>) {
            أرجع 0;
        } وإلا {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

کينډۍ <ډول_نوم T, صحيح warp_اندازه> بنية بلاک_reduce_policy<بلاک_reduce_لاره::MAX, T, warp_اندازه> {
    ساكن T reduce(T val) {
        إذا constexpr (is_any<T, عائم, sycl::half2>) {
            أرجع warp_reduce_max<warp_اندازه>(val);
        } وإلا {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ساكن T sentinel() {
        إذا constexpr (std::is_same_v<T, عائم>) {
            أرجع -INFINITY;
        } وإلا إذا constexpr (std::is_same_v<T, sycl::half2>) {
            أرجع sycl::half2(-INFINITY, -INFINITY);
        } وإلا {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


کينډۍ <بلاک_reduce_لاره reduce_لاره_t, صحيح warp_اندازه, ډول_نوم T>
ساكن T بلاک_reduce(T val, T * shared_vals, صحيح بلاک_اندازه_template) {
    خپلکاری توکی_ct1                 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<3>();
    val                           = بلاک_reduce_policy<reduce_لاره_t, T,warp_اندازه>::reduce(val);
    ثابت صحيح بلاک_اندازه_3 = بلاک_اندازه_template == 0 ? توکی_ct1.get_local_range(2) : بلاک_اندازه_template;
    ثابت صحيح nthreads = توکی_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    إذا (بلاک_اندازه_3 > warp_اندازه) {
        أكد((بلاک_اندازه_3 <= 1024) && (بلاک_اندازه_3 % warp_اندازه) == 0);
        ثابت صحيح warp_id = توکی_ct1.get_local_id(2) / warp_اندازه;
        ثابت صحيح lane_id = توکی_ct1.get_local_id(2) % warp_اندازه;
        إذا (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        توکی_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_اندازه;
        عائم tmp = 0.f;
        إذا (lane_id < (static_cast<صحيح>(بلاک_اندازه_3) / warp_اندازه)) {
            لكل (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_اندازه];
            }
        }
        أرجع بلاک_reduce_policy<reduce_لاره_t, T, warp_اندازه>::reduce(tmp);
    }
    أرجع val;
}

ساكن __dpct_inline__ عائم ggml_sycl_ue4m3_منزل_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    إذا (x == 0 || x == 0x7F) {
        أرجع 0.0f;
    }
    ثابت صحيح توانیز_تابع = (x >> 3) & 0xF;
    ثابت صحيح man = x & 0x7;
    عائم raw;
    إذا (توانیز_تابع == 0) {
        raw = man * (1.0f / 8.0f) * sycl::توان(2.0f, -6.0f);
    } وإلا {
        raw = (1.0f + man / 8.0f) * sycl::توان(2.0f, (عائم) توانیز_تابع - 7.0f);
    }
    أرجع raw * 0.5f;
}

#د_که_پای // GGML_SYCL_COMMON_HPP
