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

#اگر_تعریف_نشده GGML_SYCL_COMMON_HPP
#تعریف_کن GGML_SYCL_COMMON_HPP

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
    #اگر_تعریف_نشده GGML_SYCL_HAS_BF16
        #تعریف_کن GGML_SYCL_HAS_BF16
    #پایان_اگر
#پایان_اگر

#إذا defined(__INTEL_LLVM_COMPILER)
    #تعریف_کن GGML_SYCL_DMMV_HAS_ESIMD
#پایان_اگر

#إذا GGML_SYCL_DNNL
#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"
#پایان_اگر

#تعریف_کن GGML_COMMON_DECL_SYCL
#تعریف_کن GGML_COMMON_IMPL_SYCL
#تعریف_کن SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#تعریف_کن SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#تعریف_کن GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#رهنمود clang diagnostic افزودن
#رهنمود clang diagnostic ignored "-Wnested-anon-types"
#ضمن "ggml-common.h"
#رهنمود clang diagnostic برداشت
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
#اگر_تعریف_شده GGML_SYCL_F16
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#وإلا
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#پایان_اگر
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
خارجي صحيح g_ggml_sycl_memtrace_گام;


#تعریف_کن CHECK_TRY_ERROR(عبارت)                                            \
  [&]() {                                                                \
    حاول {                                                                \
      عبارت;                                                              \
      أرجع dpct::success;                                              \
    } امسك (std::exception ثابت& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::پایان_سطر;                                            \
      أرجع dpct::default_error;                                        \
    }                                                                    \
  }()


#تعریف_کن __SYCL_ARCH__ DPCT_COMPATIBILITY_موقت
#تعریف_کن VER_4VEC 610 // todo for hardware optimize.
#تعریف_کن VER_GEN9 700 // todo for hardware optimize.
#تعریف_کن VER_GEN12 1000000 // todo for hardware optimize.
#تعریف_کن VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#تعریف_کن GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#إذا !defined(GGML_SYCL_FORCE_MMQ)
    #تعریف_کن SYCL_USE_XMX
#پایان_اگر

// max batch size to use MMQ kernels when tensor cores are available
#تعریف_کن MMQ_MAX_BATCH_اندازه 32

// dmmv = dequantize_mul_mat_vec
#اگر_تعریف_نشده GGML_SYCL_DMMV_X
#تعریف_کن GGML_SYCL_DMMV_X 32
#پایان_اگر
#اگر_تعریف_نشده GGML_SYCL_MMV_Y
#تعریف_کن GGML_SYCL_MMV_Y 1
#پایان_اگر

تعريفنوع sycl::queue *صف_ptr;

تعداد ggml_sycl_backend_gpu_نما {
  SYCL_UNSET_GPU_نما = -1,
  SYCL_SINGLE_GPU_نما = 0,
  SYCL_MUL_GPU_نما
};

تعداد ggml_sycl_dev2dev_memcpy_نما {
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
    ثابت صحيح سطر,
    ثابت حرف* پیام_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, پیام_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, سطر);
  GGML_ABORT("SYCL error");
}

#تعریف_کن SYCL_CHECK(err)                                                                                    \
    افعل {                                                                                                   \
        خودکار err_ = (err);                                                                                 \
        إذا (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } طالما (0)

#إذا DPCT_COMPAT_RT_نسخه >= 11100
#تعریف_کن GGML_SYCL_ASSUME(x) __builtin_assume(x)
#وإلا
#تعریف_کن GGML_SYCL_ASSUME(x)
#پایان_اگر // DPCT_COMPAT_RT_VERSION >= 11100

#اگر_تعریف_شده GGML_SYCL_F16
تعريفنوع sycl::half dfloat; // dequantize float
تعريفنوع sycl::half2 dfloat2;
#وإلا
تعريفنوع عائم dfloat; // dequantize float
تعريفنوع sycl::float2 dfloat2;
#پایان_اگر // GGML_SYCL_F16

#تعریف_کن MMVQ_MAX_BATCH_اندازه  8

ساكن صحيح g_all_sycl_device_شمار = -1;
ساكن منطقي g_ggml_backend_sycl_میانگیر_type_initialized = خطأ;

ساكن ggml_sycl_backend_gpu_نما g_ggml_sycl_backend_gpu_نما =
    SYCL_UNSET_GPU_نما;

ساكن فراغ* g_scratch_میانگیر = اشاره_تهی;
ساكن size_t g_scratch_اندازه = 0; // disabled by default
ساكن size_t g_scratch_جابه_جایی = 0;

[[noreturn]] ساكن مضمن فراغ bad_arch(ثابت sycl::stream& جریان_ct1) {
  جریان_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::خروج(1);

  (فراغ)bad_arch; // suppress unused function warning
}

صحيح get_current_device_id();

مضمن صحيح ggml_sycl_get_device() {
    أرجع get_current_device_id();
}

مضمن dpct::err0 ggml_sycl_مجموعه_device(ثابت صحيح device) حاول {
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
            << ", line:" << __LINE__ << std::پایان_سطر;
  crash();
  std::خروج(1);
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
    صحيح max_wg_در_هر_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    منطقي    vmm;                // virtual memory support
    منطقي    l0_device_type_valid;
    منطقي    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  مجموع_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    منطقي    usm_system_support; // support for USM system allocations
};


بنية ggml_sycl_device_info {
    صحيح device_شمار;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<عائم, GGML_SYCL_MAX_DEVICES> default_tensor_تقسیم = {};

    صحيح max_work_گروه_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    منطقي ext_oneapi_level_zero = درست; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ثابت ggml_sycl_device_info & ggml_sycl_info();

ساكن constexpr size_t SYCL_میانگیر_ALIGNMENT = 128;

بنية ggml_sycl_pool {
    مجازی ~ggml_sycl_pool() = افتراضي;

    مجازی فراغ * alloc(size_t size, size_t * actual_اندازه) = 0;
    مجازی فراغ حرر(فراغ * ptr, size_t size) = 0;
};

قالب<نام_نوع T>
بنية ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = اشاره_تهی;
    T * ptr = اشاره_تهی;
    size_t actual_اندازه = 0;

    صریح ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        إذا (ptr != اشاره_تهی) {
            pool->حرر(ptr, actual_اندازه);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != اشاره_تهی);
        إذا (ptr)
            pool->حرر(ptr, actual_اندازه);
        ptr = (T *) pool->alloc(size * حجم(T), &هذا->actual_اندازه);
        أرجع ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != اشاره_تهی);
        GGML_ASSERT(ptr == اشاره_تهی);
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
    ggml_sycl_pool_alloc(ثابت ggml_sycl_pool_alloc &) = حذف;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = حذف;
    ggml_sycl_pool_alloc& عملگر=(ثابت ggml_sycl_pool_alloc &) = حذف;
    ggml_sycl_pool_alloc& عملگر=(ggml_sycl_pool_alloc &&) = حذف;
};

// backend interface

بنية ggml_tensor_extra_gpu {
  فراغ* داده_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

خارجي صحيح g_ggml_sycl_use_level_zero_api;
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_مستقیم);
فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q);

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<صف_ptr> streams={});

بنية mmid_ردیف_mapping {
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

    صف_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { اشاره_تهی } };

    صریح ggml_backend_sycl_context(صحيح device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    صف_ptr stream(صحيح device, صحيح stream) {
        إذا (qptrs[device][stream] == اشاره_تهی) {
            qptrs[device][stream] = &(dpct::get_device(device).default_صف());
        }
        أرجع qptrs[device][stream];
    }

    صف_ptr stream() {
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

    std::unordered_map<sycl::queue*, dnnl::stream> جریان_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream جریان_dnnl(صحيح device, صحيح جریان_2) {
        خودکار q = stream(device, جریان_2);
        أرجع جریان_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        خودکار it = engine_map.find(qptr);
        إذا (it == engine_map.end()) {
            خودکار eng = make_engine(qptr);
            engine_map[qptr] = eng;
            أرجع eng;
        }
        وإلا
        {
            أرجع it->second;
        }
    }
    dnnl::stream جریان_dnnl(sycl::queue* qptr) {
        خودکار it = جریان_map.find(qptr);
        إذا (it == جریان_map.end()) {
            خودکار eng = engine_dnnl(qptr);
            خودکار stream = dnnl::sycl_interop::make_جریان(eng, *qptr);
            جریان_map[qptr] = stream;
            أرجع stream;
        }
        وإلا
        {
            أرجع it->second;
        }
    }
    dnnl::stream جریان_dnnl() {
        أرجع جریان_dnnl(device, 0);
    }
#پایان_اگر

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_ردیف_mapping> mmid_ردیف_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_زمانبندی_host;

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_device(صف_ptr qptr, صحيح device);

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_host(صف_ptr qptr, صحيح device);

    ggml_sycl_pool & pool(صحيح device) {
        إذا (pools[device] == اشاره_تهی) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        أرجع *pools[device];
    }

    ggml_sycl_pool & pool() {
        أرجع pool(device);
    }

#اگر_تعریف_شده GGML_SYCL_گراف
    std::unique_ptr<sycl_ex::command_گراف<sycl_ex::گراف_state::executable>> exec_گراف = اشاره_تهی;
#پایان_اگر

    ggml_sycl_pool & host_pool(صحيح device) {
        إذا (host_pools[device] == اشاره_تهی) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        أرجع *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { أرجع host_pool(device); }
};

// common device functions

ساكن __dpct_inline__ عائم warp_reduce_جمع(عائم x,
    ثابت sycl::nd_قلم<3>& قلم_ct1) {
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        x += dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), x, ماسک);
    }
    أرجع x;
}

ساكن __dpct_inline__ sycl::float2
warp_reduce_جمع(sycl::float2 a, ثابت sycl::nd_قلم<3>& قلم_ct1) {
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        a.x() += dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), a.x(),
            ماسک);
        a.y() += dpct::جایگشت_sub_گروه_by_xor(قلم_ct1.get_sub_گروه(), a.y(),
            ماسک);
    }
    أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ صحيح warp_reduce_جمع(صحيح x) {
  أرجع sycl::reduce_over_گروه(
      sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ عائم warp_reduce_جمع(عائم x) {
#رهنمود unroll
  لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
    x += dpct::جایگشت_sub_گروه_by_xor(
        sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), x, جابه_جایی, عرض);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ عائم warp_reduce_جمع(عائم x, ثابت sycl::nd_قلم<3>& قلم_ct1) {
#رهنمود unroll
  لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
    x += dpct::جایگشت_sub_گروه_by_xor(
        قلم_ct1.get_sub_گروه(), x, جابه_جایی);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ sycl::float2 warp_reduce_جمع(sycl::float2 a) {
#رهنمود unroll
  لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
    a.x() += dpct::جایگشت_sub_گروه_by_xor(
        sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), a.x(), جابه_جایی,
        عرض);
    a.y() += dpct::جایگشت_sub_گروه_by_xor(
        sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), a.y(), جابه_جایی,
        عرض);
  }
  أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ sycl::half2 warp_reduce_جمع(sycl::half2 a) {
#رهنمود unroll
  لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
    a = a + dpct::جایگشت_sub_گروه_by_xor(
                sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), a, جابه_جایی,
                عرض);
  }
  أرجع a;
}

ساكن constexpr صحيح ggml_sycl_get_physical_warp_اندازه() {
  // todo: for old iGPU + dGPU case, need to be changed.
  أرجع WARP_اندازه;
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ صحيح warp_reduce_all(صحيح x) {
    إذا (عرض == ggml_sycl_get_physical_warp_اندازه()) {
        أرجع sycl::all_از_گروه(
            sycl::ext::oneapi::this_work_قلم::get_sub_گروه(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_قلم::get_sub_گروه()
                         .get_local_خطی_id())) ||
                x);
    } وإلا {
#رهنمود unroll
        لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
            x = dpct::جایگشت_sub_گروه_by_xor(
                    sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), x,
                    جابه_جایی, عرض) &&
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ صحيح warp_reduce_any(صحيح x) {
    إذا (عرض == ggml_sycl_get_physical_warp_اندازه()) {
        أرجع sycl::any_از_گروه(
            sycl::ext::oneapi::this_work_قلم::get_sub_گروه(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_قلم::get_sub_گروه()
                         .get_local_خطی_id())) &&
                x);
    } وإلا {
#رهنمود unroll
        لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
            x = dpct::جایگشت_sub_گروه_by_xor(
                    sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), x,
                    جابه_جایی, عرض) ||
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
قالب <صحيح عرض>
ساكن __dpct_inline__ عائم warp_reduce_max(عائم x) {
#رهنمود unroll
  لكل (صحيح جابه_جایی = عرض / 2; جابه_جایی > 0; جابه_جایی >>= 1) {
    x = sycl::fmax(x, dpct::جایگشت_sub_گروه_by_xor(
                          sycl::ext::oneapi::this_work_قلم::get_sub_گروه(), x,
                          جابه_جایی, عرض));
  }
  أرجع x;
}

ساكن __dpct_inline__ عائم warp_reduce_max(عائم x,
    ثابت sycl::nd_قلم<3>& قلم_ct1) {
#رهنمود unroll
    لكل (صحيح ماسک = WARP_اندازه / 2; ماسک > 0; ماسک >>= 1) {
        x = sycl::fmax(x, dpct::جایگشت_sub_گروه_by_xor(
            قلم_ct1.get_sub_گروه(), x, ماسک));
    }
    أرجع x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
قالب<صحيح N>
__dpct_inline__ size_t calculate_جابه_جایی(ثابت std::array<صحيح, N> & strides, ثابت std::array<صحيح, N> & indices) {
    size_t جابه_جایی = 0;
#رهنمود unroll
    لكل (صحيح i = 0; i < N; i++) {
        خودکار شاخص_i = indices[i];
        جابه_جایی += strides[i] * شاخص_i;
    }
    أرجع جابه_جایی;
}

// Helper for vec loading aligned data
قالب <نام_نوع Tp, صحيح n>
مضمن sycl::vec<Tp, n> vec_aligned_بارگیری(ثابت Tp* aligned_ptr) {
    أرجع *reinterpret_cast<ثابت sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
قالب <نام_نوع Tp, صحيح dim>
ساكن __dpct_inline__ Tp* get_اشاره_گر(sycl::local_accessor<Tp, dim> acc) {
    أرجع acc.قالب get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلوک_رقم, int64_t بلوک_اندازه_3);

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
#پایان_اگر

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val);

قالب <صحيح N, صنف T> std::string debug_get_array_str(ثابت std::string & prefix, ثابت T array[N]) {
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
                         std::size_t رقم_src, ثابت std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        إذا (LIKELY(!g_ggml_sycl_debug)) {
            أرجع;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.داده(), func_suffix.داده());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        إذا (dst) {
            لكل (std::size_t i = 0; i < رقم_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.داده());
    }

    scope_op_debug_print(ثابت std::string_view & func, ثابت ggml_tensor * dst, std::size_t رقم_src,
                         ثابت std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, رقم_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.داده(), func_suffix.داده()); }

  خاص:
    std::string_view func;
    std::string_view func_suffix;
};

ساكن __dpct_inline__ عائم get_alibi_شیب(ثابت عائم    max_bias,
                                             ثابت uint32_t h,
                                             ثابت uint32_t n_سر_log2,
                                             ثابت عائم    m0,
                                             ثابت عائم    m1) {
    إذا (max_bias <= 0.0f) {
        أرجع 1.0f;
    }
    ثابت عائم پایه = h < n_سر_log2 ? m0 : m1;
    ثابت صحيح   exph = h < n_سر_log2 ? h + 1 : 2*(h - n_سر_log2) + 1;

    أرجع dpct::توان(پایه, exph);
}

ساكن ثابت sycl::uint3 init_fastdiv_مقادیر(uint32_t d) {
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
ساكن constexpr صحيح ggml_sycl_get_max_cpy_بایتها() {
    أرجع 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
قالب <صحيح nbytes, صحيح alignment = 0>
ساكن __dpct_inline__ فراغ ggml_sycl_memcpy_1(فراغ * dst, ثابت فراغ * src) {
    إذا constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr صحيح nb_در_هر_cpy = alignment == 0 ? nbytes : alignment;

#رهنمود unroll
    لكل (صحيح i = 0; i < nbytes/nb_در_هر_cpy; ++i) {
        إذا constexpr (nb_در_هر_cpy == 1) {
            ((حرف *) dst)[i] = ((ثابت حرف *) src)[i];
        } وإلا إذا constexpr (nb_در_هر_cpy == 2) {
            ((کوتاه *) dst)[i] = ((ثابت کوتاه *) src)[i];
        } وإلا إذا constexpr (nb_در_هر_cpy == 4) {
            ((صحيح *) dst)[i] = ((ثابت صحيح *) src)[i];
        } وإلا إذا constexpr (nb_در_هر_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ثابت sycl::int2 *) src)[i];
        } وإلا إذا constexpr (nb_در_هر_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ثابت sycl::int4 *) src)[i];
        } وإلا {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
قالب <نام_نوع T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    أرجع res;
}

ساكن __dpct_inline__ uint32_t fastdiv(uint32_t n, ثابت sycl::uint3 fastdiv_مقادیر) {
    ثابت uint32_t hi = sycl::mul_hi<بیعلامت>(n, fastdiv_مقادیر.x());
    أرجع (hi + n) >> fastdiv_مقادیر.y();
}


قالب <نام_نوع T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<عائم>(x),static_cast<عائم>(y));
    أرجع res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_مقدار(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_مقدار;
}

ساكن __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ثابت sycl::uint3 fastdiv_مقادیر) {
    ثابت uint32_t div_val = fastdiv(n, fastdiv_مقادیر);
    ثابت uint32_t mod_val = n - div_val * fastdiv_مقادیر.z();
    أرجع sycl::uint2(div_val, mod_val);
}

ساكن __dpct_inline__ صحيح ggml_sycl_dp4a(ثابت صحيح a, ثابت صحيح b, صحيح c) {
    أرجع dpct::dp4a(a, b, c);
}

ساكن __dpct_inline__ عائم ggml_sycl_e8m0_مقصد_fp32(uint8_t x) {
    uint32_t بیتها;
    إذا (x == 0) {
        بیتها = 0x00400000;
    } وإلا {
        بیتها = (uint32_t) x << 23;
    }

    عائم نتیجه;
    کپی_حافظه(&نتیجه, &بیتها, حجم(عائم));
    أرجع نتیجه;
}

sycl::float2 __dpct_inline__ __half22float2(ثابت sycl::half2 &H) {
    sycl::float2 float2_مقدار(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_مقدار;
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
#اگر_تعریف_شده GGML_SYCL_F16
    ثابت sycl::float2 tmp = (v * u).قالب convert<عائم, sycl::rounding_نما::automatic>();
    acc += tmp.x() + tmp.y();
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#پایان_اگر // GGML_SYCL_F16
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(sycl::half2 & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#اگر_تعریف_شده GGML_SYCL_F16
    acc += v*u;
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#پایان_اگر // GGML_SYCL_F16
}

قالب <صحيح n>
بنية ggml_sycl_unroll {
    قالب <نام_نوع Func, نام_نوع... آرگومانها>
    فراغ عملگر()(ثابت Func & f, آرگومانها... آرگومانها_2) ثابت {
        f(n - 1, آرگومانها_2...);
        ggml_sycl_unroll<n - 1>{}(f, آرگومانها_2...);
    }
};

قالب <>
بنية ggml_sycl_unroll<1> {
    قالب <نام_نوع Func, نام_نوع... آرگومانها>
    فراغ عملگر()(ثابت Func & f, آرگومانها... آرگومانها_2) ثابت {
        f(0, آرگومانها_2...);
    }
};

ساكن __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_نما::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_نما::automatic>()[0];
    أرجع ret;
}

ساكن __dpct_inline__ sycl::half ggml_sycl_hmax(ثابت sycl::half a, ثابت sycl::half b) {
    أرجع sycl::vec<عائم, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<عائم, sycl::rounding_نما::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<عائم, sycl::rounding_نما::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_نما::automatic>()[0];
}

ساكن __dpct_inline__ uint32_t __hgt2_mask(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    ثابت uint32_t ماسک_کم  = 0x0000FFFF * (عائم(a[0]) > عائم(b[0]));
    ثابت uint32_t ماسک_بالا = 0xFFFF0000 * (عائم(a[1]) > عائم(b[1]));
    أرجع ماسک_کم | ماسک_بالا;
}

ساكن __dpct_inline__ uint32_t fastmodulo(uint32_t n, ثابت sycl::uint3 fastdiv_مقادیر) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    أرجع n - fastdiv(n, fastdiv_مقادیر) * fastdiv_مقادیر.z();
}

ساكن منطقي fast_fp16_موجود(ثابت صحيح cc) {
    GGML_استفادهنشده(cc);
    أرجع درست;   //Intel GPUs always support FP16.
}

تعداد صنف بلوک_reduce_روش {
    MAX,
    جمع_3,
};

قالب<بلوک_reduce_روش روش_t, نام_نوع T, صحيح warp_اندازه>
بنية بلوک_reduce_policy;

قالب <نام_نوع T, نام_نوع... Ts>
مضمن constexpr منطقي is_any = (std::is_same_v<T, Ts> || ...);

قالب<نام_نوع...>
مضمن constexpr منطقي ggml_sycl_dependent_false_v = خطأ;

#تعریف_کن WARP_32_اندازه 32

قالب <نام_نوع T, صحيح warp_اندازه> بنية بلوک_reduce_policy<بلوک_reduce_روش::جمع_3, T, warp_اندازه> {
    ساكن T reduce(T val) {
        إذا constexpr (is_any<T, عائم, sycl::float2, sycl::half2, صحيح>) {
            أرجع warp_reduce_جمع<warp_اندازه>(val);
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

قالب <نام_نوع T, صحيح warp_اندازه> بنية بلوک_reduce_policy<بلوک_reduce_روش::MAX, T, warp_اندازه> {
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


قالب <بلوک_reduce_روش reduce_روش_t, صحيح warp_اندازه, نام_نوع T>
ساكن T بلوک_reduce(T val, T * shared_vals, صحيح بلوک_اندازه_template) {
    خودکار قلم_ct1                 = sycl::ext::oneapi::this_work_قلم::get_nd_قلم<3>();
    val                           = بلوک_reduce_policy<reduce_روش_t, T,warp_اندازه>::reduce(val);
    ثابت صحيح بلوک_اندازه_3 = بلوک_اندازه_template == 0 ? قلم_ct1.get_local_range(2) : بلوک_اندازه_template;
    ثابت صحيح nthreads = قلم_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_اندازه;

    إذا (بلوک_اندازه_3 > warp_اندازه) {
        أكد((بلوک_اندازه_3 <= 1024) && (بلوک_اندازه_3 % warp_اندازه) == 0);
        ثابت صحيح warp_id = قلم_ct1.get_local_id(2) / warp_اندازه;
        ثابت صحيح lane_id = قلم_ct1.get_local_id(2) % warp_اندازه;
        إذا (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        قلم_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_اندازه;
        عائم tmp = 0.f;
        إذا (lane_id < (static_cast<صحيح>(بلوک_اندازه_3) / warp_اندازه)) {
            لكل (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_اندازه];
            }
        }
        أرجع بلوک_reduce_policy<reduce_روش_t, T, warp_اندازه>::reduce(tmp);
    }
    أرجع val;
}

ساكن __dpct_inline__ عائم ggml_sycl_ue4m3_مقصد_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    إذا (x == 0 || x == 0x7F) {
        أرجع 0.0f;
    }
    ثابت صحيح نمایی_تابع = (x >> 3) & 0xF;
    ثابت صحيح man = x & 0x7;
    عائم raw;
    إذا (نمایی_تابع == 0) {
        raw = man * (1.0f / 8.0f) * sycl::توان(2.0f, -6.0f);
    } وإلا {
        raw = (1.0f + man / 8.0f) * sycl::توان(2.0f, (عائم) نمایی_تابع - 7.0f);
    }
    أرجع raw * 0.5f;
}

#پایان_اگر // GGML_SYCL_COMMON_HPP
