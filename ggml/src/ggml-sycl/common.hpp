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

#جے_بنا_تعریف GGML_SYCL_COMMON_HPP
#تعریف_کرو GGML_SYCL_COMMON_HPP

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
    #جے_بنا_تعریف GGML_SYCL_HAS_BF16
        #تعریف_کرو GGML_SYCL_HAS_BF16
    #جے_انت
#جے_انت

#إذا defined(__INTEL_LLVM_COMPILER)
    #تعریف_کرو GGML_SYCL_DMMV_HAS_ESIMD
#جے_انت

#إذا GGML_SYCL_DNNL
#ضمن "dnnl.hpp"
#ضمن "dnnl_sycl.hpp"
#جے_انت

#تعریف_کرو GGML_COMMON_DECL_SYCL
#تعریف_کرو GGML_COMMON_IMPL_SYCL
#تعریف_کرو SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#تعریف_کرو SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#تعریف_کرو GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#ہدایت clang diagnostic ڈالو
#ہدایت clang diagnostic ignored "-Wnested-anon-types"
#ضمن "ggml-common.h"
#ہدایت clang diagnostic نکالو
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
#جے_تعریف_شدہ GGML_SYCL_F16
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#وإلا
ساكن constexpr صحيح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#جے_انت
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
خارجي صحيح g_ggml_sycl_memtrace_قدم;


#تعریف_کرو CHECK_TRY_ERROR(اظہاریہ)                                            \
  [&]() {                                                                \
    حاول {                                                                \
      اظہاریہ;                                                              \
      أرجع dpct::success;                                              \
    } امسك (std::exception ثابت& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::سطر_انت;                                            \
      أرجع dpct::default_error;                                        \
    }                                                                    \
  }()


#تعریف_کرو __SYCL_ARCH__ DPCT_COMPATIBILITY_عارضی
#تعریف_کرو VER_4VEC 610 // todo for hardware optimize.
#تعریف_کرو VER_GEN9 700 // todo for hardware optimize.
#تعریف_کرو VER_GEN12 1000000 // todo for hardware optimize.
#تعریف_کرو VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#تعریف_کرو GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#إذا !defined(GGML_SYCL_FORCE_MMQ)
    #تعریف_کرو SYCL_USE_XMX
#جے_انت

// max batch size to use MMQ kernels when tensor cores are available
#تعریف_کرو MMQ_MAX_BATCH_ناپ 32

// dmmv = dequantize_mul_mat_vec
#جے_بنا_تعریف GGML_SYCL_DMMV_X
#تعریف_کرو GGML_SYCL_DMMV_X 32
#جے_انت
#جے_بنا_تعریف GGML_SYCL_MMV_Y
#تعریف_کرو GGML_SYCL_MMV_Y 1
#جے_انت

تعريفنوع sycl::queue *قطار_ptr;

تعداد ggml_sycl_backend_gpu_منوال {
  SYCL_UNSET_GPU_منوال = -1,
  SYCL_SINGLE_GPU_منوال = 0,
  SYCL_MUL_GPU_منوال
};

تعداد ggml_sycl_dev2dev_memcpy_منوال {
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
    ثابت حرف* پیغام) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, پیغام);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, سطر);
  GGML_ABORT("SYCL error");
}

#تعریف_کرو SYCL_CHECK(err)                                                                                    \
    افعل {                                                                                                   \
        خودکار err_ = (err);                                                                                 \
        إذا (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } طالما (0)

#إذا DPCT_COMPAT_RT_نسخہ >= 11100
#تعریف_کرو GGML_SYCL_ASSUME(x) __builtin_assume(x)
#وإلا
#تعریف_کرو GGML_SYCL_ASSUME(x)
#جے_انت // DPCT_COMPAT_RT_VERSION >= 11100

#جے_تعریف_شدہ GGML_SYCL_F16
تعريفنوع sycl::half dfloat; // dequantize float
تعريفنوع sycl::half2 dfloat2;
#وإلا
تعريفنوع عائم dfloat; // dequantize float
تعريفنوع sycl::float2 dfloat2;
#جے_انت // GGML_SYCL_F16

#تعریف_کرو MMVQ_MAX_BATCH_ناپ  8

ساكن صحيح g_all_sycl_device_گنتی = -1;
ساكن منطقي g_ggml_backend_sycl_بفر_type_initialized = خطأ;

ساكن ggml_sycl_backend_gpu_منوال g_ggml_sycl_backend_gpu_منوال =
    SYCL_UNSET_GPU_منوال;

ساكن فراغ* g_scratch_بفر = خالی_سوچک;
ساكن size_t g_scratch_ناپ = 0; // disabled by default
ساكن size_t g_scratch_وتھ = 0;

[[noreturn]] ساكن مضمن فراغ bad_arch(ثابت sycl::stream& دھارا_ct1) {
  دھارا_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::باہر_نکلو(1);

  (فراغ)bad_arch; // suppress unused function warning
}

صحيح get_current_device_id();

مضمن صحيح ggml_sycl_get_device() {
    أرجع get_current_device_id();
}

مضمن dpct::err0 ggml_sycl_سیٹ_device(ثابت صحيح device) حاول {
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
            << ", line:" << __LINE__ << std::سطر_انت;
  crash();
  std::باہر_نکلو(1);
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
    صحيح warp_ناپ;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    صحيح max_wg_فی_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    منطقي    vmm;                // virtual memory support
    منطقي    l0_device_type_valid;
    منطقي    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  کل_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    منطقي    usm_system_support; // support for USM system allocations
};


بنية ggml_sycl_device_info {
    صحيح device_گنتی;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<عائم, GGML_SYCL_MAX_DEVICES> default_tensor_ونڈو = {};

    صحيح max_work_گروہ_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    منطقي ext_oneapi_level_zero = سچ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ثابت ggml_sycl_device_info & ggml_sycl_info();

ساكن constexpr size_t SYCL_بفر_ALIGNMENT = 128;

بنية ggml_sycl_pool {
    مجازی ~ggml_sycl_pool() = افتراضي;

    مجازی فراغ * alloc(size_t size, size_t * actual_ناپ) = 0;
    مجازی فراغ حرر(فراغ * ptr, size_t size) = 0;
};

سانچا<قسم_ناں T>
بنية ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = خالی_سوچک;
    T * ptr = خالی_سوچک;
    size_t actual_ناپ = 0;

    واضح ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        إذا (ptr != خالی_سوچک) {
            pool->حرر(ptr, actual_ناپ);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != خالی_سوچک);
        إذا (ptr)
            pool->حرر(ptr, actual_ناپ);
        ptr = (T *) pool->alloc(size * حجم(T), &هذا->actual_ناپ);
        أرجع ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != خالی_سوچک);
        GGML_ASSERT(ptr == خالی_سوچک);
        ptr = (T *) pool->alloc(size * حجم(T), &هذا->actual_ناپ);
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
    ggml_sycl_pool_alloc(ثابت ggml_sycl_pool_alloc &) = مٹاؤ;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = مٹاؤ;
    ggml_sycl_pool_alloc& عامل=(ثابت ggml_sycl_pool_alloc &) = مٹاؤ;
    ggml_sycl_pool_alloc& عامل=(ggml_sycl_pool_alloc &&) = مٹاؤ;
};

// backend interface

بنية ggml_tensor_extra_gpu {
  فراغ* انکڑے_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

خارجي صحيح g_ggml_sycl_use_level_zero_api;
فراغ * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_سدھا);
فراغ ggml_sycl_free_device(فراغ *ptr, sycl::queue &q);

فراغ release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<قطار_ptr> streams={});

بنية mmid_صف_mapping {
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

    قطار_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { خالی_سوچک } };

    واضح ggml_backend_sycl_context(صحيح device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    قطار_ptr stream(صحيح device, صحيح stream) {
        إذا (qptrs[device][stream] == خالی_سوچک) {
            qptrs[device][stream] = &(dpct::get_device(device).default_قطار());
        }
        أرجع qptrs[device][stream];
    }

    قطار_ptr stream() {
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

    std::unordered_map<sycl::queue*, dnnl::stream> دھارا_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream دھارا_dnnl(صحيح device, صحيح دھارا_2) {
        خودکار q = stream(device, دھارا_2);
        أرجع دھارا_dnnl(q);
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
    dnnl::stream دھارا_dnnl(sycl::queue* qptr) {
        خودکار it = دھارا_map.find(qptr);
        إذا (it == دھارا_map.end()) {
            خودکار eng = engine_dnnl(qptr);
            خودکار stream = dnnl::sycl_interop::make_دھارا(eng, *qptr);
            دھارا_map[qptr] = stream;
            أرجع stream;
        }
        وإلا
        {
            أرجع it->second;
        }
    }
    dnnl::stream دھارا_dnnl() {
        أرجع دھارا_dnnl(device, 0);
    }
#جے_انت

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_صف_mapping> mmid_صف_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_ویلا_سارنی_host;

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_device(قطار_ptr qptr, صحيح device);

    ساكن std::unique_ptr<ggml_sycl_pool> new_pool_for_host(قطار_ptr qptr, صحيح device);

    ggml_sycl_pool & pool(صحيح device) {
        إذا (pools[device] == خالی_سوچک) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        أرجع *pools[device];
    }

    ggml_sycl_pool & pool() {
        أرجع pool(device);
    }

#جے_تعریف_شدہ GGML_SYCL_گراف
    std::unique_ptr<sycl_ex::command_گراف<sycl_ex::گراف_state::executable>> exec_گراف = خالی_سوچک;
#جے_انت

    ggml_sycl_pool & host_pool(صحيح device) {
        إذا (host_pools[device] == خالی_سوچک) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        أرجع *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { أرجع host_pool(device); }
};

// common device functions

ساكن __dpct_inline__ عائم warp_reduce_جوڑ(عائم x,
    ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), x, نقاب);
    }
    أرجع x;
}

ساكن __dpct_inline__ sycl::float2
warp_reduce_جوڑ(sycl::float2 a, ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        a.x() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), a.x(),
            نقاب);
        a.y() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), a.y(),
            نقاب);
    }
    أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ صحيح warp_reduce_جوڑ(صحيح x) {
  أرجع sycl::reduce_over_گروہ(
      sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ عائم warp_reduce_جوڑ(عائم x) {
#ہدایت unroll
  لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
    x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x, وتھ, چوڑائی);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ عائم warp_reduce_جوڑ(عائم x, ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
  لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
    x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        شے_ct1.get_sub_گروہ(), x, وتھ);
  }
  أرجع x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ sycl::float2 warp_reduce_جوڑ(sycl::float2 a) {
#ہدایت unroll
  لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
    a.x() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a.x(), وتھ,
        چوڑائی);
    a.y() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a.y(), وتھ,
        چوڑائی);
  }
  أرجع a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ sycl::half2 warp_reduce_جوڑ(sycl::half2 a) {
#ہدایت unroll
  لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
    a = a + dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a, وتھ,
                چوڑائی);
  }
  أرجع a;
}

ساكن constexpr صحيح ggml_sycl_get_physical_warp_ناپ() {
  // todo: for old iGPU + dGPU case, need to be changed.
  أرجع WARP_ناپ;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ صحيح warp_reduce_all(صحيح x) {
    إذا (چوڑائی == ggml_sycl_get_physical_warp_ناپ()) {
        أرجع sycl::all_دا_گروہ(
            sycl::ext::oneapi::this_work_شے::get_sub_گروہ(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_شے::get_sub_گروہ()
                         .get_local_خطی_id())) ||
                x);
    } وإلا {
#ہدایت unroll
        لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
            x = dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                    sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                    وتھ, چوڑائی) &&
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ صحيح warp_reduce_any(صحيح x) {
    إذا (چوڑائی == ggml_sycl_get_physical_warp_ناپ()) {
        أرجع sycl::any_دا_گروہ(
            sycl::ext::oneapi::this_work_شے::get_sub_گروہ(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_شے::get_sub_گروہ()
                         .get_local_خطی_id())) &&
                x);
    } وإلا {
#ہدایت unroll
        لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
            x = dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                    sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                    وتھ, چوڑائی) ||
                x;
        }
        أرجع x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچا <صحيح چوڑائی>
ساكن __dpct_inline__ عائم warp_reduce_max(عائم x) {
#ہدایت unroll
  لكل (صحيح وتھ = چوڑائی / 2; وتھ > 0; وتھ >>= 1) {
    x = sycl::fmax(x, dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                          sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                          وتھ, چوڑائی));
  }
  أرجع x;
}

ساكن __dpct_inline__ عائم warp_reduce_max(عائم x,
    ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    لكل (صحيح نقاب = WARP_ناپ / 2; نقاب > 0; نقاب >>= 1) {
        x = sycl::fmax(x, dpct::ترتیب_بدلو_sub_گروہ_by_xor(
            شے_ct1.get_sub_گروہ(), x, نقاب));
    }
    أرجع x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
سانچا<صحيح N>
__dpct_inline__ size_t calculate_وتھ(ثابت std::array<صحيح, N> & strides, ثابت std::array<صحيح, N> & indices) {
    size_t وتھ = 0;
#ہدایت unroll
    لكل (صحيح i = 0; i < N; i++) {
        خودکار اشاریہ_i = indices[i];
        وتھ += strides[i] * اشاریہ_i;
    }
    أرجع وتھ;
}

// Helper for vec loading aligned data
سانچا <قسم_ناں Tp, صحيح n>
مضمن sycl::vec<Tp, n> vec_aligned_لدو(ثابت Tp* aligned_ptr) {
    أرجع *reinterpret_cast<ثابت sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
سانچا <قسم_ناں Tp, صحيح dim>
ساكن __dpct_inline__ Tp* get_اشارہ_گر(sycl::local_accessor<Tp, dim> acc) {
    أرجع acc.سانچا get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_ناپ_3);

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
#جے_انت

صحيح ggml_sycl_get_env(ثابت حرف *env_name, صحيح default_val);

سانچا <صحيح N, صنف T> std::string debug_get_array_str(ثابت std::string & prefix, ثابت T array[N]) {
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
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.انکڑے(), func_suffix.انکڑے());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        إذا (dst) {
            لكل (std::size_t i = 0; i < عدد_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.انکڑے());
    }

    scope_op_debug_print(ثابت std::string_view & func, ثابت ggml_tensor * dst, std::size_t عدد_src,
                         ثابت std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, عدد_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.انکڑے(), func_suffix.انکڑے()); }

  خاص:
    std::string_view func;
    std::string_view func_suffix;
};

ساكن __dpct_inline__ عائم get_alibi_ڈھلوان(ثابت عائم    max_bias,
                                             ثابت uint32_t h,
                                             ثابت uint32_t n_سر_log2,
                                             ثابت عائم    m0,
                                             ثابت عائم    m1) {
    إذا (max_bias <= 0.0f) {
        أرجع 1.0f;
    }
    ثابت عائم ادھار = h < n_سر_log2 ? m0 : m1;
    ثابت صحيح   exph = h < n_سر_log2 ? h + 1 : 2*(h - n_سر_log2) + 1;

    أرجع dpct::قوت(ادھار, exph);
}

ساكن ثابت sycl::uint3 init_fastdiv_اقدار(uint32_t d) {
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
ساكن constexpr صحيح ggml_sycl_get_max_cpy_بائٹاں() {
    أرجع 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
سانچا <صحيح nbytes, صحيح alignment = 0>
ساكن __dpct_inline__ فراغ ggml_sycl_memcpy_1(فراغ * dst, ثابت فراغ * src) {
    إذا constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr صحيح nb_فی_cpy = alignment == 0 ? nbytes : alignment;

#ہدایت unroll
    لكل (صحيح i = 0; i < nbytes/nb_فی_cpy; ++i) {
        إذا constexpr (nb_فی_cpy == 1) {
            ((حرف *) dst)[i] = ((ثابت حرف *) src)[i];
        } وإلا إذا constexpr (nb_فی_cpy == 2) {
            ((چھوٹا *) dst)[i] = ((ثابت چھوٹا *) src)[i];
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
سانچا <قسم_ناں T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    أرجع res;
}

ساكن __dpct_inline__ uint32_t fastdiv(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    ثابت uint32_t hi = sycl::mul_hi<بنانشانی>(n, fastdiv_اقدار.x());
    أرجع (hi + n) >> fastdiv_اقدار.y();
}


سانچا <قسم_ناں T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<عائم>(x),static_cast<عائم>(y));
    أرجع res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_قدر(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_قدر;
}

ساكن __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    ثابت uint32_t div_val = fastdiv(n, fastdiv_اقدار);
    ثابت uint32_t mod_val = n - div_val * fastdiv_اقدار.z();
    أرجع sycl::uint2(div_val, mod_val);
}

ساكن __dpct_inline__ صحيح ggml_sycl_dp4a(ثابت صحيح a, ثابت صحيح b, صحيح c) {
    أرجع dpct::dp4a(a, b, c);
}

ساكن __dpct_inline__ عائم ggml_sycl_e8m0_منزل_fp32(uint8_t x) {
    uint32_t بٹ;
    إذا (x == 0) {
        بٹ = 0x00400000;
    } وإلا {
        بٹ = (uint32_t) x << 23;
    }

    عائم نتیجہ;
    حافظہ_نقل(&نتیجہ, &بٹ, حجم(عائم));
    أرجع نتیجہ;
}

sycl::float2 __dpct_inline__ __half22float2(ثابت sycl::half2 &H) {
    sycl::float2 float2_قدر(static_cast<عائم>(H.x()), static_cast<عائم>(H.y()));
    أرجع float2_قدر;
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
#جے_تعریف_شدہ GGML_SYCL_F16
    ثابت sycl::float2 tmp = (v * u).سانچا convert<عائم, sycl::rounding_منوال::automatic>();
    acc += tmp.x() + tmp.y();
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#جے_انت // GGML_SYCL_F16
}

ساكن __dpct_inline__ فراغ ggml_sycl_mad(sycl::half2 & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#جے_تعریف_شدہ GGML_SYCL_F16
    acc += v*u;
#وإلا
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#جے_انت // GGML_SYCL_F16
}

سانچا <صحيح n>
بنية ggml_sycl_unroll {
    سانچا <قسم_ناں Func, قسم_ناں... دلیلاں>
    فراغ عامل()(ثابت Func & f, دلیلاں... دلیلاں_2) ثابت {
        f(n - 1, دلیلاں_2...);
        ggml_sycl_unroll<n - 1>{}(f, دلیلاں_2...);
    }
};

سانچا <>
بنية ggml_sycl_unroll<1> {
    سانچا <قسم_ناں Func, قسم_ناں... دلیلاں>
    فراغ عامل()(ثابت Func & f, دلیلاں... دلیلاں_2) ثابت {
        f(0, دلیلاں_2...);
    }
};

ساكن __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<عائم, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
    أرجع ret;
}

ساكن __dpct_inline__ sycl::half ggml_sycl_hmax(ثابت sycl::half a, ثابت sycl::half b) {
    أرجع sycl::vec<عائم, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<عائم, sycl::rounding_منوال::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<عائم, sycl::rounding_منوال::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
}

ساكن __dpct_inline__ uint32_t __hgt2_mask(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    ثابت uint32_t نقاب_کم  = 0x0000FFFF * (عائم(a[0]) > عائم(b[0]));
    ثابت uint32_t نقاب_بلند = 0xFFFF0000 * (عائم(a[1]) > عائم(b[1]));
    أرجع نقاب_کم | نقاب_بلند;
}

ساكن __dpct_inline__ uint32_t fastmodulo(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    أرجع n - fastdiv(n, fastdiv_اقدار) * fastdiv_اقدار.z();
}

ساكن منطقي fast_fp16_دستیاب(ثابت صحيح cc) {
    GGML_بن_ورتیا(cc);
    أرجع سچ;   //Intel GPUs always support FP16.
}

تعداد صنف بلاک_reduce_طریقہ {
    MAX,
    جوڑ_3,
};

سانچا<بلاک_reduce_طریقہ طریقہ_t, قسم_ناں T, صحيح warp_ناپ>
بنية بلاک_reduce_policy;

سانچا <قسم_ناں T, قسم_ناں... Ts>
مضمن constexpr منطقي is_any = (std::is_same_v<T, Ts> || ...);

سانچا<قسم_ناں...>
مضمن constexpr منطقي ggml_sycl_dependent_false_v = خطأ;

#تعریف_کرو WARP_32_ناپ 32

سانچا <قسم_ناں T, صحيح warp_ناپ> بنية بلاک_reduce_policy<بلاک_reduce_طریقہ::جوڑ_3, T, warp_ناپ> {
    ساكن T reduce(T val) {
        إذا constexpr (is_any<T, عائم, sycl::float2, sycl::half2, صحيح>) {
            أرجع warp_reduce_جوڑ<warp_ناپ>(val);
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

سانچا <قسم_ناں T, صحيح warp_ناپ> بنية بلاک_reduce_policy<بلاک_reduce_طریقہ::MAX, T, warp_ناپ> {
    ساكن T reduce(T val) {
        إذا constexpr (is_any<T, عائم, sycl::half2>) {
            أرجع warp_reduce_max<warp_ناپ>(val);
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


سانچا <بلاک_reduce_طریقہ reduce_طریقہ_t, صحيح warp_ناپ, قسم_ناں T>
ساكن T بلاک_reduce(T val, T * shared_vals, صحيح بلاک_ناپ_template) {
    خودکار شے_ct1                 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    val                           = بلاک_reduce_policy<reduce_طریقہ_t, T,warp_ناپ>::reduce(val);
    ثابت صحيح بلاک_ناپ_3 = بلاک_ناپ_template == 0 ? شے_ct1.get_local_range(2) : بلاک_ناپ_template;
    ثابت صحيح nthreads = شے_ct1.get_local_range(2);
    ثابت صحيح nwarps = nthreads / WARP_ناپ;

    إذا (بلاک_ناپ_3 > warp_ناپ) {
        أكد((بلاک_ناپ_3 <= 1024) && (بلاک_ناپ_3 % warp_ناپ) == 0);
        ثابت صحيح warp_id = شے_ct1.get_local_id(2) / warp_ناپ;
        ثابت صحيح lane_id = شے_ct1.get_local_id(2) % warp_ناپ;
        إذا (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_ناپ;
        عائم tmp = 0.f;
        إذا (lane_id < (static_cast<صحيح>(بلاک_ناپ_3) / warp_ناپ)) {
            لكل (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_ناپ];
            }
        }
        أرجع بلاک_reduce_policy<reduce_طریقہ_t, T, warp_ناپ>::reduce(tmp);
    }
    أرجع val;
}

ساكن __dpct_inline__ عائم ggml_sycl_ue4m3_منزل_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    إذا (x == 0 || x == 0x7F) {
        أرجع 0.0f;
    }
    ثابت صحيح قوت_نما = (x >> 3) & 0xF;
    ثابت صحيح man = x & 0x7;
    عائم raw;
    إذا (قوت_نما == 0) {
        raw = man * (1.0f / 8.0f) * sycl::قوت(2.0f, -6.0f);
    } وإلا {
        raw = (1.0f + man / 8.0f) * sycl::قوت(2.0f, (عائم) قوت_نما - 7.0f);
    }
    أرجع raw * 0.5f;
}

#جے_انت // GGML_SYCL_COMMON_HPP
