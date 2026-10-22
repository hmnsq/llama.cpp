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

#اگر_غیر_تعریف_شدہ GGML_SYCL_COMMON_HPP
#تعریف_کرو GGML_SYCL_COMMON_HPP

#شامل <cstddef>
#شامل <fstream>
#شامل <iostream>
#شامل <string>

#شامل "base.hpp"
#شامل "dpct/helper.hpp"
#شامل "ggml.h"
#شامل "ggml-impl.h"
#شامل "ggml-sycl.h"
#شامل "presets.hpp"
#شامل "type.hpp"
#شامل "sycl_hw.hpp"
#شامل "memtrace.hpp"

نامخلا syclexp = sycl::ext::oneapi::experimental;

#اگر defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #شامل <sycl/ext/oneapi/bfloat16.hpp>
    #اگر_غیر_تعریف_شدہ GGML_SYCL_HAS_BF16
        #تعریف_کرو GGML_SYCL_HAS_BF16
    #اگر_ختم
#اگر_ختم

#اگر defined(__INTEL_LLVM_COMPILER)
    #تعریف_کرو GGML_SYCL_DMMV_HAS_ESIMD
#اگر_ختم

#اگر GGML_SYCL_DNNL
#شامل "dnnl.hpp"
#شامل "dnnl_sycl.hpp"
#اگر_ختم

#تعریف_کرو GGML_COMMON_DECL_SYCL
#تعریف_کرو GGML_COMMON_IMPL_SYCL
#تعریف_کرو SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#تعریف_کرو SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#تعریف_کرو GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#ہدایت clang diagnostic ڈالو
#ہدایت clang diagnostic ignored "-Wnested-anon-types"
#شامل "ggml-common.h"
#ہدایت clang diagnostic نکالو
#شامل "ggml-impl.h"

خالی* ggml_sycl_host_malloc(size_t size);
خالی ggml_sycl_host_free(خالی* ptr);


خارجی صحیح g_ggml_sycl_debug;
خارجی صحیح g_ggml_sycl_enable_optimize;
خارجی صحیح g_ggml_sycl_enable_fusion;
خارجی صحیح g_ggml_sycl_enable_esimd;
خارجی صحیح g_ggml_sycl_mmvq_wide;
خارجی صحیح g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
شمار ggml_sycl_xmx_gather_type {
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
ساکن constexpr صحیح GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
خارجی صحیح g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
ساکن constexpr صحیح GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
خارجی صحیح g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
شمار ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#اگر_تعریف_شدہ GGML_SYCL_F16
ساکن constexpr صحیح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#ورنہ
ساکن constexpr صحیح GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#اگر_ختم
خارجی صحیح g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
خارجی صحیح g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
ساکن اندرونی int32_t ggml_sycl_src1_prec(ثابت ggml_tensor * dst) {
    واپس g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
ساکن اندرونی منطقی ggml_sycl_src1_f16_ok(ثابت ggml_tensor * dst) {
    ثابت int32_t src1_prec = ggml_sycl_src1_prec(dst);
    واپس g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
خارجی صحیح g_ggml_sycl_enable_flash_attention;
خارجی صحیح g_ggml_sycl_dev2dev_memcpy;
خارجی صحیح g_ggml_sycl_fa_onednn;
خارجی صحیح g_ggml_sycl_fa_onednn_max_kv;
خارجی صحیح g_ggml_sycl_enable_mkl_fa;
خارجی صحیح g_ggml_sycl_memtrace;
خارجی صحیح g_ggml_sycl_memtrace_قدم;


#تعریف_کرو CHECK_TRY_ERROR(اظہاریہ)                                            \
  [&]() {                                                                \
    کوشش {                                                                \
      اظہاریہ;                                                              \
      واپس dpct::success;                                              \
    } پکڑو (std::exception ثابت& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::سطر_ختم;                                            \
      واپس dpct::default_error;                                        \
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
#اگر !defined(GGML_SYCL_FORCE_MMQ)
    #تعریف_کرو SYCL_USE_XMX
#اگر_ختم

// max batch size to use MMQ kernels when tensor cores are available
#تعریف_کرو MMQ_MAX_BATCH_سائز 32

// dmmv = dequantize_mul_mat_vec
#اگر_غیر_تعریف_شدہ GGML_SYCL_DMMV_X
#تعریف_کرو GGML_SYCL_DMMV_X 32
#اگر_ختم
#اگر_غیر_تعریف_شدہ GGML_SYCL_MMV_Y
#تعریف_کرو GGML_SYCL_MMV_Y 1
#اگر_ختم

قسمتعریف sycl::queue *قطار_ptr;

شمار ggml_sycl_backend_gpu_منوال {
  SYCL_UNSET_GPU_منوال = -1,
  SYCL_SINGLE_GPU_منوال = 0,
  SYCL_MUL_GPU_منوال
};

شمار ggml_sycl_dev2dev_memcpy_منوال {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(حجم(sycl::half) == حجم(ggml_fp16_t), "wrong fp16 size");

ساکن خالی crash() {
  صحیح* ptr = تہی;
  *ptr = 0;
}

[[noreturn]] ساکن خالی ggml_sycl_error(
    ثابت حرف* stmt,
    ثابت حرف* func,
    ثابت حرف* file,
    ثابت صحیح سطر,
    ثابت حرف* پیغام_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, پیغام_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, سطر);
  GGML_ABORT("SYCL error");
}

#تعریف_کرو SYCL_CHECK(err)                                                                                    \
    کرو {                                                                                                   \
        خودکار err_ = (err);                                                                                 \
        اگر (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } جب (0)

#اگر DPCT_COMPAT_RT_نسخہ >= 11100
#تعریف_کرو GGML_SYCL_ASSUME(x) __builtin_assume(x)
#ورنہ
#تعریف_کرو GGML_SYCL_ASSUME(x)
#اگر_ختم // DPCT_COMPAT_RT_VERSION >= 11100

#اگر_تعریف_شدہ GGML_SYCL_F16
قسمتعریف sycl::half dfloat; // dequantize float
قسمتعریف sycl::half2 dfloat2;
#ورنہ
قسمتعریف اعشاری dfloat; // dequantize float
قسمتعریف sycl::float2 dfloat2;
#اگر_ختم // GGML_SYCL_F16

#تعریف_کرو MMVQ_MAX_BATCH_سائز  8

ساکن صحیح g_all_sycl_device_گنتی = -1;
ساکن منطقی g_ggml_backend_sycl_بفر_type_initialized = جھوٹ;

ساکن ggml_sycl_backend_gpu_منوال g_ggml_sycl_backend_gpu_منوال =
    SYCL_UNSET_GPU_منوال;

ساکن خالی* g_scratch_بفر = خالی_اشاریہ;
ساکن size_t g_scratch_سائز = 0; // disabled by default
ساکن size_t g_scratch_انحراف = 0;

[[noreturn]] ساکن اندرونی خالی bad_arch(ثابت sycl::stream& دھارا_ct1) {
  دھارا_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::باہر_نکلو(1);

  (خالی)bad_arch; // suppress unused function warning
}

صحیح get_current_device_id();

اندرونی صحیح ggml_sycl_get_device() {
    واپس get_current_device_id();
}

اندرونی dpct::err0 ggml_sycl_سیٹ_device(ثابت صحیح device) کوشش {
  صحیح current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  اگر (device == current_device_id) {
    واپس 0;
  }

  واپس CHECK_TRY_ERROR(dpct::select_device(device));
} پکڑو (sycl::exception ثابت& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  crash();
  std::باہر_نکلو(1);
}

//////////////////////
ساخت optimize_feature {
    منطقی reorder=جھوٹ;
    منطقی onednn_optimized_gemm=جھوٹ;
};

ساخت sycl_device_info {
    صحیح cc;  // compute capability
    صحیح nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    صحیح warp_سائز;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    صحیح max_wg_فی_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    منطقی    vmm;                // virtual memory support
    منطقی    l0_device_type_valid;
    منطقی    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  کل_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    منطقی    usm_system_support; // support for USM system allocations
};


ساخت ggml_sycl_device_info {
    صحیح device_گنتی;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<اعشاری, GGML_SYCL_MAX_DEVICES> default_tensor_بانٹو = {};

    صحیح max_work_گروہ_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    منطقی ext_oneapi_level_zero = سچ; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ثابت ggml_sycl_device_info & ggml_sycl_info();

ساکن constexpr size_t SYCL_بفر_ALIGNMENT = 128;

ساخت ggml_sycl_pool {
    مجازی ~ggml_sycl_pool() = طےشدہ;

    مجازی خالی * alloc(size_t size, size_t * actual_سائز) = 0;
    مجازی خالی آزاد(خالی * ptr, size_t size) = 0;
};

سانچہ<قسم_نام T>
ساخت ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = خالی_اشاریہ;
    T * ptr = خالی_اشاریہ;
    size_t actual_سائز = 0;

    صریح ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        اگر (ptr != خالی_اشاریہ) {
            pool->آزاد(ptr, actual_سائز);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != خالی_اشاریہ);
        اگر (ptr)
            pool->آزاد(ptr, actual_سائز);
        ptr = (T *) pool->alloc(size * حجم(T), &یہ->actual_سائز);
        واپس ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != خالی_اشاریہ);
        GGML_ASSERT(ptr == خالی_اشاریہ);
        ptr = (T *) pool->alloc(size * حجم(T), &یہ->actual_سائز);
        واپس ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        یہ->pool = &pool;
        واپس alloc(size);
    }

    T * get() {
        واپس ptr;
    }

    ggml_sycl_pool_alloc() = طےشدہ;
    ggml_sycl_pool_alloc(ثابت ggml_sycl_pool_alloc &) = حذف;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = حذف;
    ggml_sycl_pool_alloc& عامل=(ثابت ggml_sycl_pool_alloc &) = حذف;
    ggml_sycl_pool_alloc& عامل=(ggml_sycl_pool_alloc &&) = حذف;
};

// backend interface

ساخت ggml_tensor_extra_gpu {
  خالی* ڈیٹا_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

خارجی صحیح g_ggml_sycl_use_level_zero_api;
خالی * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_براہ_راست);
خالی ggml_sycl_free_device(خالی *ptr, sycl::queue &q);

خالی release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<قطار_ptr> streams={});

ساخت mmid_صف_mapping {
    int32_t i1;
    int32_t i2;
};

ساخت ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

نامخلا sycl_ex = sycl::ext::oneapi::experimental;
ساخت ggml_backend_sycl_context {
    صحیح device;
    std::string name;
    optimize_feature opt_feature;

    قطار_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { خالی_اشاریہ } };

    صریح ggml_backend_sycl_context(صحیح device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    قطار_ptr stream(صحیح device, صحیح stream) {
        اگر (qptrs[device][stream] == خالی_اشاریہ) {
            qptrs[device][stream] = &(dpct::get_device(device).default_قطار());
        }
        واپس qptrs[device][stream];
    }

    قطار_ptr stream() {
        واپس stream(device, 0);
    }

#اگر GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        ثابت dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        واپس eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> دھارا_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream دھارا_dnnl(صحیح device, صحیح دھارا_2) {
        خودکار q = stream(device, دھارا_2);
        واپس دھارا_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        خودکار it = engine_map.find(qptr);
        اگر (it == engine_map.end()) {
            خودکار eng = make_engine(qptr);
            engine_map[qptr] = eng;
            واپس eng;
        }
        ورنہ
        {
            واپس it->second;
        }
    }
    dnnl::stream دھارا_dnnl(sycl::queue* qptr) {
        خودکار it = دھارا_map.find(qptr);
        اگر (it == دھارا_map.end()) {
            خودکار eng = engine_dnnl(qptr);
            خودکار stream = dnnl::sycl_interop::make_دھارا(eng, *qptr);
            دھارا_map[qptr] = stream;
            واپس stream;
        }
        ورنہ
        {
            واپس it->second;
        }
    }
    dnnl::stream دھارا_dnnl() {
        واپس دھارا_dnnl(device, 0);
    }
#اگر_ختم

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_صف_mapping> mmid_صف_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_جدول_وقت_host;

    ساکن std::unique_ptr<ggml_sycl_pool> new_pool_for_device(قطار_ptr qptr, صحیح device);

    ساکن std::unique_ptr<ggml_sycl_pool> new_pool_for_host(قطار_ptr qptr, صحیح device);

    ggml_sycl_pool & pool(صحیح device) {
        اگر (pools[device] == خالی_اشاریہ) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        واپس *pools[device];
    }

    ggml_sycl_pool & pool() {
        واپس pool(device);
    }

#اگر_تعریف_شدہ GGML_SYCL_گراف
    std::unique_ptr<sycl_ex::command_گراف<sycl_ex::گراف_state::executable>> exec_گراف = خالی_اشاریہ;
#اگر_ختم

    ggml_sycl_pool & host_pool(صحیح device) {
        اگر (host_pools[device] == خالی_اشاریہ) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        واپس *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { واپس host_pool(device); }
};

// common device functions

ساکن __dpct_inline__ اعشاری warp_reduce_مجموع(اعشاری x,
    ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), x, نقاب);
    }
    واپس x;
}

ساکن __dpct_inline__ sycl::float2
warp_reduce_مجموع(sycl::float2 a, ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        a.x() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), a.x(),
            نقاب);
        a.y() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), a.y(),
            نقاب);
    }
    واپس a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ صحیح warp_reduce_مجموع(صحیح x) {
  واپس sycl::reduce_over_گروہ(
      sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ اعشاری warp_reduce_مجموع(اعشاری x) {
#ہدایت unroll
  برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
    x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x, انحراف, چوڑائی);
  }
  واپس x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ اعشاری warp_reduce_مجموع(اعشاری x, ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
  برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
    x += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        شے_ct1.get_sub_گروہ(), x, انحراف);
  }
  واپس x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ sycl::float2 warp_reduce_مجموع(sycl::float2 a) {
#ہدایت unroll
  برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
    a.x() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a.x(), انحراف,
        چوڑائی);
    a.y() += dpct::ترتیب_بدلو_sub_گروہ_by_xor(
        sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a.y(), انحراف,
        چوڑائی);
  }
  واپس a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ sycl::half2 warp_reduce_مجموع(sycl::half2 a) {
#ہدایت unroll
  برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
    a = a + dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), a, انحراف,
                چوڑائی);
  }
  واپس a;
}

ساکن constexpr صحیح ggml_sycl_get_physical_warp_سائز() {
  // todo: for old iGPU + dGPU case, need to be changed.
  واپس WARP_سائز;
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ صحیح warp_reduce_all(صحیح x) {
    اگر (چوڑائی == ggml_sycl_get_physical_warp_سائز()) {
        واپس sycl::all_کا_گروہ(
            sycl::ext::oneapi::this_work_شے::get_sub_گروہ(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_شے::get_sub_گروہ()
                         .get_local_خطی_id())) ||
                x);
    } ورنہ {
#ہدایت unroll
        برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
            x = dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                    sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                    انحراف, چوڑائی) &&
                x;
        }
        واپس x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ صحیح warp_reduce_any(صحیح x) {
    اگر (چوڑائی == ggml_sycl_get_physical_warp_سائز()) {
        واپس sycl::any_کا_گروہ(
            sycl::ext::oneapi::this_work_شے::get_sub_گروہ(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_شے::get_sub_گروہ()
                         .get_local_خطی_id())) &&
                x);
    } ورنہ {
#ہدایت unroll
        برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
            x = dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                    sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                    انحراف, چوڑائی) ||
                x;
        }
        واپس x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
سانچہ <صحیح چوڑائی>
ساکن __dpct_inline__ اعشاری warp_reduce_max(اعشاری x) {
#ہدایت unroll
  برائے (صحیح انحراف = چوڑائی / 2; انحراف > 0; انحراف >>= 1) {
    x = sycl::fmax(x, dpct::ترتیب_بدلو_sub_گروہ_by_xor(
                          sycl::ext::oneapi::this_work_شے::get_sub_گروہ(), x,
                          انحراف, چوڑائی));
  }
  واپس x;
}

ساکن __dpct_inline__ اعشاری warp_reduce_max(اعشاری x,
    ثابت sycl::nd_شے<3>& شے_ct1) {
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        x = sycl::fmax(x, dpct::ترتیب_بدلو_sub_گروہ_by_xor(
            شے_ct1.get_sub_گروہ(), x, نقاب));
    }
    واپس x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
سانچہ<صحیح N>
__dpct_inline__ size_t calculate_انحراف(ثابت std::array<صحیح, N> & strides, ثابت std::array<صحیح, N> & indices) {
    size_t انحراف = 0;
#ہدایت unroll
    برائے (صحیح i = 0; i < N; i++) {
        خودکار اشاریہ_i = indices[i];
        انحراف += strides[i] * اشاریہ_i;
    }
    واپس انحراف;
}

// Helper for vec loading aligned data
سانچہ <قسم_نام Tp, صحیح n>
اندرونی sycl::vec<Tp, n> vec_aligned_بھرو(ثابت Tp* aligned_ptr) {
    واپس *reinterpret_cast<ثابت sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
سانچہ <قسم_نام Tp, صحیح dim>
ساکن __dpct_inline__ Tp* get_اشارہ_گر(sycl::local_accessor<Tp, dim> acc) {
    واپس acc.سانچہ get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_بلاک_عدد, int64_t بلاک_سائز_3);

constexpr size_t ceil_div(ثابت size_t m, ثابت size_t n) {
    واپس (m + n - 1) / n;
}

منطقی gpu_has_xmx(sycl::device &dev);

#اگر GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
اندرونی منطقی ggml_sycl_dnnl_has_optimized_gemm(صحیح device) {
    واپس ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#اگر_ختم

صحیح ggml_sycl_get_env(ثابت حرف *env_name, صحیح default_val);

سانچہ <صحیح N, جماعت T> std::string debug_get_array_str(ثابت std::string & prefix, ثابت T array[N]) {
    اگر (LIKELY(!g_ggml_sycl_debug)) {
        واپس "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    برائے (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    اگر constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    واپس ss.str();
}

اندرونی std::string debug_get_tensor_str(ثابت std::string &prefix,
        ثابت ggml_tensor *tensor, ثابت std::string &suffix = "") {
    std::stringstream ss;
    اگر (LIKELY(!g_ggml_sycl_debug)) { واپس ss.str(); }
    ss << prefix.c_str() << "=";
    اگر (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        اگر (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        اگر (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } ورنہ {
        ss << "nullptr";
    }
    ss << suffix;
    واپس ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
ساخت scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(ثابت std::string_view & func, ثابت std::string_view & func_suffix, ثابت ggml_tensor * dst,
                         std::size_t عدد_src, ثابت std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        اگر (LIKELY(!g_ggml_sycl_debug)) {
            واپس;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.ڈیٹا(), func_suffix.ڈیٹا());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        اگر (dst) {
            برائے (std::size_t i = 0; i < عدد_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.ڈیٹا());
    }

    scope_op_debug_print(ثابت std::string_view & func, ثابت ggml_tensor * dst, std::size_t عدد_src,
                         ثابت std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, عدد_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.ڈیٹا(), func_suffix.ڈیٹا()); }

  نجی:
    std::string_view func;
    std::string_view func_suffix;
};

ساکن __dpct_inline__ اعشاری get_alibi_ڈھلان(ثابت اعشاری    max_bias,
                                             ثابت uint32_t h,
                                             ثابت uint32_t n_سر_log2,
                                             ثابت اعشاری    m0,
                                             ثابت اعشاری    m1) {
    اگر (max_bias <= 0.0f) {
        واپس 1.0f;
    }
    ثابت اعشاری بنیاد = h < n_سر_log2 ? m0 : m1;
    ثابت صحیح   exph = h < n_سر_log2 ? h + 1 : 2*(h - n_سر_log2) + 1;

    واپس dpct::قوت(بنیاد, exph);
}

ساکن ثابت sycl::uint3 init_fastdiv_اقدار(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    جب (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    واپس sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
ساکن constexpr صحیح ggml_sycl_get_max_cpy_بائٹیں() {
    واپس 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
سانچہ <صحیح nbytes, صحیح alignment = 0>
ساکن __dpct_inline__ خالی ggml_sycl_memcpy_1(خالی * dst, ثابت خالی * src) {
    اگر constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr صحیح nb_فی_cpy = alignment == 0 ? nbytes : alignment;

#ہدایت unroll
    برائے (صحیح i = 0; i < nbytes/nb_فی_cpy; ++i) {
        اگر constexpr (nb_فی_cpy == 1) {
            ((حرف *) dst)[i] = ((ثابت حرف *) src)[i];
        } ورنہ اگر constexpr (nb_فی_cpy == 2) {
            ((مختصر *) dst)[i] = ((ثابت مختصر *) src)[i];
        } ورنہ اگر constexpr (nb_فی_cpy == 4) {
            ((صحیح *) dst)[i] = ((ثابت صحیح *) src)[i];
        } ورنہ اگر constexpr (nb_فی_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((ثابت sycl::int2 *) src)[i];
        } ورنہ اگر constexpr (nb_فی_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((ثابت sycl::int4 *) src)[i];
        } ورنہ {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
سانچہ <قسم_نام T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    واپس res;
}

ساکن __dpct_inline__ uint32_t fastdiv(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    ثابت uint32_t hi = sycl::mul_hi<غیرعلامتی>(n, fastdiv_اقدار.x());
    واپس (hi + n) >> fastdiv_اقدار.y();
}


سانچہ <قسم_نام T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<اعشاری>(x),static_cast<اعشاری>(y));
    واپس res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_قدر(static_cast<اعشاری>(H.x()), static_cast<اعشاری>(H.y()));
    واپس float2_قدر;
}

ساکن __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    ثابت uint32_t div_val = fastdiv(n, fastdiv_اقدار);
    ثابت uint32_t mod_val = n - div_val * fastdiv_اقدار.z();
    واپس sycl::uint2(div_val, mod_val);
}

ساکن __dpct_inline__ صحیح ggml_sycl_dp4a(ثابت صحیح a, ثابت صحیح b, صحیح c) {
    واپس dpct::dp4a(a, b, c);
}

ساکن __dpct_inline__ اعشاری ggml_sycl_e8m0_مقصد_fp32(uint8_t x) {
    uint32_t بٹ;
    اگر (x == 0) {
        بٹ = 0x00400000;
    } ورنہ {
        بٹ = (uint32_t) x << 23;
    }

    اعشاری نتیجہ;
    حافظہ_نقل(&نتیجہ, &بٹ, حجم(اعشاری));
    واپس نتیجہ;
}

sycl::float2 __dpct_inline__ __half22float2(ثابت sycl::half2 &H) {
    sycl::float2 float2_قدر(static_cast<اعشاری>(H.x()), static_cast<اعشاری>(H.y()));
    واپس float2_قدر;
}

اعشاری __dpct_inline__ __half2float(sycl::half H) {
    واپس static_cast<اعشاری>(H);
}

ساکن __dpct_inline__ خالی ggml_sycl_mad(اعشاری & acc, ثابت اعشاری v, ثابت اعشاری u) {
    acc += v*u;
}

ساکن __dpct_inline__ خالی ggml_sycl_mad(اعشاری & acc, ثابت sycl::float2 v, ثابت sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

ساکن __dpct_inline__ خالی ggml_sycl_mad(اعشاری & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#اگر_تعریف_شدہ GGML_SYCL_F16
    ثابت sycl::float2 tmp = (v * u).سانچہ convert<اعشاری, sycl::rounding_منوال::automatic>();
    acc += tmp.x() + tmp.y();
#ورنہ
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی ggml_sycl_mad(sycl::half2 & acc, ثابت sycl::half2 v, ثابت sycl::half2 u) {
#اگر_تعریف_شدہ GGML_SYCL_F16
    acc += v*u;
#ورنہ
    ثابت sycl::float2 tmpv = __half22float2(v);
    ثابت sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#اگر_ختم // GGML_SYCL_F16
}

سانچہ <صحیح n>
ساخت ggml_sycl_unroll {
    سانچہ <قسم_نام Func, قسم_نام... دلائل>
    خالی عامل()(ثابت Func & f, دلائل... دلائل_2) ثابت {
        f(n - 1, دلائل_2...);
        ggml_sycl_unroll<n - 1>{}(f, دلائل_2...);
    }
};

سانچہ <>
ساخت ggml_sycl_unroll<1> {
    سانچہ <قسم_نام Func, قسم_نام... دلائل>
    خالی عامل()(ثابت Func & f, دلائل... دلائل_2) ثابت {
        f(0, دلائل_2...);
    }
};

ساکن __dpct_inline__ sycl::half2 ggml_sycl_hmax2(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<اعشاری, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<اعشاری, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
    واپس ret;
}

ساکن __dpct_inline__ sycl::half ggml_sycl_hmax(ثابت sycl::half a, ثابت sycl::half b) {
    واپس sycl::vec<اعشاری, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<اعشاری, sycl::rounding_منوال::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<اعشاری, sycl::rounding_منوال::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_منوال::automatic>()[0];
}

ساکن __dpct_inline__ uint32_t __hgt2_mask(ثابت sycl::half2 a, ثابت sycl::half2 b) {
    ثابت uint32_t نقاب_کم  = 0x0000FFFF * (اعشاری(a[0]) > اعشاری(b[0]));
    ثابت uint32_t نقاب_بلند = 0xFFFF0000 * (اعشاری(a[1]) > اعشاری(b[1]));
    واپس نقاب_کم | نقاب_بلند;
}

ساکن __dpct_inline__ uint32_t fastmodulo(uint32_t n, ثابت sycl::uint3 fastdiv_اقدار) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    واپس n - fastdiv(n, fastdiv_اقدار) * fastdiv_اقدار.z();
}

ساکن منطقی fast_fp16_دستیاب(ثابت صحیح cc) {
    GGML_غیر_مستعمل(cc);
    واپس سچ;   //Intel GPUs always support FP16.
}

شمار جماعت بلاک_reduce_طریقہ {
    MAX,
    مجموع_3,
};

سانچہ<بلاک_reduce_طریقہ طریقہ_t, قسم_نام T, صحیح warp_سائز>
ساخت بلاک_reduce_policy;

سانچہ <قسم_نام T, قسم_نام... Ts>
اندرونی constexpr منطقی is_any = (std::is_same_v<T, Ts> || ...);

سانچہ<قسم_نام...>
اندرونی constexpr منطقی ggml_sycl_dependent_false_v = جھوٹ;

#تعریف_کرو WARP_32_سائز 32

سانچہ <قسم_نام T, صحیح warp_سائز> ساخت بلاک_reduce_policy<بلاک_reduce_طریقہ::مجموع_3, T, warp_سائز> {
    ساکن T reduce(T val) {
        اگر constexpr (is_any<T, اعشاری, sycl::float2, sycl::half2, صحیح>) {
            واپس warp_reduce_مجموع<warp_سائز>(val);
        } ورنہ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    ساکن T sentinel() {
        اگر constexpr (std::is_same_v<T, اعشاری>) {
            واپس 0.0f;
        } ورنہ اگر constexpr (std::is_same_v<T, sycl::float2>) {
            واپس sycl::float2(0.0f, 0.0f);
        } ورنہ اگر constexpr (std::is_same_v<T, sycl::half2>) {
            واپس sycl::half2(0.0f, 0.0f);
        } ورنہ اگر constexpr (std::is_same_v<T, صحیح>) {
            واپس 0;
        } ورنہ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

سانچہ <قسم_نام T, صحیح warp_سائز> ساخت بلاک_reduce_policy<بلاک_reduce_طریقہ::MAX, T, warp_سائز> {
    ساکن T reduce(T val) {
        اگر constexpr (is_any<T, اعشاری, sycl::half2>) {
            واپس warp_reduce_max<warp_سائز>(val);
        } ورنہ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    ساکن T sentinel() {
        اگر constexpr (std::is_same_v<T, اعشاری>) {
            واپس -INFINITY;
        } ورنہ اگر constexpr (std::is_same_v<T, sycl::half2>) {
            واپس sycl::half2(-INFINITY, -INFINITY);
        } ورنہ {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


سانچہ <بلاک_reduce_طریقہ reduce_طریقہ_t, صحیح warp_سائز, قسم_نام T>
ساکن T بلاک_reduce(T val, T * shared_vals, صحیح بلاک_سائز_template) {
    خودکار شے_ct1                 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    val                           = بلاک_reduce_policy<reduce_طریقہ_t, T,warp_سائز>::reduce(val);
    ثابت صحیح بلاک_سائز_3 = بلاک_سائز_template == 0 ? شے_ct1.get_local_range(2) : بلاک_سائز_template;
    ثابت صحیح nthreads = شے_ct1.get_local_range(2);
    ثابت صحیح nwarps = nthreads / WARP_سائز;

    اگر (بلاک_سائز_3 > warp_سائز) {
        أكد((بلاک_سائز_3 <= 1024) && (بلاک_سائز_3 % warp_سائز) == 0);
        ثابت صحیح warp_id = شے_ct1.get_local_id(2) / warp_سائز;
        ثابت صحیح lane_id = شے_ct1.get_local_id(2) % warp_سائز;
        اگر (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        شے_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_سائز;
        اعشاری tmp = 0.f;
        اگر (lane_id < (static_cast<صحیح>(بلاک_سائز_3) / warp_سائز)) {
            برائے (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_سائز];
            }
        }
        واپس بلاک_reduce_policy<reduce_طریقہ_t, T, warp_سائز>::reduce(tmp);
    }
    واپس val;
}

ساکن __dpct_inline__ اعشاری ggml_sycl_ue4m3_مقصد_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    اگر (x == 0 || x == 0x7F) {
        واپس 0.0f;
    }
    ثابت صحیح قوت_نما = (x >> 3) & 0xF;
    ثابت صحیح man = x & 0x7;
    اعشاری raw;
    اگر (قوت_نما == 0) {
        raw = man * (1.0f / 8.0f) * sycl::قوت(2.0f, -6.0f);
    } ورنہ {
        raw = (1.0f + man / 8.0f) * sycl::قوت(2.0f, (اعشاری) قوت_نما - 7.0f);
    }
    واپس raw * 0.5f;
}

#اگر_ختم // GGML_SYCL_COMMON_HPP
