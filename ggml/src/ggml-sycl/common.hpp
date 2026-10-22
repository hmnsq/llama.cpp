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

#నిర్వచితం_కాకపోతే GGML_SYCL_COMMON_HPP
#నిర్వచించు GGML_SYCL_COMMON_HPP

#అన్తర్భావ <cstddef>
#అన్తర్భావ <fstream>
#అన్తర్భావ <iostream>
#అన్తర్భావ <string>

#అన్తర్భావ "base.hpp"
#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-sycl.h"
#అన్తర్భావ "presets.hpp"
#అన్తర్భావ "type.hpp"
#అన్తర్భావ "sycl_hw.hpp"
#అన్తర్భావ "memtrace.hpp"

నామాకాశ syclexp = sycl::ext::oneapi::experimental;

#ఒకవేళ defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #అన్తర్భావ <sycl/ext/oneapi/bfloat16.hpp>
    #నిర్వచితం_కాకపోతే GGML_SYCL_HAS_BF16
        #నిర్వచించు GGML_SYCL_HAS_BF16
    #అయితే_ముగింపు
#అయితే_ముగింపు

#ఒకవేళ defined(__INTEL_LLVM_COMPILER)
    #నిర్వచించు GGML_SYCL_DMMV_HAS_ESIMD
#అయితే_ముగింపు

#ఒకవేళ GGML_SYCL_DNNL
#అన్తర్భావ "dnnl.hpp"
#అన్తర్భావ "dnnl_sycl.hpp"
#అయితే_ముగింపు

#నిర్వచించు GGML_COMMON_DECL_SYCL
#నిర్వచించు GGML_COMMON_IMPL_SYCL
#నిర్వచించు SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#నిర్వచించు SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building
#నిర్వచించు GGML_SYCL_FA_ALL_QUANTS //define it to enable all quantization types in flash attention. undefine it to only support F16, Q4_0 and Q8_0 in flash attention.

/* suppress warning spam */
#సూచన clang diagnostic నెట్టు
#సూచన clang diagnostic ignored "-Wnested-anon-types"
#అన్తర్భావ "ggml-common.h"
#సూచన clang diagnostic తీయి
#అన్తర్భావ "ggml-impl.h"

శూన్య* ggml_sycl_host_malloc(size_t size);
శూన్య ggml_sycl_host_free(శూన్య* ptr);


బాహ్య పూర్ణాంకం g_ggml_sycl_debug;
బాహ్య పూర్ణాంకం g_ggml_sycl_enable_optimize;
బాహ్య పూర్ణాంకం g_ggml_sycl_enable_fusion;
బాహ్య పూర్ణాంకం g_ggml_sycl_enable_esimd;
బాహ్య పూర్ణాంకం g_ggml_sycl_mmvq_wide;
బాహ్య పూర్ణాంకం g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
గణన ggml_sycl_xmx_gather_type {
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
స్థిర constexpr పూర్ణాంకం GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
బాహ్య పూర్ణాంకం g_ggml_sycl_xmx_gather_types;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
స్థిర constexpr పూర్ణాంకం GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
బాహ్య పూర్ణాంకం g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
గణన ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#నిర్వచితమైతే GGML_SYCL_F16
స్థిర constexpr పూర్ణాంకం GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#లేకపోతే
స్థిర constexpr పూర్ణాంకం GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#అయితే_ముగింపు
బాహ్య పూర్ణాంకం g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
బాహ్య పూర్ణాంకం g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
స్థిర అంతఃవరుస int32_t ggml_sycl_src1_prec(స్థిరరాశి ggml_tensor * dst) {
    తిరిగి g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       dst->op_params[3];
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
స్థిర అంతఃవరుస తార్కిక ggml_sycl_src1_f16_ok(స్థిరరాశి ggml_tensor * dst) {
    స్థిరరాశి int32_t src1_prec = ggml_sycl_src1_prec(dst);
    తిరిగి g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
బాహ్య పూర్ణాంకం g_ggml_sycl_enable_flash_attention;
బాహ్య పూర్ణాంకం g_ggml_sycl_dev2dev_memcpy;
బాహ్య పూర్ణాంకం g_ggml_sycl_fa_onednn;
బాహ్య పూర్ణాంకం g_ggml_sycl_fa_onednn_max_kv;
బాహ్య పూర్ణాంకం g_ggml_sycl_enable_mkl_fa;
బాహ్య పూర్ణాంకం g_ggml_sycl_memtrace;
బాహ్య పూర్ణాంకం g_ggml_sycl_memtrace_అడుగు;


#నిర్వచించు CHECK_TRY_ERROR(సమాసం)                                            \
  [&]() {                                                                \
    ప్రయత్నించు {                                                                \
      సమాసం;                                                              \
      తిరిగి dpct::success;                                              \
    } పట్టుకో (std::exception స్థిరరాశి& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::పంక్తి_ముగింపు;                                            \
      తిరిగి dpct::default_error;                                        \
    }                                                                    \
  }()


#నిర్వచించు __SYCL_ARCH__ DPCT_COMPATIBILITY_తాత్కాలిక
#నిర్వచించు VER_4VEC 610 // todo for hardware optimize.
#నిర్వచించు VER_GEN9 700 // todo for hardware optimize.
#నిర్వచించు VER_GEN12 1000000 // todo for hardware optimize.
#నిర్వచించు VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#నిర్వచించు GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// define for XMX in Intel GPU
// TODO: currently, it's not used for XMX really.
#ఒకవేళ !defined(GGML_SYCL_FORCE_MMQ)
    #నిర్వచించు SYCL_USE_XMX
#అయితే_ముగింపు

// max batch size to use MMQ kernels when tensor cores are available
#నిర్వచించు MMQ_MAX_BATCH_కొలత 32

// dmmv = dequantize_mul_mat_vec
#నిర్వచితం_కాకపోతే GGML_SYCL_DMMV_X
#నిర్వచించు GGML_SYCL_DMMV_X 32
#అయితే_ముగింపు
#నిర్వచితం_కాకపోతే GGML_SYCL_MMV_Y
#నిర్వచించు GGML_SYCL_MMV_Y 1
#అయితే_ముగింపు

రకనిర్వచనం sycl::queue *వరుస_ptr;

గణన ggml_sycl_backend_gpu_బహుళకం {
  SYCL_UNSET_GPU_బహుళకం = -1,
  SYCL_SINGLE_GPU_బహుళకం = 0,
  SYCL_MUL_GPU_బహుళకం
};

గణన ggml_sycl_dev2dev_memcpy_బహుళకం {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(పరిమాణం(sycl::half) == పరిమాణం(ggml_fp16_t), "wrong fp16 size");

స్థిర శూన్య crash() {
  పూర్ణాంకం* ptr = ఖాళీ;
  *ptr = 0;
}

[[noreturn]] స్థిర శూన్య ggml_sycl_error(
    స్థిరరాశి అక్షరం* stmt,
    స్థిరరాశి అక్షరం* func,
    స్థిరరాశి అక్షరం* file,
    స్థిరరాశి పూర్ణాంకం పంక్తి,
    స్థిరరాశి అక్షరం* సందేశం_2) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, సందేశం_2);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, పంక్తి);
  GGML_ABORT("SYCL error");
}

#నిర్వచించు SYCL_CHECK(err)                                                                                    \
    చేయి {                                                                                                   \
        స్వయంచాలక err_ = (err);                                                                                 \
        ఒకవేళ (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } అయితే (0)

#ఒకవేళ DPCT_COMPAT_RT_సంస్కరణ >= 11100
#నిర్వచించు GGML_SYCL_ASSUME(x) __builtin_assume(x)
#లేకపోతే
#నిర్వచించు GGML_SYCL_ASSUME(x)
#అయితే_ముగింపు // DPCT_COMPAT_RT_VERSION >= 11100

#నిర్వచితమైతే GGML_SYCL_F16
రకనిర్వచనం sycl::half dfloat; // dequantize float
రకనిర్వచనం sycl::half2 dfloat2;
#లేకపోతే
రకనిర్వచనం దశాంశం dfloat; // dequantize float
రకనిర్వచనం sycl::float2 dfloat2;
#అయితే_ముగింపు // GGML_SYCL_F16

#నిర్వచించు MMVQ_MAX_BATCH_కొలత  8

స్థిర పూర్ణాంకం g_all_sycl_device_లెక్క = -1;
స్థిర తార్కిక g_ggml_backend_sycl_బఫరు_type_initialized = అబద్ధం;

స్థిర ggml_sycl_backend_gpu_బహుళకం g_ggml_sycl_backend_gpu_బహుళకం =
    SYCL_UNSET_GPU_బహుళకం;

స్థిర శూన్య* g_scratch_బఫరు = శూన్యసూచిక;
స్థిర size_t g_scratch_కొలత = 0; // disabled by default
స్థిర size_t g_scratch_స్థానభ్రంశం = 0;

[[noreturn]] స్థిర అంతఃవరుస శూన్య bad_arch(స్థిరరాశి sycl::stream& ప్రవాహం_ct1) {
  ప్రవాహం_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::నిష్క్రమించు(1);

  (శూన్య)bad_arch; // suppress unused function warning
}

పూర్ణాంకం get_current_device_id();

అంతఃవరుస పూర్ణాంకం ggml_sycl_get_device() {
    తిరిగి get_current_device_id();
}

అంతఃవరుస dpct::err0 ggml_sycl_సమితి_device(స్థిరరాశి పూర్ణాంకం device) ప్రయత్నించు {
  పూర్ణాంకం current_device_id;
  SYCL_CHECK(CHECK_TRY_ERROR(current_device_id = get_current_device_id()));

  // GGML_SYCL_DEBUG("ggml_sycl_set_device device_id=%d,
  // current_device_id=%d\n", device, current_device);
  ఒకవేళ (device == current_device_id) {
    తిరిగి 0;
  }

  తిరిగి CHECK_TRY_ERROR(dpct::select_device(device));
} పట్టుకో (sycl::exception స్థిరరాశి& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  crash();
  std::నిష్క్రమించు(1);
}

//////////////////////
నిర్మాణం optimize_feature {
    తార్కిక reorder=అబద్ధం;
    తార్కిక onednn_optimized_gemm=అబద్ధం;
};

నిర్మాణం sycl_device_info {
    పూర్ణాంకం cc;  // compute capability
    పూర్ణాంకం nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    పూర్ణాంకం warp_కొలత;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    పూర్ణాంకం max_wg_చొప్పున_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    తార్కిక    vmm;                // virtual memory support
    తార్కిక    l0_device_type_valid;
    తార్కిక    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  మొత్తం_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    తార్కిక    usm_system_support; // support for USM system allocations
};


నిర్మాణం ggml_sycl_device_info {
    పూర్ణాంకం device_లెక్క;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> default_tensor_విభజించు = {};

    పూర్ణాంకం max_work_సమూహం_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    తార్కిక ext_oneapi_level_zero = నిజం; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

స్థిరరాశి ggml_sycl_device_info & ggml_sycl_info();

స్థిర constexpr size_t SYCL_బఫరు_ALIGNMENT = 128;

నిర్మాణం ggml_sycl_pool {
    మాయా ~ggml_sycl_pool() = అప్రమేయ;

    మాయా శూన్య * alloc(size_t size, size_t * actual_కొలత) = 0;
    మాయా శూన్య విడుదల(శూన్య * ptr, size_t size) = 0;
};

మూస<రకం_పేరు T>
నిర్మాణం ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = శూన్యసూచిక;
    T * ptr = శూన్యసూచిక;
    size_t actual_కొలత = 0;

    స్పష్టమైన ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        ఒకవేళ (ptr != శూన్యసూచిక) {
            pool->విడుదల(ptr, actual_కొలత);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != శూన్యసూచిక);
        ఒకవేళ (ptr)
            pool->విడుదల(ptr, actual_కొలత);
        ptr = (T *) pool->alloc(size * పరిమాణం(T), &ఇది->actual_కొలత);
        తిరిగి ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != శూన్యసూచిక);
        GGML_ASSERT(ptr == శూన్యసూచిక);
        ptr = (T *) pool->alloc(size * పరిమాణం(T), &ఇది->actual_కొలత);
        తిరిగి ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        ఇది->pool = &pool;
        తిరిగి alloc(size);
    }

    T * get() {
        తిరిగి ptr;
    }

    ggml_sycl_pool_alloc() = అప్రమేయ;
    ggml_sycl_pool_alloc(స్థిరరాశి ggml_sycl_pool_alloc &) = తొలగించు;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = తొలగించు;
    ggml_sycl_pool_alloc& క్రియాకారకం=(స్థిరరాశి ggml_sycl_pool_alloc &) = తొలగించు;
    ggml_sycl_pool_alloc& క్రియాకారకం=(ggml_sycl_pool_alloc &&) = తొలగించు;
};

// backend interface

నిర్మాణం ggml_tensor_extra_gpu {
  శూన్య* దత్తాంశం_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

బాహ్య పూర్ణాంకం g_ggml_sycl_use_level_zero_api;
శూన్య * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_ప్రత్యక్ష);
శూన్య ggml_sycl_free_device(శూన్య *ptr, sycl::queue &q);

శూన్య release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<వరుస_ptr> streams={});

నిర్మాణం mmid_అడ్డువరుస_mapping {
    int32_t i1;
    int32_t i2;
};

నిర్మాణం ggml_sycl_gg_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

నామాకాశ sycl_ex = sycl::ext::oneapi::experimental;
నిర్మాణం ggml_backend_sycl_context {
    పూర్ణాంకం device;
    std::string name;
    optimize_feature opt_feature;

    వరుస_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { శూన్యసూచిక } };

    స్పష్టమైన ggml_backend_sycl_context(పూర్ణాంకం device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    వరుస_ptr stream(పూర్ణాంకం device, పూర్ణాంకం stream) {
        ఒకవేళ (qptrs[device][stream] == శూన్యసూచిక) {
            qptrs[device][stream] = &(dpct::get_device(device).default_వరుస());
        }
        తిరిగి qptrs[device][stream];
    }

    వరుస_ptr stream() {
        తిరిగి stream(device, 0);
    }

#ఒకవేళ GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        స్థిరరాశి dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        తిరిగి eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> ప్రవాహం_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream ప్రవాహం_dnnl(పూర్ణాంకం device, పూర్ణాంకం ప్రవాహం_2) {
        స్వయంచాలక q = stream(device, ప్రవాహం_2);
        తిరిగి ప్రవాహం_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        స్వయంచాలక it = engine_map.find(qptr);
        ఒకవేళ (it == engine_map.end()) {
            స్వయంచాలక eng = make_engine(qptr);
            engine_map[qptr] = eng;
            తిరిగి eng;
        }
        లేకపోతే
        {
            తిరిగి it->second;
        }
    }
    dnnl::stream ప్రవాహం_dnnl(sycl::queue* qptr) {
        స్వయంచాలక it = ప్రవాహం_map.find(qptr);
        ఒకవేళ (it == ప్రవాహం_map.end()) {
            స్వయంచాలక eng = engine_dnnl(qptr);
            స్వయంచాలక stream = dnnl::sycl_interop::make_ప్రవాహం(eng, *qptr);
            ప్రవాహం_map[qptr] = stream;
            తిరిగి stream;
        }
        లేకపోతే
        {
            తిరిగి it->second;
        }
    }
    dnnl::stream ప్రవాహం_dnnl() {
        తిరిగి ప్రవాహం_dnnl(device, 0);
    }
#అయితే_ముగింపు

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_అడ్డువరుస_mapping> mmid_అడ్డువరుస_mapping_host;
    std::vector<ggml_sycl_gg_tile> mmid_tile_కాలపట్టిక_host;

    స్థిర std::unique_ptr<ggml_sycl_pool> new_pool_for_device(వరుస_ptr qptr, పూర్ణాంకం device);

    స్థిర std::unique_ptr<ggml_sycl_pool> new_pool_for_host(వరుస_ptr qptr, పూర్ణాంకం device);

    ggml_sycl_pool & pool(పూర్ణాంకం device) {
        ఒకవేళ (pools[device] == శూన్యసూచిక) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        తిరిగి *pools[device];
    }

    ggml_sycl_pool & pool() {
        తిరిగి pool(device);
    }

#నిర్వచితమైతే GGML_SYCL_లేఖాచిత్రం
    std::unique_ptr<sycl_ex::command_లేఖాచిత్రం<sycl_ex::లేఖాచిత్రం_state::executable>> exec_లేఖాచిత్రం = శూన్యసూచిక;
#అయితే_ముగింపు

    ggml_sycl_pool & host_pool(పూర్ణాంకం device) {
        ఒకవేళ (host_pools[device] == శూన్యసూచిక) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        తిరిగి *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { తిరిగి host_pool(device); }
};

// common device functions

స్థిర __dpct_inline__ దశాంశం warp_reduce_కూడిక(దశాంశం x,
    స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1) {
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        x += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), x, ముసుగు);
    }
    తిరిగి x;
}

స్థిర __dpct_inline__ sycl::float2
warp_reduce_కూడిక(sycl::float2 a, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1) {
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        a.x() += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), a.x(),
            ముసుగు);
        a.y() += dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), a.y(),
            ముసుగు);
    }
    తిరిగి a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ పూర్ణాంకం warp_reduce_కూడిక(పూర్ణాంకం x) {
  తిరిగి sycl::reduce_over_సమూహం(
      sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ దశాంశం warp_reduce_కూడిక(దశాంశం x) {
#సూచన unroll
  కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
    x += dpct::క్రమమార్పు_sub_సమూహం_by_xor(
        sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), x, స్థానభ్రంశం, వెడల్పు);
  }
  తిరిగి x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ దశాంశం warp_reduce_కూడిక(దశాంశం x, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1) {
#సూచన unroll
  కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
    x += dpct::క్రమమార్పు_sub_సమూహం_by_xor(
        అంశం_ct1.get_sub_సమూహం(), x, స్థానభ్రంశం);
  }
  తిరిగి x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ sycl::float2 warp_reduce_కూడిక(sycl::float2 a) {
#సూచన unroll
  కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
    a.x() += dpct::క్రమమార్పు_sub_సమూహం_by_xor(
        sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), a.x(), స్థానభ్రంశం,
        వెడల్పు);
    a.y() += dpct::క్రమమార్పు_sub_సమూహం_by_xor(
        sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), a.y(), స్థానభ్రంశం,
        వెడల్పు);
  }
  తిరిగి a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ sycl::half2 warp_reduce_కూడిక(sycl::half2 a) {
#సూచన unroll
  కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
    a = a + dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), a, స్థానభ్రంశం,
                వెడల్పు);
  }
  తిరిగి a;
}

స్థిర constexpr పూర్ణాంకం ggml_sycl_get_physical_warp_కొలత() {
  // todo: for old iGPU + dGPU case, need to be changed.
  తిరిగి WARP_కొలత;
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ పూర్ణాంకం warp_reduce_all(పూర్ణాంకం x) {
    ఒకవేళ (వెడల్పు == ggml_sycl_get_physical_warp_కొలత()) {
        తిరిగి sycl::all_యొక్క_సమూహం(
            sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం()
                         .get_local_రేఖీయ_id())) ||
                x);
    } లేకపోతే {
#సూచన unroll
        కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
            x = dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                    sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), x,
                    స్థానభ్రంశం, వెడల్పు) &&
                x;
        }
        తిరిగి x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ పూర్ణాంకం warp_reduce_any(పూర్ణాంకం x) {
    ఒకవేళ (వెడల్పు == ggml_sycl_get_physical_warp_కొలత()) {
        తిరిగి sycl::any_యొక్క_సమూహం(
            sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం()
                         .get_local_రేఖీయ_id())) &&
                x);
    } లేకపోతే {
#సూచన unroll
        కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
            x = dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                    sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), x,
                    స్థానభ్రంశం, వెడల్పు) ||
                x;
        }
        తిరిగి x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
మూస <పూర్ణాంకం వెడల్పు>
స్థిర __dpct_inline__ దశాంశం warp_reduce_max(దశాంశం x) {
#సూచన unroll
  కొరకు (పూర్ణాంకం స్థానభ్రంశం = వెడల్పు / 2; స్థానభ్రంశం > 0; స్థానభ్రంశం >>= 1) {
    x = sycl::fmax(x, dpct::క్రమమార్పు_sub_సమూహం_by_xor(
                          sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం(), x,
                          స్థానభ్రంశం, వెడల్పు));
  }
  తిరిగి x;
}

స్థిర __dpct_inline__ దశాంశం warp_reduce_max(దశాంశం x,
    స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1) {
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        x = sycl::fmax(x, dpct::క్రమమార్పు_sub_సమూహం_by_xor(
            అంశం_ct1.get_sub_సమూహం(), x, ముసుగు));
    }
    తిరిగి x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
మూస<పూర్ణాంకం N>
__dpct_inline__ size_t calculate_స్థానభ్రంశం(స్థిరరాశి std::array<పూర్ణాంకం, N> & strides, స్థిరరాశి std::array<పూర్ణాంకం, N> & indices) {
    size_t స్థానభ్రంశం = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < N; i++) {
        స్వయంచాలక సూచీ_i = indices[i];
        స్థానభ్రంశం += strides[i] * సూచీ_i;
    }
    తిరిగి స్థానభ్రంశం;
}

// Helper for vec loading aligned data
మూస <రకం_పేరు Tp, పూర్ణాంకం n>
అంతఃవరుస sycl::vec<Tp, n> vec_aligned_ఎక్కించు(స్థిరరాశి Tp* aligned_ptr) {
    తిరిగి *reinterpret_cast<స్థిరరాశి sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
మూస <రకం_పేరు Tp, పూర్ణాంకం dim>
స్థిర __dpct_inline__ Tp* get_సూచిక(sycl::local_accessor<Tp, dim> acc) {
    తిరిగి acc.మూస get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_ఖండం_అంకె, int64_t ఖండం_కొలత_3);

constexpr size_t ceil_div(స్థిరరాశి size_t m, స్థిరరాశి size_t n) {
    తిరిగి (m + n - 1) / n;
}

తార్కిక gpu_has_xmx(sycl::device &dev);

#ఒకవేళ GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
అంతఃవరుస తార్కిక ggml_sycl_dnnl_has_optimized_gemm(పూర్ణాంకం device) {
    తిరిగి ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#అయితే_ముగింపు

పూర్ణాంకం ggml_sycl_get_env(స్థిరరాశి అక్షరం *env_name, పూర్ణాంకం default_val);

మూస <పూర్ణాంకం N, తరగతి T> std::string debug_get_array_str(స్థిరరాశి std::string & prefix, స్థిరరాశి T array[N]) {
    ఒకవేళ (LIKELY(!g_ggml_sycl_debug)) {
        తిరిగి "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    కొరకు (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    ఒకవేళ constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    తిరిగి ss.str();
}

అంతఃవరుస std::string debug_get_tensor_str(స్థిరరాశి std::string &prefix,
        స్థిరరాశి ggml_tensor *tensor, స్థిరరాశి std::string &suffix = "") {
    std::stringstream ss;
    ఒకవేళ (LIKELY(!g_ggml_sycl_debug)) { తిరిగి ss.str(); }
    ss << prefix.c_str() << "=";
    ఒకవేళ (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        ఒకవేళ (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        ఒకవేళ (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } లేకపోతే {
        ss << "nullptr";
    }
    ss << suffix;
    తిరిగి ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
నిర్మాణం scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(స్థిరరాశి std::string_view & func, స్థిరరాశి std::string_view & func_suffix, స్థిరరాశి ggml_tensor * dst,
                         std::size_t అంకె_src, స్థిరరాశి std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        ఒకవేళ (LIKELY(!g_ggml_sycl_debug)) {
            తిరిగి;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.దత్తాంశం(), func_suffix.దత్తాంశం());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        ఒకవేళ (dst) {
            కొరకు (std::size_t i = 0; i < అంకె_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.దత్తాంశం());
    }

    scope_op_debug_print(స్థిరరాశి std::string_view & func, స్థిరరాశి ggml_tensor * dst, std::size_t అంకె_src,
                         స్థిరరాశి std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, అంకె_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.దత్తాంశం(), func_suffix.దత్తాంశం()); }

  అంతరంగ:
    std::string_view func;
    std::string_view func_suffix;
};

స్థిర __dpct_inline__ దశాంశం get_alibi_వాలు(స్థిరరాశి దశాంశం    max_bias,
                                             స్థిరరాశి uint32_t h,
                                             స్థిరరాశి uint32_t n_తల_log2,
                                             స్థిరరాశి దశాంశం    m0,
                                             స్థిరరాశి దశాంశం    m1) {
    ఒకవేళ (max_bias <= 0.0f) {
        తిరిగి 1.0f;
    }
    స్థిరరాశి దశాంశం పునాది = h < n_తల_log2 ? m0 : m1;
    స్థిరరాశి పూర్ణాంకం   exph = h < n_తల_log2 ? h + 1 : 2*(h - n_తల_log2) + 1;

    తిరిగి dpct::ఘాతం(పునాది, exph);
}

స్థిర స్థిరరాశి sycl::uint3 init_fastdiv_విలువలు(uint32_t d) {
    GGML_ASSERT(d != 0);

    uint32_t L = 0;
    అయితే (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    తిరిగి sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
స్థిర constexpr పూర్ణాంకం ggml_sycl_get_max_cpy_బైట్లు() {
    తిరిగి 16;
}

// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
మూస <పూర్ణాంకం nbytes, పూర్ణాంకం alignment = 0>
స్థిర __dpct_inline__ శూన్య ggml_sycl_memcpy_1(శూన్య * dst, స్థిరరాశి శూన్య * src) {
    ఒకవేళ constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr పూర్ణాంకం nb_చొప్పున_cpy = alignment == 0 ? nbytes : alignment;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < nbytes/nb_చొప్పున_cpy; ++i) {
        ఒకవేళ constexpr (nb_చొప్పున_cpy == 1) {
            ((అక్షరం *) dst)[i] = ((స్థిరరాశి అక్షరం *) src)[i];
        } లేకపోతే ఒకవేళ constexpr (nb_చొప్పున_cpy == 2) {
            ((పొట్టి *) dst)[i] = ((స్థిరరాశి పొట్టి *) src)[i];
        } లేకపోతే ఒకవేళ constexpr (nb_చొప్పున_cpy == 4) {
            ((పూర్ణాంకం *) dst)[i] = ((స్థిరరాశి పూర్ణాంకం *) src)[i];
        } లేకపోతే ఒకవేళ constexpr (nb_చొప్పున_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((స్థిరరాశి sycl::int2 *) src)[i];
        } లేకపోతే ఒకవేళ constexpr (nb_చొప్పున_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((స్థిరరాశి sycl::int4 *) src)[i];
        } లేకపోతే {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
మూస <రకం_పేరు T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    తిరిగి res;
}

స్థిర __dpct_inline__ uint32_t fastdiv(uint32_t n, స్థిరరాశి sycl::uint3 fastdiv_విలువలు) {
    స్థిరరాశి uint32_t hi = sycl::mul_hi<గుర్తులేని>(n, fastdiv_విలువలు.x());
    తిరిగి (hi + n) >> fastdiv_విలువలు.y();
}


మూస <రకం_పేరు T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<దశాంశం>(x),static_cast<దశాంశం>(y));
    తిరిగి res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_విలువ(static_cast<దశాంశం>(H.x()), static_cast<దశాంశం>(H.y()));
    తిరిగి float2_విలువ;
}

స్థిర __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, స్థిరరాశి sycl::uint3 fastdiv_విలువలు) {
    స్థిరరాశి uint32_t div_val = fastdiv(n, fastdiv_విలువలు);
    స్థిరరాశి uint32_t mod_val = n - div_val * fastdiv_విలువలు.z();
    తిరిగి sycl::uint2(div_val, mod_val);
}

స్థిర __dpct_inline__ పూర్ణాంకం ggml_sycl_dp4a(స్థిరరాశి పూర్ణాంకం a, స్థిరరాశి పూర్ణాంకం b, పూర్ణాంకం c) {
    తిరిగి dpct::dp4a(a, b, c);
}

స్థిర __dpct_inline__ దశాంశం ggml_sycl_e8m0_గమ్యం_fp32(uint8_t x) {
    uint32_t బిట్లు;
    ఒకవేళ (x == 0) {
        బిట్లు = 0x00400000;
    } లేకపోతే {
        బిట్లు = (uint32_t) x << 23;
    }

    దశాంశం ఫలితం;
    జ్ఞాపకం_నకలు(&ఫలితం, &బిట్లు, పరిమాణం(దశాంశం));
    తిరిగి ఫలితం;
}

sycl::float2 __dpct_inline__ __half22float2(స్థిరరాశి sycl::half2 &H) {
    sycl::float2 float2_విలువ(static_cast<దశాంశం>(H.x()), static_cast<దశాంశం>(H.y()));
    తిరిగి float2_విలువ;
}

దశాంశం __dpct_inline__ __half2float(sycl::half H) {
    తిరిగి static_cast<దశాంశం>(H);
}

స్థిర __dpct_inline__ శూన్య ggml_sycl_mad(దశాంశం & acc, స్థిరరాశి దశాంశం v, స్థిరరాశి దశాంశం u) {
    acc += v*u;
}

స్థిర __dpct_inline__ శూన్య ggml_sycl_mad(దశాంశం & acc, స్థిరరాశి sycl::float2 v, స్థిరరాశి sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

స్థిర __dpct_inline__ శూన్య ggml_sycl_mad(దశాంశం & acc, స్థిరరాశి sycl::half2 v, స్థిరరాశి sycl::half2 u) {
#నిర్వచితమైతే GGML_SYCL_F16
    స్థిరరాశి sycl::float2 tmp = (v * u).మూస convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    acc += tmp.x() + tmp.y();
#లేకపోతే
    స్థిరరాశి sycl::float2 tmpv = __half22float2(v);
    స్థిరరాశి sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య ggml_sycl_mad(sycl::half2 & acc, స్థిరరాశి sycl::half2 v, స్థిరరాశి sycl::half2 u) {
#నిర్వచితమైతే GGML_SYCL_F16
    acc += v*u;
#లేకపోతే
    స్థిరరాశి sycl::float2 tmpv = __half22float2(v);
    స్థిరరాశి sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#అయితే_ముగింపు // GGML_SYCL_F16
}

మూస <పూర్ణాంకం n>
నిర్మాణం ggml_sycl_unroll {
    మూస <రకం_పేరు Func, రకం_పేరు... వాదనలు>
    శూన్య క్రియాకారకం()(స్థిరరాశి Func & f, వాదనలు... వాదనలు_2) స్థిరరాశి {
        f(n - 1, వాదనలు_2...);
        ggml_sycl_unroll<n - 1>{}(f, వాదనలు_2...);
    }
};

మూస <>
నిర్మాణం ggml_sycl_unroll<1> {
    మూస <రకం_పేరు Func, రకం_పేరు... వాదనలు>
    శూన్య క్రియాకారకం()(స్థిరరాశి Func & f, వాదనలు... వాదనలు_2) స్థిరరాశి {
        f(0, వాదనలు_2...);
    }
};

స్థిర __dpct_inline__ sycl::half2 ggml_sycl_hmax2(స్థిరరాశి sycl::half2 a, స్థిరరాశి sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<దశాంశం, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_బహుళకం::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<దశాంశం, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_బహుళకం::automatic>()[0];
    తిరిగి ret;
}

స్థిర __dpct_inline__ sycl::half ggml_sycl_hmax(స్థిరరాశి sycl::half a, స్థిరరాశి sycl::half b) {
    తిరిగి sycl::vec<దశాంశం, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_బహుళకం::automatic>()[0];
}

స్థిర __dpct_inline__ uint32_t __hgt2_mask(స్థిరరాశి sycl::half2 a, స్థిరరాశి sycl::half2 b) {
    స్థిరరాశి uint32_t ముసుగు_దిగువ  = 0x0000FFFF * (దశాంశం(a[0]) > దశాంశం(b[0]));
    స్థిరరాశి uint32_t ముసుగు_ఎగువ = 0xFFFF0000 * (దశాంశం(a[1]) > దశాంశం(b[1]));
    తిరిగి ముసుగు_దిగువ | ముసుగు_ఎగువ;
}

స్థిర __dpct_inline__ uint32_t fastmodulo(uint32_t n, స్థిరరాశి sycl::uint3 fastdiv_విలువలు) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    తిరిగి n - fastdiv(n, fastdiv_విలువలు) * fastdiv_విలువలు.z();
}

స్థిర తార్కిక fast_fp16_అందుబాటు(స్థిరరాశి పూర్ణాంకం cc) {
    GGML_వాడని(cc);
    తిరిగి నిజం;   //Intel GPUs always support FP16.
}

గణన తరగతి ఖండం_reduce_పద్ధతి {
    MAX,
    కూడిక_3,
};

మూస<ఖండం_reduce_పద్ధతి పద్ధతి_t, రకం_పేరు T, పూర్ణాంకం warp_కొలత>
నిర్మాణం ఖండం_reduce_policy;

మూస <రకం_పేరు T, రకం_పేరు... Ts>
అంతఃవరుస constexpr తార్కిక is_any = (std::is_same_v<T, Ts> || ...);

మూస<రకం_పేరు...>
అంతఃవరుస constexpr తార్కిక ggml_sycl_dependent_false_v = అబద్ధం;

#నిర్వచించు WARP_32_కొలత 32

మూస <రకం_పేరు T, పూర్ణాంకం warp_కొలత> నిర్మాణం ఖండం_reduce_policy<ఖండం_reduce_పద్ధతి::కూడిక_3, T, warp_కొలత> {
    స్థిర T reduce(T val) {
        ఒకవేళ constexpr (is_any<T, దశాంశం, sycl::float2, sycl::half2, పూర్ణాంకం>) {
            తిరిగి warp_reduce_కూడిక<warp_కొలత>(val);
        } లేకపోతే {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    స్థిర T sentinel() {
        ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
            తిరిగి 0.0f;
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, sycl::float2>) {
            తిరిగి sycl::float2(0.0f, 0.0f);
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, sycl::half2>) {
            తిరిగి sycl::half2(0.0f, 0.0f);
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, పూర్ణాంకం>) {
            తిరిగి 0;
        } లేకపోతే {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

మూస <రకం_పేరు T, పూర్ణాంకం warp_కొలత> నిర్మాణం ఖండం_reduce_policy<ఖండం_reduce_పద్ధతి::MAX, T, warp_కొలత> {
    స్థిర T reduce(T val) {
        ఒకవేళ constexpr (is_any<T, దశాంశం, sycl::half2>) {
            తిరిగి warp_reduce_max<warp_కొలత>(val);
        } లేకపోతే {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    స్థిర T sentinel() {
        ఒకవేళ constexpr (std::is_same_v<T, దశాంశం>) {
            తిరిగి -INFINITY;
        } లేకపోతే ఒకవేళ constexpr (std::is_same_v<T, sycl::half2>) {
            తిరిగి sycl::half2(-INFINITY, -INFINITY);
        } లేకపోతే {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


మూస <ఖండం_reduce_పద్ధతి reduce_పద్ధతి_t, పూర్ణాంకం warp_కొలత, రకం_పేరు T>
స్థిర T ఖండం_reduce(T val, T * shared_vals, పూర్ణాంకం ఖండం_కొలత_template) {
    స్వయంచాలక అంశం_ct1                 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    val                           = ఖండం_reduce_policy<reduce_పద్ధతి_t, T,warp_కొలత>::reduce(val);
    స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = ఖండం_కొలత_template == 0 ? అంశం_ct1.get_local_range(2) : ఖండం_కొలత_template;
    స్థిరరాశి పూర్ణాంకం nthreads = అంశం_ct1.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం nwarps = nthreads / WARP_కొలత;

    ఒకవేళ (ఖండం_కొలత_3 > warp_కొలత) {
        ప్రతిజ్ఞా((ఖండం_కొలత_3 <= 1024) && (ఖండం_కొలత_3 % warp_కొలత) == 0);
        స్థిరరాశి పూర్ణాంకం warp_id = అంశం_ct1.get_local_id(2) / warp_కొలత;
        స్థిరరాశి పూర్ణాంకం lane_id = అంశం_ct1.get_local_id(2) % warp_కొలత;
        ఒకవేళ (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_కొలత;
        దశాంశం tmp = 0.f;
        ఒకవేళ (lane_id < (static_cast<పూర్ణాంకం>(ఖండం_కొలత_3) / warp_కొలత)) {
            కొరకు (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_కొలత];
            }
        }
        తిరిగి ఖండం_reduce_policy<reduce_పద్ధతి_t, T, warp_కొలత>::reduce(tmp);
    }
    తిరిగి val;
}

స్థిర __dpct_inline__ దశాంశం ggml_sycl_ue4m3_గమ్యం_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    ఒకవేళ (x == 0 || x == 0x7F) {
        తిరిగి 0.0f;
    }
    స్థిరరాశి పూర్ణాంకం ఘాతాంకం = (x >> 3) & 0xF;
    స్థిరరాశి పూర్ణాంకం man = x & 0x7;
    దశాంశం raw;
    ఒకవేళ (ఘాతాంకం == 0) {
        raw = man * (1.0f / 8.0f) * sycl::ఘాతం(2.0f, -6.0f);
    } లేకపోతే {
        raw = (1.0f + man / 8.0f) * sycl::ఘాతం(2.0f, (దశాంశం) ఘాతాంకం - 7.0f);
    }
    తిరిగి raw * 0.5f;
}

#అయితే_ముగింపు // GGML_SYCL_COMMON_HPP
