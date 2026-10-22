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

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <array>
#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <atomic>
#ಅನ್ತರ್ಭಾವ <cctype>
#ಅನ್ತರ್ಭಾವ <cinttypes>
#ಅನ್ತರ್ಭಾವ <cstddef>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <float.h>
#ಅನ್ತರ್ಭಾವ <limits>
#ಅನ್ತರ್ಭಾವ <optional>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <fstream>
#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <regex>

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/backend.hpp>
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ಅನ್ತರ್ಭಾವ <level_zero/ze_api.h>
#ಅನ್ತರ್ಭಾವ <level_zero/zes_api.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_SYCL_ನಕ್ಷೆ) && SYCL_EXT_ONEAPI_ASYNC_ಸ್ಮರಣೆ_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <sycl/half_type.hpp>

#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "ggml-sycl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"

#ಅನ್ತರ್ಭಾವ "ggml-sycl/add-id.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/backend.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/common.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/element_wise.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/fwht.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/fused-gemm.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/gemm.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/getrows.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/mem.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/norm.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/presets.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/quantize.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/repeat_back.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/set_rows.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/set.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/dsv4-hc.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/lightning-indexer.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/conv2d.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/conv2d-dw.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/conv2d-transpose.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/ssm_conv.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/sycl_hw.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/ssm_scan.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/fill.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/cumsum.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/diag.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/opt-step.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/solve_tri.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/gated_delta_net.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/pool.hpp"
#ಅನ್ತರ್ಭಾವ "ggml-sycl/cross_entropy_loss.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು MEM_ಗಾತ್ರ_2_M	0x00200000
#ವ್ಯಾಖ್ಯಾನಿಸು MEM_ಗಾತ್ರ_1_G	0x40000000

ಸ್ಥಿರ ತಾರ್ಕಿಕ g_sycl_loaded = ಸುಳ್ಳು;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_debug = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_dev_debug = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_optimize = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_ನಕ್ಷೆ = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_dnn = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_fa_onednn = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_fa_onednn_max_kv = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_mkl_fa = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_memtrace = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ = 64;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_vmm = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_fusion = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_esimd = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_mmvq_wide = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_prioritize_dmmv = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_sycl_dynamic_precision_ಹೆಸರುಗಳು[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_get_env_precision(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ಪೂರ್ಣಾಂಕ def) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env = getenv(name);
    ಆದರೆ (!env) {
        ಹಿಂತಿರುಗಿ def;
    }
    std::string mode(env);
    ಗಾಗಿ (ಅಕ್ಷರ & c : mode) {
        c = (ಅಕ್ಷರ) std::ದೊಡ್ಡ_ಅಕ್ಷರ((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ) c);
    }
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        ಆದರೆ (mode == ggml_sycl_dynamic_precision_ಹೆಸರುಗಳು[i]) {
            ಹಿಂತಿರುಗಿ i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_ಹೆಸರುಗಳು[def]);
    ಹಿಂತಿರುಗಿ def;
}
ಪೂರ್ಣಾಂಕ g_ggml_sycl_use_async_mem_op = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_use_async_mem_op_requested = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_use_level_zero_api = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_flash_attention = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_usm_system = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_host_pinned_mem = 1;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_host_pinned_mem_2g = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_upload_staging_slots = 4;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_get_mem_api = ಸ್ಮರಣೆ_API_TYPE_LEVEL_ZERO;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_enable_sparse_fa = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_debug_sparse_fa = 0;
ಪೂರ್ಣಾಂಕ g_ggml_sycl_sparse_fa_margin = 256;

#ಆದರೆ GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_dnnl_detect_optimized_gemm(ಪೂರ್ಣಾಂಕ device) {
    ಬಳಸಿ dt = dnnl::ಸ್ಮರಣೆ::ದತ್ತಾಂಶ_type;

    ಪ್ರಯತ್ನಿಸು {
        ಸ್ಥಿರಾಂಕ dnnl::ಸ್ಮರಣೆ::dims dims    = { 1, 64, 64 };
        ಸ್ಥಿರಾಂಕ dnnl::ಸ್ಮರಣೆ::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.ಗಣ_scratchpad_ಬಹುಲಕ(dnnl::scratchpad_ಬಹುಲಕ::user);

        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & q   = dpct::dev_mgr::instance().get_device(device).default_ಸರತಿ();
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::ಸ್ಮರಣೆ::desc(dims, dt::f16, strides),
                                                        dnnl::ಸ್ಮರಣೆ::desc(dims, dt::f16, strides),
                                                        dnnl::ಸ್ಮರಣೆ::desc(dims, dt::f32, strides), attr);

        ಸ್ಥಿರಾಂಕ std::string impl = pd.impl_info_str();
        ಆದರೆ (impl.find("ref") == std::string::npos) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    ಪ್ರಯತ್ನಿಸು {
        info.device_ಎಣಿಕೆ = dpct::dev_mgr::instance().device_ಎಣಿಕೆ();
    } ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_ಎಣಿಕೆ = 0;
    }
    ಆದರೆ (info.device_ಎಣಿಕೆ == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        ಹಿಂತಿರುಗಿ info;
    }

    GGML_ASSERT(info.device_ಎಣಿಕೆ <= GGML_SYCL_MAX_DEVICES);

    int64_t ಒಟ್ಟು_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < info.device_ಎಣಿಕೆ; ++i) {
        dpct::device_info prop;
        ಸ್ವಯಂಚಾಲಿತ & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#ಆದರೆ !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#ಇಲ್ಲದಿದ್ದರೆ
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        ಆದರೆ (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ಸ್ಥಿರಾಂಕ size_t physical_ಪುಟ = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_ಪುಟ);
        }
#ಆದರೆ_ಅಂತ್ಯ

        info.default_tensor_ವಿಭಜಿಸು[i] = ಒಟ್ಟು_vram;
        ಒಟ್ಟು_vram += prop.get_global_mem_ಗಾತ್ರ();

        info.devices[i].cc =
            100 * prop.get_major_ಆವೃತ್ತಿ() + 10 * prop.get_minor_ಆವೃತ್ತಿ();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#ಆದರೆ GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#ಆದರೆ_ಅಂತ್ಯ
        info.devices[i].smpbo = prop.get_local_mem_ಗಾತ್ರ();
        info.devices[i].warp_ಗಾತ್ರ = WARP_ಗಾತ್ರ;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_ಗುಂಪು_sizes[i] = prop.get_max_work_ಗುಂಪು_ಗಾತ್ರ();
        info.devices[i].max_wg_ಪ್ರತಿ_cu = info.max_work_ಗುಂಪು_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        ಆದರೆ (device.is_gpu() && device.default_ಸರತಿ().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = ಸುಳ್ಳು;
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        ಆದರೆ (info.ext_oneapi_level_zero && device.is_gpu() && device.default_ಸರತಿ().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_ಸರತಿ().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_ಫಲಿತಾಂಶ_t r = zeDeviceGetProperties(ze_dev, &props);
            ಆದರೆ (r == ZE_ಫಲಿತಾಂಶ_SUCCESS) {
                info.devices[i].l0_device_type_valid = ನಿಜ;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < info.device_ಎಣಿಕೆ; ++id) {
        info.default_tensor_ವಿಭಜಿಸು[id] /= ಒಟ್ಟು_vram;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#ಆದರೆ_ಅಂತ್ಯ

    ಹಿಂತಿರುಗಿ info;
}

ಸ್ಥಿರಾಂಕ ggml_sycl_device_info & ggml_sycl_info() {
    ಸ್ಥಿರ ggml_sycl_device_info info = ggml_sycl_init();
    ಹಿಂತಿರುಗಿ info;
}

ಸ್ಥಿರ ಶೂನ್ಯ print_device_detail(ಪೂರ್ಣಾಂಕ id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string ಆವೃತ್ತಿ_2;
    ಆವೃತ್ತಿ_2 += std::to_string(prop.get_major_ಆವೃತ್ತಿ());
    ಆವೃತ್ತಿ_2 += ".";
    ಆವೃತ್ತಿ_2 += std::to_string(prop.get_minor_ಆವೃತ್ತಿ());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    ಸ್ವಯಂಚಾಲಿತ global_mem_ಗಾತ್ರ = prop.get_global_mem_ಗಾತ್ರ()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), ಆವೃತ್ತಿ_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_ಗುಂಪು_ಗಾತ್ರ(), prop.get_max_sub_ಗುಂಪು_ಗಾತ್ರ(),
            global_mem_ಗಾತ್ರ, device.get_info<sycl::info::device::driver_ಆವೃತ್ತಿ>().c_str());
}

ಸ್ಥಿರ ಶೂನ್ಯ print_device_opt_feature(ಪೂರ್ಣಾಂಕ device_ಎಣಿಕೆ) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < device_ಎಣಿಕೆ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ಪೂರ್ಣಾಂಕ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
ಶೂನ್ಯ ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    ಪೂರ್ಣಾಂಕ device_ಎಣಿಕೆ = dpct::dev_mgr::instance().device_ಎಣಿಕೆ();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_ಎಣಿಕೆ);

    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |Max    |        |Max  |Global |                     |\n");
    GGML_LOG_INFO(
        "|  |                   |                                       |      "
        " |compute|Max work|sub  |mem    |                     |\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|                                   "
        "Name|Version|units  |group   |group|size   |       Driver version|\n");
    GGML_LOG_INFO(
        "|--|-------------------|---------------------------------------|------"
        "-|-------|--------|-----|-------|---------------------|\n");

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < device_ಎಣಿಕೆ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ಪೂರ್ಣಾಂಕ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_ಎಣಿಕೆ);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* dev2dev_int2str(ಪೂರ್ಣಾಂಕ dev2dev) {
    ಆದರೆ (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        ಹಿಂತಿರುಗಿ "SYCL API";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dev2dev == DEV2DEV_MEMCPY_L0) {
        ಹಿಂತಿರುಗಿ "Level Zero API";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        ಹಿಂತಿರುಗಿ "Host Forward";
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ಸ್ಥಿರ ze_ಫಲಿತಾಂಶ_t init_zes() {
    ze_ಫಲಿತಾಂಶ_t res = zesInit(0);
    ಆದರೆ (res != ZE_ಫಲಿತಾಂಶ_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (ಪೂರ್ಣಾಂಕ) res);
    }
    ಹಿಂತಿರುಗಿ res;
}

ಸ್ಥಿರ ze_ಫಲಿತಾಂಶ_t get_zes_init_res() {
    ಸ್ಥಿರ ze_ಫಲಿತಾಂಶ_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (ಪೂರ್ಣಾಂಕ) zes_init_res);
    ಹಿಂತಿರುಗಿ zes_init_res;
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ initialize_sycl_begining() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_check_sycl() ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    ಆದರೆ (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_ನಕ್ಷೆ = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
        g_ggml_sycl_enable_vmm = ggml_sycl_get_env("GGML_SYCL_ENABLE_VMM", 1);
        g_ggml_sycl_enable_fusion = ggml_sycl_get_env("GGML_SYCL_ENABLE_FUSION", 1);
        g_ggml_sycl_enable_esimd = ggml_sycl_get_env("GGML_SYCL_ENABLE_ESIMD", 1);
        g_ggml_sycl_mmvq_wide = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE", 1);
        g_ggml_sycl_prioritize_dmmv = ggml_sycl_get_env("GGML_SYCL_PRIORITIZE_DMMV", 0);
        g_ggml_sycl_xmx_gather_types = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_TYPES", GGML_SYCL_XMX_GATHER_TYPES_DEFAULT);
        g_ggml_sycl_xmx_gather_shapes = ggml_sycl_get_env("GGML_SYCL_XMX_GATHER_SHAPES", GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT);
        g_ggml_sycl_dynamic_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_DEFAULT);
        g_ggml_sycl_dynamic_required_precision =
            ggml_sycl_get_env_precision("GGML_SYCL_DYNAMIC_REQUIRED_PRECISION", GGML_SYCL_DYNAMIC_PRECISION_F32);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#ಇಲ್ಲದಿದ್ದರೆ
        g_ggml_sycl_use_level_zero_api = 0;
#ಆದರೆ_ಅಂತ್ಯ
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", ಸ್ಮರಣೆ_API_TYPE_LEVEL_ZERO);
        ಆದರೆ (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = ಸ್ಮರಣೆ_API_TYPE_SYCL;
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#ಇಲ್ಲದಿದ್ದರೆ
        g_ggml_sycl_enable_flash_attention = 0;
#ಆದರೆ_ಅಂತ್ಯ

        g_ggml_sycl_usm_system = ggml_sycl_get_env("GGML_SYCL_USM_SYSTEM", 0);
        g_ggml_sycl_enable_host_pinned_mem =
            ggml_sycl_get_env("GGML_SYCL_ENABLE_HOST_PINNED_MEM", 1);

        g_ggml_sycl_host_pinned_mem_2g =
            ggml_sycl_get_env("GGML_SYCL_HOST_PINNED_MEM_2G", 0) & g_ggml_sycl_enable_host_pinned_mem;
        g_ggml_sycl_upload_staging_slots = std::max(0, ggml_sycl_get_env("GGML_SYCL_UPLOAD_STAGING_SLOTS", 4));

        g_ggml_sycl_enable_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA", 0);
        g_ggml_sycl_debug_sparse_fa = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_DEBUG", 0);
        g_ggml_sycl_sparse_fa_margin = ggml_sycl_get_env("GGML_SYCL_SPARSE_FA_MARGIN", 256);

        GGML_SYCL_DEBUG("[SYCL] call ggml_check_sycl\n");

        GGML_LOG_INFO("Build with Macros:\n");
#ಆದರೆ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_SYCL_ನಕ್ಷೆ)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#ಆದರೆ_ಅಂತ್ಯ

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ಆದರೆ_ಅಂತ್ಯ
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_ಹೆಜ್ಜೆ);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_ನಕ್ಷೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_ನಕ್ಷೆ);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#ಆದರೆ_ಅಂತ್ಯ

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_ಹೆಸರುಗಳು[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_ಹೆಸರುಗಳು[g_ggml_sycl_dynamic_required_precision]);

#ಆದರೆ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#ಆದರೆ_ಅಂತ್ಯ

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#ಆದರೆ defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#ಆದರೆ_ಅಂತ್ಯ
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#ಆದರೆ_ಅಂತ್ಯ

        GGML_LOG_INFO("  GGML_SYCL_USM_SYSTEM: %d\n", g_ggml_sycl_usm_system);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_HOST_PINNED_MEM: %d\n", g_ggml_sycl_enable_host_pinned_mem);
        GGML_LOG_INFO("  GGML_SYCL_HOST_PINNED_MEM_2G: %d\n", g_ggml_sycl_host_pinned_mem_2g);

        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA: %d\n", g_ggml_sycl_enable_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_DEBUG: %d\n", g_ggml_sycl_debug_sparse_fa);
        GGML_LOG_INFO("  GGML_SYCL_SPARSE_FA_MARGIN: %d\n", g_ggml_sycl_sparse_fa_margin);

/* NOT REMOVE, keep it for next optimize for XMX.
#if defined(SYCL_USE_XMX)
        fprintf(stderr, "%s: SYCL_USE_XMX: yes\n", __func__);
#else
        fprintf(stderr, "%s: SYCL_USE_XMX: no\n", __func__);
#endif
*/
        // Async USM allocation/free is also useful outside the graph path: it avoids the host waits in the reorder
        // staging path while preserving queue ordering semantics. Graph support still depends on the extension being
        // available, but it no longer needs to control the non-graph fast path.
#ಆದರೆ defined(GGML_SYCL_ನಕ್ಷೆ) && SYCL_EXT_ONEAPI_ASYNC_ಸ್ಮರಣೆ_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_ನಕ್ಷೆ;
        ಆದರೆ (g_ggml_sycl_use_async_mem_op) {
            ಗಾಗಿ (ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i = 0; i < dpct::dev_mgr::instance().device_ಎಣಿಕೆ(); ++i) {
                ಆದರೆ (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_ಸ್ಮರಣೆ_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    ನಿಲ್ಲಿಸು;
                }
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಆದರೆ (CHECK_TRY_ERROR(g_all_sycl_device_ಎಣಿಕೆ =
                            dpct::dev_mgr::instance().device_ಎಣಿಕೆ()) != 0) {
            initialized = ನಿಜ;
            g_sycl_loaded = ಸುಳ್ಳು;
            ಹಿಂತಿರುಗಿ;
        }
        GGML_ASSERT(g_all_sycl_device_ಎಣಿಕೆ <= GGML_SYCL_MAX_DEVICES);

        initialized = ನಿಜ;
        g_sycl_loaded = ನಿಜ;
        ggml_backend_sycl_print_sycl_devices();
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
ಸಾಲೊಳಗೆ ಶೂನ್ಯ check_allow_gpu_ಸೂಚ್ಯಂಕ(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device_ಸೂಚ್ಯಂಕ) {
  ಆದರೆ (device_ಸೂಚ್ಯಂಕ >= ggml_sycl_info().device_ಎಣಿಕೆ) {
    ಅಕ್ಷರ error_buf[256];
    snprintf(
        error_buf,
        ಗಾತ್ರ(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_ಸೂಚ್ಯಂಕ,
        ggml_sycl_info().device_ಎಣಿಕೆ - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
  }
}

GGML_API ಶೂನ್ಯ ggml_backend_sycl_get_gpu_list(ಪೂರ್ಣಾಂಕ *id_list, ಪೂರ್ಣಾಂಕ max_len) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    ಗಾಗಿ(ಪೂರ್ಣಾಂಕ i=0;i<max_len;i++) id_list[i] = -1;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i=0;i< ggml_sycl_info().device_ಎಣಿಕೆ;i++){
        ಆದರೆ (i>=max_len) ನಿಲ್ಲಿಸು;
        id_list[i] = i;
    }
    ಹಿಂತಿರುಗಿ;
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ free_aligned_mem_host(ಶೂನ್ಯ * memblock) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    _aligned_free(memblock);
#ಇಲ್ಲದಿದ್ದರೆ
    ಬಿಡುಗಡೆ(memblock);
#ಆದರೆ_ಅಂತ್ಯ
}

// sycl buffer

ರಚನೆ ggml_backend_sycl_ಬಫರ್_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    ಸ್ಥಿರ constexpr size_t staging_ಸ್ಥಾನ_ಗಾತ್ರ = 8*1024*1024;

    ರಚನೆ host_staging {
        ಶೂನ್ಯ * ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
        std::vector<sycl::event> events;
        std::vector<ತಾರ್ಕಿಕ> submitted;
        ಪೂರ್ಣಾಂಕ ಮುಂದಿನ = 0;
    };

    ಪೂರ್ಣಾಂಕ device;
    ಶೂನ್ಯ * dev_ptr = ಶೂನ್ಯಸೂಚಕ;
    ಸರತಿ_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    ತಾರ್ಕಿಕ is_usm_system;

    ggml_backend_sycl_ಬಫರ್_context(ಪೂರ್ಣಾಂಕ device, ಶೂನ್ಯ * dev_ptr, ಸರತಿ_ptr stream, ತಾರ್ಕಿಕ is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_ಸೂಚ್ಯಂಕ(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    ಶೂನ್ಯ drop_host_staging() {
        ಗಾಗಿ (size_t i = 0; i < staging.submitted.size(); ++i) {
            ಆದರೆ (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = ಸುಳ್ಳು;
            }
        }
        ಆದರೆ (staging.ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ) {
            sycl::ಬಿಡುಗಡೆ(staging.ದತ್ತಾಂಶ, *stream);
            staging.ದತ್ತಾಂಶ = ಶೂನ್ಯಸೂಚಕ;
        }
    }

    ~ggml_backend_sycl_ಬಫರ್_context() {
        drop_host_staging();
        ಆದರೆ (dev_ptr != ಶೂನ್ಯಸೂಚಕ) {
            ggml_sycl_ಗಣ_device(device);
            ಆದರೆ (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            ಇಲ್ಲದಿದ್ದರೆ
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        ಗಾಗಿ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft);

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_sycl(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಹಿಂತಿರುಗಿ ಬಫರ್->buft->iface.get_name == ggml_backend_sycl_ಬಫರ್_type_get_name;
}

ಸ್ಥಿರ ಶೂನ್ಯ
ggml_backend_sycl_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) ಪ್ರಯತ್ನಿಸು {
    ggml_backend_sycl_ಬಫರ್_context * ctx = ( ggml_backend_sycl_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_sycl_ಗಣ_device(ctx->device);

    ಅಳಿಸು ctx;
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_sycl_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_sycl_ಬಫರ್_context * ctx = ( ggml_backend_sycl_ಬಫರ್_context *)ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ctx->dev_ptr;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ
ggml_backend_sycl_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                     ggml_tensor *tensor) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_ಬಫರ್_context * ctx = (ggml_backend_sycl_ಬಫರ್_context *)ಬಫರ್->context;

    ಆದರೆ (tensor->view_src != ಖಾಲಿ) {
        ಪ್ರತಿಜ್ಞಾ(tensor->view_src->ಬಫರ್->buft == ಬಫರ್->buft);
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    ಆದರೆ (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        ಆಯ್ಕೆ (tensor->type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = ಹೊಸ ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                ನಿಲ್ಲಿಸು;
            }
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
    }

    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_ಗಾತ್ರ = ggml_nbytes(tensor);
        size_t padded_ಗಾತ್ರ = ggml_backend_buft_get_alloc_ಗಾತ್ರ(ಬಫರ್->buft, tensor);

        ಆದರೆ (padded_ಗಾತ್ರ > original_ಗಾತ್ರ && tensor->view_src == ಶೂನ್ಯಸೂಚಕ) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + original_ಗಾತ್ರ, 0,
                padded_ಗಾತ್ರ - original_ಗಾತ್ರ).wait()));
        }
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                ggml_tensor *tensor,
                                                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    ggml_backend_sycl_ಬಫರ್_context * ctx = ( ggml_backend_sycl_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_sycl_ಗಣ_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_slots = g_ggml_sycl_upload_staging_slots;
    ಆದರೆ (n_slots > 0 && ctx->staging.ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ctx->staging.ದತ್ತಾಂಶ = sycl::malloc_host(n_slots * ctx->staging_ಸ್ಥಾನ_ಗಾತ್ರ, *ctx->stream);
        ಆದರೆ (ctx->staging.ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.ನಿಯೋಜಿಸು(n_slots, ಸುಳ್ಳು);
        }
    }
    ಆದರೆ (ctx->staging.ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ) {
        ಸರತಿ_ptr    stream    = ctx->stream;
        ಅಕ್ಷರ *       dst       = (ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src       = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ದತ್ತಾಂಶ;
        size_t       ಉಳಿದ = size;
        ಆಗ (ಉಳಿದ > 0) {
            ಸ್ಥಿರಾಂಕ size_t chunk = std::min(ಉಳಿದ, ctx->staging_ಸ್ಥಾನ_ಗಾತ್ರ);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಸ್ಥಾನ  = ctx->staging.ಮುಂದಿನ;
            ctx->staging.ಮುಂದಿನ = (ctx->staging.ಮುಂದಿನ + 1) % (ಪೂರ್ಣಾಂಕ) ctx->staging.submitted.size();
            ಆದರೆ (ctx->staging.submitted[ಸ್ಥಾನ]) {
                ctx->staging.events[ಸ್ಥಾನ].wait_and_throw();
            }
            ಶೂನ್ಯ * stage = (ಅಕ್ಷರ *) ctx->staging.ದತ್ತಾಂಶ + ಸ್ಥಾನ * ctx->staging_ಸ್ಥಾನ_ಗಾತ್ರ;
            ಸ್ಮರಣೆ_ನಕಲು(stage, src, chunk);
            ctx->staging.events[ಸ್ಥಾನ] = stream->ಸ್ಮರಣೆ_ನಕಲು(dst, stage, chunk);
            ctx->staging.submitted[ಸ್ಥಾನ] = ನಿಜ;
            src       += chunk;
            dst       += chunk;
            ಉಳಿದ -= chunk;
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ವಯಂಚಾಲಿತ stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_ಸರತಿ());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    ಅಕ್ಷರ * host_buf = (ಅಕ್ಷರ *) ಸ್ಮರಣೆ_ಕೊಡು(size);
    ಸ್ಮರಣೆ_ನಕಲು(host_buf, ದತ್ತಾಂಶ, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, host_buf, size).wait()));
    ಬಿಡುಗಡೆ(host_buf);
#ಇಲ್ಲದಿದ್ದರೆ
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *) tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size).wait()));
#ಆದರೆ_ಅಂತ್ಯ
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                                ಸ್ಥಿರಾಂಕ ggml_tensor *tensor,
                                                ಶೂನ್ಯ *ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                                size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    ggml_backend_sycl_ಬಫರ್_context * ctx = ( ggml_backend_sycl_ಬಫರ್_context *)ಬಫರ್->context;

    ggml_sycl_ಗಣ_device(ctx->device);
    ಸ್ವಯಂಚಾಲಿತ stream = dpct::dev_mgr::instance().get_device(ctx->device).default_ಸರತಿ();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.ಸ್ಮರಣೆ_ನಕಲು(ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size)
            .wait()));
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_is_l0_discrete_gpu(ಪೂರ್ಣಾಂಕ device) {
    ಹಿಂತಿರುಗಿ ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ಶೂನ್ಯ memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, ಶೂನ್ಯ *ptr_dst,
                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr_src, size_t size) {
    ಅಕ್ಷರ *host_buf = (ಅಕ್ಷರ *)ಸ್ಮರಣೆ_ಕೊಡು(size);
    ಆದರೆ (host_buf == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.ಸ್ಮರಣೆ_ನಕಲು(host_buf, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ptr_src, size).wait();
    q_dst.ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)ptr_dst, host_buf, size).wait();
    ಬಿಡುಗಡೆ(host_buf);
}

ಸ್ಥಿರ ಶೂನ್ಯ dev2dev_memcpy(ಪೂರ್ಣಾಂಕ device_dst, sycl::queue &q_dst, ಪೂರ್ಣಾಂಕ device_src, sycl::queue &q_src, ಶೂನ್ಯ *ptr_dst,
                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr_src, size_t size) {

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ಆದರೆ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        ಆದರೆ (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            ಸ್ವಯಂಚಾಲಿತ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            ಸ್ವಯಂಚಾಲಿತ ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_ಸರತಿ_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_ಸರತಿ_DESC, ಶೂನ್ಯಸೂಚಕ, 0, 0,
                                            0, ZE_COMMAND_ಸರತಿ_ಬಹುಲಕ_SYNCHRONOUS, ZE_COMMAND_ಸರತಿ_ಆದ್ಯತೆ_ಸಾಮಾನ್ಯ};
            ze_command_list_handle_t cl;
            ze_ಫಲಿತಾಂಶ_t r = ze_Command_List_Create_ತಕ್ಷಣ(ze_ctx, ze_dev, &cq_desc, &cl);
            ಆದರೆ (r == ZE_ಫಲಿತಾಂಶ_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_ಸ್ಮರಣೆ_Copy(cl, ptr_dst, ptr_src, size, ಶೂನ್ಯಸೂಚಕ, 0, ಶೂನ್ಯಸೂಚಕ);
                zeCommandListDestroy(cl);
                ಆದರೆ (r == ZE_ಫಲಿತಾಂಶ_SUCCESS) {
                    ಹಿಂತಿರುಗಿ;
                }
            }
        }
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        ಆದರೆ (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.ಸ್ಮರಣೆ_ನಕಲು(ptr_dst, ptr_src, size).wait()));
            ಹಿಂತಿರುಗಿ;
        }
    }

    // Host-staged copy
    ಆದರೆ(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ
ggml_backend_sycl_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                    ಸ್ಥಿರಾಂಕ ggml_tensor *src,
                                    ggml_tensor *dst) ಪ್ರಯತ್ನಿಸು {
    ತಾರ್ಕಿಕ is_cpy_supported = ggml_backend_ಬಫರ್_is_sycl(src->ಬಫರ್);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ಆದರೆ (is_cpy_supported) {
        ggml_backend_sycl_ಬಫರ್_context * src_ctx = (ggml_backend_sycl_ಬಫರ್_context *)src->ಬಫರ್->context;
        ggml_backend_sycl_ಬಫರ್_context * dst_ctx = (ggml_backend_sycl_ಬಫರ್_context *)dst->ಬಫರ್->context;

        ggml_sycl_ಗಣ_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_ಗಣ_device(dst_ctx->device);
        /*
        DPCT1009:199: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
        /*
        DPCT1009:200: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */

        ಸರತಿ_ptr ಹರಿವು_dst = dst_ctx->stream;
        ಸರತಿ_ptr ಹರಿವು_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *ಹರಿವು_dst, src_ctx->device, *ಹರಿವು_src, dst->ದತ್ತಾಂಶ, src->ದತ್ತಾಂಶ, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#ಆದರೆ 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).ಸ್ಮರಣೆ_ನಕಲು(
            (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src->ದತ್ತಾಂಶ, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#ಆದರೆ_ಅಂತ್ಯ
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    GGML_ಬಳಸದ(ಬಫರ್);
} ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                           uint8_t value) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, ಬಫರ್->size);
    ggml_backend_sycl_ಬಫರ್_context * ctx = (ggml_backend_sycl_ಬಫರ್_context *) ಬಫರ್->context;

    ggml_sycl_ಗಣ_device(ctx->device);
    ಸರತಿ_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    ಗಾಗಿ (size_t off = 0; off < ಬಫರ್->size; off += MAX_CHUNK) {
        size_t chunk = std::min(ಬಫರ್->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<ಅಕ್ಷರ*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಬಫರ್_memset_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor, uint8_t value,
                                                   size_t ಸ್ಥಾನಾಂತರ, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, ಸ್ಥಾನಾಂತರ, value);
    ggml_backend_sycl_ಬಫರ್_context * ctx = (ggml_backend_sycl_ಬಫರ್_context *) ಬಫರ್->context;
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx->device));
    ಸ್ವಯಂಚಾಲಿತ stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_ಸರತಿ());
    ಆದರೆ (size == 0) {
        ಹಿಂತಿರುಗಿ;  // Nothing to do
    }
    ಆದರೆ (tensor->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    ಶೂನ್ಯ * ಗುರಿ_ptr = static_cast<ಅಕ್ಷರ *>(tensor->ದತ್ತಾಂಶ) + ಸ್ಥಾನಾಂತರ;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(ಗುರಿ_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಬಫರ್_reset(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ಆದರೆ (ಬಫರ್ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_sycl_ಬಫರ್_context * ctx = (ggml_backend_sycl_ಬಫರ್_context *) ಬಫರ್->context;

    ಆದರೆ (ctx != ಶೂನ್ಯಸೂಚಕ) {
        ಗಾಗಿ (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_i ggml_backend_sycl_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_sycl_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_sycl_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_ಬಫರ್_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_sycl_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_ಬಫರ್_clear,
    /* .reset           = */ ggml_backend_sycl_ಬಫರ್_reset,
};

// sycl buffer type
ರಚನೆ ggml_backend_sycl_ಬಫರ್_type_context {
    ಪೂರ್ಣಾಂಕ device;
    std::string name;

    // each buffer type has its own stream
    ಸರತಿ_ptr stream = ಶೂನ್ಯಸೂಚಕ;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_sycl_ಬಫರ್_type_context * ctx = (ggml_backend_sycl_ಬಫರ್_type_context *)buft->context;

    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ check_usm_system(ಪೂರ್ಣಾಂಕ device, size_t size) {
    ತಾರ್ಕಿಕ use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_ಗಾತ್ರ_1_G);

    ಆದರೆ (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ use_usm_system;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ * aligned_malloc_host(size_t alignment, size_t size) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ _aligned_malloc(size, alignment);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ aligned_alloc(alignment, size);
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t
ggml_backend_sycl_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft,
                                           size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ಬಫರ್_type_context *)buft->context;
    ggml_sycl_ಗಣ_device(buft_ctx->device);
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_ಗಾತ್ರ_2_M;
    size_t aligned_ಗಾತ್ರ = ((size + alignment - 1) / alignment) * alignment;
    ತಾರ್ಕಿಕ use_usm_system = check_usm_system(buft_ctx->device, aligned_ಗಾತ್ರ);

    ಶೂನ್ಯ * dev_ptr;
    ಆದರೆ (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (ಶೂನ್ಯ *)aligned_malloc_host(alignment, aligned_ಗಾತ್ರ);
        ಆದರೆ (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (ಶೂನ್ಯ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_ಬಫರ್)));
        ಆದರೆ (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }
    ggml_backend_sycl_ಬಫರ್_context * ctx = ಹೊಸ  ggml_backend_sycl_ಬಫರ್_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_sycl_ಬಫರ್_interface, ctx, size);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ size_t ggml_backend_sycl_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ SYCL_ಬಫರ್_ALIGNMENT;
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ is_bmg_g31_arch(ಪೂರ್ಣಾಂಕ device) {
    ಹಿಂತಿರುಗಿ ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

ಸ್ಥಿರ size_t ggml_backend_sycl_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ggml_backend_sycl_ಬಫರ್_type_context * ctx = (ggml_backend_sycl_ಬಫರ್_type_context *)buft->context;
    ಪೂರ್ಣಾಂಕ device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_ಗಾತ್ರ = dpct::get_device(device).get_max_mem_alloc_ಗಾತ್ರ();
    ಆದರೆ (g_ggml_sycl_host_pinned_mem_2g) {
        ಹಿಂತಿರುಗಿ std::min(max_alloc_ಗಾತ್ರ, (size_t) 2LL*1024*1024*1024);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_ಗಾತ್ರ = max_alloc_ಗಾತ್ರ*0.6;
        }
        ಹಿಂತಿರುಗಿ max_alloc_ಗಾತ್ರ;

    }
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_sycl_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_ಗಾತ್ರ(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    ಆದರೆ (ggml_is_quantized(tensor->type)) {
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            size += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }
    }

    ಹಿಂತಿರುಗಿ size;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_ಬಫರ್_type_i ggml_backend_sycl_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_sycl_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_sycl_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ಖಾಲಿ,
};

ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_ಬಫರ್_type(ಪೂರ್ಣಾಂಕ device) {
    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);


    ಸ್ವಯಂಚಾಲಿತ dev_ಎಣಿಕೆ = ggml_backend_sycl_get_device_ಎಣಿಕೆ();

    ಆದರೆ (device>=dev_ಎಣಿಕೆ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_ಎಣಿಕೆ-1);
        GGML_ASSERT(device<dev_ಎಣಿಕೆ);
    }
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_sycl_ಬಫರ್_types[GGML_SYCL_MAX_DEVICES];

    ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_ಬಫರ್_type_initialized = ಸುಳ್ಳು;

    ಆದರೆ (!ggml_backend_sycl_ಬಫರ್_type_initialized) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < dev_ಎಣಿಕೆ; i++) {
            ಸ್ವಯಂಚಾಲಿತ & device_i = dpct::dev_mgr::instance().get_device(i);
            ಸರತಿ_ptr stream = &(device_i.default_ಸರತಿ());
            ggml_backend_sycl_ಬಫರ್_types[i] = {
                /* .iface    = */ ggml_backend_sycl_ಬಫರ್_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ ಹೊಸ ggml_backend_sycl_ಬಫರ್_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_ಬಫರ್_type_initialized = ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ &ggml_backend_sycl_ಬಫರ್_types[device];
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_ಬಫರ್_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    ಪೂರ್ಣಾಂಕ device = ctx->device;
    ಆದರೆ (device>=ggml_sycl_info().device_ಎಣಿಕೆ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_ಎಣಿಕೆ-1);
        GGML_ASSERT(device<ggml_sycl_info().device_ಎಣಿಕೆ);
    }
    ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_type ggml_backend_sycl_ಬಫರ್_types[GGML_SYCL_MAX_DEVICES];

    ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_ಬಫರ್_type_initialized = ಸುಳ್ಳು;

    ಆದರೆ (!ggml_backend_sycl_ಬಫರ್_type_initialized) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; i++) {
            ggml_backend_sycl_ಬಫರ್_types[i] = {
                /* .iface    = */ ggml_backend_sycl_ಬಫರ್_type_interface,
                /* .device   = */ ಶೂನ್ಯಸೂಚಕ,
                /* .context  = */ ಹೊಸ ggml_backend_sycl_ಬಫರ್_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_ಬಫರ್_type_initialized = ನಿಜ;
    }
    ಹಿಂತಿರುಗಿ &ggml_backend_sycl_ಬಫರ್_types[device];
}

// sycl split buffer

ಸ್ಥಿರ int64_t get_ಅಡ್ಡಸಾಲು_rounding(ggml_type type, ಸ್ಥಿರಾಂಕ std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> & tensor_ವಿಭಜಿಸು) {
    int64_t min_compute_capability = ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ;
    int64_t max_compute_capability = INT_MIN;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        ಆದರೆ (tensor_ವಿಭಜಿಸು[i] < (i + 1 < ggml_sycl_info().device_ಎಣಿಕೆ ? tensor_ವಿಭಜಿಸು[i + 1] : 1.0f)) {
            ಆದರೆ (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            ಆದರೆ (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    ಆಯ್ಕೆ(type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            ಹಿಂತಿರುಗಿ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ 64;
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            ಹಿಂತಿರುಗಿ 1;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            ಹಿಂತಿರುಗಿ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಹಿಂತಿರುಗಿ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಹಿಂತಿರುಗಿ 64;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("fatal error");
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು(int64_t * ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, int64_t * ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor, ಸ್ಥಿರಾಂಕ std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> & tensor_ವಿಭಜಿಸು, ಪೂರ್ಣಾಂಕ id) {
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(tensor);
    ಸ್ಥಿರಾಂಕ int64_t rounding = get_ಅಡ್ಡಸಾಲು_rounding(tensor->type, tensor_ವಿಭಜಿಸು);

    *ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ = id == 0 ? 0 : nrows*tensor_ವಿಭಜಿಸು[id];
    *ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ -= *ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ % rounding;
    ಆದರೆ (id == ggml_sycl_info().device_ಎಣಿಕೆ - 1) {
        *ಅಡ್ಡಸಾಲು_ಮೇಲಿನ = nrows;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *ಅಡ್ಡಸಾಲು_ಮೇಲಿನ = nrows*tensor_ವಿಭಜಿಸು[id + 1];
        *ಅಡ್ಡಸಾಲು_ಮೇಲಿನ -= *ಅಡ್ಡಸಾಲು_ಮೇಲಿನ % rounding;
    }
}

ಸ್ಥಿರ size_t ggml_nbytes_ವಿಭಜಿಸು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * tensor, ಪೂರ್ಣಾಂಕ nrows_ವಿಭಜಿಸು) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ಹಿಂತಿರುಗಿ nrows_ವಿಭಜಿಸು*ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, tensor->ne[0]);
}

ರಚನೆ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context {
    std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> tensor_ವಿಭಜಿಸು;
};

ರಚನೆ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context {
    ~ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context() ಪ್ರಯತ್ನಿಸು {
        ಗಾಗಿ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
      std::ನಿರ್ಗಮಿಸು(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<ಸರತಿ_ptr> streams;
};

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context * ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context *)ಬಫರ್->context;
    ಅಳಿಸು ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)0x1000;

    GGML_ಬಳಸದ(ಬಫರ್);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_ಸ್ಥಿತಿ
ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                           ggml_tensor *tensor) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == ಶೂನ್ಯಸೂಚಕ); // views of split tensors are not supported

    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context * ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *)ಬಫರ್->buft->context;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = ಹೊಸ ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    ಆದರೆ (ctx->streams.ಖಾಲಿಗಳು()) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_ಸರತಿ()));
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಅಡ್ಡಸಾಲು_ಮೇಲಿನ;
        get_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು(&ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, &ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, tensor, buft_ctx->tensor_ವಿಭಜಿಸು, i);

        int64_t nrows_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
        ಆದರೆ (nrows_ವಿಭಜಿಸು == 0) {
            ಮುಂದುವರಿ;
        }

        size_t size = ggml_nbytes_ವಿಭಜಿಸು(tensor, nrows_ವಿಭಜಿಸು);
        ಸ್ಥಿರಾಂಕ size_t original_ಗಾತ್ರ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            size += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }

        ggml_sycl_ಗಣ_device(i);
        ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx->streams[i];
        ಅಕ್ಷರ * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (ಅಕ್ಷರ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_ಬಫರ್)));
        ಆದರೆ (!buf) {
            ಅಕ್ಷರ err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            ಎಸೆ std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        ಆದರೆ (size > original_ಗಾತ್ರ) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_ಗಾತ್ರ, 0, size - original_ಗಾತ್ರ)
                    .wait()));
        }

        extra->ದತ್ತಾಂಶ_device[i] = buf;

        ಗಾಗಿ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = ಹೊಸ sycl::event()));
        }
    }
    tensor->extra = extra;
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ
ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                          ggml_tensor *tensor, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ದತ್ತಾಂಶ,
                                          size_t ಸ್ಥಾನಾಂತರ, size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context * ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *)ಬಫರ್->buft->context;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];
    ಸ್ಥಿರಾಂಕ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಅಡ್ಡಸಾಲು_ಮೇಲಿನ;
        get_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು(&ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, &ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, tensor, buft_ctx->tensor_ವಿಭಜಿಸು, i);

        int64_t nrows_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
        ಆದರೆ (nrows_ವಿಭಜಿಸು == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ*nb1;
        size_t size = ggml_nbytes_ವಿಭಜಿಸು(tensor, nrows_ವಿಭಜಿಸು);
        ಸ್ಥಿರಾಂಕ size_t original_ಗಾತ್ರ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            size += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * buf_host = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ವಿಭಜಿಸು;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_ಗಣ_device(i);
        ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .ಸ್ಮರಣೆ_ನಕಲು(extra->ದತ್ತಾಂಶ_device[i], buf_host, original_ಗಾತ್ರ)
                .wait()));
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ
ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                          ಸ್ಥಿರಾಂಕ ggml_tensor *tensor, ಶೂನ್ಯ *ದತ್ತಾಂಶ,
                                          size_t ಸ್ಥಾನಾಂತರ, size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context * ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context *)ಬಫರ್->context;
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *)ಬಫರ್->buft->context;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];
    ಸ್ಥಿರಾಂಕ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಅಡ್ಡಸಾಲು_ಮೇಲಿನ;
        get_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು(&ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, &ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, tensor, buft_ctx->tensor_ವಿಭಜಿಸು, i);

        int64_t nrows_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
        ಆದರೆ (nrows_ವಿಭಜಿಸು == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ*nb1;
        size_t size = ggml_nbytes_ವಿಭಜಿಸು(tensor, nrows_ವಿಭಜಿಸು);
        ಸ್ಥಿರಾಂಕ size_t original_ಗಾತ್ರ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            size += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }

        ಅಕ್ಷರ * buf_host = (ಅಕ್ಷರ *)ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ_ವಿಭಜಿಸು;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_ಗಣ_device(i);
        ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .ಸ್ಮರಣೆ_ನಕಲು(buf_host, extra->ದತ್ತಾಂಶ_device[i], original_ಗಾತ್ರ)
                .wait()));
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    GGML_ಬಳಸದ(ಬಫರ್);
    GGML_ಬಳಸದ(value);
}

ಸ್ಥಿರ ರಚನೆ ggml_backend_ಬಫರ್_i ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ಖಾಲಿ,
    /* .clear           = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

// sycl split buffer type

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ GGML_SYCL_NAME "_Split";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(ggml_backend_ಬಫರ್_t ಬಫರ್) {
   ಹಿಂತಿರುಗಿ ಬಫರ್->buft->iface.get_name == ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_name;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context * ctx = ಹೊಸ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_context();

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_interface, ctx, size);
}

ಸ್ಥಿರ size_t ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ SYCL_ಬಫರ್_ALIGNMENT;
    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ size_t ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *)buft->context;

    size_t ಒಟ್ಟು_ಗಾತ್ರ = 0;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = tensor->ne[0];

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಅಡ್ಡಸಾಲು_ಮೇಲಿನ;
        get_ಅಡ್ಡಸಾಲು_ವಿಭಜಿಸು(&ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, &ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, tensor, ctx->tensor_ವಿಭಜಿಸು, i);

        int64_t nrows_ವಿಭಜಿಸು = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
        ಆದರೆ (nrows_ವಿಭಜಿಸು == 0) {
            ಮುಂದುವರಿ;
        }

        ಒಟ್ಟು_ಗಾತ್ರ += ggml_nbytes_ವಿಭಜಿಸು(tensor, nrows_ವಿಭಜಿಸು);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ಆದರೆ (ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING != 0) {
            ಒಟ್ಟು_ಗಾತ್ರ += ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(tensor->type, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING - ne0 % ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
        }
    }

    ಹಿಂತಿರುಗಿ ಒಟ್ಟು_ಗಾತ್ರ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_i ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ಖಾಲಿ, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_is_host,
};

ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type(ಪೂರ್ಣಾಂಕ main_device, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * tensor_ವಿಭಜಿಸು) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_ಬಳಸದ(main_device);

    ಸ್ಥಿರ std::mutex mutex;
    std::lock_guard<std::mutex> ಬೀಗ(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    ಸ್ಥಿರ std::map<std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES>, ರಚನೆ ggml_backend_ಬಫರ್_type> buft_map;

    std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> tensor_ವಿಭಜಿಸು_arr = {};

    ತಾರ್ಕಿಕ all_zero = tensor_ವಿಭಜಿಸು == ಶೂನ್ಯಸೂಚಕ || std::all_of(tensor_ವಿಭಜಿಸು, tensor_ವಿಭಜಿಸು + GGML_SYCL_MAX_DEVICES, [](ದಶಮಾಂಶ x) { ಹಿಂತಿರುಗಿ x == 0.0f; });
    ಆದರೆ (all_zero) {
        tensor_ವಿಭಜಿಸು_arr = ggml_sycl_info().default_tensor_ವಿಭಜಿಸು;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ದಶಮಾಂಶ ವಿಭಜಿಸು_ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
            tensor_ವಿಭಜಿಸು_arr[i] = ವಿಭಜಿಸು_ಮೊತ್ತ;
            ವಿಭಜಿಸು_ಮೊತ್ತ += tensor_ವಿಭಜಿಸು[i];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
            tensor_ವಿಭಜಿಸು_arr[i] /= ವಿಭಜಿಸು_ಮೊತ್ತ;
        }
    }

    ಸ್ವಯಂಚಾಲಿತ it = buft_map.find(tensor_ವಿಭಜಿಸು_arr);
    ಆದರೆ (it != buft_map.end()) {
        ಹಿಂತಿರುಗಿ &it->second;
    }

    ರಚನೆ ggml_backend_ಬಫರ್_type buft {
        /* .iface   = */ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ ಹೊಸ ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context{tensor_ವಿಭಜಿಸು_arr},
    };

    ಸ್ವಯಂಚಾಲಿತ ಫಲಿತಾಂಶ = buft_map.emplace(tensor_ವಿಭಜಿಸು_arr, buft);
    ಹಿಂತಿರುಗಿ &ಫಲಿತಾಂಶ.ಮೊದಲ->second;
}

// host buffer type

ರಚನೆ ggml_backend_sycl_device_context {
    ಪೂರ್ಣಾಂಕ device;
    std::string name;
    std::string description;
    ಪೂರ್ಣಾಂಕ op_offload_min_batch_ಗಾತ್ರ;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_host_ಬಫರ್_type_name(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ GGML_SYCL_NAME "_Host";

    GGML_ಬಳಸದ(buft);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_backend_sycl_host_ಬಫರ್_type_device(ggml_backend_ಬಫರ್_type_t buft) {
    ಹಿಂತಿರುಗಿ static_cast<ಸ್ಥಿರಾಂಕ ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_sycl_host_malloc(ಪೂರ್ಣಾಂಕ device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
    ಪ್ರಯತ್ನಿಸು {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        ಸ್ವಯಂಚಾಲಿತ & q = dpct::dev_mgr::instance().get_device(device).default_ಸರತಿ();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } ಹಿಡಿ (...) {
        ptr = ಶೂನ್ಯಸೂಚಕ;
    }
    ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    ಹಿಂತಿರುಗಿ ptr;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_host_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ಆದರೆ (ಬಫರ್->context == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (g_ggml_sycl_enable_host_pinned_mem) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device = ggml_backend_sycl_host_ಬಫರ್_type_device(ಬಫರ್->buft);
        ಸ್ವಯಂಚಾಲಿತ & q = dpct::dev_mgr::instance().get_device(device).default_ಸರತಿ();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(ಬಫರ್->context, q)));
    } ಇಲ್ಲದಿದ್ದರೆ {
        free_aligned_mem_host((ಶೂನ್ಯ *) ಬಫರ್->context);
    }
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_sycl_host_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ಶೂನ್ಯ * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_ಬಫರ್_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    ಆದರೆ (ptr == ಶೂನ್ಯಸೂಚಕ) {
        // fallback to cpu buffer
        ಹಿಂತಿರುಗಿ ggml_backend_buft_alloc_ಬಫರ್(ggml_backend_cpu_ಬಫರ್_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_ಬಫರ್_t ಬಫರ್ = ggml_backend_cpu_ಬಫರ್_from_ptr(ptr, size);
    ಬಫರ್->buft = buft;
    ಬಫರ್->iface.free_ಬಫರ್ = ggml_backend_sycl_host_ಬಫರ್_free_ಬಫರ್;

    ಹಿಂತಿರುಗಿ ಬಫರ್;
}

ಸ್ಥಿರ size_t ggml_backend_sycl_host_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {

    ಆದರೆ (g_ggml_sycl_enable_host_pinned_mem) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ device = ggml_backend_sycl_host_ಬಫರ್_type_device(buft);
        size_t max_alloc_ಗಾತ್ರ = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_ಗಾತ್ರ();
        ಆದರೆ (g_ggml_sycl_host_pinned_mem_2g) {
            ಹಿಂತಿರುಗಿ std::min(max_alloc_ಗಾತ್ರ, (size_t) 2LL*1024*1024*1024);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹಿಂತಿರುಗಿ max_alloc_ಗಾತ್ರ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ SIZE_MAX;
    }
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_host_ಬಫರ್_type_for_device(ಪೂರ್ಣಾಂಕ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    ಸ್ಥಿರ std::vector<ggml_backend_ಬಫರ್_type> ಬಫರ್_types_host = [] {
        std::vector<ggml_backend_ಬಫರ್_type> bufts(ggml_backend_sycl_get_device_ಎಣಿಕೆ());
        ಗಾಗಿ (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_ಬಫರ್_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_ಬಫರ್_type_alloc_ಬಫರ್,
                    /* .alloc_buffer_n      = */ ಖಾಲಿ,
                    /* .get_alignment       = */ ggml_backend_cpu_ಬಫರ್_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_ಬಫರ್_type_get_max_ಗಾತ್ರ,
                    /* .get_alloc_size      = */ ggml_backend_cpu_ಬಫರ್_type()->iface.get_alloc_ಗಾತ್ರ,
                    /* .get_alloc_size_n    = */ ಖಾಲಿ,
                    /* .is_host             = */ ggml_backend_cpu_ಬಫರ್_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ ಶೂನ್ಯಸೂಚಕ,
            };
        }
        ಹಿಂತಿರುಗಿ bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (ಪೂರ್ಣಾಂಕ) ಬಫರ್_types_host.size());

    ಹಿಂತಿರುಗಿ &ಬಫರ್_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_host_ಬಫರ್_type() {
    ಹಿಂತಿರುಗಿ ggml_backend_sycl_host_ಬಫರ್_type_for_device(0);
}

// buffer pool for sycl (legacy)
ರಚನೆ ggml_sycl_pool_leg : ಸಾರ್ವಜನಿಕ ggml_sycl_pool {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ MAX_SYCL_BUFFERS = 256;

    ಪೂರ್ಣಾಂಕ device;
    ಸರತಿ_ptr qptr;
    ರಚನೆ ggml_sycl_ಬಫರ್ {
        ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
        size_t size = 0;
    };

    ggml_sycl_ಬಫರ್ ಬಫರ್_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_ಗಾತ್ರ = 0;

    ಸ್ಪಷ್ಟ ggml_sycl_pool_leg(ಸರತಿ_ptr qptr_, ಪೂರ್ಣಾಂಕ device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_POOL
        ಪೂರ್ಣಾಂಕ    n_cached    = 0;
        size_t ಬೈಟುಗಳು_cached = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ಆದರೆ (ಬಫರ್_pool[i].ptr != ಶೂನ್ಯಸೂಚಕ) {
                ++n_cached;
                ಬೈಟುಗಳು_cached += ಬಫರ್_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, ಬೈಟುಗಳು_cached / 1024.0 / 1024.0);
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ slots = format_slots_ಒಳ_alloc_ಕ್ರಮ();
        ಆದರೆ (!slots.ಖಾಲಿಗಳು()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#ಆದರೆ_ಅಂತ್ಯ

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr != ಶೂನ್ಯಸೂಚಕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_ಗಾತ್ರ -= b.size;
            }
        }
        GGML_ASSERT(pool_ಗಾತ್ರ == 0);
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_POOL
    std::string format_slots_ಒಳ_alloc_ಕ್ರಮ() ಸ್ಥಿರಾಂಕ {
        std::string ಸಾಲು;
        ಅಕ್ಷರ buf[32];
        ತಾರ್ಕಿಕ ಮೊದಲ = ನಿಜ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ಆದರೆ (ಬಫರ್_pool[i].ptr == ಶೂನ್ಯಸೂಚಕ) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (!ಮೊದಲ) {
                ಸಾಲು += '/';
            }
            ಮೊದಲ = ಸುಳ್ಳು;
            snprintf(buf, ಗಾತ್ರ(buf), "%.2f", ಬಫರ್_pool[i].size / 1024.0 / 1024.0);
            ಸಾಲು += buf;
        }
        ಹಿಂತಿರುಗಿ ಸಾಲು;
    }
#ಆದರೆ_ಅಂತ್ಯ

    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_sycl_MALLOC
        ಪೂರ್ಣಾಂಕ nnz = 0;
        size_t max_size = 0;
#ಆದರೆ_ಅಂತ್ಯ
        size_t ಉತ್ತಮ_diff = 1ull << 36;
        ಪೂರ್ಣಾಂಕ ibest = -1;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ಬಫರ್& b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr != ಶೂನ್ಯಸೂಚಕ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_sycl_MALLOC
                ++nnz;
                ಆದರೆ (b.size > max_size) max_size = b.size;
#ಆದರೆ_ಅಂತ್ಯ
                ಆದರೆ (b.size >= size) {
                    size_t diff = b.size - size;
                    ಆದರೆ (diff < ಉತ್ತಮ_diff) {
                        ಉತ್ತಮ_diff = diff;
                        ibest = i;
                        ಆದರೆ (!ಉತ್ತಮ_diff) {
                            ಶೂನ್ಯ * ptr = b.ptr;
                            *actual_ಗಾತ್ರ = b.size;
                            b.ptr = ಶೂನ್ಯಸೂಚಕ;
                            b.size = 0;
                            ಹಿಂತಿರುಗಿ ptr;
                        }
                    }
                }
            }
        }
        ಆದರೆ (ibest >= 0) {
            ggml_sycl_ಬಫರ್& b = ಬಫರ್_pool[ibest];
            ಶೂನ್ಯ * ptr = b.ptr;
            *actual_ಗಾತ್ರ = b.size;
            b.ptr = ಶೂನ್ಯಸೂಚಕ;
            b.size = 0;
            ಹಿಂತಿರುಗಿ ptr;
        }
        ಶೂನ್ಯ * ptr;
        size_t look_ahead_ಗಾತ್ರ = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ಶೂನ್ಯ *)ggml_sycl_malloc_device(look_ahead_ಗಾತ್ರ, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        ಆದರೆ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_ಗಾತ್ರ);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        *actual_ಗಾತ್ರ = look_ahead_ಗಾತ್ರ;
        pool_ಗಾತ್ರ += look_ahead_ಗಾತ್ರ;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_ಗಾತ್ರ[id]/1024/1024), (uint32_t)(size/1024/1024));
#ಆದರೆ_ಅಂತ್ಯ

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        ಹಿಂತಿರುಗಿ ptr;
    }

    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ಬಫರ್& b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr == ಶೂನ್ಯಸೂಚಕ) {
                b.ptr = ptr;
                b.size = size;
                ಹಿಂತಿರುಗಿ;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_ಗಾತ್ರ -= size;
    }
};

// pool with virtual memory management
#ಆದರೆ defined(GGML_SYCL_SUPPORT_VMM)
ರಚನೆ ggml_sycl_pool_vmm : ಸಾರ್ವಜನಿಕ ggml_sycl_pool {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ size_t SYCL_POOL_VMM_MAX_ಗಾತ್ರ = 1ull << 35; // 32 GB

    ಪೂರ್ಣಾಂಕ           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_ಬಳಸಿದ = 0;
    size_t    pool_ಗಾತ್ರ = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    ರಚನೆ mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        ಶೂನ್ಯ * map_ptr;
    };
    std::vector<mapping> mappings;

    ಸ್ಪಷ್ಟ ggml_sycl_pool_vmm(ಸರತಿ_ptr qptr_, ಪೂರ್ಣಾಂಕ device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        ಆದರೆ (pool_addr == 0) {
            ಹಿಂತಿರುಗಿ;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_ಗಾತ್ರ, ctx)));
    }

    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_ಬಫರ್_ALIGNMENT);

        size_t avail = pool_ಗಾತ್ರ - pool_ಬಳಸಿದ;

        ಆದರೆ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_ಗಾತ್ರ = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_ಗಾತ್ರ + reserve_ಗಾತ್ರ <= SYCL_POOL_VMM_MAX_ಗಾತ್ರ);

            ಆದರೆ (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ಟ್ಯಾಗ್ " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_ಗಾತ್ರ / (1024 * 1024), pool_ಗಾತ್ರ / (1024 * 1024),
                              (pool_ಗಾತ್ರ + reserve_ಗಾತ್ರ) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_ಗಾತ್ರ)));

            // reserve virtual address space (if not already reserved)
            ಆದರೆ (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_ಗಾತ್ರ, ctx)));
            }

            // map at the end of the pool
            ಶೂನ್ಯ * map_ptr = ಶೂನ್ಯಸೂಚಕ;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_ಗಾತ್ರ, reserve_ಗಾತ್ರ,
                                    sycl::ext::oneapi::experimental::ವಿಳಾಸ_access_ಬಹುಲಕ::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_ಗಾತ್ರ += reserve_ಗಾತ್ರ;
            ggml_sycl_memtrace_ಕೂಡಿಸು(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_ಗಾತ್ರ);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) (pool_ಗಾತ್ರ/1024/1024),
                          (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) (reserve_ಗಾತ್ರ/1024/1024));
#ಆದರೆ_ಅಂತ್ಯ
        }

        GGML_ASSERT(pool_addr != 0);

        ಶೂನ್ಯ * ptr = reinterpret_cast<ಶೂನ್ಯ *>(pool_addr + pool_ಬಳಸಿದ);
        *actual_ಗಾತ್ರ = size;
        pool_ಬಳಸಿದ += size;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) size, ptr);
#ಆದರೆ_ಅಂತ್ಯ

        ಹಿಂತಿರುಗಿ ptr;
    }

    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ) size, ptr);
#ಆದರೆ_ಅಂತ್ಯ

        pool_ಬಳಸಿದ -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<ಶೂನ್ಯ *>(pool_addr + pool_ಬಳಸಿದ));
    }
};
#ಆದರೆ_ಅಂತ್ಯ // defined(GGML_SYCL_SUPPORT_VMM)

ರಚನೆ ggml_sycl_pool_host : ಸಾರ್ವಜನಿಕ ggml_sycl_pool {
    ಸರತಿ_ptr qptr;
    ಪೂರ್ಣಾಂಕ       device;

    ರಚನೆ ggml_sycl_ಬಫರ್ {
        ಶೂನ್ಯ * ptr  = ಶೂನ್ಯಸೂಚಕ;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ          MAX_POOL_ಗಾತ್ರ{ 64 };
    std::vector<ggml_sycl_ಬಫರ್> ಬಫರ್_pool = std::vector<ggml_sycl_ಬಫರ್>(MAX_POOL_ಗಾತ್ರ);
    size_t                        pool_ಗಾತ್ರ   = 0;
    ಪೂರ್ಣಾಂಕ                           counter     = 0;

    ಸ್ಪಷ್ಟ ggml_sycl_pool_host(ಸರತಿ_ptr qptr_, ಪೂರ್ಣಾಂಕ device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_POOL_ಗಾತ್ರ; ++i) {
            ggml_sycl_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr != ಶೂನ್ಯಸೂಚಕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(b.ptr, *qptr)));
                b.ptr = ಶೂನ್ಯಸೂಚಕ;
                pool_ಗಾತ್ರ -= b.size;
                b.size = 0;
            }
        }
    }

    ಶೂನ್ಯ * alloc(size_t size, size_t * actual_ಗಾತ್ರ) override {
        ಆದರೆ (counter == MAX_POOL_ಗಾತ್ರ) {
            // the pool wrapped around, no slot is reserved for this allocation
            ಶೂನ್ಯ * ptr = ಶೂನ್ಯಸೂಚಕ;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ಶೂನ್ಯ *) sycl::malloc_host(size, *qptr)));
            ಆದರೆ (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            }
            pool_ಗಾತ್ರ    += size;
            *actual_ಗಾತ್ರ = size;
            counter      = 0;
            ಹಿಂತಿರುಗಿ ptr;
        }
        ggml_sycl_ಬಫರ್ & b = ಬಫರ್_pool[counter];
        ++counter;

        ಆದರೆ (b.ptr != ಶೂನ್ಯಸೂಚಕ) {
            ಆದರೆ (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                ಶೂನ್ಯ * ptr = b.ptr;
                *actual_ಗಾತ್ರ = b.size;
                b.ptr = ಶೂನ್ಯಸೂಚಕ;
                b.size = 0;
                ಹಿಂತಿರುಗಿ ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(b.ptr, *qptr)));
            pool_ಗಾತ್ರ -= b.size;
            b.ptr = ಶೂನ್ಯಸೂಚಕ;
            b.size = 0;
        }

        ಶೂನ್ಯ * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ಶೂನ್ಯ *) sycl::malloc_host(size, *qptr)));
        ಆದರೆ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
        pool_ಗಾತ್ರ    += size;
        *actual_ಗಾತ್ರ = size;
        ಹಿಂತಿರುಗಿ ptr;
    }

    ಶೂನ್ಯ ಬಿಡುಗಡೆ(ಶೂನ್ಯ * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < MAX_POOL_ಗಾತ್ರ; ++i) {
            ggml_sycl_ಬಫರ್ & b = ಬಫರ್_pool[i];
            ಆದರೆ (b.ptr == ಶೂನ್ಯಸೂಚಕ) {
                b.ptr  = ptr;
                b.size = size;
                ಹಿಂತಿರುಗಿ;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ಬಿಡುಗಡೆ(ptr, *qptr)));
        pool_ಗಾತ್ರ -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(ಸರತಿ_ptr qptr, ಪೂರ್ಣಾಂಕ device) {
    // return pool for the host to speed up memory management
    ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_sycl_pool>(ಹೊಸ ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(ಸರತಿ_ptr qptr, ಪೂರ್ಣಾಂಕ device) {
#ಆದರೆ defined(GGML_SYCL_SUPPORT_VMM)
    ಆದರೆ (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_sycl_pool>(ಹೊಸ ggml_sycl_pool_vmm(qptr, device));
    }
#ಆದರೆ_ಅಂತ್ಯ // defined(GGML_SYCL_SUPPORT_VMM)
    ಹಿಂತಿರುಗಿ std::unique_ptr<ggml_sycl_pool>(ಹೊಸ ggml_sycl_pool_leg(qptr, device));
}

/// kernels
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr &stream);



ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_p021_f16_f32(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ y, ದಶಮಾಂಶ * __restrict__ dst,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_y,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ sycl::half *x = (ಸ್ಥಿರಾಂಕ sycl::half *)vx;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_x = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                      ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                        ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_x = channel / (nchannels_y / nchannels_x);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_y = ncols_x;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst = nrows_x;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst = ಅಡ್ಡಸಾಲು_x;

    ದಶಮಾಂಶ tmp = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_x0 = 0; ಲಂಬಸಾಲು_x0 < ncols_x;
         ಲಂಬಸಾಲು_x0 += ಅಂಶ_ct1.get_local_range(2)) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_x = ಲಂಬಸಾಲು_x0 + ಅಂಶ_ct1.get_local_id(2);

        ಆದರೆ (ಲಂಬಸಾಲು_x >= ncols_x) {
            ನಿಲ್ಲಿಸು;
        }

        // x is transposed and permuted
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = ಅಡ್ಡಸಾಲು_x*nchannels_x*ncols_x + channel_x*ncols_x + ಲಂಬಸಾಲು_x;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_y = ಲಂಬಸಾಲು_x;


        // y is not transposed but permuted
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iy = channel*nrows_y + ಅಡ್ಡಸಾಲು_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idst = channel*nrows_dst + ಅಡ್ಡಸಾಲು_dst;

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ y, ದಶಮಾಂಶ * __restrict__ dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_stride_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_stride_x,ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_stride_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_x_ಭಾಜಕ,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ sycl::half *x = (ಸ್ಥಿರಾಂಕ sycl::half *)vx;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_x = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                      ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel = ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) +
                        ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_x = channel / channel_x_ಭಾಜಕ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_dst = nrows_x;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_dst   = ಅಡ್ಡಸಾಲು_x;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idst = channel*nrows_dst + ಅಡ್ಡಸಾಲು_dst;

    ದಶಮಾಂಶ tmp = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_x0 = 0; ಲಂಬಸಾಲು_x0 < ncols_x;
         ಲಂಬಸಾಲು_x0 += ಅಂಶ_ct1.get_local_range(2)) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_x = ಲಂಬಸಾಲು_x0 + ಅಂಶ_ct1.get_local_id(2);

        ಆದರೆ (ಲಂಬಸಾಲು_x >= ncols_x) {
            ನಿಲ್ಲಿಸು;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_y = ಲಂಬಸಾಲು_x;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ix = channel_x*channel_stride_x + ಅಡ್ಡಸಾಲು_x*ಅಡ್ಡಸಾಲು_stride_x + ಲಂಬಸಾಲು_x;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iy = channel * channel_stride_y + ಅಡ್ಡಸಾಲು_y;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ = WARP_ಗಾತ್ರ / 2; ಮುಖವಾಡ > 0; ಮುಖವಾಡ >>= 1) {
        tmp +=
            dpct::ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(ಅಂಶ_ct1.get_sub_ಗುಂಪು(), tmp, ಮುಖವಾಡ);
    }

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ k_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                           ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಅಂಶ_ct1.get_local_id(2);

    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಲಂಬಸಾಲು; i < ncols; i += ಅಂಶ_ct1.get_local_range(2)) {
        ಮೊತ್ತ += x[ಅಡ್ಡಸಾಲು * ncols + i];
    }

    ಮೊತ್ತ = warp_reduce_ಮೊತ್ತ(ಮೊತ್ತ, ಅಂಶ_ct1);

    ಆದರೆ (ಲಂಬಸಾಲು == 0) {
        dst[ಅಡ್ಡಸಾಲು] = ಮೊತ್ತ;
    }
}


ಮಾದರಿ<ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_ಅದಲುಬದಲು(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

ಮಾದರಿ <ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
k_argsort_f32_i32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ಪೂರ್ಣಾಂಕ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಪೂರ್ಣಾಂಕ ncols_pad,
                  ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tasks_ಪ್ರತಿ_thread, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು_ಸೂಚ್ಯಂಕ =  ಅಂಶ_ct1.get_local_id(2);
    ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(1);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < tasks_ಪ್ರತಿ_thread; i++) {
        ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಲಂಬಸಾಲು_ಸೂಚ್ಯಂಕ * tasks_ಪ್ರತಿ_thread + i;
        ಆದರೆ (ಲಂಬಸಾಲು >= ncols_pad) {
            ಹಿಂತಿರುಗಿ;
        }
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಅಡ್ಡಸಾಲು = x + ಅಡ್ಡಸಾಲು * ncols;
    ಸ್ವಯಂಚಾಲಿತ dst_ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ *)dpct_local;

    // initialize indices
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i=0;i<tasks_ಪ್ರತಿ_thread;i++){
        ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಲಂಬಸಾಲು_ಸೂಚ್ಯಂಕ*tasks_ಪ್ರತಿ_thread+i;
        dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] = ಲಂಬಸಾಲು;
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 2; k <= ncols_pad; k *= 2) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = k / 2; j > 0; j /= 2) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < tasks_ಪ್ರತಿ_thread; i++) {
                ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಲಂಬಸಾಲು_ಸೂಚ್ಯಂಕ * tasks_ಪ್ರತಿ_thread + i;
                ಪೂರ್ಣಾಂಕ ixj = ಲಂಬಸಾಲು ^ j;
                ಆದರೆ (ixj > ಲಂಬಸಾಲು) {
                    ಆದರೆ ((ಲಂಬಸಾಲು & k) == 0) {
                        ಆದರೆ (dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] >= ncols ||
                            (dst_ಅಡ್ಡಸಾಲು[ixj] < ncols &&
                             (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC
                                  ? x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು]] > x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ixj]]
                                  : x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು]] <
                                        x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ixj]]))) {
                            ggml_sycl_ಅದಲುಬದಲು(dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು], dst_ಅಡ್ಡಸಾಲು[ixj]);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಆದರೆ (dst_ಅಡ್ಡಸಾಲು[ixj] >= ncols ||
                            (dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] < ncols &&
                             (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC
                                  ? x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು]] < x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ixj]]
                                  : x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು]] >
                                        x_ಅಡ್ಡಸಾಲು[dst_ಅಡ್ಡಸಾಲು[ixj]]))) {
                            ggml_sycl_ಅದಲುಬದಲು(dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು], dst_ಅಡ್ಡಸಾಲು[ixj]);
                        }
                    }
                }
                ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < tasks_ಪ್ರತಿ_thread; i++) {
        ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಲಂಬಸಾಲು_ಸೂಚ್ಯಂಕ * tasks_ಪ್ರತಿ_thread + i;
        ಆದರೆ (ಲಂಬಸಾಲು < ncols) {
            dst[ಅಡ್ಡಸಾಲು * ncols + ಲಂಬಸಾಲು] = dst_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು];
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ diag_ಮುಖವಾಡ_inf_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_channel, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past,
                              ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                    ಅಂಶ_ct1.get_local_id(2);

    ಆದರೆ (ಲಂಬಸಾಲು >= ncols) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = ಅಡ್ಡಸಾಲು*ncols + ಲಂಬಸಾಲು;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (ಲಂಬಸಾಲು > n_past + ಅಡ್ಡಸಾಲು % ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_channel) * FLT_MAX;
}

ಸ್ಥಿರ ಶೂನ್ಯ scale_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ bias, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
                      ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು(2) +
                  ಅಂಶ_ct1.get_local_id(2);

    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    dst[i] = scale * x[i] + bias;
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_p021_f16_f32_sycl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y,
                                           ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_x,
                                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_y,
                                           ಸರತಿ_ptr stream) {

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(nchannels_y, nrows_x, 1);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, ಅಂಶ_ct1);
            });
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_mul_mat_vec_nc_f16_f32_sycl(
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *y, ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು_stride_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_x,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchannels_y, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_stride_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channel_stride_y, ಸರತಿ_ptr stream) {

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(nchannels_y, nrows_x, 1);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       ಅಡ್ಡಸಾಲು_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, ಅಂಶ_ct1);
            });
    }
}



ಸ್ಥಿರ ಶೂನ್ಯ scale_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ bias,
                           ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k, ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (k + SYCL_SCALE_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_SCALE_ಖಂಡ_ಗಾತ್ರ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ಅಂಕೆ_ಖಂಡಗಳು) *
                              sycl::range<3>(1, 1, SYCL_SCALE_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<3>(1, 1, SYCL_SCALE_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            scale_f32(x, dst, scale, bias, k, ಅಂಶ_ct1);
        });
}


ಸ್ಥಿರ ಶೂನ್ಯ ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ದಶಮಾಂಶ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, WARP_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1)
                             [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
                                 k_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32(x, dst, ncols, ಅಂಶ_ct1);
                             });
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ಮುಂದಿನ_power_ದ_2(ಪೂರ್ಣಾಂಕ x) {
    ಪೂರ್ಣಾಂಕ n = 1;
    ಆಗ (n < x) {
        n *= 2;
    }
    ಹಿಂತಿರುಗಿ n;
}

ಸ್ಥಿರ ಶೂನ್ಯ init_argsort_indices_padded(
        ಪೂರ್ಣಾಂಕ * idx,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_pad,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ size_t gid = ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) + ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = (size_t) nrows * (size_t) ncols_pad;

    ಆದರೆ (gid >= ಒಟ್ಟು) {
        ಹಿಂತಿರುಗಿ;
    }

    idx[gid] = (ಪೂರ್ಣಾಂಕ) (gid % (size_t) ncols_pad);
}

ಮಾದರಿ <ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ>
ಸ್ಥಿರ ಶೂನ್ಯ argsort_f32_i32_global_pass(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *            x,
                                        ಪೂರ್ಣಾಂಕ *                    idx,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                ncols,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                nrows,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                ncols_pad,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                j,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                k,
                                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ size_t gid   = ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) + ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = (size_t) nrows * (size_t) ncols_pad;

    ಆದರೆ (gid >= ಒಟ್ಟು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ) (gid / (size_t) ncols_pad);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = (ಪೂರ್ಣಾಂಕ) (gid % (size_t) ncols_pad);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ixj = ಲಂಬಸಾಲು ^ j;

    ಆದರೆ (ixj <= ಲಂಬಸಾಲು || ixj >= ncols_pad) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ size_t ತಳ  = (size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols_pad;
    ಸ್ಥಿರಾಂಕ size_t pos_a = ತಳ + (size_t) ಲಂಬಸಾಲು;
    ಸ್ಥಿರಾಂಕ size_t pos_b = ತಳ + (size_t) ixj;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ a = idx[pos_a];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b = idx[pos_b];

    ತಾರ್ಕಿಕ do_ಅದಲುಬದಲು = ಸುಳ್ಳು;

    ಆದರೆ ((ಲಂಬಸಾಲು & k) == 0) {
        ಆದರೆ (a >= ncols ||
            (b < ncols &&
             (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC ?
                  x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) a] > x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) b] :
                  x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) a] < x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) b]))) {
            do_ಅದಲುಬದಲು = ನಿಜ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (b >= ncols ||
            (a < ncols &&
             (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC ?
                  x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) a] < x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) b] :
                  x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) a] > x[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) b]))) {
            do_ಅದಲುಬದಲು = ನಿಜ;
        }
    }

    ಆದರೆ (do_ಅದಲುಬದಲು) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ copy_argsort_indices_unpadded(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *              idx_padded,
                                          ಪೂರ್ಣಾಂಕ *                    dst,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                nrows,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                ncols,
                                          ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ                ncols_pad,
                                          ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ size_t gid   = ಅಂಶ_ct1.get_local_range(0) * ಅಂಶ_ct1.get_ಗುಂಪು(0) + ಅಂಶ_ct1.get_local_id(0);
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = (size_t) nrows * (size_t) ncols;

    ಆದರೆ (gid >= ಒಟ್ಟು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = (ಪೂರ್ಣಾಂಕ) (gid / (size_t) ncols);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = (ಪೂರ್ಣಾಂಕ) (gid % (size_t) ncols);

    dst[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols + (size_t) ಲಂಬಸಾಲು] = idx_padded[(size_t) ಅಡ್ಡಸಾಲು * (size_t) ncols_pad + (size_t) ಲಂಬಸಾಲು];
}

ಸ್ಥಿರ ಶೂನ್ಯ argsort_f32_i32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ಪೂರ್ಣಾಂಕ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                                 ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ,
                                 ಸರತಿ_ptr stream, ಪೂರ್ಣಾಂಕ device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_pad = ಮುಂದಿನ_power_ದ_2(ncols);
    ಸ್ಥಿರಾಂಕ size_t shared_mem = (size_t) ncols_pad * ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ);
    ಸ್ಥಿರಾಂಕ size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    ಆದರೆ (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<ಪೂರ್ಣಾಂಕ> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        ಪೂರ್ಣಾಂಕ *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t ಖಂಡ_ಗಾತ್ರ_3     = 256;
        ಸ್ಥಿರಾಂಕ size_t     ಒಟ್ಟು_padded   = (size_t) nrows * (size_t) ncols_pad;
        ಸ್ಥಿರಾಂಕ size_t     nblocks_padded = (ಒಟ್ಟು_padded + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, ಅಂಶ_ct1); });

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 2; k <= ncols_pad; k *= 2) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = k / 2; j > 0; j /= 2) {
                ಆದರೆ (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
                        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                            argsort_f32_i32_global_pass<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, ಅಂಶ_ct1);
                        });
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
                        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                            argsort_f32_i32_global_pass<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, ಅಂಶ_ct1);
                        });
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು   = (size_t) nrows * (size_t) ncols;
        ಸ್ಥಿರಾಂಕ size_t nblocks = (ಒಟ್ಟು + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
                             [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, ಅಂಶ_ct1);
                             });

        ಹಿಂತಿರುಗಿ;
    }

    ಪೂರ್ಣಾಂಕ nth = 1;
    ಪೂರ್ಣಾಂಕ max_ಖಂಡ_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[device];
    ಆಗ (nth < ncols_pad && nth < max_ಖಂಡ_ಗಾತ್ರ)
        nth *= 2;
    ಆದರೆ (nth > max_ಖಂಡ_ಗಾತ್ರ)
        nth = max_ಖಂಡ_ಗಾತ್ರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tasks_ಪ್ರತಿ_thread = ncols_pad / nth;

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, nth);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, nrows, 1);

    ಆದರೆ (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    k_argsort_f32_i32<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC>(
                        x, dst, ncols, ncols_pad, tasks_ಪ್ರತಿ_thread, ಅಂಶ_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    k_argsort_f32_i32<GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC>(
                        x, dst, ncols, ncols_pad, tasks_ಪ್ರತಿ_thread, ಅಂಶ_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ABORT("fatal error");
    }
}

// Scan and block merge, shared by every launch shape below so a partitioned row uses the
// same insertion order as an unpartitioned one.
//
// src_map  != nullptr: report src_map[col] instead of col, so a merge pass can carry the
//                      original column index through.
// out_vals != nullptr: also emit the k winning values, for a later merge pass.
// swap01:              emit in the output order the single-pass path uses.
ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_ಪರಿಶೀಲನೆ_ವಿಲೀನ_f32(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src_vals,
    ಸ್ಥಿರಾಂಕ int32_t * src_map,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       begin,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       end,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       k,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ಖಂಡ_ಗಾತ್ರ_3,
    ದಶಮಾಂಶ *         shared_vals,
    ಪೂರ್ಣಾಂಕ *           shared_idx,
    ದಶಮಾಂಶ *         ಹೊರಹೋಗು_vals,
    int32_t *       ಹೊರಹೋಗು_idx,
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ      swap01,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1
) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    ದಶಮಾಂಶ * lv = shared_vals + tid;
    ಪೂರ್ಣಾಂಕ * li = shared_idx + tid;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < k; i++) {
        lv[i * ಖಂಡ_ಗಾತ್ರ_3] = -FLT_MAX;
        li[i * ಖಂಡ_ಗಾತ್ರ_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    ದಶಮಾಂಶ kth = -FLT_MAX;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = begin + tid; ಲಂಬಸಾಲು < end; ಲಂಬಸಾಲು += ಖಂಡ_ಗಾತ್ರ_3) {
        ದಶಮಾಂಶ val = src_vals[ಲಂಬಸಾಲು];

        ಆದರೆ (val > kth) {
            ಪೂರ್ಣಾಂಕ pos = k - 1;
            ಆಗ (pos > 0 && val > lv[(pos - 1) * ಖಂಡ_ಗಾತ್ರ_3]) {
                pos--;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = k - 1; i > pos; i--) {
                lv[i * ಖಂಡ_ಗಾತ್ರ_3] = lv[(i - 1) * ಖಂಡ_ಗಾತ್ರ_3];
                li[i * ಖಂಡ_ಗಾತ್ರ_3] = li[(i - 1) * ಖಂಡ_ಗಾತ್ರ_3];
            }
            lv[pos * ಖಂಡ_ಗಾತ್ರ_3] = val;
            li[pos * ಖಂಡ_ಗಾತ್ರ_3] = src_map ? src_map[ಲಂಬಸಾಲು] : ಲಂಬಸಾಲು;

            kth = lv[(k - 1) * ಖಂಡ_ಗಾತ್ರ_3];
        }
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    ಆದರೆ (tid != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    ದಶಮಾಂಶ * fv = shared_vals + (size_t) k * ಖಂಡ_ಗಾತ್ರ_3;
    ಪೂರ್ಣಾಂಕ * fi = shared_idx + (size_t) k * ಖಂಡ_ಗಾತ್ರ_3;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    ದಶಮಾಂಶ fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ t = 0; t < ಖಂಡ_ಗಾತ್ರ_3; t++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < k; i++) {
            ದಶಮಾಂಶ val = shared_vals[i * ಖಂಡ_ಗಾತ್ರ_3 + t];

            ಆದರೆ (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                ನಿಲ್ಲಿಸು;
            }

            ಪೂರ್ಣಾಂಕ idx = shared_idx[i * ಖಂಡ_ಗಾತ್ರ_3 + t];

            ಪೂರ್ಣಾಂಕ pos = k - 1;
            ಆಗ (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    ಆದರೆ (ಹೊರಹೋಗು_vals) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < k; i++) {
            ಹೊರಹೋಗು_vals[i] = fv[i];
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < k; i++) {
        ಹೊರಹೋಗು_idx[i] = fi[i];
    }

    ಆದರೆ (swap01 && k > 1) {
        int32_t ತಾತ್ಕಾಲಿಕ = ಹೊರಹೋಗು_idx[0];
        ಹೊರಹೋಗು_idx[0] = ಹೊರಹೋಗು_idx[1];
        ಹೊರಹೋಗು_idx[1] = ತಾತ್ಕಾಲಿಕ;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
    int32_t * dst_indices,
    ಸ್ಥಿರಾಂಕ int64_t ncols,
    ಸ್ಥಿರಾಂಕ int64_t nrows,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k,
    dpct::ಸರತಿ_ptr main_ಹರಿವು
) {
    // A row is scanned by exactly one work-group, so a vocabulary-sized row leaves the
    // rest of the device idle. What the scan is short of is memory requests in flight,
    // not bandwidth or per-request latency, so lanes in flight is the lever: split the
    // row across independent work-groups, have each emit its partition's top-k, and
    // merge those nsplit*k candidates in a second launch.
    //
    // split_block trades parallelism against SLM residency. Its cost is
    // (split_block + 1) * k * 8 bytes of SLM per group, so at the k <= 32 ceiling 128
    // lanes need about 33 KB, which leaves a single resident group per Xe-core. Revisit
    // if the supported k ever grows.
    constexpr ಪೂರ್ಣಾಂಕ ವಿಭಜಿಸು_ಖಂಡ = 128;
    constexpr ಪೂರ್ಣಾಂಕ max_splits  = 128;
    constexpr ಪೂರ್ಣಾಂಕ min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ    = 8192;

    ಪೂರ್ಣಾಂಕ nsplit = 1;
    ಆದರೆ (ncols >= min_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ಸ್ಥಿರಾಂಕ int64_t want = ncols / ವಿಭಜಿಸು_ಖಂಡ;
        nsplit = (ಪೂರ್ಣಾಂಕ) (want > max_splits ? max_splits : want);
    }

    ಆದರೆ (nsplit > 1) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nchunk = (ಪೂರ್ಣಾಂಕ) ((ncols + nsplit - 1) / nsplit);
        ಸ್ಥಿರಾಂಕ size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<ದಶಮಾಂಶ>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        ದಶಮಾಂಶ *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_dims(ವಿಭಜಿಸು_ಖಂಡ);

        main_ಹರಿವು->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_vals(sycl::range<1>((ವಿಭಜಿಸು_ಖಂಡ + 1) * k), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> shared_idx(sycl::range<1>((ವಿಭಜಿಸು_ಖಂಡ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grp = ಅಂಶ_ct1.get_ಗುಂಪು(0);
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = grp / nsplit;
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ part = grp % nsplit;

                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ begin = part * nchunk;
                    ಪೂರ್ಣಾಂಕ end = begin + nchunk;
                    ಆದರೆ (end > (ಪೂರ್ಣಾಂಕ) ncols) {
                        end = (ಪೂರ್ಣಾಂಕ) ncols;
                    }

                    ತುದಿ_k_ಪರಿಶೀಲನೆ_ವಿಲೀನ_f32(
                        src + (int64_t) ಅಡ್ಡಸಾಲು * ncols, ಶೂನ್ಯಸೂಚಕ, begin, end, k, ವಿಭಜಿಸು_ಖಂಡ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, ಸುಳ್ಳು, ಅಂಶ_ct1);
                });
        });

        main_ಹರಿವು->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_vals(sycl::range<1>((ವಿಭಜಿಸು_ಖಂಡ + 1) * k), cgh);
            sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> shared_idx(sycl::range<1>((ವಿಭಜಿಸು_ಖಂಡ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * ಖಂಡ_dims, ಖಂಡ_dims),
                [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(0);
                    ಸ್ಥಿರಾಂಕ size_t off = (size_t) ಅಡ್ಡಸಾಲು * nsplit * k;

                    ತುದಿ_k_ಪರಿಶೀಲನೆ_ವಿಲೀನ_f32(
                        pv + off, pi + off, 0, nsplit * k, k, ವಿಭಜಿಸು_ಖಂಡ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        ಶೂನ್ಯಸೂಚಕ, dst_indices + (int64_t) ಅಡ್ಡಸಾಲು * k, ನಿಜ, ಅಂಶ_ct1);
                });
        });

        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಗಾತ್ರ_3 = 128;

    ಸ್ಥಿರಾಂಕ sycl::range<1> ಖಂಡ_dims(ಖಂಡ_ಗಾತ್ರ_3);
    ಸ್ಥಿರಾಂಕ sycl::range<1> grid_dims(nrows);

    main_ಹರಿವು->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_vals(sycl::range<1>((ಖಂಡ_ಗಾತ್ರ_3 + 1) * k), cgh);
        sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> shared_idx(sycl::range<1>((ಖಂಡ_ಗಾತ್ರ_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(0);

                ಆದರೆ (ಅಡ್ಡಸಾಲು >= nrows) ಹಿಂತಿರುಗಿ;

                ತುದಿ_k_ಪರಿಶೀಲನೆ_ವಿಲೀನ_f32(
                    src + (int64_t) ಅಡ್ಡಸಾಲು * ncols, ಶೂನ್ಯಸೂಚಕ, 0, (ಪೂರ್ಣಾಂಕ) ncols, k, ಖಂಡ_ಗಾತ್ರ_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ಶೂನ್ಯಸೂಚಕ, dst_indices + (int64_t) ಅಡ್ಡಸಾಲು * k, ನಿಜ, ಅಂಶ_ct1);
            });
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ argmax_f32_i32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ಪೂರ್ಣಾಂಕ *dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols,
                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_ARGMAX_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, nrows, 1);
    ಸ್ಥಿರಾಂಕ size_t shared_mem = 256 * ಗಾತ್ರ(ದಶಮಾಂಶ);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ದಶಮಾಂಶ, 1> shared_ದತ್ತಾಂಶ(
            sycl::range<1>(shared_mem/ಗಾತ್ರ(ದಶಮಾಂಶ)), cgh);
        sycl::local_accessor<ಪೂರ್ಣಾಂಕ, 1> shared_indices(
            sycl::range<1>(shared_mem/ಗಾತ್ರ(ದಶಮಾಂಶ)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid = ಅಂಶ_ct1.get_local_id(2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_global_id(1);

                ದಶಮಾಂಶ max_val = -INFINITY;
                ಪೂರ್ಣಾಂಕ max_idx = -1;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = tid; ಲಂಬಸಾಲು < ncols; ಲಂಬಸಾಲು += 256) {
                    ದಶಮಾಂಶ val = x[ಅಡ್ಡಸಾಲು * ncols + ಲಂಬಸಾಲು];
                    ಆದರೆ (val > max_val) {
                        max_val = val;
                        max_idx = ಲಂಬಸಾಲು;
                    }
                }

                shared_ದತ್ತಾಂಶ[tid] = max_val;
                shared_indices[tid] = max_idx;
                ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ stride = 256/2; stride > 0; stride >>= 1) {
                    ಆದರೆ (tid < stride) {
                        ದಶಮಾಂಶ val1 = shared_ದತ್ತಾಂಶ[tid];
                        ದಶಮಾಂಶ val2 = shared_ದತ್ತಾಂಶ[tid + stride];
                        ಆದರೆ (val2 > val1) {
                            shared_ದತ್ತಾಂಶ[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
                }


                ಆದರೆ (tid == 0) {
                    dst[ಅಡ್ಡಸಾಲು] = shared_indices[0];
                }
            });
    });
}
ಸ್ಥಿರ ಶೂನ್ಯ diag_ಮುಖವಾಡ_inf_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *x, ದಶಮಾಂಶ *dst,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols_x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows_x,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_channel, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past,
                                   ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, SYCL_DIAG_ಮುಖವಾಡ_INF_ಖಂಡ_ಗಾತ್ರ, 1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (ncols_x + SYCL_DIAG_ಮುಖವಾಡ_INF_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_DIAG_ಮುಖವಾಡ_INF_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             diag_ಮುಖವಾಡ_inf_f32(x, dst, ncols_x,
                                               ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_channel, n_past,
                                               ಅಂಶ_ct1);
                         });
}

ಸ್ಥಿರ dpct::err0 ggml_sycl_cpy_tensor_2d(ಶೂನ್ಯ *dst,
                                          ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_ಕೆಳಗಿನ, int64_t i1_ಮೇಲಿನ,
                                          ಸರತಿ_ptr stream) ಪ್ರಯತ್ನಿಸು {

    dpct::memcpy_direction ಬಗೆ;
    ಅಕ್ಷರ * src_ptr;
    ಆದರೆ (ggml_backend_ಬಫರ್_is_host(src->ಬಫರ್)) {
        ಬಗೆ = dpct::host_ಗಮ್ಯ_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (ಅಕ್ಷರ *) src->ದತ್ತಾಂಶ;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_backend_ಬಫರ್_is_sycl(src->ಬಫರ್)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        ಬಗೆ    = dpct::device_ಗಮ್ಯ_device;
        src_ptr = (ಅಕ್ಷರ *) src->ದತ್ತಾಂಶ;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src->ಬಫರ್)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_ಕೆಳಗಿನ == 0 && i1_ಮೇಲಿನ == src->ne[1]);
        ಬಗೆ = dpct::device_ಗಮ್ಯ_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        ಪೂರ್ಣಾಂಕ id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (ಅಕ್ಷರ *) extra->ದತ್ತಾಂಶ_device[id];
    } ಇಲ್ಲದಿದ್ದರೆ {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    ಅಕ್ಷರ * dst_ptr = (ಅಕ್ಷರ *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src->type;
    ಸ್ಥಿರಾಂಕ int64_t ts = ggml_type_ಗಾತ್ರ(type);
    ಸ್ಥಿರಾಂಕ int64_t bs = ggml_blck_ಗಾತ್ರ(type);
    int64_t i1_diff = i1_ಮೇಲಿನ - i1_ಕೆಳಗಿನ;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * x = src_ptr + i1_ಕೆಳಗಿನ*nb1 + i2*nb2 + i3*nb3;
    ಆದರೆ (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        ಹಿಂತಿರುಗಿ CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    ಬಗೆ, *stream));

    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (nb0 == ts) {
        ಹಿಂತಿರುಗಿ CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, ಬಗೆ, *stream));
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * rx = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) x + i1*nb1);
            ಶೂನ್ಯ * rd = (ಶೂನ್ಯ *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, ಬಗೆ, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            ಆದರೆ (r != 0) ಹಿಂತಿರುಗಿ r;
        }
        ಹಿಂತಿರುಗಿ 0;
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src0_dd_i, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1_ddf_i, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *src1_ddq_i,
    ದಶಮಾಂಶ *dst_dd_i, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ,
    ಸ್ಥಿರಾಂಕ int64_t src1_ncols, ಸ್ಥಿರಾಂಕ int64_t src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr &stream) ಪ್ರಯತ್ನಿಸು {

    GGML_ASSERT(src0_dd_i  != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(src1_ddf_i != ಶೂನ್ಯಸೂಚಕ);
    GGML_ASSERT(dst_dd_i   != ಶೂನ್ಯಸೂಚಕ);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_diff = ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;

    ಪೂರ್ಣಾಂಕ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    ಪೂರ್ಣಾಂಕ ldc = id == ctx.device ? ne0 : ಅಡ್ಡಸಾಲು_diff; // used by MKL only

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ತಾರ್ಕಿಕ use_fp16 = ನಿಜ;  // TODO(Yu) SYCL capability check
#ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ use_fp16 = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    ಆದರೆ (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        ಅಡ್ಡಸಾಲು_diff == src0->ne[1]) {
        ಬಳಸಿ bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        ಆದರೆ (src1->type != GGML_TYPE_BF16) {
            ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_bf16_sycl_t ಗಮ್ಯ_bf16_sycl = ggml_get_ಗಮ್ಯ_bf16_sycl(src1->type, dst);
            GGML_ASSERT(ಗಮ್ಯ_bf16_sycl != ಶೂನ್ಯಸೂಚಕ);
            ಗಮ್ಯ_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            stream->ಸ್ಮರಣೆ_ನಕಲು(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*ಗಾತ್ರ(bf16_t));
        }
        DnnlGemmWrapper::ಅಡ್ಡಸಾಲು_gemm(ctx, ಅಡ್ಡಸಾಲು_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::ಗಮ್ಯ_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::ಗಮ್ಯ_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(), stream);
        GGML_ಬಳಸದ(dst);
        GGML_ಬಳಸದ(src1_ddq_i);
        GGML_ಬಳಸದ(src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    ತಾರ್ಕಿಕ src0_reordered = ಸುಳ್ಳು;
    ಆದರೆ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * src0_extra = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    ಆದರೆ (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && ಅಡ್ಡಸಾಲು_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     ಅಡ್ಡಸಾಲು_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        ಹಿಂತಿರುಗಿ;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    ಆದರೆ ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        ಅಡ್ಡಸಾಲು_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        ಆದರೆ (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16_sycl = ggml_get_ಗಮ್ಯ_fp16_sycl(src1->type, dst);
            GGML_ASSERT(ಗಮ್ಯ_fp16_sycl != ಶೂನ್ಯಸೂಚಕ);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            ಗಮ್ಯ_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ಸ್ಥಿರಾಂಕ sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ಸ್ಥಿರಾಂಕ sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        ಆದರೆ (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16_sycl = ggml_get_ಗಮ್ಯ_fp16_sycl(src0->type, dst);
            GGML_ASSERT(ಗಮ್ಯ_fp16_sycl != ಶೂನ್ಯಸೂಚಕ);
            size_t ne = ಅಡ್ಡಸಾಲು_diff*ne00;
            src0_as_f16.alloc(ne);
            ಗಮ್ಯ_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ಸ್ಥಿರಾಂಕ sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ಸ್ಥಿರಾಂಕ sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#ಆದರೆ GGML_SYCL_DNNL
        ಆದರೆ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::ಅಡ್ಡಸಾಲು_gemm(ctx,ಅಡ್ಡಸಾಲು_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(), stream);
        }
        ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
        {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, ಅಡ್ಡಸಾಲು_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_ದತ್ತಾಂಶ_t::real_half, ne00,
                src1_ptr, dpct::library_ದತ್ತಾಂಶ_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_ದತ್ತಾಂಶ_t::real_float, ldc,
                dpct::library_ದತ್ತಾಂಶ_t::real_float)));
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_sycl_pool_alloc<ದಶಮಾಂಶ> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<ದಶಮಾಂಶ> src1_ddq_as_f32(ctx.pool());
        ಆದರೆ (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp32_sycl_t ಗಮ್ಯ_fp32_sycl = ggml_get_ಗಮ್ಯ_fp32_sycl(src0->type, dst);
            GGML_ASSERT(ಗಮ್ಯ_fp32_sycl != ಶೂನ್ಯಸೂಚಕ);
            src0_ddq_as_f32.alloc(ಅಡ್ಡಸಾಲು_diff*ne00);
            ಗಮ್ಯ_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), ಅಡ್ಡಸಾಲು_diff*ne00, stream);
        }
        ಆದರೆ (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp32_sycl_t ಗಮ್ಯ_fp32_sycl = ggml_get_ಗಮ್ಯ_fp32_sycl(src1->type, dst);
            GGML_ASSERT(ಗಮ್ಯ_fp32_sycl != ಶೂನ್ಯಸೂಚಕ);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            ಗಮ್ಯ_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0_dd_i : src0_ddq_as_f32.get();
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#ಆದರೆ GGML_SYCL_DNNL
            ಸ್ಥಿರಾಂಕ int64_t gemm_flops = (int64_t)ಅಡ್ಡಸಾಲು_diff * src1_ncols * ne10;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_mkl_ನೇರ = gemm_flops < 256 * 256 * 256;
            ಆದರೆ (g_ggml_sycl_enable_dnn && !use_mkl_ನೇರ && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::ಅಡ್ಡಸಾಲು_gemm(ctx, ಅಡ್ಡಸಾಲು_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(), src1_ddf1_i, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(),
                                          dst_dd_i, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(), stream);
            }
            ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
            {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha = 1.0f;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ಅಡ್ಡಸಾಲು_diff,
                    src1_ncols, ne10, dpct::get_ಮೌಲ್ಯ(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_ಮೌಲ್ಯ(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(src1_ddq_i);
    GGML_ಬಳಸದ(src1_padded_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಮೊತ್ತ(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_sycl(src0_dd, dst_dd, ne, 1, main_ಹರಿವು);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ncols = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ಹರಿವು);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_ಸರಾಸರಿ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ncols = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ಹರಿವು);

    main_ಹರಿವು->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> ಅಡ್ಡಸಾಲು) {
            dst_dd[ಅಡ್ಡಸಾಲು] /= ncols;
        }
    );
}


ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ದತ್ತಾಂಶ);


    ಸ್ಥಿರಾಂಕ int64_t ncols = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(dst->src[0]);

    ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (ಪೂರ್ಣಾಂಕ *)dst_dd, ncols, nrows, ಕ್ರಮ,
                         main_ಹರಿವು, ctx.device, ctx.pool());
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_ತುದಿ_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
    int32_t * dst_dd = static_cast<int32_t *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ncols = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    ಆದರೆ (k <= SYCL_ತುದಿ_K_ಪರಿಶೀಲನೆ_ವಿಲೀನ_MAX_K) {
        ತುದಿ_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ಹರಿವು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_sycl_ತುದಿ_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ಹರಿವು);
    }
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ncols = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_ಹರಿವು);
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_diag_ಮುಖವಾಡ_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = dst->src[0]->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = dst->src[0]->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_past = ((int32_t *) dst->op_params)[0];

    diag_ಮುಖವಾಡ_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_ಹರಿವು);
}

ಸ್ಥಿರ ಶೂನ್ಯ tri_f32_sycl(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
    ದಶಮಾಂಶ * dst,
    ಸ್ಥಿರಾಂಕ int64_t ne0,
    ಸ್ಥಿರಾಂಕ int64_t ne1,
    ಸ್ಥಿರಾಂಕ int64_t ne2,
    ಸ್ಥಿರಾಂಕ int64_t ne3,
    ಸ್ಥಿರಾಂಕ ggml_tri_type ttype,
    dpct::ಸರತಿ_ptr main_ಹರಿವು
) {
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_ಹರಿವು->parallel_for(sycl::range<1>(ಒಟ್ಟು), [=](sycl::id<1> tid) {
        ಸ್ಥಿರಾಂಕ int64_t idx = (int64_t) tid[0];

        ಸ್ಥಿರಾಂಕ int64_t i0 = idx % ne0;
        ಸ್ಥಿರಾಂಕ int64_t t1 = idx / ne0;
        ಸ್ಥಿರಾಂಕ int64_t i1 = t1 % ne1;

        ತಾರ್ಕಿಕ keep = ಸುಳ್ಳು;
        ಆಯ್ಕೆ (ttype) {
            ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ:      keep = (i0 <  i1); ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TRI_TYPE_ಅಧೋ_DIAG: keep = (i0 <= i1); ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ:      keep = (i0 >  i1); ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_TRI_TYPE_ಊರ್ಧ್ವ_DIAG: keep = (i0 >= i1); ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ: keep = ಸುಳ್ಳು; ನಿಲ್ಲಿಸು;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ಸ್ಥಿರಾಂಕ int64_t ne0 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_ಹರಿವು);
}


ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_dd = static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(dst->src[0]->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *       dst_dd  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ bias;
    ಸ್ಮರಣೆ_ನಕಲು(&scale, (ದಶಮಾಂಶ *) dst->op_params + 0, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&bias,  (ದಶಮಾಂಶ *) dst->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_ಹರಿವು);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಗಣ_peer_access(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_ಟೋಕನುಗಳು, [[maybe_unused]] ಪೂರ್ಣಾಂಕ main_device) {
    ಸ್ಥಿರ ತಾರ್ಕಿಕ peer_access_enabled = ಸುಳ್ಳು;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ enable_peer_access = n_ಟೋಕನುಗಳು <= GGML_SYCL_PEER_MAX_BATCH_ಗಾತ್ರ;

    ಆದರೆ (peer_access_enabled == enable_peer_access) {
        ಹಿಂತಿರುಗಿ;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ NDEBUG
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        SYCL_CHECK(ggml_sycl_ಗಣ_device(i));
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        SYCL_CHECK(ggml_sycl_ಗಣ_device(i));

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id_other = 0; id_other < ggml_sycl_info().device_ಎಣಿಕೆ; ++id_other) {
            ಆದರೆ (i == id_other) {
                ಮುಂದುವರಿ;
            }
            ಆದರೆ (i != main_device && id_other != main_device) {
                ಮುಂದುವರಿ;
            }

            // int can_access_peer;
            // SYCL_CHECK(syclDeviceCanAccessPeer(&can_access_peer, id, id_other));
            // if (can_access_peer) {
            //     if (enable_peer_access) {
            //         SYCL_CHECK(syclDeviceEnablePeerAccess(id_other, 0));
            //     } else {
            //         SYCL_CHECK(syclDeviceDisablePeerAccess(id_other));
            //     }
            // }
        }
    }
#ಆದರೆ_ಅಂತ್ಯ // NDEBUG

    peer_access_enabled = enable_peer_access;
}

ಮಾದರಿ <ಮಾದರಿ <ಪೂರ್ಣಾಂಕ> ಬಗೆ_ಹೆಸರು quantize_f>
ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0,
                                 ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) ಪ್ರಯತ್ನಿಸು {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ಸ್ಥಿರಾಂಕ int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(dst->ಬಫರ್));
    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src1->ಬಫರ್));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ಸ್ಥಿರಾಂಕ int64_t i02_ಭಾಜಕ = ne12 / ne02;

    ಸ್ಥಿರಾಂಕ size_t src0_ts = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t src0_bs = ggml_blck_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t q8_1_ts = ಗಾತ್ರ(ಖಂಡ_q8_1);
    ಸ್ಥಿರಾಂಕ size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_is_contiguous = ggml_is_contiguous(src0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ = GGML_PAD(ne10, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ವಿಭಜಿಸು = ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್);
    GGML_ASSERT(!(ವಿಭಜಿಸು && ne02 > 1));
    GGML_ASSERT(!(ವಿಭಜಿಸು && ne03 > 1));
    GGML_ASSERT(!(ವಿಭಜಿಸು && ne02 < ne12));

    std::array<ದಶಮಾಂಶ, GGML_SYCL_MAX_DEVICES> tensor_ವಿಭಜಿಸು;
    ಆದರೆ (ವಿಭಜಿಸು) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *) src0->ಬಫರ್->buft->context;
        tensor_ವಿಭಜಿಸು = buft_ctx->tensor_ವಿಭಜಿಸು;
    }

    ರಚನೆ dev_ದತ್ತಾಂಶ {
        ggml_sycl_pool_alloc<ಅಕ್ಷರ> src0_dd_alloc;
        ggml_sycl_pool_alloc<ದಶಮಾಂಶ> src1_ddf_alloc;
        ggml_sycl_pool_alloc<ಅಕ್ಷರ> src1_ddq_alloc;
        ggml_sycl_pool_alloc<ದಶಮಾಂಶ> dst_dd_alloc;

        ಅಕ್ಷರ *src0_dd = ಶೂನ್ಯಸೂಚಕ;
        ದಶಮಾಂಶ *src1_ddf = ಶೂನ್ಯಸೂಚಕ; // float
        ಅಕ್ಷರ *src1_ddq = ಶೂನ್ಯಸೂಚಕ;  // q8_1
        ದಶಮಾಂಶ *dst_dd = ಶೂನ್ಯಸೂಚಕ;

        int64_t ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;
        int64_t ಅಡ್ಡಸಾಲು_ಮೇಲಿನ;
    };

    dev_ದತ್ತಾಂಶ dev[GGML_SYCL_MAX_DEVICES];

    ಪೂರ್ಣಾಂಕ ಬಳಸಿದ_devices = 0;
    ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        // by default, use all rows
        dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ  = 0;
        dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        ಆದರೆ (ವಿಭಜಿಸು) {
            ಸ್ಥಿರಾಂಕ int64_t rounding = get_ಅಡ್ಡಸಾಲು_rounding(src0->type, tensor_ವಿಭಜಿಸು);

            ಆದರೆ (i != 0) {
                dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ  = ne01*tensor_ವಿಭಜಿಸು[i];
                ಆದರೆ (dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ < ne01) {
                    dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ -= dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ % rounding;
                }
            }

            ಆದರೆ (i != ggml_sycl_info().device_ಎಣಿಕೆ - 1) {
                dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ  = ne01*tensor_ವಿಭಜಿಸು[i + 1];
                ಆದರೆ (dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ < ne01) {
                    dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ -= dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ % rounding;
                }
            }
        }
    }

    constexpr ತಾರ್ಕಿಕ quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_ಗಾತ್ರ>,
                                                      no_quantize_q8_1<QK8_1 / WARP_ಗಾತ್ರ>>;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
        ಆದರೆ ((!ವಿಭಜಿಸು && i != ctx.device) || dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ == dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ) {
            ಮುಂದುವರಿ;
        }

        ಬಳಸಿದ_devices++;

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_on_device = i == ctx.device;
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  dst_on_device = i == ctx.device;

        ggml_sycl_ಗಣ_device(i);
        ಸರತಿ_ptr stream = ctx.stream(i, 0);

        ಆದರೆ (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = ವಿಭಜಿಸು ? (ಅಕ್ಷರ *) src0_extra->ದತ್ತಾಂಶ_device[i] : (ಅಕ್ಷರ *) src0->ದತ್ತಾಂಶ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        ಆದರೆ (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        ಆದರೆ constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ*q8_1_ts/q8_1_bs);

            ಆದರೆ (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                ಪ್ರಯತ್ನಿಸು {
                    quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ, stream);
                } ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
                    std::ನಿರ್ಗಮಿಸು(1);
                }
            }
        }

        ಆದರೆ (dst_on_device) {
            dev[i].dst_dd = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_dst_ddf = ವಿಭಜಿಸು ? (dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), ಗಾತ್ರ_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    ಆದರೆ (ವಿಭಜಿಸು && ಬಳಸಿದ_devices > 1) {
        ggml_sycl_ಗಣ_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ಸ್ಥಿರಾಂಕ int64_t src1_ಲಂಬಸಾಲು_stride = ವಿಭಜಿಸು && ಬಳಸಿದ_devices > 1 ? MUL_MAT_SRC1_ಲಂಬಸಾಲು_STRIDE : ne11;
    ಗಾಗಿ (int64_t src1_ಲಂಬಸಾಲು_0 = 0; src1_ಲಂಬಸಾಲು_0 < ne11; src1_ಲಂಬಸಾಲು_0 += src1_ಲಂಬಸಾಲು_stride) {
        ಸ್ಥಿರಾಂಕ int64_t is = ವಿಭಜಿಸು ? (src1_ಲಂಬಸಾಲು_0/src1_ಲಂಬಸಾಲು_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ಸ್ಥಿರಾಂಕ int64_t src1_ncols = src1_ಲಂಬಸಾಲು_0 + src1_ಲಂಬಸಾಲು_stride > ne11 ? ne11 - src1_ಲಂಬಸಾಲು_0 : src1_ಲಂಬಸಾಲು_stride;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
            ಆದರೆ ((!ವಿಭಜಿಸು && i != ctx.device) || dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ == dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src1_on_device = i == ctx.device;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  dst_on_device = i == ctx.device;
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_diff = dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ - dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;

            ggml_sycl_ಗಣ_device(i);
            ಸರತಿ_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            ಆದರೆ (ವಿಭಜಿಸು && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            ಗಾಗಿ (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ಸ್ಥಿರಾಂಕ int64_t i03 = i0 / ne12;
                ಸ್ಥಿರಾಂಕ int64_t i02 = i0 % ne12;

                ಸ್ಥಿರಾಂಕ size_t src1_ddq_i_ಸ್ಥಾನಾಂತರ = (i0*ne11 + src1_ಲಂಬಸಾಲು_0) * src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                ಅಕ್ಷರ  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ಭಾಜಕ) * (ne01*ne00*src0_ts)/src0_bs;
                ದಶಮಾಂಶ * src1_ddf_i = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dev[i].src1_ddf +
                    (i0*ne11 + src1_ಲಂಬಸಾಲು_0) * ne10 * ggml_type_ಗಾತ್ರ(src1->type));
                ಅಕ್ಷರ  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_ಸ್ಥಾನಾಂತರ;
                ದಶಮಾಂಶ *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_ಲಂಬಸಾಲು_0) * (dst_on_device ? ne0 : ಅಡ್ಡಸಾಲು_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                ಆದರೆ (i == ctx.device) {
                    dst_dd_i += dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                ಆದರೆ (src1_is_contiguous) {
                    ಆದರೆ (i != ctx.device) {
                        ಆದರೆ constexpr (quantize_enabled) {
                            ಅಕ್ಷರ * src1_ddq_i_ಮೂಲ = dev[ctx.device].src1_ddq + src1_ddq_i_ಸ್ಥಾನಾಂತರ;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->ಸ್ಮರಣೆ_ನಕಲು(src1_ddq_i, src1_ddq_i_ಮೂಲ,
                                                             src1_ncols * src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ddf_i_ಮೂಲ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_ಲಂಬಸಾಲು_0) * ne10 * ggml_type_ಗಾತ್ರ(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_ಹರಿವು, src1_ddf_i, src1_ddf_i_ಮೂಲ,
                                                               src1_ncols * ne10 * ggml_type_ಗಾತ್ರ(src1->type))));
                        }
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಆದರೆ (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_ಲಂಬಸಾಲು_0,
                                                           src1_ಲಂಬಸಾಲು_0 + src1_ncols, stream));
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    ಆದರೆ constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        ಪ್ರಯತ್ನಿಸು {
                            quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ, stream);
                        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
                            std::ನಿರ್ಗಮಿಸು(1);
                        }
                    }
                }

                ಆದರೆ (src1_ಲಂಬಸಾಲು_0 == 0 && !src0_is_contiguous && i02 % i02_ಭಾಜಕ == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ಭಾಜಕ, dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, stream));
                }
                ಆದರೆ (src1->type == GGML_TYPE_F16) {
                    src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ = (i0 * ne11 + src1_ಲಂಬಸಾಲು_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ, dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ, src1_ncols, src1_padded_ಲಂಬಸಾಲು_ಗಾತ್ರ, stream)));

                // copy dst to host or other device if necessary
                ಆದರೆ (!dst_on_device) {
                    ಶೂನ್ಯ * dst_off_device = dst->ದತ್ತಾಂಶ;
                    ಆದರೆ (ವಿಭಜಿಸು) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        ದಶಮಾಂಶ * dhf_dst_i = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ಗಾತ್ರ(ದಶಮಾಂಶ));
                        dhf_dst_i += src1_ಲಂಬಸಾಲು_0*ne0 + dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ), dst_dd_i,
                            ಅಡ್ಡಸಾಲು_diff * ಗಾತ್ರ(ದಶಮಾಂಶ), ಅಡ್ಡಸಾಲು_diff * ಗಾತ್ರ(ದಶಮಾಂಶ),
                            src1_ncols, dpct::device_ಗಮ್ಯ_device, *stream)));
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ದಶಮಾಂಶ * dhf_dst_i = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ಗಾತ್ರ(ದಶಮಾಂಶ));
                        dhf_dst_i += src1_ಲಂಬಸಾಲು_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->ಸ್ಮರಣೆ_ನಕಲು(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * ಗಾತ್ರ(ದಶಮಾಂಶ)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                ಆದರೆ (ವಿಭಜಿಸು && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    ಆದರೆ (ವಿಭಜಿಸು && ggml_sycl_info().device_ಎಣಿಕೆ > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_ಲಂಬಸಾಲು_STRIDE - 1) / MUL_MAT_SRC1_ಲಂಬಸಾಲು_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_ಗಣ_device(ctx.device);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; ++i) {
            ಆದರೆ (dev[i].ಅಡ್ಡಸಾಲು_ಕೆಳಗಿನ == dev[i].ಅಡ್ಡಸಾಲು_ಮೇಲಿನ) {
                ಮುಂದುವರಿ;
            }
            ಗಾಗಿ (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_repeat_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_ಹಿಂಭಾಗ(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_rms_norm_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_ಹಿಂಭಾಗ(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಗುಂಪು_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ಗುಂಪು_norm(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0,
                                       ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                                       ggml_tensor *dst) ಪ್ರಯತ್ನಿಸು {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];

    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();

    ಶೂನ್ಯ  * src0_ddq = src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src1_ddf = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ddf  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_ಹರಿವು);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0,
                                     ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                                     ggml_tensor *dst) ಪ್ರಯತ್ನಿಸು {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];

    ಸ್ಥಿರಾಂಕ int64_t nb01 = src0->nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb02 = src0->nb[2];

    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸರತಿ_ptr main_ಹರಿವು = ctx.stream();

    ಶೂನ್ಯ  * src0_ddq = src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * src1_ddf = (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ddf  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_stride_x = nb01 / ಗಾತ್ರ(sycl::half);
    ಸ್ಥಿರಾಂಕ int64_t channel_stride_x = nb02 / ಗಾತ್ರ(sycl::half);
    ಸ್ಥಿರಾಂಕ int64_t channel_stride_y = nb11 / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ಅಡ್ಡಸಾಲು_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_ಹರಿವು);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ k_compute_batched_ptrs(ಸ್ಥಿರಾಂಕ sycl::half * src0_as_f16, ಸ್ಥಿರಾಂಕ sycl::half * src1_as_f16, ಶೂನ್ಯ * dst,
                                   ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ ** ptrs_src, ಶೂನ್ಯ ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t i13 = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) + ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t i12 = ಅಂಶ_ct1.get_ಗುಂಪು(1) * ಅಂಶ_ct1.get_local_range(1) + ಅಂಶ_ct1.get_local_id(1);

    ಆದರೆ (i13 >= ne13 || i12 >= ne12) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t i03 = i13 / r3;
    ಸ್ಥಿರಾಂಕ int64_t i02 = i12 / r2;

    ಸ್ಥಿರಾಂಕ uint8_t * src0_ಬೈಟುಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src0_as_f16);
    ಸ್ಥಿರಾಂಕ uint8_t * src1_ಬೈಟುಗಳು = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(src1_as_f16);
    uint8_t *       dst_ಬೈಟುಗಳು  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_ಬೈಟುಗಳು + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_ಬೈಟುಗಳು + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_ಬೈಟುಗಳು + i12 * nbd2 + i13 * nbd3;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                           ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) ಪ್ರಯತ್ನಿಸು {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));
    ಸರತಿ_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ಸ್ಥಿರಾಂಕ sycl::half * src0_f16 = static_cast<ಸ್ಥಿರಾಂಕ sycl::half *>(src0->ದತ್ತಾಂಶ);
    ದಶಮಾಂಶ *            dst_ddf  = static_cast<ದಶಮಾಂಶ *>(dst->ದತ್ತಾಂಶ);

    ಸ್ಥಿರಾಂಕ sycl::half * src1_f16       = static_cast<ಸ್ಥಿರಾಂಕ sycl::half *>(src1->ದತ್ತಾಂಶ);
#ಆದರೆ GGML_SYCL_DNNL
    ಸ್ಥಿರಾಂಕ size_t       type_ಗಾತ್ರ_src0 = ggml_type_ಗಾತ್ರ(src0->type);
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ size_t       type_ಗಾತ್ರ_src1 = ggml_type_ಗಾತ್ರ(src1->type);

    ತಾರ್ಕಿಕ is_src0_cont_2 = ggml_is_contiguous_2(src0);
    ತಾರ್ಕಿಕ is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_ಗಾತ್ರ_src1;
    int64_t                          s12 = nb12 / type_ಗಾತ್ರ_src1;
    int64_t                          s13 = nb13 / type_ಗಾತ್ರ_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    ಆದರೆ (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#ಆದರೆ GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_dim=0;
        ಪೂರ್ಣಾಂಕ ಕೊನೆಯ_str=0;
        size_t ಅತಿದೊಡ್ಡ_str=0;
        ಗಾಗಿ(ಪೂರ್ಣಾಂಕ i = 0; i< 4; i++){
            // last stride is always the largest
            ಆದರೆ(src1->nb[i] == ಅತಿದೊಡ್ಡ_str){
                ಆದರೆ(src1->ne[ಕೊನೆಯ_dim] == 1){
                    ಕೊನೆಯ_str = i;
                    ಕೊನೆಯ_dim = i;
                }
            }
            ಆದರೆ(src1->nb[i] > ಅತಿದೊಡ್ಡ_str){
                ಅತಿದೊಡ್ಡ_str = src1->nb[i];
                ಕೊನೆಯ_str = i;
                ಕೊನೆಯ_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ಸ್ಥಿರಾಂಕ int64_t ne_src1 = src1->nb[ಕೊನೆಯ_str] * src1->ne[ಕೊನೆಯ_dim] / type_ಗಾತ್ರ_src1;
        src1_f16_alloc.alloc(ne_src1);
        ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp16_sycl_t ಗಮ್ಯ_fp16_sycl = ggml_get_ಗಮ್ಯ_fp16_sycl(src1->type, dst);
        GGML_ASSERT(ಗಮ್ಯ_fp16_sycl != ಶೂನ್ಯಸೂಚಕ);
        ಗಮ್ಯ_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# ಇಲ್ಲದಿದ್ದರೆ
        ಸ್ಥಿರಾಂಕ int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ಸ್ಥಿರಾಂಕ ಗಮ್ಯ_fp16_nc_sycl_t ಗಮ್ಯ_fp16_nc_sycl = ggml_get_ಗಮ್ಯ_fp16_nc_sycl(src1->type);
        GGML_ASSERT(ಗಮ್ಯ_fp16_nc_sycl != ಶೂನ್ಯಸೂಚಕ);
        ಗಮ್ಯ_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#ಆದರೆ_ಅಂತ್ಯ

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = ನಿಜ;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_ದತ್ತಾಂಶ_t mkl_compute_type = dpct::library_ದತ್ತಾಂಶ_t::real_float;
    dpct::library_ದತ್ತಾಂಶ_t mkl_ದತ್ತಾಂಶ_type    = dpct::library_ದತ್ತಾಂಶ_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_f32 = 1.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta_f32  = 0.0f;

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha = &alpha_f32;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13 / ne03;

#ಆದರೆ GGML_SYCL_DNNL
    ಆದರೆ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_ಗಾತ್ರ_src0;
            int64_t str_a1 = nb01 / type_ಗಾತ್ರ_src0;
            int64_t str_a2 = nb02 / type_ಗಾತ್ರ_src0;

            int64_t str_b0 = nb10 / type_ಗಾತ್ರ_src1;
            int64_t str_b1 = nb11 / type_ಗಾತ್ರ_src1;
            int64_t str_b2 = nb12 / type_ಗಾತ್ರ_src1;

            ಸ್ವಯಂಚಾಲಿತ launch_gemm_for_batches = [&ctx, queue](ಸ್ಥಿರಾಂಕ sycl::half *src0,
                                                ಸ್ಥಿರಾಂಕ sycl::half *src1, ದಶಮಾಂಶ *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                ತಾರ್ಕಿಕ supported_ಪ್ರಸಾರ = batchb == batcha ? ನಿಜ
                        : batchb == 1 || batcha == 1        ? ನಿಜ
                                                            : ಸುಳ್ಳು;
                ಆದರೆ (supported_ಪ್ರಸಾರ) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(), queue, batcha, batchb);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    ಆದರೆ (batches0 > batches1) {
                        int64_t ಅಂಕೆ_mul_mats = batches1;
                        int64_t sub_batch = batches0 / ಅಂಕೆ_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        ಗಾಗಿ (int64_t i0 = 0; i0 < ಅಂಕೆ_mul_mats; i0++) {
                            ಸ್ಥಿರಾಂಕ sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ಸ್ಥಿರಾಂಕ sycl::half *src1_shifted = src1 + (sb2 * i0);
                            ದಶಮಾಂಶ *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(),
                                    queue, sub_batch, 1);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        int64_t ಅಂಕೆ_mul_mats = batches0;
                        int64_t sub_batch = batches1 / ಅಂಕೆ_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        ಗಾಗಿ (int64_t i1 = 0; i1 < ಅಂಕೆ_mul_mats; i1++) {
                            ಸ್ಥಿರಾಂಕ sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ಸ್ಥಿರಾಂಕ sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            ದಶಮಾಂಶ *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::ಗಮ್ಯ_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::ಗಮ್ಯ_dt<ದಶಮಾಂಶ>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cont_batches_dim2_a = nb02 * ne02 == nb03;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cont_batches_dim2_b = nb12 * ne12 == nb13;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            ಆದರೆ (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ));
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_ಗಾತ್ರ_src0;
                int64_t str_b3 = nb13 / type_ಗಾತ್ರ_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ));
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಗಾಗಿ (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ಸ್ಥಿರಾಂಕ sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_ಗಾತ್ರ_src0);
                    ಸ್ಥಿರಾಂಕ sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_ಗಾತ್ರ_src1);
                    ದಶಮಾಂಶ *dst_shifted = dst_ddf + (nb3 * b_a / ಗಾತ್ರ(ದಶಮಾಂಶ));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / ಗಾತ್ರ(ದಶಮಾಂಶ));
                }
            }

    }
    ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
    {
        ಆದರೆ (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ಸ್ಥಿರಾಂಕ int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ಸ್ಥಿರಾಂಕ int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_ದತ್ತಾಂಶ_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_ದತ್ತಾಂಶ_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_ದತ್ತಾಂಶ_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<ಶೂನ್ಯ *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<ಮಾತೃಕೆ_info_t<ದಶಮಾಂಶ>> ಮಾತೃಕೆ_info(ctx.host_pool(), 1);

            sycl::range<3> ಖಂಡ_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ ** ptrs_src_get = ptrs_src.get();
                ಶೂನ್ಯ **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * ಗಾತ್ರ(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * ಗಾತ್ರ(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(ಖಂಡ_dims, ಖಂಡ_dims), [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, ಅಂಶ_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ **) (ptrs_src.get() + 0 * ne23), dpct::library_ದತ್ತಾಂಶ_t::real_half, nb01 / nb00,
                (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ **) (ptrs_src.get() + 1 * ne23), dpct::library_ದತ್ತಾಂಶ_t::real_half, s11, beta,
                (ಶೂನ್ಯ **) (ptrs_dst.get() + 0 * ne23), mkl_ದತ್ತಾಂಶ_type, ne0, ne23, mkl_compute_type, ಮಾತೃಕೆ_info.get())));
        }
    }
} ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

ಎಣಿಕೆ ವರ್ಗ mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_supports_mmq(ಎಣಿಕೆ ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_ಬಳಸದ(type);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_supports_reorder_mul_mat_sycl(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಹಿಂತಿರುಗಿ !g_ggml_sycl_prioritize_dmmv;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_supports_reorder_dmmv(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸಾಲೊಳಗೆ ತಾರ್ಕಿಕ ggml_sycl_supports_reorder_mmvq(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_supports_reorder_esimd(ಎಣಿಕೆ ggml_type type) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(type);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_supports_dmmv(ಎಣಿಕೆ ggml_type type) {
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
        ಸ್ಥಿತಿ GGML_TYPE_F16:
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ * sycl_ext_malloc_device(dpct::ಸರತಿ_ptr stream, size_t size) {
    ತಾರ್ಕಿಕ use_async = g_ggml_sycl_use_async_mem_op;
#ಆದರೆ defined(GGML_SYCL_ನಕ್ಷೆ) && SYCL_EXT_ONEAPI_ASYNC_ಸ್ಮರಣೆ_ALLOC
    ಆದರೆ (use_async) {
        ಶೂನ್ಯ * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_ಕೂಡಿಸು(GGML_SYCL_MEM_ASYNC, ptr, size);
        ಹಿಂತಿರುಗಿ ptr;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ggml_sycl_malloc_device(size, *stream);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sycl_ext_free(dpct::ಸರತಿ_ptr stream, ಶೂನ್ಯ * ptr) {
    ತಾರ್ಕಿಕ use_async = g_ggml_sycl_use_async_mem_op;
#ಆದರೆ defined(GGML_SYCL_ನಕ್ಷೆ) && SYCL_EXT_ONEAPI_ASYNC_ಸ್ಮರಣೆ_ALLOC
    ಆದರೆ (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        ಹಿಂತಿರುಗಿ;
    }
#ಇಲ್ಲದಿದ್ದರೆ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#ಆದರೆ_ಅಂತ್ಯ
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
ರಚನೆ sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ {
    ಶೂನ್ಯ *          ptr  = ಶೂನ್ಯಸೂಚಕ;
    dpct::ಸರತಿ_ptr stream;

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್(dpct::ಸರತಿ_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HOST_MEM_FALLBACK
        ಆದರೆ (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            ಆದರೆ (ptr) {
                host_fallback = ನಿಜ;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
    }

    ~sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್() {
        ಆದರೆ (!ptr) {
            ಹಿಂತಿರುಗಿ;
        }
        ಆದರೆ (host_fallback) {
            sycl::ಬಿಡುಗಡೆ(ptr, *stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            sycl_ext_free(stream, ptr);
        }
    }

    ಸ್ಪಷ್ಟ ಕ್ರಿಯಾಕಾರಕ ತಾರ್ಕಿಕ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ptr != ಶೂನ್ಯಸೂಚಕ; }

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್(ಸ್ಥಿರಾಂಕ sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ &)            = ಅಳಿಸು;
    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ & ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ &) = ಅಳಿಸು;

ಖಾಸಗಿ:
    ತಾರ್ಕಿಕ host_fallback = ಸುಳ್ಳು;
};

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q4_0(uint8_t * ದತ್ತಾಂಶ_device, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, size_t size, size_t ಸ್ಥಾನಾಂತರ,
                            dpct::ಸರತಿ_ptr stream) {
    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ಗಾತ್ರ(ಖಂಡ_q4_0) == 0));
    GGML_ASSERT((ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q4_0) == 0));
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_blks = ಸ್ಥಾನಾಂತರ / ಗಾತ್ರ(ಖಂಡ_q4_0);
    ಸ್ವಯಂಚಾಲಿತ qs_ptr      = ದತ್ತಾಂಶ_device + ಸ್ಥಾನಾಂತರ_blks * QK4_0 / 2;
    ಸ್ವಯಂಚಾಲಿತ d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + ಸ್ಥಾನಾಂತರ_blks;

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(
        size / ಗಾತ್ರ(ಖಂಡ_q4_0),
            [=](ಸ್ವಯಂಚಾಲಿತ i) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0* x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0*)tmp_buf;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q8_0(uint8_t * ದತ್ತಾಂಶ_device, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ncols, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nrows, size_t size, size_t ಸ್ಥಾನಾಂತರ,
                            dpct::ಸರತಿ_ptr stream) {
    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ಗಾತ್ರ(ಖಂಡ_q8_0) == 0));
    GGML_ASSERT((ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q8_0) == 0));
    ಪೂರ್ಣಾಂಕ ಸ್ಥಾನಾಂತರ_blks = ಸ್ಥಾನಾಂತರ / ಗಾತ್ರ(ಖಂಡ_q8_0);
    ಸ್ವಯಂಚಾಲಿತ qs_ptr = ದತ್ತಾಂಶ_device + ಸ್ಥಾನಾಂತರ_blks * QK8_0;
    ಸ್ವಯಂಚಾಲಿತ d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + ಸ್ಥಾನಾಂತರ_blks;

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(
        size / ಗಾತ್ರ(ಖಂಡ_q8_0),
            [=](ಸ್ವಯಂಚಾಲಿತ i) [[sycl::reqd_sub_ಗುಂಪು_ಗಾತ್ರ(WARP_ಗಾತ್ರ)]] {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0* x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0*)tmp_buf;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q8_0_moe(uint8_t * ದತ್ತಾಂಶ_device, size_t expert_ಬೈಟುಗಳು, int64_t n_expert, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(expert_ಬೈಟುಗಳು % ಗಾತ್ರ(ಖಂಡ_q8_0) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಖಂಡಗಳು_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) (expert_ಬೈಟುಗಳು / ಗಾತ್ರ(ಖಂಡ_q8_0));
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಬೈಟುಗಳು       = expert_ಬೈಟುಗಳು * (size_t) n_expert;

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, ಒಟ್ಟು_ಬೈಟುಗಳು);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ಒಟ್ಟು_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, ಒಟ್ಟು_ಬೈಟುಗಳು)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಖಂಡಗಳು = ಖಂಡಗಳು_ಪ್ರತಿ_expert * (ಪೂರ್ಣಾಂಕ) n_expert;
    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(ಒಟ್ಟು_ಖಂಡಗಳು, [=](ಸ್ವಯಂಚಾಲಿತ gb_) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          gb   = gb_;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          e    = gb / ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib   = gb % ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (tmp_buf + (size_t) e * expert_ಬೈಟುಗಳು);
        uint8_t *          ತಳ = ದತ್ತಾಂಶ_device + (size_t) e * expert_ಬೈಟುಗಳು;

        ಸ್ವಯಂಚಾಲಿತ * qs_ptr = ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * d_ptr  = (sycl::half *) (qs_ptr + (size_t) ಖಂಡಗಳು_ಪ್ರತಿ_expert * QK8_0);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q4_k(uint8_t * ದತ್ತಾಂಶ_device, size_t size, size_t ಸ್ಥಾನಾಂತರ, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_q4_K) == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q4_K) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_q4_K);

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ * qs_ptr     = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    ಸ್ವಯಂಚಾಲಿತ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ಗಾತ್ರ * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_SCALE_ಗಾತ್ರ; ++j) {
            scales_ptr[ib * K_SCALE_ಗಾತ್ರ + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Reorder each expert slice into a self-contained SoA layout.
ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q4_k_moe(uint8_t * ದತ್ತಾಂಶ_device, size_t expert_ಬೈಟುಗಳು, int64_t n_expert, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(expert_ಬೈಟುಗಳು % ಗಾತ್ರ(ಖಂಡ_q4_K) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಖಂಡಗಳು_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) (expert_ಬೈಟುಗಳು / ಗಾತ್ರ(ಖಂಡ_q4_K));
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಬೈಟುಗಳು       = expert_ಬೈಟುಗಳು * (size_t) n_expert;

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, ಒಟ್ಟು_ಬೈಟುಗಳು);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ಒಟ್ಟು_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, ಒಟ್ಟು_ಬೈಟುಗಳು)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಖಂಡಗಳು = ಖಂಡಗಳು_ಪ್ರತಿ_expert * (ಪೂರ್ಣಾಂಕ) n_expert;
    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(ಒಟ್ಟು_ಖಂಡಗಳು, [=](ಸ್ವಯಂಚಾಲಿತ gb_) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          gb   = gb_;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          e    = gb / ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib   = gb % ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) (tmp_buf + (size_t) e * expert_ಬೈಟುಗಳು);
        uint8_t *          ತಳ = ದತ್ತಾಂಶ_device + (size_t) e * expert_ಬೈಟುಗಳು;

        ಸ್ವಯಂಚಾಲಿತ * qs_ptr     = ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * scales_ptr = qs_ptr + QK_K / 2 * ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ವಯಂಚಾಲಿತ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ಗಾತ್ರ * ಖಂಡಗಳು_ಪ್ರತಿ_expert);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_SCALE_ಗಾತ್ರ; ++j) {
            scales_ptr[ib * K_SCALE_ಗಾತ್ರ + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q5_k_moe(uint8_t * ದತ್ತಾಂಶ_device, size_t expert_ಬೈಟುಗಳು, int64_t n_expert, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(expert_ಬೈಟುಗಳು % ಗಾತ್ರ(ಖಂಡ_q5_K) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಖಂಡಗಳು_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) (expert_ಬೈಟುಗಳು / ಗಾತ್ರ(ಖಂಡ_q5_K));
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಬೈಟುಗಳು       = expert_ಬೈಟುಗಳು * (size_t) n_expert;

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, ಒಟ್ಟು_ಬೈಟುಗಳು);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ಒಟ್ಟು_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, ಒಟ್ಟು_ಬೈಟುಗಳು)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಖಂಡಗಳು = ಖಂಡಗಳು_ಪ್ರತಿ_expert * (ಪೂರ್ಣಾಂಕ) n_expert;
    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(ಒಟ್ಟು_ಖಂಡಗಳು, [=](ಸ್ವಯಂಚಾಲಿತ gb_) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          gb   = gb_;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          e    = gb / ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib   = gb % ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) (tmp_buf + (size_t) e * expert_ಬೈಟುಗಳು);
        uint8_t *          ತಳ = ದತ್ತಾಂಶ_device + (size_t) e * expert_ಬೈಟುಗಳು;

        ಸ್ವಯಂಚಾಲಿತ * qs_ptr     = ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * qh_ptr     = qs_ptr + (QK_K / 2) * ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ವಯಂಚಾಲಿತ * scales_ptr = qh_ptr + (QK_K / 8) * ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ವಯಂಚಾಲಿತ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ಗಾತ್ರ * ಖಂಡಗಳು_ಪ್ರತಿ_expert);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_SCALE_ಗಾತ್ರ; ++j) {
            scales_ptr[ib * K_SCALE_ಗಾತ್ರ + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q6_k_moe(uint8_t * ದತ್ತಾಂಶ_device, size_t expert_ಬೈಟುಗಳು, int64_t n_expert, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(expert_ಬೈಟುಗಳು % ಗಾತ್ರ(ಖಂಡ_q6_K) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ    ಖಂಡಗಳು_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) (expert_ಬೈಟುಗಳು / ಗಾತ್ರ(ಖಂಡ_q6_K));
    ಸ್ಥಿರಾಂಕ size_t ಒಟ್ಟು_ಬೈಟುಗಳು       = expert_ಬೈಟುಗಳು * (size_t) n_expert;

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, ಒಟ್ಟು_ಬೈಟುಗಳು);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ಒಟ್ಟು_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, ಒಟ್ಟು_ಬೈಟುಗಳು)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಖಂಡಗಳು = ಖಂಡಗಳು_ಪ್ರತಿ_expert * (ಪೂರ್ಣಾಂಕ) n_expert;
    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(ಒಟ್ಟು_ಖಂಡಗಳು, [=](ಸ್ವಯಂಚಾಲಿತ gb_) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          gb   = gb_;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          e    = gb / ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib   = gb % ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) (tmp_buf + (size_t) e * expert_ಬೈಟುಗಳು);
        uint8_t *          ತಳ = ದತ್ತಾಂಶ_device + (size_t) e * expert_ಬೈಟುಗಳು;

        ಸ್ವಯಂಚಾಲಿತ * ql_ptr     = ತಳ;
        ಸ್ವಯಂಚಾಲಿತ * qh_ptr     = ql_ptr + (QK_K / 2) * ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ವಯಂಚಾಲಿತ * scales_ptr = qh_ptr + (QK_K / 4) * ಖಂಡಗಳು_ಪ್ರತಿ_expert;
        ಸ್ವಯಂಚಾಲಿತ * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * ಖಂಡಗಳು_ಪ್ರತಿ_expert);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_mxfp4_moe(uint8_t * ದತ್ತಾಂಶ_device, size_t expert_ಬೈಟುಗಳು, int64_t n_expert, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(expert_ಬೈಟುಗಳು % ಗಾತ್ರ(ಖಂಡ_mxfp4) == 0);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಖಂಡಗಳು_ಪ್ರತಿ_expert = (ಪೂರ್ಣಾಂಕ) (expert_ಬೈಟುಗಳು / ಗಾತ್ರ(ಖಂಡ_mxfp4));
    ಸ್ಥಿರಾಂಕ size_t  max_chunk_ಬೈಟುಗಳು   = 32u << 20;
    ಸ್ಥಿರಾಂಕ int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_ಬೈಟುಗಳು / expert_ಬೈಟುಗಳು)));

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, (size_t) chunk_experts * expert_ಬೈಟುಗಳು);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_ಬೈಟುಗಳು);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    ಗಾಗಿ (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ಸ್ಥಿರಾಂಕ int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = ದತ್ತಾಂಶ_device + (size_t) e0 * expert_ಬೈಟುಗಳು;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, chunk, (size_t) n_chunk * expert_ಬೈಟುಗಳು)));
        ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_ಖಂಡಗಳು = ಖಂಡಗಳು_ಪ್ರತಿ_expert * (ಪೂರ್ಣಾಂಕ) n_chunk;
        ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(ಒಟ್ಟು_ಖಂಡಗಳು, [=](ಸ್ವಯಂಚಾಲಿತ gb_) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           gb   = gb_;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           e    = gb / ಖಂಡಗಳು_ಪ್ರತಿ_expert;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ           ib   = gb % ಖಂಡಗಳು_ಪ್ರತಿ_expert;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * x    = (ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *) (tmp_buf + (size_t) e * expert_ಬೈಟುಗಳು);
            uint8_t *           ತಳ = chunk + (size_t) e * expert_ಬೈಟುಗಳು;

            uint8_t * qs_ptr = ತಳ;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) ಖಂಡಗಳು_ಪ್ರತಿ_expert;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q2_k(uint8_t * ದತ್ತಾಂಶ_device, size_t size, size_t ಸ್ಥಾನಾಂತರ, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_q2_K) == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q2_K) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_q2_K);

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ *        qs_ptr     = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * x  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q3_k(uint8_t * ದತ್ತಾಂಶ_device, size_t size, size_t ಸ್ಥಾನಾಂತರ, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_q3_K) == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q3_K) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_q3_K);

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ *       qs_ptr     = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * x  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q5_k(uint8_t * ದತ್ತಾಂಶ_device, size_t size, size_t ಸ್ಥಾನಾಂತರ, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_q5_K) == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q5_K) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_q5_K);

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ * qs_ptr     = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ಗಾತ್ರ * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < K_SCALE_ಗಾತ್ರ; ++j) {
            scales_ptr[ib * K_SCALE_ಗಾತ್ರ + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_q6_k(uint8_t * ದತ್ತಾಂಶ_device, size_t size, size_t ಸ್ಥಾನಾಂತರ, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_q6_K) == 0);
    GGML_ASSERT(ಸ್ಥಾನಾಂತರ % ಗಾತ್ರ(ಖಂಡ_q6_K) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_q6_K);

    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ *       ql_ptr     = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * x  = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          ib = i;

        ಸ್ಥಿರಾಂಕ uint8_t * ql              = x[ib].ql;
        ಸ್ಥಿರಾಂಕ uint8_t * qh              = x[ib].qh;
        uint8_t *       ತಳ_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       ತಳ_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       ತಳ_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 2; ++j) {
            ತಳ_ql_ptr[j] = ql[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 4; ++j) {
            ತಳ_qh_ptr[j] = qh[j];
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 16; ++j) {
            ತಳ_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_iq3_s(uint8_t * ದತ್ತಾಂಶ_device, size_t size, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_iq3_s) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_iq3_s);
    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ * qs_ptr       = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    ಸ್ವಯಂಚಾಲಿತ * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (ಗಾತ್ರ(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[ಗಾತ್ರ(ggml_half) + j] = x[ib].scales[j];
        }
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw_iq3_xxs(uint8_t * ದತ್ತಾಂಶ_device, size_t size, dpct::ಸರತಿ_ptr stream) {
    GGML_ASSERT(size % ಗಾತ್ರ(ಖಂಡ_iq3_xxs) == 0);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nblocks = size / ಗಾತ್ರ(ಖಂಡ_iq3_xxs);
    sycl_reorder_ತಾತ್ಕಾಲಿಕ_ಬಫರ್ tmp(stream, size);
    ಆದರೆ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ಸ್ಮರಣೆ_ನಕಲು(tmp_buf, ದತ್ತಾಂಶ_device, size)));
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ಸ್ವಯಂಚಾಲಿತ * qs_ptr = ದತ್ತಾಂಶ_device;
    ಸ್ವಯಂಚಾಲಿತ * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    ಸ್ವಯಂಚಾಲಿತ reorder_event = stream->parallel_for(nblocks, [=](ಸ್ವಯಂಚಾಲಿತ i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs *) tmp_buf;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ reorder_qw(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, dpct::ಸರತಿ_ptr stream) {
    uint8_t * ದತ್ತಾಂಶ_device = (uint8_t *) src0->ದತ್ತಾಂಶ;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    ಆದರೆ (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        ಆಯ್ಕೆ (src0->type) {
            ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                ಹಿಂತಿರುಗಿ reorder_qw_q8_0_moe(ದತ್ತಾಂಶ_device, src0->nb[2], src0->ne[2], stream);
            ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                ಹಿಂತಿರುಗಿ reorder_qw_q4_k_moe(ದತ್ತಾಂಶ_device, src0->nb[2], src0->ne[2], stream);
            ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                ಹಿಂತಿರುಗಿ reorder_qw_q5_k_moe(ದತ್ತಾಂಶ_device, src0->nb[2], src0->ne[2], stream);
            ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                ಹಿಂತಿರುಗಿ reorder_qw_q6_k_moe(ದತ್ತಾಂಶ_device, src0->nb[2], src0->ne[2], stream);
            ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                ಹಿಂತಿರುಗಿ reorder_qw_mxfp4_moe(ದತ್ತಾಂಶ_device, src0->nb[2], src0->ne[2], stream);
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಹಿಂತಿರುಗಿ reorder_qw_q4_0(ದತ್ತಾಂಶ_device, ncols, nrows, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ reorder_qw_q8_0(ದತ್ತಾಂಶ_device, ncols, nrows, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಹಿಂತಿರುಗಿ reorder_qw_q2_k(ದತ್ತಾಂಶ_device, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಹಿಂತಿರುಗಿ reorder_qw_q3_k(ದತ್ತಾಂಶ_device, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಹಿಂತಿರುಗಿ reorder_qw_q4_k(ದತ್ತಾಂಶ_device, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಹಿಂತಿರುಗಿ reorder_qw_q5_k(ದತ್ತಾಂಶ_device, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಹಿಂತಿರುಗಿ reorder_qw_q6_k(ದತ್ತಾಂಶ_device, size, 0, stream);
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            ಹಿಂತಿರುಗಿ reorder_qw_iq3_xxs(ದತ್ತಾಂಶ_device, size, stream);
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಹಿಂತಿರುಗಿ reorder_qw_iq3_s(ದತ್ತಾಂಶ_device, size, stream);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ should_reorder_tensor(ggml_backend_sycl_context& ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

ಸ್ಥಿರ ಶೂನ್ಯ opt_for_reorder(ggml_backend_sycl_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    ಆದರೆ (!should_reorder_tensor(*ctx, dst)) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ಆದರೆ (!extra || extra->optimized_feature.reorder) {
        ಹಿಂತಿರುಗಿ;  // Skip permutations and already reordered tensors
    }

    ಆಯ್ಕೆ (mm_algorithm) {
        ಸ್ಥಿತಿ mul_mat_algo::DMMV:
            ಆದರೆ (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                ಹಿಂತಿರುಗಿ;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ mul_mat_algo::MMVQ:
            ಆದರೆ (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                ಹಿಂತಿರುಗಿ;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ mul_mat_algo::MUL_MAT_SYCL:
            ಆದರೆ (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                ಹಿಂತಿರುಗಿ;
            }
            ನಿಲ್ಲಿಸು;
    }

    ಆದರೆ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ನಿಜ;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
ಸ್ಥಿರ ಶೂನ್ಯ opt_for_reorder_id(ggml_backend_sycl_context * ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0) {
    ಆದರೆ (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        ಹಿಂತಿರುಗಿ;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    ಆದರೆ (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->ದತ್ತಾಂಶ % 16 != 0)) {
        ಹಿಂತಿರುಗಿ;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ಆದರೆ (!extra || extra->optimized_feature.reorder) {
        ಹಿಂತಿರುಗಿ;
    }
    ಆದರೆ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ನಿಜ;
    }
}


ಸ್ಥಿರ ತಾರ್ಕಿಕ can_use_dequantize_mul_mat_vec(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ಸ್ಥಿರಾಂಕ int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    ಹಿಂತಿರುಗಿ ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ can_use_mul_mat_vec_q(ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    ಹಿಂತಿರುಗಿ ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_ಗಾತ್ರ;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0, ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    ಆದರೆ (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ವಿಭಜಿಸು = ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್);
    int64_t min_compute_capability = ಗರಿಷ್ಠ_ಪೂರ್ಣಾಂಕ;

    ಆದರೆ (ವಿಭಜಿಸು) {
        ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context * buft_ctx =
            (ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type_context *) src0->ಬಫರ್->buft->context;
        ಸ್ವಯಂಚಾಲಿತ & tensor_ವಿಭಜಿಸು = buft_ctx->tensor_ವಿಭಜಿಸು;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ id = 0; id < ggml_sycl_info().device_ಎಣಿಕೆ; ++id) {
            // skip devices that are not going to do any work:
            ಆದರೆ (tensor_ವಿಭಜಿಸು[id] >= (id + 1 < ggml_sycl_info().device_ಎಣಿಕೆ ? tensor_ವಿಭಜಿಸು[id + 1] : 1.0f)) {
                ಮುಂದುವರಿ;
            }

            ಆದರೆ (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    ತಾರ್ಕಿಕ use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#ಆದರೆ_ಅಂತ್ಯ

    ತಾರ್ಕಿಕ use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    ತಾರ್ಕಿಕ use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    ಆದರೆ (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        ತಾರ್ಕಿಕ use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        ಆದರೆ (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    ಆದರೆ (!ವಿಭಜಿಸು && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        ಆದರೆ (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ವಿಭಜಿಸು && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!ವಿಭಜಿಸು && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        ಆದರೆ (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * ಮೇಲೆ, ಸ್ಥಿರಾಂಕ ggml_tensor * wu,
                                             ಸ್ಥಿರಾಂಕ ggml_tensor * wg, ಸ್ಥಿರಾಂಕ ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_u = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wu->extra);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_g = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wg->extra);
    ಆದರೆ ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ plain_type = [](ggml_type t) { ಹಿಂತಿರುಗಿ t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    ಆದರೆ (!plain_type(wu->type) || !plain_type(wg->type)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ಮೇಲೆ, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ಸ್ಥಿರಾಂಕ int64_t ne00 = wu->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = act->ne[1];

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx.stream();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = GGML_PAD((ಪೂರ್ಣಾಂಕ) ne00, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);

    ggml_sycl_pool_alloc<ಅಕ್ಷರ> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಗಾತ್ರ(ಖಂಡ_q8_1) / QK8_1);
    ಅಕ್ಷರ * src1_ddq = src1_q8_alloc.get();

    quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_q8_1>((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) act->ದತ್ತಾಂಶ, src1_ddq, (ಪೂರ್ಣಾಂಕ) ne00, (ಪೂರ್ಣಾಂಕ) ne11,
                                          src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, stream);

    ಹಿಂತಿರುಗಿ ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->ದತ್ತಾಂಶ, wu->ದತ್ತಾಂಶ,
                                             src1_ddq, (ದಶಮಾಂಶ *) glu->ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ne00, (ಪೂರ್ಣಾಂಕ) wu->ne[1],
                                             (ಪೂರ್ಣಾಂಕ) ne11,
                                             /*stride_col_y=*/src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ / QK8_1,
                                             /*stride_col_dst=*/(ಪೂರ್ಣಾಂಕ) glu->ne[0], stream);
    GGML_ಬಳಸದ(gate);
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * ಮೇಲೆ, ಸ್ಥಿರಾಂಕ ggml_tensor * wu,
                                             ಸ್ಥಿರಾಂಕ ggml_tensor * wg, ಸ್ಥಿರಾಂಕ ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * act_begin = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) act->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * glu_begin = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) glu->ದತ್ತಾಂಶ;
    ಆದರೆ (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, ಮೇಲೆ, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_u = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wu->extra);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_g = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wg->extra);
    ಆದರೆ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ಮೇಲೆ, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    ಹಿಂತಿರುಗಿ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->ದತ್ತಾಂಶ, wg->ದತ್ತಾಂಶ,
                                                              (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) act->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) glu->ದತ್ತಾಂಶ,
                                                              (ಪೂರ್ಣಾಂಕ) wu->ne[0], (ಪೂರ್ಣಾಂಕ) wu->ne[1], ctx.stream());
}
#ಆದರೆ_ಅಂತ್ಯ

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಆದರೆ (!ggml_sycl_can_fuse(cgraph, ಗಂಟು_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ggml_tensor *       glu  = cgraph->nodes[ಗಂಟು_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       ಮೇಲೆ   = glu->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * wu   = ಮೇಲೆ->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * wg   = gate->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * act  = ಮೇಲೆ->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    ಆದರೆ (ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(wu->ಬಫರ್) || ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(wg->ಬಫರ್)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    ಆದರೆ (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, ಮೇಲೆ, wu, wg, act)) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }
#ಆದರೆ_ಅಂತ್ಯ

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    ಆದರೆ (g_ggml_sycl_prioritize_dmmv) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    ಆದರೆ (!reorder_pair) {
        ಹಿಂತಿರುಗಿ ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, ಮೇಲೆ, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    ಆದರೆ (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, ಮೇಲೆ, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_u = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wu->extra);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ * extra_g = static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(wg->extra);
    ಆದರೆ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ಮೇಲೆ, /*num_src=*/2, " : fused with gate + GLU");

    ಸ್ಥಿರಾಂಕ int64_t ne00 = wu->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = act->ne[1];

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream           = ctx.stream();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = GGML_PAD((ಪೂರ್ಣಾಂಕ) ne00, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<ಅಕ್ಷರ> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಗಾತ್ರ(ಖಂಡ_q8_1) / QK8_1);
    ಅಕ್ಷರ *                     src1_ddq = src1_q8_alloc.get();

    quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) act->ದತ್ತಾಂಶ, src1_ddq, (ಪೂರ್ಣಾಂಕ) ne00, (ಪೂರ್ಣಾಂಕ) ne11,
                                                          src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, stream);

    ಹಿಂತಿರುಗಿ ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->ದತ್ತಾಂಶ, wg->ದತ್ತಾಂಶ, src1_ddq,
                                               (ದಶಮಾಂಶ *) glu->ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ne00, (ಪೂರ್ಣಾಂಕ) wu->ne[1], (ಪೂರ್ಣಾಂಕ) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * (ಪೂರ್ಣಾಂಕ) ಗಾತ್ರ(ಖಂಡ_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(ಪೂರ್ಣಾಂಕ) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಗಂಟು = cgraph->nodes[ಗಂಟು_idx];
    ಆದರೆ (ggml_sycl_info().device_ಎಣಿಕೆ != 1 || ಗಂಟು->type != GGML_TYPE_F32 ||
        ಗಂಟು->src[0]->type != GGML_TYPE_F32 || ಗಂಟು->src[0]->ne[0] >= 1024) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    ಪೂರ್ಣಾಂಕ           count = 0;
    ಪೂರ್ಣಾಂಕ           ಕೊನೆಯ  = ಗಂಟು_idx;
    ದಶಮಾಂಶ         eps0;
    ಸ್ಮರಣೆ_ನಕಲು(&eps0, ಗಂಟು->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    ಸ್ವಯಂಚಾಲಿತ overlaps = [](ಸ್ಥಿರಾಂಕ ggml_tensor * a, ಸ್ಥಿರಾಂಕ ggml_tensor * b) {
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ab = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) a->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * bb = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) b->ದತ್ತಾಂಶ;
        ಹಿಂತಿರುಗಿ ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಗಂಟು_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        ಆದರೆ (ggml_is_ಖಾಲಿಗಳು(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;  // not a launch; cannot break a run of adjacent norms
        }
        ಆದರೆ (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ಗಂಟು) ||
            !ggml_are_same_shape(nj->src[0], ಗಂಟು->src[0])) {
            ನಿಲ್ಲಿಸು;  // any other launch ends the run
        }
        ತಾರ್ಕಿಕ same_nb = ನಿಜ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < GGML_MAX_DIMS; ++d) {
            ಆದರೆ (nj->nb[d] != ಗಂಟು->nb[d] || nj->src[0]->nb[d] != ಗಂಟು->src[0]->nb[d]) {
                same_nb = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (!same_nb) {
            ನಿಲ್ಲಿಸು;  // one nb[] stride set is shared by the whole batch
        }
        ದಶಮಾಂಶ epsj;
        ಸ್ಮರಣೆ_ನಕಲು(&epsj, nj->op_params, ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಆದರೆ (epsj != eps0) {
            ನಿಲ್ಲಿಸು;  // eps mismatch ends the run
        }
        ತಾರ್ಕಿಕ indep = ನಿಜ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < count; ++k) {
            ಆದರೆ (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = ಸುಳ್ಳು;
                ನಿಲ್ಲಿಸು;
            }
        }
        ಆದರೆ (!indep) {
            ನಿಲ್ಲಿಸು;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        ಕೊನೆಯ           = j;
    }
    ಆದರೆ (count < 2) {
        ಹಿಂತಿರುಗಿ 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    ಹಿಂತಿರುಗಿ ಕೊನೆಯ - ಗಂಟು_idx;
}


__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ k_copy_src1_ಗಮ್ಯ_contiguous(
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *__restrict__ src1_original, ಅಕ್ಷರ *__restrict__ src1_contiguous,
    ಸ್ಥಿರಾಂಕ mmid_ಅಡ್ಡಸಾಲು_mapping *__restrict__ ಅಡ್ಡಸಾಲು_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int32_t src1_ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ int32_t iid1 = ಅಡ್ಡಸಾಲು_mapping[src1_ಅಡ್ಡಸಾಲು].i2;
    ಸ್ಥಿರಾಂಕ int32_t id   = ಅಡ್ಡಸಾಲು_mapping[src1_ಅಡ್ಡಸಾಲು].i1;

    ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
    ಸ್ಥಿರಾಂಕ int64_t i12 = iid1;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_ಅಡ್ಡಸಾಲು_original = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(src1_original + i11*nb11 + i12*nb12);
    ದಶಮಾಂಶ * src1_ಅಡ್ಡಸಾಲು_contiguous = (ದಶಮಾಂಶ *)(src1_contiguous + src1_ಅಡ್ಡಸಾಲು*nb11);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = ಅಂಶ_ct1.get_local_id(2); i < ne10;
         i += ಅಂಶ_ct1.get_local_range(2)) {
        src1_ಅಡ್ಡಸಾಲು_contiguous[i] = src1_ಅಡ್ಡಸಾಲು_original[i];
    }
}

__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ k_copy_dst_from_contiguous(
    ಅಕ್ಷರ *__restrict__ dst_original, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *__restrict__ dst_contiguous,
    ಸ್ಥಿರಾಂಕ mmid_ಅಡ್ಡಸಾಲು_mapping *__restrict__ ಅಡ್ಡಸಾಲು_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    int32_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    ಸ್ಥಿರಾಂಕ int32_t i1 = ಅಡ್ಡಸಾಲು_mapping[i].i1;
    ಸ್ಥಿರಾಂಕ int32_t i2 = ಅಡ್ಡಸಾಲು_mapping[i].i2;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು_contiguous = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)(dst_contiguous + i*nb1);
    ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು_original = (ದಶಮಾಂಶ *)(dst_original + i1*nb1 + i2*nb2);

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಅಂಶ_ct1.get_local_id(2); j < ne0;
         j += ಅಂಶ_ct1.get_local_range(2)) {
        dst_ಅಡ್ಡಸಾಲು_original[j] = dst_ಅಡ್ಡಸಾಲು_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ಸ್ಥಿರಾಂಕ ggml_tensor * ids, ggml_tensor * dst)
{
    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];
    ಆದರೆ (ne12 != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (!ggml_is_contiguous(src1)) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಸ್ಥಿರಾಂಕ int64_t n_ids_ಪ್ರತಿ_ಗುಂಪು = ids->ne[0];
    ಆದರೆ (ids->ne[1] != 1) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    ಆದರೆ (ne11 != 1 && ne11 != n_ids_ಪ್ರತಿ_ಗುಂಪು) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream           = ctx.stream();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = GGML_PAD((ಪೂರ್ಣಾಂಕ) ne10, ಮಾತೃಕೆ_ಅಡ್ಡಸಾಲು_PADDING);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       n_experts_ಬಳಸಿದ   = (ಪೂರ್ಣಾಂಕ) n_ids_ಪ್ರತಿ_ಗುಂಪು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       nrows            = (ಪೂರ್ಣಾಂಕ) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu * src0_extra =
        static_cast<ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *>(src0->extra);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<ಅಕ್ಷರ> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಗಾತ್ರ(ಖಂಡ_q8_1) / QK8_1);
    ಅಕ್ಷರ * src1_ddq = src1_q8_alloc.get();
    ಆದರೆ (use_reorder) {
        quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ, src1_ddq, (ಪೂರ್ಣಾಂಕ) ne10, (ಪೂರ್ಣಾಂಕ) ne11,
            src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        quantize_ಅಡ್ಡಸಾಲು_q8_1_sycl<quantize_q8_1>(
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ, src1_ddq, (ಪೂರ್ಣಾಂಕ) ne10, (ಪೂರ್ಣಾಂಕ) ne11,
            src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ, stream);
    }

    ಸ್ಥಿರಾಂಕ size_t ಬೈಟುಗಳು_ಪ್ರತಿ_qrow = (size_t) src1_padded_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ * ಗಾತ್ರ(ಖಂಡ_q8_1) / QK8_1;
    ಸ್ಥಿರಾಂಕ size_t src1_ಅಡ್ಡಸಾಲು_stride = (ne11 == 1) ? 0 : ಬೈಟುಗಳು_ಪ್ರತಿ_qrow;

    ಆದರೆ (use_reorder) {
        ಹಿಂತಿರುಗಿ ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->ದತ್ತಾಂಶ, src1_ddq, (ಸ್ಥಿರಾಂಕ int32_t *) ids->ದತ್ತಾಂಶ,
            (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ne10, nrows, n_experts_ಬಳಸಿದ,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_ಅಡ್ಡಸಾಲು_stride, stream);
    }
    ಹಿಂತಿರುಗಿ ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->ದತ್ತಾಂಶ, src1_ddq, (ಸ್ಥಿರಾಂಕ int32_t *) ids->ದತ್ತಾಂಶ,
        (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ, (ಪೂರ್ಣಾಂಕ) ne10, nrows, n_experts_ಬಳಸಿದ,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_ಅಡ್ಡಸಾಲು_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
ಸ್ಥಿರ ಶೂನ್ಯ mmid_counting_ವಿಂಗಡಿಸು_ಅಡ್ಡಸಾಲುಗಳು(
        ಸ್ಥಿರಾಂಕ ggml_tensor * ids, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_ಅಡ್ಡಸಾಲುಗಳು,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_ಅಡ್ಡಸಾಲು_offsets,
        std::vector<mmid_ಅಡ್ಡಸಾಲು_mapping> & routed_ಅಡ್ಡಸಾಲು_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.ನಿಯೋಜಿಸು(n_as, 0);
    ಗಾಗಿ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ಗಾಗಿ (int64_t id = 0; id < n_ids; id++) {
            ಸ್ಥಿರಾಂಕ int32_t ಅಡ್ಡಸಾಲು_id_i = *(ಸ್ಥಿರಾಂಕ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(ಅಡ್ಡಸಾಲು_id_i >= 0 && ಅಡ್ಡಸಾಲು_id_i < n_as);
            expert_counts[ಅಡ್ಡಸಾಲು_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_ಅಡ್ಡಸಾಲು_offsets.ನಿಯೋಜಿಸು(n_as + 1, 0);
    ಗಾಗಿ (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_ಅಡ್ಡಸಾಲು_offsets[i02 + 1] = expert_ಅಡ್ಡಸಾಲು_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_ಅಡ್ಡಸಾಲು_ಮುಂದಿನ = expert_ಅಡ್ಡಸಾಲು_offsets;
    routed_ಅಡ್ಡಸಾಲು_src.resize(n_routed_ಅಡ್ಡಸಾಲುಗಳು);
    ಗಾಗಿ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ಗಾಗಿ (int64_t id = 0; id < n_ids; id++) {
            ಸ್ಥಿರಾಂಕ int32_t ಅಡ್ಡಸಾಲು_id_i = *(ಸ್ಥಿರಾಂಕ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(ಅಡ್ಡಸಾಲು_id_i >= 0 && ಅಡ್ಡಸಾಲು_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ಸ್ಥಿರಾಂಕ int64_t routed_ಅಡ್ಡಸಾಲು = expert_ಅಡ್ಡಸಾಲು_ಮುಂದಿನ[ಅಡ್ಡಸಾಲು_id_i]++;
            GGML_ASSERT(routed_ಅಡ್ಡಸಾಲು >= expert_ಅಡ್ಡಸಾಲು_offsets[ಅಡ್ಡಸಾಲು_id_i]);
            GGML_ASSERT(routed_ಅಡ್ಡಸಾಲು < expert_ಅಡ್ಡಸಾಲು_offsets[ಅಡ್ಡಸಾಲು_id_i + 1]);
            routed_ಅಡ್ಡಸಾಲು_src[routed_ಅಡ್ಡಸಾಲು] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) ಪ್ರಯತ್ನಿಸು {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(src0->ಬಫರ್) && "mul_mat_id does not support split buffers");

    ಸ್ಥಿರಾಂಕ ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int64_t n_as = ne02;
    ಸ್ಥಿರಾಂಕ int64_t n_ids = ids->ne[0];

    ಆದರೆ (ne12 == 1) {
        ಆದರೆ (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            ಹಿಂತಿರುಗಿ;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    ಆದರೆ (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<ಅಕ್ಷರ> ids_host(ggml_nbytes(ids));
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ids_dev = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ids->ದತ್ತಾಂಶ;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->ಸ್ಮರಣೆ_ನಕಲು(ids_host.ದತ್ತಾಂಶ(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_ಅಡ್ಡಸಾಲು = *src0;
    ggml_tensor src1_ಅಡ್ಡಸಾಲು = *src1;
    ggml_tensor dst_ಅಡ್ಡಸಾಲು = *dst;

    ಅಕ್ಷರ *src0_original = (ಅಕ್ಷರ *)src0->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *src1_original = (ಅಕ್ಷರ *)src1->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *dst_original = (ಅಕ್ಷರ *)dst->ದತ್ತಾಂಶ;

    src0_ಅಡ್ಡಸಾಲು.ne[2] = 1;
    src0_ಅಡ್ಡಸಾಲು.ne[3] = 1;
    src0_ಅಡ್ಡಸಾಲು.nb[3] = nb02;

    src1_ಅಡ್ಡಸಾಲು.ne[1] = 1;
    src1_ಅಡ್ಡಸಾಲು.ne[2] = 1;
    src1_ಅಡ್ಡಸಾಲು.ne[3] = 1;
    src1_ಅಡ್ಡಸಾಲು.nb[2] = nb11;
    src1_ಅಡ್ಡಸಾಲು.nb[3] = nb11;

    dst_ಅಡ್ಡಸಾಲು.ne[1] = 1;
    dst_ಅಡ್ಡಸಾಲು.ne[2] = 1;
    dst_ಅಡ್ಡಸಾಲು.ne[3] = 1;
    dst_ಅಡ್ಡಸಾಲು.nb[2] = nb1;
    dst_ಅಡ್ಡಸಾಲು.nb[3] = nb1;
    ಆದರೆ (ne12 == 1) {
        ಗಾಗಿ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            ಗಾಗಿ (int64_t id = 0; id < n_ids; id++) {
                ಸ್ಥಿರಾಂಕ int32_t i02 = *(ಸ್ಥಿರಾಂಕ int32_t *) (ids_host.ದತ್ತಾಂಶ() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ಸ್ಥಿರಾಂಕ int64_t i11 = id % ne11;
                ಸ್ಥಿರಾಂಕ int64_t i12 = iid1;

                ಸ್ಥಿರಾಂಕ int64_t i1 = id;
                ಸ್ಥಿರಾಂಕ int64_t i2 = i12;

            src0_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = src0_original + i02*nb02;
            src1_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = src1_original + i11*nb11 + i12*nb12;
            dst_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_ಅಡ್ಡಸಾಲು, &src1_ಅಡ್ಡಸಾಲು, &dst_ಅಡ್ಡಸಾಲು);
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ int64_t n_routed_ಅಡ್ಡಸಾಲುಗಳು = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<ಅಕ್ಷರ> src1_contiguous(ctx.pool(), ಗಾತ್ರ(ದಶಮಾಂಶ)*n_routed_ಅಡ್ಡಸಾಲುಗಳು*ne10);
        ggml_sycl_pool_alloc<ಅಕ್ಷರ>  dst_contiguous(ctx.pool(), ಗಾತ್ರ(ದಶಮಾಂಶ)*n_routed_ಅಡ್ಡಸಾಲುಗಳು*ne0);

        src1_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = src1_contiguous.get();
        dst_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_ಅಡ್ಡಸಾಲು_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_ಅಡ್ಡಸಾಲು_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_ಅಡ್ಡಸಾಲು_mapping> & routed_ಅಡ್ಡಸಾಲು_src = ctx.mmid_ಅಡ್ಡಸಾಲು_mapping_host;

        mmid_counting_ವಿಂಗಡಿಸು_ಅಡ್ಡಸಾಲುಗಳು(ids, ids_host.ದತ್ತಾಂಶ(), n_ids, n_as, n_routed_ಅಡ್ಡಸಾಲುಗಳು,
                                expert_ಅಡ್ಡಸಾಲು_counts, expert_ಅಡ್ಡಸಾಲು_offsets, routed_ಅಡ್ಡಸಾಲು_src);

        ggml_sycl_pool_alloc<mmid_ಅಡ್ಡಸಾಲು_mapping> dev_ಅಡ್ಡಸಾಲು_mapping(ctx.pool(), n_routed_ಅಡ್ಡಸಾಲುಗಳು);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->ಸ್ಮರಣೆ_ನಕಲು(dev_ಅಡ್ಡಸಾಲು_mapping.get(), routed_ಅಡ್ಡಸಾಲು_src.ದತ್ತಾಂಶ(), n_routed_ಅಡ್ಡಸಾಲುಗಳು*ಗಾತ್ರ(mmid_ಅಡ್ಡಸಾಲು_mapping))));

        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ max_work_ಗುಂಪು_ಗಾತ್ರ = ggml_sycl_info().max_work_ಗುಂಪು_sizes[ctx.device];
        ಪ್ರತಿಜ್ಞಾ(max_work_ಗುಂಪು_ಗಾತ್ರ % (WARP_ಗಾತ್ರ * WARP_ಗಾತ್ರ) == 0);

        {
            sycl::range<3> ಖಂಡ_dims(1, 1, std::min((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ne10, max_work_ಗುಂಪು_ಗಾತ್ರ));
            sycl::range<3> grid_dims(1, 1, n_routed_ಅಡ್ಡಸಾಲುಗಳು);
            stream->submit([&](sycl::handler &cgh) {
                ಅಕ್ಷರ *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_ಅಡ್ಡಸಾಲು_mapping *__restrict dev_ಅಡ್ಡಸಾಲು_mapping_get =
                    dev_ಅಡ್ಡಸಾಲು_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                    [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                        k_copy_src1_ಗಮ್ಯ_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_ಅಡ್ಡಸಾಲು_mapping_get,
                            ne11, ne10, nb11, nb12,
                            ಅಂಶ_ct1);
                    });
            });
        }

        ತಾರ್ಕಿಕ grouped = ಸುಳ್ಳು;
        ಆದರೆ (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == ಗಾತ್ರ(ದಶಮಾಂಶ)*ne10 && nb1 == ಗಾತ್ರ(ದಶಮಾಂಶ)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            ಆದರೆ (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_reordered =
                src0->extra && ((ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (ದಶಮಾಂಶ *) dst_contiguous.get(),
                                                     expert_ಅಡ್ಡಸಾಲು_offsets.ದತ್ತಾಂಶ(), n_as, ne01, ne10, n_routed_ಅಡ್ಡಸಾಲುಗಳು,
                                                     ctx.mmid_tile_ವೇಳಾಪಟ್ಟಿ_host, ctx.pool(), stream);
        }

        ಗಾಗಿ (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು = expert_ಅಡ್ಡಸಾಲು_counts[i02];

            ಆದರೆ (ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು == 0) {
                ಮುಂದುವರಿ;
            }

            ಸ್ಥಿರಾಂಕ int64_t expert_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ = expert_ಅಡ್ಡಸಾಲು_offsets[i02];

            src0_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == ಗಾತ್ರ(ದಶಮಾಂಶ)*ne10);
            GGML_ASSERT(nb1 == ಗಾತ್ರ(ದಶಮಾಂಶ)*ne0);
            src1_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = src1_contiguous.get() + expert_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ*nb11;
            src1_ಅಡ್ಡಸಾಲು.ne[1] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು;

            src1_ಅಡ್ಡಸಾಲು.nb[1] = nb11;
            src1_ಅಡ್ಡಸಾಲು.nb[2] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು*nb11;
            src1_ಅಡ್ಡಸಾಲು.nb[3] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು*nb11;

            dst_ಅಡ್ಡಸಾಲು.ದತ್ತಾಂಶ = dst_contiguous.get() + expert_ಅಡ್ಡಸಾಲು_ಸ್ಥಾನಾಂತರ*nb1;
            dst_ಅಡ್ಡಸಾಲು.ne[1] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು;
            dst_ಅಡ್ಡಸಾಲು.nb[1] = nb1;
            dst_ಅಡ್ಡಸಾಲು.nb[2] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು*nb1;
            dst_ಅಡ್ಡಸಾಲು.nb[3] = ಅಂಕೆ_src1_ಅಡ್ಡಸಾಲುಗಳು*nb1;

            ggml_sycl_mul_mat(ctx, &src0_ಅಡ್ಡಸಾಲು, &src1_ಅಡ್ಡಸಾಲು, &dst_ಅಡ್ಡಸಾಲು);
        }

        {
            sycl::range<3> ಖಂಡ_dims(1, 1, std::min((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ne0, max_work_ಗುಂಪು_ಗಾತ್ರ));
            sycl::range<3> grid_dims(1, 1, n_routed_ಅಡ್ಡಸಾಲುಗಳು);
            stream->submit([&](sycl::handler &cgh) {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ಸ್ಥಿರಾಂಕ mmid_ಅಡ್ಡಸಾಲು_mapping *__restrict dev_ಅಡ್ಡಸಾಲು_mapping_get =
                    dev_ಅಡ್ಡಸಾಲು_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ಖಂಡ_dims, ಖಂಡ_dims),
                    [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_ಅಡ್ಡಸಾಲು_mapping_get,
                                                   ne0, nb1, nb2, ಅಂಶ_ct1);
                    });
            });
        }
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_diag_ಮುಖವಾಡ_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_ಮುಖವಾಡ_inf(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಮೊತ್ತ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ಮೊತ್ತ(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಸರಾಸರಿ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ಸರಾಸರಿ(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_ಗಣ_main_device(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ main_device) ಪ್ರಯತ್ನಿಸು {
    ಆದರೆ (dpct::get_current_device_id() == static_cast<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ> (main_device)) {
        ಹಿಂತಿರುಗಿ;
    }
    check_allow_gpu_ಸೂಚ್ಯಂಕ(main_device);
    dpct::select_device(main_device);

    ಆದರೆ (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ರಚನೆ ggml_tensor * dst) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    ಆದರೆ (!g_sycl_loaded) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;

    ಆದರೆ (dst->src[0] != ಶೂನ್ಯಸೂಚಕ && ggml_backend_ಬಫರ್_is_sycl_ವಿಭಜಿಸು(dst->src[0]->ಬಫರ್)) {
        ggml_sycl_ಗಣ_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    ಆಯ್ಕೆ (dst->op) {
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
            ggml_sycl_repeat_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_sycl_get_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
            ggml_sycl_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            ggml_sycl_op_ಗಣ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_sycl_op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_ಕೂಡಿಸು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
            ggml_sycl_ಕೂಡಿಸು_id(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
            ggml_sycl_ಎಣಿಕೆ_equal(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(dst)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                    ggml_sycl_ಹೆಜ್ಜೆ(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                    ggml_sycl_gelu_ಶೀಘ್ರ(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                    ggml_sycl_ಸುತ್ತು(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(dst)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                    ggml_sycl_geglu_ಶೀಘ್ರ(ctx, dst);
                    ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            ggml_sycl_ಗುಂಪು_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
            ggml_sycl_rms_norm_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಆದರೆ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            ಆದರೆ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ggml_sycl_op_ಹೊರಹೋಗು_prod(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            ggml_sycl_diag_ಮುಖವಾಡ_inf(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ:
            ggml_sycl_op_soft_max_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
            ggml_sycl_cross_entropy_loss_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
            ggml_sycl_rope_ಹಿಂಭಾಗ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            ggml_sycl_im2col(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
            ggml_sycl_im2col_3d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
            ggml_sycl_ಮೊತ್ತ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            ggml_sycl_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ggml_sycl_ಸರಾಸರಿ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K:
            ggml_sycl_op_ತುದಿ_k(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
            ggml_sycl_op_gated_ರೇಖೀಯ_attn(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
            ggml_sycl_opt_ಹೆಜ್ಜೆ_adamw(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            ggml_sycl_opt_ಹೆಜ್ಜೆ_sgd(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ggml_sycl_ssm_ಪರಿಶೀಲನೆ(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
} ಹಿಡಿ (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

GGML_API ಶೂನ್ಯ ggml_backend_sycl_get_device_description(ಪೂರ್ಣಾಂಕ device, ಅಕ್ಷರ *description,
                                      size_t description_ಗಾತ್ರ) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_ಗಾತ್ರ, "%s", prop.get_name());
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ sycl_get_mem_info(ಪೂರ್ಣಾಂಕ device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    ಸ್ಮರಣೆ_APIType mem_api_type = ಸ್ಮರಣೆ_APIType::ಸ್ಮರಣೆ_API_TYPE_SYCL;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_ಫಲಿತಾಂಶ_SUCCESS ?
        (ಸ್ಮರಣೆ_APIType) g_ggml_sycl_get_mem_api : ಸ್ಮರಣೆ_APIType::ಸ್ಮರಣೆ_API_TYPE_SYCL;
#ಇಲ್ಲದಿದ್ದರೆ
    mem_api_type = ಸ್ಮರಣೆ_APIType::ಸ್ಮರಣೆ_API_TYPE_SYCL;
#ಆದರೆ_ಅಂತ್ಯ
    ತಾರ್ಕಿಕ res = get_ಸ್ಮರಣೆ_ಗಾತ್ರ(dpct::dev_mgr::instance().get_device(device),
        *ಬಿಡುಗಡೆ, *ಒಟ್ಟು, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *ಒಟ್ಟು, *ಬಿಡುಗಡೆ);
    ಹಿಂತಿರುಗಿ res;
}

ಶೂನ್ಯ ggml_backend_sycl_get_device_ಸ್ಮರಣೆ(ಪೂರ್ಣಾಂಕ device, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ಆದರೆ (!sycl_get_mem_info(device, ಬಿಡುಗಡೆ, ಒಟ್ಟು)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *ಬಿಡುಗಡೆ, *ಒಟ್ಟು);
} ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ಹಿಂತಿರುಗಿ sycl_ctx->name.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ಅಳಿಸು sycl_ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ಗಣ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                               size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_ಬಫರ್_type(sycl_ctx->device) && "unsupported buffer type");
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->ಸ್ಮರಣೆ_ನಕಲು((ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, ದತ್ತಾಂಶ, size)));
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor *tensor,
                                               ಶೂನ್ಯ *ದತ್ತಾಂಶ, size_t ಸ್ಥಾನಾಂತರ,
                                               size_t size) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ಸ್ಥಾನಾಂತರ);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_ಬಫರ್_t buf = tensor->view_src ? tensor->view_src->ಬಫರ್ : tensor->ಬಫರ್;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_ಬಫರ್_type(sycl_ctx->device) && "unsupported buffer type");
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->ಸ್ಮರಣೆ_ನಕಲು(
        ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)tensor->ದತ್ತಾಂಶ + ಸ್ಥಾನಾಂತರ, size)));
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ಸ್ಥಿರಾಂಕ ggml_tensor *src,
                                               ggml_tensor *dst) ಪ್ರಯತ್ನಿಸು {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ತಾರ್ಕಿಕ is_cpy_supported                = dst->ಬಫರ್->buft == ggml_backend_sycl_ಬಫರ್_type(sycl_ctx->device) &&
                            ggml_backend_ಬಫರ್_is_sycl(src->ಬಫರ್);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ಆದರೆ (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->ಸ್ಮರಣೆ_ನಕಲು(
            dst->ದತ್ತಾಂಶ, src->ದತ್ತಾಂಶ, ggml_nbytes(dst))));
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_synchronize(ggml_backend_t backend) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_ಬಳಸದ(backend);
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_sycl_is_view_or_noop(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ ggml_is_ಖಾಲಿಗಳು(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ggml_sycl_try_gdn_cache_fusion(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    ಆದರೆ (!g_ggml_sycl_enable_fusion) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * gdn = cgraph->nodes[ಗಂಟು_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    ಆದರೆ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * src_v     = gdn->src[2];
    ಸ್ಥಿರಾಂಕ int64_t       S_v       = src_v->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t       H         = src_v->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t       n_ಟೋಕನುಗಳು  = src_v->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t       n_seqs    = src_v->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t       D         = S_v * S_v * H;
    ಸ್ಥಿರಾಂಕ int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ಸ್ಥಿರಾಂಕ int64_t       n_written = std::min<int64_t>(n_ಟೋಕನುಗಳು, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ಸ್ಥಿರಾಂಕ size_t ಬಾಲ_off = ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, S_v * H * n_ಟೋಕನುಗಳು * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ಸ್ಥಿರಾಂಕ ggml_tensor * cpy  = ಶೂನ್ಯಸೂಚಕ;
    ಪೂರ್ಣಾಂಕ                 skip = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = ಗಂಟು_idx + 1; j < cgraph->n_nodes && cpy == ಶೂನ್ಯಸೂಚಕ; ++j) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * n = cgraph->nodes[j];
        ಆದರೆ (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ಹಿಂತಿರುಗಿ 0;
        }
        cpy  = n;
        skip = j - ಗಂಟು_idx;
    }
    ಆದರೆ (cpy == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ಸ್ಥಿರಾಂಕ ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ಆದರೆ (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != ಬಾಲ_off ||
        !ggml_is_contiguous(src)) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ಸ್ಥಿರಾಂಕ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ಆದರೆ (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_ಗಾತ್ರ(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(GGML_TYPE_F32, D)) {
        ಹಿಂತಿರುಗಿ 0;
    }

    fused_state_cpy.ದತ್ತಾಂಶ        = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ; // rollback group 0 (newest)
    fused_state_cpy.ಸ್ಥಾನ_stride = K > 1 ? (int64_t) (dst->nb[2] / ಗಾತ್ರ(ದಶಮಾಂಶ)) : 0;
    ಹಿಂತಿರುಗಿ skip;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_ನಕ್ಷೆ_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_ಗಣ_main_device(sycl_ctx->device);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];
        ಆದರೆ (ggml_sycl_is_view_or_noop(ಗಂಟು)) {
            ಮುಂದುವರಿ;
        }
        ಆದರೆ ((ಗಂಟು->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nodes_ಗಮ್ಯ_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        ಆದರೆ (nodes_ಗಮ್ಯ_skip != 0) {
            i += nodes_ಗಮ್ಯ_skip;
            ಮುಂದುವರಿ;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        ಪ್ರತಿಜ್ಞಾ(ಗಂಟು->ಬಫರ್->buft == ggml_backend_sycl_ಬಫರ್_type(sycl_ctx->device));
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < GGML_MAX_SRC; j++) {
            ಆದರೆ (ಗಂಟು->src[j] != ಶೂನ್ಯಸೂಚಕ) {
                ಪ್ರತಿಜ್ಞಾ(ಗಂಟು->src[j]->ಬಫರ್->buft == ggml_backend_sycl_ಬಫರ್_type(sycl_ctx->device));
            }
        }
#ಆದರೆ_ಅಂತ್ಯ
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        ಆದರೆ (ಗಂಟು->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ gdn_nodes_ಗಮ್ಯ_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            ಆದರೆ (gdn_nodes_ಗಮ್ಯ_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ಗಂಟು, fused_state_cpy);
                i += gdn_nodes_ಗಮ್ಯ_skip;
                ಮುಂದುವರಿ;
            }
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ಕೂಡಿಸು }, {})) {
            ggml_sycl_op_rms_norm_fused_ಕೂಡಿಸು(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;
            ಮುಂದುವರಿ;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        ಆದರೆ (ಗಂಟು->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_ಕೂಡಿಸು &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ಕೂಡಿಸು, GGML_OP_ಕೂಡಿಸು }, {})) {
            ggml_sycl_op_ಕೂಡಿಸು_ಕೂಡಿಸು_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (ಗಂಟು->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ಗಂಟು) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;
            ಮುಂದುವರಿ;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        ಆದರೆ (ಗಂಟು->op == GGML_OP_ಕೂಡಿಸು && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ಕೂಡಿಸು, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_ಕೂಡಿಸು_unary_mul_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ಮುಂದುವರಿ;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        ಆದರೆ (ಗಂಟು->op == GGML_OP_L2_NORM) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            ಆದರೆ (l2_batch_skip > 0) {
                i += l2_batch_skip;
                ಮುಂದುವರಿ;
            }
        }

        ಆದರೆ (ಗಂಟು->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ಕೂಡಿಸು, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ಗಂಟು, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (ಗಂಟು->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ಗಂಟು, ಶೂನ್ಯಸೂಚಕ, cgraph->nodes[i + 1]);
            i++;
            ಮುಂದುವರಿ;
        }

        ಆದರೆ (ಗಂಟು->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            ಮುಂದುವರಿ;
        }

        ತಾರ್ಕಿಕ ok = ggml_sycl_compute_forward(*sycl_ctx, ಗಂಟು);
        ಆದರೆ (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ಗಂಟು->name, ggml_op_name(ಗಂಟು->op));
        }
        GGML_ASSERT(ok);
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_ನಕ್ಷೆ
ಸ್ಥಿರ ತಾರ್ಕಿಕ check_ನಕ್ಷೆ_compatibility(ggml_cgraph * cgraph) {
    ಆದರೆ (ggml_sycl_info().device_ಎಣಿಕೆ > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ಸ್ಥಿರಾಂಕ ggml_op ಗಂಟು_op = cgraph->nodes[i]->op;
        ಆಯ್ಕೆ (ಗಂಟು_op) {
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ಗಂಟು_op));
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                ಆದರೆ (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ಗಂಟು_op));
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_sycl_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ಸ್ವಯಂಚಾಲಿತ * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_ನಕ್ಷೆ
    ತಾರ್ಕಿಕ use_sycl_ನಕ್ಷೆ = ಸುಳ್ಳು;
    ಆದರೆ (g_ggml_sycl_enable_ನಕ್ಷೆ) {
        use_sycl_ನಕ್ಷೆ = check_ನಕ್ಷೆ_compatibility(cgraph);
    }
    ಆದರೆ (use_sycl_ನಕ್ಷೆ) {
        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ನಕ್ಷೆ_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_ನಕ್ಷೆ);
        ಆದರೆ (!ನಕ್ಷೆ_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_ನಕ್ಷೆ_compute_impl(sycl_ctx, cgraph);
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
        }

        sycl_ex::command_ನಕ್ಷೆ model_sycl_ನಕ್ಷೆ(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_ಬಫರ್_outlives_ನಕ್ಷೆ{}});

        model_sycl_ನಕ್ಷೆ.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_ನಕ್ಷೆ_compute_impl(sycl_ctx, cgraph);
        model_sycl_ನಕ್ಷೆ.ಕೊನೆ_recording();

        ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ ನಕ್ಷೆ_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_ನಕ್ಷೆ);
        ಆದರೆ (!sycl_ctx->exec_ನಕ್ಷೆ || !ನಕ್ಷೆ_update_support) {
            ಸ್ವಯಂಚಾಲಿತ exec_ನಕ್ಷೆ = ನಕ್ಷೆ_update_support ? model_sycl_ನಕ್ಷೆ.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_ನಕ್ಷೆ.finalize();
            sycl_ctx->exec_ನಕ್ಷೆ = std::make_unique<
                sycl_ex::command_ನಕ್ಷೆ<sycl_ex::ನಕ್ಷೆ_state::executable>>(exec_ನಕ್ಷೆ);
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಪ್ರಯತ್ನಿಸು {
                sycl_ctx->exec_ನಕ್ಷೆ->update(model_sycl_ನಕ್ಷೆ);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                ಸ್ವಯಂಚಾಲಿತ exec_ನಕ್ಷೆ = model_sycl_ನಕ್ಷೆ.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_ನಕ್ಷೆ = std::make_unique<
                    sycl_ex::command_ನಕ್ಷೆ<sycl_ex::ನಕ್ಷೆ_state::executable>>(exec_ನಕ್ಷೆ);
            }
        }

        sycl_ctx->stream()->ext_oneapi_ನಕ್ಷೆ(*(sycl_ctx->exec_ನಕ್ಷೆ));
    } ಇಲ್ಲದಿದ್ದರೆ
#ಆದರೆ_ಅಂತ್ಯ
    {
        ggml_backend_sycl_ನಕ್ಷೆ_compute_impl(sycl_ctx, cgraph);
    }
    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_event_ದಾಖಲೆ(ggml_backend_t backend, ggml_backend_event_t event)
ಪ್ರಯತ್ನಿಸು
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) ಪ್ರಯತ್ನಿಸು {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    ಆದರೆ (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } ಇಲ್ಲದಿದ್ದರೆ
        GGML_ABORT("fatal error");
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
    std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_sycl_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ggml_backend_sycl_event_ದಾಖಲೆ,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ggml_guid_t ggml_backend_sycl_guid() {
    ಸ್ಥಿರ ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    ಹಿಂತಿರುಗಿ &guid;
}

ತಾರ್ಕಿಕ ggml_backend_is_sycl(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

ಪೂರ್ಣಾಂಕ ggml_backend_sycl_get_device_ಎಣಿಕೆ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ ggml_sycl_info().device_ಎಣಿಕೆ;
}


// backend device

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ctx->description.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ಆದರೆ (!sycl_get_mem_info(ctx->device, ಬಿಡುಗಡೆ, ಒಟ್ಟು)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *ಒಟ್ಟು, *ಬಿಡುಗಡೆ);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *ಬಿಡುಗಡೆ, *ಒಟ್ಟು);
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಸ್ಥಿರಾಂಕ sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    ಆದರೆ (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);

    ತಾರ್ಕಿಕ host_ಬಫರ್ = getenv("GGML_SYCL_NO_PINNED") == ಶೂನ್ಯಸೂಚಕ;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_NO_PEER_COPY
    ತಾರ್ಕಿಕ events = ಸುಳ್ಳು;
#ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ events = ನಿಜ;
#ಆದರೆ_ಅಂತ್ಯ

    props->caps = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ host_ಬಫರ್,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ events,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    GGML_ಬಳಸದ(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_sycl_init(ctx->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_sycl_ಬಫರ್_type(ctx->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_sycl_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_sycl_host_ಬಫರ್_type_for_device(ctx->device);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_sycl_device_ಬಫರ್_from_host_ptr(ggml_backend_dev_t dev, ಶೂನ್ಯ * ptr, size_t size, size_t max_tensor_ಗಾತ್ರ) {
    GGML_ಬಳಸದ(dev);
    GGML_ಬಳಸದ(ptr);
    GGML_ಬಳಸದ(size);
    GGML_ಬಳಸದ(max_tensor_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    ಪೂರ್ಣಾಂಕ device = sycl_ctx->device;
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                ಆದರೆ (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                }
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            ಹಿಂತಿರುಗಿ (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_ಗಾತ್ರ(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_ಗಾತ್ರ(op->src[0]->type));
        ಸ್ಥಿತಿ GGML_OP_CONV_2D_DW:
        ಸ್ಥಿತಿ GGML_OP_CONV_TRANSPOSE_2D:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_XIELU:
                ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                    ಹಿಂತಿರುಗಿ ನಿಜ;
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_GLU:
            ಆಯ್ಕೆ (ggml_get_glu_op(op)) {
                ಸ್ಥಿತಿ GGML_GLU_OP_REGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_OAI:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ERF:
                ಸ್ಥಿತಿ GGML_GLU_OP_GEGLU_ಶೀಘ್ರ:
                ಸ್ಥಿತಿ GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    ಹಿಂತಿರುಗಿ ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                ಪೂರ್ವನಿಯೋಜಿತ:
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            {
                ರಚನೆ ggml_tensor * a = op->src[0];
                ರಚನೆ ggml_tensor * b = op->src[1];

                ಆದರೆ (a->ne[3] != b->ne[3]) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                ಆದರೆ (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                ಆದರೆ (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಆದರೆ (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_ಹೊರಹೋಗು_PROD:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆಯ್ಕೆ (op->src[0]->type) {
                    ಸ್ಥಿತಿ GGML_TYPE_I32:
                    ಸ್ಥಿತಿ GGML_TYPE_F16:
                    ಸ್ಥಿತಿ GGML_TYPE_BF16:
                    ಸ್ಥಿತಿ GGML_TYPE_F32:
                    ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
                    ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
                    ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
                    ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
                    ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
                    ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
                        ಹಿಂತಿರುಗಿ ನಿಜ;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
            }
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ:
            // return true;
             ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         ಸ್ಥಿತಿ GGML_OP_ಗಣ:
               ಹಿಂತಿರುಗಿ (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            {
                ಆದರೆ (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }
                ಸ್ವಯಂಚಾಲಿತ res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                ಹಿಂತಿರುಗಿ res;
            }
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_PRE:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_COMB:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_DSV4_HC_POST:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == ಶೂನ್ಯಸೂಚಕ || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_LIGHTNING_INDEXER:
            ಹಿಂತಿರುಗಿ op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_ಗಾತ್ರ * 8;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                ಆದರೆ (src0_type == GGML_TYPE_F16) {
                    ಆದರೆ (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }

                ಆದರೆ (src0_type == GGML_TYPE_BF16) {
                    ಆದರೆ (src1_type == GGML_TYPE_Q4_0 || //big error in ut
                        src1_type == GGML_TYPE_Q4_1 || //big error in ut
                        src1_type == GGML_TYPE_Q8_0 || //big error in ut
                        src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }

                ಆದರೆ (src0_type == GGML_TYPE_F32) {
                    ಆದರೆ (src1_type == GGML_TYPE_Q2_K ||
                        src1_type == GGML_TYPE_Q3_K ||
                        src1_type == GGML_TYPE_Q4_K ||
                        src1_type == GGML_TYPE_Q5_K ||
                        src1_type == GGML_TYPE_Q6_K ||
                        src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ4_XS) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }

                ಆದರೆ (src1_type == GGML_TYPE_F32) {
                    ಆದರೆ (src0_type == GGML_TYPE_Q1_0 ||
                        src0_type == GGML_TYPE_NVFP4 ||
                        src0_type == GGML_TYPE_Q2_K ||
                        src0_type == GGML_TYPE_Q3_K ||
                        src0_type == GGML_TYPE_Q4_K ||
                        src0_type == GGML_TYPE_Q5_K ||
                        src0_type == GGML_TYPE_Q6_K ||
                        src0_type == GGML_TYPE_IQ2_XXS ||
                        src0_type == GGML_TYPE_IQ2_XS ||
                        src0_type == GGML_TYPE_IQ2_S ||
                        src0_type == GGML_TYPE_IQ3_XXS ||
                        src0_type == GGML_TYPE_IQ1_S ||
                        src0_type == GGML_TYPE_IQ1_M ||
                        src0_type == GGML_TYPE_IQ3_S ||
                        src0_type == GGML_TYPE_IQ4_NL ||
                        src0_type == GGML_TYPE_IQ4_XS
                    ) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }

                ಆದರೆ (src0_type == src1_type) {
                    ಆದರೆ (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                    }
                }

                ಆದರೆ (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
                }

                ಹಿಂತಿರುಗಿ ನಿಜ;
            }
        ಸ್ಥಿತಿ GGML_OP_REPEAT_ಹಿಂಭಾಗ:
            {
                ggml_type src0_type = op->src[0]->type;
                ಹಿಂತಿರುಗಿ src0_type == GGML_TYPE_F32;
            }
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
        ಸ್ಥಿತಿ GGML_OP_DUP:
        ಸ್ಥಿತಿ GGML_OP_ARGMAX:
        ಸ್ಥಿತಿ GGML_OP_NONE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_ADD1:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು_ID:
        ಸ್ಥಿತಿ GGML_OP_SUB:
        ಸ್ಥಿತಿ GGML_OP_ಎಣಿಕೆ_EQUAL:
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_DIV:
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_PAD_REFLECT_1D:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SQR:
        ಸ್ಥಿತಿ GGML_OP_SQRT:
        ಸ್ಥಿತಿ GGML_OP_SIN:
        ಸ್ಥಿತಿ GGML_OP_COS:
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
        ಸ್ಥಿತಿ GGML_OP_LOG:
        ಸ್ಥಿತಿ GGML_OP_NORM:
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            // the scale kernel is f32 only
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            {
                ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
                ಹಿಂತಿರುಗಿ src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        ಸ್ಥಿತಿ GGML_OP_DIAG_ಮುಖವಾಡ_INF:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX_ಹಿಂಭಾಗ: {
            ದಶಮಾಂಶ max_bias = 0.0f;
            ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 1, ಗಾತ್ರ(ದಶಮಾಂಶ));
            ಹಿಂತಿರುಗಿ max_bias == 0.0f;
        }
        ಸ್ಥಿತಿ GGML_OP_ROPE:
        ಸ್ಥಿತಿ GGML_OP_ROPE_ಹಿಂಭಾಗ:
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು_3_D:
        ಸ್ಥಿತಿ GGML_OP_UPSCALE:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_COL2IM_1D:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#ಆದರೆ_ಅಂತ್ಯ
                   ) &&
                   op->src[0]->type == op->type;
        ಸ್ಥಿತಿ GGML_OP_CONV_3D: {
            ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ಸ್ಥಿರಾಂಕ int64_t knl_n_ಒಟ್ಟು = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ಸ್ಥಿರಾಂಕ int64_t patch_ಒಟ್ಟು = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_ಒಟ್ಟು > 0 && patch_ಒಟ್ಟು > 0;
        }
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ:
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            ಹಿಂತಿರುಗಿ ggml_is_contiguous(op->src[0]);
        ಸ್ಥಿತಿ GGML_OP_ARGSORT:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_ತುದಿ_K: {
            ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = op->src[0];
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ k = op->ne[0];
            ಹಿಂತಿರುಗಿ src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        ಸ್ಥಿತಿ GGML_OP_POOL_2D:
        ಸ್ಥಿತಿ GGML_OP_POOL_1D:
        ಸ್ಥಿತಿ GGML_OP_ACC:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            ಆದರೆ (ggml_get_op_params_i32(op, 8) != 0) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_LEAKY_RELU:
        ಸ್ಥಿತಿ GGML_OP_TIMESTEP_EMBEDDING:
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
        ಸ್ಥಿತಿ GGML_OP_GATED_ರೇಖೀಯ_ATTN:
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_ADAMW:
        ಸ್ಥಿತಿ GGML_OP_OPT_ಹೆಜ್ಜೆ_SGD:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ROLL:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_ARANGE:
            ಹಿಂತಿರುಗಿ op->type == GGML_TYPE_F32;
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            ಆದರೆ (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                ಹಿಂತಿರುಗಿ (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_ಗಾತ್ರ == 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                // TODO Mamba-1 not yet ported to SYCL
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        ಸ್ಥಿತಿ GGML_OP_FILL:
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
        ಸ್ಥಿತಿ GGML_OP_DIAG:
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS:
        ಸ್ಥಿತಿ GGML_OP_CROSS_ENTROPY_LOSS_ಹಿಂಭಾಗ:
            ಹಿಂತಿರುಗಿ ನಿಜ;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            ಹಿಂತಿರುಗಿ op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಹಿಂತಿರುಗಿ ggml_sycl_flash_attn_ext_supported(device, op);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ತಾರ್ಕಿಕ res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    ಹಿಂತಿರುಗಿ res;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    ಆದರೆ (buft->iface.get_name != ggml_backend_sycl_ಬಫರ್_type_get_name) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ggml_backend_sycl_ಬಫರ್_type_context * buft_ctx = (ggml_backend_sycl_ಬಫರ್_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ buft_ctx->device == sycl_ctx->device;
}

ಸ್ಥಿರ int64_t get_op_batch_ಗಾತ್ರ(ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            ಹಿಂತಿರುಗಿ 0;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            ಹಿಂತಿರುಗಿ op->ne[1];
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            ಹಿಂತಿರುಗಿ op->ne[2];
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ ggml_nrows(op);
    }
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ಹಿಂತಿರುಗಿ get_op_batch_ಗಾತ್ರ(op) >= sycl_ctx->op_offload_min_batch_ಗಾತ್ರ;
}

ಸ್ಥಿರ ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_NO_PEER_COPY
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
#ಇಲ್ಲದಿದ್ದರೆ
  sycl::event *event_ptr = ಹೊಸ sycl::event();

  ಹಿಂತಿರುಗಿ ಹೊಸ ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) ಪ್ರಯತ್ನಿಸು {
  GGML_ಬಳಸದ(dev);
  ಆದರೆ (event == ಶೂನ್ಯಸೂಚಕ) {
    ಹಿಂತಿರುಗಿ;
  }

  ಆದರೆ (event->context != ಶೂನ್ಯಸೂಚಕ) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    ಅಳಿಸು sycl_event;
    event->context = ಶೂನ್ಯಸೂಚಕ;
  }

  ಅಳಿಸು event;
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}


ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) ಪ್ರಯತ್ನಿಸು {
  GGML_ಬಳಸದ(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
  std::ನಿರ್ಗಮಿಸು(1);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_ಸ್ಮರಣೆ,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_ಬಫರ್_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

ರಚನೆ ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ಬಳಸದ(reg);
    ಹಿಂತಿರುಗಿ GGML_SYCL_NAME;
}

ಸ್ಥಿರ size_t ggml_backend_sycl_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    ಹಿಂತಿರುಗಿ ctx->devices.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t ಸೂಚ್ಯಂಕ) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < ctx->devices.size());
    ಹಿಂತಿರುಗಿ ctx->devices[ಸೂಚ್ಯಂಕ];
}

// ==========================================================================
// Tensor parallelism (--split-mode tensor) for the SYCL backend.
//
// The meta-backend invokes these three entry points via get_proc_address:
//   * ggml_backend_sycl_comm_init             - one-time per-graph setup
//   * ggml_backend_sycl_comm_allreduce_tensor - per-allreduce step
//   * ggml_backend_sycl_comm_free             - tear-down
//
// For N=2 (dual-GPU), this is a degenerate ring allreduce with dual paths
// chosen by tensor size:
//
//   * Small (nelem < 32K): FP32 direct memcpy + per-device ADD
//     kernel. The kernel depends_on() its corresponding memcpy event
//     so it doesn't read partial data. Both devices run in parallel.
//
//   * Large (nelem >= 32K): BF16-compressed. Each device compresses
//     its FP32 partial to BF16 locally, cross-device memcpys
//     to the peer (half the PCI bandwidth), where it is decompressed
//     and added into the local FP32 partial. 6 SYCL submissions per
//     allreduce (2 compress + 2 memcpy + 2 decompress-add) vs the
//     4 for the small path, but the bandwidth saving > 6 GB/s PCIe x 2
//     dominates for larger tensors.
//
// Storage: A persistent uint8_t buffer per device, sized to
// 4 * nelem bytes. Both paths reinterpret the same bytes (small path
// as nelem floats; large path as outbox + inbox = 2*nelem uint16_t
// each, using the full 4*nelem byte budget either way). Single
// alloc+free per device keeps the SYCL pool's strict-LIFO invariant
// trivial.
//
// For non-(N=2 FP32 contiguous) cases, comm_init or comm_allreduce_tensor
// returns null/false, causing the meta-backend to use its generic
// butterfly all-reduce fallback.
// ==========================================================================

ರಚನೆ ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_ಹೊರಹೋಗು[2] = { ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ };
    uint8_t * host_ಒಳ[2]  = { ಶೂನ್ಯಸೂಚಕ, ಶೂನ್ಯಸೂಚಕ };
    size_t    host_ಬೈಟುಗಳು  = 0;

    ಶೂನ್ಯ free_host() {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 2; ++i) {
            ಸ್ಥಿರಾಂಕ sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            ಆದರೆ (host_ಹೊರಹೋಗು[i]) { sycl::ಬಿಡುಗಡೆ(host_ಹೊರಹೋಗು[i], sctx); host_ಹೊರಹೋಗು[i] = ಶೂನ್ಯಸೂಚಕ; }
            ಆದರೆ (host_ಒಳ[i])  { sycl::ಬಿಡುಗಡೆ(host_ಒಳ[i],  sctx); host_ಒಳ[i]  = ಶೂನ್ಯಸೂಚಕ; }
        }
        host_ಬೈಟುಗಳು = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
ಸ್ಥಿರ ಶೂನ್ಯ ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, ಸರತಿ_ptr q0, ಸರತಿ_ptr q1,
                                    ಶೂನ್ಯ * dst0, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src1, ಶೂನ್ಯ * dst1, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0,
                                    size_t nbytes) {
    ಆದರೆ (comm->host_ಬೈಟುಗಳು < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_ಹೊರಹೋಗು[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_ಒಳ[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_ಹೊರಹೋಗು[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_ಒಳ[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_ಹೊರಹೋಗು[0] && comm->host_ಒಳ[0] && comm->host_ಹೊರಹೋಗು[1] && comm->host_ಒಳ[1]);
        comm->host_ಬೈಟುಗಳು = nbytes;
    }

    sycl::event e0 = q0->ಸ್ಮರಣೆ_ನಕಲು(comm->host_ಹೊರಹೋಗು[0], src0, nbytes);
    sycl::event e1 = q1->ಸ್ಮರಣೆ_ನಕಲು(comm->host_ಹೊರಹೋಗು[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::ಸ್ಮರಣೆ_ನಕಲು(comm->host_ಒಳ[0], comm->host_ಹೊರಹೋಗು[1], nbytes);
    std::ಸ್ಮರಣೆ_ನಕಲು(comm->host_ಒಳ[1], comm->host_ಹೊರಹೋಗು[0], nbytes);
    q0->ಸ್ಮರಣೆ_ನಕಲು(dst0, comm->host_ಒಳ[0], nbytes);
    q1->ಸ್ಮರಣೆ_ನಕಲು(dst1, comm->host_ಒಳ[1], nbytes);
}

ಶೂನ್ಯ * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) ಪ್ರಯತ್ನಿಸು {
    ಗಾಗಿ (size_t i = 0; i < n_backends; ++i) {
        ಆದರೆ (!ggml_backend_is_sycl(backends[i])) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    ಆದರೆ (n_backends != 2) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಸ್ವಯಂಚಾಲಿತ * ctx = ಹೊಸ ggml_backend_sycl_comm_context;
    ctx->backends.ನಿಯೋಜಿಸು(backends, backends + n_backends);
    ಸ್ವಯಂಚಾಲಿತ * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    ಸ್ವಯಂಚಾಲಿತ * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    ಹಿಂತಿರುಗಿ ctx;
}
ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception &) { ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ; }
ಹಿಡಿ (...)                     { ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ; }

ಶೂನ್ಯ ggml_backend_sycl_comm_free(ಶೂನ್ಯ * comm_ctx_v) {
    ಸ್ವಯಂಚಾಲಿತ * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ಆದರೆ (comm_ctx == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    ಆದರೆ (comm_ctx->backends.size() == 2) {
        ಸ್ವಯಂಚಾಲಿತ * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        ಸ್ವಯಂಚಾಲಿತ * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        ಪ್ರಯತ್ನಿಸು {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } ಹಿಡಿ (...) { /* best effort during shutdown */ }
    }

    ಅಳಿಸು comm_ctx;
}

ತಾರ್ಕಿಕ ggml_backend_sycl_comm_allreduce_tensor(ಶೂನ್ಯ * comm_ctx_v, ರಚನೆ ggml_tensor ** tensors) ಪ್ರಯತ್ನಿಸು {
    ಆದರೆ (comm_ctx_v == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ವಯಂಚಾಲಿತ * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ಸ್ಥಿರಾಂಕ size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    ಆದರೆ (n_backends != 2) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    ಆದರೆ (tensors[0]->type != tensors[1]->type) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಆದರೆ (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಸ್ಥಿರಾಂಕ int64_t nelem  = ggml_nelements(tensors[0]);
    ಸ್ಥಿರಾಂಕ size_t  nbytes = ggml_nbytes(tensors[0]);
    ಆದರೆ (nelem == 0) {
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ಸ್ವಯಂಚಾಲಿತ * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    ಸ್ವಯಂಚಾಲಿತ * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    ಸರತಿ_ptr q0 = ctx0->stream();
    ಸರತಿ_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    ಆದರೆ (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    ಆದರೆ (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->ದತ್ತಾಂಶ;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->ದತ್ತಾಂಶ;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->ದತ್ತಾಂಶ, f16_tmp1, tensors[0]->ದತ್ತಾಂಶ, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((ದಶಮಾಂಶ) f16_out0[i] + (ದಶಮಾಂಶ) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((ದಶಮಾಂಶ) f16_out1[i] + (ದಶಮಾಂಶ) f16_tmp1[i]);
            });
        });
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    ದಶಮಾಂಶ * out0 = (ದಶಮಾಂಶ *) tensors[0]->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * out1 = (ದಶಮಾಂಶ *) tensors[1]->ದತ್ತಾಂಶ;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    ಸ್ಥಿರ constexpr int64_t BF16_ಮಿತಿ = 32768;

    ಆದರೆ (nelem < BF16_ಮಿತಿ) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        ದಶಮಾಂಶ * tmp0 = (ದಶಮಾಂಶ *) buf0;
        ದಶಮಾಂಶ * tmp1 = (ದಶಮಾಂಶ *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->ದತ್ತಾಂಶ, tmp1, tensors[0]->ದತ್ತಾಂಶ, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out0[i] += tmp0[i];
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                out1[i] += tmp1[i];
            });
        });
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    // BF16 large path: 6 SYCL submissions per allreduce, but the
    // cross-device memcpy is HALF the bytes. Pure bit-shift
    // conversion (no rounding) — matches ggml's truncating fp32->bf16.
    uint16_t * outbox0 = (uint16_t *) buf0;
    uint16_t * inbox0  = outbox0 + nelem;
    uint16_t * outbox1 = (uint16_t *) buf1;
    uint16_t * inbox1  = outbox1 + nelem;

    // Phase A: compress each device's local partial in parallel.
    q0->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox0[i] = (uint16_t) (sycl::ಬಿಟ್_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::ಬಿಟ್_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    ಸ್ಥಿರಾಂಕ size_t bf16_ಬೈಟುಗಳು = nelem * ಗಾತ್ರ(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_ಬೈಟುಗಳು);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::ಬಿಟ್_cast<ದಶಮಾಂಶ>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::ಬಿಟ್_cast<ದಶಮಾಂಶ>(((uint32_t) inbox1[i]) << 16);
        });
    });

    ಹಿಂತಿರುಗಿ ನಿಜ;
}
ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception &) { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }
ಹಿಡಿ (...)                     { ಹಿಂತಿರುಗಿ ಸುಳ್ಳು; }

ಸ್ಥಿರ ಶೂನ್ಯ *ggml_backend_sycl_reg_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name) {
    GGML_ಬಳಸದ(reg);

    ಆದರೆ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_sycl_ವಿಭಜಿಸು_ಬಫರ್_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    ಆದರೆ (strcmp(name, "ggml_backend_comm_init") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_sycl_comm_init;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_comm_free") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_sycl_comm_free;
    }
    ಆದರೆ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ಹಿಂತಿರುಗಿ (ಶೂನ್ಯ *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_ಬಳಸದ(name);
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_ಎಣಿಕೆ,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_ವಿಳಾಸ,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    ಸ್ಥಿರ ggml_backend_reg reg;
    ಸ್ಥಿರ ತಾರ್ಕಿಕ initialized = ಸುಳ್ಳು;

    {
        ಸ್ಥಿರ std::mutex mutex;
        std::lock_guard<std::mutex> ಬೀಗ(mutex);
        ಆದರೆ (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = ಹೊಸ ggml_backend_sycl_reg_context;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ min_batch_ಗಾತ್ರ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_sycl_info().device_ಎಣಿಕೆ; i++) {
                ggml_backend_sycl_device_context * dev_ctx = ಹೊಸ ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_ಗಣ_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_ಗಾತ್ರ = min_batch_ಗಾತ್ರ;

                ggml_backend_dev_t dev = ಹೊಸ ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = ನಿಜ;
    }

    ಹಿಂತಿರುಗಿ &reg;
}

ggml_backend_t ggml_backend_sycl_init(ಪೂರ್ಣಾಂಕ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_ಸೂಚ್ಯಂಕ(device);

    ggml_backend_sycl_context * ctx = ಹೊಸ ggml_backend_sycl_context(device);
    ಆದರೆ (ctx == ಶೂನ್ಯಸೂಚಕ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    };

    ggml_backend_t sycl_backend = ಹೊಸ ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    ಹಿಂತಿರುಗಿ sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
