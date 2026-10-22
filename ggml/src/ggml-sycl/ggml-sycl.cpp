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

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <array>
#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <atomic>
#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>
#ਅਨ੍ਤਰ੍ਭਾਵ <cinttypes>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstddef>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <float.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>
#ਅਨ੍ਤਰ੍ਭਾਵ <optional>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <fstream>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <regex>

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/backend.hpp>
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ਅਨ੍ਤਰ੍ਭਾਵ <level_zero/ze_api.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <level_zero/zes_api.h>
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_SYCL_ਗ੍ਰਾਫ) && SYCL_EXT_ONEAPI_ASYNC_ਸਮ੍ਰਿਤੀ_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#ਜੇ_ਅੰਤ
#ਜੇ SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/half_type.hpp>

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/add-id.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/backend.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/element_wise.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/fwht.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/fused-gemm.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/gemm.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/getrows.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/mem.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/norm.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/presets.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/quantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/repeat_back.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/set_rows.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/set.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/dsv4-hc.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/lightning-indexer.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/conv2d.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/conv2d-dw.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/conv2d-transpose.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/ssm_conv.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/sycl_hw.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/ssm_scan.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/fill.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/cumsum.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/diag.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/opt-step.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/solve_tri.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/gated_delta_net.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/pool.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl/cross_entropy_loss.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MEM_ਮਾਪ_2_M	0x00200000
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MEM_ਮਾਪ_1_G	0x40000000

ਸਥਿਰ ਤਾਰਕਿਕ g_sycl_loaded = ਝੂਠ;
ਪੂਰਨ g_ggml_sycl_debug = 0;
ਪੂਰਨ g_ggml_sycl_dev_debug = 0;
ਪੂਰਨ g_ggml_sycl_enable_optimize = 1;
ਪੂਰਨ g_ggml_sycl_enable_ਗ੍ਰਾਫ = 0;
ਪੂਰਨ g_ggml_sycl_enable_dnn = 1;
ਪੂਰਨ g_ggml_sycl_fa_onednn = 1;
ਪੂਰਨ g_ggml_sycl_fa_onednn_max_kv = 0;
ਪੂਰਨ g_ggml_sycl_enable_mkl_fa = 1;
ਪੂਰਨ g_ggml_sycl_memtrace = 0;
ਪੂਰਨ g_ggml_sycl_memtrace_ਕਦਮ = 64;
ਪੂਰਨ g_ggml_sycl_enable_vmm = 1;
ਪੂਰਨ g_ggml_sycl_enable_fusion = 1;
ਪੂਰਨ g_ggml_sycl_enable_esimd = 1;
ਪੂਰਨ g_ggml_sycl_mmvq_wide = 1;
ਪੂਰਨ g_ggml_sycl_prioritize_dmmv = 0;
ਪੂਰਨ g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
ਪੂਰਨ g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
ਪੂਰਨ g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
ਪੂਰਨ g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_sycl_dynamic_precision_ਨਾਂ[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
ਸਥਿਰ ਪੂਰਨ ggml_sycl_get_env_precision(ਅਚਲ ਅੱਖਰ * name, ਪੂਰਨ def) {
    ਅਚਲ ਅੱਖਰ * env = getenv(name);
    ਜੇ (!env) {
        ਵਾਪਸ def;
    }
    std::string mode(env);
    ਲਈ (ਅੱਖਰ & c : mode) {
        c = (ਅੱਖਰ) std::ਵੱਡਾ_ਅੱਖਰ((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ) c);
    }
    ਲਈ (ਪੂਰਨ i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        ਜੇ (mode == ggml_sycl_dynamic_precision_ਨਾਂ[i]) {
            ਵਾਪਸ i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_ਨਾਂ[def]);
    ਵਾਪਸ def;
}
ਪੂਰਨ g_ggml_sycl_use_async_mem_op = 0;
ਪੂਰਨ g_ggml_sycl_use_async_mem_op_requested = 1;
ਪੂਰਨ g_ggml_sycl_use_level_zero_api = 0;
ਪੂਰਨ g_ggml_sycl_enable_flash_attention = 1;
ਪੂਰਨ g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
ਪੂਰਨ g_ggml_sycl_usm_system = 0;
ਪੂਰਨ g_ggml_sycl_enable_host_pinned_mem = 1;
ਪੂਰਨ g_ggml_sycl_host_pinned_mem_2g = 0;
ਪੂਰਨ g_ggml_sycl_upload_staging_slots = 4;
ਪੂਰਨ g_ggml_sycl_get_mem_api = ਸਮ੍ਰਿਤੀ_API_TYPE_LEVEL_ZERO;
ਪੂਰਨ g_ggml_sycl_enable_sparse_fa = 0;
ਪੂਰਨ g_ggml_sycl_debug_sparse_fa = 0;
ਪੂਰਨ g_ggml_sycl_sparse_fa_margin = 256;

#ਜੇ GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_dnnl_detect_optimized_gemm(ਪੂਰਨ device) {
    ਵਰਤੋਂ dt = dnnl::ਸਮ੍ਰਿਤੀ::ਅੰਕੜੇ_type;

    ਕੋਸ਼ਿਸ਼ {
        ਅਚਲ dnnl::ਸਮ੍ਰਿਤੀ::dims dims    = { 1, 64, 64 };
        ਅਚਲ dnnl::ਸਮ੍ਰਿਤੀ::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.ਸੈੱਟ_scratchpad_ਬਹੁਲਕ(dnnl::scratchpad_ਬਹੁਲਕ::user);

        ਅਚਲ ਸਵੈਚਾਲਿਤ & q   = dpct::dev_mgr::instance().get_device(device).default_ਕਤਾਰ();
        ਅਚਲ ਸਵੈਚਾਲਿਤ   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ਅਚਲ ਸਵੈਚਾਲਿਤ   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::ਸਮ੍ਰਿਤੀ::desc(dims, dt::f16, strides),
                                                        dnnl::ਸਮ੍ਰਿਤੀ::desc(dims, dt::f16, strides),
                                                        dnnl::ਸਮ੍ਰਿਤੀ::desc(dims, dt::f32, strides), attr);

        ਅਚਲ std::string impl = pd.impl_info_str();
        ਜੇ (impl.find("ref") == std::string::npos) {
            ਵਾਪਸ ਸੱਚ;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } ਫੜੋ (ਅਚਲ std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    ਵਾਪਸ ਝੂਠ;
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    ਕੋਸ਼ਿਸ਼ {
        info.device_ਗਿਣਤੀ = dpct::dev_mgr::instance().device_ਗਿਣਤੀ();
    } ਫੜੋ (sycl::exception ਅਚਲ &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_ਗਿਣਤੀ = 0;
    }
    ਜੇ (info.device_ਗਿਣਤੀ == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        ਵਾਪਸ info;
    }

    GGML_ASSERT(info.device_ਗਿਣਤੀ <= GGML_SYCL_MAX_DEVICES);

    int64_t ਕੁੱਲ_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    ਲਈ (ਪੂਰਨ i = 0; i < info.device_ਗਿਣਤੀ; ++i) {
        dpct::device_info prop;
        ਸਵੈਚਾਲਿਤ & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#ਜੇ !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#ਨਹੀਂਤਾਂ
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        ਜੇ (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ਅਚਲ size_t physical_ਪੰਨਾ = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_ਪੰਨਾ);
        }
#ਜੇ_ਅੰਤ

        info.default_tensor_ਵੰਡੋ[i] = ਕੁੱਲ_vram;
        ਕੁੱਲ_vram += prop.get_global_mem_ਮਾਪ();

        info.devices[i].cc =
            100 * prop.get_major_ਸੰਸਕਰਣ() + 10 * prop.get_minor_ਸੰਸਕਰਣ();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#ਜੇ GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#ਜੇ_ਅੰਤ
        info.devices[i].smpbo = prop.get_local_mem_ਮਾਪ();
        info.devices[i].warp_ਮਾਪ = WARP_ਮਾਪ;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_ਸਮੂਹ_sizes[i] = prop.get_max_work_ਸਮੂਹ_ਮਾਪ();
        info.devices[i].max_wg_ਪ੍ਰਤੀ_cu = info.max_work_ਸਮੂਹ_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        ਜੇ (device.is_gpu() && device.default_ਕਤਾਰ().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = ਝੂਠ;
        }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        ਜੇ (info.ext_oneapi_level_zero && device.is_gpu() && device.default_ਕਤਾਰ().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_ਕਤਾਰ().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_ਨਤੀਜਾ_t r = zeDeviceGetProperties(ze_dev, &props);
            ਜੇ (r == ZE_ਨਤੀਜਾ_SUCCESS) {
                info.devices[i].l0_device_type_valid = ਸੱਚ;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#ਜੇ_ਅੰਤ
    }

    ਲਈ (ਪੂਰਨ id = 0; id < info.device_ਗਿਣਤੀ; ++id) {
        info.default_tensor_ਵੰਡੋ[id] /= ਕੁੱਲ_vram;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#ਜੇ_ਅੰਤ

    ਵਾਪਸ info;
}

ਅਚਲ ggml_sycl_device_info & ggml_sycl_info() {
    ਸਥਿਰ ggml_sycl_device_info info = ggml_sycl_init();
    ਵਾਪਸ info;
}

ਸਥਿਰ ਖਾਲੀ print_device_detail(ਪੂਰਨ id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string ਸੰਸਕਰਣ_2;
    ਸੰਸਕਰਣ_2 += std::to_string(prop.get_major_ਸੰਸਕਰਣ());
    ਸੰਸਕਰਣ_2 += ".";
    ਸੰਸਕਰਣ_2 += std::to_string(prop.get_minor_ਸੰਸਕਰਣ());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    ਸਵੈਚਾਲਿਤ global_mem_ਮਾਪ = prop.get_global_mem_ਮਾਪ()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), ਸੰਸਕਰਣ_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_ਸਮੂਹ_ਮਾਪ(), prop.get_max_sub_ਸਮੂਹ_ਮਾਪ(),
            global_mem_ਮਾਪ, device.get_info<sycl::info::device::driver_ਸੰਸਕਰਣ>().c_str());
}

ਸਥਿਰ ਖਾਲੀ print_device_opt_feature(ਪੂਰਨ device_ਗਿਣਤੀ) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    ਲਈ (ਪੂਰਨ id = 0; id < device_ਗਿਣਤੀ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ਪੂਰਨ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
ਖਾਲੀ ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    ਪੂਰਨ device_ਗਿਣਤੀ = dpct::dev_mgr::instance().device_ਗਿਣਤੀ();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_ਗਿਣਤੀ);

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

    ਲਈ (ਪੂਰਨ id = 0; id < device_ਗਿਣਤੀ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ਪੂਰਨ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_ਗਿਣਤੀ);
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ* dev2dev_int2str(ਪੂਰਨ dev2dev) {
    ਜੇ (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        ਵਾਪਸ "SYCL API";
    } ਨਹੀਂਤਾਂ ਜੇ (dev2dev == DEV2DEV_MEMCPY_L0) {
        ਵਾਪਸ "Level Zero API";
    } ਨਹੀਂਤਾਂ ਜੇ (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        ਵਾਪਸ "Host Forward";
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ਸਥਿਰ ze_ਨਤੀਜਾ_t init_zes() {
    ze_ਨਤੀਜਾ_t res = zesInit(0);
    ਜੇ (res != ZE_ਨਤੀਜਾ_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (ਪੂਰਨ) res);
    }
    ਵਾਪਸ res;
}

ਸਥਿਰ ze_ਨਤੀਜਾ_t get_zes_init_res() {
    ਸਥਿਰ ze_ਨਤੀਜਾ_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (ਪੂਰਨ) zes_init_res);
    ਵਾਪਸ zes_init_res;
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ initialize_sycl_begining() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਖਾਲੀ ggml_check_sycl() ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    ਜੇ (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_ਗ੍ਰਾਫ = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_ਕਦਮ = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
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

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#ਨਹੀਂਤਾਂ
        g_ggml_sycl_use_level_zero_api = 0;
#ਜੇ_ਅੰਤ
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", ਸਮ੍ਰਿਤੀ_API_TYPE_LEVEL_ZERO);
        ਜੇ (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL;
        }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#ਨਹੀਂਤਾਂ
        g_ggml_sycl_enable_flash_attention = 0;
#ਜੇ_ਅੰਤ

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
#ਜੇ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_SYCL_ਗ੍ਰਾਫ)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#ਜੇ_ਅੰਤ
#ਜੇ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#ਜੇ_ਅੰਤ

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ਜੇ_ਅੰਤ
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_ਕਦਮ);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ਗ੍ਰਾਫ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_ਗ੍ਰਾਫ);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#ਜੇ_ਅੰਤ

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_ਨਾਂ[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_ਨਾਂ[g_ggml_sycl_dynamic_required_precision]);

#ਜੇ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#ਜੇ_ਅੰਤ

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#ਜੇ defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#ਜੇ_ਅੰਤ
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#ਨਹੀਂਤਾਂ
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#ਜੇ_ਅੰਤ

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
#ਜੇ defined(GGML_SYCL_ਗ੍ਰਾਫ) && SYCL_EXT_ONEAPI_ASYNC_ਸਮ੍ਰਿਤੀ_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_ਗ੍ਰਾਫ;
        ਜੇ (g_ggml_sycl_use_async_mem_op) {
            ਲਈ (ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i = 0; i < dpct::dev_mgr::instance().device_ਗਿਣਤੀ(); ++i) {
                ਜੇ (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_ਸਮ੍ਰਿਤੀ_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    ਰੋਕੋ;
                }
            }
        }
#ਜੇ_ਅੰਤ
        ਜੇ (CHECK_TRY_ERROR(g_all_sycl_device_ਗਿਣਤੀ =
                            dpct::dev_mgr::instance().device_ਗਿਣਤੀ()) != 0) {
            initialized = ਸੱਚ;
            g_sycl_loaded = ਝੂਠ;
            ਵਾਪਸ;
        }
        GGML_ASSERT(g_all_sycl_device_ਗਿਣਤੀ <= GGML_SYCL_MAX_DEVICES);

        initialized = ਸੱਚ;
        g_sycl_loaded = ਸੱਚ;
        ggml_backend_sycl_print_sycl_devices();
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
ਸਤਰਅੰਦਰ ਖਾਲੀ check_allow_gpu_ਸੂਚਕਾਂਕ(ਅਚਲ ਪੂਰਨ device_ਸੂਚਕਾਂਕ) {
  ਜੇ (device_ਸੂਚਕਾਂਕ >= ggml_sycl_info().device_ਗਿਣਤੀ) {
    ਅੱਖਰ error_buf[256];
    snprintf(
        error_buf,
        ਆਕਾਰ(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_ਸੂਚਕਾਂਕ,
        ggml_sycl_info().device_ਗਿਣਤੀ - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
  }
}

GGML_API ਖਾਲੀ ggml_backend_sycl_get_gpu_list(ਪੂਰਨ *id_list, ਪੂਰਨ max_len) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    ਲਈ(ਪੂਰਨ i=0;i<max_len;i++) id_list[i] = -1;

    ਲਈ (ਪੂਰਨ i=0;i< ggml_sycl_info().device_ਗਿਣਤੀ;i++){
        ਜੇ (i>=max_len) ਰੋਕੋ;
        id_list[i] = i;
    }
    ਵਾਪਸ;
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ free_aligned_mem_host(ਖਾਲੀ * memblock) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    _aligned_free(memblock);
#ਨਹੀਂਤਾਂ
    ਮੁਕਤ(memblock);
#ਜੇ_ਅੰਤ
}

// sycl buffer

ਬਣਤਰ ggml_backend_sycl_ਬਫਰ_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    ਸਥਿਰ constexpr size_t staging_ਥਾਂ_ਮਾਪ = 8*1024*1024;

    ਬਣਤਰ host_staging {
        ਖਾਲੀ * ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
        std::vector<sycl::event> events;
        std::vector<ਤਾਰਕਿਕ> submitted;
        ਪੂਰਨ ਅਗਲਾ = 0;
    };

    ਪੂਰਨ device;
    ਖਾਲੀ * dev_ptr = ਖਾਲੀ_ਸੂਚਕ;
    ਕਤਾਰ_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    ਤਾਰਕਿਕ is_usm_system;

    ggml_backend_sycl_ਬਫਰ_context(ਪੂਰਨ device, ਖਾਲੀ * dev_ptr, ਕਤਾਰ_ptr stream, ਤਾਰਕਿਕ is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_ਸੂਚਕਾਂਕ(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    ਖਾਲੀ drop_host_staging() {
        ਲਈ (size_t i = 0; i < staging.submitted.size(); ++i) {
            ਜੇ (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = ਝੂਠ;
            }
        }
        ਜੇ (staging.ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ) {
            sycl::ਮੁਕਤ(staging.ਅੰਕੜੇ, *stream);
            staging.ਅੰਕੜੇ = ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    ~ggml_backend_sycl_ਬਫਰ_context() {
        drop_host_staging();
        ਜੇ (dev_ptr != ਖਾਲੀ_ਸੂਚਕ) {
            ggml_sycl_ਸੈੱਟ_device(device);
            ਜੇ (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            ਨਹੀਂਤਾਂ
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        ਲਈ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft);

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_sycl(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਵਾਪਸ ਬਫਰ->buft->iface.get_name == ggml_backend_sycl_ਬਫਰ_type_get_name;
}

ਸਥਿਰ ਖਾਲੀ
ggml_backend_sycl_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) ਕੋਸ਼ਿਸ਼ {
    ggml_backend_sycl_ਬਫਰ_context * ctx = ( ggml_backend_sycl_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_sycl_ਸੈੱਟ_device(ctx->device);

    ਮਿਟਾਓ ctx;
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_sycl_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_sycl_ਬਫਰ_context * ctx = ( ggml_backend_sycl_ਬਫਰ_context *)ਬਫਰ->context;
    ਵਾਪਸ ctx->dev_ptr;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ
ggml_backend_sycl_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                     ggml_tensor *tensor) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਬਫਰ_context *)ਬਫਰ->context;

    ਜੇ (tensor->view_src != ਨਿਰਾ) {
        ਪ੍ਰਤਿਜ੍ਞਾ(tensor->view_src->ਬਫਰ->buft == ਬਫਰ->buft);
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    ਜੇ (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        ਚੋਣ (tensor->type) {
            ਹਾਲਤ GGML_TYPE_Q4_0:
            ਹਾਲਤ GGML_TYPE_Q8_0:
            ਹਾਲਤ GGML_TYPE_Q2_K:
            ਹਾਲਤ GGML_TYPE_Q3_K:
            ਹਾਲਤ GGML_TYPE_Q4_K:
            ਹਾਲਤ GGML_TYPE_Q5_K:
            ਹਾਲਤ GGML_TYPE_Q6_K:
            ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਹਾਲਤ GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = ਨਵਾਂ ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                ਰੋਕੋ;
            }
            ਮੂਲ:
                ਰੋਕੋ;
        }
    }

    ਜੇ (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_ਮਾਪ = ggml_nbytes(tensor);
        size_t padded_ਮਾਪ = ggml_backend_buft_get_alloc_ਮਾਪ(ਬਫਰ->buft, tensor);

        ਜੇ (padded_ਮਾਪ > original_ਮਾਪ && tensor->view_src == ਖਾਲੀ_ਸੂਚਕ) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (ਅੱਖਰ *)tensor->ਅੰਕੜੇ + original_ਮਾਪ, 0,
                padded_ਮਾਪ - original_ਮਾਪ).wait()));
        }
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                ggml_tensor *tensor,
                                                ਅਚਲ ਖਾਲੀ *ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    ggml_backend_sycl_ਬਫਰ_context * ctx = ( ggml_backend_sycl_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_sycl_ਸੈੱਟ_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ਅਚਲ ਪੂਰਨ n_slots = g_ggml_sycl_upload_staging_slots;
    ਜੇ (n_slots > 0 && ctx->staging.ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
        ctx->staging.ਅੰਕੜੇ = sycl::malloc_host(n_slots * ctx->staging_ਥਾਂ_ਮਾਪ, *ctx->stream);
        ਜੇ (ctx->staging.ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.ਸੌਂਪੋ(n_slots, ਝੂਠ);
        }
    }
    ਜੇ (ctx->staging.ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ) {
        ਕਤਾਰ_ptr    stream    = ctx->stream;
        ਅੱਖਰ *       dst       = (ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ;
        ਅਚਲ ਅੱਖਰ * src       = (ਅਚਲ ਅੱਖਰ *) ਅੰਕੜੇ;
        size_t       ਬਚਿਆ = size;
        ਜਦੋਂ (ਬਚਿਆ > 0) {
            ਅਚਲ size_t chunk = std::min(ਬਚਿਆ, ctx->staging_ਥਾਂ_ਮਾਪ);
            ਅਚਲ ਪੂਰਨ    ਥਾਂ  = ctx->staging.ਅਗਲਾ;
            ctx->staging.ਅਗਲਾ = (ctx->staging.ਅਗਲਾ + 1) % (ਪੂਰਨ) ctx->staging.submitted.size();
            ਜੇ (ctx->staging.submitted[ਥਾਂ]) {
                ctx->staging.events[ਥਾਂ].wait_and_throw();
            }
            ਖਾਲੀ * stage = (ਅੱਖਰ *) ctx->staging.ਅੰਕੜੇ + ਥਾਂ * ctx->staging_ਥਾਂ_ਮਾਪ;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(stage, src, chunk);
            ctx->staging.events[ਥਾਂ] = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst, stage, chunk);
            ctx->staging.submitted[ਥਾਂ] = ਸੱਚ;
            src       += chunk;
            dst       += chunk;
            ਬਚਿਆ -= chunk;
        }
        ਵਾਪਸ;
    }

    ਸਵੈਚਾਲਿਤ stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_ਕਤਾਰ());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    ਅੱਖਰ * host_buf = (ਅੱਖਰ *) ਸਮ੍ਰਿਤੀ_ਦਿਓ(size);
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(host_buf, ਅੰਕੜੇ, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, host_buf, size).wait()));
    ਮੁਕਤ(host_buf);
#ਨਹੀਂਤਾਂ
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *) tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size).wait()));
#ਜੇ_ਅੰਤ
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                                ਅਚਲ ggml_tensor *tensor,
                                                ਖਾਲੀ *ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                                size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    ggml_backend_sycl_ਬਫਰ_context * ctx = ( ggml_backend_sycl_ਬਫਰ_context *)ਬਫਰ->context;

    ggml_sycl_ਸੈੱਟ_device(ctx->device);
    ਸਵੈਚਾਲਿਤ stream = dpct::dev_mgr::instance().get_device(ctx->device).default_ਕਤਾਰ();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, size)
            .wait()));
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_is_l0_discrete_gpu(ਪੂਰਨ device) {
    ਵਾਪਸ ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਖਾਲੀ memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, ਖਾਲੀ *ptr_dst,
                         ਅਚਲ ਖਾਲੀ *ptr_src, size_t size) {
    ਅੱਖਰ *host_buf = (ਅੱਖਰ *)ਸਮ੍ਰਿਤੀ_ਦਿਓ(size);
    ਜੇ (host_buf == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.ਸਮ੍ਰਿਤੀ_ਨਕਲ(host_buf, (ਅਚਲ ਅੱਖਰ *)ptr_src, size).wait();
    q_dst.ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)ptr_dst, host_buf, size).wait();
    ਮੁਕਤ(host_buf);
}

ਸਥਿਰ ਖਾਲੀ dev2dev_memcpy(ਪੂਰਨ device_dst, sycl::queue &q_dst, ਪੂਰਨ device_src, sycl::queue &q_src, ਖਾਲੀ *ptr_dst,
                    ਅਚਲ ਖਾਲੀ *ptr_src, size_t size) {

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ਜੇ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ਅਚਲ ਤਾਰਕਿਕ l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        ਜੇ (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            ਸਵੈਚਾਲਿਤ ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            ਸਵੈਚਾਲਿਤ ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_ਕਤਾਰ_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_ਕਤਾਰ_DESC, ਖਾਲੀ_ਸੂਚਕ, 0, 0,
                                            0, ZE_COMMAND_ਕਤਾਰ_ਬਹੁਲਕ_SYNCHRONOUS, ZE_COMMAND_ਕਤਾਰ_ਤਰਜੀਹ_ਸਾਧਾਰਨ};
            ze_command_list_handle_t cl;
            ze_ਨਤੀਜਾ_t r = ze_Command_List_Create_ਤੁਰੰਤ(ze_ctx, ze_dev, &cq_desc, &cl);
            ਜੇ (r == ZE_ਨਤੀਜਾ_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_ਸਮ੍ਰਿਤੀ_Copy(cl, ptr_dst, ptr_src, size, ਖਾਲੀ_ਸੂਚਕ, 0, ਖਾਲੀ_ਸੂਚਕ);
                zeCommandListDestroy(cl);
                ਜੇ (r == ZE_ਨਤੀਜਾ_SUCCESS) {
                    ਵਾਪਸ;
                }
            }
        }
    }
#ਜੇ_ਅੰਤ

    ਜੇ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        ਜੇ (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.ਸਮ੍ਰਿਤੀ_ਨਕਲ(ptr_dst, ptr_src, size).wait()));
            ਵਾਪਸ;
        }
    }

    // Host-staged copy
    ਜੇ(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } ਨਹੀਂਤਾਂ {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

ਸਥਿਰ ਤਾਰਕਿਕ
ggml_backend_sycl_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                    ਅਚਲ ggml_tensor *src,
                                    ggml_tensor *dst) ਕੋਸ਼ਿਸ਼ {
    ਤਾਰਕਿਕ is_cpy_supported = ggml_backend_ਬਫਰ_is_sycl(src->ਬਫਰ);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ਜੇ (is_cpy_supported) {
        ggml_backend_sycl_ਬਫਰ_context * src_ctx = (ggml_backend_sycl_ਬਫਰ_context *)src->ਬਫਰ->context;
        ggml_backend_sycl_ਬਫਰ_context * dst_ctx = (ggml_backend_sycl_ਬਫਰ_context *)dst->ਬਫਰ->context;

        ggml_sycl_ਸੈੱਟ_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_ਸੈੱਟ_device(dst_ctx->device);
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

        ਕਤਾਰ_ptr ਧਾਰਾ_dst = dst_ctx->stream;
        ਕਤਾਰ_ptr ਧਾਰਾ_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *ਧਾਰਾ_dst, src_ctx->device, *ਧਾਰਾ_src, dst->ਅੰਕੜੇ, src->ਅੰਕੜੇ, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#ਜੇ 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).ਸਮ੍ਰਿਤੀ_ਨਕਲ(
            (ਅੱਖਰ *)dst->ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *)src->ਅੰਕੜੇ, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#ਜੇ_ਅੰਤ
        ਵਾਪਸ ਸੱਚ;
    }
    ਵਾਪਸ ਝੂਠ;
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
} ਫੜੋ (ਅਚਲ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                           uint8_t value) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, ਬਫਰ->size);
    ggml_backend_sycl_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਬਫਰ_context *) ਬਫਰ->context;

    ggml_sycl_ਸੈੱਟ_device(ctx->device);
    ਕਤਾਰ_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    ਲਈ (size_t off = 0; off < ਬਫਰ->size; off += MAX_CHUNK) {
        size_t chunk = std::min(ਬਫਰ->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<ਅੱਖਰ*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਬਫਰ_memset_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor, uint8_t value,
                                                   size_t ਵਿੱਥ, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, ਵਿੱਥ, value);
    ggml_backend_sycl_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਬਫਰ_context *) ਬਫਰ->context;
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx->device));
    ਸਵੈਚਾਲਿਤ stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_ਕਤਾਰ());
    ਜੇ (size == 0) {
        ਵਾਪਸ;  // Nothing to do
    }
    ਜੇ (tensor->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    ਖਾਲੀ * ਨਿਸ਼ਾਨਾ_ptr = static_cast<ਅੱਖਰ *>(tensor->ਅੰਕੜੇ) + ਵਿੱਥ;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(ਨਿਸ਼ਾਨਾ_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਬਫਰ_reset(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ਜੇ (ਬਫਰ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    ggml_backend_sycl_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਬਫਰ_context *) ਬਫਰ->context;

    ਜੇ (ctx != ਖਾਲੀ_ਸੂਚਕ) {
        ਲਈ (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_i ggml_backend_sycl_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_sycl_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_sycl_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_ਬਫਰ_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_sycl_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_ਬਫਰ_clear,
    /* .reset           = */ ggml_backend_sycl_ਬਫਰ_reset,
};

// sycl buffer type
ਬਣਤਰ ggml_backend_sycl_ਬਫਰ_type_context {
    ਪੂਰਨ device;
    std::string name;

    // each buffer type has its own stream
    ਕਤਾਰ_ptr stream = ਖਾਲੀ_ਸੂਚਕ;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_sycl_ਬਫਰ_type_context * ctx = (ggml_backend_sycl_ਬਫਰ_type_context *)buft->context;

    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਤਾਰਕਿਕ check_usm_system(ਪੂਰਨ device, size_t size) {
    ਤਾਰਕਿਕ use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_ਮਾਪ_1_G);

    ਜੇ (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = ਝੂਠ;
    }

    ਵਾਪਸ use_usm_system;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ * aligned_malloc_host(size_t alignment, size_t size) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ _aligned_malloc(size, alignment);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ aligned_alloc(alignment, size);
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t
ggml_backend_sycl_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft,
                                           size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਬਫਰ_type_context *)buft->context;
    ggml_sycl_ਸੈੱਟ_device(buft_ctx->device);
    ਅਚਲ ਕਤਾਰ_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_ਮਾਪ_2_M;
    size_t aligned_ਮਾਪ = ((size + alignment - 1) / alignment) * alignment;
    ਤਾਰਕਿਕ use_usm_system = check_usm_system(buft_ctx->device, aligned_ਮਾਪ);

    ਖਾਲੀ * dev_ptr;
    ਜੇ (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (ਖਾਲੀ *)aligned_malloc_host(alignment, aligned_ਮਾਪ);
        ਜੇ (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    } ਨਹੀਂਤਾਂ {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (ਖਾਲੀ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_ਬਫਰ)));
        ਜੇ (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }
    ggml_backend_sycl_ਬਫਰ_context * ctx = ਨਵਾਂ  ggml_backend_sycl_ਬਫਰ_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_sycl_ਬਫਰ_interface, ctx, size);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ size_t ggml_backend_sycl_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ SYCL_ਬਫਰ_ALIGNMENT;
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ is_bmg_g31_arch(ਪੂਰਨ device) {
    ਵਾਪਸ ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

ਸਥਿਰ size_t ggml_backend_sycl_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ggml_backend_sycl_ਬਫਰ_type_context * ctx = (ggml_backend_sycl_ਬਫਰ_type_context *)buft->context;
    ਪੂਰਨ device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_ਮਾਪ = dpct::get_device(device).get_max_mem_alloc_ਮਾਪ();
    ਜੇ (g_ggml_sycl_host_pinned_mem_2g) {
        ਵਾਪਸ std::min(max_alloc_ਮਾਪ, (size_t) 2LL*1024*1024*1024);
    } ਨਹੀਂਤਾਂ {
        ਜੇ(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_ਮਾਪ = max_alloc_ਮਾਪ*0.6;
        }
        ਵਾਪਸ max_alloc_ਮਾਪ;

    }
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_sycl_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_ਮਾਪ(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    ਜੇ (ggml_is_quantized(tensor->type)) {
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            size += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }
    }

    ਵਾਪਸ size;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਅਚਲ ggml_backend_ਬਫਰ_type_i ggml_backend_sycl_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_sycl_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_sycl_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ਨਿਰਾ,
};

ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_ਬਫਰ_type(ਪੂਰਨ device) {
    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);


    ਸਵੈਚਾਲਿਤ dev_ਗਿਣਤੀ = ggml_backend_sycl_get_device_ਗਿਣਤੀ();

    ਜੇ (device>=dev_ਗਿਣਤੀ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_ਗਿਣਤੀ-1);
        GGML_ASSERT(device<dev_ਗਿਣਤੀ);
    }
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_sycl_ਬਫਰ_types[GGML_SYCL_MAX_DEVICES];

    ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_ਬਫਰ_type_initialized = ਝੂਠ;

    ਜੇ (!ggml_backend_sycl_ਬਫਰ_type_initialized) {
        ਲਈ (ਪੂਰਨ i = 0; i < dev_ਗਿਣਤੀ; i++) {
            ਸਵੈਚਾਲਿਤ & device_i = dpct::dev_mgr::instance().get_device(i);
            ਕਤਾਰ_ptr stream = &(device_i.default_ਕਤਾਰ());
            ggml_backend_sycl_ਬਫਰ_types[i] = {
                /* .iface    = */ ggml_backend_sycl_ਬਫਰ_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ ਨਵਾਂ ggml_backend_sycl_ਬਫਰ_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_ਬਫਰ_type_initialized = ਸੱਚ;
    }
    ਵਾਪਸ &ggml_backend_sycl_ਬਫਰ_types[device];
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_ਬਫਰ_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    ਪੂਰਨ device = ctx->device;
    ਜੇ (device>=ggml_sycl_info().device_ਗਿਣਤੀ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_ਗਿਣਤੀ-1);
        GGML_ASSERT(device<ggml_sycl_info().device_ਗਿਣਤੀ);
    }
    ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_type ggml_backend_sycl_ਬਫਰ_types[GGML_SYCL_MAX_DEVICES];

    ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_ਬਫਰ_type_initialized = ਝੂਠ;

    ਜੇ (!ggml_backend_sycl_ਬਫਰ_type_initialized) {
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; i++) {
            ggml_backend_sycl_ਬਫਰ_types[i] = {
                /* .iface    = */ ggml_backend_sycl_ਬਫਰ_type_interface,
                /* .device   = */ ਖਾਲੀ_ਸੂਚਕ,
                /* .context  = */ ਨਵਾਂ ggml_backend_sycl_ਬਫਰ_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_ਬਫਰ_type_initialized = ਸੱਚ;
    }
    ਵਾਪਸ &ggml_backend_sycl_ਬਫਰ_types[device];
}

// sycl split buffer

ਸਥਿਰ int64_t get_ਪੰਕਤੀ_rounding(ggml_type type, ਅਚਲ std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> & tensor_ਵੰਡੋ) {
    int64_t min_compute_capability = ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ;
    int64_t max_compute_capability = INT_MIN;
    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        ਜੇ (tensor_ਵੰਡੋ[i] < (i + 1 < ggml_sycl_info().device_ਗਿਣਤੀ ? tensor_ਵੰਡੋ[i + 1] : 1.0f)) {
            ਜੇ (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            ਜੇ (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    ਚੋਣ(type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
            ਵਾਪਸ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ 64;
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ 1;
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
        ਹਾਲਤ GGML_TYPE_IQ2_S:
        ਹਾਲਤ GGML_TYPE_IQ1_S:
        ਹਾਲਤ GGML_TYPE_IQ1_M:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            ਵਾਪਸ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਵਾਪਸ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਵਾਪਸ 64;
        ਮੂਲ:
            GGML_ABORT("fatal error");
    }
}

ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀ_ਵੰਡੋ(int64_t * ਪੰਕਤੀ_ਨੀਵਾਂ, int64_t * ਪੰਕਤੀ_ਉੱਚਾ, ਅਚਲ ggml_tensor * tensor, ਅਚਲ std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> & tensor_ਵੰਡੋ, ਪੂਰਨ id) {
    ਅਚਲ int64_t nrows = ggml_nrows(tensor);
    ਅਚਲ int64_t rounding = get_ਪੰਕਤੀ_rounding(tensor->type, tensor_ਵੰਡੋ);

    *ਪੰਕਤੀ_ਨੀਵਾਂ = id == 0 ? 0 : nrows*tensor_ਵੰਡੋ[id];
    *ਪੰਕਤੀ_ਨੀਵਾਂ -= *ਪੰਕਤੀ_ਨੀਵਾਂ % rounding;
    ਜੇ (id == ggml_sycl_info().device_ਗਿਣਤੀ - 1) {
        *ਪੰਕਤੀ_ਉੱਚਾ = nrows;
    } ਨਹੀਂਤਾਂ {
        *ਪੰਕਤੀ_ਉੱਚਾ = nrows*tensor_ਵੰਡੋ[id + 1];
        *ਪੰਕਤੀ_ਉੱਚਾ -= *ਪੰਕਤੀ_ਉੱਚਾ % rounding;
    }
}

ਸਥਿਰ size_t ggml_nbytes_ਵੰਡੋ(ਅਚਲ ਬਣਤਰ ggml_tensor * tensor, ਪੂਰਨ nrows_ਵੰਡੋ) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ਵਾਪਸ nrows_ਵੰਡੋ*ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, tensor->ne[0]);
}

ਬਣਤਰ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context {
    std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> tensor_ਵੰਡੋ;
};

ਬਣਤਰ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context {
    ~ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context() ਕੋਸ਼ਿਸ਼ {
        ਲਈ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    ਫੜੋ (sycl::exception ਅਚਲ &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
      std::ਬਾਹਰ_ਨਿਕਲੋ(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<ਕਤਾਰ_ptr> streams;
};

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context *)ਬਫਰ->context;
    ਮਿਟਾਓ ctx;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    ਵਾਪਸ (ਖਾਲੀ *)0x1000;

    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_ਸਥਿਤੀ
ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                           ggml_tensor *tensor) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == ਖਾਲੀ_ਸੂਚਕ); // views of split tensors are not supported

    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *)ਬਫਰ->buft->context;

    ਅਚਲ int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = ਨਵਾਂ ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    ਜੇ (ctx->streams.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_ਕਤਾਰ()));
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਪੰਕਤੀ_ਉੱਚਾ;
        get_ਪੰਕਤੀ_ਵੰਡੋ(&ਪੰਕਤੀ_ਨੀਵਾਂ, &ਪੰਕਤੀ_ਉੱਚਾ, tensor, buft_ctx->tensor_ਵੰਡੋ, i);

        int64_t nrows_ਵੰਡੋ = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;
        ਜੇ (nrows_ਵੰਡੋ == 0) {
            ਜਾਰੀ;
        }

        size_t size = ggml_nbytes_ਵੰਡੋ(tensor, nrows_ਵੰਡੋ);
        ਅਚਲ size_t original_ਮਾਪ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            size += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }

        ggml_sycl_ਸੈੱਟ_device(i);
        ਅਚਲ ਕਤਾਰ_ptr stream = ctx->streams[i];
        ਅੱਖਰ * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (ਅੱਖਰ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_ਬਫਰ)));
        ਜੇ (!buf) {
            ਅੱਖਰ err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            ਸੁੱਟੋ std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        ਜੇ (size > original_ਮਾਪ) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_ਮਾਪ, 0, size - original_ਮਾਪ)
                    .wait()));
        }

        extra->ਅੰਕੜੇ_device[i] = buf;

        ਲਈ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = ਨਵਾਂ sycl::event()));
        }
    }
    tensor->extra = extra;
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ
ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                          ggml_tensor *tensor, ਅਚਲ ਖਾਲੀ *ਅੰਕੜੇ,
                                          size_t ਵਿੱਥ, size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *)ਬਫਰ->buft->context;

    ਅਚਲ int64_t ne0 = tensor->ne[0];
    ਅਚਲ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਪੰਕਤੀ_ਉੱਚਾ;
        get_ਪੰਕਤੀ_ਵੰਡੋ(&ਪੰਕਤੀ_ਨੀਵਾਂ, &ਪੰਕਤੀ_ਉੱਚਾ, tensor, buft_ctx->tensor_ਵੰਡੋ, i);

        int64_t nrows_ਵੰਡੋ = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;
        ਜੇ (nrows_ਵੰਡੋ == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ size_t ਵਿੱਥ_ਵੰਡੋ = ਪੰਕਤੀ_ਨੀਵਾਂ*nb1;
        size_t size = ggml_nbytes_ਵੰਡੋ(tensor, nrows_ਵੰਡੋ);
        ਅਚਲ size_t original_ਮਾਪ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            size += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }

        ਅਚਲ ਅੱਖਰ * buf_host = (ਅਚਲ ਅੱਖਰ *)ਅੰਕੜੇ + ਵਿੱਥ_ਵੰਡੋ;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_ਸੈੱਟ_device(i);
        ਅਚਲ ਕਤਾਰ_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .ਸਮ੍ਰਿਤੀ_ਨਕਲ(extra->ਅੰਕੜੇ_device[i], buf_host, original_ਮਾਪ)
                .wait()));
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ
ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                          ਅਚਲ ggml_tensor *tensor, ਖਾਲੀ *ਅੰਕੜੇ,
                                          size_t ਵਿੱਥ, size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(ਵਿੱਥ == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context * ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context *)ਬਫਰ->context;
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *)ਬਫਰ->buft->context;

    ਅਚਲ int64_t ne0 = tensor->ne[0];
    ਅਚਲ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਪੰਕਤੀ_ਉੱਚਾ;
        get_ਪੰਕਤੀ_ਵੰਡੋ(&ਪੰਕਤੀ_ਨੀਵਾਂ, &ਪੰਕਤੀ_ਉੱਚਾ, tensor, buft_ctx->tensor_ਵੰਡੋ, i);

        int64_t nrows_ਵੰਡੋ = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;
        ਜੇ (nrows_ਵੰਡੋ == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ size_t ਵਿੱਥ_ਵੰਡੋ = ਪੰਕਤੀ_ਨੀਵਾਂ*nb1;
        size_t size = ggml_nbytes_ਵੰਡੋ(tensor, nrows_ਵੰਡੋ);
        ਅਚਲ size_t original_ਮਾਪ = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            size += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }

        ਅੱਖਰ * buf_host = (ਅੱਖਰ *)ਅੰਕੜੇ + ਵਿੱਥ_ਵੰਡੋ;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_ਸੈੱਟ_device(i);
        ਅਚਲ ਕਤਾਰ_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .ਸਮ੍ਰਿਤੀ_ਨਕਲ(buf_host, extra->ਅੰਕੜੇ_device[i], original_ਮਾਪ)
                .wait()));
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    GGML_ਅਣਵਰਤਿਆ(value);
}

ਸਥਿਰ ਬਣਤਰ ggml_backend_ਬਫਰ_i ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ਨਿਰਾ,
    /* .clear           = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

// sycl split buffer type

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ GGML_SYCL_NAME "_Split";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
   ਵਾਪਸ ਬਫਰ->buft->iface.get_name == ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_name;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context * ctx = ਨਵਾਂ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_context();

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_interface, ctx, size);
}

ਸਥਿਰ size_t ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ SYCL_ਬਫਰ_ALIGNMENT;
    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ size_t ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *)buft->context;

    size_t ਕੁੱਲ_ਮਾਪ = 0;

    ਅਚਲ int64_t ne0 = tensor->ne[0];

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਪੰਕਤੀ_ਉੱਚਾ;
        get_ਪੰਕਤੀ_ਵੰਡੋ(&ਪੰਕਤੀ_ਨੀਵਾਂ, &ਪੰਕਤੀ_ਉੱਚਾ, tensor, ctx->tensor_ਵੰਡੋ, i);

        int64_t nrows_ਵੰਡੋ = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;
        ਜੇ (nrows_ਵੰਡੋ == 0) {
            ਜਾਰੀ;
        }

        ਕੁੱਲ_ਮਾਪ += ggml_nbytes_ਵੰਡੋ(tensor, nrows_ਵੰਡੋ);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ਜੇ (ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING != 0) {
            ਕੁੱਲ_ਮਾਪ += ggml_ਪੰਕਤੀ_ਮਾਪ(tensor->type, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING - ne0 % ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
        }
    }

    ਵਾਪਸ ਕੁੱਲ_ਮਾਪ;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ ਝੂਠ;

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_i ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ਨਿਰਾ, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_is_host,
};

ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type(ਪੂਰਨ main_device, ਅਚਲ ਦਸ਼ਮਲਵ * tensor_ਵੰਡੋ) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_ਅਣਵਰਤਿਆ(main_device);

    ਸਥਿਰ std::mutex mutex;
    std::lock_guard<std::mutex> ਤਾਲਾ(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    ਸਥਿਰ std::map<std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES>, ਬਣਤਰ ggml_backend_ਬਫਰ_type> buft_map;

    std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> tensor_ਵੰਡੋ_arr = {};

    ਤਾਰਕਿਕ all_zero = tensor_ਵੰਡੋ == ਖਾਲੀ_ਸੂਚਕ || std::all_of(tensor_ਵੰਡੋ, tensor_ਵੰਡੋ + GGML_SYCL_MAX_DEVICES, [](ਦਸ਼ਮਲਵ x) { ਵਾਪਸ x == 0.0f; });
    ਜੇ (all_zero) {
        tensor_ਵੰਡੋ_arr = ggml_sycl_info().default_tensor_ਵੰਡੋ;
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ ਵੰਡੋ_ਜੋੜ = 0.0f;
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
            tensor_ਵੰਡੋ_arr[i] = ਵੰਡੋ_ਜੋੜ;
            ਵੰਡੋ_ਜੋੜ += tensor_ਵੰਡੋ[i];
        }
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
            tensor_ਵੰਡੋ_arr[i] /= ਵੰਡੋ_ਜੋੜ;
        }
    }

    ਸਵੈਚਾਲਿਤ it = buft_map.find(tensor_ਵੰਡੋ_arr);
    ਜੇ (it != buft_map.end()) {
        ਵਾਪਸ &it->second;
    }

    ਬਣਤਰ ggml_backend_ਬਫਰ_type buft {
        /* .iface   = */ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ ਨਵਾਂ ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context{tensor_ਵੰਡੋ_arr},
    };

    ਸਵੈਚਾਲਿਤ ਨਤੀਜਾ = buft_map.emplace(tensor_ਵੰਡੋ_arr, buft);
    ਵਾਪਸ &ਨਤੀਜਾ.ਪਹਿਲਾ->second;
}

// host buffer type

ਬਣਤਰ ggml_backend_sycl_device_context {
    ਪੂਰਨ device;
    std::string name;
    std::string description;
    ਪੂਰਨ op_offload_min_batch_ਮਾਪ;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_host_ਬਫਰ_type_name(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ GGML_SYCL_NAME "_Host";

    GGML_ਅਣਵਰਤਿਆ(buft);
}

ਸਥਿਰ ਪੂਰਨ ggml_backend_sycl_host_ਬਫਰ_type_device(ggml_backend_ਬਫਰ_type_t buft) {
    ਵਾਪਸ static_cast<ਅਚਲ ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
ਸਥਿਰ ਖਾਲੀ * ggml_backend_sycl_host_malloc(ਪੂਰਨ device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
    ਕੋਸ਼ਿਸ਼ {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        ਸਵੈਚਾਲਿਤ & q = dpct::dev_mgr::instance().get_device(device).default_ਕਤਾਰ();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } ਫੜੋ (...) {
        ptr = ਖਾਲੀ_ਸੂਚਕ;
    }
    ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    ਵਾਪਸ ptr;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_host_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ਜੇ (ਬਫਰ->context == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }
    ਜੇ (g_ggml_sycl_enable_host_pinned_mem) {
        ਅਚਲ ਪੂਰਨ device = ggml_backend_sycl_host_ਬਫਰ_type_device(ਬਫਰ->buft);
        ਸਵੈਚਾਲਿਤ & q = dpct::dev_mgr::instance().get_device(device).default_ਕਤਾਰ();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(ਬਫਰ->context, q)));
    } ਨਹੀਂਤਾਂ {
        free_aligned_mem_host((ਖਾਲੀ *) ਬਫਰ->context);
    }
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_sycl_host_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ਖਾਲੀ * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_ਬਫਰ_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    ਜੇ (ptr == ਖਾਲੀ_ਸੂਚਕ) {
        // fallback to cpu buffer
        ਵਾਪਸ ggml_backend_buft_alloc_ਬਫਰ(ggml_backend_cpu_ਬਫਰ_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_ਬਫਰ_t ਬਫਰ = ggml_backend_cpu_ਬਫਰ_from_ptr(ptr, size);
    ਬਫਰ->buft = buft;
    ਬਫਰ->iface.free_ਬਫਰ = ggml_backend_sycl_host_ਬਫਰ_free_ਬਫਰ;

    ਵਾਪਸ ਬਫਰ;
}

ਸਥਿਰ size_t ggml_backend_sycl_host_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {

    ਜੇ (g_ggml_sycl_enable_host_pinned_mem) {
        ਅਚਲ ਪੂਰਨ device = ggml_backend_sycl_host_ਬਫਰ_type_device(buft);
        size_t max_alloc_ਮਾਪ = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_ਮਾਪ();
        ਜੇ (g_ggml_sycl_host_pinned_mem_2g) {
            ਵਾਪਸ std::min(max_alloc_ਮਾਪ, (size_t) 2LL*1024*1024*1024);
        } ਨਹੀਂਤਾਂ {
            ਵਾਪਸ max_alloc_ਮਾਪ;
        }
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ SIZE_MAX;
    }
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_host_ਬਫਰ_type_for_device(ਪੂਰਨ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    ਸਥਿਰ std::vector<ggml_backend_ਬਫਰ_type> ਬਫਰ_types_host = [] {
        std::vector<ggml_backend_ਬਫਰ_type> bufts(ggml_backend_sycl_get_device_ਗਿਣਤੀ());
        ਲਈ (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_ਬਫਰ_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_ਬਫਰ_type_alloc_ਬਫਰ,
                    /* .alloc_buffer_n      = */ ਨਿਰਾ,
                    /* .get_alignment       = */ ggml_backend_cpu_ਬਫਰ_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_ਬਫਰ_type_get_max_ਮਾਪ,
                    /* .get_alloc_size      = */ ggml_backend_cpu_ਬਫਰ_type()->iface.get_alloc_ਮਾਪ,
                    /* .get_alloc_size_n    = */ ਨਿਰਾ,
                    /* .is_host             = */ ggml_backend_cpu_ਬਫਰ_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ ਖਾਲੀ_ਸੂਚਕ,
            };
        }
        ਵਾਪਸ bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (ਪੂਰਨ) ਬਫਰ_types_host.size());

    ਵਾਪਸ &ਬਫਰ_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_host_ਬਫਰ_type() {
    ਵਾਪਸ ggml_backend_sycl_host_ਬਫਰ_type_for_device(0);
}

// buffer pool for sycl (legacy)
ਬਣਤਰ ggml_sycl_pool_leg : ਜਨਤਕ ggml_sycl_pool {
    ਸਥਿਰ ਅਚਲ ਪੂਰਨ MAX_SYCL_BUFFERS = 256;

    ਪੂਰਨ device;
    ਕਤਾਰ_ptr qptr;
    ਬਣਤਰ ggml_sycl_ਬਫਰ {
        ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
        size_t size = 0;
    };

    ggml_sycl_ਬਫਰ ਬਫਰ_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_ਮਾਪ = 0;

    ਸਪਸ਼ਟ ggml_sycl_pool_leg(ਕਤਾਰ_ptr qptr_, ਪੂਰਨ device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_POOL
        ਪੂਰਨ    n_cached    = 0;
        size_t ਬਾਈਟਾਂ_cached = 0;
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ਜੇ (ਬਫਰ_pool[i].ptr != ਖਾਲੀ_ਸੂਚਕ) {
                ++n_cached;
                ਬਾਈਟਾਂ_cached += ਬਫਰ_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, ਬਾਈਟਾਂ_cached / 1024.0 / 1024.0);
        ਅਚਲ ਸਵੈਚਾਲਿਤ slots = format_slots_ਵਿੱਚ_alloc_ਕ੍ਰਮ();
        ਜੇ (!slots.ਖਾਲੀ_ਥਾਵਾਂ()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#ਜੇ_ਅੰਤ

        ਲਈ (ਪੂਰਨ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr != ਖਾਲੀ_ਸੂਚਕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_ਮਾਪ -= b.size;
            }
        }
        GGML_ASSERT(pool_ਮਾਪ == 0);
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_POOL
    std::string format_slots_ਵਿੱਚ_alloc_ਕ੍ਰਮ() ਅਚਲ {
        std::string ਸਤਰ;
        ਅੱਖਰ buf[32];
        ਤਾਰਕਿਕ ਪਹਿਲਾ = ਸੱਚ;
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ਜੇ (ਬਫਰ_pool[i].ptr == ਖਾਲੀ_ਸੂਚਕ) {
                ਜਾਰੀ;
            }
            ਜੇ (!ਪਹਿਲਾ) {
                ਸਤਰ += '/';
            }
            ਪਹਿਲਾ = ਝੂਠ;
            snprintf(buf, ਆਕਾਰ(buf), "%.2f", ਬਫਰ_pool[i].size / 1024.0 / 1024.0);
            ਸਤਰ += buf;
        }
        ਵਾਪਸ ਸਤਰ;
    }
#ਜੇ_ਅੰਤ

    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_sycl_MALLOC
        ਪੂਰਨ nnz = 0;
        size_t max_size = 0;
#ਜੇ_ਅੰਤ
        size_t ਉੱਤਮ_diff = 1ull << 36;
        ਪੂਰਨ ibest = -1;
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ਬਫਰ& b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr != ਖਾਲੀ_ਸੂਚਕ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_sycl_MALLOC
                ++nnz;
                ਜੇ (b.size > max_size) max_size = b.size;
#ਜੇ_ਅੰਤ
                ਜੇ (b.size >= size) {
                    size_t diff = b.size - size;
                    ਜੇ (diff < ਉੱਤਮ_diff) {
                        ਉੱਤਮ_diff = diff;
                        ibest = i;
                        ਜੇ (!ਉੱਤਮ_diff) {
                            ਖਾਲੀ * ptr = b.ptr;
                            *actual_ਮਾਪ = b.size;
                            b.ptr = ਖਾਲੀ_ਸੂਚਕ;
                            b.size = 0;
                            ਵਾਪਸ ptr;
                        }
                    }
                }
            }
        }
        ਜੇ (ibest >= 0) {
            ggml_sycl_ਬਫਰ& b = ਬਫਰ_pool[ibest];
            ਖਾਲੀ * ptr = b.ptr;
            *actual_ਮਾਪ = b.size;
            b.ptr = ਖਾਲੀ_ਸੂਚਕ;
            b.size = 0;
            ਵਾਪਸ ptr;
        }
        ਖਾਲੀ * ptr;
        size_t look_ahead_ਮਾਪ = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ਖਾਲੀ *)ggml_sycl_malloc_device(look_ahead_ਮਾਪ, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        ਜੇ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_ਮਾਪ);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        *actual_ਮਾਪ = look_ahead_ਮਾਪ;
        pool_ਮਾਪ += look_ahead_ਮਾਪ;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_ਮਾਪ[id]/1024/1024), (uint32_t)(size/1024/1024));
#ਜੇ_ਅੰਤ

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        ਵਾਪਸ ptr;
    }

    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_ਬਫਰ& b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr == ਖਾਲੀ_ਸੂਚਕ) {
                b.ptr = ptr;
                b.size = size;
                ਵਾਪਸ;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_ਮਾਪ -= size;
    }
};

// pool with virtual memory management
#ਜੇ defined(GGML_SYCL_SUPPORT_VMM)
ਬਣਤਰ ggml_sycl_pool_vmm : ਜਨਤਕ ggml_sycl_pool {
    ਸਥਿਰ ਅਚਲ size_t SYCL_POOL_VMM_MAX_ਮਾਪ = 1ull << 35; // 32 GB

    ਪੂਰਨ           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_ਵਰਤਿਆ = 0;
    size_t    pool_ਮਾਪ = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    ਬਣਤਰ mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        ਖਾਲੀ * map_ptr;
    };
    std::vector<mapping> mappings;

    ਸਪਸ਼ਟ ggml_sycl_pool_vmm(ਕਤਾਰ_ptr qptr_, ਪੂਰਨ device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        ਜੇ (pool_addr == 0) {
            ਵਾਪਸ;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        ਲਈ (ਸਵੈਚਾਲਿਤ & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_ਮਾਪ, ctx)));
    }

    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_ਬਫਰ_ALIGNMENT);

        size_t avail = pool_ਮਾਪ - pool_ਵਰਤਿਆ;

        ਜੇ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_ਮਾਪ = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_ਮਾਪ + reserve_ਮਾਪ <= SYCL_POOL_VMM_MAX_ਮਾਪ);

            ਜੇ (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ਟੈਗ " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_ਮਾਪ / (1024 * 1024), pool_ਮਾਪ / (1024 * 1024),
                              (pool_ਮਾਪ + reserve_ਮਾਪ) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_ਮਾਪ)));

            // reserve virtual address space (if not already reserved)
            ਜੇ (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_ਮਾਪ, ctx)));
            }

            // map at the end of the pool
            ਖਾਲੀ * map_ptr = ਖਾਲੀ_ਸੂਚਕ;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_ਮਾਪ, reserve_ਮਾਪ,
                                    sycl::ext::oneapi::experimental::ਪਤਾ_access_ਬਹੁਲਕ::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_ਮਾਪ += reserve_ਮਾਪ;
            ggml_sycl_memtrace_ਜੋੜੋ(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_ਮਾਪ);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) (pool_ਮਾਪ/1024/1024),
                          (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) (reserve_ਮਾਪ/1024/1024));
#ਜੇ_ਅੰਤ
        }

        GGML_ASSERT(pool_addr != 0);

        ਖਾਲੀ * ptr = reinterpret_cast<ਖਾਲੀ *>(pool_addr + pool_ਵਰਤਿਆ);
        *actual_ਮਾਪ = size;
        pool_ਵਰਤਿਆ += size;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) size, ptr);
#ਜੇ_ਅੰਤ

        ਵਾਪਸ ptr;
    }

    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ) size, ptr);
#ਜੇ_ਅੰਤ

        pool_ਵਰਤਿਆ -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<ਖਾਲੀ *>(pool_addr + pool_ਵਰਤਿਆ));
    }
};
#ਜੇ_ਅੰਤ // defined(GGML_SYCL_SUPPORT_VMM)

ਬਣਤਰ ggml_sycl_pool_host : ਜਨਤਕ ggml_sycl_pool {
    ਕਤਾਰ_ptr qptr;
    ਪੂਰਨ       device;

    ਬਣਤਰ ggml_sycl_ਬਫਰ {
        ਖਾਲੀ * ptr  = ਖਾਲੀ_ਸੂਚਕ;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    ਸਥਿਰ constexpr ਪੂਰਨ          MAX_POOL_ਮਾਪ{ 64 };
    std::vector<ggml_sycl_ਬਫਰ> ਬਫਰ_pool = std::vector<ggml_sycl_ਬਫਰ>(MAX_POOL_ਮਾਪ);
    size_t                        pool_ਮਾਪ   = 0;
    ਪੂਰਨ                           counter     = 0;

    ਸਪਸ਼ਟ ggml_sycl_pool_host(ਕਤਾਰ_ptr qptr_, ਪੂਰਨ device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_POOL_ਮਾਪ; ++i) {
            ggml_sycl_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr != ਖਾਲੀ_ਸੂਚਕ) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(b.ptr, *qptr)));
                b.ptr = ਖਾਲੀ_ਸੂਚਕ;
                pool_ਮਾਪ -= b.size;
                b.size = 0;
            }
        }
    }

    ਖਾਲੀ * alloc(size_t size, size_t * actual_ਮਾਪ) override {
        ਜੇ (counter == MAX_POOL_ਮਾਪ) {
            // the pool wrapped around, no slot is reserved for this allocation
            ਖਾਲੀ * ptr = ਖਾਲੀ_ਸੂਚਕ;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ਖਾਲੀ *) sycl::malloc_host(size, *qptr)));
            ਜੇ (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            }
            pool_ਮਾਪ    += size;
            *actual_ਮਾਪ = size;
            counter      = 0;
            ਵਾਪਸ ptr;
        }
        ggml_sycl_ਬਫਰ & b = ਬਫਰ_pool[counter];
        ++counter;

        ਜੇ (b.ptr != ਖਾਲੀ_ਸੂਚਕ) {
            ਜੇ (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                ਖਾਲੀ * ptr = b.ptr;
                *actual_ਮਾਪ = b.size;
                b.ptr = ਖਾਲੀ_ਸੂਚਕ;
                b.size = 0;
                ਵਾਪਸ ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(b.ptr, *qptr)));
            pool_ਮਾਪ -= b.size;
            b.ptr = ਖਾਲੀ_ਸੂਚਕ;
            b.size = 0;
        }

        ਖਾਲੀ * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ਖਾਲੀ *) sycl::malloc_host(size, *qptr)));
        ਜੇ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
        pool_ਮਾਪ    += size;
        *actual_ਮਾਪ = size;
        ਵਾਪਸ ptr;
    }

    ਖਾਲੀ ਮੁਕਤ(ਖਾਲੀ * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        ਲਈ (ਪੂਰਨ i = 0; i < MAX_POOL_ਮਾਪ; ++i) {
            ggml_sycl_ਬਫਰ & b = ਬਫਰ_pool[i];
            ਜੇ (b.ptr == ਖਾਲੀ_ਸੂਚਕ) {
                b.ptr  = ptr;
                b.size = size;
                ਵਾਪਸ;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ਮੁਕਤ(ptr, *qptr)));
        pool_ਮਾਪ -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(ਕਤਾਰ_ptr qptr, ਪੂਰਨ device) {
    // return pool for the host to speed up memory management
    ਵਾਪਸ std::unique_ptr<ggml_sycl_pool>(ਨਵਾਂ ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(ਕਤਾਰ_ptr qptr, ਪੂਰਨ device) {
#ਜੇ defined(GGML_SYCL_SUPPORT_VMM)
    ਜੇ (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        ਵਾਪਸ std::unique_ptr<ggml_sycl_pool>(ਨਵਾਂ ggml_sycl_pool_vmm(qptr, device));
    }
#ਜੇ_ਅੰਤ // defined(GGML_SYCL_SUPPORT_VMM)
    ਵਾਪਸ std::unique_ptr<ggml_sycl_pool>(ਨਵਾਂ ggml_sycl_pool_leg(qptr, device));
}

/// kernels
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ ਕਤਾਰ_ptr &stream);



ਸਥਿਰ ਖਾਲੀ mul_mat_p021_f16_f32(
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ y, ਦਸ਼ਮਲਵ * __restrict__ dst,
    ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ nchannels_x, ਅਚਲ ਪੂਰਨ nchannels_y,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ sycl::half *x = (ਅਚਲ sycl::half *)vx;

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_x = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                      ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ channel = ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                        ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ ਪੂਰਨ channel_x = channel / (nchannels_y / nchannels_x);

    ਅਚਲ ਪੂਰਨ nrows_y = ncols_x;
    ਅਚਲ ਪੂਰਨ nrows_dst = nrows_x;
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst = ਪੰਕਤੀ_x;

    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਲਈ (ਪੂਰਨ ਖਾਨਾ_x0 = 0; ਖਾਨਾ_x0 < ncols_x;
         ਖਾਨਾ_x0 += ਵਸਤੂ_ct1.get_local_range(2)) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ_x = ਖਾਨਾ_x0 + ਵਸਤੂ_ct1.get_local_id(2);

        ਜੇ (ਖਾਨਾ_x >= ncols_x) {
            ਰੋਕੋ;
        }

        // x is transposed and permuted
        ਅਚਲ ਪੂਰਨ ix = ਪੰਕਤੀ_x*nchannels_x*ncols_x + channel_x*ncols_x + ਖਾਨਾ_x;
        ਅਚਲ ਦਸ਼ਮਲਵ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];

        ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_y = ਖਾਨਾ_x;


        // y is not transposed but permuted
        ਅਚਲ ਪੂਰਨ iy = channel*nrows_y + ਪੰਕਤੀ_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ਅਚਲ ਪੂਰਨ idst = channel*nrows_dst + ਪੰਕਤੀ_dst;

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ਅਚਲ ਖਾਲੀ * __restrict__ vx, ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ y, ਦਸ਼ਮਲਵ * __restrict__ dst, ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x,
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_stride_x, ਅਚਲ ਪੂਰਨ channel_stride_x,ਅਚਲ ਪੂਰਨ channel_stride_y, ਅਚਲ ਪੂਰਨ channel_x_ਭਾਜਕ,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ sycl::half *x = (ਅਚਲ sycl::half *)vx;

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_x = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                      ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ channel = ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) +
                        ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ ਪੂਰਨ channel_x = channel / channel_x_ਭਾਜਕ;

    ਅਚਲ ਪੂਰਨ nrows_dst = nrows_x;
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_dst   = ਪੰਕਤੀ_x;

    ਅਚਲ ਪੂਰਨ idst = channel*nrows_dst + ਪੰਕਤੀ_dst;

    ਦਸ਼ਮਲਵ tmp = 0.0f;

    ਲਈ (ਪੂਰਨ ਖਾਨਾ_x0 = 0; ਖਾਨਾ_x0 < ncols_x;
         ਖਾਨਾ_x0 += ਵਸਤੂ_ct1.get_local_range(2)) {
        ਅਚਲ ਪੂਰਨ ਖਾਨਾ_x = ਖਾਨਾ_x0 + ਵਸਤੂ_ct1.get_local_id(2);

        ਜੇ (ਖਾਨਾ_x >= ncols_x) {
            ਰੋਕੋ;
        }

        ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_y = ਖਾਨਾ_x;

        ਅਚਲ ਪੂਰਨ ix = channel_x*channel_stride_x + ਪੰਕਤੀ_x*ਪੰਕਤੀ_stride_x + ਖਾਨਾ_x;
        ਅਚਲ ਪੂਰਨ iy = channel * channel_stride_y + ਪੰਕਤੀ_y;

        ਅਚਲ ਦਸ਼ਮਲਵ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ ਨਕਾਬ = WARP_ਮਾਪ / 2; ਨਕਾਬ > 0; ਨਕਾਬ >>= 1) {
        tmp +=
            dpct::ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(ਵਸਤੂ_ct1.get_sub_ਸਮੂਹ(), tmp, ਨਕਾਬ);
    }

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ਸਥਿਰ ਖਾਲੀ k_ਜੋੜ_ਪੰਕਤੀਆਂ_f32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols,
                           ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = ਵਸਤੂ_ct1.get_local_id(2);

    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
    ਲਈ (ਪੂਰਨ i = ਖਾਨਾ; i < ncols; i += ਵਸਤੂ_ct1.get_local_range(2)) {
        ਜੋੜ += x[ਪੰਕਤੀ * ncols + i];
    }

    ਜੋੜ = warp_reduce_ਜੋੜ(ਜੋੜ, ਵਸਤੂ_ct1);

    ਜੇ (ਖਾਨਾ == 0) {
        dst[ਪੰਕਤੀ] = ਜੋੜ;
    }
}


ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_ਅਦਲਾ_ਬਦਲੀ(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

ਸਾਂਚਾ <ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
k_argsort_f32_i32(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਪੂਰਨ *dst, ਅਚਲ ਪੂਰਨ ncols, ਪੂਰਨ ncols_pad,
                  ਅਚਲ ਪੂਰਨ tasks_ਪ੍ਰਤੀ_thread, ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    ਪੂਰਨ ਖਾਨਾ_ਸੂਚਕਾਂਕ =  ਵਸਤੂ_ct1.get_local_id(2);
    ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);

    ਲਈ (ਪੂਰਨ i = 0; i < tasks_ਪ੍ਰਤੀ_thread; i++) {
        ਪੂਰਨ ਖਾਨਾ = ਖਾਨਾ_ਸੂਚਕਾਂਕ * tasks_ਪ੍ਰਤੀ_thread + i;
        ਜੇ (ਖਾਨਾ >= ncols_pad) {
            ਵਾਪਸ;
        }
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * x_ਪੰਕਤੀ = x + ਪੰਕਤੀ * ncols;
    ਸਵੈਚਾਲਿਤ dst_ਪੰਕਤੀ = (ਪੂਰਨ *)dpct_local;

    // initialize indices
    ਲਈ (ਪੂਰਨ i=0;i<tasks_ਪ੍ਰਤੀ_thread;i++){
        ਪੂਰਨ ਖਾਨਾ = ਖਾਨਾ_ਸੂਚਕਾਂਕ*tasks_ਪ੍ਰਤੀ_thread+i;
        dst_ਪੰਕਤੀ[ਖਾਨਾ] = ਖਾਨਾ;
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਲਈ (ਪੂਰਨ k = 2; k <= ncols_pad; k *= 2) {
        ਲਈ (ਪੂਰਨ j = k / 2; j > 0; j /= 2) {
            ਲਈ (ਪੂਰਨ i = 0; i < tasks_ਪ੍ਰਤੀ_thread; i++) {
                ਪੂਰਨ ਖਾਨਾ = ਖਾਨਾ_ਸੂਚਕਾਂਕ * tasks_ਪ੍ਰਤੀ_thread + i;
                ਪੂਰਨ ixj = ਖਾਨਾ ^ j;
                ਜੇ (ixj > ਖਾਨਾ) {
                    ਜੇ ((ਖਾਨਾ & k) == 0) {
                        ਜੇ (dst_ਪੰਕਤੀ[ਖਾਨਾ] >= ncols ||
                            (dst_ਪੰਕਤੀ[ixj] < ncols &&
                             (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC
                                  ? x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ਖਾਨਾ]] > x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ixj]]
                                  : x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ਖਾਨਾ]] <
                                        x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ixj]]))) {
                            ggml_sycl_ਅਦਲਾ_ਬਦਲੀ(dst_ਪੰਕਤੀ[ਖਾਨਾ], dst_ਪੰਕਤੀ[ixj]);
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਜੇ (dst_ਪੰਕਤੀ[ixj] >= ncols ||
                            (dst_ਪੰਕਤੀ[ਖਾਨਾ] < ncols &&
                             (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC
                                  ? x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ਖਾਨਾ]] < x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ixj]]
                                  : x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ਖਾਨਾ]] >
                                        x_ਪੰਕਤੀ[dst_ਪੰਕਤੀ[ixj]]))) {
                            ggml_sycl_ਅਦਲਾ_ਬਦਲੀ(dst_ਪੰਕਤੀ[ਖਾਨਾ], dst_ਪੰਕਤੀ[ixj]);
                        }
                    }
                }
                ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    ਲਈ (ਪੂਰਨ i = 0; i < tasks_ਪ੍ਰਤੀ_thread; i++) {
        ਪੂਰਨ ਖਾਨਾ = ਖਾਨਾ_ਸੂਚਕਾਂਕ * tasks_ਪ੍ਰਤੀ_thread + i;
        ਜੇ (ਖਾਨਾ < ncols) {
            dst[ਪੰਕਤੀ * ncols + ਖਾਨਾ] = dst_ਪੰਕਤੀ[ਖਾਨਾ];
        }
    }
}

ਸਥਿਰ ਖਾਲੀ diag_ਨਕਾਬ_inf_f32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_channel, ਅਚਲ ਪੂਰਨ n_past,
                              ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                    ਵਸਤੂ_ct1.get_local_id(2);

    ਜੇ (ਖਾਨਾ >= ncols) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i = ਪੰਕਤੀ*ncols + ਖਾਨਾ;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (ਖਾਨਾ > n_past + ਪੰਕਤੀ % ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_channel) * FLT_MAX;
}

ਸਥਿਰ ਖਾਲੀ scale_f32(ਅਚਲ ਦਸ਼ਮਲਵ * x, ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ bias, ਅਚਲ ਪੂਰਨ k,
                      ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) +
                  ਵਸਤੂ_ct1.get_local_id(2);

    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    dst[i] = scale * x[i] + bias;
}


ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_p021_f16_f32_sycl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y,
                                           ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
                                           ਅਚਲ ਪੂਰਨ nrows_x,
                                           ਅਚਲ ਪੂਰਨ nchannels_x,
                                           ਅਚਲ ਪੂਰਨ nchannels_y,
                                           ਕਤਾਰ_ptr stream) {

    ਅਚਲ sycl::range<3> ਖੰਡ_nums(nchannels_y, nrows_x, 1);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, ਵਸਤੂ_ct1);
            });
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_mul_mat_vec_nc_f16_f32_sycl(
    ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ ਦਸ਼ਮਲਵ *y, ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols_x,
    ਅਚਲ ਪੂਰਨ nrows_x, ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ_stride_x, ਅਚਲ ਪੂਰਨ nchannels_x,
    ਅਚਲ ਪੂਰਨ nchannels_y, ਅਚਲ ਪੂਰਨ channel_stride_x, ਅਚਲ ਪੂਰਨ channel_stride_y, ਕਤਾਰ_ptr stream) {

    ਅਚਲ sycl::range<3> ਖੰਡ_nums(nchannels_y, nrows_x, 1);
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       ਪੰਕਤੀ_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, ਵਸਤੂ_ct1);
            });
    }
}



ਸਥਿਰ ਖਾਲੀ scale_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਦਸ਼ਮਲਵ scale, ਅਚਲ ਦਸ਼ਮਲਵ bias,
                           ਅਚਲ ਪੂਰਨ k, ਕਤਾਰ_ptr stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (k + SYCL_SCALE_ਖੰਡ_ਮਾਪ - 1) / SYCL_SCALE_ਖੰਡ_ਮਾਪ;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ਅੰਕ_ਖੰਡਾਂ) *
                              sycl::range<3>(1, 1, SYCL_SCALE_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_SCALE_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            scale_f32(x, dst, scale, bias, k, ਵਸਤੂ_ct1);
        });
}


ਸਥਿਰ ਖਾਲੀ ਜੋੜ_ਪੰਕਤੀਆਂ_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਦਸ਼ਮਲਵ *dst, ਅਚਲ ਪੂਰਨ ncols,
                              ਅਚਲ ਪੂਰਨ nrows, ਕਤਾਰ_ptr stream) {
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, WARP_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1)
                             [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
                                 k_ਜੋੜ_ਪੰਕਤੀਆਂ_f32(x, dst, ncols, ਵਸਤੂ_ct1);
                             });
}

ਸਥਿਰ ਪੂਰਨ ਅਗਲਾ_power_ਦਾ_2(ਪੂਰਨ x) {
    ਪੂਰਨ n = 1;
    ਜਦੋਂ (n < x) {
        n *= 2;
    }
    ਵਾਪਸ n;
}

ਸਥਿਰ ਖਾਲੀ init_argsort_indices_padded(
        ਪੂਰਨ * idx,
        ਅਚਲ ਪੂਰਨ nrows,
        ਅਚਲ ਪੂਰਨ ncols_pad,
        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1) {
    ਅਚਲ size_t gid = ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) + ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ size_t ਕੁੱਲ = (size_t) nrows * (size_t) ncols_pad;

    ਜੇ (gid >= ਕੁੱਲ) {
        ਵਾਪਸ;
    }

    idx[gid] = (ਪੂਰਨ) (gid % (size_t) ncols_pad);
}

ਸਾਂਚਾ <ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ>
ਸਥਿਰ ਖਾਲੀ argsort_f32_i32_global_pass(ਅਚਲ ਦਸ਼ਮਲਵ *            x,
                                        ਪੂਰਨ *                    idx,
                                        ਅਚਲ ਪੂਰਨ                ncols,
                                        ਅਚਲ ਪੂਰਨ                nrows,
                                        ਅਚਲ ਪੂਰਨ                ncols_pad,
                                        ਅਚਲ ਪੂਰਨ                j,
                                        ਅਚਲ ਪੂਰਨ                k,
                                        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1) {
    ਅਚਲ size_t gid   = ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) + ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ size_t ਕੁੱਲ = (size_t) nrows * (size_t) ncols_pad;

    ਜੇ (gid >= ਕੁੱਲ) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ) (gid / (size_t) ncols_pad);
    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = (ਪੂਰਨ) (gid % (size_t) ncols_pad);
    ਅਚਲ ਪੂਰਨ ixj = ਖਾਨਾ ^ j;

    ਜੇ (ixj <= ਖਾਨਾ || ixj >= ncols_pad) {
        ਵਾਪਸ;
    }

    ਅਚਲ size_t ਅਧਾਰ  = (size_t) ਪੰਕਤੀ * (size_t) ncols_pad;
    ਅਚਲ size_t pos_a = ਅਧਾਰ + (size_t) ਖਾਨਾ;
    ਅਚਲ size_t pos_b = ਅਧਾਰ + (size_t) ixj;

    ਅਚਲ ਪੂਰਨ a = idx[pos_a];
    ਅਚਲ ਪੂਰਨ b = idx[pos_b];

    ਤਾਰਕਿਕ do_ਅਦਲਾ_ਬਦਲੀ = ਝੂਠ;

    ਜੇ ((ਖਾਨਾ & k) == 0) {
        ਜੇ (a >= ncols ||
            (b < ncols &&
             (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC ?
                  x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) a] > x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) b] :
                  x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) a] < x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) b]))) {
            do_ਅਦਲਾ_ਬਦਲੀ = ਸੱਚ;
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (b >= ncols ||
            (a < ncols &&
             (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC ?
                  x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) a] < x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) b] :
                  x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) a] > x[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) b]))) {
            do_ਅਦਲਾ_ਬਦਲੀ = ਸੱਚ;
        }
    }

    ਜੇ (do_ਅਦਲਾ_ਬਦਲੀ) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

ਸਥਿਰ ਖਾਲੀ copy_argsort_indices_unpadded(ਅਚਲ ਪੂਰਨ *              idx_padded,
                                          ਪੂਰਨ *                    dst,
                                          ਅਚਲ ਪੂਰਨ                nrows,
                                          ਅਚਲ ਪੂਰਨ                ncols,
                                          ਅਚਲ ਪੂਰਨ                ncols_pad,
                                          ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1) {
    ਅਚਲ size_t gid   = ਵਸਤੂ_ct1.get_local_range(0) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) + ਵਸਤੂ_ct1.get_local_id(0);
    ਅਚਲ size_t ਕੁੱਲ = (size_t) nrows * (size_t) ncols;

    ਜੇ (gid >= ਕੁੱਲ) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = (ਪੂਰਨ) (gid / (size_t) ncols);
    ਅਚਲ ਪੂਰਨ ਖਾਨਾ = (ਪੂਰਨ) (gid % (size_t) ncols);

    dst[(size_t) ਪੰਕਤੀ * (size_t) ncols + (size_t) ਖਾਨਾ] = idx_padded[(size_t) ਪੰਕਤੀ * (size_t) ncols_pad + (size_t) ਖਾਨਾ];
}

ਸਥਿਰ ਖਾਲੀ argsort_f32_i32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਪੂਰਨ *dst, ਅਚਲ ਪੂਰਨ ncols,
                                 ਅਚਲ ਪੂਰਨ nrows, ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ,
                                 ਕਤਾਰ_ptr stream, ਪੂਰਨ device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ਅਚਲ ਪੂਰਨ ncols_pad = ਅਗਲਾ_power_ਦਾ_2(ncols);
    ਅਚਲ size_t shared_mem = (size_t) ncols_pad * ਆਕਾਰ(ਪੂਰਨ);
    ਅਚਲ size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    ਜੇ (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<ਪੂਰਨ> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        ਪੂਰਨ *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t ਖੰਡ_ਮਾਪ_3     = 256;
        ਅਚਲ size_t     ਕੁੱਲ_padded   = (size_t) nrows * (size_t) ncols_pad;
        ਅਚਲ size_t     nblocks_padded = (ਕੁੱਲ_padded + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, ਵਸਤੂ_ct1); });

        ਲਈ (ਪੂਰਨ k = 2; k <= ncols_pad; k *= 2) {
            ਲਈ (ਪੂਰਨ j = k / 2; j > 0; j /= 2) {
                ਜੇ (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
                        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                            argsort_f32_i32_global_pass<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, ਵਸਤੂ_ct1);
                        });
                } ਨਹੀਂਤਾਂ ਜੇ (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
                        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                            argsort_f32_i32_global_pass<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, ਵਸਤੂ_ct1);
                        });
                } ਨਹੀਂਤਾਂ {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ਅਚਲ size_t ਕੁੱਲ   = (size_t) nrows * (size_t) ncols;
        ਅਚਲ size_t nblocks = (ਕੁੱਲ + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
                             [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, ਵਸਤੂ_ct1);
                             });

        ਵਾਪਸ;
    }

    ਪੂਰਨ nth = 1;
    ਪੂਰਨ max_ਖੰਡ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[device];
    ਜਦੋਂ (nth < ncols_pad && nth < max_ਖੰਡ_ਮਾਪ)
        nth *= 2;
    ਜੇ (nth > max_ਖੰਡ_ਮਾਪ)
        nth = max_ਖੰਡ_ਮਾਪ;

    ਅਚਲ ਪੂਰਨ tasks_ਪ੍ਰਤੀ_thread = ncols_pad / nth;

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, nth);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, nrows, 1);

    ਜੇ (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    k_argsort_f32_i32<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC>(
                        x, dst, ncols, ncols_pad, tasks_ਪ੍ਰਤੀ_thread, ਵਸਤੂ_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ਨਹੀਂਤਾਂ ਜੇ (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    k_argsort_f32_i32<GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC>(
                        x, dst, ncols, ncols_pad, tasks_ਪ੍ਰਤੀ_thread, ਵਸਤੂ_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ਨਹੀਂਤਾਂ {
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
ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_ਜਾਂਚ_ਮਿਲਾਓ_f32(
    ਅਚਲ ਦਸ਼ਮਲਵ *   src_vals,
    ਅਚਲ int32_t * src_map,
    ਅਚਲ ਪੂਰਨ       begin,
    ਅਚਲ ਪੂਰਨ       end,
    ਅਚਲ ਪੂਰਨ       k,
    ਅਚਲ ਪੂਰਨ       ਖੰਡ_ਮਾਪ_3,
    ਦਸ਼ਮਲਵ *         shared_vals,
    ਪੂਰਨ *           shared_idx,
    ਦਸ਼ਮਲਵ *         ਬਾਹਰਲਾ_vals,
    int32_t *       ਬਾਹਰਲਾ_idx,
    ਅਚਲ ਤਾਰਕਿਕ      swap01,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1
) {
    ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    ਦਸ਼ਮਲਵ * lv = shared_vals + tid;
    ਪੂਰਨ * li = shared_idx + tid;

    ਲਈ (ਪੂਰਨ i = 0; i < k; i++) {
        lv[i * ਖੰਡ_ਮਾਪ_3] = -FLT_MAX;
        li[i * ਖੰਡ_ਮਾਪ_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    ਦਸ਼ਮਲਵ kth = -FLT_MAX;

    ਲਈ (ਪੂਰਨ ਖਾਨਾ = begin + tid; ਖਾਨਾ < end; ਖਾਨਾ += ਖੰਡ_ਮਾਪ_3) {
        ਦਸ਼ਮਲਵ val = src_vals[ਖਾਨਾ];

        ਜੇ (val > kth) {
            ਪੂਰਨ pos = k - 1;
            ਜਦੋਂ (pos > 0 && val > lv[(pos - 1) * ਖੰਡ_ਮਾਪ_3]) {
                pos--;
            }

            ਲਈ (ਪੂਰਨ i = k - 1; i > pos; i--) {
                lv[i * ਖੰਡ_ਮਾਪ_3] = lv[(i - 1) * ਖੰਡ_ਮਾਪ_3];
                li[i * ਖੰਡ_ਮਾਪ_3] = li[(i - 1) * ਖੰਡ_ਮਾਪ_3];
            }
            lv[pos * ਖੰਡ_ਮਾਪ_3] = val;
            li[pos * ਖੰਡ_ਮਾਪ_3] = src_map ? src_map[ਖਾਨਾ] : ਖਾਨਾ;

            kth = lv[(k - 1) * ਖੰਡ_ਮਾਪ_3];
        }
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    ਜੇ (tid != 0) {
        ਵਾਪਸ;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    ਦਸ਼ਮਲਵ * fv = shared_vals + (size_t) k * ਖੰਡ_ਮਾਪ_3;
    ਪੂਰਨ * fi = shared_idx + (size_t) k * ਖੰਡ_ਮਾਪ_3;

    ਲਈ (ਪੂਰਨ i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    ਦਸ਼ਮਲਵ fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    ਲਈ (ਪੂਰਨ t = 0; t < ਖੰਡ_ਮਾਪ_3; t++) {
        ਲਈ (ਪੂਰਨ i = 0; i < k; i++) {
            ਦਸ਼ਮਲਵ val = shared_vals[i * ਖੰਡ_ਮਾਪ_3 + t];

            ਜੇ (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                ਰੋਕੋ;
            }

            ਪੂਰਨ idx = shared_idx[i * ਖੰਡ_ਮਾਪ_3 + t];

            ਪੂਰਨ pos = k - 1;
            ਜਦੋਂ (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            ਲਈ (ਪੂਰਨ j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    ਜੇ (ਬਾਹਰਲਾ_vals) {
        ਲਈ (ਪੂਰਨ i = 0; i < k; i++) {
            ਬਾਹਰਲਾ_vals[i] = fv[i];
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < k; i++) {
        ਬਾਹਰਲਾ_idx[i] = fi[i];
    }

    ਜੇ (swap01 && k > 1) {
        int32_t ਆਰਜ਼ੀ = ਬਾਹਰਲਾ_idx[0];
        ਬਾਹਰਲਾ_idx[0] = ਬਾਹਰਲਾ_idx[1];
        ਬਾਹਰਲਾ_idx[1] = ਆਰਜ਼ੀ;
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ਦਸ਼ਮਲਵ * src,
    int32_t * dst_indices,
    ਅਚਲ int64_t ncols,
    ਅਚਲ int64_t nrows,
    ਅਚਲ ਪੂਰਨ k,
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ
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
    constexpr ਪੂਰਨ ਵੰਡੋ_ਖੰਡ = 128;
    constexpr ਪੂਰਨ max_splits  = 128;
    constexpr ਪੂਰਨ min_ਖਾਨੇ_ਗਿਣਤੀ    = 8192;

    ਪੂਰਨ nsplit = 1;
    ਜੇ (ncols >= min_ਖਾਨੇ_ਗਿਣਤੀ) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ਅਚਲ int64_t want = ncols / ਵੰਡੋ_ਖੰਡ;
        nsplit = (ਪੂਰਨ) (want > max_splits ? max_splits : want);
    }

    ਜੇ (nsplit > 1) {
        ਅਚਲ ਪੂਰਨ nchunk = (ਪੂਰਨ) ((ncols + nsplit - 1) / nsplit);
        ਅਚਲ size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        ਦਸ਼ਮਲਵ *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ਅਚਲ sycl::range<1> ਖੰਡ_dims(ਵੰਡੋ_ਖੰਡ);

        main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_vals(sycl::range<1>((ਵੰਡੋ_ਖੰਡ + 1) * k), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> shared_idx(sycl::range<1>((ਵੰਡੋ_ਖੰਡ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                    ਅਚਲ ਪੂਰਨ grp = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
                    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = grp / nsplit;
                    ਅਚਲ ਪੂਰਨ part = grp % nsplit;

                    ਅਚਲ ਪੂਰਨ begin = part * nchunk;
                    ਪੂਰਨ end = begin + nchunk;
                    ਜੇ (end > (ਪੂਰਨ) ncols) {
                        end = (ਪੂਰਨ) ncols;
                    }

                    ਸਿਖਰ_k_ਜਾਂਚ_ਮਿਲਾਓ_f32(
                        src + (int64_t) ਪੰਕਤੀ * ncols, ਖਾਲੀ_ਸੂਚਕ, begin, end, k, ਵੰਡੋ_ਖੰਡ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, ਝੂਠ, ਵਸਤੂ_ct1);
                });
        });

        main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_vals(sycl::range<1>((ਵੰਡੋ_ਖੰਡ + 1) * k), cgh);
            sycl::local_accessor<ਪੂਰਨ, 1> shared_idx(sycl::range<1>((ਵੰਡੋ_ਖੰਡ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * ਖੰਡ_dims, ਖੰਡ_dims),
                [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                    ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);
                    ਅਚਲ size_t off = (size_t) ਪੰਕਤੀ * nsplit * k;

                    ਸਿਖਰ_k_ਜਾਂਚ_ਮਿਲਾਓ_f32(
                        pv + off, pi + off, 0, nsplit * k, k, ਵੰਡੋ_ਖੰਡ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        ਖਾਲੀ_ਸੂਚਕ, dst_indices + (int64_t) ਪੰਕਤੀ * k, ਸੱਚ, ਵਸਤੂ_ct1);
                });
        });

        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਮਾਪ_3 = 128;

    ਅਚਲ sycl::range<1> ਖੰਡ_dims(ਖੰਡ_ਮਾਪ_3);
    ਅਚਲ sycl::range<1> grid_dims(nrows);

    main_ਧਾਰਾ->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_vals(sycl::range<1>((ਖੰਡ_ਮਾਪ_3 + 1) * k), cgh);
        sycl::local_accessor<ਪੂਰਨ, 1> shared_idx(sycl::range<1>((ਖੰਡ_ਮਾਪ_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0);

                ਜੇ (ਪੰਕਤੀ >= nrows) ਵਾਪਸ;

                ਸਿਖਰ_k_ਜਾਂਚ_ਮਿਲਾਓ_f32(
                    src + (int64_t) ਪੰਕਤੀ * ncols, ਖਾਲੀ_ਸੂਚਕ, 0, (ਪੂਰਨ) ncols, k, ਖੰਡ_ਮਾਪ_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    ਖਾਲੀ_ਸੂਚਕ, dst_indices + (int64_t) ਪੰਕਤੀ * k, ਸੱਚ, ਵਸਤੂ_ct1);
            });
    });
}

ਸਥਿਰ ਖਾਲੀ argmax_f32_i32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਪੂਰਨ *dst, ਅਚਲ ਪੂਰਨ ncols,
                               ਅਚਲ ਪੂਰਨ nrows, ਕਤਾਰ_ptr stream) {
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_ARGMAX_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, nrows, 1);
    ਅਚਲ size_t shared_mem = 256 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ਦਸ਼ਮਲਵ, 1> shared_ਅੰਕੜੇ(
            sycl::range<1>(shared_mem/ਆਕਾਰ(ਦਸ਼ਮਲਵ)), cgh);
        sycl::local_accessor<ਪੂਰਨ, 1> shared_indices(
            sycl::range<1>(shared_mem/ਆਕਾਰ(ਦਸ਼ਮਲਵ)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                ਅਚਲ ਪੂਰਨ tid = ਵਸਤੂ_ct1.get_local_id(2);
                ਅਚਲ ਪੂਰਨ ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_global_id(1);

                ਦਸ਼ਮਲਵ max_val = -INFINITY;
                ਪੂਰਨ max_idx = -1;

                ਲਈ (ਪੂਰਨ ਖਾਨਾ = tid; ਖਾਨਾ < ncols; ਖਾਨਾ += 256) {
                    ਦਸ਼ਮਲਵ val = x[ਪੰਕਤੀ * ncols + ਖਾਨਾ];
                    ਜੇ (val > max_val) {
                        max_val = val;
                        max_idx = ਖਾਨਾ;
                    }
                }

                shared_ਅੰਕੜੇ[tid] = max_val;
                shared_indices[tid] = max_idx;
                ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

                ਲਈ (ਪੂਰਨ stride = 256/2; stride > 0; stride >>= 1) {
                    ਜੇ (tid < stride) {
                        ਦਸ਼ਮਲਵ val1 = shared_ਅੰਕੜੇ[tid];
                        ਦਸ਼ਮਲਵ val2 = shared_ਅੰਕੜੇ[tid + stride];
                        ਜੇ (val2 > val1) {
                            shared_ਅੰਕੜੇ[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
                }


                ਜੇ (tid == 0) {
                    dst[ਪੰਕਤੀ] = shared_indices[0];
                }
            });
    });
}
ਸਥਿਰ ਖਾਲੀ diag_ਨਕਾਬ_inf_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *x, ਦਸ਼ਮਲਵ *dst,
                                   ਅਚਲ ਪੂਰਨ ncols_x, ਅਚਲ ਪੂਰਨ nrows_x,
                                   ਅਚਲ ਪੂਰਨ ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_channel, ਅਚਲ ਪੂਰਨ n_past,
                                   ਕਤਾਰ_ptr stream) {
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, SYCL_DIAG_ਨਕਾਬ_INF_ਖੰਡ_ਮਾਪ, 1);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (ncols_x + SYCL_DIAG_ਨਕਾਬ_INF_ਖੰਡ_ਮਾਪ - 1) / SYCL_DIAG_ਨਕਾਬ_INF_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             diag_ਨਕਾਬ_inf_f32(x, dst, ncols_x,
                                               ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_channel, n_past,
                                               ਵਸਤੂ_ct1);
                         });
}

ਸਥਿਰ dpct::err0 ggml_sycl_cpy_tensor_2d(ਖਾਲੀ *dst,
                                          ਅਚਲ ਬਣਤਰ ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_ਨੀਵਾਂ, int64_t i1_ਉੱਚਾ,
                                          ਕਤਾਰ_ptr stream) ਕੋਸ਼ਿਸ਼ {

    dpct::memcpy_direction ਕਿਸਮ_2;
    ਅੱਖਰ * src_ptr;
    ਜੇ (ggml_backend_ਬਫਰ_is_host(src->ਬਫਰ)) {
        ਕਿਸਮ_2 = dpct::host_ਮੰਜ਼ਿਲ_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (ਅੱਖਰ *) src->ਅੰਕੜੇ;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_backend_ਬਫਰ_is_sycl(src->ਬਫਰ)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        ਕਿਸਮ_2    = dpct::device_ਮੰਜ਼ਿਲ_device;
        src_ptr = (ਅੱਖਰ *) src->ਅੰਕੜੇ;
    } ਨਹੀਂਤਾਂ ਜੇ (ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src->ਬਫਰ)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_ਨੀਵਾਂ == 0 && i1_ਉੱਚਾ == src->ne[1]);
        ਕਿਸਮ_2 = dpct::device_ਮੰਜ਼ਿਲ_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        ਪੂਰਨ id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (ਅੱਖਰ *) extra->ਅੰਕੜੇ_device[id];
    } ਨਹੀਂਤਾਂ {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    ਅੱਖਰ * dst_ptr = (ਅੱਖਰ *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ਅਚਲ ਗਿਣਤੀ ggml_type type = src->type;
    ਅਚਲ int64_t ts = ggml_type_ਮਾਪ(type);
    ਅਚਲ int64_t bs = ggml_blck_ਮਾਪ(type);
    int64_t i1_diff = i1_ਉੱਚਾ - i1_ਨੀਵਾਂ;

    ਅਚਲ ਅੱਖਰ * x = src_ptr + i1_ਨੀਵਾਂ*nb1 + i2*nb2 + i3*nb3;
    ਜੇ (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        ਵਾਪਸ CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    ਕਿਸਮ_2, *stream));

    } ਨਹੀਂਤਾਂ ਜੇ (nb0 == ts) {
        ਵਾਪਸ CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, ਕਿਸਮ_2, *stream));
    } ਨਹੀਂਤਾਂ {
        ਲਈ (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ਅਚਲ ਖਾਲੀ * rx = (ਅਚਲ ਖਾਲੀ *) ((ਅਚਲ ਅੱਖਰ *) x + i1*nb1);
            ਖਾਲੀ * rd = (ਖਾਲੀ *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, ਕਿਸਮ_2, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            ਜੇ (r != 0) ਵਾਪਸ r;
        }
        ਵਾਪਸ 0;
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
    ਅਚਲ ਅੱਖਰ *src0_dd_i, ਅਚਲ ਦਸ਼ਮਲਵ *src1_ddf_i, ਅਚਲ ਅੱਖਰ *src1_ddq_i,
    ਦਸ਼ਮਲਵ *dst_dd_i, ਅਚਲ int64_t ਪੰਕਤੀ_ਨੀਵਾਂ, ਅਚਲ int64_t ਪੰਕਤੀ_ਉੱਚਾ,
    ਅਚਲ int64_t src1_ncols, ਅਚਲ int64_t src1_padded_ਪੰਕਤੀ_ਮਾਪ,
    ਅਚਲ ਕਤਾਰ_ptr &stream) ਕੋਸ਼ਿਸ਼ {

    GGML_ASSERT(src0_dd_i  != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(src1_ddf_i != ਖਾਲੀ_ਸੂਚਕ);
    GGML_ASSERT(dst_dd_i   != ਖਾਲੀ_ਸੂਚਕ);

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ਅਚਲ int64_t ਪੰਕਤੀ_diff = ਪੰਕਤੀ_ਉੱਚਾ - ਪੰਕਤੀ_ਨੀਵਾਂ;

    ਪੂਰਨ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ਅਚਲ int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    ਪੂਰਨ ldc = id == ctx.device ? ne0 : ਪੰਕਤੀ_diff; // used by MKL only

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਤਾਰਕਿਕ use_fp16 = ਸੱਚ;  // TODO(Yu) SYCL capability check
#ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ use_fp16 = ਝੂਠ;
#ਜੇ_ਅੰਤ

#ਜੇ GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    ਜੇ (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        ਪੰਕਤੀ_diff == src0->ne[1]) {
        ਵਰਤੋਂ bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        ਜੇ (src1->type != GGML_TYPE_BF16) {
            ਅਚਲ ਮੰਜ਼ਿਲ_bf16_sycl_t ਮੰਜ਼ਿਲ_bf16_sycl = ggml_get_ਮੰਜ਼ਿਲ_bf16_sycl(src1->type, dst);
            GGML_ASSERT(ਮੰਜ਼ਿਲ_bf16_sycl != ਖਾਲੀ_ਸੂਚਕ);
            ਮੰਜ਼ਿਲ_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } ਨਹੀਂਤਾਂ {
            stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*ਆਕਾਰ(bf16_t));
        }
        DnnlGemmWrapper::ਪੰਕਤੀ_gemm(ctx, ਪੰਕਤੀ_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(), stream);
        GGML_ਅਣਵਰਤਿਆ(dst);
        GGML_ਅਣਵਰਤਿਆ(src1_ddq_i);
        GGML_ਅਣਵਰਤਿਆ(src1_padded_ਪੰਕਤੀ_ਮਾਪ);
        ਵਾਪਸ;
    }
#ਜੇ_ਅੰਤ

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    ਤਾਰਕਿਕ src0_reordered = ਝੂਠ;
    ਜੇ (ਅਚਲ ਸਵੈਚਾਲਿਤ * src0_extra = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    ਜੇ (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && ਪੰਕਤੀ_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     ਪੰਕਤੀ_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        ਵਾਪਸ;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    ਜੇ ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        ਪੰਕਤੀ_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        ਜੇ (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ਅਚਲ ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(src1->type, dst);
            GGML_ASSERT(ਮੰਜ਼ਿਲ_fp16_sycl != ਖਾਲੀ_ਸੂਚਕ);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            ਮੰਜ਼ਿਲ_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ਅਚਲ sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ਅਚਲ sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        ਜੇ (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ਅਚਲ ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(src0->type, dst);
            GGML_ASSERT(ਮੰਜ਼ਿਲ_fp16_sycl != ਖਾਲੀ_ਸੂਚਕ);
            size_t ne = ਪੰਕਤੀ_diff*ne00;
            src0_as_f16.alloc(ne);
            ਮੰਜ਼ਿਲ_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ਅਚਲ sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ਅਚਲ sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#ਜੇ GGML_SYCL_DNNL
        ਜੇ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::ਪੰਕਤੀ_gemm(ctx,ਪੰਕਤੀ_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(), stream);
        }
        ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
        {
            ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
            ਅਚਲ ਦਸ਼ਮਲਵ beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, ਪੰਕਤੀ_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_ਅੰਕੜੇ_t::real_half, ne00,
                src1_ptr, dpct::library_ਅੰਕੜੇ_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_ਅੰਕੜੇ_t::real_float, ldc,
                dpct::library_ਅੰਕੜੇ_t::real_float)));
        }
    } ਨਹੀਂਤਾਂ {
        ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> src1_ddq_as_f32(ctx.pool());
        ਜੇ (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ਅਚਲ ਮੰਜ਼ਿਲ_fp32_sycl_t ਮੰਜ਼ਿਲ_fp32_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp32_sycl(src0->type, dst);
            GGML_ASSERT(ਮੰਜ਼ਿਲ_fp32_sycl != ਖਾਲੀ_ਸੂਚਕ);
            src0_ddq_as_f32.alloc(ਪੰਕਤੀ_diff*ne00);
            ਮੰਜ਼ਿਲ_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), ਪੰਕਤੀ_diff*ne00, stream);
        }
        ਜੇ (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ਅਚਲ ਮੰਜ਼ਿਲ_fp32_sycl_t ਮੰਜ਼ਿਲ_fp32_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp32_sycl(src1->type, dst);
            GGML_ASSERT(ਮੰਜ਼ਿਲ_fp32_sycl != ਖਾਲੀ_ਸੂਚਕ);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            ਮੰਜ਼ਿਲ_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ਅਚਲ ਦਸ਼ਮਲਵ * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) src0_dd_i : src0_ddq_as_f32.get();
        ਅਚਲ ਦਸ਼ਮਲਵ * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#ਜੇ GGML_SYCL_DNNL
            ਅਚਲ int64_t gemm_flops = (int64_t)ਪੰਕਤੀ_diff * src1_ncols * ne10;
            ਅਚਲ ਤਾਰਕਿਕ use_mkl_ਸਿੱਧਾ = gemm_flops < 256 * 256 * 256;
            ਜੇ (g_ggml_sycl_enable_dnn && !use_mkl_ਸਿੱਧਾ && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::ਪੰਕਤੀ_gemm(ctx, ਪੰਕਤੀ_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(), src1_ddf1_i, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(),
                                          dst_dd_i, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(), stream);
            }
            ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
            {
                ਅਚਲ ਦਸ਼ਮਲਵ alpha = 1.0f;
                ਅਚਲ ਦਸ਼ਮਲਵ beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::ਕਾਲਮ_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ਪੰਕਤੀ_diff,
                    src1_ncols, ne10, dpct::get_ਮੁੱਲ(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_ਮੁੱਲ(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(src1_ddq_i);
    GGML_ਅਣਵਰਤਿਆ(src1_padded_ਪੰਕਤੀ_ਮਾਪ);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਜੋੜ(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    ਜੋੜ_ਪੰਕਤੀਆਂ_f32_sycl(src0_dd, dst_dd, ne, 1, main_ਧਾਰਾ);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ncols = dst->src[0]->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    ਜੋੜ_ਪੰਕਤੀਆਂ_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ਧਾਰਾ);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_ਔਸਤ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ncols = dst->src[0]->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    ਜੋੜ_ਪੰਕਤੀਆਂ_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ਧਾਰਾ);

    main_ਧਾਰਾ->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> ਪੰਕਤੀ) {
            dst_dd[ਪੰਕਤੀ] /= ncols;
        }
    );
}


ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ਅੰਕੜੇ);


    ਅਚਲ int64_t ncols = dst->src[0]->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(dst->src[0]);

    ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (ਪੂਰਨ *)dst_dd, ncols, nrows, ਕ੍ਰਮ,
                         main_ਧਾਰਾ, ctx.device, ctx.pool());
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_ਸਿਖਰ_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
    int32_t * dst_dd = static_cast<int32_t *>(dst->ਅੰਕੜੇ);

    ਅਚਲ ਪੂਰਨ k = dst->ne[0];
    ਅਚਲ int64_t ncols = src0->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    ਜੇ (k <= SYCL_ਸਿਖਰ_K_ਜਾਂਚ_ਮਿਲਾਓ_MAX_K) {
        ਸਿਖਰ_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ਧਾਰਾ);
    } ਨਹੀਂਤਾਂ {
        ggml_sycl_ਸਿਖਰ_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ਧਾਰਾ);
    }
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ncols = dst->src[0]->ne[0];
    ਅਚਲ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_ਧਾਰਾ);
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_diag_ਨਕਾਬ_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ int64_t ne00 = dst->src[0]->ne[0];
    ਅਚਲ int64_t ne01 = dst->src[0]->ne[1];
    ਅਚਲ ਪੂਰਨ nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ਅਚਲ ਪੂਰਨ n_past = ((int32_t *) dst->op_params)[0];

    diag_ਨਕਾਬ_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_ਧਾਰਾ);
}

ਸਥਿਰ ਖਾਲੀ tri_f32_sycl(
    ਅਚਲ ਦਸ਼ਮਲਵ * src,
    ਦਸ਼ਮਲਵ * dst,
    ਅਚਲ int64_t ne0,
    ਅਚਲ int64_t ne1,
    ਅਚਲ int64_t ne2,
    ਅਚਲ int64_t ne3,
    ਅਚਲ ggml_tri_type ttype,
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ
) {
    ਅਚਲ size_t ਕੁੱਲ = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_ਧਾਰਾ->parallel_for(sycl::range<1>(ਕੁੱਲ), [=](sycl::id<1> tid) {
        ਅਚਲ int64_t idx = (int64_t) tid[0];

        ਅਚਲ int64_t i0 = idx % ne0;
        ਅਚਲ int64_t t1 = idx / ne0;
        ਅਚਲ int64_t i1 = t1 % ne1;

        ਤਾਰਕਿਕ keep = ਝੂਠ;
        ਚੋਣ (ttype) {
            ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ:      keep = (i0 <  i1); ਰੋਕੋ;
            ਹਾਲਤ GGML_TRI_TYPE_ਹੇਠਲਾ_DIAG: keep = (i0 <= i1); ਰੋਕੋ;
            ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ:      keep = (i0 >  i1); ਰੋਕੋ;
            ਹਾਲਤ GGML_TRI_TYPE_ਉੱਤਲਾ_DIAG: keep = (i0 >= i1); ਰੋਕੋ;
            ਮੂਲ: keep = ਝੂਠ; ਰੋਕੋ;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ਅਚਲ int64_t ne0 = src0->ne[0];
    ਅਚਲ int64_t ne1 = src0->ne[1];
    ਅਚਲ int64_t ne2 = src0->ne[2];
    ਅਚਲ int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_ਧਾਰਾ);
}


ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_dd = static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(dst->src[0]->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *       dst_dd  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ bias;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&scale, (ਦਸ਼ਮਲਵ *) dst->op_params + 0, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&bias,  (ਦਸ਼ਮਲਵ *) dst->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_ਧਾਰਾ);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਸੈੱਟ_peer_access(ਅਚਲ ਪੂਰਨ n_ਟੋਕਨ, [[maybe_unused]] ਪੂਰਨ main_device) {
    ਸਥਿਰ ਤਾਰਕਿਕ peer_access_enabled = ਝੂਠ;

    ਅਚਲ ਤਾਰਕਿਕ enable_peer_access = n_ਟੋਕਨ <= GGML_SYCL_PEER_MAX_BATCH_ਮਾਪ;

    ਜੇ (peer_access_enabled == enable_peer_access) {
        ਵਾਪਸ;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ NDEBUG
    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(i));
    }

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(i));

        ਲਈ (ਪੂਰਨ id_other = 0; id_other < ggml_sycl_info().device_ਗਿਣਤੀ; ++id_other) {
            ਜੇ (i == id_other) {
                ਜਾਰੀ;
            }
            ਜੇ (i != main_device && id_other != main_device) {
                ਜਾਰੀ;
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
#ਜੇ_ਅੰਤ // NDEBUG

    peer_access_enabled = enable_peer_access;
}

ਸਾਂਚਾ <ਸਾਂਚਾ <ਪੂਰਨ> ਕਿਸਮ_ਨਾਂ quantize_f>
ਸਥਿਰ ਖਾਲੀ ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0,
                                 ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) ਕੋਸ਼ਿਸ਼ {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ਅਚਲ int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];

    ਅਚਲ ਪੂਰਨ nb2 = dst->nb[2];
    ਅਚਲ ਪੂਰਨ nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(dst->ਬਫਰ));
    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src1->ਬਫਰ));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ਅਚਲ int64_t i02_ਭਾਜਕ = ne12 / ne02;

    ਅਚਲ size_t src0_ts = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t src0_bs = ggml_blck_ਮਾਪ(src0->type);
    ਅਚਲ size_t q8_1_ts = ਆਕਾਰ(ਖੰਡ_q8_1);
    ਅਚਲ size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ਅਚਲ ਤਾਰਕਿਕ src0_is_contiguous = ggml_is_contiguous(src0);
    ਅਚਲ ਤਾਰਕਿਕ src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_ਖਾਨਾ_ਮਾਪ = GGML_PAD(ne10, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);

    ਅਚਲ ਤਾਰਕਿਕ ਵੰਡੋ = ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ);
    GGML_ASSERT(!(ਵੰਡੋ && ne02 > 1));
    GGML_ASSERT(!(ਵੰਡੋ && ne03 > 1));
    GGML_ASSERT(!(ਵੰਡੋ && ne02 < ne12));

    std::array<ਦਸ਼ਮਲਵ, GGML_SYCL_MAX_DEVICES> tensor_ਵੰਡੋ;
    ਜੇ (ਵੰਡੋ) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *) src0->ਬਫਰ->buft->context;
        tensor_ਵੰਡੋ = buft_ctx->tensor_ਵੰਡੋ;
    }

    ਬਣਤਰ dev_ਅੰਕੜੇ {
        ggml_sycl_pool_alloc<ਅੱਖਰ> src0_dd_alloc;
        ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> src1_ddf_alloc;
        ggml_sycl_pool_alloc<ਅੱਖਰ> src1_ddq_alloc;
        ggml_sycl_pool_alloc<ਦਸ਼ਮਲਵ> dst_dd_alloc;

        ਅੱਖਰ *src0_dd = ਖਾਲੀ_ਸੂਚਕ;
        ਦਸ਼ਮਲਵ *src1_ddf = ਖਾਲੀ_ਸੂਚਕ; // float
        ਅੱਖਰ *src1_ddq = ਖਾਲੀ_ਸੂਚਕ;  // q8_1
        ਦਸ਼ਮਲਵ *dst_dd = ਖਾਲੀ_ਸੂਚਕ;

        int64_t ਪੰਕਤੀ_ਨੀਵਾਂ;
        int64_t ਪੰਕਤੀ_ਉੱਚਾ;
    };

    dev_ਅੰਕੜੇ dev[GGML_SYCL_MAX_DEVICES];

    ਪੂਰਨ ਵਰਤਿਆ_devices = 0;
    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();

    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        // by default, use all rows
        dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ  = 0;
        dev[i].ਪੰਕਤੀ_ਉੱਚਾ = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        ਜੇ (ਵੰਡੋ) {
            ਅਚਲ int64_t rounding = get_ਪੰਕਤੀ_rounding(src0->type, tensor_ਵੰਡੋ);

            ਜੇ (i != 0) {
                dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ  = ne01*tensor_ਵੰਡੋ[i];
                ਜੇ (dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ < ne01) {
                    dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ -= dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ % rounding;
                }
            }

            ਜੇ (i != ggml_sycl_info().device_ਗਿਣਤੀ - 1) {
                dev[i].ਪੰਕਤੀ_ਉੱਚਾ  = ne01*tensor_ਵੰਡੋ[i + 1];
                ਜੇ (dev[i].ਪੰਕਤੀ_ਉੱਚਾ < ne01) {
                    dev[i].ਪੰਕਤੀ_ਉੱਚਾ -= dev[i].ਪੰਕਤੀ_ਉੱਚਾ % rounding;
                }
            }
        }
    }

    constexpr ਤਾਰਕਿਕ quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_ਮਾਪ>,
                                                      no_quantize_q8_1<QK8_1 / WARP_ਮਾਪ>>;
    ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
        ਜੇ ((!ਵੰਡੋ && i != ctx.device) || dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ == dev[i].ਪੰਕਤੀ_ਉੱਚਾ) {
            ਜਾਰੀ;
        }

        ਵਰਤਿਆ_devices++;

        ਅਚਲ ਤਾਰਕਿਕ src1_on_device = i == ctx.device;
        ਅਚਲ ਤਾਰਕਿਕ  dst_on_device = i == ctx.device;

        ggml_sycl_ਸੈੱਟ_device(i);
        ਕਤਾਰ_ptr stream = ctx.stream(i, 0);

        ਜੇ (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = ਵੰਡੋ ? (ਅੱਖਰ *) src0_extra->ਅੰਕੜੇ_device[i] : (ਅੱਖਰ *) src0->ਅੰਕੜੇ;
        } ਨਹੀਂਤਾਂ {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        ਜੇ (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
        } ਨਹੀਂਤਾਂ {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        ਜੇ constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_ਖਾਨਾ_ਮਾਪ*q8_1_ts/q8_1_bs);

            ਜੇ (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                ਕੋਸ਼ਿਸ਼ {
                    quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_ਖਾਨਾ_ਮਾਪ, stream);
                } ਫੜੋ (sycl::exception ਅਚਲ &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
                    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
                }
            }
        }

        ਜੇ (dst_on_device) {
            dev[i].dst_dd = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ size_t ਮਾਪ_dst_ddf = ਵੰਡੋ ? (dev[i].ਪੰਕਤੀ_ਉੱਚਾ - dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), ਮਾਪ_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    ਜੇ (ਵੰਡੋ && ਵਰਤਿਆ_devices > 1) {
        ggml_sycl_ਸੈੱਟ_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ਅਚਲ int64_t src1_ਖਾਨਾ_stride = ਵੰਡੋ && ਵਰਤਿਆ_devices > 1 ? MUL_MAT_SRC1_ਖਾਨਾ_STRIDE : ne11;
    ਲਈ (int64_t src1_ਖਾਨਾ_0 = 0; src1_ਖਾਨਾ_0 < ne11; src1_ਖਾਨਾ_0 += src1_ਖਾਨਾ_stride) {
        ਅਚਲ int64_t is = ਵੰਡੋ ? (src1_ਖਾਨਾ_0/src1_ਖਾਨਾ_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ਅਚਲ int64_t src1_ncols = src1_ਖਾਨਾ_0 + src1_ਖਾਨਾ_stride > ne11 ? ne11 - src1_ਖਾਨਾ_0 : src1_ਖਾਨਾ_stride;
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
            ਜੇ ((!ਵੰਡੋ && i != ctx.device) || dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ == dev[i].ਪੰਕਤੀ_ਉੱਚਾ) {
                ਜਾਰੀ;
            }

            ਅਚਲ ਤਾਰਕਿਕ src1_on_device = i == ctx.device;
            ਅਚਲ ਤਾਰਕਿਕ  dst_on_device = i == ctx.device;
            ਅਚਲ int64_t ਪੰਕਤੀ_diff = dev[i].ਪੰਕਤੀ_ਉੱਚਾ - dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ;

            ggml_sycl_ਸੈੱਟ_device(i);
            ਕਤਾਰ_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            ਜੇ (ਵੰਡੋ && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            ਲਈ (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ਅਚਲ int64_t i03 = i0 / ne12;
                ਅਚਲ int64_t i02 = i0 % ne12;

                ਅਚਲ size_t src1_ddq_i_ਵਿੱਥ = (i0*ne11 + src1_ਖਾਨਾ_0) * src1_padded_ਖਾਨਾ_ਮਾਪ*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                ਅੱਖਰ  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ਭਾਜਕ) * (ne01*ne00*src0_ts)/src0_bs;
                ਦਸ਼ਮਲਵ * src1_ddf_i = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dev[i].src1_ddf +
                    (i0*ne11 + src1_ਖਾਨਾ_0) * ne10 * ggml_type_ਮਾਪ(src1->type));
                ਅੱਖਰ  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_ਵਿੱਥ;
                ਦਸ਼ਮਲਵ *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_ਖਾਨਾ_0) * (dst_on_device ? ne0 : ਪੰਕਤੀ_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                ਜੇ (i == ctx.device) {
                    dst_dd_i += dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                ਜੇ (src1_is_contiguous) {
                    ਜੇ (i != ctx.device) {
                        ਜੇ constexpr (quantize_enabled) {
                            ਅੱਖਰ * src1_ddq_i_ਸਰੋਤ = dev[ctx.device].src1_ddq + src1_ddq_i_ਵਿੱਥ;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->ਸਮ੍ਰਿਤੀ_ਨਕਲ(src1_ddq_i, src1_ddq_i_ਸਰੋਤ,
                                                             src1_ncols * src1_padded_ਖਾਨਾ_ਮਾਪ * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } ਨਹੀਂਤਾਂ {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ਅਚਲ ਅੱਖਰ * src1_ddf_i_ਸਰੋਤ = (ਅਚਲ ਅੱਖਰ *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_ਖਾਨਾ_0) * ne10 * ggml_type_ਮਾਪ(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_ਧਾਰਾ, src1_ddf_i, src1_ddf_i_ਸਰੋਤ,
                                                               src1_ncols * ne10 * ggml_type_ਮਾਪ(src1->type))));
                        }
                    }
                } ਨਹੀਂਤਾਂ {
                    ਜੇ (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_ਖਾਨਾ_0,
                                                           src1_ਖਾਨਾ_0 + src1_ncols, stream));
                    } ਨਹੀਂਤਾਂ {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    ਜੇ constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        ਕੋਸ਼ਿਸ਼ {
                            quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_ਖਾਨਾ_ਮਾਪ, stream);
                        } ਫੜੋ (ਅਚਲ sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
                            std::ਬਾਹਰ_ਨਿਕਲੋ(1);
                        }
                    }
                }

                ਜੇ (src1_ਖਾਨਾ_0 == 0 && !src0_is_contiguous && i02 % i02_ਭਾਜਕ == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ਭਾਜਕ, dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ, dev[i].ਪੰਕਤੀ_ਉੱਚਾ, stream));
                }
                ਜੇ (src1->type == GGML_TYPE_F16) {
                    src1_padded_ਖਾਨਾ_ਮਾਪ = (i0 * ne11 + src1_ਖਾਨਾ_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ, dev[i].ਪੰਕਤੀ_ਉੱਚਾ, src1_ncols, src1_padded_ਖਾਨਾ_ਮਾਪ, stream)));

                // copy dst to host or other device if necessary
                ਜੇ (!dst_on_device) {
                    ਖਾਲੀ * dst_off_device = dst->ਅੰਕੜੇ;
                    ਜੇ (ਵੰਡੋ) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        ਦਸ਼ਮਲਵ * dhf_dst_i = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                        dhf_dst_i += src1_ਖਾਨਾ_0*ne0 + dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), dst_dd_i,
                            ਪੰਕਤੀ_diff * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਪੰਕਤੀ_diff * ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                            src1_ncols, dpct::device_ਮੰਜ਼ਿਲ_device, *stream)));
                    } ਨਹੀਂਤਾਂ {
                        ਦਸ਼ਮਲਵ * dhf_dst_i = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                        dhf_dst_i += src1_ਖਾਨਾ_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                ਜੇ (ਵੰਡੋ && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    ਜੇ (ਵੰਡੋ && ggml_sycl_info().device_ਗਿਣਤੀ > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_ਖਾਨਾ_STRIDE - 1) / MUL_MAT_SRC1_ਖਾਨਾ_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_ਸੈੱਟ_device(ctx.device);
        ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; ++i) {
            ਜੇ (dev[i].ਪੰਕਤੀ_ਨੀਵਾਂ == dev[i].ਪੰਕਤੀ_ਉੱਚਾ) {
                ਜਾਰੀ;
            }
            ਲਈ (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_repeat_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_ਪਿੱਠ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_get_ਪੰਕਤੀਆਂ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_ਪੰਕਤੀਆਂ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_rms_norm_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_ਪਿੱਠ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਸਮੂਹ_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ਸਮੂਹ_norm(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0,
                                       ਅਚਲ ggml_tensor *src1,
                                       ggml_tensor *dst) ਕੋਸ਼ਿਸ਼ {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];

    ਅਚਲ int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();

    ਖਾਲੀ  * src0_ddq = src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ddf = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ddf  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_ਧਾਰਾ);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0,
                                     ਅਚਲ ggml_tensor *src1,
                                     ggml_tensor *dst) ਕੋਸ਼ਿਸ਼ {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    ਅਚਲ int64_t ne00 = src0->ne[0];
    ਅਚਲ int64_t ne01 = src0->ne[1];
    ਅਚਲ int64_t ne02 = src0->ne[2];

    ਅਚਲ int64_t nb01 = src0->nb[1];
    ਅਚਲ int64_t nb02 = src0->nb[2];

    ਅਚਲ int64_t ne12 = src1->ne[2];
    ਅਚਲ int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਕਤਾਰ_ptr main_ਧਾਰਾ = ctx.stream();

    ਖਾਲੀ  * src0_ddq = src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * src1_ddf = (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ddf  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t ਪੰਕਤੀ_stride_x = nb01 / ਆਕਾਰ(sycl::half);
    ਅਚਲ int64_t channel_stride_x = nb02 / ਆਕਾਰ(sycl::half);
    ਅਚਲ int64_t channel_stride_y = nb11 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ਪੰਕਤੀ_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_ਧਾਰਾ);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ k_compute_batched_ptrs(ਅਚਲ sycl::half * src0_as_f16, ਅਚਲ sycl::half * src1_as_f16, ਖਾਲੀ * dst,
                                   ਅਚਲ ਖਾਲੀ ** ptrs_src, ਖਾਲੀ ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t i13 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) + ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t i12 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) * ਵਸਤੂ_ct1.get_local_range(1) + ਵਸਤੂ_ct1.get_local_id(1);

    ਜੇ (i13 >= ne13 || i12 >= ne12) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t i03 = i13 / r3;
    ਅਚਲ int64_t i02 = i12 / r2;

    ਅਚਲ uint8_t * src0_ਬਾਈਟਾਂ = reinterpret_cast<ਅਚਲ uint8_t *>(src0_as_f16);
    ਅਚਲ uint8_t * src1_ਬਾਈਟਾਂ = reinterpret_cast<ਅਚਲ uint8_t *>(src1_as_f16);
    uint8_t *       dst_ਬਾਈਟਾਂ  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_ਬਾਈਟਾਂ + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_ਬਾਈਟਾਂ + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_ਬਾਈਟਾਂ + i12 * nbd2 + i13 * nbd3;
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0,
                                           ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) ਕੋਸ਼ਿਸ਼ {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));
    ਕਤਾਰ_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ਅਚਲ sycl::half * src0_f16 = static_cast<ਅਚਲ sycl::half *>(src0->ਅੰਕੜੇ);
    ਦਸ਼ਮਲਵ *            dst_ddf  = static_cast<ਦਸ਼ਮਲਵ *>(dst->ਅੰਕੜੇ);

    ਅਚਲ sycl::half * src1_f16       = static_cast<ਅਚਲ sycl::half *>(src1->ਅੰਕੜੇ);
#ਜੇ GGML_SYCL_DNNL
    ਅਚਲ size_t       type_ਮਾਪ_src0 = ggml_type_ਮਾਪ(src0->type);
#ਜੇ_ਅੰਤ
    ਅਚਲ size_t       type_ਮਾਪ_src1 = ggml_type_ਮਾਪ(src1->type);

    ਤਾਰਕਿਕ is_src0_cont_2 = ggml_is_contiguous_2(src0);
    ਤਾਰਕਿਕ is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_ਮਾਪ_src1;
    int64_t                          s12 = nb12 / type_ਮਾਪ_src1;
    int64_t                          s13 = nb13 / type_ਮਾਪ_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    ਜੇ (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#ਜੇ GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        ਪੂਰਨ ਆਖਰੀ_dim=0;
        ਪੂਰਨ ਆਖਰੀ_str=0;
        size_t ਸਭ_ਤੋਂ_ਵੱਡਾ_str=0;
        ਲਈ(ਪੂਰਨ i = 0; i< 4; i++){
            // last stride is always the largest
            ਜੇ(src1->nb[i] == ਸਭ_ਤੋਂ_ਵੱਡਾ_str){
                ਜੇ(src1->ne[ਆਖਰੀ_dim] == 1){
                    ਆਖਰੀ_str = i;
                    ਆਖਰੀ_dim = i;
                }
            }
            ਜੇ(src1->nb[i] > ਸਭ_ਤੋਂ_ਵੱਡਾ_str){
                ਸਭ_ਤੋਂ_ਵੱਡਾ_str = src1->nb[i];
                ਆਖਰੀ_str = i;
                ਆਖਰੀ_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ਅਚਲ int64_t ne_src1 = src1->nb[ਆਖਰੀ_str] * src1->ne[ਆਖਰੀ_dim] / type_ਮਾਪ_src1;
        src1_f16_alloc.alloc(ne_src1);
        ਅਚਲ ਮੰਜ਼ਿਲ_fp16_sycl_t ਮੰਜ਼ਿਲ_fp16_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp16_sycl(src1->type, dst);
        GGML_ASSERT(ਮੰਜ਼ਿਲ_fp16_sycl != ਖਾਲੀ_ਸੂਚਕ);
        ਮੰਜ਼ਿਲ_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# ਨਹੀਂਤਾਂ
        ਅਚਲ int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ਅਚਲ ਮੰਜ਼ਿਲ_fp16_nc_sycl_t ਮੰਜ਼ਿਲ_fp16_nc_sycl = ggml_get_ਮੰਜ਼ਿਲ_fp16_nc_sycl(src1->type);
        GGML_ASSERT(ਮੰਜ਼ਿਲ_fp16_nc_sycl != ਖਾਲੀ_ਸੂਚਕ);
        ਮੰਜ਼ਿਲ_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#ਜੇ_ਅੰਤ

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = ਸੱਚ;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_ਅੰਕੜੇ_t mkl_compute_type = dpct::library_ਅੰਕੜੇ_t::real_float;
    dpct::library_ਅੰਕੜੇ_t mkl_ਅੰਕੜੇ_type    = dpct::library_ਅੰਕੜੇ_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ਅਚਲ ਦਸ਼ਮਲਵ alpha_f32 = 1.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ beta_f32  = 0.0f;

    ਅਚਲ ਖਾਲੀ * alpha = &alpha_f32;
    ਅਚਲ ਖਾਲੀ * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ਅਚਲ int64_t r2 = ne12 / ne02;
    ਅਚਲ int64_t r3 = ne13 / ne03;

#ਜੇ GGML_SYCL_DNNL
    ਜੇ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_ਮਾਪ_src0;
            int64_t str_a1 = nb01 / type_ਮਾਪ_src0;
            int64_t str_a2 = nb02 / type_ਮਾਪ_src0;

            int64_t str_b0 = nb10 / type_ਮਾਪ_src1;
            int64_t str_b1 = nb11 / type_ਮਾਪ_src1;
            int64_t str_b2 = nb12 / type_ਮਾਪ_src1;

            ਸਵੈਚਾਲਿਤ launch_gemm_for_batches = [&ctx, queue](ਅਚਲ sycl::half *src0,
                                                ਅਚਲ sycl::half *src1, ਦਸ਼ਮਲਵ *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                ਤਾਰਕਿਕ supported_ਪ੍ਰਸਾਰਣ = batchb == batcha ? ਸੱਚ
                        : batchb == 1 || batcha == 1        ? ਸੱਚ
                                                            : ਝੂਠ;
                ਜੇ (supported_ਪ੍ਰਸਾਰਣ) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(), queue, batcha, batchb);
                } ਨਹੀਂਤਾਂ {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    ਜੇ (batches0 > batches1) {
                        int64_t ਅੰਕ_mul_mats = batches1;
                        int64_t sub_batch = batches0 / ਅੰਕ_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        ਲਈ (int64_t i0 = 0; i0 < ਅੰਕ_mul_mats; i0++) {
                            ਅਚਲ sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ਅਚਲ sycl::half *src1_shifted = src1 + (sb2 * i0);
                            ਦਸ਼ਮਲਵ *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(),
                                    queue, sub_batch, 1);
                        }
                    } ਨਹੀਂਤਾਂ {
                        int64_t ਅੰਕ_mul_mats = batches0;
                        int64_t sub_batch = batches1 / ਅੰਕ_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        ਲਈ (int64_t i1 = 0; i1 < ਅੰਕ_mul_mats; i1++) {
                            ਅਚਲ sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ਅਚਲ sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            ਦਸ਼ਮਲਵ *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::ਮੰਜ਼ਿਲ_dt<ਦਸ਼ਮਲਵ>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ਅਚਲ ਤਾਰਕਿਕ cont_batches_dim2_a = nb02 * ne02 == nb03;
            ਅਚਲ ਤਾਰਕਿਕ cont_batches_dim2_b = nb12 * ne12 == nb13;
            ਅਚਲ ਤਾਰਕਿਕ cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ਅਚਲ ਤਾਰਕਿਕ cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            ਜੇ (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            } ਨਹੀਂਤਾਂ ਜੇ (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_ਮਾਪ_src0;
                int64_t str_b3 = nb13 / type_ਮਾਪ_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            } ਨਹੀਂਤਾਂ {
                ਲਈ (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ਅਚਲ sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_ਮਾਪ_src0);
                    ਅਚਲ sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_ਮਾਪ_src1);
                    ਦਸ਼ਮਲਵ *dst_shifted = dst_ddf + (nb3 * b_a / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                }
            }

    }
    ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
    {
        ਜੇ (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ਅਚਲ int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ਅਚਲ int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_ਅੰਕੜੇ_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_ਅੰਕੜੇ_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_ਅੰਕੜੇ_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ਪੂਰਨ ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ਅਚਲ ਖਾਲੀ *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<ਖਾਲੀ *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<ਮੈਟ੍ਰਿਕਸ_info_t<ਦਸ਼ਮਲਵ>> ਮੈਟ੍ਰਿਕਸ_info(ctx.host_pool(), 1);

            sycl::range<3> ਖੰਡ_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ਅਚਲ ਖਾਲੀ ** ptrs_src_get = ptrs_src.get();
                ਖਾਲੀ **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * ਆਕਾਰ(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * ਆਕਾਰ(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(ਖੰਡ_dims, ਖੰਡ_dims), [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, ਵਸਤੂ_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ਅਚਲ ਖਾਲੀ **) (ptrs_src.get() + 0 * ne23), dpct::library_ਅੰਕੜੇ_t::real_half, nb01 / nb00,
                (ਅਚਲ ਖਾਲੀ **) (ptrs_src.get() + 1 * ne23), dpct::library_ਅੰਕੜੇ_t::real_half, s11, beta,
                (ਖਾਲੀ **) (ptrs_dst.get() + 0 * ne23), mkl_ਅੰਕੜੇ_type, ne0, ne23, mkl_compute_type, ਮੈਟ੍ਰਿਕਸ_info.get())));
        }
    }
} ਫੜੋ (ਅਚਲ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਗਿਣਤੀ ਵਰਗ mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_supports_mmq(ਗਿਣਤੀ ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_ਅਣਵਰਤਿਆ(type);
    ਵਾਪਸ ਝੂਠ;
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_supports_reorder_mul_mat_sycl(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਵਾਪਸ !g_ggml_sycl_prioritize_dmmv;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_supports_reorder_dmmv(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਤਰਅੰਦਰ ਤਾਰਕਿਕ ggml_sycl_supports_reorder_mmvq(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_supports_reorder_esimd(ਗਿਣਤੀ ggml_type type) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(type);
    ਵਾਪਸ ਝੂਠ;
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_supports_dmmv(ਗਿਣਤੀ ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q1_0:
        ਹਾਲਤ GGML_TYPE_Q4_0:
        ਹਾਲਤ GGML_TYPE_Q4_1:
        ਹਾਲਤ GGML_TYPE_Q5_0:
        ਹਾਲਤ GGML_TYPE_Q5_1:
        ਹਾਲਤ GGML_TYPE_Q8_0:
        ਹਾਲਤ GGML_TYPE_Q2_K:
        ਹਾਲਤ GGML_TYPE_Q3_K:
        ਹਾਲਤ GGML_TYPE_Q4_K:
        ਹਾਲਤ GGML_TYPE_Q5_K:
        ਹਾਲਤ GGML_TYPE_Q6_K:
        ਹਾਲਤ GGML_TYPE_F16:
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ ਸੱਚ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ * sycl_ext_malloc_device(dpct::ਕਤਾਰ_ptr stream, size_t size) {
    ਤਾਰਕਿਕ use_async = g_ggml_sycl_use_async_mem_op;
#ਜੇ defined(GGML_SYCL_ਗ੍ਰਾਫ) && SYCL_EXT_ONEAPI_ASYNC_ਸਮ੍ਰਿਤੀ_ALLOC
    ਜੇ (use_async) {
        ਖਾਲੀ * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_ਜੋੜੋ(GGML_SYCL_MEM_ASYNC, ptr, size);
        ਵਾਪਸ ptr;
    }
#ਨਹੀਂਤਾਂ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ggml_sycl_malloc_device(size, *stream);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sycl_ext_free(dpct::ਕਤਾਰ_ptr stream, ਖਾਲੀ * ptr) {
    ਤਾਰਕਿਕ use_async = g_ggml_sycl_use_async_mem_op;
#ਜੇ defined(GGML_SYCL_ਗ੍ਰਾਫ) && SYCL_EXT_ONEAPI_ASYNC_ਸਮ੍ਰਿਤੀ_ALLOC
    ਜੇ (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        ਵਾਪਸ;
    }
#ਨਹੀਂਤਾਂ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#ਜੇ_ਅੰਤ
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
ਬਣਤਰ sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ {
    ਖਾਲੀ *          ptr  = ਖਾਲੀ_ਸੂਚਕ;
    dpct::ਕਤਾਰ_ptr stream;

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ(dpct::ਕਤਾਰ_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HOST_MEM_FALLBACK
        ਜੇ (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            ਜੇ (ptr) {
                host_fallback = ਸੱਚ;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#ਜੇ_ਅੰਤ
    }

    ~sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ() {
        ਜੇ (!ptr) {
            ਵਾਪਸ;
        }
        ਜੇ (host_fallback) {
            sycl::ਮੁਕਤ(ptr, *stream);
        } ਨਹੀਂਤਾਂ {
            sycl_ext_free(stream, ptr);
        }
    }

    ਸਪਸ਼ਟ ਸੰਚਾਲਕ ਤਾਰਕਿਕ() ਅਚਲ { ਵਾਪਸ ptr != ਖਾਲੀ_ਸੂਚਕ; }

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ(ਅਚਲ sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ &)            = ਮਿਟਾਓ;
    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ & ਸੰਚਾਲਕ=(ਅਚਲ sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ &) = ਮਿਟਾਓ;

ਨਿਜੀ:
    ਤਾਰਕਿਕ host_fallback = ਝੂਠ;
};

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q4_0(uint8_t * ਅੰਕੜੇ_device, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, size_t size, size_t ਵਿੱਥ,
                            dpct::ਕਤਾਰ_ptr stream) {
    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ਆਕਾਰ(ਖੰਡ_q4_0) == 0));
    GGML_ASSERT((ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q4_0) == 0));
    ਪੂਰਨ ਵਿੱਥ_blks = ਵਿੱਥ / ਆਕਾਰ(ਖੰਡ_q4_0);
    ਸਵੈਚਾਲਿਤ qs_ptr      = ਅੰਕੜੇ_device + ਵਿੱਥ_blks * QK4_0 / 2;
    ਸਵੈਚਾਲਿਤ d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + ਵਿੱਥ_blks;

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(
        size / ਆਕਾਰ(ਖੰਡ_q4_0),
            [=](ਸਵੈਚਾਲਿਤ i) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            ਅਚਲ ਖੰਡ_q4_0* x = (ਅਚਲ ਖੰਡ_q4_0*)tmp_buf;
            ਅਚਲ ਪੂਰਨ ib = i;

            ਲਈ (ਪੂਰਨ j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q8_0(uint8_t * ਅੰਕੜੇ_device, ਅਚਲ ਪੂਰਨ ncols, ਅਚਲ ਪੂਰਨ nrows, size_t size, size_t ਵਿੱਥ,
                            dpct::ਕਤਾਰ_ptr stream) {
    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ਆਕਾਰ(ਖੰਡ_q8_0) == 0));
    GGML_ASSERT((ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q8_0) == 0));
    ਪੂਰਨ ਵਿੱਥ_blks = ਵਿੱਥ / ਆਕਾਰ(ਖੰਡ_q8_0);
    ਸਵੈਚਾਲਿਤ qs_ptr = ਅੰਕੜੇ_device + ਵਿੱਥ_blks * QK8_0;
    ਸਵੈਚਾਲਿਤ d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + ਵਿੱਥ_blks;

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(
        size / ਆਕਾਰ(ਖੰਡ_q8_0),
            [=](ਸਵੈਚਾਲਿਤ i) [[sycl::reqd_sub_ਸਮੂਹ_ਮਾਪ(WARP_ਮਾਪ)]] {
            ਅਚਲ ਖੰਡ_q8_0* x = (ਅਚਲ ਖੰਡ_q8_0*)tmp_buf;
            ਅਚਲ ਪੂਰਨ ib = i;

            ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q8_0_moe(uint8_t * ਅੰਕੜੇ_device, size_t expert_ਬਾਈਟਾਂ, int64_t n_expert, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(expert_ਬਾਈਟਾਂ % ਆਕਾਰ(ਖੰਡ_q8_0) == 0);
    ਅਚਲ ਪੂਰਨ    ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) (expert_ਬਾਈਟਾਂ / ਆਕਾਰ(ਖੰਡ_q8_0));
    ਅਚਲ size_t ਕੁੱਲ_ਬਾਈਟਾਂ       = expert_ਬਾਈਟਾਂ * (size_t) n_expert;

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, ਕੁੱਲ_ਬਾਈਟਾਂ);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ਕੁੱਲ_ਬਾਈਟਾਂ);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, ਕੁੱਲ_ਬਾਈਟਾਂ)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਖੰਡਾਂ = ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * (ਪੂਰਨ) n_expert;
    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(ਕੁੱਲ_ਖੰਡਾਂ, [=](ਸਵੈਚਾਲਿਤ gb_) {
        ਅਚਲ ਪੂਰਨ          gb   = gb_;
        ਅਚਲ ਪੂਰਨ          e    = gb / ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਪੂਰਨ          ib   = gb % ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਖੰਡ_q8_0 * x    = (ਅਚਲ ਖੰਡ_q8_0 *) (tmp_buf + (size_t) e * expert_ਬਾਈਟਾਂ);
        uint8_t *          ਅਧਾਰ = ਅੰਕੜੇ_device + (size_t) e * expert_ਬਾਈਟਾਂ;

        ਸਵੈਚਾਲਿਤ * qs_ptr = ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * d_ptr  = (sycl::half *) (qs_ptr + (size_t) ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * QK8_0);

        ਲਈ (ਪੂਰਨ j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q4_k(uint8_t * ਅੰਕੜੇ_device, size_t size, size_t ਵਿੱਥ, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_q4_K) == 0);
    GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q4_K) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_q4_K);

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ * qs_ptr     = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    ਸਵੈਚਾਲਿਤ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ਮਾਪ * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_q4_K * x  = (ਅਚਲ ਖੰਡ_q4_K *) tmp_buf;
        ਅਚਲ ਪੂਰਨ          ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < K_SCALE_ਮਾਪ; ++j) {
            scales_ptr[ib * K_SCALE_ਮਾਪ + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

// Reorder each expert slice into a self-contained SoA layout.
ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q4_k_moe(uint8_t * ਅੰਕੜੇ_device, size_t expert_ਬਾਈਟਾਂ, int64_t n_expert, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(expert_ਬਾਈਟਾਂ % ਆਕਾਰ(ਖੰਡ_q4_K) == 0);
    ਅਚਲ ਪੂਰਨ    ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) (expert_ਬਾਈਟਾਂ / ਆਕਾਰ(ਖੰਡ_q4_K));
    ਅਚਲ size_t ਕੁੱਲ_ਬਾਈਟਾਂ       = expert_ਬਾਈਟਾਂ * (size_t) n_expert;

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, ਕੁੱਲ_ਬਾਈਟਾਂ);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ਕੁੱਲ_ਬਾਈਟਾਂ);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, ਕੁੱਲ_ਬਾਈਟਾਂ)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਖੰਡਾਂ = ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * (ਪੂਰਨ) n_expert;
    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(ਕੁੱਲ_ਖੰਡਾਂ, [=](ਸਵੈਚਾਲਿਤ gb_) {
        ਅਚਲ ਪੂਰਨ          gb   = gb_;
        ਅਚਲ ਪੂਰਨ          e    = gb / ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਪੂਰਨ          ib   = gb % ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਖੰਡ_q4_K * x    = (ਅਚਲ ਖੰਡ_q4_K *) (tmp_buf + (size_t) e * expert_ਬਾਈਟਾਂ);
        uint8_t *          ਅਧਾਰ = ਅੰਕੜੇ_device + (size_t) e * expert_ਬਾਈਟਾਂ;

        ਸਵੈਚਾਲਿਤ * qs_ptr     = ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * scales_ptr = qs_ptr + QK_K / 2 * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਸਵੈਚਾਲਿਤ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ਮਾਪ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert);

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < K_SCALE_ਮਾਪ; ++j) {
            scales_ptr[ib * K_SCALE_ਮਾਪ + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q5_k_moe(uint8_t * ਅੰਕੜੇ_device, size_t expert_ਬਾਈਟਾਂ, int64_t n_expert, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(expert_ਬਾਈਟਾਂ % ਆਕਾਰ(ਖੰਡ_q5_K) == 0);
    ਅਚਲ ਪੂਰਨ    ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) (expert_ਬਾਈਟਾਂ / ਆਕਾਰ(ਖੰਡ_q5_K));
    ਅਚਲ size_t ਕੁੱਲ_ਬਾਈਟਾਂ       = expert_ਬਾਈਟਾਂ * (size_t) n_expert;

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, ਕੁੱਲ_ਬਾਈਟਾਂ);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ਕੁੱਲ_ਬਾਈਟਾਂ);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, ਕੁੱਲ_ਬਾਈਟਾਂ)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਖੰਡਾਂ = ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * (ਪੂਰਨ) n_expert;
    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(ਕੁੱਲ_ਖੰਡਾਂ, [=](ਸਵੈਚਾਲਿਤ gb_) {
        ਅਚਲ ਪੂਰਨ          gb   = gb_;
        ਅਚਲ ਪੂਰਨ          e    = gb / ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਪੂਰਨ          ib   = gb % ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਖੰਡ_q5_K * x    = (ਅਚਲ ਖੰਡ_q5_K *) (tmp_buf + (size_t) e * expert_ਬਾਈਟਾਂ);
        uint8_t *          ਅਧਾਰ = ਅੰਕੜੇ_device + (size_t) e * expert_ਬਾਈਟਾਂ;

        ਸਵੈਚਾਲਿਤ * qs_ptr     = ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * qh_ptr     = qs_ptr + (QK_K / 2) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਸਵੈਚਾਲਿਤ * scales_ptr = qh_ptr + (QK_K / 8) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਸਵੈਚਾਲਿਤ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ਮਾਪ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert);

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < K_SCALE_ਮਾਪ; ++j) {
            scales_ptr[ib * K_SCALE_ਮਾਪ + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q6_k_moe(uint8_t * ਅੰਕੜੇ_device, size_t expert_ਬਾਈਟਾਂ, int64_t n_expert, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(expert_ਬਾਈਟਾਂ % ਆਕਾਰ(ਖੰਡ_q6_K) == 0);
    ਅਚਲ ਪੂਰਨ    ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) (expert_ਬਾਈਟਾਂ / ਆਕਾਰ(ਖੰਡ_q6_K));
    ਅਚਲ size_t ਕੁੱਲ_ਬਾਈਟਾਂ       = expert_ਬਾਈਟਾਂ * (size_t) n_expert;

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, ਕੁੱਲ_ਬਾਈਟਾਂ);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ਕੁੱਲ_ਬਾਈਟਾਂ);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, ਕੁੱਲ_ਬਾਈਟਾਂ)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਖੰਡਾਂ = ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * (ਪੂਰਨ) n_expert;
    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(ਕੁੱਲ_ਖੰਡਾਂ, [=](ਸਵੈਚਾਲਿਤ gb_) {
        ਅਚਲ ਪੂਰਨ          gb   = gb_;
        ਅਚਲ ਪੂਰਨ          e    = gb / ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਪੂਰਨ          ib   = gb % ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਅਚਲ ਖੰਡ_q6_K * x    = (ਅਚਲ ਖੰਡ_q6_K *) (tmp_buf + (size_t) e * expert_ਬਾਈਟਾਂ);
        uint8_t *          ਅਧਾਰ = ਅੰਕੜੇ_device + (size_t) e * expert_ਬਾਈਟਾਂ;

        ਸਵੈਚਾਲਿਤ * ql_ptr     = ਅਧਾਰ;
        ਸਵੈਚਾਲਿਤ * qh_ptr     = ql_ptr + (QK_K / 2) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਸਵੈਚਾਲਿਤ * scales_ptr = qh_ptr + (QK_K / 4) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
        ਸਵੈਚਾਲਿਤ * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert);

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_mxfp4_moe(uint8_t * ਅੰਕੜੇ_device, size_t expert_ਬਾਈਟਾਂ, int64_t n_expert, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(expert_ਬਾਈਟਾਂ % ਆਕਾਰ(ਖੰਡ_mxfp4) == 0);
    ਅਚਲ ਪੂਰਨ     ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert = (ਪੂਰਨ) (expert_ਬਾਈਟਾਂ / ਆਕਾਰ(ਖੰਡ_mxfp4));
    ਅਚਲ size_t  max_chunk_ਬਾਈਟਾਂ   = 32u << 20;
    ਅਚਲ int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_ਬਾਈਟਾਂ / expert_ਬਾਈਟਾਂ)));

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, (size_t) chunk_experts * expert_ਬਾਈਟਾਂ);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_ਬਾਈਟਾਂ);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    ਲਈ (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ਅਚਲ int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = ਅੰਕੜੇ_device + (size_t) e0 * expert_ਬਾਈਟਾਂ;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, chunk, (size_t) n_chunk * expert_ਬਾਈਟਾਂ)));
        ਜੇ (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        ਅਚਲ ਪੂਰਨ ਕੁੱਲ_ਖੰਡਾਂ = ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert * (ਪੂਰਨ) n_chunk;
        ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(ਕੁੱਲ_ਖੰਡਾਂ, [=](ਸਵੈਚਾਲਿਤ gb_) {
            ਅਚਲ ਪੂਰਨ           gb   = gb_;
            ਅਚਲ ਪੂਰਨ           e    = gb / ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
            ਅਚਲ ਪੂਰਨ           ib   = gb % ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;
            ਅਚਲ ਖੰਡ_mxfp4 * x    = (ਅਚਲ ਖੰਡ_mxfp4 *) (tmp_buf + (size_t) e * expert_ਬਾਈਟਾਂ);
            uint8_t *           ਅਧਾਰ = chunk + (size_t) e * expert_ਬਾਈਟਾਂ;

            uint8_t * qs_ptr = ਅਧਾਰ;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) ਖੰਡਾਂ_ਪ੍ਰਤੀ_expert;

            ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        ਜੇ (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q2_k(uint8_t * ਅੰਕੜੇ_device, size_t size, size_t ਵਿੱਥ, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_q2_K) == 0);
    GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q2_K) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_q2_K);

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ *        qs_ptr     = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_q2_K * x  = (ਅਚਲ ਖੰਡ_q2_K *) tmp_buf;
        ਅਚਲ ਪੂਰਨ          ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q3_k(uint8_t * ਅੰਕੜੇ_device, size_t size, size_t ਵਿੱਥ, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_q3_K) == 0);
    GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q3_K) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_q3_K);

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ *       qs_ptr     = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    ਸਵੈਚਾਲਿਤ *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_q3_K * x  = (ਅਚਲ ਖੰਡ_q3_K *) tmp_buf;
        ਅਚਲ ਪੂਰਨ          ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q5_k(uint8_t * ਅੰਕੜੇ_device, size_t size, size_t ਵਿੱਥ, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_q5_K) == 0);
    GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q5_K) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_q5_K);

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ * qs_ptr     = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    ਸਵੈਚਾਲਿਤ * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    ਸਵੈਚਾਲਿਤ * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_ਮਾਪ * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_q5_K * x  = (ਅਚਲ ਖੰਡ_q5_K *) tmp_buf;
        ਅਚਲ ਪੂਰਨ          ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < K_SCALE_ਮਾਪ; ++j) {
            scales_ptr[ib * K_SCALE_ਮਾਪ + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_q6_k(uint8_t * ਅੰਕੜੇ_device, size_t size, size_t ਵਿੱਥ, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_q6_K) == 0);
    GGML_ASSERT(ਵਿੱਥ % ਆਕਾਰ(ਖੰਡ_q6_K) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_q6_K);

    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ *       ql_ptr     = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    ਸਵੈਚਾਲਿਤ *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_q6_K * x  = (ਅਚਲ ਖੰਡ_q6_K *) tmp_buf;
        ਅਚਲ ਪੂਰਨ          ib = i;

        ਅਚਲ uint8_t * ql              = x[ib].ql;
        ਅਚਲ uint8_t * qh              = x[ib].qh;
        uint8_t *       ਅਧਾਰ_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       ਅਧਾਰ_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       ਅਧਾਰ_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 2; ++j) {
            ਅਧਾਰ_ql_ptr[j] = ql[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; ++j) {
            ਅਧਾਰ_qh_ptr[j] = qh[j];
        }

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 16; ++j) {
            ਅਧਾਰ_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_iq3_s(uint8_t * ਅੰਕੜੇ_device, size_t size, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_iq3_s) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_iq3_s);
    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ * qs_ptr       = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    ਸਵੈਚਾਲਿਤ * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    ਸਵੈਚਾਲਿਤ * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_iq3_s * x = (ਅਚਲ ਖੰਡ_iq3_s *) tmp_buf;
        ਅਚਲ ਪੂਰਨ ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (ਆਕਾਰ(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        ਲਈ (ਪੂਰਨ j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[ਆਕਾਰ(ggml_half) + j] = x[ib].scales[j];
        }
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw_iq3_xxs(uint8_t * ਅੰਕੜੇ_device, size_t size, dpct::ਕਤਾਰ_ptr stream) {
    GGML_ASSERT(size % ਆਕਾਰ(ਖੰਡ_iq3_xxs) == 0);

    ਅਚਲ ਪੂਰਨ nblocks = size / ਆਕਾਰ(ਖੰਡ_iq3_xxs);
    sycl_reorder_ਆਰਜ਼ੀ_ਬਫਰ tmp(stream, size);
    ਜੇ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ਵਾਪਸ ਝੂਠ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_buf, ਅੰਕੜੇ_device, size)));
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ਸਵੈਚਾਲਿਤ * qs_ptr = ਅੰਕੜੇ_device;
    ਸਵੈਚਾਲਿਤ * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    ਸਵੈਚਾਲਿਤ reorder_event = stream->parallel_for(nblocks, [=](ਸਵੈਚਾਲਿਤ i) {
        ਅਚਲ ਖੰਡ_iq3_xxs * x = (ਅਚਲ ਖੰਡ_iq3_xxs *) tmp_buf;
        ਅਚਲ ਪੂਰਨ ib = i;

        ਲਈ (ਪੂਰਨ j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ਜੇ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ਤਾਰਕਿਕ reorder_qw(ਅਚਲ ggml_tensor * src0, dpct::ਕਤਾਰ_ptr stream) {
    uint8_t * ਅੰਕੜੇ_device = (uint8_t *) src0->ਅੰਕੜੇ;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    ਜੇ (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        ਚੋਣ (src0->type) {
            ਹਾਲਤ GGML_TYPE_Q8_0:
                ਵਾਪਸ reorder_qw_q8_0_moe(ਅੰਕੜੇ_device, src0->nb[2], src0->ne[2], stream);
            ਹਾਲਤ GGML_TYPE_Q4_K:
                ਵਾਪਸ reorder_qw_q4_k_moe(ਅੰਕੜੇ_device, src0->nb[2], src0->ne[2], stream);
            ਹਾਲਤ GGML_TYPE_Q5_K:
                ਵਾਪਸ reorder_qw_q5_k_moe(ਅੰਕੜੇ_device, src0->nb[2], src0->ne[2], stream);
            ਹਾਲਤ GGML_TYPE_Q6_K:
                ਵਾਪਸ reorder_qw_q6_k_moe(ਅੰਕੜੇ_device, src0->nb[2], src0->ne[2], stream);
            ਹਾਲਤ GGML_TYPE_MXFP4:
                ਵਾਪਸ reorder_qw_mxfp4_moe(ਅੰਕੜੇ_device, src0->nb[2], src0->ne[2], stream);
            ਮੂਲ:
                ਵਾਪਸ ਝੂਠ;
        }
    }

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਵਾਪਸ reorder_qw_q4_0(ਅੰਕੜੇ_device, ncols, nrows, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ reorder_qw_q8_0(ਅੰਕੜੇ_device, ncols, nrows, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਵਾਪਸ reorder_qw_q2_k(ਅੰਕੜੇ_device, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ਵਾਪਸ reorder_qw_q3_k(ਅੰਕੜੇ_device, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਵਾਪਸ reorder_qw_q4_k(ਅੰਕੜੇ_device, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ਵਾਪਸ reorder_qw_q5_k(ਅੰਕੜੇ_device, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਵਾਪਸ reorder_qw_q6_k(ਅੰਕੜੇ_device, size, 0, stream);
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            ਵਾਪਸ reorder_qw_iq3_xxs(ਅੰਕੜੇ_device, size, stream);
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਵਾਪਸ reorder_qw_iq3_s(ਅੰਕੜੇ_device, size, stream);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ should_reorder_tensor(ggml_backend_sycl_context& ctx, ਅਚਲ ggml_tensor * dst) {
    ਵਾਪਸ g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

ਸਥਿਰ ਖਾਲੀ opt_for_reorder(ggml_backend_sycl_context * ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    ਜੇ (!should_reorder_tensor(*ctx, dst)) {
        ਵਾਪਸ;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ਜੇ (!extra || extra->optimized_feature.reorder) {
        ਵਾਪਸ;  // Skip permutations and already reordered tensors
    }

    ਚੋਣ (mm_algorithm) {
        ਹਾਲਤ mul_mat_algo::DMMV:
            ਜੇ (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                ਵਾਪਸ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ mul_mat_algo::MMVQ:
            ਜੇ (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                ਵਾਪਸ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ mul_mat_algo::MUL_MAT_SYCL:
            ਜੇ (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                ਵਾਪਸ;
            }
            ਰੋਕੋ;
    }

    ਜੇ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ਸੱਚ;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
ਸਥਿਰ ਖਾਲੀ opt_for_reorder_id(ggml_backend_sycl_context * ctx, ਅਚਲ ggml_tensor * src0) {
    ਜੇ (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        ਵਾਪਸ;
    }
    ਜੇ (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        ਵਾਪਸ;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    ਜੇ (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->ਅੰਕੜੇ % 16 != 0)) {
        ਵਾਪਸ;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ਜੇ (!extra || extra->optimized_feature.reorder) {
        ਵਾਪਸ;
    }
    ਜੇ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ਸੱਚ;
    }
}


ਸਥਿਰ ਤਾਰਕਿਕ can_use_dequantize_mul_mat_vec(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ਅਚਲ int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    ਵਾਪਸ ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

ਸਥਿਰ ਤਾਰਕਿਕ can_use_mul_mat_vec_q(ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    ਵਾਪਸ ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_ਮਾਪ;
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0, ਅਚਲ ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    ਜੇ (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਤਾਰਕਿਕ ਵੰਡੋ = ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ);
    int64_t min_compute_capability = ਵੱਧ_ਤੋਂ_ਵੱਧ_ਅੰਕ;

    ਜੇ (ਵੰਡੋ) {
        ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context * buft_ctx =
            (ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type_context *) src0->ਬਫਰ->buft->context;
        ਸਵੈਚਾਲਿਤ & tensor_ਵੰਡੋ = buft_ctx->tensor_ਵੰਡੋ;
        ਲਈ (ਪੂਰਨ id = 0; id < ggml_sycl_info().device_ਗਿਣਤੀ; ++id) {
            // skip devices that are not going to do any work:
            ਜੇ (tensor_ਵੰਡੋ[id] >= (id + 1 < ggml_sycl_info().device_ਗਿਣਤੀ ? tensor_ਵੰਡੋ[id + 1] : 1.0f)) {
                ਜਾਰੀ;
            }

            ਜੇ (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } ਨਹੀਂਤਾਂ {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    ਤਾਰਕਿਕ use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#ਜੇ_ਅੰਤ

    ਤਾਰਕਿਕ use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    ਤਾਰਕਿਕ use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_ਮਾਪ);
#ਜੇ_ਅੰਤ // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    ਜੇ (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        ਤਾਰਕਿਕ use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        ਜੇ (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    ਜੇ (!ਵੰਡੋ && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        ਜੇ (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } ਨਹੀਂਤਾਂ {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (!ਵੰਡੋ && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (!ਵੰਡੋ && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } ਨਹੀਂਤਾਂ ਜੇ (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } ਨਹੀਂਤਾਂ ਜੇ (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        ਜੇ (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } ਨਹੀਂਤਾਂ {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } ਨਹੀਂਤਾਂ ਜੇ (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } ਨਹੀਂਤਾਂ {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * ਉੱਪਰ, ਅਚਲ ggml_tensor * wu,
                                             ਅਚਲ ggml_tensor * wg, ਅਚਲ ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_u = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wu->extra);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_g = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wg->extra);
    ਜੇ ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ਅਚਲ ਸਵੈਚਾਲਿਤ plain_type = [](ggml_type t) { ਵਾਪਸ t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    ਜੇ (!plain_type(wu->type) || !plain_type(wg->type)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ਉੱਪਰ, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ਅਚਲ int64_t ne00 = wu->ne[0];
    ਅਚਲ int64_t ne11 = act->ne[1];

    ਅਚਲ ਕਤਾਰ_ptr stream = ctx.stream();
    ਅਚਲ ਪੂਰਨ src1_padded_ਖਾਨੇ_ਗਿਣਤੀ = GGML_PAD((ਪੂਰਨ) ne00, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);

    ggml_sycl_pool_alloc<ਅੱਖਰ> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ਖਾਨੇ_ਗਿਣਤੀ * ਆਕਾਰ(ਖੰਡ_q8_1) / QK8_1);
    ਅੱਖਰ * src1_ddq = src1_q8_alloc.get();

    quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_q8_1>((ਅਚਲ ਦਸ਼ਮਲਵ *) act->ਅੰਕੜੇ, src1_ddq, (ਪੂਰਨ) ne00, (ਪੂਰਨ) ne11,
                                          src1_padded_ਖਾਨੇ_ਗਿਣਤੀ, stream);

    ਵਾਪਸ ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->ਅੰਕੜੇ, wu->ਅੰਕੜੇ,
                                             src1_ddq, (ਦਸ਼ਮਲਵ *) glu->ਅੰਕੜੇ, (ਪੂਰਨ) ne00, (ਪੂਰਨ) wu->ne[1],
                                             (ਪੂਰਨ) ne11,
                                             /*stride_col_y=*/src1_padded_ਖਾਨੇ_ਗਿਣਤੀ / QK8_1,
                                             /*stride_col_dst=*/(ਪੂਰਨ) glu->ne[0], stream);
    GGML_ਅਣਵਰਤਿਆ(gate);
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * ਉੱਪਰ, ਅਚਲ ggml_tensor * wu,
                                             ਅਚਲ ggml_tensor * wg, ਅਚਲ ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ਅਚਲ ਅੱਖਰ * act_begin = (ਅਚਲ ਅੱਖਰ *) act->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * glu_begin = (ਅਚਲ ਅੱਖਰ *) glu->ਅੰਕੜੇ;
    ਜੇ (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        ਵਾਪਸ ਝੂਠ;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, ਉੱਪਰ, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_u = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wu->extra);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_g = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wg->extra);
    ਜੇ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ਵਾਪਸ ਝੂਠ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ਉੱਪਰ, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    ਵਾਪਸ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->ਅੰਕੜੇ, wg->ਅੰਕੜੇ,
                                                              (ਅਚਲ ਦਸ਼ਮਲਵ *) act->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) glu->ਅੰਕੜੇ,
                                                              (ਪੂਰਨ) wu->ne[0], (ਪੂਰਨ) wu->ne[1], ctx.stream());
}
#ਜੇ_ਅੰਤ

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਜੇ (!ggml_sycl_can_fuse(cgraph, ਗੰਢ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        ਵਾਪਸ ਝੂਠ;
    }

    ggml_tensor *       glu  = cgraph->nodes[ਗੰਢ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       ਉੱਪਰ   = glu->src[1];
    ਅਚਲ ggml_tensor * wu   = ਉੱਪਰ->src[0];
    ਅਚਲ ggml_tensor * wg   = gate->src[0];
    ਅਚਲ ggml_tensor * act  = ਉੱਪਰ->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    ਜੇ (ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(wu->ਬਫਰ) || ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(wg->ਬਫਰ)) {
        ਵਾਪਸ ਝੂਠ;
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    ਜੇ (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, ਉੱਪਰ, wu, wg, act)) {
        ਵਾਪਸ ਸੱਚ;
    }
#ਜੇ_ਅੰਤ

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    ਜੇ (g_ggml_sycl_prioritize_dmmv) {
        ਵਾਪਸ ਝੂਠ;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ਅਚਲ ਤਾਰਕਿਕ reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    ਜੇ (!reorder_pair) {
        ਵਾਪਸ ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, ਉੱਪਰ, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    ਜੇ (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        ਵਾਪਸ ਝੂਠ;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, ਉੱਪਰ, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_u = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wu->extra);
    ਅਚਲ ਸਵੈਚਾਲਿਤ * extra_g = static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(wg->extra);
    ਜੇ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ਵਾਪਸ ਝੂਠ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ਉੱਪਰ, /*num_src=*/2, " : fused with gate + GLU");

    ਅਚਲ int64_t ne00 = wu->ne[0];
    ਅਚਲ int64_t ne11 = act->ne[1];

    ਅਚਲ ਕਤਾਰ_ptr stream           = ctx.stream();
    ਅਚਲ ਪੂਰਨ       src1_padded_ਖਾਨੇ_ਗਿਣਤੀ = GGML_PAD((ਪੂਰਨ) ne00, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<ਅੱਖਰ> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_ਖਾਨੇ_ਗਿਣਤੀ * ਆਕਾਰ(ਖੰਡ_q8_1) / QK8_1);
    ਅੱਖਰ *                     src1_ddq = src1_q8_alloc.get();

    quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ਅਚਲ ਦਸ਼ਮਲਵ *) act->ਅੰਕੜੇ, src1_ddq, (ਪੂਰਨ) ne00, (ਪੂਰਨ) ne11,
                                                          src1_padded_ਖਾਨੇ_ਗਿਣਤੀ, stream);

    ਵਾਪਸ ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->ਅੰਕੜੇ, wg->ਅੰਕੜੇ, src1_ddq,
                                               (ਦਸ਼ਮਲਵ *) glu->ਅੰਕੜੇ, (ਪੂਰਨ) ne00, (ਪੂਰਨ) wu->ne[1], (ਪੂਰਨ) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_ਖਾਨੇ_ਗਿਣਤੀ * (ਪੂਰਨ) ਆਕਾਰ(ਖੰਡ_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(ਪੂਰਨ) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
ਸਥਿਰ ਪੂਰਨ ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx) {
    ਅਚਲ ggml_tensor * ਗੰਢ = cgraph->nodes[ਗੰਢ_idx];
    ਜੇ (ggml_sycl_info().device_ਗਿਣਤੀ != 1 || ਗੰਢ->type != GGML_TYPE_F32 ||
        ਗੰਢ->src[0]->type != GGML_TYPE_F32 || ਗੰਢ->src[0]->ne[0] >= 1024) {
        ਵਾਪਸ 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    ਪੂਰਨ           count = 0;
    ਪੂਰਨ           ਆਖਰੀ  = ਗੰਢ_idx;
    ਦਸ਼ਮਲਵ         eps0;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&eps0, ਗੰਢ->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    ਸਵੈਚਾਲਿਤ overlaps = [](ਅਚਲ ggml_tensor * a, ਅਚਲ ggml_tensor * b) {
        ਅਚਲ ਅੱਖਰ * ab = (ਅਚਲ ਅੱਖਰ *) a->ਅੰਕੜੇ;
        ਅਚਲ ਅੱਖਰ * bb = (ਅਚਲ ਅੱਖਰ *) b->ਅੰਕੜੇ;
        ਵਾਪਸ ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    ਲਈ (ਪੂਰਨ j = ਗੰਢ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        ਜੇ (ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;  // not a launch; cannot break a run of adjacent norms
        }
        ਜੇ (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ਗੰਢ) ||
            !ggml_are_same_shape(nj->src[0], ਗੰਢ->src[0])) {
            ਰੋਕੋ;  // any other launch ends the run
        }
        ਤਾਰਕਿਕ same_nb = ਸੱਚ;
        ਲਈ (ਪੂਰਨ d = 0; d < GGML_MAX_DIMS; ++d) {
            ਜੇ (nj->nb[d] != ਗੰਢ->nb[d] || nj->src[0]->nb[d] != ਗੰਢ->src[0]->nb[d]) {
                same_nb = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (!same_nb) {
            ਰੋਕੋ;  // one nb[] stride set is shared by the whole batch
        }
        ਦਸ਼ਮਲਵ epsj;
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&epsj, nj->op_params, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਜੇ (epsj != eps0) {
            ਰੋਕੋ;  // eps mismatch ends the run
        }
        ਤਾਰਕਿਕ indep = ਸੱਚ;
        ਲਈ (ਪੂਰਨ k = 0; k < count; ++k) {
            ਜੇ (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = ਝੂਠ;
                ਰੋਕੋ;
            }
        }
        ਜੇ (!indep) {
            ਰੋਕੋ;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        ਆਖਰੀ           = j;
    }
    ਜੇ (count < 2) {
        ਵਾਪਸ 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    ਵਾਪਸ ਆਖਰੀ - ਗੰਢ_idx;
}


__dpct_inline__ ਸਥਿਰ ਖਾਲੀ k_copy_src1_ਮੰਜ਼ਿਲ_contiguous(
    ਅਚਲ ਅੱਖਰ *__restrict__ src1_original, ਅੱਖਰ *__restrict__ src1_contiguous,
    ਅਚਲ mmid_ਪੰਕਤੀ_mapping *__restrict__ ਪੰਕਤੀ_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ int32_t src1_ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ int32_t iid1 = ਪੰਕਤੀ_mapping[src1_ਪੰਕਤੀ].i2;
    ਅਚਲ int32_t id   = ਪੰਕਤੀ_mapping[src1_ਪੰਕਤੀ].i1;

    ਅਚਲ int64_t i11 = id % ne11;
    ਅਚਲ int64_t i12 = iid1;

    ਅਚਲ ਦਸ਼ਮਲਵ * src1_ਪੰਕਤੀ_original = (ਅਚਲ ਦਸ਼ਮਲਵ *)(src1_original + i11*nb11 + i12*nb12);
    ਦਸ਼ਮਲਵ * src1_ਪੰਕਤੀ_contiguous = (ਦਸ਼ਮਲਵ *)(src1_contiguous + src1_ਪੰਕਤੀ*nb11);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = ਵਸਤੂ_ct1.get_local_id(2); i < ne10;
         i += ਵਸਤੂ_ct1.get_local_range(2)) {
        src1_ਪੰਕਤੀ_contiguous[i] = src1_ਪੰਕਤੀ_original[i];
    }
}

__dpct_inline__ ਸਥਿਰ ਖਾਲੀ k_copy_dst_from_contiguous(
    ਅੱਖਰ *__restrict__ dst_original, ਅਚਲ ਅੱਖਰ *__restrict__ dst_contiguous,
    ਅਚਲ mmid_ਪੰਕਤੀ_mapping *__restrict__ ਪੰਕਤੀ_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    int32_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    ਅਚਲ int32_t i1 = ਪੰਕਤੀ_mapping[i].i1;
    ਅਚਲ int32_t i2 = ਪੰਕਤੀ_mapping[i].i2;

    ਅਚਲ ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ_contiguous = (ਅਚਲ ਦਸ਼ਮਲਵ *)(dst_contiguous + i*nb1);
    ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ_original = (ਦਸ਼ਮਲਵ *)(dst_original + i1*nb1 + i2*nb2);

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = ਵਸਤੂ_ct1.get_local_id(2); j < ne0;
         j += ਵਸਤੂ_ct1.get_local_range(2)) {
        dst_ਪੰਕਤੀ_original[j] = dst_ਪੰਕਤੀ_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0,
    ਅਚਲ ggml_tensor * src1, ਅਚਲ ggml_tensor * ids, ggml_tensor * dst)
{
    ਅਚਲ int64_t ne10 = src1->ne[0];
    ਅਚਲ int64_t ne11 = src1->ne[1];
    ਅਚਲ int64_t ne12 = src1->ne[2];
    ਜੇ (ne12 != 1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) ਵਾਪਸ ਝੂਠ;
    ਜੇ (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) ਵਾਪਸ ਝੂਠ;
    ਜੇ (!ggml_is_contiguous(src1)) ਵਾਪਸ ਝੂਠ;

    ਅਚਲ int64_t n_ids_ਪ੍ਰਤੀ_ਸਮੂਹ = ids->ne[0];
    ਜੇ (ids->ne[1] != 1) ਵਾਪਸ ਝੂਠ;
    ਜੇ (ne11 != 1 && ne11 != n_ids_ਪ੍ਰਤੀ_ਸਮੂਹ) ਵਾਪਸ ਝੂਠ;

    ਅਚਲ ਕਤਾਰ_ptr stream           = ctx.stream();
    ਅਚਲ ਪੂਰਨ       src1_padded_ਖਾਨੇ_ਗਿਣਤੀ = GGML_PAD((ਪੂਰਨ) ne10, ਮੈਟ੍ਰਿਕਸ_ਪੰਕਤੀ_PADDING);
    ਅਚਲ ਪੂਰਨ       n_experts_ਵਰਤਿਆ   = (ਪੂਰਨ) n_ids_ਪ੍ਰਤੀ_ਸਮੂਹ;
    ਅਚਲ ਪੂਰਨ       nrows            = (ਪੂਰਨ) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ਅਚਲ ggml_tensor_extra_gpu * src0_extra =
        static_cast<ਅਚਲ ggml_tensor_extra_gpu *>(src0->extra);
    ਅਚਲ ਤਾਰਕਿਕ use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<ਅੱਖਰ> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ਖਾਨੇ_ਗਿਣਤੀ * ਆਕਾਰ(ਖੰਡ_q8_1) / QK8_1);
    ਅੱਖਰ * src1_ddq = src1_q8_alloc.get();
    ਜੇ (use_reorder) {
        quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ, src1_ddq, (ਪੂਰਨ) ne10, (ਪੂਰਨ) ne11,
            src1_padded_ਖਾਨੇ_ਗਿਣਤੀ, stream);
    } ਨਹੀਂਤਾਂ {
        quantize_ਪੰਕਤੀ_q8_1_sycl<quantize_q8_1>(
            (ਅਚਲ ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ, src1_ddq, (ਪੂਰਨ) ne10, (ਪੂਰਨ) ne11,
            src1_padded_ਖਾਨੇ_ਗਿਣਤੀ, stream);
    }

    ਅਚਲ size_t ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_qrow = (size_t) src1_padded_ਖਾਨੇ_ਗਿਣਤੀ * ਆਕਾਰ(ਖੰਡ_q8_1) / QK8_1;
    ਅਚਲ size_t src1_ਪੰਕਤੀ_stride = (ne11 == 1) ? 0 : ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_qrow;

    ਜੇ (use_reorder) {
        ਵਾਪਸ ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->ਅੰਕੜੇ, src1_ddq, (ਅਚਲ int32_t *) ids->ਅੰਕੜੇ,
            (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ, (ਪੂਰਨ) ne10, nrows, n_experts_ਵਰਤਿਆ,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_ਪੰਕਤੀ_stride, stream);
    }
    ਵਾਪਸ ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->ਅੰਕੜੇ, src1_ddq, (ਅਚਲ int32_t *) ids->ਅੰਕੜੇ,
        (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ, (ਪੂਰਨ) ne10, nrows, n_experts_ਵਰਤਿਆ,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_ਪੰਕਤੀ_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
ਸਥਿਰ ਖਾਲੀ mmid_counting_ਛਾਂਟੋ_ਪੰਕਤੀਆਂ(
        ਅਚਲ ggml_tensor * ids, ਅਚਲ ਅੱਖਰ * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_ਪੰਕਤੀਆਂ,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_ਪੰਕਤੀ_offsets,
        std::vector<mmid_ਪੰਕਤੀ_mapping> & routed_ਪੰਕਤੀ_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.ਸੌਂਪੋ(n_as, 0);
    ਲਈ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ਲਈ (int64_t id = 0; id < n_ids; id++) {
            ਅਚਲ int32_t ਪੰਕਤੀ_id_i = *(ਅਚਲ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(ਪੰਕਤੀ_id_i >= 0 && ਪੰਕਤੀ_id_i < n_as);
            expert_counts[ਪੰਕਤੀ_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_ਪੰਕਤੀ_offsets.ਸੌਂਪੋ(n_as + 1, 0);
    ਲਈ (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_ਪੰਕਤੀ_offsets[i02 + 1] = expert_ਪੰਕਤੀ_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_ਪੰਕਤੀ_ਅਗਲਾ = expert_ਪੰਕਤੀ_offsets;
    routed_ਪੰਕਤੀ_src.resize(n_routed_ਪੰਕਤੀਆਂ);
    ਲਈ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ਲਈ (int64_t id = 0; id < n_ids; id++) {
            ਅਚਲ int32_t ਪੰਕਤੀ_id_i = *(ਅਚਲ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(ਪੰਕਤੀ_id_i >= 0 && ਪੰਕਤੀ_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ਅਚਲ int64_t routed_ਪੰਕਤੀ = expert_ਪੰਕਤੀ_ਅਗਲਾ[ਪੰਕਤੀ_id_i]++;
            GGML_ASSERT(routed_ਪੰਕਤੀ >= expert_ਪੰਕਤੀ_offsets[ਪੰਕਤੀ_id_i]);
            GGML_ASSERT(routed_ਪੰਕਤੀ < expert_ਪੰਕਤੀ_offsets[ਪੰਕਤੀ_id_i + 1]);
            routed_ਪੰਕਤੀ_src[routed_ਪੰਕਤੀ] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) ਕੋਸ਼ਿਸ਼ {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ਅਚਲ ggml_tensor *src0 = dst->src[0];
    ਅਚਲ ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(src0->ਬਫਰ) && "mul_mat_id does not support split buffers");

    ਅਚਲ ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ ਕਤਾਰ_ptr stream = ctx.stream();

    ਅਚਲ int64_t n_as = ne02;
    ਅਚਲ int64_t n_ids = ids->ne[0];

    ਜੇ (ne12 == 1) {
        ਜੇ (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            ਵਾਪਸ;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    ਜੇ (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<ਅੱਖਰ> ids_host(ggml_nbytes(ids));
    ਅਚਲ ਅੱਖਰ * ids_dev = (ਅਚਲ ਅੱਖਰ *) ids->ਅੰਕੜੇ;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(ids_host.ਅੰਕੜੇ(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_ਪੰਕਤੀ = *src0;
    ggml_tensor src1_ਪੰਕਤੀ = *src1;
    ggml_tensor dst_ਪੰਕਤੀ = *dst;

    ਅੱਖਰ *src0_original = (ਅੱਖਰ *)src0->ਅੰਕੜੇ;
    ਅੱਖਰ *src1_original = (ਅੱਖਰ *)src1->ਅੰਕੜੇ;
    ਅੱਖਰ *dst_original = (ਅੱਖਰ *)dst->ਅੰਕੜੇ;

    src0_ਪੰਕਤੀ.ne[2] = 1;
    src0_ਪੰਕਤੀ.ne[3] = 1;
    src0_ਪੰਕਤੀ.nb[3] = nb02;

    src1_ਪੰਕਤੀ.ne[1] = 1;
    src1_ਪੰਕਤੀ.ne[2] = 1;
    src1_ਪੰਕਤੀ.ne[3] = 1;
    src1_ਪੰਕਤੀ.nb[2] = nb11;
    src1_ਪੰਕਤੀ.nb[3] = nb11;

    dst_ਪੰਕਤੀ.ne[1] = 1;
    dst_ਪੰਕਤੀ.ne[2] = 1;
    dst_ਪੰਕਤੀ.ne[3] = 1;
    dst_ਪੰਕਤੀ.nb[2] = nb1;
    dst_ਪੰਕਤੀ.nb[3] = nb1;
    ਜੇ (ne12 == 1) {
        ਲਈ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            ਲਈ (int64_t id = 0; id < n_ids; id++) {
                ਅਚਲ int32_t i02 = *(ਅਚਲ int32_t *) (ids_host.ਅੰਕੜੇ() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ਅਚਲ int64_t i11 = id % ne11;
                ਅਚਲ int64_t i12 = iid1;

                ਅਚਲ int64_t i1 = id;
                ਅਚਲ int64_t i2 = i12;

            src0_ਪੰਕਤੀ.ਅੰਕੜੇ = src0_original + i02*nb02;
            src1_ਪੰਕਤੀ.ਅੰਕੜੇ = src1_original + i11*nb11 + i12*nb12;
            dst_ਪੰਕਤੀ.ਅੰਕੜੇ = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_ਪੰਕਤੀ, &src1_ਪੰਕਤੀ, &dst_ਪੰਕਤੀ);
            }
        }
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ int64_t n_routed_ਪੰਕਤੀਆਂ = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<ਅੱਖਰ> src1_contiguous(ctx.pool(), ਆਕਾਰ(ਦਸ਼ਮਲਵ)*n_routed_ਪੰਕਤੀਆਂ*ne10);
        ggml_sycl_pool_alloc<ਅੱਖਰ>  dst_contiguous(ctx.pool(), ਆਕਾਰ(ਦਸ਼ਮਲਵ)*n_routed_ਪੰਕਤੀਆਂ*ne0);

        src1_ਪੰਕਤੀ.ਅੰਕੜੇ = src1_contiguous.get();
        dst_ਪੰਕਤੀ.ਅੰਕੜੇ  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_ਪੰਕਤੀ_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_ਪੰਕਤੀ_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_ਪੰਕਤੀ_mapping> & routed_ਪੰਕਤੀ_src = ctx.mmid_ਪੰਕਤੀ_mapping_host;

        mmid_counting_ਛਾਂਟੋ_ਪੰਕਤੀਆਂ(ids, ids_host.ਅੰਕੜੇ(), n_ids, n_as, n_routed_ਪੰਕਤੀਆਂ,
                                expert_ਪੰਕਤੀ_counts, expert_ਪੰਕਤੀ_offsets, routed_ਪੰਕਤੀ_src);

        ggml_sycl_pool_alloc<mmid_ਪੰਕਤੀ_mapping> dev_ਪੰਕਤੀ_mapping(ctx.pool(), n_routed_ਪੰਕਤੀਆਂ);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dev_ਪੰਕਤੀ_mapping.get(), routed_ਪੰਕਤੀ_src.ਅੰਕੜੇ(), n_routed_ਪੰਕਤੀਆਂ*ਆਕਾਰ(mmid_ਪੰਕਤੀ_mapping))));

        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ max_work_ਸਮੂਹ_ਮਾਪ = ggml_sycl_info().max_work_ਸਮੂਹ_sizes[ctx.device];
        ਪ੍ਰਤਿਜ੍ਞਾ(max_work_ਸਮੂਹ_ਮਾਪ % (WARP_ਮਾਪ * WARP_ਮਾਪ) == 0);

        {
            sycl::range<3> ਖੰਡ_dims(1, 1, std::min((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ne10, max_work_ਸਮੂਹ_ਮਾਪ));
            sycl::range<3> grid_dims(1, 1, n_routed_ਪੰਕਤੀਆਂ);
            stream->submit([&](sycl::handler &cgh) {
                ਅੱਖਰ *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_ਪੰਕਤੀ_mapping *__restrict dev_ਪੰਕਤੀ_mapping_get =
                    dev_ਪੰਕਤੀ_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                    [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                        k_copy_src1_ਮੰਜ਼ਿਲ_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_ਪੰਕਤੀ_mapping_get,
                            ne11, ne10, nb11, nb12,
                            ਵਸਤੂ_ct1);
                    });
            });
        }

        ਤਾਰਕਿਕ grouped = ਝੂਠ;
        ਜੇ (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne10 && nb1 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            ਜੇ (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ਅਚਲ ਤਾਰਕਿਕ src0_reordered =
                src0->extra && ((ਅਚਲ ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ਅਚਲ ਦਸ਼ਮਲਵ *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (ਦਸ਼ਮਲਵ *) dst_contiguous.get(),
                                                     expert_ਪੰਕਤੀ_offsets.ਅੰਕੜੇ(), n_as, ne01, ne10, n_routed_ਪੰਕਤੀਆਂ,
                                                     ctx.mmid_tile_ਸਮਾਂ_ਸਾਰਣੀ_host, ctx.pool(), stream);
        }

        ਲਈ (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ਅਚਲ int64_t ਅੰਕ_src1_ਪੰਕਤੀਆਂ = expert_ਪੰਕਤੀ_counts[i02];

            ਜੇ (ਅੰਕ_src1_ਪੰਕਤੀਆਂ == 0) {
                ਜਾਰੀ;
            }

            ਅਚਲ int64_t expert_ਪੰਕਤੀ_ਵਿੱਥ = expert_ਪੰਕਤੀ_offsets[i02];

            src0_ਪੰਕਤੀ.ਅੰਕੜੇ = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne10);
            GGML_ASSERT(nb1 == ਆਕਾਰ(ਦਸ਼ਮਲਵ)*ne0);
            src1_ਪੰਕਤੀ.ਅੰਕੜੇ = src1_contiguous.get() + expert_ਪੰਕਤੀ_ਵਿੱਥ*nb11;
            src1_ਪੰਕਤੀ.ne[1] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ;

            src1_ਪੰਕਤੀ.nb[1] = nb11;
            src1_ਪੰਕਤੀ.nb[2] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ*nb11;
            src1_ਪੰਕਤੀ.nb[3] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ*nb11;

            dst_ਪੰਕਤੀ.ਅੰਕੜੇ = dst_contiguous.get() + expert_ਪੰਕਤੀ_ਵਿੱਥ*nb1;
            dst_ਪੰਕਤੀ.ne[1] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ;
            dst_ਪੰਕਤੀ.nb[1] = nb1;
            dst_ਪੰਕਤੀ.nb[2] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ*nb1;
            dst_ਪੰਕਤੀ.nb[3] = ਅੰਕ_src1_ਪੰਕਤੀਆਂ*nb1;

            ggml_sycl_mul_mat(ctx, &src0_ਪੰਕਤੀ, &src1_ਪੰਕਤੀ, &dst_ਪੰਕਤੀ);
        }

        {
            sycl::range<3> ਖੰਡ_dims(1, 1, std::min((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ne0, max_work_ਸਮੂਹ_ਮਾਪ));
            sycl::range<3> grid_dims(1, 1, n_routed_ਪੰਕਤੀਆਂ);
            stream->submit([&](sycl::handler &cgh) {
                ਅਚਲ ਅੱਖਰ *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ਅਚਲ mmid_ਪੰਕਤੀ_mapping *__restrict dev_ਪੰਕਤੀ_mapping_get =
                    dev_ਪੰਕਤੀ_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ਖੰਡ_dims, ਖੰਡ_dims),
                    [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_ਪੰਕਤੀ_mapping_get,
                                                   ne0, nb1, nb2, ਵਸਤੂ_ct1);
                    });
            });
        }
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_diag_ਨਕਾਬ_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_ਨਕਾਬ_inf(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਜੋੜ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ਜੋੜ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਜੋੜ_ਪੰਕਤੀਆਂ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਔਸਤ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_ਔਸਤ(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

ਸਥਿਰ ਖਾਲੀ ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


ਸਥਿਰ ਖਾਲੀ ggml_sycl_ਸੈੱਟ_main_device(ਅਚਲ ਪੂਰਨ main_device) ਕੋਸ਼ਿਸ਼ {
    ਜੇ (dpct::get_current_device_id() == static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ> (main_device)) {
        ਵਾਪਸ;
    }
    check_allow_gpu_ਸੂਚਕਾਂਕ(main_device);
    dpct::select_device(main_device);

    ਜੇ (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ਬਣਤਰ ggml_tensor * dst) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    ਜੇ (!g_sycl_loaded) ਵਾਪਸ ਝੂਠ;

    ਜੇ (dst->src[0] != ਖਾਲੀ_ਸੂਚਕ && ggml_backend_ਬਫਰ_is_sycl_ਵੰਡੋ(dst->src[0]->ਬਫਰ)) {
        ggml_sycl_ਸੈੱਟ_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    ਚੋਣ (dst->op) {
        ਹਾਲਤ GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
            ggml_sycl_repeat_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ggml_sycl_get_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
            ggml_sycl_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            ggml_sycl_op_ਸੈੱਟ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            ggml_sycl_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_ਜੋੜੋ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
            ggml_sycl_ਜੋੜੋ_id(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
            ggml_sycl_ਗਿਣਤੀ_equal(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(dst)) {
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                    ggml_sycl_ਕਦਮ(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                    ggml_sycl_gelu_ਤੇਜ਼(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                    ggml_sycl_ਗੇੜਾ(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(dst)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    ਰੋਕੋ;
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                    ggml_sycl_geglu_ਤੇਜ਼(ctx, dst);
                    ਰੋਕੋ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            ggml_sycl_ਸਮੂਹ_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
            ggml_sycl_rms_norm_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਜੇ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            ਜੇ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ਵਾਪਸ ਝੂਠ;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ggml_sycl_op_ਬਾਹਰਲਾ_prod(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            ggml_sycl_diag_ਨਕਾਬ_inf(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ:
            ggml_sycl_op_soft_max_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
            ggml_sycl_cross_entropy_loss_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
            ggml_sycl_rope_ਪਿੱਠ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            ggml_sycl_im2col(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
            ggml_sycl_im2col_3d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ:
            ggml_sycl_ਜੋੜ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            ggml_sycl_ਜੋੜ_ਪੰਕਤੀਆਂ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            ggml_sycl_ਔਸਤ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K:
            ggml_sycl_op_ਸਿਖਰ_k(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
            ggml_sycl_op_gated_ਰੇਖਿਕ_attn(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
            ggml_sycl_opt_ਕਦਮ_adamw(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            ggml_sycl_opt_ਕਦਮ_sgd(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ggml_sycl_ssm_ਜਾਂਚ(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            ਰੋਕੋ;
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    ਵਾਪਸ ਸੱਚ;
} ਫੜੋ (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

GGML_API ਖਾਲੀ ggml_backend_sycl_get_device_description(ਪੂਰਨ device, ਅੱਖਰ *description,
                                      size_t description_ਮਾਪ) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_ਮਾਪ, "%s", prop.get_name());
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਤਾਰਕਿਕ sycl_get_mem_info(ਪੂਰਨ device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    ਸਮ੍ਰਿਤੀ_APIType mem_api_type = ਸਮ੍ਰਿਤੀ_APIType::ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_ਨਤੀਜਾ_SUCCESS ?
        (ਸਮ੍ਰਿਤੀ_APIType) g_ggml_sycl_get_mem_api : ਸਮ੍ਰਿਤੀ_APIType::ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL;
#ਨਹੀਂਤਾਂ
    mem_api_type = ਸਮ੍ਰਿਤੀ_APIType::ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL;
#ਜੇ_ਅੰਤ
    ਤਾਰਕਿਕ res = get_ਸਮ੍ਰਿਤੀ_ਮਾਪ(dpct::dev_mgr::instance().get_device(device),
        *ਮੁਕਤ, *ਕੁੱਲ, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *ਕੁੱਲ, *ਮੁਕਤ);
    ਵਾਪਸ res;
}

ਖਾਲੀ ggml_backend_sycl_get_device_ਸਮ੍ਰਿਤੀ(ਪੂਰਨ device, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ਜੇ (!sycl_get_mem_info(device, ਮੁਕਤ, ਕੁੱਲ)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *ਮੁਕਤ, *ਕੁੱਲ);
} ਫੜੋ (ਅਚਲ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ਵਾਪਸ sycl_ctx->name.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ਮਿਟਾਓ sycl_ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਸੈੱਟ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ਅਚਲ ਖਾਲੀ *ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                               size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_ਬਫਰ_type(sycl_ctx->device) && "unsupported buffer type");
    ਅਚਲ ਕਤਾਰ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->ਸਮ੍ਰਿਤੀ_ਨਕਲ((ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, ਅੰਕੜੇ, size)));
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ਅਚਲ ggml_tensor *tensor,
                                               ਖਾਲੀ *ਅੰਕੜੇ, size_t ਵਿੱਥ,
                                               size_t size) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, ਵਿੱਥ);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_ਬਫਰ_t buf = tensor->view_src ? tensor->view_src->ਬਫਰ : tensor->ਬਫਰ;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_ਬਫਰ_type(sycl_ctx->device) && "unsupported buffer type");
    ਅਚਲ ਕਤਾਰ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->ਸਮ੍ਰਿਤੀ_ਨਕਲ(
        ਅੰਕੜੇ, (ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + ਵਿੱਥ, size)));
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ਅਚਲ ggml_tensor *src,
                                               ggml_tensor *dst) ਕੋਸ਼ਿਸ਼ {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ਤਾਰਕਿਕ is_cpy_supported                = dst->ਬਫਰ->buft == ggml_backend_sycl_ਬਫਰ_type(sycl_ctx->device) &&
                            ggml_backend_ਬਫਰ_is_sycl(src->ਬਫਰ);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ਜੇ (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ਅਚਲ ਕਤਾਰ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->ਸਮ੍ਰਿਤੀ_ਨਕਲ(
            dst->ਅੰਕੜੇ, src->ਅੰਕੜੇ, ggml_nbytes(dst))));
        ਵਾਪਸ ਸੱਚ;
    }

    ਵਾਪਸ ਝੂਠ;
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_synchronize(ggml_backend_t backend) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ਅਚਲ ਕਤਾਰ_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_ਅਣਵਰਤਿਆ(backend);
}
ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_sycl_is_view_or_noop(ਅਚਲ ggml_tensor * t) {
    ਵਾਪਸ ggml_is_ਖਾਲੀ_ਥਾਵਾਂ(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
ਸਥਿਰ ਪੂਰਨ ggml_sycl_try_gdn_cache_fusion(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    ਜੇ (!g_ggml_sycl_enable_fusion) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ggml_tensor * gdn = cgraph->nodes[ਗੰਢ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    ਜੇ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ggml_tensor * src_v     = gdn->src[2];
    ਅਚਲ int64_t       S_v       = src_v->ne[0];
    ਅਚਲ int64_t       H         = src_v->ne[1];
    ਅਚਲ int64_t       n_ਟੋਕਨ  = src_v->ne[2];
    ਅਚਲ int64_t       n_seqs    = src_v->ne[3];
    ਅਚਲ int64_t       D         = S_v * S_v * H;
    ਅਚਲ int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ਅਚਲ int64_t       n_written = std::min<int64_t>(n_ਟੋਕਨ, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ਅਚਲ size_t ਪੂਛ_off = ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, S_v * H * n_ਟੋਕਨ * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ਅਚਲ ggml_tensor * cpy  = ਖਾਲੀ_ਸੂਚਕ;
    ਪੂਰਨ                 skip = 0;
    ਲਈ (ਪੂਰਨ j = ਗੰਢ_idx + 1; j < cgraph->n_nodes && cpy == ਖਾਲੀ_ਸੂਚਕ; ++j) {
        ਅਚਲ ggml_tensor * n = cgraph->nodes[j];
        ਜੇ (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }
        ਜੇ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ਵਾਪਸ 0;
        }
        cpy  = n;
        skip = j - ਗੰਢ_idx;
    }
    ਜੇ (cpy == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ਅਚਲ ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ਜੇ (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != ਪੂਛ_off ||
        !ggml_is_contiguous(src)) {
        ਵਾਪਸ 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ਅਚਲ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ਜੇ (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_ਮਾਪ(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_ਪੰਕਤੀ_ਮਾਪ(GGML_TYPE_F32, D)) {
        ਵਾਪਸ 0;
    }

    fused_state_cpy.ਅੰਕੜੇ        = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ; // rollback group 0 (newest)
    fused_state_cpy.ਥਾਂ_stride = K > 1 ? (int64_t) (dst->nb[2] / ਆਕਾਰ(ਦਸ਼ਮਲਵ)) : 0;
    ਵਾਪਸ skip;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_ਗ੍ਰਾਫ_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_ਸੈੱਟ_main_device(sycl_ctx->device);

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];
        ਜੇ (ggml_sycl_is_view_or_noop(ਗੰਢ)) {
            ਜਾਰੀ;
        }
        ਜੇ ((ਗੰਢ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ਜਾਰੀ;
        }

        ਅਚਲ ਪੂਰਨ nodes_ਮੰਜ਼ਿਲ_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        ਜੇ (nodes_ਮੰਜ਼ਿਲ_skip != 0) {
            i += nodes_ਮੰਜ਼ਿਲ_skip;
            ਜਾਰੀ;
        }
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        ਪ੍ਰਤਿਜ੍ਞਾ(ਗੰਢ->ਬਫਰ->buft == ggml_backend_sycl_ਬਫਰ_type(sycl_ctx->device));
        ਲਈ (ਪੂਰਨ j = 0; j < GGML_MAX_SRC; j++) {
            ਜੇ (ਗੰਢ->src[j] != ਖਾਲੀ_ਸੂਚਕ) {
                ਪ੍ਰਤਿਜ੍ਞਾ(ਗੰਢ->src[j]->ਬਫਰ->buft == ggml_backend_sycl_ਬਫਰ_type(sycl_ctx->device));
            }
        }
#ਜੇ_ਅੰਤ
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        ਜੇ (ਗੰਢ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ਅਚਲ ਪੂਰਨ gdn_nodes_ਮੰਜ਼ਿਲ_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            ਜੇ (gdn_nodes_ਮੰਜ਼ਿਲ_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ਗੰਢ, fused_state_cpy);
                i += gdn_nodes_ਮੰਜ਼ਿਲ_skip;
                ਜਾਰੀ;
            }
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ਜੋੜੋ }, {})) {
            ggml_sycl_op_rms_norm_fused_ਜੋੜੋ(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ਜਾਰੀ;
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;
            ਜਾਰੀ;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        ਜੇ (ਗੰਢ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;
            ਜਾਰੀ;
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_ਜੋੜੋ &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ਜੋੜੋ, GGML_OP_ਜੋੜੋ }, {})) {
            ggml_sycl_op_ਜੋੜੋ_ਜੋੜੋ_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;
            ਜਾਰੀ;
        }
        ਜੇ (ਗੰਢ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ਗੰਢ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;
            ਜਾਰੀ;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        ਜੇ (ਗੰਢ->op == GGML_OP_ਜੋੜੋ && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ਜੋੜੋ, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_ਜੋੜੋ_unary_mul_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ਜਾਰੀ;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        ਜੇ (ਗੰਢ->op == GGML_OP_L2_NORM) {
            ਅਚਲ ਪੂਰਨ l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            ਜੇ (l2_batch_skip > 0) {
                i += l2_batch_skip;
                ਜਾਰੀ;
            }
        }

        ਜੇ (ਗੰਢ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ਜੋੜੋ, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ਗੰਢ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ਜਾਰੀ;
        }

        ਜੇ (ਗੰਢ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ਗੰਢ, ਖਾਲੀ_ਸੂਚਕ, cgraph->nodes[i + 1]);
            i++;
            ਜਾਰੀ;
        }

        ਜੇ (ਗੰਢ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            ਜਾਰੀ;
        }

        ਤਾਰਕਿਕ ok = ggml_sycl_compute_forward(*sycl_ctx, ਗੰਢ);
        ਜੇ (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ਗੰਢ->name, ggml_op_name(ਗੰਢ->op));
        }
        GGML_ASSERT(ok);
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ਗ੍ਰਾਫ
ਸਥਿਰ ਤਾਰਕਿਕ check_ਗ੍ਰਾਫ_compatibility(ggml_cgraph * cgraph) {
    ਜੇ (ggml_sycl_info().device_ਗਿਣਤੀ > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        ਵਾਪਸ ਝੂਠ;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ਅਚਲ ggml_op ਗੰਢ_op = cgraph->nodes[i]->op;
        ਚੋਣ (ਗੰਢ_op) {
            ਮੂਲ:
                ਰੋਕੋ;
            ਹਾਲਤ GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ਗੰਢ_op));
                ਵਾਪਸ ਝੂਠ;
            ਹਾਲਤ GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                ਜੇ (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ਗੰਢ_op));
                    ਵਾਪਸ ਝੂਠ;
                }
        }
    }
    ਵਾਪਸ ਸੱਚ;
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_sycl_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ਸਵੈਚਾਲਿਤ * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_ਗ੍ਰਾਫ
    ਤਾਰਕਿਕ use_sycl_ਗ੍ਰਾਫ = ਝੂਠ;
    ਜੇ (g_ggml_sycl_enable_ਗ੍ਰਾਫ) {
        use_sycl_ਗ੍ਰਾਫ = check_ਗ੍ਰਾਫ_compatibility(cgraph);
    }
    ਜੇ (use_sycl_ਗ੍ਰਾਫ) {
        ਅਚਲ ਤਾਰਕਿਕ ਗ੍ਰਾਫ_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_ਗ੍ਰਾਫ);
        ਜੇ (!ਗ੍ਰਾਫ_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_ਗ੍ਰਾਫ_compute_impl(sycl_ctx, cgraph);
            ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
        }

        sycl_ex::command_ਗ੍ਰਾਫ model_sycl_ਗ੍ਰਾਫ(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_ਬਫਰ_outlives_ਗ੍ਰਾਫ{}});

        model_sycl_ਗ੍ਰਾਫ.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_ਗ੍ਰਾਫ_compute_impl(sycl_ctx, cgraph);
        model_sycl_ਗ੍ਰਾਫ.ਸਿਰਾ_recording();

        ਅਚਲ ਤਾਰਕਿਕ ਗ੍ਰਾਫ_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_ਗ੍ਰਾਫ);
        ਜੇ (!sycl_ctx->exec_ਗ੍ਰਾਫ || !ਗ੍ਰਾਫ_update_support) {
            ਸਵੈਚਾਲਿਤ exec_ਗ੍ਰਾਫ = ਗ੍ਰਾਫ_update_support ? model_sycl_ਗ੍ਰਾਫ.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_ਗ੍ਰਾਫ.finalize();
            sycl_ctx->exec_ਗ੍ਰਾਫ = std::make_unique<
                sycl_ex::command_ਗ੍ਰਾਫ<sycl_ex::ਗ੍ਰਾਫ_state::executable>>(exec_ਗ੍ਰਾਫ);
        } ਨਹੀਂਤਾਂ {
            ਕੋਸ਼ਿਸ਼ {
                sycl_ctx->exec_ਗ੍ਰਾਫ->update(model_sycl_ਗ੍ਰਾਫ);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } ਫੜੋ (sycl::exception ਅਚਲ & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                ਸਵੈਚਾਲਿਤ exec_ਗ੍ਰਾਫ = model_sycl_ਗ੍ਰਾਫ.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_ਗ੍ਰਾਫ = std::make_unique<
                    sycl_ex::command_ਗ੍ਰਾਫ<sycl_ex::ਗ੍ਰਾਫ_state::executable>>(exec_ਗ੍ਰਾਫ);
            }
        }

        sycl_ctx->stream()->ext_oneapi_ਗ੍ਰਾਫ(*(sycl_ctx->exec_ਗ੍ਰਾਫ));
    } ਨਹੀਂਤਾਂ
#ਜੇ_ਅੰਤ
    {
        ggml_backend_sycl_ਗ੍ਰਾਫ_compute_impl(sycl_ctx, cgraph);
    }
    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_event_ਰਿਕਾਰਡ(ggml_backend_t backend, ggml_backend_event_t event)
ਕੋਸ਼ਿਸ਼
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ਅਚਲ ਕਤਾਰ_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
ਫੜੋ (sycl::exception ਅਚਲ &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) ਕੋਸ਼ਿਸ਼ {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    ਜੇ (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } ਨਹੀਂਤਾਂ
        GGML_ABORT("fatal error");
} ਫੜੋ (sycl::exception ਅਚਲ& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
    std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_sycl_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ggml_backend_sycl_event_ਰਿਕਾਰਡ,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ਸਥਿਰ ggml_guid_t ggml_backend_sycl_guid() {
    ਸਥਿਰ ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    ਵਾਪਸ &guid;
}

ਤਾਰਕਿਕ ggml_backend_is_sycl(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

ਪੂਰਨ ggml_backend_sycl_get_device_ਗਿਣਤੀ(ਖਾਲੀ) {
    ਵਾਪਸ ggml_sycl_info().device_ਗਿਣਤੀ;
}


// backend device

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ ctx->description.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ਜੇ (!sycl_get_mem_info(ctx->device, ਮੁਕਤ, ਕੁੱਲ)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *ਕੁੱਲ, *ਮੁਕਤ);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *ਮੁਕਤ, *ਕੁੱਲ);
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਅਚਲ sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    ਜੇ (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);

    ਤਾਰਕਿਕ host_ਬਫਰ = getenv("GGML_SYCL_NO_PINNED") == ਖਾਲੀ_ਸੂਚਕ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_NO_PEER_COPY
    ਤਾਰਕਿਕ events = ਝੂਠ;
#ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ events = ਸੱਚ;
#ਜੇ_ਅੰਤ

    props->caps = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ host_ਬਫਰ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ events,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    GGML_ਅਣਵਰਤਿਆ(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ ggml_backend_sycl_init(ctx->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ ggml_backend_sycl_ਬਫਰ_type(ctx->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_sycl_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ਵਾਪਸ ggml_backend_sycl_host_ਬਫਰ_type_for_device(ctx->device);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_sycl_device_ਬਫਰ_from_host_ptr(ggml_backend_dev_t dev, ਖਾਲੀ * ptr, size_t size, size_t max_tensor_ਮਾਪ) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    GGML_ਅਣਵਰਤਿਆ(ptr);
    GGML_ਅਣਵਰਤਿਆ(size);
    GGML_ਅਣਵਰਤਿਆ(max_tensor_ਮਾਪ);
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਤਾਰਕਿਕ do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    ਪੂਰਨ device = sycl_ctx->device;
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                ਜੇ (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    ਵਾਪਸ ਸੱਚ;
                }
                ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            ਵਾਪਸ (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_ਮਾਪ(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_ਮਾਪ(op->src[0]->type));
        ਹਾਲਤ GGML_OP_CONV_2D_DW:
        ਹਾਲਤ GGML_OP_CONV_TRANSPOSE_2D:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_UNARY:
            ਚੋਣ (ggml_get_unary_op(op)) {
                ਹਾਲਤ GGML_UNARY_OP_SGN:
                ਹਾਲਤ GGML_UNARY_OP_ABS:
                ਹਾਲਤ GGML_UNARY_OP_NEG:
                ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                ਹਾਲਤ GGML_UNARY_OP_RELU:
                ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_TANH:
                ਹਾਲਤ GGML_UNARY_OP_GELU:
                ਹਾਲਤ GGML_UNARY_OP_SILU:
                ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                ਹਾਲਤ GGML_UNARY_OP_EXP:
                ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                ਹਾਲਤ GGML_UNARY_OP_ELU:
                ਹਾਲਤ GGML_UNARY_OP_XIELU:
                ਹਾਲਤ GGML_UNARY_OP_CEIL:
                    ਵਾਪਸ ਸੱਚ;
                ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                    ਵਾਪਸ ਸੱਚ;
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_GLU:
            ਚੋਣ (ggml_get_glu_op(op)) {
                ਹਾਲਤ GGML_GLU_OP_REGLU:
                ਹਾਲਤ GGML_GLU_OP_GEGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_OAI:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ERF:
                ਹਾਲਤ GGML_GLU_OP_GEGLU_ਤੇਜ਼:
                ਹਾਲਤ GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    ਵਾਪਸ ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                ਮੂਲ:
                    ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_MUL_MAT:
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            {
                ਬਣਤਰ ggml_tensor * a = op->src[0];
                ਬਣਤਰ ggml_tensor * b = op->src[1];

                ਜੇ (a->ne[3] != b->ne[3]) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                ਜੇ (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  ਵਾਪਸ ਝੂਠ;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                ਜੇ (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਜੇ (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_ਬਾਹਰਲਾ_PROD:
            ਵਾਪਸ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            {
                ਚੋਣ (op->src[0]->type) {
                    ਹਾਲਤ GGML_TYPE_I32:
                    ਹਾਲਤ GGML_TYPE_F16:
                    ਹਾਲਤ GGML_TYPE_BF16:
                    ਹਾਲਤ GGML_TYPE_F32:
                    ਹਾਲਤ GGML_TYPE_Q1_0:
                    ਹਾਲਤ GGML_TYPE_MXFP4:
                    ਹਾਲਤ GGML_TYPE_NVFP4:
                    ਹਾਲਤ GGML_TYPE_IQ2_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ2_XS:
                    ਹਾਲਤ GGML_TYPE_IQ2_S:
                    ਹਾਲਤ GGML_TYPE_IQ3_XXS:
                    ਹਾਲਤ GGML_TYPE_IQ1_S:
                    ਹਾਲਤ GGML_TYPE_IQ1_M:
                    ਹਾਲਤ GGML_TYPE_IQ3_S:
                    ਹਾਲਤ GGML_TYPE_IQ4_NL:
                    ਹਾਲਤ GGML_TYPE_IQ4_XS:
                    ਹਾਲਤ GGML_TYPE_Q2_K:
                    ਹਾਲਤ GGML_TYPE_Q3_K:
                    ਹਾਲਤ GGML_TYPE_Q4_0:
                    ਹਾਲਤ GGML_TYPE_Q4_1:
                    ਹਾਲਤ GGML_TYPE_Q4_K:
                    ਹਾਲਤ GGML_TYPE_Q5_0:
                    ਹਾਲਤ GGML_TYPE_Q5_1:
                    ਹਾਲਤ GGML_TYPE_Q5_K:
                    ਹਾਲਤ GGML_TYPE_Q6_K:
                    ਹਾਲਤ GGML_TYPE_Q8_0:
                        ਵਾਪਸ ਸੱਚ;
                    ਮੂਲ:
                        ਵਾਪਸ ਝੂਠ;
                }
            }
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ_ਪਿੱਠ:
            // return true;
             ਵਾਪਸ op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         ਹਾਲਤ GGML_OP_ਸੈੱਟ:
               ਵਾਪਸ (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            {
                ਜੇ (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    ਵਾਪਸ ਝੂਠ;
                }
                ਸਵੈਚਾਲਿਤ res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                ਵਾਪਸ res;
            }
        ਹਾਲਤ GGML_OP_DSV4_HC_PRE:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_DSV4_HC_COMB:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_DSV4_HC_POST:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == ਖਾਲੀ_ਸੂਚਕ || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_LIGHTNING_INDEXER:
            ਵਾਪਸ op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_ਮਾਪ * 8;
        ਹਾਲਤ GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                ਜੇ (src0_type == GGML_TYPE_F16) {
                    ਜੇ (src1_type == GGML_TYPE_Q2_K ||
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
                        ਵਾਪਸ ਝੂਠ;
                    }
                }

                ਜੇ (src0_type == GGML_TYPE_BF16) {
                    ਜੇ (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        ਵਾਪਸ ਝੂਠ;
                    }
                }

                ਜੇ (src0_type == GGML_TYPE_F32) {
                    ਜੇ (src1_type == GGML_TYPE_Q2_K ||
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
                        ਵਾਪਸ ਝੂਠ;
                    }
                }

                ਜੇ (src1_type == GGML_TYPE_F32) {
                    ਜੇ (src0_type == GGML_TYPE_Q1_0 ||
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
                        ਵਾਪਸ ਝੂਠ;
                    }
                }

                ਜੇ (src0_type == src1_type) {
                    ਜੇ (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        ਵਾਪਸ ਝੂਠ;
                    }
                }

                ਜੇ (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    ਵਾਪਸ ਝੂਠ;
                }

                ਵਾਪਸ ਸੱਚ;
            }
        ਹਾਲਤ GGML_OP_REPEAT_ਪਿੱਠ:
            {
                ggml_type src0_type = op->src[0]->type;
                ਵਾਪਸ src0_type == GGML_TYPE_F32;
            }
        ਹਾਲਤ GGML_OP_CONCAT:
        ਹਾਲਤ GGML_OP_DUP:
        ਹਾਲਤ GGML_OP_ARGMAX:
        ਹਾਲਤ GGML_OP_NONE:
        ਹਾਲਤ GGML_OP_RESHAPE:
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_ADD1:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ_ID:
        ਹਾਲਤ GGML_OP_SUB:
        ਹਾਲਤ GGML_OP_ਗਿਣਤੀ_EQUAL:
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_DIV:
        ਹਾਲਤ GGML_OP_REPEAT:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_PAD_REFLECT_1D:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SQR:
        ਹਾਲਤ GGML_OP_SQRT:
        ਹਾਲਤ GGML_OP_SIN:
        ਹਾਲਤ GGML_OP_COS:
        ਹਾਲਤ GGML_OP_CLAMP:
        ਹਾਲਤ GGML_OP_LOG:
        ਹਾਲਤ GGML_OP_NORM:
        ਹਾਲਤ GGML_OP_L2_NORM:
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
        ਹਾਲਤ GGML_OP_RMS_NORM:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_RMS_NORM_ਪਿੱਠ:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]);
        ਹਾਲਤ GGML_OP_SCALE:
            // the scale kernel is f32 only
            ਵਾਪਸ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_CONT:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_TRI:
            {
                ਅਚਲ ggml_tensor * src0 = op->src[0];
                ਵਾਪਸ src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        ਹਾਲਤ GGML_OP_DIAG_ਨਕਾਬ_INF:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_SOFT_MAX_ਪਿੱਠ: {
            ਦਸ਼ਮਲਵ max_bias = 0.0f;
            ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 1, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            ਵਾਪਸ max_bias == 0.0f;
        }
        ਹਾਲਤ GGML_OP_ROPE:
        ਹਾਲਤ GGML_OP_ROPE_ਪਿੱਠ:
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ_3_D:
        ਹਾਲਤ GGML_OP_UPSCALE:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_COL2IM_1D:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#ਜੇ_ਅੰਤ
                   ) &&
                   op->src[0]->type == op->type;
        ਹਾਲਤ GGML_OP_CONV_3D: {
            ਅਚਲ int32_t * opts = (ਅਚਲ int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ਅਚਲ int64_t knl_n_ਕੁੱਲ = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ਅਚਲ int64_t patch_ਕੁੱਲ = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            ਵਾਪਸ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_ਕੁੱਲ > 0 && patch_ਕੁੱਲ > 0;
        }
        ਹਾਲਤ GGML_OP_ਜੋੜ:
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            ਵਾਪਸ ggml_is_contiguous(op->src[0]);
        ਹਾਲਤ GGML_OP_ARGSORT:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_ਸਿਖਰ_K: {
            ਅਚਲ ggml_tensor * src0 = op->src[0];
            ਅਚਲ ਪੂਰਨ k = op->ne[0];
            ਵਾਪਸ src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        ਹਾਲਤ GGML_OP_POOL_2D:
        ਹਾਲਤ GGML_OP_POOL_1D:
        ਹਾਲਤ GGML_OP_ACC:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_PAD:
            ਜੇ (ggml_get_op_params_i32(op, 8) != 0) {
                ਵਾਪਸ ਝੂਠ;
            }
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_LEAKY_RELU:
        ਹਾਲਤ GGML_OP_TIMESTEP_EMBEDDING:
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
        ਹਾਲਤ GGML_OP_GATED_ਰੇਖਿਕ_ATTN:
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_ADAMW:
        ਹਾਲਤ GGML_OP_OPT_ਕਦਮ_SGD:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            ਵਾਪਸ op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ROLL:
            ਵਾਪਸ op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_ARANGE:
            ਵਾਪਸ op->type == GGML_TYPE_F32;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            ਜੇ (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                ਵਾਪਸ (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_ਮਾਪ == 0;
            } ਨਹੀਂਤਾਂ {
                // TODO Mamba-1 not yet ported to SYCL
                ਵਾਪਸ ਝੂਠ;
            }
        ਹਾਲਤ GGML_OP_FILL:
        ਹਾਲਤ GGML_OP_CUMSUM:
        ਹਾਲਤ GGML_OP_DIAG:
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS:
        ਹਾਲਤ GGML_OP_CROSS_ENTROPY_LOSS_ਪਿੱਠ:
            ਵਾਪਸ ਸੱਚ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            ਵਾਪਸ op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਵਾਪਸ ggml_sycl_flash_attn_ext_supported(device, op);
        ਮੂਲ:
            ਵਾਪਸ ਝੂਠ;
    }

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ਤਾਰਕਿਕ res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    ਵਾਪਸ res;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    ਜੇ (buft->iface.get_name != ggml_backend_sycl_ਬਫਰ_type_get_name) {
        ਵਾਪਸ ਝੂਠ;
    }
    ggml_backend_sycl_ਬਫਰ_type_context * buft_ctx = (ggml_backend_sycl_ਬਫਰ_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ buft_ctx->device == sycl_ctx->device;
}

ਸਥਿਰ int64_t get_op_batch_ਮਾਪ(ਅਚਲ ggml_tensor * op) {
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            ਵਾਪਸ 0;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            ਵਾਪਸ op->ne[1];
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
        ਹਾਲਤ GGML_OP_ROPE:
            ਵਾਪਸ op->ne[2];
        ਮੂਲ:
            ਵਾਪਸ ggml_nrows(op);
    }
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ਵਾਪਸ get_op_batch_ਮਾਪ(op) >= sycl_ctx->op_offload_min_batch_ਮਾਪ;
}

ਸਥਿਰ ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_NO_PEER_COPY
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
#ਨਹੀਂਤਾਂ
  sycl::event *event_ptr = ਨਵਾਂ sycl::event();

  ਵਾਪਸ ਨਵਾਂ ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) ਕੋਸ਼ਿਸ਼ {
  GGML_ਅਣਵਰਤਿਆ(dev);
  ਜੇ (event == ਖਾਲੀ_ਸੂਚਕ) {
    ਵਾਪਸ;
  }

  ਜੇ (event->context != ਖਾਲੀ_ਸੂਚਕ) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    ਮਿਟਾਓ sycl_event;
    event->context = ਖਾਲੀ_ਸੂਚਕ;
  }

  ਮਿਟਾਓ event;
} ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}


ਸਥਿਰ ਖਾਲੀ ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) ਕੋਸ਼ਿਸ਼ {
  GGML_ਅਣਵਰਤਿਆ(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} ਫੜੋ (sycl::exception ਅਚਲ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
  std::ਬਾਹਰ_ਨਿਕਲੋ(1);
}

ਸਥਿਰ ਅਚਲ ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_ਬਫਰ_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

ਬਣਤਰ ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    ਵਾਪਸ GGML_SYCL_NAME;
}

ਸਥਿਰ size_t ggml_backend_sycl_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    ਵਾਪਸ ctx->devices.size();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t ਸੂਚਕਾਂਕ) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(ਸੂਚਕਾਂਕ < ctx->devices.size());
    ਵਾਪਸ ctx->devices[ਸੂਚਕਾਂਕ];
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

ਬਣਤਰ ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_ਬਾਹਰਲਾ[2] = { ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ };
    uint8_t * host_ਵਿੱਚ[2]  = { ਖਾਲੀ_ਸੂਚਕ, ਖਾਲੀ_ਸੂਚਕ };
    size_t    host_ਬਾਈਟਾਂ  = 0;

    ਖਾਲੀ free_host() {
        ਲਈ (ਪੂਰਨ i = 0; i < 2; ++i) {
            ਅਚਲ sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            ਜੇ (host_ਬਾਹਰਲਾ[i]) { sycl::ਮੁਕਤ(host_ਬਾਹਰਲਾ[i], sctx); host_ਬਾਹਰਲਾ[i] = ਖਾਲੀ_ਸੂਚਕ; }
            ਜੇ (host_ਵਿੱਚ[i])  { sycl::ਮੁਕਤ(host_ਵਿੱਚ[i],  sctx); host_ਵਿੱਚ[i]  = ਖਾਲੀ_ਸੂਚਕ; }
        }
        host_ਬਾਈਟਾਂ = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
ਸਥਿਰ ਖਾਲੀ ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, ਕਤਾਰ_ptr q0, ਕਤਾਰ_ptr q1,
                                    ਖਾਲੀ * dst0, ਅਚਲ ਖਾਲੀ * src1, ਖਾਲੀ * dst1, ਅਚਲ ਖਾਲੀ * src0,
                                    size_t nbytes) {
    ਜੇ (comm->host_ਬਾਈਟਾਂ < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_ਬਾਹਰਲਾ[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_ਵਿੱਚ[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_ਬਾਹਰਲਾ[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_ਵਿੱਚ[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_ਬਾਹਰਲਾ[0] && comm->host_ਵਿੱਚ[0] && comm->host_ਬਾਹਰਲਾ[1] && comm->host_ਵਿੱਚ[1]);
        comm->host_ਬਾਈਟਾਂ = nbytes;
    }

    sycl::event e0 = q0->ਸਮ੍ਰਿਤੀ_ਨਕਲ(comm->host_ਬਾਹਰਲਾ[0], src0, nbytes);
    sycl::event e1 = q1->ਸਮ੍ਰਿਤੀ_ਨਕਲ(comm->host_ਬਾਹਰਲਾ[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(comm->host_ਵਿੱਚ[0], comm->host_ਬਾਹਰਲਾ[1], nbytes);
    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(comm->host_ਵਿੱਚ[1], comm->host_ਬਾਹਰਲਾ[0], nbytes);
    q0->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst0, comm->host_ਵਿੱਚ[0], nbytes);
    q1->ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst1, comm->host_ਵਿੱਚ[1], nbytes);
}

ਖਾਲੀ * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) ਕੋਸ਼ਿਸ਼ {
    ਲਈ (size_t i = 0; i < n_backends; ++i) {
        ਜੇ (!ggml_backend_is_sycl(backends[i])) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    ਜੇ (n_backends != 2) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਸਵੈਚਾਲਿਤ * ctx = ਨਵਾਂ ggml_backend_sycl_comm_context;
    ctx->backends.ਸੌਂਪੋ(backends, backends + n_backends);
    ਸਵੈਚਾਲਿਤ * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    ਸਵੈਚਾਲਿਤ * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    ਵਾਪਸ ctx;
}
ਫੜੋ (ਅਚਲ sycl::exception &) { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }
ਫੜੋ (...)                     { ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ; }

ਖਾਲੀ ggml_backend_sycl_comm_free(ਖਾਲੀ * comm_ctx_v) {
    ਸਵੈਚਾਲਿਤ * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ਜੇ (comm_ctx == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    ਜੇ (comm_ctx->backends.size() == 2) {
        ਸਵੈਚਾਲਿਤ * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        ਸਵੈਚਾਲਿਤ * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        ਕੋਸ਼ਿਸ਼ {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } ਫੜੋ (...) { /* best effort during shutdown */ }
    }

    ਮਿਟਾਓ comm_ctx;
}

ਤਾਰਕਿਕ ggml_backend_sycl_comm_allreduce_tensor(ਖਾਲੀ * comm_ctx_v, ਬਣਤਰ ggml_tensor ** tensors) ਕੋਸ਼ਿਸ਼ {
    ਜੇ (comm_ctx_v == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਸਵੈਚਾਲਿਤ * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ਅਚਲ size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    ਜੇ (n_backends != 2) {
        ਵਾਪਸ ਝੂਠ;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    ਜੇ (tensors[0]->type != tensors[1]->type) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਜੇ (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਅਚਲ int64_t nelem  = ggml_nelements(tensors[0]);
    ਅਚਲ size_t  nbytes = ggml_nbytes(tensors[0]);
    ਜੇ (nelem == 0) {
        ਵਾਪਸ ਸੱਚ;
    }

    ਸਵੈਚਾਲਿਤ * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    ਸਵੈਚਾਲਿਤ * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    ਕਤਾਰ_ptr q0 = ctx0->stream();
    ਕਤਾਰ_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    ਜੇ (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    ਜੇ (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->ਅੰਕੜੇ;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->ਅੰਕੜੇ;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->ਅੰਕੜੇ, f16_tmp1, tensors[0]->ਅੰਕੜੇ, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((ਦਸ਼ਮਲਵ) f16_out0[i] + (ਦਸ਼ਮਲਵ) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((ਦਸ਼ਮਲਵ) f16_out1[i] + (ਦਸ਼ਮਲਵ) f16_tmp1[i]);
            });
        });
        ਵਾਪਸ ਸੱਚ;
    }

    ਦਸ਼ਮਲਵ * out0 = (ਦਸ਼ਮਲਵ *) tensors[0]->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * out1 = (ਦਸ਼ਮਲਵ *) tensors[1]->ਅੰਕੜੇ;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    ਸਥਿਰ constexpr int64_t BF16_ਹੱਦ = 32768;

    ਜੇ (nelem < BF16_ਹੱਦ) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        ਦਸ਼ਮਲਵ * tmp0 = (ਦਸ਼ਮਲਵ *) buf0;
        ਦਸ਼ਮਲਵ * tmp1 = (ਦਸ਼ਮਲਵ *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->ਅੰਕੜੇ, tmp1, tensors[0]->ਅੰਕੜੇ, nbytes);

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
        ਵਾਪਸ ਸੱਚ;
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
        outbox0[i] = (uint16_t) (sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    ਅਚਲ size_t bf16_ਬਾਈਟਾਂ = nelem * ਆਕਾਰ(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_ਬਾਈਟਾਂ);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<ਦਸ਼ਮਲਵ>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<ਦਸ਼ਮਲਵ>(((uint32_t) inbox1[i]) << 16);
        });
    });

    ਵਾਪਸ ਸੱਚ;
}
ਫੜੋ (ਅਚਲ sycl::exception &) { ਵਾਪਸ ਝੂਠ; }
ਫੜੋ (...)                     { ਵਾਪਸ ਝੂਠ; }

ਸਥਿਰ ਖਾਲੀ *ggml_backend_sycl_reg_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ *name) {
    GGML_ਅਣਵਰਤਿਆ(reg);

    ਜੇ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_sycl_ਵੰਡੋ_ਬਫਰ_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    ਜੇ (strcmp(name, "ggml_backend_comm_init") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_sycl_comm_init;
    }
    ਜੇ (strcmp(name, "ggml_backend_comm_free") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_sycl_comm_free;
    }
    ਜੇ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ਵਾਪਸ (ਖਾਲੀ *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_ਅਣਵਰਤਿਆ(name);
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਅਚਲ ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_ਗਿਣਤੀ,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_ਪਤਾ,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    ਸਥਿਰ ggml_backend_reg reg;
    ਸਥਿਰ ਤਾਰਕਿਕ initialized = ਝੂਠ;

    {
        ਸਥਿਰ std::mutex mutex;
        std::lock_guard<std::mutex> ਤਾਲਾ(mutex);
        ਜੇ (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = ਨਵਾਂ ggml_backend_sycl_reg_context;
            ਅਚਲ ਪੂਰਨ min_batch_ਮਾਪ = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            ਲਈ (ਪੂਰਨ i = 0; i < ggml_sycl_info().device_ਗਿਣਤੀ; i++) {
                ggml_backend_sycl_device_context * dev_ctx = ਨਵਾਂ ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_ਸੈੱਟ_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_ਮਾਪ = min_batch_ਮਾਪ;

                ggml_backend_dev_t dev = ਨਵਾਂ ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = ਸੱਚ;
    }

    ਵਾਪਸ &reg;
}

ggml_backend_t ggml_backend_sycl_init(ਪੂਰਨ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_ਸੂਚਕਾਂਕ(device);

    ggml_backend_sycl_context * ctx = ਨਵਾਂ ggml_backend_sycl_context(device);
    ਜੇ (ctx == ਖਾਲੀ_ਸੂਚਕ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    };

    ggml_backend_t sycl_backend = ਨਵਾਂ ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    ਵਾਪਸ sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
