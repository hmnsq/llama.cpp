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

#అన్తర్భావ <algorithm>
#అన్తర్భావ <array>
#అన్తర్భావ <assert.h>
#అన్తర్భావ <atomic>
#అన్తర్భావ <cctype>
#అన్తర్భావ <cinttypes>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <float.h>
#అన్తర్భావ <limits>
#అన్తర్భావ <optional>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <vector>
#అన్తర్భావ <cmath>
#అన్తర్భావ <iostream>
#అన్తర్భావ <fstream>
#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <regex>

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/backend.hpp>
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#అన్తర్భావ <level_zero/ze_api.h>
#అన్తర్భావ <level_zero/zes_api.h>
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_SYCL_లేఖాచిత్రం) && SYCL_EXT_ONEAPI_ASYNC_జ్ఞాపకం_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#అయితే_ముగింపు
#ఒకవేళ SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#అయితే_ముగింపు
#అన్తర్భావ <sycl/half_type.hpp>

#అన్తర్భావ "ggml.h"
#అన్తర్భావ "ggml-sycl.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ "ggml-sycl/add-id.hpp"
#అన్తర్భావ "ggml-sycl/backend.hpp"
#అన్తర్భావ "ggml-sycl/common.hpp"
#అన్తర్భావ "ggml-sycl/element_wise.hpp"
#అన్తర్భావ "ggml-sycl/fwht.hpp"
#అన్తర్భావ "ggml-sycl/fused-gemm.hpp"
#అన్తర్భావ "ggml-sycl/gemm.hpp"
#అన్తర్భావ "ggml-sycl/getrows.hpp"
#అన్తర్భావ "ggml-sycl/mem.hpp"
#అన్తర్భావ "ggml-sycl/norm.hpp"
#అన్తర్భావ "ggml-sycl/presets.hpp"
#అన్తర్భావ "ggml-sycl/quantize.hpp"
#అన్తర్భావ "ggml-sycl/repeat_back.hpp"
#అన్తర్భావ "ggml-sycl/set_rows.hpp"
#అన్తర్భావ "ggml-sycl/set.hpp"
#అన్తర్భావ "ggml-sycl/dsv4-hc.hpp"
#అన్తర్భావ "ggml-sycl/lightning-indexer.hpp"
#అన్తర్భావ "ggml-sycl/conv2d.hpp"
#అన్తర్భావ "ggml-sycl/conv2d-dw.hpp"
#అన్తర్భావ "ggml-sycl/conv2d-transpose.hpp"
#అన్తర్భావ "ggml-sycl/ssm_conv.hpp"
#అన్తర్భావ "ggml-sycl/sycl_hw.hpp"
#అన్తర్భావ "ggml-sycl/ssm_scan.hpp"
#అన్తర్భావ "ggml-sycl/fill.hpp"
#అన్తర్భావ "ggml-sycl/cumsum.hpp"
#అన్తర్భావ "ggml-sycl/diag.hpp"
#అన్తర్భావ "ggml-sycl/opt-step.hpp"
#అన్తర్భావ "ggml-sycl/solve_tri.hpp"
#అన్తర్భావ "ggml-sycl/gated_delta_net.hpp"
#అన్తర్భావ "ggml-sycl/pool.hpp"
#అన్తర్భావ "ggml-sycl/cross_entropy_loss.hpp"

#నిర్వచించు MEM_కొలత_2_M	0x00200000
#నిర్వచించు MEM_కొలత_1_G	0x40000000

స్థిర తార్కిక g_sycl_loaded = అబద్ధం;
పూర్ణాంకం g_ggml_sycl_debug = 0;
పూర్ణాంకం g_ggml_sycl_dev_debug = 0;
పూర్ణాంకం g_ggml_sycl_enable_optimize = 1;
పూర్ణాంకం g_ggml_sycl_enable_లేఖాచిత్రం = 0;
పూర్ణాంకం g_ggml_sycl_enable_dnn = 1;
పూర్ణాంకం g_ggml_sycl_fa_onednn = 1;
పూర్ణాంకం g_ggml_sycl_fa_onednn_max_kv = 0;
పూర్ణాంకం g_ggml_sycl_enable_mkl_fa = 1;
పూర్ణాంకం g_ggml_sycl_memtrace = 0;
పూర్ణాంకం g_ggml_sycl_memtrace_అడుగు = 64;
పూర్ణాంకం g_ggml_sycl_enable_vmm = 1;
పూర్ణాంకం g_ggml_sycl_enable_fusion = 1;
పూర్ణాంకం g_ggml_sycl_enable_esimd = 1;
పూర్ణాంకం g_ggml_sycl_mmvq_wide = 1;
పూర్ణాంకం g_ggml_sycl_prioritize_dmmv = 0;
పూర్ణాంకం g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
పూర్ణాంకం g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
పూర్ణాంకం g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
పూర్ణాంకం g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
స్థిర స్థిరరాశి అక్షరం * ggml_sycl_dynamic_precision_పేర్లు[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
స్థిర పూర్ణాంకం ggml_sycl_get_env_precision(స్థిరరాశి అక్షరం * name, పూర్ణాంకం def) {
    స్థిరరాశి అక్షరం * env = getenv(name);
    ఒకవేళ (!env) {
        తిరిగి def;
    }
    std::string mode(env);
    కొరకు (అక్షరం & c : mode) {
        c = (అక్షరం) std::పెద్ద_అక్షరం((గుర్తులేని అక్షరం) c);
    }
    కొరకు (పూర్ణాంకం i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        ఒకవేళ (mode == ggml_sycl_dynamic_precision_పేర్లు[i]) {
            తిరిగి i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_పేర్లు[def]);
    తిరిగి def;
}
పూర్ణాంకం g_ggml_sycl_use_async_mem_op = 0;
పూర్ణాంకం g_ggml_sycl_use_async_mem_op_requested = 1;
పూర్ణాంకం g_ggml_sycl_use_level_zero_api = 0;
పూర్ణాంకం g_ggml_sycl_enable_flash_attention = 1;
పూర్ణాంకం g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
పూర్ణాంకం g_ggml_sycl_usm_system = 0;
పూర్ణాంకం g_ggml_sycl_enable_host_pinned_mem = 1;
పూర్ణాంకం g_ggml_sycl_host_pinned_mem_2g = 0;
పూర్ణాంకం g_ggml_sycl_upload_staging_slots = 4;
పూర్ణాంకం g_ggml_sycl_get_mem_api = జ్ఞాపకం_API_TYPE_LEVEL_ZERO;
పూర్ణాంకం g_ggml_sycl_enable_sparse_fa = 0;
పూర్ణాంకం g_ggml_sycl_debug_sparse_fa = 0;
పూర్ణాంకం g_ggml_sycl_sparse_fa_margin = 256;

#ఒకవేళ GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
స్థిర తార్కిక ggml_sycl_dnnl_detect_optimized_gemm(పూర్ణాంకం device) {
    ఉపయోగించి dt = dnnl::జ్ఞాపకం::దత్తాంశం_type;

    ప్రయత్నించు {
        స్థిరరాశి dnnl::జ్ఞాపకం::dims dims    = { 1, 64, 64 };
        స్థిరరాశి dnnl::జ్ఞాపకం::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.సమితి_scratchpad_బహుళకం(dnnl::scratchpad_బహుళకం::user);

        స్థిరరాశి స్వయంచాలక & q   = dpct::dev_mgr::instance().get_device(device).default_వరుస();
        స్థిరరాశి స్వయంచాలక   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        స్థిరరాశి స్వయంచాలక   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::జ్ఞాపకం::desc(dims, dt::f16, strides),
                                                        dnnl::జ్ఞాపకం::desc(dims, dt::f16, strides),
                                                        dnnl::జ్ఞాపకం::desc(dims, dt::f32, strides), attr);

        స్థిరరాశి std::string impl = pd.impl_info_str();
        ఒకవేళ (impl.find("ref") == std::string::npos) {
            తిరిగి నిజం;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } పట్టుకో (స్థిరరాశి std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    తిరిగి అబద్ధం;
}
#అయితే_ముగింపు

స్థిర ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    ప్రయత్నించు {
        info.device_లెక్క = dpct::dev_mgr::instance().device_లెక్క();
    } పట్టుకో (sycl::exception స్థిరరాశి &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_లెక్క = 0;
    }
    ఒకవేళ (info.device_లెక్క == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        తిరిగి info;
    }

    GGML_ASSERT(info.device_లెక్క <= GGML_SYCL_MAX_DEVICES);

    int64_t మొత్తం_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    కొరకు (పూర్ణాంకం i = 0; i < info.device_లెక్క; ++i) {
        dpct::device_info prop;
        స్వయంచాలక & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#ఒకవేళ !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#లేకపోతే
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        ఒకవేళ (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            స్థిరరాశి size_t physical_పుట = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_పుట);
        }
#అయితే_ముగింపు

        info.default_tensor_విభజించు[i] = మొత్తం_vram;
        మొత్తం_vram += prop.get_global_mem_కొలత();

        info.devices[i].cc =
            100 * prop.get_major_సంస్కరణ() + 10 * prop.get_minor_సంస్కరణ();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#ఒకవేళ GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#అయితే_ముగింపు
        info.devices[i].smpbo = prop.get_local_mem_కొలత();
        info.devices[i].warp_కొలత = WARP_కొలత;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_సమూహం_sizes[i] = prop.get_max_work_సమూహం_కొలత();
        info.devices[i].max_wg_చొప్పున_cu = info.max_work_సమూహం_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        ఒకవేళ (device.is_gpu() && device.default_వరుస().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = అబద్ధం;
        }

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        ఒకవేళ (info.ext_oneapi_level_zero && device.is_gpu() && device.default_వరుస().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_వరుస().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_ఫలితం_t r = zeDeviceGetProperties(ze_dev, &props);
            ఒకవేళ (r == ZE_ఫలితం_SUCCESS) {
                info.devices[i].l0_device_type_valid = నిజం;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#అయితే_ముగింపు
    }

    కొరకు (పూర్ణాంకం id = 0; id < info.device_లెక్క; ++id) {
        info.default_tensor_విభజించు[id] /= మొత్తం_vram;
    }

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#అయితే_ముగింపు

    తిరిగి info;
}

స్థిరరాశి ggml_sycl_device_info & ggml_sycl_info() {
    స్థిర ggml_sycl_device_info info = ggml_sycl_init();
    తిరిగి info;
}

స్థిర శూన్య print_device_detail(పూర్ణాంకం id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string సంస్కరణ_2;
    సంస్కరణ_2 += std::to_string(prop.get_major_సంస్కరణ());
    సంస్కరణ_2 += ".";
    సంస్కరణ_2 += std::to_string(prop.get_minor_సంస్కరణ());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    స్వయంచాలక global_mem_కొలత = prop.get_global_mem_కొలత()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), సంస్కరణ_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_సమూహం_కొలత(), prop.get_max_sub_సమూహం_కొలత(),
            global_mem_కొలత, device.get_info<sycl::info::device::driver_సంస్కరణ>().c_str());
}

స్థిర శూన్య print_device_opt_feature(పూర్ణాంకం device_లెక్క) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    కొరకు (పూర్ణాంకం id = 0; id < device_లెక్క; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      పూర్ణాంకం type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
శూన్య ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    పూర్ణాంకం device_లెక్క = dpct::dev_mgr::instance().device_లెక్క();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_లెక్క);

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

    కొరకు (పూర్ణాంకం id = 0; id < device_లెక్క; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      పూర్ణాంకం type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_లెక్క);
}

స్థిర స్థిరరాశి అక్షరం* dev2dev_int2str(పూర్ణాంకం dev2dev) {
    ఒకవేళ (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        తిరిగి "SYCL API";
    } లేకపోతే ఒకవేళ (dev2dev == DEV2DEV_MEMCPY_L0) {
        తిరిగి "Level Zero API";
    } లేకపోతే ఒకవేళ (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        తిరిగి "Host Forward";
    } లేకపోతే {
        తిరిగి "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
స్థిర ze_ఫలితం_t init_zes() {
    ze_ఫలితం_t res = zesInit(0);
    ఒకవేళ (res != ZE_ఫలితం_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (పూర్ణాంకం) res);
    }
    తిరిగి res;
}

స్థిర ze_ఫలితం_t get_zes_init_res() {
    స్థిర ze_ఫలితం_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (పూర్ణాంకం) zes_init_res);
    తిరిగి zes_init_res;
}
#అయితే_ముగింపు

స్థిర శూన్య initialize_sycl_begining() {
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#అయితే_ముగింపు
}

స్థిర శూన్య ggml_check_sycl() ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    స్థిర తార్కిక initialized = అబద్ధం;

    ఒకవేళ (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_లేఖాచిత్రం = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_అడుగు = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
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

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#లేకపోతే
        g_ggml_sycl_use_level_zero_api = 0;
#అయితే_ముగింపు
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", జ్ఞాపకం_API_TYPE_LEVEL_ZERO);
        ఒకవేళ (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = జ్ఞాపకం_API_TYPE_SYCL;
        }

#నిర్వచితమైతే SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#లేకపోతే
        g_ggml_sycl_enable_flash_attention = 0;
#అయితే_ముగింపు

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
#ఒకవేళ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_SYCL_లేఖాచిత్రం)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#అయితే_ముగింపు
#ఒకవేళ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#అయితే_ముగింపు

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#అయితే_ముగింపు
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_అడుగు);
#నిర్వచితమైతే SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#అయితే_ముగింపు

#నిర్వచితమైతే GGML_SYCL_లేఖాచిత్రం
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_లేఖాచిత్రం);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#అయితే_ముగింపు

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_పేర్లు[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_పేర్లు[g_ggml_sycl_dynamic_required_precision]);

#ఒకవేళ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#అయితే_ముగింపు

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#ఒకవేళ defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#అయితే_ముగింపు
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#లేకపోతే
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#అయితే_ముగింపు

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
#ఒకవేళ defined(GGML_SYCL_లేఖాచిత్రం) && SYCL_EXT_ONEAPI_ASYNC_జ్ఞాపకం_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_లేఖాచిత్రం;
        ఒకవేళ (g_ggml_sycl_use_async_mem_op) {
            కొరకు (గుర్తులేని పూర్ణాంకం i = 0; i < dpct::dev_mgr::instance().device_లెక్క(); ++i) {
                ఒకవేళ (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_జ్ఞాపకం_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    ఆపు;
                }
            }
        }
#అయితే_ముగింపు
        ఒకవేళ (CHECK_TRY_ERROR(g_all_sycl_device_లెక్క =
                            dpct::dev_mgr::instance().device_లెక్క()) != 0) {
            initialized = నిజం;
            g_sycl_loaded = అబద్ధం;
            తిరిగి;
        }
        GGML_ASSERT(g_all_sycl_device_లెక్క <= GGML_SYCL_MAX_DEVICES);

        initialized = నిజం;
        g_sycl_loaded = నిజం;
        ggml_backend_sycl_print_sycl_devices();
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
అంతఃవరుస శూన్య check_allow_gpu_సూచీ(స్థిరరాశి పూర్ణాంకం device_సూచీ) {
  ఒకవేళ (device_సూచీ >= ggml_sycl_info().device_లెక్క) {
    అక్షరం error_buf[256];
    snprintf(
        error_buf,
        పరిమాణం(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_సూచీ,
        ggml_sycl_info().device_లెక్క - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    ప్రతిజ్ఞా(అబద్ధం);
  }
}

GGML_API శూన్య ggml_backend_sycl_get_gpu_list(పూర్ణాంకం *id_list, పూర్ణాంకం max_len) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    కొరకు(పూర్ణాంకం i=0;i<max_len;i++) id_list[i] = -1;

    కొరకు (పూర్ణాంకం i=0;i< ggml_sycl_info().device_లెక్క;i++){
        ఒకవేళ (i>=max_len) ఆపు;
        id_list[i] = i;
    }
    తిరిగి;
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

అంతఃవరుస శూన్య free_aligned_mem_host(శూన్య * memblock) {
#నిర్వచితమైతే _WIN32
    _aligned_free(memblock);
#లేకపోతే
    విడుదల(memblock);
#అయితే_ముగింపు
}

// sycl buffer

నిర్మాణం ggml_backend_sycl_బఫరు_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    స్థిర constexpr size_t staging_స్థానం_కొలత = 8*1024*1024;

    నిర్మాణం host_staging {
        శూన్య * దత్తాంశం = శూన్యసూచిక;
        std::vector<sycl::event> events;
        std::vector<తార్కిక> submitted;
        పూర్ణాంకం తదుపరి = 0;
    };

    పూర్ణాంకం device;
    శూన్య * dev_ptr = శూన్యసూచిక;
    వరుస_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    తార్కిక is_usm_system;

    ggml_backend_sycl_బఫరు_context(పూర్ణాంకం device, శూన్య * dev_ptr, వరుస_ptr stream, తార్కిక is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_సూచీ(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    శూన్య drop_host_staging() {
        కొరకు (size_t i = 0; i < staging.submitted.size(); ++i) {
            ఒకవేళ (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = అబద్ధం;
            }
        }
        ఒకవేళ (staging.దత్తాంశం != శూన్యసూచిక) {
            sycl::విడుదల(staging.దత్తాంశం, *stream);
            staging.దత్తాంశం = శూన్యసూచిక;
        }
    }

    ~ggml_backend_sycl_బఫరు_context() {
        drop_host_staging();
        ఒకవేళ (dev_ptr != శూన్యసూచిక) {
            ggml_sycl_సమితి_device(device);
            ఒకవేళ (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            లేకపోతే
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        కొరకు (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft);

స్థిర తార్కిక ggml_backend_బఫరు_is_sycl(ggml_backend_బఫరు_t బఫరు) {
    తిరిగి బఫరు->buft->iface.get_name == ggml_backend_sycl_బఫరు_type_get_name;
}

స్థిర శూన్య
ggml_backend_sycl_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) ప్రయత్నించు {
    ggml_backend_sycl_బఫరు_context * ctx = ( ggml_backend_sycl_బఫరు_context *)బఫరు->context;
    ggml_sycl_సమితి_device(ctx->device);

    తొలగించు ctx;
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య * ggml_backend_sycl_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_sycl_బఫరు_context * ctx = ( ggml_backend_sycl_బఫరు_context *)బఫరు->context;
    తిరిగి ctx->dev_ptr;
}

స్థిర గణన ggml_స్థితి
ggml_backend_sycl_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు,
                                     ggml_tensor *tensor) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_బఫరు_context * ctx = (ggml_backend_sycl_బఫరు_context *)బఫరు->context;

    ఒకవేళ (tensor->view_src != ఖాళీ) {
        ప్రతిజ్ఞా(tensor->view_src->బఫరు->buft == బఫరు->buft);
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    ఒకవేళ (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        ఎంపిక (tensor->type) {
            స్థితి GGML_TYPE_Q4_0:
            స్థితి GGML_TYPE_Q8_0:
            స్థితి GGML_TYPE_Q2_K:
            స్థితి GGML_TYPE_Q3_K:
            స్థితి GGML_TYPE_Q4_K:
            స్థితి GGML_TYPE_Q5_K:
            స్థితి GGML_TYPE_Q6_K:
            స్థితి GGML_TYPE_IQ3_XXS:
            స్థితి GGML_TYPE_IQ3_S:
            స్థితి GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = కొత్త ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                ఆపు;
            }
            అప్రమేయ:
                ఆపు;
        }
    }

    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_కొలత = ggml_nbytes(tensor);
        size_t padded_కొలత = ggml_backend_buft_get_alloc_కొలత(బఫరు->buft, tensor);

        ఒకవేళ (padded_కొలత > original_కొలత && tensor->view_src == శూన్యసూచిక) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (అక్షరం *)tensor->దత్తాంశం + original_కొలత, 0,
                padded_కొలత - original_కొలత).wait()));
        }
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                                ggml_tensor *tensor,
                                                స్థిరరాశి శూన్య *దత్తాంశం, size_t స్థానభ్రంశం,
                                                size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    ggml_backend_sycl_బఫరు_context * ctx = ( ggml_backend_sycl_బఫరు_context *)బఫరు->context;
    ggml_sycl_సమితి_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    స్థిరరాశి పూర్ణాంకం n_slots = g_ggml_sycl_upload_staging_slots;
    ఒకవేళ (n_slots > 0 && ctx->staging.దత్తాంశం == శూన్యసూచిక) {
        ctx->staging.దత్తాంశం = sycl::malloc_host(n_slots * ctx->staging_స్థానం_కొలత, *ctx->stream);
        ఒకవేళ (ctx->staging.దత్తాంశం != శూన్యసూచిక) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.కేటాయించు(n_slots, అబద్ధం);
        }
    }
    ఒకవేళ (ctx->staging.దత్తాంశం != శూన్యసూచిక) {
        వరుస_ptr    stream    = ctx->stream;
        అక్షరం *       dst       = (అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం;
        స్థిరరాశి అక్షరం * src       = (స్థిరరాశి అక్షరం *) దత్తాంశం;
        size_t       మిగిలిన = size;
        అయితే (మిగిలిన > 0) {
            స్థిరరాశి size_t chunk = std::min(మిగిలిన, ctx->staging_స్థానం_కొలత);
            స్థిరరాశి పూర్ణాంకం    స్థానం  = ctx->staging.తదుపరి;
            ctx->staging.తదుపరి = (ctx->staging.తదుపరి + 1) % (పూర్ణాంకం) ctx->staging.submitted.size();
            ఒకవేళ (ctx->staging.submitted[స్థానం]) {
                ctx->staging.events[స్థానం].wait_and_throw();
            }
            శూన్య * stage = (అక్షరం *) ctx->staging.దత్తాంశం + స్థానం * ctx->staging_స్థానం_కొలత;
            జ్ఞాపకం_నకలు(stage, src, chunk);
            ctx->staging.events[స్థానం] = stream->జ్ఞాపకం_నకలు(dst, stage, chunk);
            ctx->staging.submitted[స్థానం] = నిజం;
            src       += chunk;
            dst       += chunk;
            మిగిలిన -= chunk;
        }
        తిరిగి;
    }

    స్వయంచాలక stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_వరుస());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#నిర్వచితం_కాకపోతే _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    అక్షరం * host_buf = (అక్షరం *) జ్ఞాపకం_ఇవ్వు(size);
    జ్ఞాపకం_నకలు(host_buf, దత్తాంశం, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, host_buf, size).wait()));
    విడుదల(host_buf);
#లేకపోతే
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).జ్ఞాపకం_నకలు((అక్షరం *) tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size).wait()));
#అయితే_ముగింపు
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                                స్థిరరాశి ggml_tensor *tensor,
                                                శూన్య *దత్తాంశం, size_t స్థానభ్రంశం,
                                                size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    ggml_backend_sycl_బఫరు_context * ctx = ( ggml_backend_sycl_బఫరు_context *)బఫరు->context;

    ggml_sycl_సమితి_device(ctx->device);
    స్వయంచాలక stream = dpct::dev_mgr::instance().get_device(ctx->device).default_వరుస();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.జ్ఞాపకం_నకలు(దత్తాంశం, (స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, size)
            .wait()));
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
స్థిర తార్కిక ggml_sycl_is_l0_discrete_gpu(పూర్ణాంకం device) {
    తిరిగి ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#అయితే_ముగింపు

స్థిర శూన్య memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, శూన్య *ptr_dst,
                         స్థిరరాశి శూన్య *ptr_src, size_t size) {
    అక్షరం *host_buf = (అక్షరం *)జ్ఞాపకం_ఇవ్వు(size);
    ఒకవేళ (host_buf == శూన్యసూచిక) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.జ్ఞాపకం_నకలు(host_buf, (స్థిరరాశి అక్షరం *)ptr_src, size).wait();
    q_dst.జ్ఞాపకం_నకలు((అక్షరం *)ptr_dst, host_buf, size).wait();
    విడుదల(host_buf);
}

స్థిర శూన్య dev2dev_memcpy(పూర్ణాంకం device_dst, sycl::queue &q_dst, పూర్ణాంకం device_src, sycl::queue &q_src, శూన్య *ptr_dst,
                    స్థిరరాశి శూన్య *ptr_src, size_t size) {

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ఒకవేళ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        స్థిరరాశి తార్కిక l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        ఒకవేళ (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            స్వయంచాలక ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            స్వయంచాలక ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_వరుస_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_వరుస_DESC, శూన్యసూచిక, 0, 0,
                                            0, ZE_COMMAND_వరుస_బహుళకం_SYNCHRONOUS, ZE_COMMAND_వరుస_ప్రాధాన్యత_సాధారణ};
            ze_command_list_handle_t cl;
            ze_ఫలితం_t r = ze_Command_List_Create_తక్షణ(ze_ctx, ze_dev, &cq_desc, &cl);
            ఒకవేళ (r == ZE_ఫలితం_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_జ్ఞాపకం_Copy(cl, ptr_dst, ptr_src, size, శూన్యసూచిక, 0, శూన్యసూచిక);
                zeCommandListDestroy(cl);
                ఒకవేళ (r == ZE_ఫలితం_SUCCESS) {
                    తిరిగి;
                }
            }
        }
    }
#అయితే_ముగింపు

    ఒకవేళ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        ఒకవేళ (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.జ్ఞాపకం_నకలు(ptr_dst, ptr_src, size).wait()));
            తిరిగి;
        }
    }

    // Host-staged copy
    ఒకవేళ(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } లేకపోతే {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

స్థిర తార్కిక
ggml_backend_sycl_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు,
                                    స్థిరరాశి ggml_tensor *src,
                                    ggml_tensor *dst) ప్రయత్నించు {
    తార్కిక is_cpy_supported = ggml_backend_బఫరు_is_sycl(src->బఫరు);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ఒకవేళ (is_cpy_supported) {
        ggml_backend_sycl_బఫరు_context * src_ctx = (ggml_backend_sycl_బఫరు_context *)src->బఫరు->context;
        ggml_backend_sycl_బఫరు_context * dst_ctx = (ggml_backend_sycl_బఫరు_context *)dst->బఫరు->context;

        ggml_sycl_సమితి_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_సమితి_device(dst_ctx->device);
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

        వరుస_ptr ప్రవాహం_dst = dst_ctx->stream;
        వరుస_ptr ప్రవాహం_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *ప్రవాహం_dst, src_ctx->device, *ప్రవాహం_src, dst->దత్తాంశం, src->దత్తాంశం, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#ఒకవేళ 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).జ్ఞాపకం_నకలు(
            (అక్షరం *)dst->దత్తాంశం, (స్థిరరాశి అక్షరం *)src->దత్తాంశం, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#అయితే_ముగింపు
        తిరిగి నిజం;
    }
    తిరిగి అబద్ధం;
    GGML_వాడని(బఫరు);
} పట్టుకో (స్థిరరాశి sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_బఫరు_clear(ggml_backend_బఫరు_t బఫరు,
                                           uint8_t value) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, బఫరు->size);
    ggml_backend_sycl_బఫరు_context * ctx = (ggml_backend_sycl_బఫరు_context *) బఫరు->context;

    ggml_sycl_సమితి_device(ctx->device);
    వరుస_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    కొరకు (size_t off = 0; off < బఫరు->size; off += MAX_CHUNK) {
        size_t chunk = std::min(బఫరు->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<అక్షరం*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_బఫరు_memset_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor, uint8_t value,
                                                   size_t స్థానభ్రంశం, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, స్థానభ్రంశం, value);
    ggml_backend_sycl_బఫరు_context * ctx = (ggml_backend_sycl_బఫరు_context *) బఫరు->context;
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx->device));
    స్వయంచాలక stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_వరుస());
    ఒకవేళ (size == 0) {
        తిరిగి;  // Nothing to do
    }
    ఒకవేళ (tensor->దత్తాంశం == శూన్యసూచిక) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    శూన్య * లక్ష్యం_ptr = static_cast<అక్షరం *>(tensor->దత్తాంశం) + స్థానభ్రంశం;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(లక్ష్యం_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

స్థిర శూన్య ggml_backend_sycl_బఫరు_reset(ggml_backend_బఫరు_t బఫరు) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ఒకవేళ (బఫరు == శూన్యసూచిక) {
        తిరిగి;
    }

    ggml_backend_sycl_బఫరు_context * ctx = (ggml_backend_sycl_బఫరు_context *) బఫరు->context;

    ఒకవేళ (ctx != శూన్యసూచిక) {
        కొరకు (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

స్థిర స్థిరరాశి ggml_backend_బఫరు_i ggml_backend_sycl_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_sycl_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_sycl_బఫరు_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_బఫరు_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_sycl_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_బఫరు_clear,
    /* .reset           = */ ggml_backend_sycl_బఫరు_reset,
};

// sycl buffer type
నిర్మాణం ggml_backend_sycl_బఫరు_type_context {
    పూర్ణాంకం device;
    std::string name;

    // each buffer type has its own stream
    వరుస_ptr stream = శూన్యసూచిక;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_sycl_బఫరు_type_context * ctx = (ggml_backend_sycl_బఫరు_type_context *)buft->context;

    తిరిగి ctx->name.c_str();
}

స్థిర తార్కిక check_usm_system(పూర్ణాంకం device, size_t size) {
    తార్కిక use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_కొలత_1_G);

    ఒకవేళ (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = అబద్ధం;
    }

    తిరిగి use_usm_system;
}

అంతఃవరుస శూన్య * aligned_malloc_host(size_t alignment, size_t size) {
#నిర్వచితమైతే _WIN32
    తిరిగి _aligned_malloc(size, alignment);
#లేకపోతే
    తిరిగి aligned_alloc(alignment, size);
#అయితే_ముగింపు
}

స్థిర ggml_backend_బఫరు_t
ggml_backend_sycl_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft,
                                           size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_బఫరు_type_context *)buft->context;
    ggml_sycl_సమితి_device(buft_ctx->device);
    స్థిరరాశి వరుస_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_కొలత_2_M;
    size_t aligned_కొలత = ((size + alignment - 1) / alignment) * alignment;
    తార్కిక use_usm_system = check_usm_system(buft_ctx->device, aligned_కొలత);

    శూన్య * dev_ptr;
    ఒకవేళ (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (శూన్య *)aligned_malloc_host(alignment, aligned_కొలత);
        ఒకవేళ (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            తిరిగి శూన్యసూచిక;
        }
    } లేకపోతే {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (శూన్య *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_బఫరు)));
        ఒకవేళ (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          తిరిగి శూన్యసూచిక;
        }
    }
    ggml_backend_sycl_బఫరు_context * ctx = కొత్త  ggml_backend_sycl_బఫరు_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_sycl_బఫరు_interface, ctx, size);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర size_t ggml_backend_sycl_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి SYCL_బఫరు_ALIGNMENT;
    GGML_వాడని(buft);
}

స్థిర తార్కిక is_bmg_g31_arch(పూర్ణాంకం device) {
    తిరిగి ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

స్థిర size_t ggml_backend_sycl_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ggml_backend_sycl_బఫరు_type_context * ctx = (ggml_backend_sycl_బఫరు_type_context *)buft->context;
    పూర్ణాంకం device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_కొలత = dpct::get_device(device).get_max_mem_alloc_కొలత();
    ఒకవేళ (g_ggml_sycl_host_pinned_mem_2g) {
        తిరిగి std::min(max_alloc_కొలత, (size_t) 2LL*1024*1024*1024);
    } లేకపోతే {
        ఒకవేళ(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_కొలత = max_alloc_కొలత*0.6;
        }
        తిరిగి max_alloc_కొలత;

    }
    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_sycl_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_కొలత(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            size += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }
    }

    తిరిగి size;

    GGML_వాడని(buft);
}

స్థిర స్థిరరాశి ggml_backend_బఫరు_type_i ggml_backend_sycl_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_sycl_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_sycl_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ఖాళీ,
};

ggml_backend_బఫరు_type_t ggml_backend_sycl_బఫరు_type(పూర్ణాంకం device) {
    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);


    స్వయంచాలక dev_లెక్క = ggml_backend_sycl_get_device_లెక్క();

    ఒకవేళ (device>=dev_లెక్క or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_లెక్క-1);
        GGML_ASSERT(device<dev_లెక్క);
    }
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_sycl_బఫరు_types[GGML_SYCL_MAX_DEVICES];

    స్థిర తార్కిక ggml_backend_sycl_బఫరు_type_initialized = అబద్ధం;

    ఒకవేళ (!ggml_backend_sycl_బఫరు_type_initialized) {
        కొరకు (పూర్ణాంకం i = 0; i < dev_లెక్క; i++) {
            స్వయంచాలక & device_i = dpct::dev_mgr::instance().get_device(i);
            వరుస_ptr stream = &(device_i.default_వరుస());
            ggml_backend_sycl_బఫరు_types[i] = {
                /* .iface    = */ ggml_backend_sycl_బఫరు_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ కొత్త ggml_backend_sycl_బఫరు_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_బఫరు_type_initialized = నిజం;
    }
    తిరిగి &ggml_backend_sycl_బఫరు_types[device];
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_sycl_బఫరు_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    పూర్ణాంకం device = ctx->device;
    ఒకవేళ (device>=ggml_sycl_info().device_లెక్క or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_లెక్క-1);
        GGML_ASSERT(device<ggml_sycl_info().device_లెక్క);
    }
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_sycl_బఫరు_types[GGML_SYCL_MAX_DEVICES];

    స్థిర తార్కిక ggml_backend_sycl_బఫరు_type_initialized = అబద్ధం;

    ఒకవేళ (!ggml_backend_sycl_బఫరు_type_initialized) {
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; i++) {
            ggml_backend_sycl_బఫరు_types[i] = {
                /* .iface    = */ ggml_backend_sycl_బఫరు_type_interface,
                /* .device   = */ శూన్యసూచిక,
                /* .context  = */ కొత్త ggml_backend_sycl_బఫరు_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_బఫరు_type_initialized = నిజం;
    }
    తిరిగి &ggml_backend_sycl_బఫరు_types[device];
}

// sycl split buffer

స్థిర int64_t get_అడ్డువరుస_rounding(ggml_type type, స్థిరరాశి std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> & tensor_విభజించు) {
    int64_t min_compute_capability = గరిష్ఠ_పూర్ణాంకం;
    int64_t max_compute_capability = INT_MIN;
    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        ఒకవేళ (tensor_విభజించు[i] < (i + 1 < ggml_sycl_info().device_లెక్క ? tensor_విభజించు[i + 1] : 1.0f)) {
            ఒకవేళ (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            ఒకవేళ (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    ఎంపిక(type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
            తిరిగి max_compute_capability >= VER_GEN9 ? 128 : 64;
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి 64;
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_F32:
            తిరిగి 1;
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_IQ2_XXS:
        స్థితి GGML_TYPE_IQ2_XS:
        స్థితి GGML_TYPE_IQ2_S:
        స్థితి GGML_TYPE_IQ1_S:
        స్థితి GGML_TYPE_IQ1_M:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ4_XS:
        స్థితి GGML_TYPE_IQ4_NL:
            తిరిగి max_compute_capability >= VER_GEN9 ? 128 : 64;
        స్థితి GGML_TYPE_IQ3_S:
            తిరిగి max_compute_capability >= VER_GEN9 ? 128 : 64;
        స్థితి GGML_TYPE_Q6_K:
            తిరిగి 64;
        అప్రమేయ:
            GGML_ABORT("fatal error");
    }
}

స్థిర శూన్య get_అడ్డువరుస_విభజించు(int64_t * అడ్డువరుస_దిగువ, int64_t * అడ్డువరుస_ఎగువ, స్థిరరాశి ggml_tensor * tensor, స్థిరరాశి std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> & tensor_విభజించు, పూర్ణాంకం id) {
    స్థిరరాశి int64_t nrows = ggml_nrows(tensor);
    స్థిరరాశి int64_t rounding = get_అడ్డువరుస_rounding(tensor->type, tensor_విభజించు);

    *అడ్డువరుస_దిగువ = id == 0 ? 0 : nrows*tensor_విభజించు[id];
    *అడ్డువరుస_దిగువ -= *అడ్డువరుస_దిగువ % rounding;
    ఒకవేళ (id == ggml_sycl_info().device_లెక్క - 1) {
        *అడ్డువరుస_ఎగువ = nrows;
    } లేకపోతే {
        *అడ్డువరుస_ఎగువ = nrows*tensor_విభజించు[id + 1];
        *అడ్డువరుస_ఎగువ -= *అడ్డువరుస_ఎగువ % rounding;
    }
}

స్థిర size_t ggml_nbytes_విభజించు(స్థిరరాశి నిర్మాణం ggml_tensor * tensor, పూర్ణాంకం nrows_విభజించు) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    తిరిగి nrows_విభజించు*ggml_అడ్డువరుస_కొలత(tensor->type, tensor->ne[0]);
}

నిర్మాణం ggml_backend_sycl_విభజించు_బఫరు_type_context {
    std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> tensor_విభజించు;
};

నిర్మాణం ggml_backend_sycl_విభజించు_బఫరు_context {
    ~ggml_backend_sycl_విభజించు_బఫరు_context() ప్రయత్నించు {
        కొరకు (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    పట్టుకో (sycl::exception స్థిరరాశి &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
      std::నిష్క్రమించు(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<వరుస_ptr> streams;
};

స్థిర శూన్య ggml_backend_sycl_విభజించు_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_sycl_విభజించు_బఫరు_context * ctx = (ggml_backend_sycl_విభజించు_బఫరు_context *)బఫరు->context;
    తొలగించు ctx;
}

స్థిర శూన్య * ggml_backend_sycl_విభజించు_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    తిరిగి (శూన్య *)0x1000;

    GGML_వాడని(బఫరు);
}

స్థిర గణన ggml_స్థితి
ggml_backend_sycl_విభజించు_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు,
                                           ggml_tensor *tensor) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == శూన్యసూచిక); // views of split tensors are not supported

    ggml_backend_sycl_విభజించు_బఫరు_context * ctx = (ggml_backend_sycl_విభజించు_బఫరు_context *)బఫరు->context;
    ggml_backend_sycl_విభజించు_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_విభజించు_బఫరు_type_context *)బఫరు->buft->context;

    స్థిరరాశి int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = కొత్త ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    ఒకవేళ (ctx->streams.ఖాళీలు()) {
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_వరుస()));
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        int64_t అడ్డువరుస_దిగువ, అడ్డువరుస_ఎగువ;
        get_అడ్డువరుస_విభజించు(&అడ్డువరుస_దిగువ, &అడ్డువరుస_ఎగువ, tensor, buft_ctx->tensor_విభజించు, i);

        int64_t nrows_విభజించు = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;
        ఒకవేళ (nrows_విభజించు == 0) {
            కొనసాగు;
        }

        size_t size = ggml_nbytes_విభజించు(tensor, nrows_విభజించు);
        స్థిరరాశి size_t original_కొలత = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            size += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }

        ggml_sycl_సమితి_device(i);
        స్థిరరాశి వరుస_ptr stream = ctx->streams[i];
        అక్షరం * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (అక్షరం *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_బఫరు)));
        ఒకవేళ (!buf) {
            అక్షరం err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            విసురు std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        ఒకవేళ (size > original_కొలత) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_కొలత, 0, size - original_కొలత)
                    .wait()));
        }

        extra->దత్తాంశం_device[i] = buf;

        కొరకు (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = కొత్త sycl::event()));
        }
    }
    tensor->extra = extra;
    తిరిగి GGML_స్థితి_SUCCESS;
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య
ggml_backend_sycl_విభజించు_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                          ggml_tensor *tensor, స్థిరరాశి శూన్య *దత్తాంశం,
                                          size_t స్థానభ్రంశం, size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_విభజించు_బఫరు_context * ctx = (ggml_backend_sycl_విభజించు_బఫరు_context *)బఫరు->context;
    ggml_backend_sycl_విభజించు_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_విభజించు_బఫరు_type_context *)బఫరు->buft->context;

    స్థిరరాశి int64_t ne0 = tensor->ne[0];
    స్థిరరాశి size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        int64_t అడ్డువరుస_దిగువ, అడ్డువరుస_ఎగువ;
        get_అడ్డువరుస_విభజించు(&అడ్డువరుస_దిగువ, &అడ్డువరుస_ఎగువ, tensor, buft_ctx->tensor_విభజించు, i);

        int64_t nrows_విభజించు = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;
        ఒకవేళ (nrows_విభజించు == 0) {
            కొనసాగు;
        }

        స్థిరరాశి size_t స్థానభ్రంశం_విభజించు = అడ్డువరుస_దిగువ*nb1;
        size_t size = ggml_nbytes_విభజించు(tensor, nrows_విభజించు);
        స్థిరరాశి size_t original_కొలత = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            size += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }

        స్థిరరాశి అక్షరం * buf_host = (స్థిరరాశి అక్షరం *)దత్తాంశం + స్థానభ్రంశం_విభజించు;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_సమితి_device(i);
        స్థిరరాశి వరుస_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .జ్ఞాపకం_నకలు(extra->దత్తాంశం_device[i], buf_host, original_కొలత)
                .wait()));
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య
ggml_backend_sycl_విభజించు_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                          స్థిరరాశి ggml_tensor *tensor, శూన్య *దత్తాంశం,
                                          size_t స్థానభ్రంశం, size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_విభజించు_బఫరు_context * ctx = (ggml_backend_sycl_విభజించు_బఫరు_context *)బఫరు->context;
    ggml_backend_sycl_విభజించు_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_విభజించు_బఫరు_type_context *)బఫరు->buft->context;

    స్థిరరాశి int64_t ne0 = tensor->ne[0];
    స్థిరరాశి size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        int64_t అడ్డువరుస_దిగువ, అడ్డువరుస_ఎగువ;
        get_అడ్డువరుస_విభజించు(&అడ్డువరుస_దిగువ, &అడ్డువరుస_ఎగువ, tensor, buft_ctx->tensor_విభజించు, i);

        int64_t nrows_విభజించు = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;
        ఒకవేళ (nrows_విభజించు == 0) {
            కొనసాగు;
        }

        స్థిరరాశి size_t స్థానభ్రంశం_విభజించు = అడ్డువరుస_దిగువ*nb1;
        size_t size = ggml_nbytes_విభజించు(tensor, nrows_విభజించు);
        స్థిరరాశి size_t original_కొలత = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            size += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }

        అక్షరం * buf_host = (అక్షరం *)దత్తాంశం + స్థానభ్రంశం_విభజించు;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_సమితి_device(i);
        స్థిరరాశి వరుస_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .జ్ఞాపకం_నకలు(buf_host, extra->దత్తాంశం_device[i], original_కొలత)
                .wait()));
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_విభజించు_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    GGML_వాడని(బఫరు);
    GGML_వాడని(value);
}

స్థిర నిర్మాణం ggml_backend_బఫరు_i ggml_backend_sycl_విభజించు_బఫరు_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_విభజించు_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_sycl_విభజించు_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_sycl_విభజించు_బఫరు_init_tensor,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_sycl_విభజించు_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_విభజించు_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ఖాళీ,
    /* .clear           = */ ggml_backend_sycl_విభజించు_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

// sycl split buffer type

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_విభజించు_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి GGML_SYCL_NAME "_Split";

    GGML_వాడని(buft);
}

స్థిర తార్కిక ggml_backend_బఫరు_is_sycl_విభజించు(ggml_backend_బఫరు_t బఫరు) {
   తిరిగి బఫరు->buft->iface.get_name == ggml_backend_sycl_విభజించు_బఫరు_type_get_name;
}

స్థిర ggml_backend_బఫరు_t ggml_backend_sycl_విభజించు_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_విభజించు_బఫరు_context * ctx = కొత్త ggml_backend_sycl_విభజించు_బఫరు_context();

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_sycl_విభజించు_బఫరు_interface, ctx, size);
}

స్థిర size_t ggml_backend_sycl_విభజించు_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి SYCL_బఫరు_ALIGNMENT;
    GGML_వాడని(buft);
}

స్థిర size_t ggml_backend_sycl_విభజించు_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    ggml_backend_sycl_విభజించు_బఫరు_type_context * ctx = (ggml_backend_sycl_విభజించు_బఫరు_type_context *)buft->context;

    size_t మొత్తం_కొలత = 0;

    స్థిరరాశి int64_t ne0 = tensor->ne[0];

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        int64_t అడ్డువరుస_దిగువ, అడ్డువరుస_ఎగువ;
        get_అడ్డువరుస_విభజించు(&అడ్డువరుస_దిగువ, &అడ్డువరుస_ఎగువ, tensor, ctx->tensor_విభజించు, i);

        int64_t nrows_విభజించు = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;
        ఒకవేళ (nrows_విభజించు == 0) {
            కొనసాగు;
        }

        మొత్తం_కొలత += ggml_nbytes_విభజించు(tensor, nrows_విభజించు);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ఒకవేళ (ne0 % మాతృక_అడ్డువరుస_PADDING != 0) {
            మొత్తం_కొలత += ggml_అడ్డువరుస_కొలత(tensor->type, మాతృక_అడ్డువరుస_PADDING - ne0 % మాతృక_అడ్డువరుస_PADDING);
        }
    }

    తిరిగి మొత్తం_కొలత;
}

స్థిర తార్కిక ggml_backend_sycl_విభజించు_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    తిరిగి అబద్ధం;

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_type_i ggml_backend_sycl_విభజించు_బఫరు_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_విభజించు_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_విభజించు_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_sycl_విభజించు_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ఖాళీ, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_విభజించు_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ggml_backend_sycl_విభజించు_బఫరు_type_is_host,
};

ggml_backend_బఫరు_type_t ggml_backend_sycl_విభజించు_బఫరు_type(పూర్ణాంకం main_device, స్థిరరాశి దశాంశం * tensor_విభజించు) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_వాడని(main_device);

    స్థిర std::mutex mutex;
    std::lock_guard<std::mutex> తాళం(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    స్థిర std::map<std::array<దశాంశం, GGML_SYCL_MAX_DEVICES>, నిర్మాణం ggml_backend_బఫరు_type> buft_map;

    std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> tensor_విభజించు_arr = {};

    తార్కిక all_zero = tensor_విభజించు == శూన్యసూచిక || std::all_of(tensor_విభజించు, tensor_విభజించు + GGML_SYCL_MAX_DEVICES, [](దశాంశం x) { తిరిగి x == 0.0f; });
    ఒకవేళ (all_zero) {
        tensor_విభజించు_arr = ggml_sycl_info().default_tensor_విభజించు;
    } లేకపోతే {
        దశాంశం విభజించు_కూడిక = 0.0f;
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
            tensor_విభజించు_arr[i] = విభజించు_కూడిక;
            విభజించు_కూడిక += tensor_విభజించు[i];
        }
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
            tensor_విభజించు_arr[i] /= విభజించు_కూడిక;
        }
    }

    స్వయంచాలక it = buft_map.find(tensor_విభజించు_arr);
    ఒకవేళ (it != buft_map.end()) {
        తిరిగి &it->second;
    }

    నిర్మాణం ggml_backend_బఫరు_type buft {
        /* .iface   = */ ggml_backend_sycl_విభజించు_బఫరు_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ కొత్త ggml_backend_sycl_విభజించు_బఫరు_type_context{tensor_విభజించు_arr},
    };

    స్వయంచాలక ఫలితం = buft_map.emplace(tensor_విభజించు_arr, buft);
    తిరిగి &ఫలితం.మొదటి->second;
}

// host buffer type

నిర్మాణం ggml_backend_sycl_device_context {
    పూర్ణాంకం device;
    std::string name;
    std::string description;
    పూర్ణాంకం op_offload_min_batch_కొలత;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_host_బఫరు_type_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి GGML_SYCL_NAME "_Host";

    GGML_వాడని(buft);
}

స్థిర పూర్ణాంకం ggml_backend_sycl_host_బఫరు_type_device(ggml_backend_బఫరు_type_t buft) {
    తిరిగి static_cast<స్థిరరాశి ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
స్థిర శూన్య * ggml_backend_sycl_host_malloc(పూర్ణాంకం device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    శూన్య * ptr = శూన్యసూచిక;
    ప్రయత్నించు {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        స్వయంచాలక & q = dpct::dev_mgr::instance().get_device(device).default_వరుస();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } పట్టుకో (...) {
        ptr = శూన్యసూచిక;
    }
    ఒకవేళ (ptr == శూన్యసూచిక) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    తిరిగి ptr;
}

స్థిర శూన్య ggml_backend_sycl_host_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ఒకవేళ (బఫరు->context == శూన్యసూచిక) {
        తిరిగి;
    }
    ఒకవేళ (g_ggml_sycl_enable_host_pinned_mem) {
        స్థిరరాశి పూర్ణాంకం device = ggml_backend_sycl_host_బఫరు_type_device(బఫరు->buft);
        స్వయంచాలక & q = dpct::dev_mgr::instance().get_device(device).default_వరుస();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(బఫరు->context, q)));
    } లేకపోతే {
        free_aligned_mem_host((శూన్య *) బఫరు->context);
    }
}

స్థిర ggml_backend_బఫరు_t ggml_backend_sycl_host_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    శూన్య * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_బఫరు_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    ఒకవేళ (ptr == శూన్యసూచిక) {
        // fallback to cpu buffer
        తిరిగి ggml_backend_buft_alloc_బఫరు(ggml_backend_cpu_బఫరు_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_బఫరు_t బఫరు = ggml_backend_cpu_బఫరు_from_ptr(ptr, size);
    బఫరు->buft = buft;
    బఫరు->iface.free_బఫరు = ggml_backend_sycl_host_బఫరు_free_బఫరు;

    తిరిగి బఫరు;
}

స్థిర size_t ggml_backend_sycl_host_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {

    ఒకవేళ (g_ggml_sycl_enable_host_pinned_mem) {
        స్థిరరాశి పూర్ణాంకం device = ggml_backend_sycl_host_బఫరు_type_device(buft);
        size_t max_alloc_కొలత = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_కొలత();
        ఒకవేళ (g_ggml_sycl_host_pinned_mem_2g) {
            తిరిగి std::min(max_alloc_కొలత, (size_t) 2LL*1024*1024*1024);
        } లేకపోతే {
            తిరిగి max_alloc_కొలత;
        }
    } లేకపోతే {
        తిరిగి SIZE_MAX;
    }
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_sycl_host_బఫరు_type_for_device(పూర్ణాంకం device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    స్థిర std::vector<ggml_backend_బఫరు_type> బఫరు_types_host = [] {
        std::vector<ggml_backend_బఫరు_type> bufts(ggml_backend_sycl_get_device_లెక్క());
        కొరకు (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_బఫరు_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_బఫరు_type_alloc_బఫరు,
                    /* .alloc_buffer_n      = */ ఖాళీ,
                    /* .get_alignment       = */ ggml_backend_cpu_బఫరు_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_బఫరు_type_get_max_కొలత,
                    /* .get_alloc_size      = */ ggml_backend_cpu_బఫరు_type()->iface.get_alloc_కొలత,
                    /* .get_alloc_size_n    = */ ఖాళీ,
                    /* .is_host             = */ ggml_backend_cpu_బఫరు_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ శూన్యసూచిక,
            };
        }
        తిరిగి bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (పూర్ణాంకం) బఫరు_types_host.size());

    తిరిగి &బఫరు_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_బఫరు_type_t ggml_backend_sycl_host_బఫరు_type() {
    తిరిగి ggml_backend_sycl_host_బఫరు_type_for_device(0);
}

// buffer pool for sycl (legacy)
నిర్మాణం ggml_sycl_pool_leg : బహిరంగ ggml_sycl_pool {
    స్థిర స్థిరరాశి పూర్ణాంకం MAX_SYCL_BUFFERS = 256;

    పూర్ణాంకం device;
    వరుస_ptr qptr;
    నిర్మాణం ggml_sycl_బఫరు {
        శూన్య * ptr = శూన్యసూచిక;
        size_t size = 0;
    };

    ggml_sycl_బఫరు బఫరు_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_కొలత = 0;

    స్పష్టమైన ggml_sycl_pool_leg(వరుస_ptr qptr_, పూర్ణాంకం device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#నిర్వచితమైతే DEBUG_SYCL_POOL
        పూర్ణాంకం    n_cached    = 0;
        size_t బైట్లు_cached = 0;
        కొరకు (పూర్ణాంకం i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ఒకవేళ (బఫరు_pool[i].ptr != శూన్యసూచిక) {
                ++n_cached;
                బైట్లు_cached += బఫరు_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, బైట్లు_cached / 1024.0 / 1024.0);
        స్థిరరాశి స్వయంచాలక slots = format_slots_లోపల_alloc_క్రమం();
        ఒకవేళ (!slots.ఖాళీలు()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#అయితే_ముగింపు

        కొరకు (పూర్ణాంకం i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr != శూన్యసూచిక) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_కొలత -= b.size;
            }
        }
        GGML_ASSERT(pool_కొలత == 0);
    }

#నిర్వచితమైతే DEBUG_SYCL_POOL
    std::string format_slots_లోపల_alloc_క్రమం() స్థిరరాశి {
        std::string పంక్తి;
        అక్షరం buf[32];
        తార్కిక మొదటి = నిజం;
        కొరకు (పూర్ణాంకం i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ఒకవేళ (బఫరు_pool[i].ptr == శూన్యసూచిక) {
                కొనసాగు;
            }
            ఒకవేళ (!మొదటి) {
                పంక్తి += '/';
            }
            మొదటి = అబద్ధం;
            snprintf(buf, పరిమాణం(buf), "%.2f", బఫరు_pool[i].size / 1024.0 / 1024.0);
            పంక్తి += buf;
        }
        తిరిగి పంక్తి;
    }
#అయితే_ముగింపు

    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
#నిర్వచితమైతే DEBUG_sycl_MALLOC
        పూర్ణాంకం nnz = 0;
        size_t max_size = 0;
#అయితే_ముగింపు
        size_t ఉత్తమ_diff = 1ull << 36;
        పూర్ణాంకం ibest = -1;
        కొరకు (పూర్ణాంకం i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_బఫరు& b = బఫరు_pool[i];
            ఒకవేళ (b.ptr != శూన్యసూచిక) {
#నిర్వచితమైతే DEBUG_sycl_MALLOC
                ++nnz;
                ఒకవేళ (b.size > max_size) max_size = b.size;
#అయితే_ముగింపు
                ఒకవేళ (b.size >= size) {
                    size_t diff = b.size - size;
                    ఒకవేళ (diff < ఉత్తమ_diff) {
                        ఉత్తమ_diff = diff;
                        ibest = i;
                        ఒకవేళ (!ఉత్తమ_diff) {
                            శూన్య * ptr = b.ptr;
                            *actual_కొలత = b.size;
                            b.ptr = శూన్యసూచిక;
                            b.size = 0;
                            తిరిగి ptr;
                        }
                    }
                }
            }
        }
        ఒకవేళ (ibest >= 0) {
            ggml_sycl_బఫరు& b = బఫరు_pool[ibest];
            శూన్య * ptr = b.ptr;
            *actual_కొలత = b.size;
            b.ptr = శూన్యసూచిక;
            b.size = 0;
            తిరిగి ptr;
        }
        శూన్య * ptr;
        size_t look_ahead_కొలత = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (శూన్య *)ggml_sycl_malloc_device(look_ahead_కొలత, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        ఒకవేళ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_కొలత);
            తిరిగి శూన్యసూచిక;
        }

        *actual_కొలత = look_ahead_కొలత;
        pool_కొలత += look_ahead_కొలత;

#నిర్వచితమైతే DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_కొలత[id]/1024/1024), (uint32_t)(size/1024/1024));
#అయితే_ముగింపు

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        తిరిగి ptr;
    }

    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
        కొరకు (పూర్ణాంకం i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_బఫరు& b = బఫరు_pool[i];
            ఒకవేళ (b.ptr == శూన్యసూచిక) {
                b.ptr = ptr;
                b.size = size;
                తిరిగి;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_కొలత -= size;
    }
};

// pool with virtual memory management
#ఒకవేళ defined(GGML_SYCL_SUPPORT_VMM)
నిర్మాణం ggml_sycl_pool_vmm : బహిరంగ ggml_sycl_pool {
    స్థిర స్థిరరాశి size_t SYCL_POOL_VMM_MAX_కొలత = 1ull << 35; // 32 GB

    పూర్ణాంకం           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_వాడిన = 0;
    size_t    pool_కొలత = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    నిర్మాణం mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        శూన్య * map_ptr;
    };
    std::vector<mapping> mappings;

    స్పష్టమైన ggml_sycl_pool_vmm(వరుస_ptr qptr_, పూర్ణాంకం device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        ఒకవేళ (pool_addr == 0) {
            తిరిగి;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        కొరకు (స్వయంచాలక & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_కొలత, ctx)));
    }

    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_బఫరు_ALIGNMENT);

        size_t avail = pool_కొలత - pool_వాడిన;

        ఒకవేళ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_కొలత = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_కొలత + reserve_కొలత <= SYCL_POOL_VMM_MAX_కొలత);

            ఒకవేళ (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ట్యాగ్ " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_కొలత / (1024 * 1024), pool_కొలత / (1024 * 1024),
                              (pool_కొలత + reserve_కొలత) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_కొలత)));

            // reserve virtual address space (if not already reserved)
            ఒకవేళ (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_కొలత, ctx)));
            }

            // map at the end of the pool
            శూన్య * map_ptr = శూన్యసూచిక;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_కొలత, reserve_కొలత,
                                    sycl::ext::oneapi::experimental::చిరునామా_access_బహుళకం::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_కొలత += reserve_కొలత;
            ggml_sycl_memtrace_కూడు(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_కొలత);

#నిర్వచితమైతే DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (గుర్తులేని దీర్ఘ దీర్ఘ) (pool_కొలత/1024/1024),
                          (గుర్తులేని దీర్ఘ దీర్ఘ) (reserve_కొలత/1024/1024));
#అయితే_ముగింపు
        }

        GGML_ASSERT(pool_addr != 0);

        శూన్య * ptr = reinterpret_cast<శూన్య *>(pool_addr + pool_వాడిన);
        *actual_కొలత = size;
        pool_వాడిన += size;

#నిర్వచితమైతే DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (గుర్తులేని దీర్ఘ దీర్ఘ) size, ptr);
#అయితే_ముగింపు

        తిరిగి ptr;
    }

    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
#నిర్వచితమైతే DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (గుర్తులేని దీర్ఘ దీర్ఘ) size, ptr);
#అయితే_ముగింపు

        pool_వాడిన -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<శూన్య *>(pool_addr + pool_వాడిన));
    }
};
#అయితే_ముగింపు // defined(GGML_SYCL_SUPPORT_VMM)

నిర్మాణం ggml_sycl_pool_host : బహిరంగ ggml_sycl_pool {
    వరుస_ptr qptr;
    పూర్ణాంకం       device;

    నిర్మాణం ggml_sycl_బఫరు {
        శూన్య * ptr  = శూన్యసూచిక;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    స్థిర constexpr పూర్ణాంకం          MAX_POOL_కొలత{ 64 };
    std::vector<ggml_sycl_బఫరు> బఫరు_pool = std::vector<ggml_sycl_బఫరు>(MAX_POOL_కొలత);
    size_t                        pool_కొలత   = 0;
    పూర్ణాంకం                           counter     = 0;

    స్పష్టమైన ggml_sycl_pool_host(వరుస_ptr qptr_, పూర్ణాంకం device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        కొరకు (పూర్ణాంకం i = 0; i < MAX_POOL_కొలత; ++i) {
            ggml_sycl_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr != శూన్యసూచిక) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(b.ptr, *qptr)));
                b.ptr = శూన్యసూచిక;
                pool_కొలత -= b.size;
                b.size = 0;
            }
        }
    }

    శూన్య * alloc(size_t size, size_t * actual_కొలత) override {
        ఒకవేళ (counter == MAX_POOL_కొలత) {
            // the pool wrapped around, no slot is reserved for this allocation
            శూన్య * ptr = శూన్యసూచిక;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (శూన్య *) sycl::malloc_host(size, *qptr)));
            ఒకవేళ (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                తిరిగి శూన్యసూచిక;
            }
            pool_కొలత    += size;
            *actual_కొలత = size;
            counter      = 0;
            తిరిగి ptr;
        }
        ggml_sycl_బఫరు & b = బఫరు_pool[counter];
        ++counter;

        ఒకవేళ (b.ptr != శూన్యసూచిక) {
            ఒకవేళ (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                శూన్య * ptr = b.ptr;
                *actual_కొలత = b.size;
                b.ptr = శూన్యసూచిక;
                b.size = 0;
                తిరిగి ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(b.ptr, *qptr)));
            pool_కొలత -= b.size;
            b.ptr = శూన్యసూచిక;
            b.size = 0;
        }

        శూన్య * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (శూన్య *) sycl::malloc_host(size, *qptr)));
        ఒకవేళ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            తిరిగి శూన్యసూచిక;
        }
        pool_కొలత    += size;
        *actual_కొలత = size;
        తిరిగి ptr;
    }

    శూన్య విడుదల(శూన్య * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        కొరకు (పూర్ణాంకం i = 0; i < MAX_POOL_కొలత; ++i) {
            ggml_sycl_బఫరు & b = బఫరు_pool[i];
            ఒకవేళ (b.ptr == శూన్యసూచిక) {
                b.ptr  = ptr;
                b.size = size;
                తిరిగి;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::విడుదల(ptr, *qptr)));
        pool_కొలత -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(వరుస_ptr qptr, పూర్ణాంకం device) {
    // return pool for the host to speed up memory management
    తిరిగి std::unique_ptr<ggml_sycl_pool>(కొత్త ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(వరుస_ptr qptr, పూర్ణాంకం device) {
#ఒకవేళ defined(GGML_SYCL_SUPPORT_VMM)
    ఒకవేళ (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        తిరిగి std::unique_ptr<ggml_sycl_pool>(కొత్త ggml_sycl_pool_vmm(qptr, device));
    }
#అయితే_ముగింపు // defined(GGML_SYCL_SUPPORT_VMM)
    తిరిగి std::unique_ptr<ggml_sycl_pool>(కొత్త ggml_sycl_pool_leg(qptr, device));
}

/// kernels
రకనిర్వచనం శూన్య (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి వరుస_ptr &stream);



స్థిర శూన్య mul_mat_p021_f16_f32(
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి దశాంశం * __restrict__ y, దశాంశం * __restrict__ dst,
    స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం nchannels_x, స్థిరరాశి పూర్ణాంకం nchannels_y,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి sycl::half *x = (స్థిరరాశి sycl::half *)vx;

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_x = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                      అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం channel = అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                        అంశం_ct1.get_local_id(0);
    స్థిరరాశి పూర్ణాంకం channel_x = channel / (nchannels_y / nchannels_x);

    స్థిరరాశి పూర్ణాంకం nrows_y = ncols_x;
    స్థిరరాశి పూర్ణాంకం nrows_dst = nrows_x;
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst = అడ్డువరుస_x;

    దశాంశం tmp = 0.0f;

    కొరకు (పూర్ణాంకం నిలువు_x0 = 0; నిలువు_x0 < ncols_x;
         నిలువు_x0 += అంశం_ct1.get_local_range(2)) {
        స్థిరరాశి పూర్ణాంకం నిలువు_x = నిలువు_x0 + అంశం_ct1.get_local_id(2);

        ఒకవేళ (నిలువు_x >= ncols_x) {
            ఆపు;
        }

        // x is transposed and permuted
        స్థిరరాశి పూర్ణాంకం ix = అడ్డువరుస_x*nchannels_x*ncols_x + channel_x*ncols_x + నిలువు_x;
        స్థిరరాశి దశాంశం xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];

        స్థిరరాశి పూర్ణాంకం అడ్డువరుస_y = నిలువు_x;


        // y is not transposed but permuted
        స్థిరరాశి పూర్ణాంకం iy = channel*nrows_y + అడ్డువరుస_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    స్థిరరాశి పూర్ణాంకం idst = channel*nrows_dst + అడ్డువరుస_dst;

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

స్థిర శూన్య mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    స్థిరరాశి శూన్య * __restrict__ vx, స్థిరరాశి దశాంశం * __restrict__ y, దశాంశం * __restrict__ dst, స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x,
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_stride_x, స్థిరరాశి పూర్ణాంకం channel_stride_x,స్థిరరాశి పూర్ణాంకం channel_stride_y, స్థిరరాశి పూర్ణాంకం channel_x_భాజకం,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి sycl::half *x = (స్థిరరాశి sycl::half *)vx;

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_x = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                      అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం channel = అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                        అంశం_ct1.get_local_id(0);
    స్థిరరాశి పూర్ణాంకం channel_x = channel / channel_x_భాజకం;

    స్థిరరాశి పూర్ణాంకం nrows_dst = nrows_x;
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస_dst   = అడ్డువరుస_x;

    స్థిరరాశి పూర్ణాంకం idst = channel*nrows_dst + అడ్డువరుస_dst;

    దశాంశం tmp = 0.0f;

    కొరకు (పూర్ణాంకం నిలువు_x0 = 0; నిలువు_x0 < ncols_x;
         నిలువు_x0 += అంశం_ct1.get_local_range(2)) {
        స్థిరరాశి పూర్ణాంకం నిలువు_x = నిలువు_x0 + అంశం_ct1.get_local_id(2);

        ఒకవేళ (నిలువు_x >= ncols_x) {
            ఆపు;
        }

        స్థిరరాశి పూర్ణాంకం అడ్డువరుస_y = నిలువు_x;

        స్థిరరాశి పూర్ణాంకం ix = channel_x*channel_stride_x + అడ్డువరుస_x*అడ్డువరుస_stride_x + నిలువు_x;
        స్థిరరాశి పూర్ణాంకం iy = channel * channel_stride_y + అడ్డువరుస_y;

        స్థిరరాశి దశాంశం xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#సూచన unroll
    కొరకు (పూర్ణాంకం ముసుగు = WARP_కొలత / 2; ముసుగు > 0; ముసుగు >>= 1) {
        tmp +=
            dpct::క్రమమార్పు_sub_సమూహం_by_xor(అంశం_ct1.get_sub_సమూహం(), tmp, ముసుగు);
    }

    ఒకవేళ (అంశం_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

స్థిర శూన్య k_కూడిక_అడ్డువరుసలు_f32(స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
                           స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం నిలువు = అంశం_ct1.get_local_id(2);

    దశాంశం కూడిక = 0.0f;
    కొరకు (పూర్ణాంకం i = నిలువు; i < ncols; i += అంశం_ct1.get_local_range(2)) {
        కూడిక += x[అడ్డువరుస * ncols + i];
    }

    కూడిక = warp_reduce_కూడిక(కూడిక, అంశం_ct1);

    ఒకవేళ (నిలువు == 0) {
        dst[అడ్డువరుస] = కూడిక;
    }
}


మూస<రకం_పేరు T>
స్థిర అంతఃవరుస శూన్య ggml_sycl_మార్పిడి(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

మూస <ggml_క్రమీకరించు_క్రమం క్రమం>
__dpct_inline__ స్థిర శూన్య
k_argsort_f32_i32(స్థిరరాశి దశాంశం *x, పూర్ణాంకం *dst, స్థిరరాశి పూర్ణాంకం ncols, పూర్ణాంకం ncols_pad,
                  స్థిరరాశి పూర్ణాంకం tasks_చొప్పున_thread, స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    పూర్ణాంకం నిలువు_సూచీ =  అంశం_ct1.get_local_id(2);
    పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(1);

    కొరకు (పూర్ణాంకం i = 0; i < tasks_చొప్పున_thread; i++) {
        పూర్ణాంకం నిలువు = నిలువు_సూచీ * tasks_చొప్పున_thread + i;
        ఒకవేళ (నిలువు >= ncols_pad) {
            తిరిగి;
        }
    }

    స్థిరరాశి దశాంశం * x_అడ్డువరుస = x + అడ్డువరుస * ncols;
    స్వయంచాలక dst_అడ్డువరుస = (పూర్ణాంకం *)dpct_local;

    // initialize indices
    కొరకు (పూర్ణాంకం i=0;i<tasks_చొప్పున_thread;i++){
        పూర్ణాంకం నిలువు = నిలువు_సూచీ*tasks_చొప్పున_thread+i;
        dst_అడ్డువరుస[నిలువు] = నిలువు;
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    కొరకు (పూర్ణాంకం k = 2; k <= ncols_pad; k *= 2) {
        కొరకు (పూర్ణాంకం j = k / 2; j > 0; j /= 2) {
            కొరకు (పూర్ణాంకం i = 0; i < tasks_చొప్పున_thread; i++) {
                పూర్ణాంకం నిలువు = నిలువు_సూచీ * tasks_చొప్పున_thread + i;
                పూర్ణాంకం ixj = నిలువు ^ j;
                ఒకవేళ (ixj > నిలువు) {
                    ఒకవేళ ((నిలువు & k) == 0) {
                        ఒకవేళ (dst_అడ్డువరుస[నిలువు] >= ncols ||
                            (dst_అడ్డువరుస[ixj] < ncols &&
                             (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC
                                  ? x_అడ్డువరుస[dst_అడ్డువరుస[నిలువు]] > x_అడ్డువరుస[dst_అడ్డువరుస[ixj]]
                                  : x_అడ్డువరుస[dst_అడ్డువరుస[నిలువు]] <
                                        x_అడ్డువరుస[dst_అడ్డువరుస[ixj]]))) {
                            ggml_sycl_మార్పిడి(dst_అడ్డువరుస[నిలువు], dst_అడ్డువరుస[ixj]);
                        }
                    } లేకపోతే {
                        ఒకవేళ (dst_అడ్డువరుస[ixj] >= ncols ||
                            (dst_అడ్డువరుస[నిలువు] < ncols &&
                             (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC
                                  ? x_అడ్డువరుస[dst_అడ్డువరుస[నిలువు]] < x_అడ్డువరుస[dst_అడ్డువరుస[ixj]]
                                  : x_అడ్డువరుస[dst_అడ్డువరుస[నిలువు]] >
                                        x_అడ్డువరుస[dst_అడ్డువరుస[ixj]]))) {
                            ggml_sycl_మార్పిడి(dst_అడ్డువరుస[నిలువు], dst_అడ్డువరుస[ixj]);
                        }
                    }
                }
                అంశం_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    కొరకు (పూర్ణాంకం i = 0; i < tasks_చొప్పున_thread; i++) {
        పూర్ణాంకం నిలువు = నిలువు_సూచీ * tasks_చొప్పున_thread + i;
        ఒకవేళ (నిలువు < ncols) {
            dst[అడ్డువరుస * ncols + నిలువు] = dst_అడ్డువరుస[నిలువు];
        }
    }
}

స్థిర శూన్య diag_ముసుగు_inf_f32(స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం అడ్డువరుసలు_చొప్పున_channel, స్థిరరాశి పూర్ణాంకం n_past,
                              స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం నిలువు = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                    అంశం_ct1.get_local_id(2);

    ఒకవేళ (నిలువు >= ncols) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i = అడ్డువరుస*ncols + నిలువు;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (నిలువు > n_past + అడ్డువరుస % అడ్డువరుసలు_చొప్పున_channel) * FLT_MAX;
}

స్థిర శూన్య scale_f32(స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం bias, స్థిరరాశి పూర్ణాంకం k,
                      స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం i = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                  అంశం_ct1.get_local_id(2);

    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    dst[i] = scale * x[i] + bias;
}


స్థిర శూన్య ggml_mul_mat_p021_f16_f32_sycl(స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y,
                                           దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
                                           స్థిరరాశి పూర్ణాంకం nrows_x,
                                           స్థిరరాశి పూర్ణాంకం nchannels_x,
                                           స్థిరరాశి పూర్ణాంకం nchannels_y,
                                           వరుస_ptr stream) {

    స్థిరరాశి sycl::range<3> ఖండం_nums(nchannels_y, nrows_x, 1);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, అంశం_ct1);
            });
    }
}

స్థిర శూన్య ggml_mul_mat_vec_nc_f16_f32_sycl(
    స్థిరరాశి శూన్య *vx, స్థిరరాశి దశాంశం *y, దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols_x,
    స్థిరరాశి పూర్ణాంకం nrows_x, స్థిరరాశి పూర్ణాంకం అడ్డువరుస_stride_x, స్థిరరాశి పూర్ణాంకం nchannels_x,
    స్థిరరాశి పూర్ణాంకం nchannels_y, స్థిరరాశి పూర్ణాంకం channel_stride_x, స్థిరరాశి పూర్ణాంకం channel_stride_y, వరుస_ptr stream) {

    స్థిరరాశి sycl::range<3> ఖండం_nums(nchannels_y, nrows_x, 1);
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       అడ్డువరుస_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, అంశం_ct1);
            });
    }
}



స్థిర శూన్య scale_f32_sycl(స్థిరరాశి దశాంశం *x, దశాంశం *dst, స్థిరరాశి దశాంశం scale, స్థిరరాశి దశాంశం bias,
                           స్థిరరాశి పూర్ణాంకం k, వరుస_ptr stream) {
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (k + SYCL_SCALE_ఖండం_కొలత - 1) / SYCL_SCALE_ఖండం_కొలత;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_ఖండాలు) *
                              sycl::range<3>(1, 1, SYCL_SCALE_ఖండం_కొలత),
                          sycl::range<3>(1, 1, SYCL_SCALE_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            scale_f32(x, dst, scale, bias, k, అంశం_ct1);
        });
}


స్థిర శూన్య కూడిక_అడ్డువరుసలు_f32_sycl(స్థిరరాశి దశాంశం *x, దశాంశం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                              స్థిరరాశి పూర్ణాంకం nrows, వరుస_ptr stream) {
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1)
                             [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                                 k_కూడిక_అడ్డువరుసలు_f32(x, dst, ncols, అంశం_ct1);
                             });
}

స్థిర పూర్ణాంకం తదుపరి_power_యొక్క_2(పూర్ణాంకం x) {
    పూర్ణాంకం n = 1;
    అయితే (n < x) {
        n *= 2;
    }
    తిరిగి n;
}

స్థిర శూన్య init_argsort_indices_padded(
        పూర్ణాంకం * idx,
        స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం ncols_pad,
        స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1) {
    స్థిరరాశి size_t gid = అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) + అంశం_ct1.get_local_id(0);
    స్థిరరాశి size_t మొత్తం = (size_t) nrows * (size_t) ncols_pad;

    ఒకవేళ (gid >= మొత్తం) {
        తిరిగి;
    }

    idx[gid] = (పూర్ణాంకం) (gid % (size_t) ncols_pad);
}

మూస <ggml_క్రమీకరించు_క్రమం క్రమం>
స్థిర శూన్య argsort_f32_i32_global_pass(స్థిరరాశి దశాంశం *            x,
                                        పూర్ణాంకం *                    idx,
                                        స్థిరరాశి పూర్ణాంకం                ncols,
                                        స్థిరరాశి పూర్ణాంకం                nrows,
                                        స్థిరరాశి పూర్ణాంకం                ncols_pad,
                                        స్థిరరాశి పూర్ణాంకం                j,
                                        స్థిరరాశి పూర్ణాంకం                k,
                                        స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1) {
    స్థిరరాశి size_t gid   = అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) + అంశం_ct1.get_local_id(0);
    స్థిరరాశి size_t మొత్తం = (size_t) nrows * (size_t) ncols_pad;

    ఒకవేళ (gid >= మొత్తం) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం) (gid / (size_t) ncols_pad);
    స్థిరరాశి పూర్ణాంకం నిలువు = (పూర్ణాంకం) (gid % (size_t) ncols_pad);
    స్థిరరాశి పూర్ణాంకం ixj = నిలువు ^ j;

    ఒకవేళ (ixj <= నిలువు || ixj >= ncols_pad) {
        తిరిగి;
    }

    స్థిరరాశి size_t పునాది  = (size_t) అడ్డువరుస * (size_t) ncols_pad;
    స్థిరరాశి size_t pos_a = పునాది + (size_t) నిలువు;
    స్థిరరాశి size_t pos_b = పునాది + (size_t) ixj;

    స్థిరరాశి పూర్ణాంకం a = idx[pos_a];
    స్థిరరాశి పూర్ణాంకం b = idx[pos_b];

    తార్కిక do_మార్పిడి = అబద్ధం;

    ఒకవేళ ((నిలువు & k) == 0) {
        ఒకవేళ (a >= ncols ||
            (b < ncols &&
             (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC ?
                  x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) a] > x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) b] :
                  x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) a] < x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) b]))) {
            do_మార్పిడి = నిజం;
        }
    } లేకపోతే {
        ఒకవేళ (b >= ncols ||
            (a < ncols &&
             (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC ?
                  x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) a] < x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) b] :
                  x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) a] > x[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) b]))) {
            do_మార్పిడి = నిజం;
        }
    }

    ఒకవేళ (do_మార్పిడి) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

స్థిర శూన్య copy_argsort_indices_unpadded(స్థిరరాశి పూర్ణాంకం *              idx_padded,
                                          పూర్ణాంకం *                    dst,
                                          స్థిరరాశి పూర్ణాంకం                nrows,
                                          స్థిరరాశి పూర్ణాంకం                ncols,
                                          స్థిరరాశి పూర్ణాంకం                ncols_pad,
                                          స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1) {
    స్థిరరాశి size_t gid   = అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) + అంశం_ct1.get_local_id(0);
    స్థిరరాశి size_t మొత్తం = (size_t) nrows * (size_t) ncols;

    ఒకవేళ (gid >= మొత్తం) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = (పూర్ణాంకం) (gid / (size_t) ncols);
    స్థిరరాశి పూర్ణాంకం నిలువు = (పూర్ణాంకం) (gid % (size_t) ncols);

    dst[(size_t) అడ్డువరుస * (size_t) ncols + (size_t) నిలువు] = idx_padded[(size_t) అడ్డువరుస * (size_t) ncols_pad + (size_t) నిలువు];
}

స్థిర శూన్య argsort_f32_i32_sycl(స్థిరరాశి దశాంశం *x, పూర్ణాంకం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                                 స్థిరరాశి పూర్ణాంకం nrows, ggml_క్రమీకరించు_క్రమం క్రమం,
                                 వరుస_ptr stream, పూర్ణాంకం device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    స్థిరరాశి పూర్ణాంకం ncols_pad = తదుపరి_power_యొక్క_2(ncols);
    స్థిరరాశి size_t shared_mem = (size_t) ncols_pad * పరిమాణం(పూర్ణాంకం);
    స్థిరరాశి size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    ఒకవేళ (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<పూర్ణాంకం> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        పూర్ణాంకం *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t ఖండం_కొలత_3     = 256;
        స్థిరరాశి size_t     మొత్తం_padded   = (size_t) nrows * (size_t) ncols_pad;
        స్థిరరాశి size_t     nblocks_padded = (మొత్తం_padded + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
            [=](sycl::nd_అంశం<1> అంశం_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, అంశం_ct1); });

        కొరకు (పూర్ణాంకం k = 2; k <= ncols_pad; k *= 2) {
            కొరకు (పూర్ణాంకం j = k / 2; j > 0; j /= 2) {
                ఒకవేళ (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
                        [=](sycl::nd_అంశం<1> అంశం_ct1) {
                            argsort_f32_i32_global_pass<GGML_క్రమీకరించు_క్రమం_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, అంశం_ct1);
                        });
                } లేకపోతే ఒకవేళ (క్రమం == GGML_క్రమీకరించు_క్రమం_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
                        [=](sycl::nd_అంశం<1> అంశం_ct1) {
                            argsort_f32_i32_global_pass<GGML_క్రమీకరించు_క్రమం_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, అంశం_ct1);
                        });
                } లేకపోతే {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        స్థిరరాశి size_t మొత్తం   = (size_t) nrows * (size_t) ncols;
        స్థిరరాశి size_t nblocks = (మొత్తం + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
                             [=](sycl::nd_అంశం<1> అంశం_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, అంశం_ct1);
                             });

        తిరిగి;
    }

    పూర్ణాంకం nth = 1;
    పూర్ణాంకం max_ఖండం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
    అయితే (nth < ncols_pad && nth < max_ఖండం_కొలత)
        nth *= 2;
    ఒకవేళ (nth > max_ఖండం_కొలత)
        nth = max_ఖండం_కొలత;

    స్థిరరాశి పూర్ణాంకం tasks_చొప్పున_thread = ncols_pad / nth;

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, nth);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, nrows, 1);

    ఒకవేళ (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    k_argsort_f32_i32<GGML_క్రమీకరించు_క్రమం_ASC>(
                        x, dst, ncols, ncols_pad, tasks_చొప్పున_thread, అంశం_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } లేకపోతే ఒకవేళ (క్రమం == GGML_క్రమీకరించు_క్రమం_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    k_argsort_f32_i32<GGML_క్రమీకరించు_క్రమం_DESC>(
                        x, dst, ncols, ncols_pad, tasks_చొప్పున_thread, అంశం_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } లేకపోతే {
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
స్థిర శూన్య శిఖరం_k_పరిశీలన_విలీనం_f32(
    స్థిరరాశి దశాంశం *   src_vals,
    స్థిరరాశి int32_t * src_map,
    స్థిరరాశి పూర్ణాంకం       begin,
    స్థిరరాశి పూర్ణాంకం       end,
    స్థిరరాశి పూర్ణాంకం       k,
    స్థిరరాశి పూర్ణాంకం       ఖండం_కొలత_3,
    దశాంశం *         shared_vals,
    పూర్ణాంకం *           shared_idx,
    దశాంశం *         బహిర్గతం_vals,
    int32_t *       బహిర్గతం_idx,
    స్థిరరాశి తార్కిక      swap01,
    స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1
) {
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    దశాంశం * lv = shared_vals + tid;
    పూర్ణాంకం * li = shared_idx + tid;

    కొరకు (పూర్ణాంకం i = 0; i < k; i++) {
        lv[i * ఖండం_కొలత_3] = -FLT_MAX;
        li[i * ఖండం_కొలత_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    దశాంశం kth = -FLT_MAX;

    కొరకు (పూర్ణాంకం నిలువు = begin + tid; నిలువు < end; నిలువు += ఖండం_కొలత_3) {
        దశాంశం val = src_vals[నిలువు];

        ఒకవేళ (val > kth) {
            పూర్ణాంకం pos = k - 1;
            అయితే (pos > 0 && val > lv[(pos - 1) * ఖండం_కొలత_3]) {
                pos--;
            }

            కొరకు (పూర్ణాంకం i = k - 1; i > pos; i--) {
                lv[i * ఖండం_కొలత_3] = lv[(i - 1) * ఖండం_కొలత_3];
                li[i * ఖండం_కొలత_3] = li[(i - 1) * ఖండం_కొలత_3];
            }
            lv[pos * ఖండం_కొలత_3] = val;
            li[pos * ఖండం_కొలత_3] = src_map ? src_map[నిలువు] : నిలువు;

            kth = lv[(k - 1) * ఖండం_కొలత_3];
        }
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    ఒకవేళ (tid != 0) {
        తిరిగి;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    దశాంశం * fv = shared_vals + (size_t) k * ఖండం_కొలత_3;
    పూర్ణాంకం * fi = shared_idx + (size_t) k * ఖండం_కొలత_3;

    కొరకు (పూర్ణాంకం i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    దశాంశం fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    కొరకు (పూర్ణాంకం t = 0; t < ఖండం_కొలత_3; t++) {
        కొరకు (పూర్ణాంకం i = 0; i < k; i++) {
            దశాంశం val = shared_vals[i * ఖండం_కొలత_3 + t];

            ఒకవేళ (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                ఆపు;
            }

            పూర్ణాంకం idx = shared_idx[i * ఖండం_కొలత_3 + t];

            పూర్ణాంకం pos = k - 1;
            అయితే (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            కొరకు (పూర్ణాంకం j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    ఒకవేళ (బహిర్గతం_vals) {
        కొరకు (పూర్ణాంకం i = 0; i < k; i++) {
            బహిర్గతం_vals[i] = fv[i];
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < k; i++) {
        బహిర్గతం_idx[i] = fi[i];
    }

    ఒకవేళ (swap01 && k > 1) {
        int32_t తాత్కాలిక = బహిర్గతం_idx[0];
        బహిర్గతం_idx[0] = బహిర్గతం_idx[1];
        బహిర్గతం_idx[1] = తాత్కాలిక;
    }
}

స్థిర శూన్య శిఖరం_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి దశాంశం * src,
    int32_t * dst_indices,
    స్థిరరాశి int64_t ncols,
    స్థిరరాశి int64_t nrows,
    స్థిరరాశి పూర్ణాంకం k,
    dpct::వరుస_ptr main_ప్రవాహం
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
    constexpr పూర్ణాంకం విభజించు_ఖండం = 128;
    constexpr పూర్ణాంకం max_splits  = 128;
    constexpr పూర్ణాంకం min_నిలువుల_సంఖ్య    = 8192;

    పూర్ణాంకం nsplit = 1;
    ఒకవేళ (ncols >= min_నిలువుల_సంఖ్య) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        స్థిరరాశి int64_t want = ncols / విభజించు_ఖండం;
        nsplit = (పూర్ణాంకం) (want > max_splits ? max_splits : want);
    }

    ఒకవేళ (nsplit > 1) {
        స్థిరరాశి పూర్ణాంకం nchunk = (పూర్ణాంకం) ((ncols + nsplit - 1) / nsplit);
        స్థిరరాశి size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<దశాంశం>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        దశాంశం *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        స్థిరరాశి sycl::range<1> ఖండం_dims(విభజించు_ఖండం);

        main_ప్రవాహం->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_vals(sycl::range<1>((విభజించు_ఖండం + 1) * k), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> shared_idx(sycl::range<1>((విభజించు_ఖండం + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<1> అంశం_ct1) {
                    స్థిరరాశి పూర్ణాంకం grp = అంశం_ct1.get_సమూహం(0);
                    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = grp / nsplit;
                    స్థిరరాశి పూర్ణాంకం part = grp % nsplit;

                    స్థిరరాశి పూర్ణాంకం begin = part * nchunk;
                    పూర్ణాంకం end = begin + nchunk;
                    ఒకవేళ (end > (పూర్ణాంకం) ncols) {
                        end = (పూర్ణాంకం) ncols;
                    }

                    శిఖరం_k_పరిశీలన_విలీనం_f32(
                        src + (int64_t) అడ్డువరుస * ncols, శూన్యసూచిక, begin, end, k, విభజించు_ఖండం,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, అబద్ధం, అంశం_ct1);
                });
        });

        main_ప్రవాహం->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<దశాంశం, 1> shared_vals(sycl::range<1>((విభజించు_ఖండం + 1) * k), cgh);
            sycl::local_accessor<పూర్ణాంకం, 1> shared_idx(sycl::range<1>((విభజించు_ఖండం + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<1> అంశం_ct1) {
                    స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(0);
                    స్థిరరాశి size_t off = (size_t) అడ్డువరుస * nsplit * k;

                    శిఖరం_k_పరిశీలన_విలీనం_f32(
                        pv + off, pi + off, 0, nsplit * k, k, విభజించు_ఖండం,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        శూన్యసూచిక, dst_indices + (int64_t) అడ్డువరుస * k, నిజం, అంశం_ct1);
                });
        });

        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = 128;

    స్థిరరాశి sycl::range<1> ఖండం_dims(ఖండం_కొలత_3);
    స్థిరరాశి sycl::range<1> grid_dims(nrows);

    main_ప్రవాహం->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<దశాంశం, 1> shared_vals(sycl::range<1>((ఖండం_కొలత_3 + 1) * k), cgh);
        sycl::local_accessor<పూర్ణాంకం, 1> shared_idx(sycl::range<1>((ఖండం_కొలత_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<1> అంశం_ct1) {
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(0);

                ఒకవేళ (అడ్డువరుస >= nrows) తిరిగి;

                శిఖరం_k_పరిశీలన_విలీనం_f32(
                    src + (int64_t) అడ్డువరుస * ncols, శూన్యసూచిక, 0, (పూర్ణాంకం) ncols, k, ఖండం_కొలత_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    శూన్యసూచిక, dst_indices + (int64_t) అడ్డువరుస * k, నిజం, అంశం_ct1);
            });
    });
}

స్థిర శూన్య argmax_f32_i32_sycl(స్థిరరాశి దశాంశం *x, పూర్ణాంకం *dst, స్థిరరాశి పూర్ణాంకం ncols,
                               స్థిరరాశి పూర్ణాంకం nrows, వరుస_ptr stream) {
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_ARGMAX_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, nrows, 1);
    స్థిరరాశి size_t shared_mem = 256 * పరిమాణం(దశాంశం);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<దశాంశం, 1> shared_దత్తాంశం(
            sycl::range<1>(shared_mem/పరిమాణం(దశాంశం)), cgh);
        sycl::local_accessor<పూర్ణాంకం, 1> shared_indices(
            sycl::range<1>(shared_mem/పరిమాణం(దశాంశం)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_global_id(1);

                దశాంశం max_val = -INFINITY;
                పూర్ణాంకం max_idx = -1;

                కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += 256) {
                    దశాంశం val = x[అడ్డువరుస * ncols + నిలువు];
                    ఒకవేళ (val > max_val) {
                        max_val = val;
                        max_idx = నిలువు;
                    }
                }

                shared_దత్తాంశం[tid] = max_val;
                shared_indices[tid] = max_idx;
                అంశం_ct1.barrier(sycl::access::fence_space::local_space);

                కొరకు (పూర్ణాంకం stride = 256/2; stride > 0; stride >>= 1) {
                    ఒకవేళ (tid < stride) {
                        దశాంశం val1 = shared_దత్తాంశం[tid];
                        దశాంశం val2 = shared_దత్తాంశం[tid + stride];
                        ఒకవేళ (val2 > val1) {
                            shared_దత్తాంశం[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    అంశం_ct1.barrier(sycl::access::fence_space::local_space);
                }


                ఒకవేళ (tid == 0) {
                    dst[అడ్డువరుస] = shared_indices[0];
                }
            });
    });
}
స్థిర శూన్య diag_ముసుగు_inf_f32_sycl(స్థిరరాశి దశాంశం *x, దశాంశం *dst,
                                   స్థిరరాశి పూర్ణాంకం ncols_x, స్థిరరాశి పూర్ణాంకం nrows_x,
                                   స్థిరరాశి పూర్ణాంకం అడ్డువరుసలు_చొప్పున_channel, స్థిరరాశి పూర్ణాంకం n_past,
                                   వరుస_ptr stream) {
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, SYCL_DIAG_ముసుగు_INF_ఖండం_కొలత, 1);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (ncols_x + SYCL_DIAG_ముసుగు_INF_ఖండం_కొలత - 1) / SYCL_DIAG_ముసుగు_INF_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             diag_ముసుగు_inf_f32(x, dst, ncols_x,
                                               అడ్డువరుసలు_చొప్పున_channel, n_past,
                                               అంశం_ct1);
                         });
}

స్థిర dpct::err0 ggml_sycl_cpy_tensor_2d(శూన్య *dst,
                                          స్థిరరాశి నిర్మాణం ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_దిగువ, int64_t i1_ఎగువ,
                                          వరుస_ptr stream) ప్రయత్నించు {

    dpct::memcpy_direction రకం_2;
    అక్షరం * src_ptr;
    ఒకవేళ (ggml_backend_బఫరు_is_host(src->బఫరు)) {
        రకం_2 = dpct::host_గమ్యం_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (అక్షరం *) src->దత్తాంశం;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } లేకపోతే ఒకవేళ (ggml_backend_బఫరు_is_sycl(src->బఫరు)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        రకం_2    = dpct::device_గమ్యం_device;
        src_ptr = (అక్షరం *) src->దత్తాంశం;
    } లేకపోతే ఒకవేళ (ggml_backend_బఫరు_is_sycl_విభజించు(src->బఫరు)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_దిగువ == 0 && i1_ఎగువ == src->ne[1]);
        రకం_2 = dpct::device_గమ్యం_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        పూర్ణాంకం id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (అక్షరం *) extra->దత్తాంశం_device[id];
    } లేకపోతే {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    అక్షరం * dst_ptr = (అక్షరం *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    స్థిరరాశి గణన ggml_type type = src->type;
    స్థిరరాశి int64_t ts = ggml_type_కొలత(type);
    స్థిరరాశి int64_t bs = ggml_blck_కొలత(type);
    int64_t i1_diff = i1_ఎగువ - i1_దిగువ;

    స్థిరరాశి అక్షరం * x = src_ptr + i1_దిగువ*nb1 + i2*nb2 + i3*nb3;
    ఒకవేళ (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        తిరిగి CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    రకం_2, *stream));

    } లేకపోతే ఒకవేళ (nb0 == ts) {
        తిరిగి CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, రకం_2, *stream));
    } లేకపోతే {
        కొరకు (int64_t i1 = 0; i1 < i1_diff; i1++) {
            స్థిరరాశి శూన్య * rx = (స్థిరరాశి శూన్య *) ((స్థిరరాశి అక్షరం *) x + i1*nb1);
            శూన్య * rd = (శూన్య *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, రకం_2, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            ఒకవేళ (r != 0) తిరిగి r;
        }
        తిరిగి 0;
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

అంతఃవరుస శూన్య ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
    స్థిరరాశి అక్షరం *src0_dd_i, స్థిరరాశి దశాంశం *src1_ddf_i, స్థిరరాశి అక్షరం *src1_ddq_i,
    దశాంశం *dst_dd_i, స్థిరరాశి int64_t అడ్డువరుస_దిగువ, స్థిరరాశి int64_t అడ్డువరుస_ఎగువ,
    స్థిరరాశి int64_t src1_ncols, స్థిరరాశి int64_t src1_padded_అడ్డువరుస_కొలత,
    స్థిరరాశి వరుస_ptr &stream) ప్రయత్నించు {

    GGML_ASSERT(src0_dd_i  != శూన్యసూచిక);
    GGML_ASSERT(src1_ddf_i != శూన్యసూచిక);
    GGML_ASSERT(dst_dd_i   != శూన్యసూచిక);

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    స్థిరరాశి int64_t అడ్డువరుస_diff = అడ్డువరుస_ఎగువ - అడ్డువరుస_దిగువ;

    పూర్ణాంకం id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    స్థిరరాశి int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    పూర్ణాంకం ldc = id == ctx.device ? ne0 : అడ్డువరుస_diff; // used by MKL only

#నిర్వచితమైతే GGML_SYCL_F16
    తార్కిక use_fp16 = నిజం;  // TODO(Yu) SYCL capability check
#లేకపోతే
    తార్కిక use_fp16 = అబద్ధం;
#అయితే_ముగింపు

#ఒకవేళ GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    ఒకవేళ (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        అడ్డువరుస_diff == src0->ne[1]) {
        ఉపయోగించి bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        ఒకవేళ (src1->type != GGML_TYPE_BF16) {
            స్థిరరాశి గమ్యం_bf16_sycl_t గమ్యం_bf16_sycl = ggml_get_గమ్యం_bf16_sycl(src1->type, dst);
            GGML_ASSERT(గమ్యం_bf16_sycl != శూన్యసూచిక);
            గమ్యం_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } లేకపోతే {
            stream->జ్ఞాపకం_నకలు(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*పరిమాణం(bf16_t));
        }
        DnnlGemmWrapper::అడ్డువరుస_gemm(ctx, అడ్డువరుస_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::గమ్యం_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::గమ్యం_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(), stream);
        GGML_వాడని(dst);
        GGML_వాడని(src1_ddq_i);
        GGML_వాడని(src1_padded_అడ్డువరుస_కొలత);
        తిరిగి;
    }
#అయితే_ముగింపు

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    తార్కిక src0_reordered = అబద్ధం;
    ఒకవేళ (స్థిరరాశి స్వయంచాలక * src0_extra = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    ఒకవేళ (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && అడ్డువరుస_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     అడ్డువరుస_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        తిరిగి;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    ఒకవేళ ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        అడ్డువరుస_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        ఒకవేళ (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            స్థిరరాశి గమ్యం_fp16_sycl_t గమ్యం_fp16_sycl = ggml_get_గమ్యం_fp16_sycl(src1->type, dst);
            GGML_ASSERT(గమ్యం_fp16_sycl != శూన్యసూచిక);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            గమ్యం_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        స్థిరరాశి sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (స్థిరరాశి sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        ఒకవేళ (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            స్థిరరాశి గమ్యం_fp16_sycl_t గమ్యం_fp16_sycl = ggml_get_గమ్యం_fp16_sycl(src0->type, dst);
            GGML_ASSERT(గమ్యం_fp16_sycl != శూన్యసూచిక);
            size_t ne = అడ్డువరుస_diff*ne00;
            src0_as_f16.alloc(ne);
            గమ్యం_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        స్థిరరాశి sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (స్థిరరాశి sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#ఒకవేళ GGML_SYCL_DNNL
        ఒకవేళ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::అడ్డువరుస_gemm(ctx,అడ్డువరుస_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::గమ్యం_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(), stream);
        }
        లేకపోతే
#అయితే_ముగింపు
        {
            స్థిరరాశి దశాంశం alpha = 1.0f;
            స్థిరరాశి దశాంశం beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, అడ్డువరుస_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_దత్తాంశం_t::real_half, ne00,
                src1_ptr, dpct::library_దత్తాంశం_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_దత్తాంశం_t::real_float, ldc,
                dpct::library_దత్తాంశం_t::real_float)));
        }
    } లేకపోతే {
        ggml_sycl_pool_alloc<దశాంశం> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<దశాంశం> src1_ddq_as_f32(ctx.pool());
        ఒకవేళ (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            స్థిరరాశి గమ్యం_fp32_sycl_t గమ్యం_fp32_sycl = ggml_get_గమ్యం_fp32_sycl(src0->type, dst);
            GGML_ASSERT(గమ్యం_fp32_sycl != శూన్యసూచిక);
            src0_ddq_as_f32.alloc(అడ్డువరుస_diff*ne00);
            గమ్యం_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), అడ్డువరుస_diff*ne00, stream);
        }
        ఒకవేళ (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            స్థిరరాశి గమ్యం_fp32_sycl_t గమ్యం_fp32_sycl = ggml_get_గమ్యం_fp32_sycl(src1->type, dst);
            GGML_ASSERT(గమ్యం_fp32_sycl != శూన్యసూచిక);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            గమ్యం_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        స్థిరరాశి దశాంశం * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (స్థిరరాశి దశాంశం *) src0_dd_i : src0_ddq_as_f32.get();
        స్థిరరాశి దశాంశం * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (స్థిరరాశి దశాంశం *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#ఒకవేళ GGML_SYCL_DNNL
            స్థిరరాశి int64_t gemm_flops = (int64_t)అడ్డువరుస_diff * src1_ncols * ne10;
            స్థిరరాశి తార్కిక use_mkl_ప్రత్యక్ష = gemm_flops < 256 * 256 * 256;
            ఒకవేళ (g_ggml_sycl_enable_dnn && !use_mkl_ప్రత్యక్ష && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::అడ్డువరుస_gemm(ctx, అడ్డువరుస_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(), src1_ddf1_i, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(),
                                          dst_dd_i, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(), stream);
            }
            లేకపోతే
#అయితే_ముగింపు
            {
                స్థిరరాశి దశాంశం alpha = 1.0f;
                స్థిరరాశి దశాంశం beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::స్తంభం_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, అడ్డువరుస_diff,
                    src1_ncols, ne10, dpct::get_విలువ(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_విలువ(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_వాడని(dst);
    GGML_వాడని(src1_ddq_i);
    GGML_వాడని(src1_padded_అడ్డువరుస_కొలత);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

అంతఃవరుస శూన్య ggml_sycl_op_కూడిక(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    కూడిక_అడ్డువరుసలు_f32_sycl(src0_dd, dst_dd, ne, 1, main_ప్రవాహం);
}

అంతఃవరుస శూన్య ggml_sycl_op_కూడిక_అడ్డువరుసలు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ncols = dst->src[0]->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    కూడిక_అడ్డువరుసలు_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ప్రవాహం);
}

అంతఃవరుస శూన్య ggml_sycl_op_సగటు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ncols = dst->src[0]->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    కూడిక_అడ్డువరుసలు_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_ప్రవాహం);

    main_ప్రవాహం->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> అడ్డువరుస) {
            dst_dd[అడ్డువరుస] /= ncols;
        }
    );
}


అంతఃవరుస శూన్య ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->దత్తాంశం);


    స్థిరరాశి int64_t ncols = dst->src[0]->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(dst->src[0]);

    గణన ggml_క్రమీకరించు_క్రమం క్రమం = (గణన ggml_క్రమీకరించు_క్రమం) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (పూర్ణాంకం *)dst_dd, ncols, nrows, క్రమం,
                         main_ప్రవాహం, ctx.device, ctx.pool());
}

స్థిర శూన్య ggml_sycl_op_శిఖరం_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
    int32_t * dst_dd = static_cast<int32_t *>(dst->దత్తాంశం);

    స్థిరరాశి పూర్ణాంకం k = dst->ne[0];
    స్థిరరాశి int64_t ncols = src0->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    ఒకవేళ (k <= SYCL_శిఖరం_K_పరిశీలన_విలీనం_MAX_K) {
        శిఖరం_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ప్రవాహం);
    } లేకపోతే {
        ggml_sycl_శిఖరం_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_ప్రవాహం);
    }
}

అంతఃవరుస శూన్య ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ncols = dst->src[0]->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_ప్రవాహం);
}

అంతఃవరుస శూన్య ggml_sycl_op_diag_ముసుగు_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ne00 = dst->src[0]->ne[0];
    స్థిరరాశి int64_t ne01 = dst->src[0]->ne[1];
    స్థిరరాశి పూర్ణాంకం nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    స్థిరరాశి పూర్ణాంకం n_past = ((int32_t *) dst->op_params)[0];

    diag_ముసుగు_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_ప్రవాహం);
}

స్థిర శూన్య tri_f32_sycl(
    స్థిరరాశి దశాంశం * src,
    దశాంశం * dst,
    స్థిరరాశి int64_t ne0,
    స్థిరరాశి int64_t ne1,
    స్థిరరాశి int64_t ne2,
    స్థిరరాశి int64_t ne3,
    స్థిరరాశి ggml_tri_type ttype,
    dpct::వరుస_ptr main_ప్రవాహం
) {
    స్థిరరాశి size_t మొత్తం = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_ప్రవాహం->parallel_for(sycl::range<1>(మొత్తం), [=](sycl::id<1> tid) {
        స్థిరరాశి int64_t idx = (int64_t) tid[0];

        స్థిరరాశి int64_t i0 = idx % ne0;
        స్థిరరాశి int64_t t1 = idx / ne0;
        స్థిరరాశి int64_t i1 = t1 % ne1;

        తార్కిక keep = అబద్ధం;
        ఎంపిక (ttype) {
            స్థితి GGML_TRI_TYPE_క్రింది:      keep = (i0 <  i1); ఆపు;
            స్థితి GGML_TRI_TYPE_క్రింది_DIAG: keep = (i0 <= i1); ఆపు;
            స్థితి GGML_TRI_TYPE_పై:      keep = (i0 >  i1); ఆపు;
            స్థితి GGML_TRI_TYPE_పై_DIAG: keep = (i0 >= i1); ఆపు;
            అప్రమేయ: keep = అబద్ధం; ఆపు;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

స్థిర శూన్య ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    స్థిరరాశి int64_t ne0 = src0->ne[0];
    స్థిరరాశి int64_t ne1 = src0->ne[1];
    స్థిరరాశి int64_t ne2 = src0->ne[2];
    స్థిరరాశి int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_ప్రవాహం);
}


అంతఃవరుస శూన్య ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    దశాంశం scale;
    దశాంశం bias;
    జ్ఞాపకం_నకలు(&scale, (దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&bias,  (దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_ప్రవాహం);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

స్థిర శూన్య ggml_sycl_సమితి_peer_access(స్థిరరాశి పూర్ణాంకం n_టోకెన్లు, [[maybe_unused]] పూర్ణాంకం main_device) {
    స్థిర తార్కిక peer_access_enabled = అబద్ధం;

    స్థిరరాశి తార్కిక enable_peer_access = n_టోకెన్లు <= GGML_SYCL_PEER_MAX_BATCH_కొలత;

    ఒకవేళ (peer_access_enabled == enable_peer_access) {
        తిరిగి;
    }

#నిర్వచితమైతే NDEBUG
    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        SYCL_CHECK(ggml_sycl_సమితి_device(i));
    }

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        SYCL_CHECK(ggml_sycl_సమితి_device(i));

        కొరకు (పూర్ణాంకం id_other = 0; id_other < ggml_sycl_info().device_లెక్క; ++id_other) {
            ఒకవేళ (i == id_other) {
                కొనసాగు;
            }
            ఒకవేళ (i != main_device && id_other != main_device) {
                కొనసాగు;
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
#అయితే_ముగింపు // NDEBUG

    peer_access_enabled = enable_peer_access;
}

మూస <మూస <పూర్ణాంకం> రకం_పేరు quantize_f>
స్థిర శూన్య ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0,
                                 స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) ప్రయత్నించు {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    స్థిరరాశి int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];

    స్థిరరాశి పూర్ణాంకం nb2 = dst->nb[2];
    స్థిరరాశి పూర్ణాంకం nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(dst->బఫరు));
    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(src1->బఫరు));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    స్థిరరాశి int64_t i02_భాజకం = ne12 / ne02;

    స్థిరరాశి size_t src0_ts = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t src0_bs = ggml_blck_కొలత(src0->type);
    స్థిరరాశి size_t q8_1_ts = పరిమాణం(ఖండం_q8_1);
    స్థిరరాశి size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    స్థిరరాశి తార్కిక src0_is_contiguous = ggml_is_contiguous(src0);
    స్థిరరాశి తార్కిక src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_నిలువు_కొలత = GGML_PAD(ne10, మాతృక_అడ్డువరుస_PADDING);

    స్థిరరాశి తార్కిక విభజించు = ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు);
    GGML_ASSERT(!(విభజించు && ne02 > 1));
    GGML_ASSERT(!(విభజించు && ne03 > 1));
    GGML_ASSERT(!(విభజించు && ne02 < ne12));

    std::array<దశాంశం, GGML_SYCL_MAX_DEVICES> tensor_విభజించు;
    ఒకవేళ (విభజించు) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_విభజించు_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_విభజించు_బఫరు_type_context *) src0->బఫరు->buft->context;
        tensor_విభజించు = buft_ctx->tensor_విభజించు;
    }

    నిర్మాణం dev_దత్తాంశం {
        ggml_sycl_pool_alloc<అక్షరం> src0_dd_alloc;
        ggml_sycl_pool_alloc<దశాంశం> src1_ddf_alloc;
        ggml_sycl_pool_alloc<అక్షరం> src1_ddq_alloc;
        ggml_sycl_pool_alloc<దశాంశం> dst_dd_alloc;

        అక్షరం *src0_dd = శూన్యసూచిక;
        దశాంశం *src1_ddf = శూన్యసూచిక; // float
        అక్షరం *src1_ddq = శూన్యసూచిక;  // q8_1
        దశాంశం *dst_dd = శూన్యసూచిక;

        int64_t అడ్డువరుస_దిగువ;
        int64_t అడ్డువరుస_ఎగువ;
    };

    dev_దత్తాంశం dev[GGML_SYCL_MAX_DEVICES];

    పూర్ణాంకం వాడిన_devices = 0;
    వరుస_ptr main_ప్రవాహం = ctx.stream();

    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        // by default, use all rows
        dev[i].అడ్డువరుస_దిగువ  = 0;
        dev[i].అడ్డువరుస_ఎగువ = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        ఒకవేళ (విభజించు) {
            స్థిరరాశి int64_t rounding = get_అడ్డువరుస_rounding(src0->type, tensor_విభజించు);

            ఒకవేళ (i != 0) {
                dev[i].అడ్డువరుస_దిగువ  = ne01*tensor_విభజించు[i];
                ఒకవేళ (dev[i].అడ్డువరుస_దిగువ < ne01) {
                    dev[i].అడ్డువరుస_దిగువ -= dev[i].అడ్డువరుస_దిగువ % rounding;
                }
            }

            ఒకవేళ (i != ggml_sycl_info().device_లెక్క - 1) {
                dev[i].అడ్డువరుస_ఎగువ  = ne01*tensor_విభజించు[i + 1];
                ఒకవేళ (dev[i].అడ్డువరుస_ఎగువ < ne01) {
                    dev[i].అడ్డువరుస_ఎగువ -= dev[i].అడ్డువరుస_ఎగువ % rounding;
                }
            }
        }
    }

    constexpr తార్కిక quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_కొలత>,
                                                      no_quantize_q8_1<QK8_1 / WARP_కొలత>>;
    కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
        ఒకవేళ ((!విభజించు && i != ctx.device) || dev[i].అడ్డువరుస_దిగువ == dev[i].అడ్డువరుస_ఎగువ) {
            కొనసాగు;
        }

        వాడిన_devices++;

        స్థిరరాశి తార్కిక src1_on_device = i == ctx.device;
        స్థిరరాశి తార్కిక  dst_on_device = i == ctx.device;

        ggml_sycl_సమితి_device(i);
        వరుస_ptr stream = ctx.stream(i, 0);

        ఒకవేళ (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = విభజించు ? (అక్షరం *) src0_extra->దత్తాంశం_device[i] : (అక్షరం *) src0->దత్తాంశం;
        } లేకపోతే {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        ఒకవేళ (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (దశాంశం *) src1->దత్తాంశం;
        } లేకపోతే {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        ఒకవేళ constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_నిలువు_కొలత*q8_1_ts/q8_1_bs);

            ఒకవేళ (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                ప్రయత్నించు {
                    quantize_అడ్డువరుస_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_నిలువు_కొలత, stream);
                } పట్టుకో (sycl::exception స్థిరరాశి &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
                    std::నిష్క్రమించు(1);
                }
            }
        }

        ఒకవేళ (dst_on_device) {
            dev[i].dst_dd = (దశాంశం *) dst->దత్తాంశం;
        } లేకపోతే {
            స్థిరరాశి size_t కొలత_dst_ddf = విభజించు ? (dev[i].అడ్డువరుస_ఎగువ - dev[i].అడ్డువరుస_దిగువ)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), కొలత_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    ఒకవేళ (విభజించు && వాడిన_devices > 1) {
        ggml_sycl_సమితి_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    స్థిరరాశి int64_t src1_నిలువు_stride = విభజించు && వాడిన_devices > 1 ? MUL_MAT_SRC1_నిలువు_STRIDE : ne11;
    కొరకు (int64_t src1_నిలువు_0 = 0; src1_నిలువు_0 < ne11; src1_నిలువు_0 += src1_నిలువు_stride) {
        స్థిరరాశి int64_t is = విభజించు ? (src1_నిలువు_0/src1_నిలువు_stride) % GGML_SYCL_MAX_STREAMS : 0;
        స్థిరరాశి int64_t src1_ncols = src1_నిలువు_0 + src1_నిలువు_stride > ne11 ? ne11 - src1_నిలువు_0 : src1_నిలువు_stride;
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
            ఒకవేళ ((!విభజించు && i != ctx.device) || dev[i].అడ్డువరుస_దిగువ == dev[i].అడ్డువరుస_ఎగువ) {
                కొనసాగు;
            }

            స్థిరరాశి తార్కిక src1_on_device = i == ctx.device;
            స్థిరరాశి తార్కిక  dst_on_device = i == ctx.device;
            స్థిరరాశి int64_t అడ్డువరుస_diff = dev[i].అడ్డువరుస_ఎగువ - dev[i].అడ్డువరుస_దిగువ;

            ggml_sycl_సమితి_device(i);
            వరుస_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            ఒకవేళ (విభజించు && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            కొరకు (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                స్థిరరాశి int64_t i03 = i0 / ne12;
                స్థిరరాశి int64_t i02 = i0 % ne12;

                స్థిరరాశి size_t src1_ddq_i_స్థానభ్రంశం = (i0*ne11 + src1_నిలువు_0) * src1_padded_నిలువు_కొలత*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                అక్షరం  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_భాజకం) * (ne01*ne00*src0_ts)/src0_bs;
                దశాంశం * src1_ddf_i = (దశాంశం *) ((అక్షరం *) dev[i].src1_ddf +
                    (i0*ne11 + src1_నిలువు_0) * ne10 * ggml_type_కొలత(src1->type));
                అక్షరం  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_స్థానభ్రంశం;
                దశాంశం *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_నిలువు_0) * (dst_on_device ? ne0 : అడ్డువరుస_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                ఒకవేళ (i == ctx.device) {
                    dst_dd_i += dev[i].అడ్డువరుస_దిగువ; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                ఒకవేళ (src1_is_contiguous) {
                    ఒకవేళ (i != ctx.device) {
                        ఒకవేళ constexpr (quantize_enabled) {
                            అక్షరం * src1_ddq_i_ఆధారం = dev[ctx.device].src1_ddq + src1_ddq_i_స్థానభ్రంశం;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->జ్ఞాపకం_నకలు(src1_ddq_i, src1_ddq_i_ఆధారం,
                                                             src1_ncols * src1_padded_నిలువు_కొలత * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } లేకపోతే {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            స్థిరరాశి అక్షరం * src1_ddf_i_ఆధారం = (స్థిరరాశి అక్షరం *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_నిలువు_0) * ne10 * ggml_type_కొలత(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_ప్రవాహం, src1_ddf_i, src1_ddf_i_ఆధారం,
                                                               src1_ncols * ne10 * ggml_type_కొలత(src1->type))));
                        }
                    }
                } లేకపోతే {
                    ఒకవేళ (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_నిలువు_0,
                                                           src1_నిలువు_0 + src1_ncols, stream));
                    } లేకపోతే {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    ఒకవేళ constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        ప్రయత్నించు {
                            quantize_అడ్డువరుస_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_నిలువు_కొలత, stream);
                        } పట్టుకో (స్థిరరాశి sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
                            std::నిష్క్రమించు(1);
                        }
                    }
                }

                ఒకవేళ (src1_నిలువు_0 == 0 && !src0_is_contiguous && i02 % i02_భాజకం == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_భాజకం, dev[i].అడ్డువరుస_దిగువ, dev[i].అడ్డువరుస_ఎగువ, stream));
                }
                ఒకవేళ (src1->type == GGML_TYPE_F16) {
                    src1_padded_నిలువు_కొలత = (i0 * ne11 + src1_నిలువు_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].అడ్డువరుస_దిగువ, dev[i].అడ్డువరుస_ఎగువ, src1_ncols, src1_padded_నిలువు_కొలత, stream)));

                // copy dst to host or other device if necessary
                ఒకవేళ (!dst_on_device) {
                    శూన్య * dst_off_device = dst->దత్తాంశం;
                    ఒకవేళ (విభజించు) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        దశాంశం * dhf_dst_i = (దశాంశం *) ((అక్షరం *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*పరిమాణం(దశాంశం));
                        dhf_dst_i += src1_నిలువు_0*ne0 + dev[i].అడ్డువరుస_దిగువ;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * పరిమాణం(దశాంశం), dst_dd_i,
                            అడ్డువరుస_diff * పరిమాణం(దశాంశం), అడ్డువరుస_diff * పరిమాణం(దశాంశం),
                            src1_ncols, dpct::device_గమ్యం_device, *stream)));
                    } లేకపోతే {
                        దశాంశం * dhf_dst_i = (దశాంశం *) ((అక్షరం *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*పరిమాణం(దశాంశం));
                        dhf_dst_i += src1_నిలువు_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->జ్ఞాపకం_నకలు(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * పరిమాణం(దశాంశం)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                ఒకవేళ (విభజించు && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    ఒకవేళ (విభజించు && ggml_sycl_info().device_లెక్క > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_నిలువు_STRIDE - 1) / MUL_MAT_SRC1_నిలువు_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_సమితి_device(ctx.device);
        కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; ++i) {
            ఒకవేళ (dev[i].అడ్డువరుస_దిగువ == dev[i].అడ్డువరుస_ఎగువ) {
                కొనసాగు;
            }
            కొరకు (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_sycl_repeat_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_వెనుక(ctx, dst);
}

స్థిర శూన్య ggml_sycl_get_అడ్డువరుసలు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_అడ్డువరుసలు(ctx, dst);
}

స్థిర శూన్య ggml_sycl_get_అడ్డువరుసలు_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_అడ్డువరుసలు_వెనుక(ctx, dst);
}

స్థిర శూన్య ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

స్థిర శూన్య ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

స్థిర శూన్య ggml_sycl_rms_norm_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_వెనుక(ctx, dst);
}

స్థిర శూన్య ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

స్థిర శూన్య ggml_sycl_సమూహం_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_సమూహం_norm(ctx, dst);
}

స్థిర శూన్య ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0,
                                       స్థిరరాశి ggml_tensor *src1,
                                       ggml_tensor *dst) ప్రయత్నించు {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];

    స్థిరరాశి int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    వరుస_ptr main_ప్రవాహం = ctx.stream();

    శూన్య  * src0_ddq = src0->దత్తాంశం;
    దశాంశం * src1_ddf = (దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_ddf  = (దశాంశం *) dst->దత్తాంశం;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_ప్రవాహం);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0,
                                     స్థిరరాశి ggml_tensor *src1,
                                     ggml_tensor *dst) ప్రయత్నించు {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];

    స్థిరరాశి int64_t nb01 = src0->nb[1];
    స్థిరరాశి int64_t nb02 = src0->nb[2];

    స్థిరరాశి int64_t ne12 = src1->ne[2];
    స్థిరరాశి int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    వరుస_ptr main_ప్రవాహం = ctx.stream();

    శూన్య  * src0_ddq = src0->దత్తాంశం;
    దశాంశం * src1_ddf = (దశాంశం *) src1->దత్తాంశం;
    దశాంశం * dst_ddf  = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి int64_t అడ్డువరుస_stride_x = nb01 / పరిమాణం(sycl::half);
    స్థిరరాశి int64_t channel_stride_x = nb02 / పరిమాణం(sycl::half);
    స్థిరరాశి int64_t channel_stride_y = nb11 / పరిమాణం(దశాంశం);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, అడ్డువరుస_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_ప్రవాహం);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య k_compute_batched_ptrs(స్థిరరాశి sycl::half * src0_as_f16, స్థిరరాశి sycl::half * src1_as_f16, శూన్య * dst,
                                   స్థిరరాశి శూన్య ** ptrs_src, శూన్య ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి int64_t i13 = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) + అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t i12 = అంశం_ct1.get_సమూహం(1) * అంశం_ct1.get_local_range(1) + అంశం_ct1.get_local_id(1);

    ఒకవేళ (i13 >= ne13 || i12 >= ne12) {
        తిరిగి;
    }

    స్థిరరాశి int64_t i03 = i13 / r3;
    స్థిరరాశి int64_t i02 = i12 / r2;

    స్థిరరాశి uint8_t * src0_బైట్లు = reinterpret_cast<స్థిరరాశి uint8_t *>(src0_as_f16);
    స్థిరరాశి uint8_t * src1_బైట్లు = reinterpret_cast<స్థిరరాశి uint8_t *>(src1_as_f16);
    uint8_t *       dst_బైట్లు  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_బైట్లు + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_బైట్లు + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_బైట్లు + i12 * nbd2 + i13 * nbd3;
}

స్థిర శూన్య ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0,
                                           స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) ప్రయత్నించు {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    వరుస_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    స్థిరరాశి sycl::half * src0_f16 = static_cast<స్థిరరాశి sycl::half *>(src0->దత్తాంశం);
    దశాంశం *            dst_ddf  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి sycl::half * src1_f16       = static_cast<స్థిరరాశి sycl::half *>(src1->దత్తాంశం);
#ఒకవేళ GGML_SYCL_DNNL
    స్థిరరాశి size_t       type_కొలత_src0 = ggml_type_కొలత(src0->type);
#అయితే_ముగింపు
    స్థిరరాశి size_t       type_కొలత_src1 = ggml_type_కొలత(src1->type);

    తార్కిక is_src0_cont_2 = ggml_is_contiguous_2(src0);
    తార్కిక is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_కొలత_src1;
    int64_t                          s12 = nb12 / type_కొలత_src1;
    int64_t                          s13 = nb13 / type_కొలత_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    ఒకవేళ (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#ఒకవేళ GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        పూర్ణాంకం చివరి_dim=0;
        పూర్ణాంకం చివరి_str=0;
        size_t అతిపెద్ద_str=0;
        కొరకు(పూర్ణాంకం i = 0; i< 4; i++){
            // last stride is always the largest
            ఒకవేళ(src1->nb[i] == అతిపెద్ద_str){
                ఒకవేళ(src1->ne[చివరి_dim] == 1){
                    చివరి_str = i;
                    చివరి_dim = i;
                }
            }
            ఒకవేళ(src1->nb[i] > అతిపెద్ద_str){
                అతిపెద్ద_str = src1->nb[i];
                చివరి_str = i;
                చివరి_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        స్థిరరాశి int64_t ne_src1 = src1->nb[చివరి_str] * src1->ne[చివరి_dim] / type_కొలత_src1;
        src1_f16_alloc.alloc(ne_src1);
        స్థిరరాశి గమ్యం_fp16_sycl_t గమ్యం_fp16_sycl = ggml_get_గమ్యం_fp16_sycl(src1->type, dst);
        GGML_ASSERT(గమ్యం_fp16_sycl != శూన్యసూచిక);
        గమ్యం_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# లేకపోతే
        స్థిరరాశి int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        స్థిరరాశి గమ్యం_fp16_nc_sycl_t గమ్యం_fp16_nc_sycl = ggml_get_గమ్యం_fp16_nc_sycl(src1->type);
        GGML_ASSERT(గమ్యం_fp16_nc_sycl != శూన్యసూచిక);
        గమ్యం_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#అయితే_ముగింపు

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = నిజం;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_దత్తాంశం_t mkl_compute_type = dpct::library_దత్తాంశం_t::real_float;
    dpct::library_దత్తాంశం_t mkl_దత్తాంశం_type    = dpct::library_దత్తాంశం_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    స్థిరరాశి దశాంశం alpha_f32 = 1.0f;
    స్థిరరాశి దశాంశం beta_f32  = 0.0f;

    స్థిరరాశి శూన్య * alpha = &alpha_f32;
    స్థిరరాశి శూన్య * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

#ఒకవేళ GGML_SYCL_DNNL
    ఒకవేళ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_కొలత_src0;
            int64_t str_a1 = nb01 / type_కొలత_src0;
            int64_t str_a2 = nb02 / type_కొలత_src0;

            int64_t str_b0 = nb10 / type_కొలత_src1;
            int64_t str_b1 = nb11 / type_కొలత_src1;
            int64_t str_b2 = nb12 / type_కొలత_src1;

            స్వయంచాలక launch_gemm_for_batches = [&ctx, queue](స్థిరరాశి sycl::half *src0,
                                                స్థిరరాశి sycl::half *src1, దశాంశం *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                తార్కిక supported_ప్రసారం = batchb == batcha ? నిజం
                        : batchb == 1 || batcha == 1        ? నిజం
                                                            : అబద్ధం;
                ఒకవేళ (supported_ప్రసారం) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(), queue, batcha, batchb);
                } లేకపోతే {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    ఒకవేళ (batches0 > batches1) {
                        int64_t అంకె_mul_mats = batches1;
                        int64_t sub_batch = batches0 / అంకె_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        కొరకు (int64_t i0 = 0; i0 < అంకె_mul_mats; i0++) {
                            స్థిరరాశి sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            స్థిరరాశి sycl::half *src1_shifted = src1 + (sb2 * i0);
                            దశాంశం *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(),
                                    queue, sub_batch, 1);
                        }
                    } లేకపోతే {
                        int64_t అంకె_mul_mats = batches0;
                        int64_t sub_batch = batches1 / అంకె_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        కొరకు (int64_t i1 = 0; i1 < అంకె_mul_mats; i1++) {
                            స్థిరరాశి sycl::half *src0_shifted = src0 + (sa2 * i1);
                            స్థిరరాశి sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            దశాంశం *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::గమ్యం_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::గమ్యం_dt<దశాంశం>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            స్థిరరాశి తార్కిక cont_batches_dim2_a = nb02 * ne02 == nb03;
            స్థిరరాశి తార్కిక cont_batches_dim2_b = nb12 * ne12 == nb13;
            స్థిరరాశి తార్కిక cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            స్థిరరాశి తార్కిక cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            ఒకవేళ (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / పరిమాణం(దశాంశం));
            } లేకపోతే ఒకవేళ (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_కొలత_src0;
                int64_t str_b3 = nb13 / type_కొలత_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / పరిమాణం(దశాంశం));
            } లేకపోతే {
                కొరకు (int64_t b_a = 0; b_a < ne03; b_a++) {
                    స్థిరరాశి sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_కొలత_src0);
                    స్థిరరాశి sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_కొలత_src1);
                    దశాంశం *dst_shifted = dst_ddf + (nb3 * b_a / పరిమాణం(దశాంశం));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / పరిమాణం(దశాంశం));
                }
            }

    }
    లేకపోతే
#అయితే_ముగింపు
    {
        ఒకవేళ (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            స్థిరరాశి int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            స్థిరరాశి int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_దత్తాంశం_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_దత్తాంశం_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_దత్తాంశం_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } లేకపోతే {
            స్థిరరాశి పూర్ణాంకం ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<స్థిరరాశి శూన్య *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<శూన్య *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<మాతృక_info_t<దశాంశం>> మాతృక_info(ctx.host_pool(), 1);

            sycl::range<3> ఖండం_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                స్థిరరాశి శూన్య ** ptrs_src_get = ptrs_src.get();
                శూన్య **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * పరిమాణం(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * పరిమాణం(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(ఖండం_dims, ఖండం_dims), [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, అంశం_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (స్థిరరాశి శూన్య **) (ptrs_src.get() + 0 * ne23), dpct::library_దత్తాంశం_t::real_half, nb01 / nb00,
                (స్థిరరాశి శూన్య **) (ptrs_src.get() + 1 * ne23), dpct::library_దత్తాంశం_t::real_half, s11, beta,
                (శూన్య **) (ptrs_dst.get() + 0 * ne23), mkl_దత్తాంశం_type, ne0, ne23, mkl_compute_type, మాతృక_info.get())));
        }
    }
} పట్టుకో (స్థిరరాశి sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

గణన తరగతి mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

అంతఃవరుస తార్కిక ggml_sycl_supports_mmq(గణన ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_వాడని(type);
    తిరిగి అబద్ధం;
}

అంతఃవరుస తార్కిక ggml_sycl_supports_reorder_mul_mat_sycl(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి నిజం;
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
            తిరిగి !g_ggml_sycl_prioritize_dmmv;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

అంతఃవరుస తార్కిక ggml_sycl_supports_reorder_dmmv(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

అంతఃవరుస తార్కిక ggml_sycl_supports_reorder_mmvq(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_IQ3_XXS:
        స్థితి GGML_TYPE_IQ3_S:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక ggml_sycl_supports_reorder_esimd(గణన ggml_type type) {
#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
#లేకపోతే
    GGML_వాడని(type);
    తిరిగి అబద్ధం;
#అయితే_ముగింపు
}

స్థిర తార్కిక ggml_sycl_supports_dmmv(గణన ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q1_0:
        స్థితి GGML_TYPE_Q4_0:
        స్థితి GGML_TYPE_Q4_1:
        స్థితి GGML_TYPE_Q5_0:
        స్థితి GGML_TYPE_Q5_1:
        స్థితి GGML_TYPE_Q8_0:
        స్థితి GGML_TYPE_Q2_K:
        స్థితి GGML_TYPE_Q3_K:
        స్థితి GGML_TYPE_Q4_K:
        స్థితి GGML_TYPE_Q5_K:
        స్థితి GGML_TYPE_Q6_K:
        స్థితి GGML_TYPE_F16:
        స్థితి GGML_TYPE_BF16:
            తిరిగి నిజం;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
స్థిర అంతఃవరుస శూన్య * sycl_ext_malloc_device(dpct::వరుస_ptr stream, size_t size) {
    తార్కిక use_async = g_ggml_sycl_use_async_mem_op;
#ఒకవేళ defined(GGML_SYCL_లేఖాచిత్రం) && SYCL_EXT_ONEAPI_ASYNC_జ్ఞాపకం_ALLOC
    ఒకవేళ (use_async) {
        శూన్య * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_కూడు(GGML_SYCL_MEM_ASYNC, ptr, size);
        తిరిగి ptr;
    }
#లేకపోతే
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#అయితే_ముగింపు
    తిరిగి ggml_sycl_malloc_device(size, *stream);
}

స్థిర అంతఃవరుస శూన్య sycl_ext_free(dpct::వరుస_ptr stream, శూన్య * ptr) {
    తార్కిక use_async = g_ggml_sycl_use_async_mem_op;
#ఒకవేళ defined(GGML_SYCL_లేఖాచిత్రం) && SYCL_EXT_ONEAPI_ASYNC_జ్ఞాపకం_ALLOC
    ఒకవేళ (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        తిరిగి;
    }
#లేకపోతే
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#అయితే_ముగింపు
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
నిర్మాణం sycl_reorder_తాత్కాలిక_బఫరు {
    శూన్య *          ptr  = శూన్యసూచిక;
    dpct::వరుస_ptr stream;

    sycl_reorder_తాత్కాలిక_బఫరు(dpct::వరుస_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#నిర్వచితమైతే GGML_SYCL_HOST_MEM_FALLBACK
        ఒకవేళ (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            ఒకవేళ (ptr) {
                host_fallback = నిజం;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#అయితే_ముగింపు
    }

    ~sycl_reorder_తాత్కాలిక_బఫరు() {
        ఒకవేళ (!ptr) {
            తిరిగి;
        }
        ఒకవేళ (host_fallback) {
            sycl::విడుదల(ptr, *stream);
        } లేకపోతే {
            sycl_ext_free(stream, ptr);
        }
    }

    స్పష్టమైన క్రియాకారకం తార్కిక() స్థిరరాశి { తిరిగి ptr != శూన్యసూచిక; }

    sycl_reorder_తాత్కాలిక_బఫరు(స్థిరరాశి sycl_reorder_తాత్కాలిక_బఫరు &)            = తొలగించు;
    sycl_reorder_తాత్కాలిక_బఫరు & క్రియాకారకం=(స్థిరరాశి sycl_reorder_తాత్కాలిక_బఫరు &) = తొలగించు;

అంతరంగ:
    తార్కిక host_fallback = అబద్ధం;
};

స్థిర తార్కిక reorder_qw_q4_0(uint8_t * దత్తాంశం_device, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, size_t size, size_t స్థానభ్రంశం,
                            dpct::వరుస_ptr stream) {
    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % పరిమాణం(ఖండం_q4_0) == 0));
    GGML_ASSERT((స్థానభ్రంశం % పరిమాణం(ఖండం_q4_0) == 0));
    పూర్ణాంకం స్థానభ్రంశం_blks = స్థానభ్రంశం / పరిమాణం(ఖండం_q4_0);
    స్వయంచాలక qs_ptr      = దత్తాంశం_device + స్థానభ్రంశం_blks * QK4_0 / 2;
    స్వయంచాలక d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + స్థానభ్రంశం_blks;

    స్వయంచాలక reorder_event = stream->parallel_for(
        size / పరిమాణం(ఖండం_q4_0),
            [=](స్వయంచాలక i) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            స్థిరరాశి ఖండం_q4_0* x = (స్థిరరాశి ఖండం_q4_0*)tmp_buf;
            స్థిరరాశి పూర్ణాంకం ib = i;

            కొరకు (పూర్ణాంకం j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q8_0(uint8_t * దత్తాంశం_device, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, size_t size, size_t స్థానభ్రంశం,
                            dpct::వరుస_ptr stream) {
    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % పరిమాణం(ఖండం_q8_0) == 0));
    GGML_ASSERT((స్థానభ్రంశం % పరిమాణం(ఖండం_q8_0) == 0));
    పూర్ణాంకం స్థానభ్రంశం_blks = స్థానభ్రంశం / పరిమాణం(ఖండం_q8_0);
    స్వయంచాలక qs_ptr = దత్తాంశం_device + స్థానభ్రంశం_blks * QK8_0;
    స్వయంచాలక d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + స్థానభ్రంశం_blks;

    స్వయంచాలక reorder_event = stream->parallel_for(
        size / పరిమాణం(ఖండం_q8_0),
            [=](స్వయంచాలక i) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
            స్థిరరాశి ఖండం_q8_0* x = (స్థిరరాశి ఖండం_q8_0*)tmp_buf;
            స్థిరరాశి పూర్ణాంకం ib = i;

            కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
స్థిర తార్కిక reorder_qw_q8_0_moe(uint8_t * దత్తాంశం_device, size_t expert_బైట్లు, int64_t n_expert, dpct::వరుస_ptr stream) {
    GGML_ASSERT(expert_బైట్లు % పరిమాణం(ఖండం_q8_0) == 0);
    స్థిరరాశి పూర్ణాంకం    ఖండాలు_చొప్పున_expert = (పూర్ణాంకం) (expert_బైట్లు / పరిమాణం(ఖండం_q8_0));
    స్థిరరాశి size_t మొత్తం_బైట్లు       = expert_బైట్లు * (size_t) n_expert;

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, మొత్తం_బైట్లు);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, మొత్తం_బైట్లు);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, మొత్తం_బైట్లు)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్థిరరాశి పూర్ణాంకం మొత్తం_ఖండాలు = ఖండాలు_చొప్పున_expert * (పూర్ణాంకం) n_expert;
    స్వయంచాలక reorder_event = stream->parallel_for(మొత్తం_ఖండాలు, [=](స్వయంచాలక gb_) {
        స్థిరరాశి పూర్ణాంకం          gb   = gb_;
        స్థిరరాశి పూర్ణాంకం          e    = gb / ఖండాలు_చొప్పున_expert;
        స్థిరరాశి పూర్ణాంకం          ib   = gb % ఖండాలు_చొప్పున_expert;
        స్థిరరాశి ఖండం_q8_0 * x    = (స్థిరరాశి ఖండం_q8_0 *) (tmp_buf + (size_t) e * expert_బైట్లు);
        uint8_t *          పునాది = దత్తాంశం_device + (size_t) e * expert_బైట్లు;

        స్వయంచాలక * qs_ptr = పునాది;
        స్వయంచాలక * d_ptr  = (sycl::half *) (qs_ptr + (size_t) ఖండాలు_చొప్పున_expert * QK8_0);

        కొరకు (పూర్ణాంకం j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q4_k(uint8_t * దత్తాంశం_device, size_t size, size_t స్థానభ్రంశం, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_q4_K) == 0);
    GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(ఖండం_q4_K) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_q4_K);

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక * qs_ptr     = దత్తాంశం_device;
    స్వయంచాలక * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    స్వయంచాలక * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_కొలత * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_q4_K * x  = (స్థిరరాశి ఖండం_q4_K *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం          ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < K_SCALE_కొలత; ++j) {
            scales_ptr[ib * K_SCALE_కొలత + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

// Reorder each expert slice into a self-contained SoA layout.
స్థిర తార్కిక reorder_qw_q4_k_moe(uint8_t * దత్తాంశం_device, size_t expert_బైట్లు, int64_t n_expert, dpct::వరుస_ptr stream) {
    GGML_ASSERT(expert_బైట్లు % పరిమాణం(ఖండం_q4_K) == 0);
    స్థిరరాశి పూర్ణాంకం    ఖండాలు_చొప్పున_expert = (పూర్ణాంకం) (expert_బైట్లు / పరిమాణం(ఖండం_q4_K));
    స్థిరరాశి size_t మొత్తం_బైట్లు       = expert_బైట్లు * (size_t) n_expert;

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, మొత్తం_బైట్లు);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, మొత్తం_బైట్లు);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, మొత్తం_బైట్లు)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్థిరరాశి పూర్ణాంకం మొత్తం_ఖండాలు = ఖండాలు_చొప్పున_expert * (పూర్ణాంకం) n_expert;
    స్వయంచాలక reorder_event = stream->parallel_for(మొత్తం_ఖండాలు, [=](స్వయంచాలక gb_) {
        స్థిరరాశి పూర్ణాంకం          gb   = gb_;
        స్థిరరాశి పూర్ణాంకం          e    = gb / ఖండాలు_చొప్పున_expert;
        స్థిరరాశి పూర్ణాంకం          ib   = gb % ఖండాలు_చొప్పున_expert;
        స్థిరరాశి ఖండం_q4_K * x    = (స్థిరరాశి ఖండం_q4_K *) (tmp_buf + (size_t) e * expert_బైట్లు);
        uint8_t *          పునాది = దత్తాంశం_device + (size_t) e * expert_బైట్లు;

        స్వయంచాలక * qs_ptr     = పునాది;
        స్వయంచాలక * scales_ptr = qs_ptr + QK_K / 2 * ఖండాలు_చొప్పున_expert;
        స్వయంచాలక * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_కొలత * ఖండాలు_చొప్పున_expert);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < K_SCALE_కొలత; ++j) {
            scales_ptr[ib * K_SCALE_కొలత + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
స్థిర తార్కిక reorder_qw_q5_k_moe(uint8_t * దత్తాంశం_device, size_t expert_బైట్లు, int64_t n_expert, dpct::వరుస_ptr stream) {
    GGML_ASSERT(expert_బైట్లు % పరిమాణం(ఖండం_q5_K) == 0);
    స్థిరరాశి పూర్ణాంకం    ఖండాలు_చొప్పున_expert = (పూర్ణాంకం) (expert_బైట్లు / పరిమాణం(ఖండం_q5_K));
    స్థిరరాశి size_t మొత్తం_బైట్లు       = expert_బైట్లు * (size_t) n_expert;

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, మొత్తం_బైట్లు);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, మొత్తం_బైట్లు);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, మొత్తం_బైట్లు)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్థిరరాశి పూర్ణాంకం మొత్తం_ఖండాలు = ఖండాలు_చొప్పున_expert * (పూర్ణాంకం) n_expert;
    స్వయంచాలక reorder_event = stream->parallel_for(మొత్తం_ఖండాలు, [=](స్వయంచాలక gb_) {
        స్థిరరాశి పూర్ణాంకం          gb   = gb_;
        స్థిరరాశి పూర్ణాంకం          e    = gb / ఖండాలు_చొప్పున_expert;
        స్థిరరాశి పూర్ణాంకం          ib   = gb % ఖండాలు_చొప్పున_expert;
        స్థిరరాశి ఖండం_q5_K * x    = (స్థిరరాశి ఖండం_q5_K *) (tmp_buf + (size_t) e * expert_బైట్లు);
        uint8_t *          పునాది = దత్తాంశం_device + (size_t) e * expert_బైట్లు;

        స్వయంచాలక * qs_ptr     = పునాది;
        స్వయంచాలక * qh_ptr     = qs_ptr + (QK_K / 2) * ఖండాలు_చొప్పున_expert;
        స్వయంచాలక * scales_ptr = qh_ptr + (QK_K / 8) * ఖండాలు_చొప్పున_expert;
        స్వయంచాలక * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_కొలత * ఖండాలు_చొప్పున_expert);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < K_SCALE_కొలత; ++j) {
            scales_ptr[ib * K_SCALE_కొలత + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
స్థిర తార్కిక reorder_qw_q6_k_moe(uint8_t * దత్తాంశం_device, size_t expert_బైట్లు, int64_t n_expert, dpct::వరుస_ptr stream) {
    GGML_ASSERT(expert_బైట్లు % పరిమాణం(ఖండం_q6_K) == 0);
    స్థిరరాశి పూర్ణాంకం    ఖండాలు_చొప్పున_expert = (పూర్ణాంకం) (expert_బైట్లు / పరిమాణం(ఖండం_q6_K));
    స్థిరరాశి size_t మొత్తం_బైట్లు       = expert_బైట్లు * (size_t) n_expert;

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, మొత్తం_బైట్లు);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, మొత్తం_బైట్లు);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, మొత్తం_బైట్లు)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్థిరరాశి పూర్ణాంకం మొత్తం_ఖండాలు = ఖండాలు_చొప్పున_expert * (పూర్ణాంకం) n_expert;
    స్వయంచాలక reorder_event = stream->parallel_for(మొత్తం_ఖండాలు, [=](స్వయంచాలక gb_) {
        స్థిరరాశి పూర్ణాంకం          gb   = gb_;
        స్థిరరాశి పూర్ణాంకం          e    = gb / ఖండాలు_చొప్పున_expert;
        స్థిరరాశి పూర్ణాంకం          ib   = gb % ఖండాలు_చొప్పున_expert;
        స్థిరరాశి ఖండం_q6_K * x    = (స్థిరరాశి ఖండం_q6_K *) (tmp_buf + (size_t) e * expert_బైట్లు);
        uint8_t *          పునాది = దత్తాంశం_device + (size_t) e * expert_బైట్లు;

        స్వయంచాలక * ql_ptr     = పునాది;
        స్వయంచాలక * qh_ptr     = ql_ptr + (QK_K / 2) * ఖండాలు_చొప్పున_expert;
        స్వయంచాలక * scales_ptr = qh_ptr + (QK_K / 4) * ఖండాలు_చొప్పున_expert;
        స్వయంచాలక * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * ఖండాలు_చొప్పున_expert);

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
స్థిర తార్కిక reorder_qw_mxfp4_moe(uint8_t * దత్తాంశం_device, size_t expert_బైట్లు, int64_t n_expert, dpct::వరుస_ptr stream) {
    GGML_ASSERT(expert_బైట్లు % పరిమాణం(ఖండం_mxfp4) == 0);
    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_expert = (పూర్ణాంకం) (expert_బైట్లు / పరిమాణం(ఖండం_mxfp4));
    స్థిరరాశి size_t  max_chunk_బైట్లు   = 32u << 20;
    స్థిరరాశి int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_బైట్లు / expert_బైట్లు)));

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, (size_t) chunk_experts * expert_బైట్లు);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_బైట్లు);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    కొరకు (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        స్థిరరాశి int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = దత్తాంశం_device + (size_t) e0 * expert_బైట్లు;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, chunk, (size_t) n_chunk * expert_బైట్లు)));
        ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        స్థిరరాశి పూర్ణాంకం మొత్తం_ఖండాలు = ఖండాలు_చొప్పున_expert * (పూర్ణాంకం) n_chunk;
        స్వయంచాలక reorder_event = stream->parallel_for(మొత్తం_ఖండాలు, [=](స్వయంచాలక gb_) {
            స్థిరరాశి పూర్ణాంకం           gb   = gb_;
            స్థిరరాశి పూర్ణాంకం           e    = gb / ఖండాలు_చొప్పున_expert;
            స్థిరరాశి పూర్ణాంకం           ib   = gb % ఖండాలు_చొప్పున_expert;
            స్థిరరాశి ఖండం_mxfp4 * x    = (స్థిరరాశి ఖండం_mxfp4 *) (tmp_buf + (size_t) e * expert_బైట్లు);
            uint8_t *           పునాది = chunk + (size_t) e * expert_బైట్లు;

            uint8_t * qs_ptr = పునాది;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) ఖండాలు_చొప్పున_expert;

            కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q2_k(uint8_t * దత్తాంశం_device, size_t size, size_t స్థానభ్రంశం, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_q2_K) == 0);
    GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(ఖండం_q2_K) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_q2_K);

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక *        qs_ptr     = దత్తాంశం_device;
    స్వయంచాలక *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_q2_K * x  = (స్థిరరాశి ఖండం_q2_K *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం          ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q3_k(uint8_t * దత్తాంశం_device, size_t size, size_t స్థానభ్రంశం, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_q3_K) == 0);
    GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(ఖండం_q3_K) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_q3_K);

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక *       qs_ptr     = దత్తాంశం_device;
    స్వయంచాలక *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    స్వయంచాలక *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_q3_K * x  = (స్థిరరాశి ఖండం_q3_K *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం          ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q5_k(uint8_t * దత్తాంశం_device, size_t size, size_t స్థానభ్రంశం, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_q5_K) == 0);
    GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(ఖండం_q5_K) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_q5_K);

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక * qs_ptr     = దత్తాంశం_device;
    స్వయంచాలక * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    స్వయంచాలక * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    స్వయంచాలక * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_కొలత * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_q5_K * x  = (స్థిరరాశి ఖండం_q5_K *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం          ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < K_SCALE_కొలత; ++j) {
            scales_ptr[ib * K_SCALE_కొలత + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_q6_k(uint8_t * దత్తాంశం_device, size_t size, size_t స్థానభ్రంశం, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_q6_K) == 0);
    GGML_ASSERT(స్థానభ్రంశం % పరిమాణం(ఖండం_q6_K) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_q6_K);

    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక *       ql_ptr     = దత్తాంశం_device;
    స్వయంచాలక *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    స్వయంచాలక *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_q6_K * x  = (స్థిరరాశి ఖండం_q6_K *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం          ib = i;

        స్థిరరాశి uint8_t * ql              = x[ib].ql;
        స్థిరరాశి uint8_t * qh              = x[ib].qh;
        uint8_t *       పునాది_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       పునాది_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       పునాది_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 2; ++j) {
            పునాది_ql_ptr[j] = ql[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; ++j) {
            పునాది_qh_ptr[j] = qh[j];
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 16; ++j) {
            పునాది_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_iq3_s(uint8_t * దత్తాంశం_device, size_t size, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_iq3_s) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_iq3_s);
    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక * qs_ptr       = దత్తాంశం_device;
    స్వయంచాలక * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    స్వయంచాలక * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    స్వయంచాలక * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_iq3_s * x = (స్థిరరాశి ఖండం_iq3_s *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (పరిమాణం(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        కొరకు (పూర్ణాంకం j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[పరిమాణం(ggml_half) + j] = x[ib].scales[j];
        }
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw_iq3_xxs(uint8_t * దత్తాంశం_device, size_t size, dpct::వరుస_ptr stream) {
    GGML_ASSERT(size % పరిమాణం(ఖండం_iq3_xxs) == 0);

    స్థిరరాశి పూర్ణాంకం nblocks = size / పరిమాణం(ఖండం_iq3_xxs);
    sycl_reorder_తాత్కాలిక_బఫరు tmp(stream, size);
    ఒకవేళ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        తిరిగి అబద్ధం;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->జ్ఞాపకం_నకలు(tmp_buf, దత్తాంశం_device, size)));
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    స్వయంచాలక * qs_ptr = దత్తాంశం_device;
    స్వయంచాలక * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    స్వయంచాలక reorder_event = stream->parallel_for(nblocks, [=](స్వయంచాలక i) {
        స్థిరరాశి ఖండం_iq3_xxs * x = (స్థిరరాశి ఖండం_iq3_xxs *) tmp_buf;
        స్థిరరాశి పూర్ణాంకం ib = i;

        కొరకు (పూర్ణాంకం j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    తిరిగి నిజం;
}

స్థిర తార్కిక reorder_qw(స్థిరరాశి ggml_tensor * src0, dpct::వరుస_ptr stream) {
    uint8_t * దత్తాంశం_device = (uint8_t *) src0->దత్తాంశం;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    ఒకవేళ (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        ఎంపిక (src0->type) {
            స్థితి GGML_TYPE_Q8_0:
                తిరిగి reorder_qw_q8_0_moe(దత్తాంశం_device, src0->nb[2], src0->ne[2], stream);
            స్థితి GGML_TYPE_Q4_K:
                తిరిగి reorder_qw_q4_k_moe(దత్తాంశం_device, src0->nb[2], src0->ne[2], stream);
            స్థితి GGML_TYPE_Q5_K:
                తిరిగి reorder_qw_q5_k_moe(దత్తాంశం_device, src0->nb[2], src0->ne[2], stream);
            స్థితి GGML_TYPE_Q6_K:
                తిరిగి reorder_qw_q6_k_moe(దత్తాంశం_device, src0->nb[2], src0->ne[2], stream);
            స్థితి GGML_TYPE_MXFP4:
                తిరిగి reorder_qw_mxfp4_moe(దత్తాంశం_device, src0->nb[2], src0->ne[2], stream);
            అప్రమేయ:
                తిరిగి అబద్ధం;
        }
    }

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_Q4_0:
            తిరిగి reorder_qw_q4_0(దత్తాంశం_device, ncols, nrows, size, 0, stream);
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి reorder_qw_q8_0(దత్తాంశం_device, ncols, nrows, size, 0, stream);
        స్థితి GGML_TYPE_Q2_K:
            తిరిగి reorder_qw_q2_k(దత్తాంశం_device, size, 0, stream);
        స్థితి GGML_TYPE_Q3_K:
            తిరిగి reorder_qw_q3_k(దత్తాంశం_device, size, 0, stream);
        స్థితి GGML_TYPE_Q4_K:
            తిరిగి reorder_qw_q4_k(దత్తాంశం_device, size, 0, stream);
        స్థితి GGML_TYPE_Q5_K:
            తిరిగి reorder_qw_q5_k(దత్తాంశం_device, size, 0, stream);
        స్థితి GGML_TYPE_Q6_K:
            తిరిగి reorder_qw_q6_k(దత్తాంశం_device, size, 0, stream);
        స్థితి GGML_TYPE_IQ3_XXS:
            తిరిగి reorder_qw_iq3_xxs(దత్తాంశం_device, size, stream);
        స్థితి GGML_TYPE_IQ3_S:
            తిరిగి reorder_qw_iq3_s(దత్తాంశం_device, size, stream);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }
}

స్థిర తార్కిక should_reorder_tensor(ggml_backend_sycl_context& ctx, స్థిరరాశి ggml_tensor * dst) {
    తిరిగి g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

స్థిర శూన్య opt_for_reorder(ggml_backend_sycl_context * ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    ఒకవేళ (!should_reorder_tensor(*ctx, dst)) {
        తిరిగి;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ఒకవేళ (!extra || extra->optimized_feature.reorder) {
        తిరిగి;  // Skip permutations and already reordered tensors
    }

    ఎంపిక (mm_algorithm) {
        స్థితి mul_mat_algo::DMMV:
            ఒకవేళ (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                తిరిగి;
            }
            ఆపు;
        స్థితి mul_mat_algo::MMVQ:
            ఒకవేళ (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                తిరిగి;
            }
            ఆపు;
        స్థితి mul_mat_algo::MUL_MAT_SYCL:
            ఒకవేళ (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                తిరిగి;
            }
            ఆపు;
    }

    ఒకవేళ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = నిజం;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
స్థిర శూన్య opt_for_reorder_id(ggml_backend_sycl_context * ctx, స్థిరరాశి ggml_tensor * src0) {
    ఒకవేళ (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        తిరిగి;
    }
    ఒకవేళ (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        తిరిగి;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    ఒకవేళ (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->దత్తాంశం % 16 != 0)) {
        తిరిగి;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ఒకవేళ (!extra || extra->optimized_feature.reorder) {
        తిరిగి;
    }
    ఒకవేళ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = నిజం;
    }
}


స్థిర తార్కిక can_use_dequantize_mul_mat_vec(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    స్థిరరాశి int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    తిరిగి ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

స్థిర తార్కిక can_use_mul_mat_vec_q(స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    తిరిగి ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_కొలత;
}

స్థిర శూన్య ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    ఒకవేళ (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        తిరిగి;
    }

    స్థిరరాశి తార్కిక విభజించు = ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు);
    int64_t min_compute_capability = గరిష్ఠ_పూర్ణాంకం;

    ఒకవేళ (విభజించు) {
        ggml_backend_sycl_విభజించు_బఫరు_type_context * buft_ctx =
            (ggml_backend_sycl_విభజించు_బఫరు_type_context *) src0->బఫరు->buft->context;
        స్వయంచాలక & tensor_విభజించు = buft_ctx->tensor_విభజించు;
        కొరకు (పూర్ణాంకం id = 0; id < ggml_sycl_info().device_లెక్క; ++id) {
            // skip devices that are not going to do any work:
            ఒకవేళ (tensor_విభజించు[id] >= (id + 1 < ggml_sycl_info().device_లెక్క ? tensor_విభజించు[id + 1] : 1.0f)) {
                కొనసాగు;
            }

            ఒకవేళ (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } లేకపోతే {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    తార్కిక use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#నిర్వచితమైతే GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#అయితే_ముగింపు

    తార్కిక use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    తార్కిక use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#నిర్వచితమైతే SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_కొలత);
#అయితే_ముగింపు // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    ఒకవేళ (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        తార్కిక use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        ఒకవేళ (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    ఒకవేళ (!విభజించు && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        ఒకవేళ (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } లేకపోతే {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } లేకపోతే ఒకవేళ (!విభజించు && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } లేకపోతే ఒకవేళ (!విభజించు && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } లేకపోతే ఒకవేళ (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } లేకపోతే ఒకవేళ (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        ఒకవేళ (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } లేకపోతే {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } లేకపోతే ఒకవేళ (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } లేకపోతే {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
స్థిర తార్కిక ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * పైకి, స్థిరరాశి ggml_tensor * wu,
                                             స్థిరరాశి ggml_tensor * wg, స్థిరరాశి ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    స్థిరరాశి స్వయంచాలక * extra_u = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wu->extra);
    స్థిరరాశి స్వయంచాలక * extra_g = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wg->extra);
    ఒకవేళ ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        తిరిగి అబద్ధం;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    స్థిరరాశి స్వయంచాలక plain_type = [](ggml_type t) { తిరిగి t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    ఒకవేళ (!plain_type(wu->type) || !plain_type(wg->type)) {
        తిరిగి అబద్ధం;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, పైకి, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    స్థిరరాశి int64_t ne00 = wu->ne[0];
    స్థిరరాశి int64_t ne11 = act->ne[1];

    స్థిరరాశి వరుస_ptr stream = ctx.stream();
    స్థిరరాశి పూర్ణాంకం src1_padded_నిలువుల_సంఖ్య = GGML_PAD((పూర్ణాంకం) ne00, మాతృక_అడ్డువరుస_PADDING);

    ggml_sycl_pool_alloc<అక్షరం> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_నిలువుల_సంఖ్య * పరిమాణం(ఖండం_q8_1) / QK8_1);
    అక్షరం * src1_ddq = src1_q8_alloc.get();

    quantize_అడ్డువరుస_q8_1_sycl<quantize_q8_1>((స్థిరరాశి దశాంశం *) act->దత్తాంశం, src1_ddq, (పూర్ణాంకం) ne00, (పూర్ణాంకం) ne11,
                                          src1_padded_నిలువుల_సంఖ్య, stream);

    తిరిగి ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->దత్తాంశం, wu->దత్తాంశం,
                                             src1_ddq, (దశాంశం *) glu->దత్తాంశం, (పూర్ణాంకం) ne00, (పూర్ణాంకం) wu->ne[1],
                                             (పూర్ణాంకం) ne11,
                                             /*stride_col_y=*/src1_padded_నిలువుల_సంఖ్య / QK8_1,
                                             /*stride_col_dst=*/(పూర్ణాంకం) glu->ne[0], stream);
    GGML_వాడని(gate);
}

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
స్థిర తార్కిక ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * పైకి, స్థిరరాశి ggml_tensor * wu,
                                             స్థిరరాశి ggml_tensor * wg, స్థిరరాశి ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    స్థిరరాశి అక్షరం * act_begin = (స్థిరరాశి అక్షరం *) act->దత్తాంశం;
    స్థిరరాశి అక్షరం * glu_begin = (స్థిరరాశి అక్షరం *) glu->దత్తాంశం;
    ఒకవేళ (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        తిరిగి అబద్ధం;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, పైకి, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    స్థిరరాశి స్వయంచాలక * extra_u = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wu->extra);
    స్థిరరాశి స్వయంచాలక * extra_g = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wg->extra);
    ఒకవేళ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        తిరిగి అబద్ధం;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, పైకి, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    తిరిగి ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->దత్తాంశం, wg->దత్తాంశం,
                                                              (స్థిరరాశి దశాంశం *) act->దత్తాంశం, (దశాంశం *) glu->దత్తాంశం,
                                                              (పూర్ణాంకం) wu->ne[0], (పూర్ణాంకం) wu->ne[1], ctx.stream());
}
#అయితే_ముగింపు

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
స్థిర తార్కిక ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    ఒకవేళ (!ggml_sycl_can_fuse(cgraph, కణుపు_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        తిరిగి అబద్ధం;
    }

    ggml_tensor *       glu  = cgraph->nodes[కణుపు_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       పైకి   = glu->src[1];
    స్థిరరాశి ggml_tensor * wu   = పైకి->src[0];
    స్థిరరాశి ggml_tensor * wg   = gate->src[0];
    స్థిరరాశి ggml_tensor * act  = పైకి->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    ఒకవేళ (ggml_backend_బఫరు_is_sycl_విభజించు(wu->బఫరు) || ggml_backend_బఫరు_is_sycl_విభజించు(wg->బఫరు)) {
        తిరిగి అబద్ధం;
    }

#నిర్వచితమైతే GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    ఒకవేళ (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, పైకి, wu, wg, act)) {
        తిరిగి నిజం;
    }
#అయితే_ముగింపు

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    ఒకవేళ (g_ggml_sycl_prioritize_dmmv) {
        తిరిగి అబద్ధం;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    స్థిరరాశి తార్కిక reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    ఒకవేళ (!reorder_pair) {
        తిరిగి ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, పైకి, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    ఒకవేళ (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        తిరిగి అబద్ధం;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, పైకి, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    స్థిరరాశి స్వయంచాలక * extra_u = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wu->extra);
    స్థిరరాశి స్వయంచాలక * extra_g = static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(wg->extra);
    ఒకవేళ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        తిరిగి అబద్ధం;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, పైకి, /*num_src=*/2, " : fused with gate + GLU");

    స్థిరరాశి int64_t ne00 = wu->ne[0];
    స్థిరరాశి int64_t ne11 = act->ne[1];

    స్థిరరాశి వరుస_ptr stream           = ctx.stream();
    స్థిరరాశి పూర్ణాంకం       src1_padded_నిలువుల_సంఖ్య = GGML_PAD((పూర్ణాంకం) ne00, మాతృక_అడ్డువరుస_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<అక్షరం> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_నిలువుల_సంఖ్య * పరిమాణం(ఖండం_q8_1) / QK8_1);
    అక్షరం *                     src1_ddq = src1_q8_alloc.get();

    quantize_అడ్డువరుస_q8_1_sycl<quantize_and_reorder_q8_1_soa>((స్థిరరాశి దశాంశం *) act->దత్తాంశం, src1_ddq, (పూర్ణాంకం) ne00, (పూర్ణాంకం) ne11,
                                                          src1_padded_నిలువుల_సంఖ్య, stream);

    తిరిగి ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->దత్తాంశం, wg->దత్తాంశం, src1_ddq,
                                               (దశాంశం *) glu->దత్తాంశం, (పూర్ణాంకం) ne00, (పూర్ణాంకం) wu->ne[1], (పూర్ణాంకం) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_నిలువుల_సంఖ్య * (పూర్ణాంకం) పరిమాణం(ఖండం_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(పూర్ణాంకం) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
స్థిర పూర్ణాంకం ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx) {
    స్థిరరాశి ggml_tensor * కణుపు = cgraph->nodes[కణుపు_idx];
    ఒకవేళ (ggml_sycl_info().device_లెక్క != 1 || కణుపు->type != GGML_TYPE_F32 ||
        కణుపు->src[0]->type != GGML_TYPE_F32 || కణుపు->src[0]->ne[0] >= 1024) {
        తిరిగి 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    పూర్ణాంకం           count = 0;
    పూర్ణాంకం           చివరి  = కణుపు_idx;
    దశాంశం         eps0;
    జ్ఞాపకం_నకలు(&eps0, కణుపు->op_params, పరిమాణం(దశాంశం));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    స్వయంచాలక overlaps = [](స్థిరరాశి ggml_tensor * a, స్థిరరాశి ggml_tensor * b) {
        స్థిరరాశి అక్షరం * ab = (స్థిరరాశి అక్షరం *) a->దత్తాంశం;
        స్థిరరాశి అక్షరం * bb = (స్థిరరాశి అక్షరం *) b->దత్తాంశం;
        తిరిగి ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    కొరకు (పూర్ణాంకం j = కణుపు_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        ఒకవేళ (ggml_is_ఖాళీలు(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_క్రమమార్పు || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;  // not a launch; cannot break a run of adjacent norms
        }
        ఒకవేళ (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, కణుపు) ||
            !ggml_are_same_shape(nj->src[0], కణుపు->src[0])) {
            ఆపు;  // any other launch ends the run
        }
        తార్కిక same_nb = నిజం;
        కొరకు (పూర్ణాంకం d = 0; d < GGML_MAX_DIMS; ++d) {
            ఒకవేళ (nj->nb[d] != కణుపు->nb[d] || nj->src[0]->nb[d] != కణుపు->src[0]->nb[d]) {
                same_nb = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (!same_nb) {
            ఆపు;  // one nb[] stride set is shared by the whole batch
        }
        దశాంశం epsj;
        జ్ఞాపకం_నకలు(&epsj, nj->op_params, పరిమాణం(దశాంశం));
        ఒకవేళ (epsj != eps0) {
            ఆపు;  // eps mismatch ends the run
        }
        తార్కిక indep = నిజం;
        కొరకు (పూర్ణాంకం k = 0; k < count; ++k) {
            ఒకవేళ (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = అబద్ధం;
                ఆపు;
            }
        }
        ఒకవేళ (!indep) {
            ఆపు;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        చివరి           = j;
    }
    ఒకవేళ (count < 2) {
        తిరిగి 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    తిరిగి చివరి - కణుపు_idx;
}


__dpct_inline__ స్థిర శూన్య k_copy_src1_గమ్యం_contiguous(
    స్థిరరాశి అక్షరం *__restrict__ src1_original, అక్షరం *__restrict__ src1_contiguous,
    స్థిరరాశి mmid_అడ్డువరుస_mapping *__restrict__ అడ్డువరుస_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి int32_t src1_అడ్డువరుస = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి int32_t iid1 = అడ్డువరుస_mapping[src1_అడ్డువరుస].i2;
    స్థిరరాశి int32_t id   = అడ్డువరుస_mapping[src1_అడ్డువరుస].i1;

    స్థిరరాశి int64_t i11 = id % ne11;
    స్థిరరాశి int64_t i12 = iid1;

    స్థిరరాశి దశాంశం * src1_అడ్డువరుస_original = (స్థిరరాశి దశాంశం *)(src1_original + i11*nb11 + i12*nb12);
    దశాంశం * src1_అడ్డువరుస_contiguous = (దశాంశం *)(src1_contiguous + src1_అడ్డువరుస*nb11);

#సూచన unroll
    కొరకు (పూర్ణాంకం i = అంశం_ct1.get_local_id(2); i < ne10;
         i += అంశం_ct1.get_local_range(2)) {
        src1_అడ్డువరుస_contiguous[i] = src1_అడ్డువరుస_original[i];
    }
}

__dpct_inline__ స్థిర శూన్య k_copy_dst_from_contiguous(
    అక్షరం *__restrict__ dst_original, స్థిరరాశి అక్షరం *__restrict__ dst_contiguous,
    స్థిరరాశి mmid_అడ్డువరుస_mapping *__restrict__ అడ్డువరుస_mapping, int64_t ne0, size_t nb1,
    size_t nb2, స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    int32_t i = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి int32_t i1 = అడ్డువరుస_mapping[i].i1;
    స్థిరరాశి int32_t i2 = అడ్డువరుస_mapping[i].i2;

    స్థిరరాశి దశాంశం * dst_అడ్డువరుస_contiguous = (స్థిరరాశి దశాంశం *)(dst_contiguous + i*nb1);
    దశాంశం * dst_అడ్డువరుస_original = (దశాంశం *)(dst_original + i1*nb1 + i2*nb2);

#సూచన unroll
    కొరకు (పూర్ణాంకం j = అంశం_ct1.get_local_id(2); j < ne0;
         j += అంశం_ct1.get_local_range(2)) {
        dst_అడ్డువరుస_original[j] = dst_అడ్డువరుస_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
స్థిర తార్కిక ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0,
    స్థిరరాశి ggml_tensor * src1, స్థిరరాశి ggml_tensor * ids, ggml_tensor * dst)
{
    స్థిరరాశి int64_t ne10 = src1->ne[0];
    స్థిరరాశి int64_t ne11 = src1->ne[1];
    స్థిరరాశి int64_t ne12 = src1->ne[2];
    ఒకవేళ (ne12 != 1) తిరిగి అబద్ధం;
    ఒకవేళ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) తిరిగి అబద్ధం;
    ఒకవేళ (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) తిరిగి అబద్ధం;
    ఒకవేళ (!ggml_is_contiguous(src1)) తిరిగి అబద్ధం;

    స్థిరరాశి int64_t n_ids_చొప్పున_సమూహం = ids->ne[0];
    ఒకవేళ (ids->ne[1] != 1) తిరిగి అబద్ధం;
    ఒకవేళ (ne11 != 1 && ne11 != n_ids_చొప్పున_సమూహం) తిరిగి అబద్ధం;

    స్థిరరాశి వరుస_ptr stream           = ctx.stream();
    స్థిరరాశి పూర్ణాంకం       src1_padded_నిలువుల_సంఖ్య = GGML_PAD((పూర్ణాంకం) ne10, మాతృక_అడ్డువరుస_PADDING);
    స్థిరరాశి పూర్ణాంకం       n_experts_వాడిన   = (పూర్ణాంకం) n_ids_చొప్పున_సమూహం;
    స్థిరరాశి పూర్ణాంకం       nrows            = (పూర్ణాంకం) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    స్థిరరాశి ggml_tensor_extra_gpu * src0_extra =
        static_cast<స్థిరరాశి ggml_tensor_extra_gpu *>(src0->extra);
    స్థిరరాశి తార్కిక use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<అక్షరం> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_నిలువుల_సంఖ్య * పరిమాణం(ఖండం_q8_1) / QK8_1);
    అక్షరం * src1_ddq = src1_q8_alloc.get();
    ఒకవేళ (use_reorder) {
        quantize_అడ్డువరుస_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (స్థిరరాశి దశాంశం *) src1->దత్తాంశం, src1_ddq, (పూర్ణాంకం) ne10, (పూర్ణాంకం) ne11,
            src1_padded_నిలువుల_సంఖ్య, stream);
    } లేకపోతే {
        quantize_అడ్డువరుస_q8_1_sycl<quantize_q8_1>(
            (స్థిరరాశి దశాంశం *) src1->దత్తాంశం, src1_ddq, (పూర్ణాంకం) ne10, (పూర్ణాంకం) ne11,
            src1_padded_నిలువుల_సంఖ్య, stream);
    }

    స్థిరరాశి size_t బైట్లు_చొప్పున_qrow = (size_t) src1_padded_నిలువుల_సంఖ్య * పరిమాణం(ఖండం_q8_1) / QK8_1;
    స్థిరరాశి size_t src1_అడ్డువరుస_stride = (ne11 == 1) ? 0 : బైట్లు_చొప్పున_qrow;

    ఒకవేళ (use_reorder) {
        తిరిగి ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->దత్తాంశం, src1_ddq, (స్థిరరాశి int32_t *) ids->దత్తాంశం,
            (దశాంశం *) dst->దత్తాంశం, (పూర్ణాంకం) ne10, nrows, n_experts_వాడిన,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_అడ్డువరుస_stride, stream);
    }
    తిరిగి ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->దత్తాంశం, src1_ddq, (స్థిరరాశి int32_t *) ids->దత్తాంశం,
        (దశాంశం *) dst->దత్తాంశం, (పూర్ణాంకం) ne10, nrows, n_experts_వాడిన,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_అడ్డువరుస_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
స్థిర శూన్య mmid_counting_క్రమీకరించు_అడ్డువరుసలు(
        స్థిరరాశి ggml_tensor * ids, స్థిరరాశి అక్షరం * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_అడ్డువరుసలు,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_అడ్డువరుస_offsets,
        std::vector<mmid_అడ్డువరుస_mapping> & routed_అడ్డువరుస_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.కేటాయించు(n_as, 0);
    కొరకు (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        కొరకు (int64_t id = 0; id < n_ids; id++) {
            స్థిరరాశి int32_t అడ్డువరుస_id_i = *(స్థిరరాశి int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(అడ్డువరుస_id_i >= 0 && అడ్డువరుస_id_i < n_as);
            expert_counts[అడ్డువరుస_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_అడ్డువరుస_offsets.కేటాయించు(n_as + 1, 0);
    కొరకు (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_అడ్డువరుస_offsets[i02 + 1] = expert_అడ్డువరుస_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_అడ్డువరుస_తదుపరి = expert_అడ్డువరుస_offsets;
    routed_అడ్డువరుస_src.resize(n_routed_అడ్డువరుసలు);
    కొరకు (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        కొరకు (int64_t id = 0; id < n_ids; id++) {
            స్థిరరాశి int32_t అడ్డువరుస_id_i = *(స్థిరరాశి int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(అడ్డువరుస_id_i >= 0 && అడ్డువరుస_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            స్థిరరాశి int64_t routed_అడ్డువరుస = expert_అడ్డువరుస_తదుపరి[అడ్డువరుస_id_i]++;
            GGML_ASSERT(routed_అడ్డువరుస >= expert_అడ్డువరుస_offsets[అడ్డువరుస_id_i]);
            GGML_ASSERT(routed_అడ్డువరుస < expert_అడ్డువరుస_offsets[అడ్డువరుస_id_i + 1]);
            routed_అడ్డువరుస_src[routed_అడ్డువరుస] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

స్థిర శూన్య ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) ప్రయత్నించు {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    స్థిరరాశి ggml_tensor *src0 = dst->src[0];
    స్థిరరాశి ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_బఫరు_is_sycl_విభజించు(src0->బఫరు) && "mul_mat_id does not support split buffers");

    స్థిరరాశి ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి వరుస_ptr stream = ctx.stream();

    స్థిరరాశి int64_t n_as = ne02;
    స్థిరరాశి int64_t n_ids = ids->ne[0];

    ఒకవేళ (ne12 == 1) {
        ఒకవేళ (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            తిరిగి;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    ఒకవేళ (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<అక్షరం> ids_host(ggml_nbytes(ids));
    స్థిరరాశి అక్షరం * ids_dev = (స్థిరరాశి అక్షరం *) ids->దత్తాంశం;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->జ్ఞాపకం_నకలు(ids_host.దత్తాంశం(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_అడ్డువరుస = *src0;
    ggml_tensor src1_అడ్డువరుస = *src1;
    ggml_tensor dst_అడ్డువరుస = *dst;

    అక్షరం *src0_original = (అక్షరం *)src0->దత్తాంశం;
    అక్షరం *src1_original = (అక్షరం *)src1->దత్తాంశం;
    అక్షరం *dst_original = (అక్షరం *)dst->దత్తాంశం;

    src0_అడ్డువరుస.ne[2] = 1;
    src0_అడ్డువరుస.ne[3] = 1;
    src0_అడ్డువరుస.nb[3] = nb02;

    src1_అడ్డువరుస.ne[1] = 1;
    src1_అడ్డువరుస.ne[2] = 1;
    src1_అడ్డువరుస.ne[3] = 1;
    src1_అడ్డువరుస.nb[2] = nb11;
    src1_అడ్డువరుస.nb[3] = nb11;

    dst_అడ్డువరుస.ne[1] = 1;
    dst_అడ్డువరుస.ne[2] = 1;
    dst_అడ్డువరుస.ne[3] = 1;
    dst_అడ్డువరుస.nb[2] = nb1;
    dst_అడ్డువరుస.nb[3] = nb1;
    ఒకవేళ (ne12 == 1) {
        కొరకు (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            కొరకు (int64_t id = 0; id < n_ids; id++) {
                స్థిరరాశి int32_t i02 = *(స్థిరరాశి int32_t *) (ids_host.దత్తాంశం() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                స్థిరరాశి int64_t i11 = id % ne11;
                స్థిరరాశి int64_t i12 = iid1;

                స్థిరరాశి int64_t i1 = id;
                స్థిరరాశి int64_t i2 = i12;

            src0_అడ్డువరుస.దత్తాంశం = src0_original + i02*nb02;
            src1_అడ్డువరుస.దత్తాంశం = src1_original + i11*nb11 + i12*nb12;
            dst_అడ్డువరుస.దత్తాంశం = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_అడ్డువరుస, &src1_అడ్డువరుస, &dst_అడ్డువరుస);
            }
        }
    } లేకపోతే {
        స్థిరరాశి int64_t n_routed_అడ్డువరుసలు = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<అక్షరం> src1_contiguous(ctx.pool(), పరిమాణం(దశాంశం)*n_routed_అడ్డువరుసలు*ne10);
        ggml_sycl_pool_alloc<అక్షరం>  dst_contiguous(ctx.pool(), పరిమాణం(దశాంశం)*n_routed_అడ్డువరుసలు*ne0);

        src1_అడ్డువరుస.దత్తాంశం = src1_contiguous.get();
        dst_అడ్డువరుస.దత్తాంశం  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_అడ్డువరుస_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_అడ్డువరుస_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_అడ్డువరుస_mapping> & routed_అడ్డువరుస_src = ctx.mmid_అడ్డువరుస_mapping_host;

        mmid_counting_క్రమీకరించు_అడ్డువరుసలు(ids, ids_host.దత్తాంశం(), n_ids, n_as, n_routed_అడ్డువరుసలు,
                                expert_అడ్డువరుస_counts, expert_అడ్డువరుస_offsets, routed_అడ్డువరుస_src);

        ggml_sycl_pool_alloc<mmid_అడ్డువరుస_mapping> dev_అడ్డువరుస_mapping(ctx.pool(), n_routed_అడ్డువరుసలు);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->జ్ఞాపకం_నకలు(dev_అడ్డువరుస_mapping.get(), routed_అడ్డువరుస_src.దత్తాంశం(), n_routed_అడ్డువరుసలు*పరిమాణం(mmid_అడ్డువరుస_mapping))));

        స్థిరరాశి గుర్తులేని పూర్ణాంకం max_work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[ctx.device];
        ప్రతిజ్ఞా(max_work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);

        {
            sycl::range<3> ఖండం_dims(1, 1, std::min((గుర్తులేని పూర్ణాంకం)ne10, max_work_సమూహం_కొలత));
            sycl::range<3> grid_dims(1, 1, n_routed_అడ్డువరుసలు);
            stream->submit([&](sycl::handler &cgh) {
                అక్షరం *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_అడ్డువరుస_mapping *__restrict dev_అడ్డువరుస_mapping_get =
                    dev_అడ్డువరుస_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                    [=](sycl::nd_అంశం<3> అంశం_ct1) {
                        k_copy_src1_గమ్యం_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_అడ్డువరుస_mapping_get,
                            ne11, ne10, nb11, nb12,
                            అంశం_ct1);
                    });
            });
        }

        తార్కిక grouped = అబద్ధం;
        ఒకవేళ (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == పరిమాణం(దశాంశం)*ne10 && nb1 == పరిమాణం(దశాంశం)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            ఒకవేళ (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            స్థిరరాశి తార్కిక src0_reordered =
                src0->extra && ((స్థిరరాశి ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (స్థిరరాశి దశాంశం *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (దశాంశం *) dst_contiguous.get(),
                                                     expert_అడ్డువరుస_offsets.దత్తాంశం(), n_as, ne01, ne10, n_routed_అడ్డువరుసలు,
                                                     ctx.mmid_tile_కాలపట్టిక_host, ctx.pool(), stream);
        }

        కొరకు (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            స్థిరరాశి int64_t అంకె_src1_అడ్డువరుసలు = expert_అడ్డువరుస_counts[i02];

            ఒకవేళ (అంకె_src1_అడ్డువరుసలు == 0) {
                కొనసాగు;
            }

            స్థిరరాశి int64_t expert_అడ్డువరుస_స్థానభ్రంశం = expert_అడ్డువరుస_offsets[i02];

            src0_అడ్డువరుస.దత్తాంశం = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == పరిమాణం(దశాంశం)*ne10);
            GGML_ASSERT(nb1 == పరిమాణం(దశాంశం)*ne0);
            src1_అడ్డువరుస.దత్తాంశం = src1_contiguous.get() + expert_అడ్డువరుస_స్థానభ్రంశం*nb11;
            src1_అడ్డువరుస.ne[1] = అంకె_src1_అడ్డువరుసలు;

            src1_అడ్డువరుస.nb[1] = nb11;
            src1_అడ్డువరుస.nb[2] = అంకె_src1_అడ్డువరుసలు*nb11;
            src1_అడ్డువరుస.nb[3] = అంకె_src1_అడ్డువరుసలు*nb11;

            dst_అడ్డువరుస.దత్తాంశం = dst_contiguous.get() + expert_అడ్డువరుస_స్థానభ్రంశం*nb1;
            dst_అడ్డువరుస.ne[1] = అంకె_src1_అడ్డువరుసలు;
            dst_అడ్డువరుస.nb[1] = nb1;
            dst_అడ్డువరుస.nb[2] = అంకె_src1_అడ్డువరుసలు*nb1;
            dst_అడ్డువరుస.nb[3] = అంకె_src1_అడ్డువరుసలు*nb1;

            ggml_sycl_mul_mat(ctx, &src0_అడ్డువరుస, &src1_అడ్డువరుస, &dst_అడ్డువరుస);
        }

        {
            sycl::range<3> ఖండం_dims(1, 1, std::min((గుర్తులేని పూర్ణాంకం)ne0, max_work_సమూహం_కొలత));
            sycl::range<3> grid_dims(1, 1, n_routed_అడ్డువరుసలు);
            stream->submit([&](sycl::handler &cgh) {
                స్థిరరాశి అక్షరం *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                స్థిరరాశి mmid_అడ్డువరుస_mapping *__restrict dev_అడ్డువరుస_mapping_get =
                    dev_అడ్డువరుస_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                    [=](sycl::nd_అంశం<3> అంశం_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_అడ్డువరుస_mapping_get,
                                                   ne0, nb1, nb2, అంశం_ct1);
                    });
            });
        }
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

స్థిర శూన్య ggml_sycl_diag_ముసుగు_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_ముసుగు_inf(ctx, dst);
}

స్థిర శూన్య ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

స్థిర శూన్య ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

స్థిర శూన్య ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

స్థిర శూన్య ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

స్థిర శూన్య ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

స్థిర శూన్య ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

స్థిర శూన్య ggml_sycl_కూడిక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_కూడిక(ctx, dst);
}

స్థిర శూన్య ggml_sycl_కూడిక_అడ్డువరుసలు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_కూడిక_అడ్డువరుసలు(ctx, dst);
}

స్థిర శూన్య ggml_sycl_సగటు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_సగటు(ctx, dst);
}

స్థిర శూన్య ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

స్థిర శూన్య ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


స్థిర శూన్య ggml_sycl_సమితి_main_device(స్థిరరాశి పూర్ణాంకం main_device) ప్రయత్నించు {
    ఒకవేళ (dpct::get_current_device_id() == static_cast<గుర్తులేని పూర్ణాంకం> (main_device)) {
        తిరిగి;
    }
    check_allow_gpu_సూచీ(main_device);
    dpct::select_device(main_device);

    ఒకవేళ (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర తార్కిక ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, నిర్మాణం ggml_tensor * dst) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    ఒకవేళ (!g_sycl_loaded) తిరిగి అబద్ధం;

    ఒకవేళ (dst->src[0] != శూన్యసూచిక && ggml_backend_బఫరు_is_sycl_విభజించు(dst->src[0]->బఫరు)) {
        ggml_sycl_సమితి_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    ఎంపిక (dst->op) {
        స్థితి GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_REPEAT_వెనుక:
            ggml_sycl_repeat_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            ggml_sycl_get_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
            ggml_sycl_get_అడ్డువరుసలు_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమితి:
            ggml_sycl_op_సమితి(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            ggml_sycl_op_సమితి_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_కూడు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడు_ID:
            ggml_sycl_కూడు_id(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_లెక్క_EQUAL:
            ggml_sycl_లెక్క_equal(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(dst)) {
                స్థితి GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_అడుగు:
                    ggml_sycl_అడుగు(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                    ggml_sycl_gelu_శీఘ్ర(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_విడత:
                    ggml_sycl_విడత(ctx, dst);
                    ఆపు;
                స్థితి GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(dst)) {
                స్థితి GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    ఆపు;
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                    ggml_sycl_geglu_శీఘ్ర(ctx, dst);
                    ఆపు;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సమూహం_NORM:
            ggml_sycl_సమూహం_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            ఆపు;
        స్థితి GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_RMS_NORM_వెనుక:
            ggml_sycl_rms_norm_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            ఒకవేళ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                తిరిగి అబద్ధం;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            ఒకవేళ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                తిరిగి అబద్ధం;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_బహిర్గతం_PROD:
            ggml_sycl_op_బహిర్గతం_prod(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            ఆపు;
        స్థితి GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            ఆపు;
        స్థితి GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
            ggml_sycl_diag_ముసుగు_inf(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX_వెనుక:
            ggml_sycl_op_soft_max_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
            ggml_sycl_cross_entropy_loss_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ROPE_వెనుక:
            ggml_sycl_rope_వెనుక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
            ggml_sycl_im2col(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు_3_D:
            ggml_sycl_im2col_3d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడిక:
            ggml_sycl_కూడిక(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            ggml_sycl_కూడిక_అడ్డువరుసలు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_సగటు:
            ggml_sycl_సగటు(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_శిఖరం_K:
            ggml_sycl_op_శిఖరం_k(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
            ggml_sycl_op_gated_రేఖీయ_attn(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_OPT_అడుగు_ADAMW:
            ggml_sycl_opt_అడుగు_adamw(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_OPT_అడుగు_SGD:
            ggml_sycl_opt_అడుగు_sgd(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SSM_పరిశీలన:
            ggml_sycl_ssm_పరిశీలన(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            ఆపు;
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    తిరిగి నిజం;
} పట్టుకో (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

GGML_API శూన్య ggml_backend_sycl_get_device_description(పూర్ణాంకం device, అక్షరం *description,
                                      size_t description_కొలత) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_కొలత, "%s", prop.get_name());
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర తార్కిక sycl_get_mem_info(పూర్ణాంకం device, size_t * విడుదల, size_t * మొత్తం) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    జ్ఞాపకం_APIType mem_api_type = జ్ఞాపకం_APIType::జ్ఞాపకం_API_TYPE_SYCL;

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_ఫలితం_SUCCESS ?
        (జ్ఞాపకం_APIType) g_ggml_sycl_get_mem_api : జ్ఞాపకం_APIType::జ్ఞాపకం_API_TYPE_SYCL;
#లేకపోతే
    mem_api_type = జ్ఞాపకం_APIType::జ్ఞాపకం_API_TYPE_SYCL;
#అయితే_ముగింపు
    తార్కిక res = get_జ్ఞాపకం_కొలత(dpct::dev_mgr::instance().get_device(device),
        *విడుదల, *మొత్తం, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *మొత్తం, *విడుదల);
    తిరిగి res;
}

శూన్య ggml_backend_sycl_get_device_జ్ఞాపకం(పూర్ణాంకం device, size_t * విడుదల, size_t * మొత్తం) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ఒకవేళ (!sycl_get_mem_info(device, విడుదల, మొత్తం)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *విడుదల, *మొత్తం);
} పట్టుకో (స్థిరరాశి sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    తిరిగి sycl_ctx->name.c_str();
}

స్థిర శూన్య ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    తొలగించు sycl_ctx;
    తొలగించు backend;
}

స్థిర శూన్య ggml_backend_sycl_సమితి_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               స్థిరరాశి శూన్య *దత్తాంశం, size_t స్థానభ్రంశం,
                                               size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_బఫరు_type(sycl_ctx->device) && "unsupported buffer type");
    స్థిరరాశి వరుస_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->జ్ఞాపకం_నకలు((అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, దత్తాంశం, size)));
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               స్థిరరాశి ggml_tensor *tensor,
                                               శూన్య *దత్తాంశం, size_t స్థానభ్రంశం,
                                               size_t size) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, స్థానభ్రంశం);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_బఫరు_t buf = tensor->view_src ? tensor->view_src->బఫరు : tensor->బఫరు;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_బఫరు_type(sycl_ctx->device) && "unsupported buffer type");
    స్థిరరాశి వరుస_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->జ్ఞాపకం_నకలు(
        దత్తాంశం, (స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + స్థానభ్రంశం, size)));
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర తార్కిక ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               స్థిరరాశి ggml_tensor *src,
                                               ggml_tensor *dst) ప్రయత్నించు {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    తార్కిక is_cpy_supported                = dst->బఫరు->buft == ggml_backend_sycl_బఫరు_type(sycl_ctx->device) &&
                            ggml_backend_బఫరు_is_sycl(src->బఫరు);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ఒకవేళ (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        స్థిరరాశి వరుస_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->జ్ఞాపకం_నకలు(
            dst->దత్తాంశం, src->దత్తాంశం, ggml_nbytes(dst))));
        తిరిగి నిజం;
    }

    తిరిగి అబద్ధం;
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_synchronize(ggml_backend_t backend) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    స్థిరరాశి వరుస_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_వాడని(backend);
}
పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర తార్కిక ggml_sycl_is_view_or_noop(స్థిరరాశి ggml_tensor * t) {
    తిరిగి ggml_is_ఖాళీలు(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_క్రమమార్పు || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
స్థిర పూర్ణాంకం ggml_sycl_try_gdn_cache_fusion(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    ఒకవేళ (!g_ggml_sycl_enable_fusion) {
        తిరిగి 0;
    }

    స్థిరరాశి ggml_tensor * gdn = cgraph->nodes[కణుపు_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    ఒకవేళ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        తిరిగి 0;
    }

    స్థిరరాశి ggml_tensor * src_v     = gdn->src[2];
    స్థిరరాశి int64_t       S_v       = src_v->ne[0];
    స్థిరరాశి int64_t       H         = src_v->ne[1];
    స్థిరరాశి int64_t       n_టోకెన్లు  = src_v->ne[2];
    స్థిరరాశి int64_t       n_seqs    = src_v->ne[3];
    స్థిరరాశి int64_t       D         = S_v * S_v * H;
    స్థిరరాశి int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    స్థిరరాశి int64_t       n_written = std::min<int64_t>(n_టోకెన్లు, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    స్థిరరాశి size_t తోక_off = ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, S_v * H * n_టోకెన్లు * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    స్థిరరాశి ggml_tensor * cpy  = శూన్యసూచిక;
    పూర్ణాంకం                 skip = 0;
    కొరకు (పూర్ణాంకం j = కణుపు_idx + 1; j < cgraph->n_nodes && cpy == శూన్యసూచిక; ++j) {
        స్థిరరాశి ggml_tensor * n = cgraph->nodes[j];
        ఒకవేళ (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }
        ఒకవేళ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            తిరిగి 0;
        }
        cpy  = n;
        skip = j - కణుపు_idx;
    }
    ఒకవేళ (cpy == శూన్యసూచిక) {
        తిరిగి 0;
    }

    స్థిరరాశి ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    స్థిరరాశి ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ఒకవేళ (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != తోక_off ||
        !ggml_is_contiguous(src)) {
        తిరిగి 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    స్థిరరాశి std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ఒకవేళ (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->దత్తాంశం == శూన్యసూచిక ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_కొలత(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_అడ్డువరుస_కొలత(GGML_TYPE_F32, D)) {
        తిరిగి 0;
    }

    fused_state_cpy.దత్తాంశం        = (దశాంశం *) dst->దత్తాంశం; // rollback group 0 (newest)
    fused_state_cpy.స్థానం_stride = K > 1 ? (int64_t) (dst->nb[2] / పరిమాణం(దశాంశం)) : 0;
    తిరిగి skip;
}

స్థిర శూన్య ggml_backend_sycl_లేఖాచిత్రం_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_సమితి_main_device(sycl_ctx->device);

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];
        ఒకవేళ (ggml_sycl_is_view_or_noop(కణుపు)) {
            కొనసాగు;
        }
        ఒకవేళ ((కణుపు->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            కొనసాగు;
        }

        స్థిరరాశి పూర్ణాంకం nodes_గమ్యం_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        ఒకవేళ (nodes_గమ్యం_skip != 0) {
            i += nodes_గమ్యం_skip;
            కొనసాగు;
        }
#నిర్వచితం_కాకపోతే NDEBUG
        ప్రతిజ్ఞా(కణుపు->బఫరు->buft == ggml_backend_sycl_బఫరు_type(sycl_ctx->device));
        కొరకు (పూర్ణాంకం j = 0; j < GGML_MAX_SRC; j++) {
            ఒకవేళ (కణుపు->src[j] != శూన్యసూచిక) {
                ప్రతిజ్ఞా(కణుపు->src[j]->బఫరు->buft == ggml_backend_sycl_బఫరు_type(sycl_ctx->device));
            }
        }
#అయితే_ముగింపు
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        ఒకవేళ (కణుపు->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            స్థిరరాశి పూర్ణాంకం gdn_nodes_గమ్యం_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            ఒకవేళ (gdn_nodes_గమ్యం_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, కణుపు, fused_state_cpy);
                i += gdn_nodes_గమ్యం_skip;
                కొనసాగు;
            }
        }
        ఒకవేళ (కణుపు->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_కూడు }, {})) {
            ggml_sycl_op_rms_norm_fused_కూడు(*sycl_ctx, కణుపు, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            కొనసాగు;
        }
        ఒకవేళ (కణుపు->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;
            కొనసాగు;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        ఒకవేళ (కణుపు->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;
            కొనసాగు;
        }
        ఒకవేళ (కణుపు->op == GGML_OP_కూడు &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_కూడు, GGML_OP_కూడు }, {})) {
            ggml_sycl_op_కూడు_కూడు_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;
            కొనసాగు;
        }
        ఒకవేళ (కణుపు->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(కణుపు) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;
            కొనసాగు;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        ఒకవేళ (కణుపు->op == GGML_OP_కూడు && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_కూడు, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_కూడు_unary_mul_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            కొనసాగు;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        ఒకవేళ (కణుపు->op == GGML_OP_L2_NORM) {
            స్థిరరాశి పూర్ణాంకం l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            ఒకవేళ (l2_batch_skip > 0) {
                i += l2_batch_skip;
                కొనసాగు;
            }
        }

        ఒకవేళ (కణుపు->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_కూడు, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, కణుపు, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            కొనసాగు;
        }

        ఒకవేళ (కణుపు->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, కణుపు, శూన్యసూచిక, cgraph->nodes[i + 1]);
            i++;
            కొనసాగు;
        }

        ఒకవేళ (కణుపు->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            కొనసాగు;
        }

        తార్కిక ok = ggml_sycl_compute_forward(*sycl_ctx, కణుపు);
        ఒకవేళ (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, కణుపు->name, ggml_op_name(కణుపు->op));
        }
        GGML_ASSERT(ok);
    }
}

#నిర్వచితమైతే GGML_SYCL_లేఖాచిత్రం
స్థిర తార్కిక check_లేఖాచిత్రం_compatibility(ggml_cgraph * cgraph) {
    ఒకవేళ (ggml_sycl_info().device_లెక్క > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        తిరిగి అబద్ధం;
    }

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        స్థిరరాశి ggml_op కణుపు_op = cgraph->nodes[i]->op;
        ఎంపిక (కణుపు_op) {
            అప్రమేయ:
                ఆపు;
            స్థితి GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            స్థితి GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(కణుపు_op));
                తిరిగి అబద్ధం;
            స్థితి GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                ఒకవేళ (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(కణుపు_op));
                    తిరిగి అబద్ధం;
                }
        }
    }
    తిరిగి నిజం;
}
#అయితే_ముగింపు

స్థిర ggml_స్థితి ggml_backend_sycl_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    స్వయంచాలక * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#నిర్వచితమైతే GGML_SYCL_లేఖాచిత్రం
    తార్కిక use_sycl_లేఖాచిత్రం = అబద్ధం;
    ఒకవేళ (g_ggml_sycl_enable_లేఖాచిత్రం) {
        use_sycl_లేఖాచిత్రం = check_లేఖాచిత్రం_compatibility(cgraph);
    }
    ఒకవేళ (use_sycl_లేఖాచిత్రం) {
        స్థిరరాశి తార్కిక లేఖాచిత్రం_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_లేఖాచిత్రం);
        ఒకవేళ (!లేఖాచిత్రం_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_లేఖాచిత్రం_compute_impl(sycl_ctx, cgraph);
            తిరిగి GGML_స్థితి_SUCCESS;
        }

        sycl_ex::command_లేఖాచిత్రం model_sycl_లేఖాచిత్రం(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_బఫరు_outlives_లేఖాచిత్రం{}});

        model_sycl_లేఖాచిత్రం.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_లేఖాచిత్రం_compute_impl(sycl_ctx, cgraph);
        model_sycl_లేఖాచిత్రం.చివర_recording();

        స్థిరరాశి తార్కిక లేఖాచిత్రం_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_లేఖాచిత్రం);
        ఒకవేళ (!sycl_ctx->exec_లేఖాచిత్రం || !లేఖాచిత్రం_update_support) {
            స్వయంచాలక exec_లేఖాచిత్రం = లేఖాచిత్రం_update_support ? model_sycl_లేఖాచిత్రం.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_లేఖాచిత్రం.finalize();
            sycl_ctx->exec_లేఖాచిత్రం = std::make_unique<
                sycl_ex::command_లేఖాచిత్రం<sycl_ex::లేఖాచిత్రం_state::executable>>(exec_లేఖాచిత్రం);
        } లేకపోతే {
            ప్రయత్నించు {
                sycl_ctx->exec_లేఖాచిత్రం->update(model_sycl_లేఖాచిత్రం);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } పట్టుకో (sycl::exception స్థిరరాశి & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                స్వయంచాలక exec_లేఖాచిత్రం = model_sycl_లేఖాచిత్రం.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_లేఖాచిత్రం = std::make_unique<
                    sycl_ex::command_లేఖాచిత్రం<sycl_ex::లేఖాచిత్రం_state::executable>>(exec_లేఖాచిత్రం);
            }
        }

        sycl_ctx->stream()->ext_oneapi_లేఖాచిత్రం(*(sycl_ctx->exec_లేఖాచిత్రం));
    } లేకపోతే
#అయితే_ముగింపు
    {
        ggml_backend_sycl_లేఖాచిత్రం_compute_impl(sycl_ctx, cgraph);
    }
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_sycl_event_రికార్డు(ggml_backend_t backend, ggml_backend_event_t event)
ప్రయత్నించు
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    స్థిరరాశి వరుస_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
పట్టుకో (sycl::exception స్థిరరాశి &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

స్థిర శూన్య ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) ప్రయత్నించు {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    ఒకవేళ (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } లేకపోతే
        GGML_ABORT("fatal error");
} పట్టుకో (sycl::exception స్థిరరాశి& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
    std::నిష్క్రమించు(1);
}

స్థిర ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_sycl_లేఖాచిత్రం_compute,
    /* .event_record            = */ ggml_backend_sycl_event_రికార్డు,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ఖాళీ,
};

స్థిర ggml_guid_t ggml_backend_sycl_guid() {
    స్థిర ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    తిరిగి &guid;
}

తార్కిక ggml_backend_is_sycl(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

పూర్ణాంకం ggml_backend_sycl_get_device_లెక్క(శూన్య) {
    తిరిగి ggml_sycl_info().device_లెక్క;
}


// backend device

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి ctx->description.c_str();
}

స్థిర శూన్య ggml_backend_sycl_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ఒకవేళ (!sycl_get_mem_info(ctx->device, విడుదల, మొత్తం)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *మొత్తం, *విడుదల);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *విడుదల, *మొత్తం);
}

స్థిర గణన ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    స్థిరరాశి sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    ఒకవేళ (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        తిరిగి GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;
}

స్థిర శూన్య ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);

    తార్కిక host_బఫరు = getenv("GGML_SYCL_NO_PINNED") == శూన్యసూచిక;
#నిర్వచితమైతే GGML_SYCL_NO_PEER_COPY
    తార్కిక events = అబద్ధం;
#లేకపోతే
    తార్కిక events = నిజం;
#అయితే_ముగింపు

    props->caps = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ host_బఫరు,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ events,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    GGML_వాడని(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి ggml_backend_sycl_init(ctx->device);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_sycl_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి ggml_backend_sycl_బఫరు_type(ctx->device);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_sycl_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    తిరిగి ggml_backend_sycl_host_బఫరు_type_for_device(ctx->device);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_sycl_device_బఫరు_from_host_ptr(ggml_backend_dev_t dev, శూన్య * ptr, size_t size, size_t max_tensor_కొలత) {
    GGML_వాడని(dev);
    GGML_వాడని(ptr);
    GGML_వాడని(size);
    GGML_వాడని(max_tensor_కొలత);
    తిరిగి శూన్యసూచిక;
}

స్థిర తార్కిక do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    పూర్ణాంకం device = sycl_ctx->device;
    ఎంపిక (op->op) {
        స్థితి GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                ఒకవేళ (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    తిరిగి నిజం;
                }
                తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            తిరిగి (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_కొలత(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_కొలత(op->src[0]->type));
        స్థితి GGML_OP_CONV_2D_DW:
        స్థితి GGML_OP_CONV_TRANSPOSE_2D:
            తిరిగి నిజం;
        స్థితి GGML_OP_UNARY:
            ఎంపిక (ggml_get_unary_op(op)) {
                స్థితి GGML_UNARY_OP_SGN:
                స్థితి GGML_UNARY_OP_ABS:
                స్థితి GGML_UNARY_OP_NEG:
                స్థితి GGML_UNARY_OP_అడుగు:
                స్థితి GGML_UNARY_OP_RELU:
                స్థితి GGML_UNARY_OP_HARDSIGMOID:
                స్థితి GGML_UNARY_OP_TANH:
                స్థితి GGML_UNARY_OP_GELU:
                స్థితి GGML_UNARY_OP_SILU:
                స్థితి GGML_UNARY_OP_SIGMOID:
                స్థితి GGML_UNARY_OP_HARDSWISH:
                స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                స్థితి GGML_UNARY_OP_GELU_ERF:
                స్థితి GGML_UNARY_OP_EXP:
                స్థితి GGML_UNARY_OP_EXPM1:
                స్థితి GGML_UNARY_OP_SOFTPLUS:
                స్థితి GGML_UNARY_OP_ELU:
                స్థితి GGML_UNARY_OP_XIELU:
                స్థితి GGML_UNARY_OP_CEIL:
                    తిరిగి నిజం;
                స్థితి GGML_UNARY_OP_FLOOR:
                స్థితి GGML_UNARY_OP_విడత:
                స్థితి GGML_UNARY_OP_TRUNC:
                    తిరిగి నిజం;
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_GLU:
            ఎంపిక (ggml_get_glu_op(op)) {
                స్థితి GGML_GLU_OP_REGLU:
                స్థితి GGML_GLU_OP_GEGLU:
                స్థితి GGML_GLU_OP_SWIGLU:
                స్థితి GGML_GLU_OP_SWIGLU_OAI:
                స్థితి GGML_GLU_OP_GEGLU_ERF:
                స్థితి GGML_GLU_OP_GEGLU_శీఘ్ర:
                స్థితి GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    తిరిగి ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                అప్రమేయ:
                    తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_MUL_MAT:
        స్థితి GGML_OP_MUL_MAT_ID:
            {
                నిర్మాణం ggml_tensor * a = op->src[0];
                నిర్మాణం ggml_tensor * b = op->src[1];

                ఒకవేళ (a->ne[3] != b->ne[3]) {
                    తిరిగి అబద్ధం;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                ఒకవేళ (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  తిరిగి అబద్ధం;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                ఒకవేళ (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    తిరిగి అబద్ధం;
                }

                ఒకవేళ (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    తిరిగి అబద్ధం;
                }

                తిరిగి నిజం;
            }
        స్థితి GGML_OP_బహిర్గతం_PROD:
            తిరిగి op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            {
                ఎంపిక (op->src[0]->type) {
                    స్థితి GGML_TYPE_I32:
                    స్థితి GGML_TYPE_F16:
                    స్థితి GGML_TYPE_BF16:
                    స్థితి GGML_TYPE_F32:
                    స్థితి GGML_TYPE_Q1_0:
                    స్థితి GGML_TYPE_MXFP4:
                    స్థితి GGML_TYPE_NVFP4:
                    స్థితి GGML_TYPE_IQ2_XXS:
                    స్థితి GGML_TYPE_IQ2_XS:
                    స్థితి GGML_TYPE_IQ2_S:
                    స్థితి GGML_TYPE_IQ3_XXS:
                    స్థితి GGML_TYPE_IQ1_S:
                    స్థితి GGML_TYPE_IQ1_M:
                    స్థితి GGML_TYPE_IQ3_S:
                    స్థితి GGML_TYPE_IQ4_NL:
                    స్థితి GGML_TYPE_IQ4_XS:
                    స్థితి GGML_TYPE_Q2_K:
                    స్థితి GGML_TYPE_Q3_K:
                    స్థితి GGML_TYPE_Q4_0:
                    స్థితి GGML_TYPE_Q4_1:
                    స్థితి GGML_TYPE_Q4_K:
                    స్థితి GGML_TYPE_Q5_0:
                    స్థితి GGML_TYPE_Q5_1:
                    స్థితి GGML_TYPE_Q5_K:
                    స్థితి GGML_TYPE_Q6_K:
                    స్థితి GGML_TYPE_Q8_0:
                        తిరిగి నిజం;
                    అప్రమేయ:
                        తిరిగి అబద్ధం;
                }
            }
        స్థితి GGML_OP_GET_అడ్డువరుసలు_వెనుక:
            // return true;
             తిరిగి op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         స్థితి GGML_OP_సమితి:
               తిరిగి (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            {
                ఒకవేళ (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    తిరిగి అబద్ధం;
                }
                స్వయంచాలక res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                తిరిగి res;
            }
        స్థితి GGML_OP_DSV4_HC_PRE:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_DSV4_HC_COMB:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_DSV4_HC_POST:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == శూన్యసూచిక || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_LIGHTNING_INDEXER:
            తిరిగి op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_కొలత * 8;
        స్థితి GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                ఒకవేళ (src0_type == GGML_TYPE_F16) {
                    ఒకవేళ (src1_type == GGML_TYPE_Q2_K ||
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
                        తిరిగి అబద్ధం;
                    }
                }

                ఒకవేళ (src0_type == GGML_TYPE_BF16) {
                    ఒకవేళ (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        తిరిగి అబద్ధం;
                    }
                }

                ఒకవేళ (src0_type == GGML_TYPE_F32) {
                    ఒకవేళ (src1_type == GGML_TYPE_Q2_K ||
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
                        తిరిగి అబద్ధం;
                    }
                }

                ఒకవేళ (src1_type == GGML_TYPE_F32) {
                    ఒకవేళ (src0_type == GGML_TYPE_Q1_0 ||
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
                        తిరిగి అబద్ధం;
                    }
                }

                ఒకవేళ (src0_type == src1_type) {
                    ఒకవేళ (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        తిరిగి అబద్ధం;
                    }
                }

                ఒకవేళ (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    తిరిగి అబద్ధం;
                }

                తిరిగి నిజం;
            }
        స్థితి GGML_OP_REPEAT_వెనుక:
            {
                ggml_type src0_type = op->src[0]->type;
                తిరిగి src0_type == GGML_TYPE_F32;
            }
        స్థితి GGML_OP_CONCAT:
        స్థితి GGML_OP_DUP:
        స్థితి GGML_OP_ARGMAX:
        స్థితి GGML_OP_NONE:
        స్థితి GGML_OP_RESHAPE:
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_ADD1:
        స్థితి GGML_OP_కూడు_ID:
        స్థితి GGML_OP_SUB:
        స్థితి GGML_OP_లెక్క_EQUAL:
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_DIV:
        స్థితి GGML_OP_REPEAT:
            తిరిగి నిజం;
        స్థితి GGML_OP_PAD_REFLECT_1D:
            తిరిగి ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SQR:
        స్థితి GGML_OP_SQRT:
        స్థితి GGML_OP_SIN:
        స్థితి GGML_OP_COS:
        స్థితి GGML_OP_CLAMP:
        స్థితి GGML_OP_LOG:
        స్థితి GGML_OP_NORM:
        స్థితి GGML_OP_L2_NORM:
        స్థితి GGML_OP_సమూహం_NORM:
        స్థితి GGML_OP_RMS_NORM:
            తిరిగి నిజం;
        స్థితి GGML_OP_RMS_NORM_వెనుక:
            తిరిగి ggml_is_contiguous(op->src[0]);
        స్థితి GGML_OP_SCALE:
            // the scale kernel is f32 only
            తిరిగి op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_CONT:
            తిరిగి నిజం;
        స్థితి GGML_OP_TRI:
            {
                స్థిరరాశి ggml_tensor * src0 = op->src[0];
                తిరిగి src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        స్థితి GGML_OP_DIAG_ముసుగు_INF:
            తిరిగి నిజం;
        స్థితి GGML_OP_SOFT_MAX:
            తిరిగి నిజం;
        స్థితి GGML_OP_SOFT_MAX_వెనుక: {
            దశాంశం max_bias = 0.0f;
            జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) op->op_params + 1, పరిమాణం(దశాంశం));
            తిరిగి max_bias == 0.0f;
        }
        స్థితి GGML_OP_ROPE:
        స్థితి GGML_OP_ROPE_వెనుక:
        స్థితి GGML_OP_IM2_నిలువు:
        స్థితి GGML_OP_IM2_నిలువు_3_D:
        స్థితి GGML_OP_UPSCALE:
            తిరిగి నిజం;
        స్థితి GGML_OP_COL2IM_1D:
            తిరిగి ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#అయితే_ముగింపు
                   ) &&
                   op->src[0]->type == op->type;
        స్థితి GGML_OP_CONV_3D: {
            స్థిరరాశి int32_t * opts = (స్థిరరాశి int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            స్థిరరాశి int64_t knl_n_మొత్తం = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            స్థిరరాశి int64_t patch_మొత్తం = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            తిరిగి op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_మొత్తం > 0 && patch_మొత్తం > 0;
        }
        స్థితి GGML_OP_కూడిక:
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
        స్థితి GGML_OP_సగటు:
            తిరిగి ggml_is_contiguous(op->src[0]);
        స్థితి GGML_OP_ARGSORT:
            తిరిగి నిజం;
        స్థితి GGML_OP_శిఖరం_K: {
            స్థిరరాశి ggml_tensor * src0 = op->src[0];
            స్థిరరాశి పూర్ణాంకం k = op->ne[0];
            తిరిగి src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        స్థితి GGML_OP_POOL_2D:
        స్థితి GGML_OP_POOL_1D:
        స్థితి GGML_OP_ACC:
            తిరిగి నిజం;
        స్థితి GGML_OP_PAD:
            ఒకవేళ (ggml_get_op_params_i32(op, 8) != 0) {
                తిరిగి అబద్ధం;
            }
            తిరిగి నిజం;
        స్థితి GGML_OP_LEAKY_RELU:
        స్థితి GGML_OP_TIMESTEP_EMBEDDING:
        స్థితి GGML_OP_RWKV_WKV6:
        స్థితి GGML_OP_RWKV_WKV7:
        స్థితి GGML_OP_GATED_రేఖీయ_ATTN:
        స్థితి GGML_OP_GATED_DELTA_NET:
        స్థితి GGML_OP_OPT_అడుగు_ADAMW:
        స్థితి GGML_OP_OPT_అడుగు_SGD:
            తిరిగి నిజం;
        స్థితి GGML_OP_SSM_CONV:
            తిరిగి op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        స్థితి GGML_OP_ROLL:
            తిరిగి op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_ARANGE:
            తిరిగి op->type == GGML_TYPE_F32;
        స్థితి GGML_OP_SSM_పరిశీలన:
            ఒకవేళ (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                తిరిగి (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_కొలత == 0;
            } లేకపోతే {
                // TODO Mamba-1 not yet ported to SYCL
                తిరిగి అబద్ధం;
            }
        స్థితి GGML_OP_FILL:
        స్థితి GGML_OP_CUMSUM:
        స్థితి GGML_OP_DIAG:
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS:
        స్థితి GGML_OP_CROSS_ENTROPY_LOSS_వెనుక:
            తిరిగి నిజం;
        స్థితి GGML_OP_SOLVE_TRI:
            తిరిగి op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            తిరిగి ggml_sycl_flash_attn_ext_supported(device, op);
        అప్రమేయ:
            తిరిగి అబద్ధం;
    }

    GGML_వాడని(dev);
}

స్థిర తార్కిక ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    తార్కిక res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    తిరిగి res;
}

స్థిర తార్కిక ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    ఒకవేళ (buft->iface.get_name != ggml_backend_sycl_బఫరు_type_get_name) {
        తిరిగి అబద్ధం;
    }
    ggml_backend_sycl_బఫరు_type_context * buft_ctx = (ggml_backend_sycl_బఫరు_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి buft_ctx->device == sycl_ctx->device;
}

స్థిర int64_t get_op_batch_కొలత(స్థిరరాశి ggml_tensor * op) {
    ఎంపిక (op->op) {
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            తిరిగి 0;
        స్థితి GGML_OP_MUL_MAT:
            తిరిగి op->ne[1];
        స్థితి GGML_OP_MUL_MAT_ID:
        స్థితి GGML_OP_ROPE:
            తిరిగి op->ne[2];
        అప్రమేయ:
            తిరిగి ggml_nrows(op);
    }
}

స్థిర తార్కిక ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    తిరిగి get_op_batch_కొలత(op) >= sycl_ctx->op_offload_min_batch_కొలత;
}

స్థిర ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#నిర్వచితమైతే GGML_SYCL_NO_PEER_COPY
    తిరిగి శూన్యసూచిక;
#లేకపోతే
  sycl::event *event_ptr = కొత్త sycl::event();

  తిరిగి కొత్త ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#అయితే_ముగింపు
}

స్థిర శూన్య ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) ప్రయత్నించు {
  GGML_వాడని(dev);
  ఒకవేళ (event == శూన్యసూచిక) {
    తిరిగి;
  }

  ఒకవేళ (event->context != శూన్యసూచిక) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    తొలగించు sycl_event;
    event->context = శూన్యసూచిక;
  }

  తొలగించు event;
} పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}


స్థిర శూన్య ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) ప్రయత్నించు {
  GGML_వాడని(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} పట్టుకో (sycl::exception స్థిరరాశి &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
  std::నిష్క్రమించు(1);
}

స్థిర స్థిరరాశి ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_జ్ఞాపకం,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_బఫరు_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_బఫరు_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

నిర్మాణం ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_వాడని(reg);
    తిరిగి GGML_SYCL_NAME;
}

స్థిర size_t ggml_backend_sycl_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    తిరిగి ctx->devices.size();
}

స్థిర ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t సూచీ) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(సూచీ < ctx->devices.size());
    తిరిగి ctx->devices[సూచీ];
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

నిర్మాణం ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_బహిర్గతం[2] = { శూన్యసూచిక, శూన్యసూచిక };
    uint8_t * host_లోపల[2]  = { శూన్యసూచిక, శూన్యసూచిక };
    size_t    host_బైట్లు  = 0;

    శూన్య free_host() {
        కొరకు (పూర్ణాంకం i = 0; i < 2; ++i) {
            స్థిరరాశి sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            ఒకవేళ (host_బహిర్గతం[i]) { sycl::విడుదల(host_బహిర్గతం[i], sctx); host_బహిర్గతం[i] = శూన్యసూచిక; }
            ఒకవేళ (host_లోపల[i])  { sycl::విడుదల(host_లోపల[i],  sctx); host_లోపల[i]  = శూన్యసూచిక; }
        }
        host_బైట్లు = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
స్థిర శూన్య ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, వరుస_ptr q0, వరుస_ptr q1,
                                    శూన్య * dst0, స్థిరరాశి శూన్య * src1, శూన్య * dst1, స్థిరరాశి శూన్య * src0,
                                    size_t nbytes) {
    ఒకవేళ (comm->host_బైట్లు < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_బహిర్గతం[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_లోపల[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_బహిర్గతం[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_లోపల[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_బహిర్గతం[0] && comm->host_లోపల[0] && comm->host_బహిర్గతం[1] && comm->host_లోపల[1]);
        comm->host_బైట్లు = nbytes;
    }

    sycl::event e0 = q0->జ్ఞాపకం_నకలు(comm->host_బహిర్గతం[0], src0, nbytes);
    sycl::event e1 = q1->జ్ఞాపకం_నకలు(comm->host_బహిర్గతం[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::జ్ఞాపకం_నకలు(comm->host_లోపల[0], comm->host_బహిర్గతం[1], nbytes);
    std::జ్ఞాపకం_నకలు(comm->host_లోపల[1], comm->host_బహిర్గతం[0], nbytes);
    q0->జ్ఞాపకం_నకలు(dst0, comm->host_లోపల[0], nbytes);
    q1->జ్ఞాపకం_నకలు(dst1, comm->host_లోపల[1], nbytes);
}

శూన్య * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) ప్రయత్నించు {
    కొరకు (size_t i = 0; i < n_backends; ++i) {
        ఒకవేళ (!ggml_backend_is_sycl(backends[i])) {
            తిరిగి శూన్యసూచిక;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    ఒకవేళ (n_backends != 2) {
        తిరిగి శూన్యసూచిక;
    }

    స్వయంచాలక * ctx = కొత్త ggml_backend_sycl_comm_context;
    ctx->backends.కేటాయించు(backends, backends + n_backends);
    స్వయంచాలక * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    స్వయంచాలక * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    తిరిగి ctx;
}
పట్టుకో (స్థిరరాశి sycl::exception &) { తిరిగి శూన్యసూచిక; }
పట్టుకో (...)                     { తిరిగి శూన్యసూచిక; }

శూన్య ggml_backend_sycl_comm_free(శూన్య * comm_ctx_v) {
    స్వయంచాలక * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ఒకవేళ (comm_ctx == శూన్యసూచిక) {
        తిరిగి;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    ఒకవేళ (comm_ctx->backends.size() == 2) {
        స్వయంచాలక * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        స్వయంచాలక * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        ప్రయత్నించు {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } పట్టుకో (...) { /* best effort during shutdown */ }
    }

    తొలగించు comm_ctx;
}

తార్కిక ggml_backend_sycl_comm_allreduce_tensor(శూన్య * comm_ctx_v, నిర్మాణం ggml_tensor ** tensors) ప్రయత్నించు {
    ఒకవేళ (comm_ctx_v == శూన్యసూచిక) {
        తిరిగి అబద్ధం;
    }

    స్వయంచాలక * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    స్థిరరాశి size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    ఒకవేళ (n_backends != 2) {
        తిరిగి అబద్ధం;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    ఒకవేళ (tensors[0]->type != tensors[1]->type) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        తిరిగి అబద్ధం;
    }
    ఒకవేళ (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        తిరిగి అబద్ధం;
    }

    స్థిరరాశి int64_t nelem  = ggml_nelements(tensors[0]);
    స్థిరరాశి size_t  nbytes = ggml_nbytes(tensors[0]);
    ఒకవేళ (nelem == 0) {
        తిరిగి నిజం;
    }

    స్వయంచాలక * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    స్వయంచాలక * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    వరుస_ptr q0 = ctx0->stream();
    వరుస_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    ఒకవేళ (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    ఒకవేళ (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->దత్తాంశం;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->దత్తాంశం;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->దత్తాంశం, f16_tmp1, tensors[0]->దత్తాంశం, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((దశాంశం) f16_out0[i] + (దశాంశం) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((దశాంశం) f16_out1[i] + (దశాంశం) f16_tmp1[i]);
            });
        });
        తిరిగి నిజం;
    }

    దశాంశం * out0 = (దశాంశం *) tensors[0]->దత్తాంశం;
    దశాంశం * out1 = (దశాంశం *) tensors[1]->దత్తాంశం;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    స్థిర constexpr int64_t BF16_పరిమితి = 32768;

    ఒకవేళ (nelem < BF16_పరిమితి) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        దశాంశం * tmp0 = (దశాంశం *) buf0;
        దశాంశం * tmp1 = (దశాంశం *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->దత్తాంశం, tmp1, tensors[0]->దత్తాంశం, nbytes);

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
        తిరిగి నిజం;
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
        outbox0[i] = (uint16_t) (sycl::బిట్టు_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::బిట్టు_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    స్థిరరాశి size_t bf16_బైట్లు = nelem * పరిమాణం(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_బైట్లు);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::బిట్టు_cast<దశాంశం>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::బిట్టు_cast<దశాంశం>(((uint32_t) inbox1[i]) << 16);
        });
    });

    తిరిగి నిజం;
}
పట్టుకో (స్థిరరాశి sycl::exception &) { తిరిగి అబద్ధం; }
పట్టుకో (...)                     { తిరిగి అబద్ధం; }

స్థిర శూన్య *ggml_backend_sycl_reg_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం *name) {
    GGML_వాడని(reg);

    ఒకవేళ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        తిరిగి (శూన్య *)ggml_backend_sycl_విభజించు_బఫరు_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    ఒకవేళ (strcmp(name, "ggml_backend_comm_init") == 0) {
        తిరిగి (శూన్య *)ggml_backend_sycl_comm_init;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_comm_free") == 0) {
        తిరిగి (శూన్య *)ggml_backend_sycl_comm_free;
    }
    ఒకవేళ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        తిరిగి (శూన్య *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_వాడని(name);
    తిరిగి శూన్యసూచిక;
}

స్థిర స్థిరరాశి ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_లెక్క,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_చిరునామా,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    స్థిర ggml_backend_reg reg;
    స్థిర తార్కిక initialized = అబద్ధం;

    {
        స్థిర std::mutex mutex;
        std::lock_guard<std::mutex> తాళం(mutex);
        ఒకవేళ (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = కొత్త ggml_backend_sycl_reg_context;
            స్థిరరాశి పూర్ణాంకం min_batch_కొలత = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            కొరకు (పూర్ణాంకం i = 0; i < ggml_sycl_info().device_లెక్క; i++) {
                ggml_backend_sycl_device_context * dev_ctx = కొత్త ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_సమితి_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_కొలత = min_batch_కొలత;

                ggml_backend_dev_t dev = కొత్త ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = నిజం;
    }

    తిరిగి &reg;
}

ggml_backend_t ggml_backend_sycl_init(పూర్ణాంకం device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_సూచీ(device);

    ggml_backend_sycl_context * ctx = కొత్త ggml_backend_sycl_context(device);
    ఒకవేళ (ctx == శూన్యసూచిక) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        తిరిగి శూన్యసూచిక;
    };

    ggml_backend_t sycl_backend = కొత్త ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    తిరిగి sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
