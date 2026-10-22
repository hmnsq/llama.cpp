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

#شامل <algorithm>
#شامل <array>
#شامل <assert.h>
#شامل <atomic>
#شامل <cctype>
#شامل <cinttypes>
#شامل <cstddef>
#شامل <cstdint>
#شامل <cstdlib>
#شامل <float.h>
#شامل <limits>
#شامل <optional>
#شامل <stdint.h>
#شامل <stdio.h>
#شامل <vector>
#شامل <cmath>
#شامل <iostream>
#شامل <fstream>
#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <regex>

#شامل <sycl/sycl.hpp>
#شامل <sycl/backend.hpp>
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#شامل <level_zero/ze_api.h>
#شامل <level_zero/zes_api.h>
#اگر_ختم
#اگر defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_حافظہ_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#اگر_ختم
#اگر SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#اگر_ختم
#شامل <sycl/half_type.hpp>

#شامل "ggml.h"
#شامل "ggml-sycl.h"
#شامل "ggml-impl.h"
#شامل "ggml-backend-impl.h"

#شامل "ggml-sycl/add-id.hpp"
#شامل "ggml-sycl/backend.hpp"
#شامل "ggml-sycl/common.hpp"
#شامل "ggml-sycl/element_wise.hpp"
#شامل "ggml-sycl/fwht.hpp"
#شامل "ggml-sycl/fused-gemm.hpp"
#شامل "ggml-sycl/gemm.hpp"
#شامل "ggml-sycl/getrows.hpp"
#شامل "ggml-sycl/mem.hpp"
#شامل "ggml-sycl/norm.hpp"
#شامل "ggml-sycl/presets.hpp"
#شامل "ggml-sycl/quantize.hpp"
#شامل "ggml-sycl/repeat_back.hpp"
#شامل "ggml-sycl/set_rows.hpp"
#شامل "ggml-sycl/set.hpp"
#شامل "ggml-sycl/dsv4-hc.hpp"
#شامل "ggml-sycl/lightning-indexer.hpp"
#شامل "ggml-sycl/conv2d.hpp"
#شامل "ggml-sycl/conv2d-dw.hpp"
#شامل "ggml-sycl/conv2d-transpose.hpp"
#شامل "ggml-sycl/ssm_conv.hpp"
#شامل "ggml-sycl/sycl_hw.hpp"
#شامل "ggml-sycl/ssm_scan.hpp"
#شامل "ggml-sycl/fill.hpp"
#شامل "ggml-sycl/cumsum.hpp"
#شامل "ggml-sycl/diag.hpp"
#شامل "ggml-sycl/opt-step.hpp"
#شامل "ggml-sycl/solve_tri.hpp"
#شامل "ggml-sycl/gated_delta_net.hpp"
#شامل "ggml-sycl/pool.hpp"
#شامل "ggml-sycl/cross_entropy_loss.hpp"

#تعریف_کرو MEM_سائز_2_M	0x00200000
#تعریف_کرو MEM_سائز_1_G	0x40000000

ساکن منطقی g_sycl_loaded = جھوٹ;
صحیح g_ggml_sycl_debug = 0;
صحیح g_ggml_sycl_dev_debug = 0;
صحیح g_ggml_sycl_enable_optimize = 1;
صحیح g_ggml_sycl_enable_گراف = 0;
صحیح g_ggml_sycl_enable_dnn = 1;
صحیح g_ggml_sycl_fa_onednn = 1;
صحیح g_ggml_sycl_fa_onednn_max_kv = 0;
صحیح g_ggml_sycl_enable_mkl_fa = 1;
صحیح g_ggml_sycl_memtrace = 0;
صحیح g_ggml_sycl_memtrace_قدم = 64;
صحیح g_ggml_sycl_enable_vmm = 1;
صحیح g_ggml_sycl_enable_fusion = 1;
صحیح g_ggml_sycl_enable_esimd = 1;
صحیح g_ggml_sycl_mmvq_wide = 1;
صحیح g_ggml_sycl_prioritize_dmmv = 0;
صحیح g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
صحیح g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
صحیح g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
صحیح g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
ساکن ثابت حرف * ggml_sycl_dynamic_precision_نام[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
ساکن صحیح ggml_sycl_get_env_precision(ثابت حرف * name, صحیح def) {
    ثابت حرف * env = getenv(name);
    اگر (!env) {
        واپس def;
    }
    std::string mode(env);
    برائے (حرف & c : mode) {
        c = (حرف) std::بڑا_حرف((غیرعلامتی حرف) c);
    }
    برائے (صحیح i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        اگر (mode == ggml_sycl_dynamic_precision_نام[i]) {
            واپس i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_نام[def]);
    واپس def;
}
صحیح g_ggml_sycl_use_async_mem_op = 0;
صحیح g_ggml_sycl_use_async_mem_op_requested = 1;
صحیح g_ggml_sycl_use_level_zero_api = 0;
صحیح g_ggml_sycl_enable_flash_attention = 1;
صحیح g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
صحیح g_ggml_sycl_usm_system = 0;
صحیح g_ggml_sycl_enable_host_pinned_mem = 1;
صحیح g_ggml_sycl_host_pinned_mem_2g = 0;
صحیح g_ggml_sycl_upload_staging_slots = 4;
صحیح g_ggml_sycl_get_mem_api = حافظہ_API_TYPE_LEVEL_ZERO;
صحیح g_ggml_sycl_enable_sparse_fa = 0;
صحیح g_ggml_sycl_debug_sparse_fa = 0;
صحیح g_ggml_sycl_sparse_fa_margin = 256;

#اگر GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
ساکن منطقی ggml_sycl_dnnl_detect_optimized_gemm(صحیح device) {
    استعمال dt = dnnl::حافظہ::ڈیٹا_type;

    کوشش {
        ثابت dnnl::حافظہ::dims dims    = { 1, 64, 64 };
        ثابت dnnl::حافظہ::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.سیٹ_scratchpad_منوال(dnnl::scratchpad_منوال::user);

        ثابت خودکار & q   = dpct::dev_mgr::instance().get_device(device).default_قطار();
        ثابت خودکار   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ثابت خودکار   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::حافظہ::desc(dims, dt::f16, strides),
                                                        dnnl::حافظہ::desc(dims, dt::f16, strides),
                                                        dnnl::حافظہ::desc(dims, dt::f32, strides), attr);

        ثابت std::string impl = pd.impl_info_str();
        اگر (impl.find("ref") == std::string::npos) {
            واپس سچ;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } پکڑو (ثابت std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    واپس جھوٹ;
}
#اگر_ختم

ساکن ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    کوشش {
        info.device_گنتی = dpct::dev_mgr::instance().device_گنتی();
    } پکڑو (sycl::exception ثابت &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_گنتی = 0;
    }
    اگر (info.device_گنتی == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        واپس info;
    }

    GGML_ASSERT(info.device_گنتی <= GGML_SYCL_MAX_DEVICES);

    int64_t کل_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    برائے (صحیح i = 0; i < info.device_گنتی; ++i) {
        dpct::device_info prop;
        خودکار & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#اگر !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#ورنہ
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        اگر (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ثابت size_t physical_صفحہ = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_صفحہ);
        }
#اگر_ختم

        info.default_tensor_بانٹو[i] = کل_vram;
        کل_vram += prop.get_global_mem_سائز();

        info.devices[i].cc =
            100 * prop.get_major_نسخہ() + 10 * prop.get_minor_نسخہ();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#اگر GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#اگر_ختم
        info.devices[i].smpbo = prop.get_local_mem_سائز();
        info.devices[i].warp_سائز = WARP_سائز;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_گروہ_sizes[i] = prop.get_max_work_گروہ_سائز();
        info.devices[i].max_wg_فی_cu = info.max_work_گروہ_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        اگر (device.is_gpu() && device.default_قطار().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = جھوٹ;
        }

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        اگر (info.ext_oneapi_level_zero && device.is_gpu() && device.default_قطار().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_قطار().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_نتیجہ_t r = zeDeviceGetProperties(ze_dev, &props);
            اگر (r == ZE_نتیجہ_SUCCESS) {
                info.devices[i].l0_device_type_valid = سچ;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#اگر_ختم
    }

    برائے (صحیح id = 0; id < info.device_گنتی; ++id) {
        info.default_tensor_بانٹو[id] /= کل_vram;
    }

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#اگر_ختم

    واپس info;
}

ثابت ggml_sycl_device_info & ggml_sycl_info() {
    ساکن ggml_sycl_device_info info = ggml_sycl_init();
    واپس info;
}

ساکن خالی print_device_detail(صحیح id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string نسخہ_2;
    نسخہ_2 += std::to_string(prop.get_major_نسخہ());
    نسخہ_2 += ".";
    نسخہ_2 += std::to_string(prop.get_minor_نسخہ());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    خودکار global_mem_سائز = prop.get_global_mem_سائز()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), نسخہ_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_گروہ_سائز(), prop.get_max_sub_گروہ_سائز(),
            global_mem_سائز, device.get_info<sycl::info::device::driver_نسخہ>().c_str());
}

ساکن خالی print_device_opt_feature(صحیح device_گنتی) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    برائے (صحیح id = 0; id < device_گنتی; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      صحیح type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
خالی ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    صحیح device_گنتی = dpct::dev_mgr::instance().device_گنتی();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_گنتی);

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

    برائے (صحیح id = 0; id < device_گنتی; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      صحیح type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_گنتی);
}

ساکن ثابت حرف* dev2dev_int2str(صحیح dev2dev) {
    اگر (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        واپس "SYCL API";
    } ورنہ اگر (dev2dev == DEV2DEV_MEMCPY_L0) {
        واپس "Level Zero API";
    } ورنہ اگر (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        واپس "Host Forward";
    } ورنہ {
        واپس "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساکن ze_نتیجہ_t init_zes() {
    ze_نتیجہ_t res = zesInit(0);
    اگر (res != ZE_نتیجہ_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (صحیح) res);
    }
    واپس res;
}

ساکن ze_نتیجہ_t get_zes_init_res() {
    ساکن ze_نتیجہ_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (صحیح) zes_init_res);
    واپس zes_init_res;
}
#اگر_ختم

ساکن خالی initialize_sycl_begining() {
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#اگر_ختم
}

ساکن خالی ggml_check_sycl() کوشش {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    ساکن منطقی initialized = جھوٹ;

    اگر (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_گراف = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_قدم = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
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

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#ورنہ
        g_ggml_sycl_use_level_zero_api = 0;
#اگر_ختم
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", حافظہ_API_TYPE_LEVEL_ZERO);
        اگر (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = حافظہ_API_TYPE_SYCL;
        }

#اگر_تعریف_شدہ SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#ورنہ
        g_ggml_sycl_enable_flash_attention = 0;
#اگر_ختم

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
#اگر defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#اگر_ختم

#اگر defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#اگر_ختم

#اگر defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#اگر_ختم

#اگر defined(GGML_SYCL_گراف)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#اگر_ختم

#اگر defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#اگر_ختم
#اگر defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#اگر_ختم

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#اگر_ختم

#اگر defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#اگر_ختم
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_قدم);
#اگر_تعریف_شدہ SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#اگر_ختم

#اگر_تعریف_شدہ GGML_SYCL_گراف
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_گراف);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#اگر_ختم

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_نام[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_نام[g_ggml_sycl_dynamic_required_precision]);

#اگر defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#اگر_ختم

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#اگر defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#اگر_ختم
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#ورنہ
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#اگر_ختم

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
#اگر defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_حافظہ_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_گراف;
        اگر (g_ggml_sycl_use_async_mem_op) {
            برائے (غیرعلامتی صحیح i = 0; i < dpct::dev_mgr::instance().device_گنتی(); ++i) {
                اگر (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_حافظہ_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    توڑو;
                }
            }
        }
#اگر_ختم
        اگر (CHECK_TRY_ERROR(g_all_sycl_device_گنتی =
                            dpct::dev_mgr::instance().device_گنتی()) != 0) {
            initialized = سچ;
            g_sycl_loaded = جھوٹ;
            واپس;
        }
        GGML_ASSERT(g_all_sycl_device_گنتی <= GGML_SYCL_MAX_DEVICES);

        initialized = سچ;
        g_sycl_loaded = سچ;
        ggml_backend_sycl_print_sycl_devices();
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
اندرونی خالی check_allow_gpu_اشاریہ(ثابت صحیح device_اشاریہ) {
  اگر (device_اشاریہ >= ggml_sycl_info().device_گنتی) {
    حرف error_buf[256];
    snprintf(
        error_buf,
        حجم(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_اشاریہ,
        ggml_sycl_info().device_گنتی - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    أكد(جھوٹ);
  }
}

GGML_API خالی ggml_backend_sycl_get_gpu_list(صحیح *id_list, صحیح max_len) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    برائے(صحیح i=0;i<max_len;i++) id_list[i] = -1;

    برائے (صحیح i=0;i< ggml_sycl_info().device_گنتی;i++){
        اگر (i>=max_len) توڑو;
        id_list[i] = i;
    }
    واپس;
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

اندرونی خالی free_aligned_mem_host(خالی * memblock) {
#اگر_تعریف_شدہ _WIN32
    _aligned_free(memblock);
#ورنہ
    آزاد(memblock);
#اگر_ختم
}

// sycl buffer

ساخت ggml_backend_sycl_بفر_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    ساکن constexpr size_t staging_خانہ_سائز = 8*1024*1024;

    ساخت host_staging {
        خالی * ڈیٹا = خالی_اشاریہ;
        std::vector<sycl::event> events;
        std::vector<منطقی> submitted;
        صحیح اگلا = 0;
    };

    صحیح device;
    خالی * dev_ptr = خالی_اشاریہ;
    قطار_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    منطقی is_usm_system;

    ggml_backend_sycl_بفر_context(صحیح device, خالی * dev_ptr, قطار_ptr stream, منطقی is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_اشاریہ(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    خالی drop_host_staging() {
        برائے (size_t i = 0; i < staging.submitted.size(); ++i) {
            اگر (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = جھوٹ;
            }
        }
        اگر (staging.ڈیٹا != خالی_اشاریہ) {
            sycl::آزاد(staging.ڈیٹا, *stream);
            staging.ڈیٹا = خالی_اشاریہ;
        }
    }

    ~ggml_backend_sycl_بفر_context() {
        drop_host_staging();
        اگر (dev_ptr != خالی_اشاریہ) {
            ggml_sycl_سیٹ_device(device);
            اگر (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            ورنہ
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        برائے (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

ساکن ثابت حرف * ggml_backend_sycl_بفر_type_get_name(ggml_backend_بفر_type_t buft);

ساکن منطقی ggml_backend_بفر_is_sycl(ggml_backend_بفر_t بفر) {
    واپس بفر->buft->iface.get_name == ggml_backend_sycl_بفر_type_get_name;
}

ساکن خالی
ggml_backend_sycl_بفر_free_بفر(ggml_backend_بفر_t بفر) کوشش {
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    ggml_sycl_سیٹ_device(ctx->device);

    حذف ctx;
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی * ggml_backend_sycl_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    واپس ctx->dev_ptr;
}

ساکن شمار ggml_حالت
ggml_backend_sycl_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                     ggml_tensor *tensor) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *)بفر->context;

    اگر (tensor->view_src != تہی) {
        أكد(tensor->view_src->بفر->buft == بفر->buft);
        واپس GGML_حالت_SUCCESS;
    }

    اگر (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        انتخاب (tensor->type) {
            حالت GGML_TYPE_Q4_0:
            حالت GGML_TYPE_Q8_0:
            حالت GGML_TYPE_Q2_K:
            حالت GGML_TYPE_Q3_K:
            حالت GGML_TYPE_Q4_K:
            حالت GGML_TYPE_Q5_K:
            حالت GGML_TYPE_Q6_K:
            حالت GGML_TYPE_IQ3_XXS:
            حالت GGML_TYPE_IQ3_S:
            حالت GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = نیا ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                توڑو;
            }
            طےشدہ:
                توڑو;
        }
    }

    اگر (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_سائز = ggml_nbytes(tensor);
        size_t padded_سائز = ggml_backend_buft_get_alloc_سائز(بفر->buft, tensor);

        اگر (padded_سائز > original_سائز && tensor->view_src == خالی_اشاریہ) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (حرف *)tensor->ڈیٹا + original_سائز, 0,
                padded_سائز - original_سائز).wait()));
        }
    }
    واپس GGML_حالت_SUCCESS;
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                                ggml_tensor *tensor,
                                                ثابت خالی *ڈیٹا, size_t انحراف,
                                                size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    ggml_sycl_سیٹ_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ثابت صحیح n_slots = g_ggml_sycl_upload_staging_slots;
    اگر (n_slots > 0 && ctx->staging.ڈیٹا == خالی_اشاریہ) {
        ctx->staging.ڈیٹا = sycl::malloc_host(n_slots * ctx->staging_خانہ_سائز, *ctx->stream);
        اگر (ctx->staging.ڈیٹا != خالی_اشاریہ) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.تفویض(n_slots, جھوٹ);
        }
    }
    اگر (ctx->staging.ڈیٹا != خالی_اشاریہ) {
        قطار_ptr    stream    = ctx->stream;
        حرف *       dst       = (حرف *) tensor->ڈیٹا + انحراف;
        ثابت حرف * src       = (ثابت حرف *) ڈیٹا;
        size_t       بقایا = size;
        جب (بقایا > 0) {
            ثابت size_t chunk = std::min(بقایا, ctx->staging_خانہ_سائز);
            ثابت صحیح    خانہ  = ctx->staging.اگلا;
            ctx->staging.اگلا = (ctx->staging.اگلا + 1) % (صحیح) ctx->staging.submitted.size();
            اگر (ctx->staging.submitted[خانہ]) {
                ctx->staging.events[خانہ].wait_and_throw();
            }
            خالی * stage = (حرف *) ctx->staging.ڈیٹا + خانہ * ctx->staging_خانہ_سائز;
            حافظہ_نقل(stage, src, chunk);
            ctx->staging.events[خانہ] = stream->حافظہ_نقل(dst, stage, chunk);
            ctx->staging.submitted[خانہ] = سچ;
            src       += chunk;
            dst       += chunk;
            بقایا -= chunk;
        }
        واپس;
    }

    خودکار stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_قطار());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#اگر_غیر_تعریف_شدہ _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    حرف * host_buf = (حرف *) حافظہ_دو(size);
    حافظہ_نقل(host_buf, ڈیٹا, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, host_buf, size).wait()));
    آزاد(host_buf);
#ورنہ
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).حافظہ_نقل((حرف *) tensor->ڈیٹا + انحراف, ڈیٹا, size).wait()));
#اگر_ختم
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                ثابت ggml_tensor *tensor,
                                                خالی *ڈیٹا, size_t انحراف,
                                                size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;

    ggml_sycl_سیٹ_device(ctx->device);
    خودکار stream = dpct::dev_mgr::instance().get_device(ctx->device).default_قطار();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.حافظہ_نقل(ڈیٹا, (ثابت حرف *)tensor->ڈیٹا + انحراف, size)
            .wait()));
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساکن منطقی ggml_sycl_is_l0_discrete_gpu(صحیح device) {
    واپس ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#اگر_ختم

ساکن خالی memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, خالی *ptr_dst,
                         ثابت خالی *ptr_src, size_t size) {
    حرف *host_buf = (حرف *)حافظہ_دو(size);
    اگر (host_buf == خالی_اشاریہ) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.حافظہ_نقل(host_buf, (ثابت حرف *)ptr_src, size).wait();
    q_dst.حافظہ_نقل((حرف *)ptr_dst, host_buf, size).wait();
    آزاد(host_buf);
}

ساکن خالی dev2dev_memcpy(صحیح device_dst, sycl::queue &q_dst, صحیح device_src, sycl::queue &q_src, خالی *ptr_dst,
                    ثابت خالی *ptr_src, size_t size) {

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    اگر (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ثابت منطقی l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        اگر (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            خودکار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            خودکار ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_قطار_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_قطار_DESC, خالی_اشاریہ, 0, 0,
                                            0, ZE_COMMAND_قطار_منوال_SYNCHRONOUS, ZE_COMMAND_قطار_ترجیح_معمول};
            ze_command_list_handle_t cl;
            ze_نتیجہ_t r = ze_Command_List_Create_فوری(ze_ctx, ze_dev, &cq_desc, &cl);
            اگر (r == ZE_نتیجہ_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_حافظہ_Copy(cl, ptr_dst, ptr_src, size, خالی_اشاریہ, 0, خالی_اشاریہ);
                zeCommandListDestroy(cl);
                اگر (r == ZE_نتیجہ_SUCCESS) {
                    واپس;
                }
            }
        }
    }
#اگر_ختم

    اگر (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        اگر (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.حافظہ_نقل(ptr_dst, ptr_src, size).wait()));
            واپس;
        }
    }

    // Host-staged copy
    اگر(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } ورنہ {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

ساکن منطقی
ggml_backend_sycl_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                    ثابت ggml_tensor *src,
                                    ggml_tensor *dst) کوشش {
    منطقی is_cpy_supported = ggml_backend_بفر_is_sycl(src->بفر);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    اگر (is_cpy_supported) {
        ggml_backend_sycl_بفر_context * src_ctx = (ggml_backend_sycl_بفر_context *)src->بفر->context;
        ggml_backend_sycl_بفر_context * dst_ctx = (ggml_backend_sycl_بفر_context *)dst->بفر->context;

        ggml_sycl_سیٹ_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_سیٹ_device(dst_ctx->device);
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

        قطار_ptr دھارا_dst = dst_ctx->stream;
        قطار_ptr دھارا_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *دھارا_dst, src_ctx->device, *دھارا_src, dst->ڈیٹا, src->ڈیٹا, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#اگر 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).حافظہ_نقل(
            (حرف *)dst->ڈیٹا, (ثابت حرف *)src->ڈیٹا, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#اگر_ختم
        واپس سچ;
    }
    واپس جھوٹ;
    GGML_غیر_مستعمل(بفر);
} پکڑو (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_بفر_clear(ggml_backend_بفر_t بفر,
                                           uint8_t value) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, بفر->size);
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;

    ggml_sycl_سیٹ_device(ctx->device);
    قطار_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    برائے (size_t off = 0; off < بفر->size; off += MAX_CHUNK) {
        size_t chunk = std::min(بفر->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<حرف*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value,
                                                   size_t انحراف, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, انحراف, value);
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx->device));
    خودکار stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_قطار());
    اگر (size == 0) {
        واپس;  // Nothing to do
    }
    اگر (tensor->ڈیٹا == خالی_اشاریہ) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    خالی * ہدف_ptr = static_cast<حرف *>(tensor->ڈیٹا) + انحراف;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(ہدف_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

ساکن خالی ggml_backend_sycl_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    اگر (بفر == خالی_اشاریہ) {
        واپس;
    }

    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;

    اگر (ctx != خالی_اشاریہ) {
        برائے (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

ساکن ثابت ggml_backend_بفر_i ggml_backend_sycl_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_sycl_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_sycl_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_sycl_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_بفر_clear,
    /* .reset           = */ ggml_backend_sycl_بفر_reset,
};

// sycl buffer type
ساخت ggml_backend_sycl_بفر_type_context {
    صحیح device;
    std::string name;

    // each buffer type has its own stream
    قطار_ptr stream = خالی_اشاریہ;
};

ساکن ثابت حرف * ggml_backend_sycl_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_sycl_بفر_type_context * ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;

    واپس ctx->name.c_str();
}

ساکن منطقی check_usm_system(صحیح device, size_t size) {
    منطقی use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_سائز_1_G);

    اگر (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = جھوٹ;
    }

    واپس use_usm_system;
}

اندرونی خالی * aligned_malloc_host(size_t alignment, size_t size) {
#اگر_تعریف_شدہ _WIN32
    واپس _aligned_malloc(size, alignment);
#ورنہ
    واپس aligned_alloc(alignment, size);
#اگر_ختم
}

ساکن ggml_backend_بفر_t
ggml_backend_sycl_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                           size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_بفر_type_context * buft_ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    ggml_sycl_سیٹ_device(buft_ctx->device);
    ثابت قطار_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_سائز_2_M;
    size_t aligned_سائز = ((size + alignment - 1) / alignment) * alignment;
    منطقی use_usm_system = check_usm_system(buft_ctx->device, aligned_سائز);

    خالی * dev_ptr;
    اگر (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (خالی *)aligned_malloc_host(alignment, aligned_سائز);
        اگر (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            واپس خالی_اشاریہ;
        }
    } ورنہ {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (خالی *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_بفر)));
        اگر (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          واپس خالی_اشاریہ;
        }
    }
    ggml_backend_sycl_بفر_context * ctx = نیا  ggml_backend_sycl_بفر_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    واپس ggml_backend_بفر_init(buft, ggml_backend_sycl_بفر_interface, ctx, size);
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن size_t ggml_backend_sycl_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس SYCL_بفر_ALIGNMENT;
    GGML_غیر_مستعمل(buft);
}

ساکن منطقی is_bmg_g31_arch(صحیح device) {
    واپس ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

ساکن size_t ggml_backend_sycl_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    ggml_backend_sycl_بفر_type_context * ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    صحیح device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_سائز = dpct::get_device(device).get_max_mem_alloc_سائز();
    اگر (g_ggml_sycl_host_pinned_mem_2g) {
        واپس std::min(max_alloc_سائز, (size_t) 2LL*1024*1024*1024);
    } ورنہ {
        اگر(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_سائز = max_alloc_سائز*0.6;
        }
        واپس max_alloc_سائز;

    }
    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_sycl_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_سائز(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    اگر (ggml_is_quantized(tensor->type)) {
        اگر (ne0 % میٹرکس_صف_PADDING != 0) {
            size += ggml_صف_سائز(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }
    }

    واپس size;

    GGML_غیر_مستعمل(buft);
}

ساکن ثابت ggml_backend_بفر_type_i ggml_backend_sycl_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_sycl_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_sycl_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ تہی,
};

ggml_backend_بفر_type_t ggml_backend_sycl_بفر_type(صحیح device) {
    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);


    خودکار dev_گنتی = ggml_backend_sycl_get_device_گنتی();

    اگر (device>=dev_گنتی or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_گنتی-1);
        GGML_ASSERT(device<dev_گنتی);
    }
    ساکن ساخت ggml_backend_بفر_type ggml_backend_sycl_بفر_types[GGML_SYCL_MAX_DEVICES];

    ساکن منطقی ggml_backend_sycl_بفر_type_initialized = جھوٹ;

    اگر (!ggml_backend_sycl_بفر_type_initialized) {
        برائے (صحیح i = 0; i < dev_گنتی; i++) {
            خودکار & device_i = dpct::dev_mgr::instance().get_device(i);
            قطار_ptr stream = &(device_i.default_قطار());
            ggml_backend_sycl_بفر_types[i] = {
                /* .iface    = */ ggml_backend_sycl_بفر_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ نیا ggml_backend_sycl_بفر_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_بفر_type_initialized = سچ;
    }
    واپس &ggml_backend_sycl_بفر_types[device];
}

ساکن ggml_backend_بفر_type_t ggml_backend_sycl_بفر_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    صحیح device = ctx->device;
    اگر (device>=ggml_sycl_info().device_گنتی or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_گنتی-1);
        GGML_ASSERT(device<ggml_sycl_info().device_گنتی);
    }
    ساکن ساخت ggml_backend_بفر_type ggml_backend_sycl_بفر_types[GGML_SYCL_MAX_DEVICES];

    ساکن منطقی ggml_backend_sycl_بفر_type_initialized = جھوٹ;

    اگر (!ggml_backend_sycl_بفر_type_initialized) {
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; i++) {
            ggml_backend_sycl_بفر_types[i] = {
                /* .iface    = */ ggml_backend_sycl_بفر_type_interface,
                /* .device   = */ خالی_اشاریہ,
                /* .context  = */ نیا ggml_backend_sycl_بفر_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_بفر_type_initialized = سچ;
    }
    واپس &ggml_backend_sycl_بفر_types[device];
}

// sycl split buffer

ساکن int64_t get_صف_rounding(ggml_type type, ثابت std::array<اعشاری, GGML_SYCL_MAX_DEVICES> & tensor_بانٹو) {
    int64_t min_compute_capability = زیادہ_سے_زیادہ_عدد;
    int64_t max_compute_capability = INT_MIN;
    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        اگر (tensor_بانٹو[i] < (i + 1 < ggml_sycl_info().device_گنتی ? tensor_بانٹو[i + 1] : 1.0f)) {
            اگر (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            اگر (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    انتخاب(type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
            واپس max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
            واپس 64;
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_F32:
            واپس 1;
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_IQ2_XXS:
        حالت GGML_TYPE_IQ2_XS:
        حالت GGML_TYPE_IQ2_S:
        حالت GGML_TYPE_IQ1_S:
        حالت GGML_TYPE_IQ1_M:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ4_XS:
        حالت GGML_TYPE_IQ4_NL:
            واپس max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالت GGML_TYPE_IQ3_S:
            واپس max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالت GGML_TYPE_Q6_K:
            واپس 64;
        طےشدہ:
            GGML_ABORT("fatal error");
    }
}

ساکن خالی get_صف_بانٹو(int64_t * صف_کم, int64_t * صف_بلند, ثابت ggml_tensor * tensor, ثابت std::array<اعشاری, GGML_SYCL_MAX_DEVICES> & tensor_بانٹو, صحیح id) {
    ثابت int64_t nrows = ggml_nrows(tensor);
    ثابت int64_t rounding = get_صف_rounding(tensor->type, tensor_بانٹو);

    *صف_کم = id == 0 ? 0 : nrows*tensor_بانٹو[id];
    *صف_کم -= *صف_کم % rounding;
    اگر (id == ggml_sycl_info().device_گنتی - 1) {
        *صف_بلند = nrows;
    } ورنہ {
        *صف_بلند = nrows*tensor_بانٹو[id + 1];
        *صف_بلند -= *صف_بلند % rounding;
    }
}

ساکن size_t ggml_nbytes_بانٹو(ثابت ساخت ggml_tensor * tensor, صحیح nrows_بانٹو) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    واپس nrows_بانٹو*ggml_صف_سائز(tensor->type, tensor->ne[0]);
}

ساخت ggml_backend_sycl_بانٹو_بفر_type_context {
    std::array<اعشاری, GGML_SYCL_MAX_DEVICES> tensor_بانٹو;
};

ساخت ggml_backend_sycl_بانٹو_بفر_context {
    ~ggml_backend_sycl_بانٹو_بفر_context() کوشش {
        برائے (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    پکڑو (sycl::exception ثابت &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::سطر_ختم;
      std::باہر_نکلو(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<قطار_ptr> streams;
};

ساکن خالی ggml_backend_sycl_بانٹو_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_sycl_بانٹو_بفر_context * ctx = (ggml_backend_sycl_بانٹو_بفر_context *)بفر->context;
    حذف ctx;
}

ساکن خالی * ggml_backend_sycl_بانٹو_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    واپس (خالی *)0x1000;

    GGML_غیر_مستعمل(بفر);
}

ساکن شمار ggml_حالت
ggml_backend_sycl_بانٹو_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                           ggml_tensor *tensor) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == خالی_اشاریہ); // views of split tensors are not supported

    ggml_backend_sycl_بانٹو_بفر_context * ctx = (ggml_backend_sycl_بانٹو_بفر_context *)بفر->context;
    ggml_backend_sycl_بانٹو_بفر_type_context * buft_ctx = (ggml_backend_sycl_بانٹو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = نیا ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    اگر (ctx->streams.خالی_خانے()) {
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_قطار()));
        }
    }

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        int64_t صف_کم, صف_بلند;
        get_صف_بانٹو(&صف_کم, &صف_بلند, tensor, buft_ctx->tensor_بانٹو, i);

        int64_t nrows_بانٹو = صف_بلند - صف_کم;
        اگر (nrows_بانٹو == 0) {
            جاری;
        }

        size_t size = ggml_nbytes_بانٹو(tensor, nrows_بانٹو);
        ثابت size_t original_سائز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        اگر (ne0 % میٹرکس_صف_PADDING != 0) {
            size += ggml_صف_سائز(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }

        ggml_sycl_سیٹ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        حرف * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (حرف *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_بفر)));
        اگر (!buf) {
            حرف err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            پھینکو std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        اگر (size > original_سائز) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_سائز, 0, size - original_سائز)
                    .wait()));
        }

        extra->ڈیٹا_device[i] = buf;

        برائے (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = نیا sycl::event()));
        }
    }
    tensor->extra = extra;
    واپس GGML_حالت_SUCCESS;
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی
ggml_backend_sycl_بانٹو_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                          ggml_tensor *tensor, ثابت خالی *ڈیٹا,
                                          size_t انحراف, size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_بانٹو_بفر_context * ctx = (ggml_backend_sycl_بانٹو_بفر_context *)بفر->context;
    ggml_backend_sycl_بانٹو_بفر_type_context * buft_ctx = (ggml_backend_sycl_بانٹو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];
    ثابت size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        int64_t صف_کم, صف_بلند;
        get_صف_بانٹو(&صف_کم, &صف_بلند, tensor, buft_ctx->tensor_بانٹو, i);

        int64_t nrows_بانٹو = صف_بلند - صف_کم;
        اگر (nrows_بانٹو == 0) {
            جاری;
        }

        ثابت size_t انحراف_بانٹو = صف_کم*nb1;
        size_t size = ggml_nbytes_بانٹو(tensor, nrows_بانٹو);
        ثابت size_t original_سائز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        اگر (ne0 % میٹرکس_صف_PADDING != 0) {
            size += ggml_صف_سائز(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }

        ثابت حرف * buf_host = (ثابت حرف *)ڈیٹا + انحراف_بانٹو;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_سیٹ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .حافظہ_نقل(extra->ڈیٹا_device[i], buf_host, original_سائز)
                .wait()));
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی
ggml_backend_sycl_بانٹو_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                          ثابت ggml_tensor *tensor, خالی *ڈیٹا,
                                          size_t انحراف, size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(انحراف == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_بانٹو_بفر_context * ctx = (ggml_backend_sycl_بانٹو_بفر_context *)بفر->context;
    ggml_backend_sycl_بانٹو_بفر_type_context * buft_ctx = (ggml_backend_sycl_بانٹو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];
    ثابت size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        int64_t صف_کم, صف_بلند;
        get_صف_بانٹو(&صف_کم, &صف_بلند, tensor, buft_ctx->tensor_بانٹو, i);

        int64_t nrows_بانٹو = صف_بلند - صف_کم;
        اگر (nrows_بانٹو == 0) {
            جاری;
        }

        ثابت size_t انحراف_بانٹو = صف_کم*nb1;
        size_t size = ggml_nbytes_بانٹو(tensor, nrows_بانٹو);
        ثابت size_t original_سائز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        اگر (ne0 % میٹرکس_صف_PADDING != 0) {
            size += ggml_صف_سائز(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }

        حرف * buf_host = (حرف *)ڈیٹا + انحراف_بانٹو;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_سیٹ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .حافظہ_نقل(buf_host, extra->ڈیٹا_device[i], original_سائز)
                .wait()));
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_بانٹو_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_غیر_مستعمل(بفر);
    GGML_غیر_مستعمل(value);
}

ساکن ساخت ggml_backend_بفر_i ggml_backend_sycl_بانٹو_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_بانٹو_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_sycl_بانٹو_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_sycl_بانٹو_بفر_init_tensor,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_sycl_بانٹو_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_بانٹو_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ تہی,
    /* .clear           = */ ggml_backend_sycl_بانٹو_بفر_clear,
    /* .reset           = */ تہی,
};

// sycl split buffer type

ساکن ثابت حرف * ggml_backend_sycl_بانٹو_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    واپس GGML_SYCL_NAME "_Split";

    GGML_غیر_مستعمل(buft);
}

ساکن منطقی ggml_backend_بفر_is_sycl_بانٹو(ggml_backend_بفر_t بفر) {
   واپس بفر->buft->iface.get_name == ggml_backend_sycl_بانٹو_بفر_type_get_name;
}

ساکن ggml_backend_بفر_t ggml_backend_sycl_بانٹو_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_بانٹو_بفر_context * ctx = نیا ggml_backend_sycl_بانٹو_بفر_context();

    واپس ggml_backend_بفر_init(buft, ggml_backend_sycl_بانٹو_بفر_interface, ctx, size);
}

ساکن size_t ggml_backend_sycl_بانٹو_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    واپس SYCL_بفر_ALIGNMENT;
    GGML_غیر_مستعمل(buft);
}

ساکن size_t ggml_backend_sycl_بانٹو_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    ggml_backend_sycl_بانٹو_بفر_type_context * ctx = (ggml_backend_sycl_بانٹو_بفر_type_context *)buft->context;

    size_t کل_سائز = 0;

    ثابت int64_t ne0 = tensor->ne[0];

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        int64_t صف_کم, صف_بلند;
        get_صف_بانٹو(&صف_کم, &صف_بلند, tensor, ctx->tensor_بانٹو, i);

        int64_t nrows_بانٹو = صف_بلند - صف_کم;
        اگر (nrows_بانٹو == 0) {
            جاری;
        }

        کل_سائز += ggml_nbytes_بانٹو(tensor, nrows_بانٹو);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        اگر (ne0 % میٹرکس_صف_PADDING != 0) {
            کل_سائز += ggml_صف_سائز(tensor->type, میٹرکس_صف_PADDING - ne0 % میٹرکس_صف_PADDING);
        }
    }

    واپس کل_سائز;
}

ساکن منطقی ggml_backend_sycl_بانٹو_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    واپس جھوٹ;

    GGML_غیر_مستعمل(buft);
}

ساکن ggml_backend_بفر_type_i ggml_backend_sycl_بانٹو_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_بانٹو_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_بانٹو_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_sycl_بانٹو_بفر_type_get_alignment,
    /* .get_max_size        = */ تہی, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_بانٹو_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ ggml_backend_sycl_بانٹو_بفر_type_is_host,
};

ggml_backend_بفر_type_t ggml_backend_sycl_بانٹو_بفر_type(صحیح main_device, ثابت اعشاری * tensor_بانٹو) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_غیر_مستعمل(main_device);

    ساکن std::mutex mutex;
    std::lock_guard<std::mutex> تالا(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    ساکن std::map<std::array<اعشاری, GGML_SYCL_MAX_DEVICES>, ساخت ggml_backend_بفر_type> buft_map;

    std::array<اعشاری, GGML_SYCL_MAX_DEVICES> tensor_بانٹو_arr = {};

    منطقی all_zero = tensor_بانٹو == خالی_اشاریہ || std::all_of(tensor_بانٹو, tensor_بانٹو + GGML_SYCL_MAX_DEVICES, [](اعشاری x) { واپس x == 0.0f; });
    اگر (all_zero) {
        tensor_بانٹو_arr = ggml_sycl_info().default_tensor_بانٹو;
    } ورنہ {
        اعشاری بانٹو_مجموع = 0.0f;
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
            tensor_بانٹو_arr[i] = بانٹو_مجموع;
            بانٹو_مجموع += tensor_بانٹو[i];
        }
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
            tensor_بانٹو_arr[i] /= بانٹو_مجموع;
        }
    }

    خودکار it = buft_map.find(tensor_بانٹو_arr);
    اگر (it != buft_map.end()) {
        واپس &it->second;
    }

    ساخت ggml_backend_بفر_type buft {
        /* .iface   = */ ggml_backend_sycl_بانٹو_بفر_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ نیا ggml_backend_sycl_بانٹو_بفر_type_context{tensor_بانٹو_arr},
    };

    خودکار نتیجہ = buft_map.emplace(tensor_بانٹو_arr, buft);
    واپس &نتیجہ.پہلا->second;
}

// host buffer type

ساخت ggml_backend_sycl_device_context {
    صحیح device;
    std::string name;
    std::string description;
    صحیح op_offload_min_batch_سائز;
};

ساکن ثابت حرف * ggml_backend_sycl_host_بفر_type_name(ggml_backend_بفر_type_t buft) {
    واپس GGML_SYCL_NAME "_Host";

    GGML_غیر_مستعمل(buft);
}

ساکن صحیح ggml_backend_sycl_host_بفر_type_device(ggml_backend_بفر_type_t buft) {
    واپس static_cast<ثابت ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
ساکن خالی * ggml_backend_sycl_host_malloc(صحیح device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    خالی * ptr = خالی_اشاریہ;
    کوشش {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        خودکار & q = dpct::dev_mgr::instance().get_device(device).default_قطار();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } پکڑو (...) {
        ptr = خالی_اشاریہ;
    }
    اگر (ptr == خالی_اشاریہ) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    واپس ptr;
}

ساکن خالی ggml_backend_sycl_host_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    اگر (بفر->context == خالی_اشاریہ) {
        واپس;
    }
    اگر (g_ggml_sycl_enable_host_pinned_mem) {
        ثابت صحیح device = ggml_backend_sycl_host_بفر_type_device(بفر->buft);
        خودکار & q = dpct::dev_mgr::instance().get_device(device).default_قطار();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(بفر->context, q)));
    } ورنہ {
        free_aligned_mem_host((خالی *) بفر->context);
    }
}

ساکن ggml_backend_بفر_t ggml_backend_sycl_host_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    خالی * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_بفر_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    اگر (ptr == خالی_اشاریہ) {
        // fallback to cpu buffer
        واپس ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(ptr, size);
    بفر->buft = buft;
    بفر->iface.free_بفر = ggml_backend_sycl_host_بفر_free_بفر;

    واپس بفر;
}

ساکن size_t ggml_backend_sycl_host_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {

    اگر (g_ggml_sycl_enable_host_pinned_mem) {
        ثابت صحیح device = ggml_backend_sycl_host_بفر_type_device(buft);
        size_t max_alloc_سائز = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_سائز();
        اگر (g_ggml_sycl_host_pinned_mem_2g) {
            واپس std::min(max_alloc_سائز, (size_t) 2LL*1024*1024*1024);
        } ورنہ {
            واپس max_alloc_سائز;
        }
    } ورنہ {
        واپس SIZE_MAX;
    }
}

ساکن ggml_backend_بفر_type_t ggml_backend_sycl_host_بفر_type_for_device(صحیح device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    ساکن std::vector<ggml_backend_بفر_type> بفر_types_host = [] {
        std::vector<ggml_backend_بفر_type> bufts(ggml_backend_sycl_get_device_گنتی());
        برائے (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_بفر_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_بفر_type_alloc_بفر,
                    /* .alloc_buffer_n      = */ تہی,
                    /* .get_alignment       = */ ggml_backend_cpu_بفر_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_بفر_type_get_max_سائز,
                    /* .get_alloc_size      = */ ggml_backend_cpu_بفر_type()->iface.get_alloc_سائز,
                    /* .get_alloc_size_n    = */ تہی,
                    /* .is_host             = */ ggml_backend_cpu_بفر_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ خالی_اشاریہ,
            };
        }
        واپس bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (صحیح) بفر_types_host.size());

    واپس &بفر_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_بفر_type_t ggml_backend_sycl_host_بفر_type() {
    واپس ggml_backend_sycl_host_بفر_type_for_device(0);
}

// buffer pool for sycl (legacy)
ساخت ggml_sycl_pool_leg : عوامی ggml_sycl_pool {
    ساکن ثابت صحیح MAX_SYCL_BUFFERS = 256;

    صحیح device;
    قطار_ptr qptr;
    ساخت ggml_sycl_بفر {
        خالی * ptr = خالی_اشاریہ;
        size_t size = 0;
    };

    ggml_sycl_بفر بفر_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_سائز = 0;

    صریح ggml_sycl_pool_leg(قطار_ptr qptr_, صحیح device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#اگر_تعریف_شدہ DEBUG_SYCL_POOL
        صحیح    n_cached    = 0;
        size_t بائٹیں_cached = 0;
        برائے (صحیح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            اگر (بفر_pool[i].ptr != خالی_اشاریہ) {
                ++n_cached;
                بائٹیں_cached += بفر_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, بائٹیں_cached / 1024.0 / 1024.0);
        ثابت خودکار slots = format_slots_اندر_alloc_ترتیب();
        اگر (!slots.خالی_خانے()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#اگر_ختم

        برائے (صحیح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            اگر (b.ptr != خالی_اشاریہ) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_سائز -= b.size;
            }
        }
        GGML_ASSERT(pool_سائز == 0);
    }

#اگر_تعریف_شدہ DEBUG_SYCL_POOL
    std::string format_slots_اندر_alloc_ترتیب() ثابت {
        std::string سطر;
        حرف buf[32];
        منطقی پہلا = سچ;
        برائے (صحیح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            اگر (بفر_pool[i].ptr == خالی_اشاریہ) {
                جاری;
            }
            اگر (!پہلا) {
                سطر += '/';
            }
            پہلا = جھوٹ;
            snprintf(buf, حجم(buf), "%.2f", بفر_pool[i].size / 1024.0 / 1024.0);
            سطر += buf;
        }
        واپس سطر;
    }
#اگر_ختم

    خالی * alloc(size_t size, size_t * actual_سائز) override {
#اگر_تعریف_شدہ DEBUG_sycl_MALLOC
        صحیح nnz = 0;
        size_t max_size = 0;
#اگر_ختم
        size_t بہترین_diff = 1ull << 36;
        صحیح ibest = -1;
        برائے (صحیح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر& b = بفر_pool[i];
            اگر (b.ptr != خالی_اشاریہ) {
#اگر_تعریف_شدہ DEBUG_sycl_MALLOC
                ++nnz;
                اگر (b.size > max_size) max_size = b.size;
#اگر_ختم
                اگر (b.size >= size) {
                    size_t diff = b.size - size;
                    اگر (diff < بہترین_diff) {
                        بہترین_diff = diff;
                        ibest = i;
                        اگر (!بہترین_diff) {
                            خالی * ptr = b.ptr;
                            *actual_سائز = b.size;
                            b.ptr = خالی_اشاریہ;
                            b.size = 0;
                            واپس ptr;
                        }
                    }
                }
            }
        }
        اگر (ibest >= 0) {
            ggml_sycl_بفر& b = بفر_pool[ibest];
            خالی * ptr = b.ptr;
            *actual_سائز = b.size;
            b.ptr = خالی_اشاریہ;
            b.size = 0;
            واپس ptr;
        }
        خالی * ptr;
        size_t look_ahead_سائز = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (خالی *)ggml_sycl_malloc_device(look_ahead_سائز, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        اگر (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_سائز);
            واپس خالی_اشاریہ;
        }

        *actual_سائز = look_ahead_سائز;
        pool_سائز += look_ahead_سائز;

#اگر_تعریف_شدہ DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_سائز[id]/1024/1024), (uint32_t)(size/1024/1024));
#اگر_ختم

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        واپس ptr;
    }

    خالی آزاد(خالی * ptr, size_t size) override {
        برائے (صحیح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر& b = بفر_pool[i];
            اگر (b.ptr == خالی_اشاریہ) {
                b.ptr = ptr;
                b.size = size;
                واپس;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_سائز -= size;
    }
};

// pool with virtual memory management
#اگر defined(GGML_SYCL_SUPPORT_VMM)
ساخت ggml_sycl_pool_vmm : عوامی ggml_sycl_pool {
    ساکن ثابت size_t SYCL_POOL_VMM_MAX_سائز = 1ull << 35; // 32 GB

    صحیح           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_مستعمل = 0;
    size_t    pool_سائز = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    ساخت mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        خالی * map_ptr;
    };
    std::vector<mapping> mappings;

    صریح ggml_sycl_pool_vmm(قطار_ptr qptr_, صحیح device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        اگر (pool_addr == 0) {
            واپس;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        برائے (خودکار & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_سائز, ctx)));
    }

    خالی * alloc(size_t size, size_t * actual_سائز) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_بفر_ALIGNMENT);

        size_t avail = pool_سائز - pool_مستعمل;

        اگر (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_سائز = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_سائز + reserve_سائز <= SYCL_POOL_VMM_MAX_سائز);

            اگر (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_لیبل " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_سائز / (1024 * 1024), pool_سائز / (1024 * 1024),
                              (pool_سائز + reserve_سائز) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_سائز)));

            // reserve virtual address space (if not already reserved)
            اگر (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_سائز, ctx)));
            }

            // map at the end of the pool
            خالی * map_ptr = خالی_اشاریہ;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_سائز, reserve_سائز,
                                    sycl::ext::oneapi::experimental::پتہ_access_منوال::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_سائز += reserve_سائز;
            ggml_sycl_memtrace_جوڑو(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_سائز);

#اگر_تعریف_شدہ DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (غیرعلامتی طویل طویل) (pool_سائز/1024/1024),
                          (غیرعلامتی طویل طویل) (reserve_سائز/1024/1024));
#اگر_ختم
        }

        GGML_ASSERT(pool_addr != 0);

        خالی * ptr = reinterpret_cast<خالی *>(pool_addr + pool_مستعمل);
        *actual_سائز = size;
        pool_مستعمل += size;

#اگر_تعریف_شدہ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (غیرعلامتی طویل طویل) size, ptr);
#اگر_ختم

        واپس ptr;
    }

    خالی آزاد(خالی * ptr, size_t size) override {
#اگر_تعریف_شدہ DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (غیرعلامتی طویل طویل) size, ptr);
#اگر_ختم

        pool_مستعمل -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<خالی *>(pool_addr + pool_مستعمل));
    }
};
#اگر_ختم // defined(GGML_SYCL_SUPPORT_VMM)

ساخت ggml_sycl_pool_host : عوامی ggml_sycl_pool {
    قطار_ptr qptr;
    صحیح       device;

    ساخت ggml_sycl_بفر {
        خالی * ptr  = خالی_اشاریہ;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    ساکن constexpr صحیح          MAX_POOL_سائز{ 64 };
    std::vector<ggml_sycl_بفر> بفر_pool = std::vector<ggml_sycl_بفر>(MAX_POOL_سائز);
    size_t                        pool_سائز   = 0;
    صحیح                           شمار_2     = 0;

    صریح ggml_sycl_pool_host(قطار_ptr qptr_, صحیح device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        برائے (صحیح i = 0; i < MAX_POOL_سائز; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            اگر (b.ptr != خالی_اشاریہ) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(b.ptr, *qptr)));
                b.ptr = خالی_اشاریہ;
                pool_سائز -= b.size;
                b.size = 0;
            }
        }
    }

    خالی * alloc(size_t size, size_t * actual_سائز) override {
        اگر (شمار_2 == MAX_POOL_سائز) {
            // the pool wrapped around, no slot is reserved for this allocation
            خالی * ptr = خالی_اشاریہ;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (خالی *) sycl::malloc_host(size, *qptr)));
            اگر (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                واپس خالی_اشاریہ;
            }
            pool_سائز    += size;
            *actual_سائز = size;
            شمار_2      = 0;
            واپس ptr;
        }
        ggml_sycl_بفر & b = بفر_pool[شمار_2];
        ++شمار_2;

        اگر (b.ptr != خالی_اشاریہ) {
            اگر (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                خالی * ptr = b.ptr;
                *actual_سائز = b.size;
                b.ptr = خالی_اشاریہ;
                b.size = 0;
                واپس ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(b.ptr, *qptr)));
            pool_سائز -= b.size;
            b.ptr = خالی_اشاریہ;
            b.size = 0;
        }

        خالی * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (خالی *) sycl::malloc_host(size, *qptr)));
        اگر (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            واپس خالی_اشاریہ;
        }
        pool_سائز    += size;
        *actual_سائز = size;
        واپس ptr;
    }

    خالی آزاد(خالی * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        برائے (صحیح i = 0; i < MAX_POOL_سائز; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            اگر (b.ptr == خالی_اشاریہ) {
                b.ptr  = ptr;
                b.size = size;
                واپس;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::آزاد(ptr, *qptr)));
        pool_سائز -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(قطار_ptr qptr, صحیح device) {
    // return pool for the host to speed up memory management
    واپس std::unique_ptr<ggml_sycl_pool>(نیا ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(قطار_ptr qptr, صحیح device) {
#اگر defined(GGML_SYCL_SUPPORT_VMM)
    اگر (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        واپس std::unique_ptr<ggml_sycl_pool>(نیا ggml_sycl_pool_vmm(qptr, device));
    }
#اگر_ختم // defined(GGML_SYCL_SUPPORT_VMM)
    واپس std::unique_ptr<ggml_sycl_pool>(نیا ggml_sycl_pool_leg(qptr, device));
}

/// kernels
قسمتعریف خالی (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت حرف *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t صف_کم, ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائز,
    ثابت قطار_ptr &stream);



ساکن خالی mul_mat_p021_f16_f32(
    ثابت خالی * __restrict__ vx, ثابت اعشاری * __restrict__ y, اعشاری * __restrict__ dst,
    ثابت صحیح ncols_x, ثابت صحیح nrows_x, ثابت صحیح nchannels_x, ثابت صحیح nchannels_y,
    ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    ثابت صحیح صف_x = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                      شے_ct1.get_local_id(1);
    ثابت صحیح channel = شے_ct1.get_local_range(0) * شے_ct1.get_گروہ(0) +
                        شے_ct1.get_local_id(0);
    ثابت صحیح channel_x = channel / (nchannels_y / nchannels_x);

    ثابت صحیح nrows_y = ncols_x;
    ثابت صحیح nrows_dst = nrows_x;
    ثابت صحیح صف_dst = صف_x;

    اعشاری tmp = 0.0f;

    برائے (صحیح ستون_x0 = 0; ستون_x0 < ncols_x;
         ستون_x0 += شے_ct1.get_local_range(2)) {
        ثابت صحیح ستون_x = ستون_x0 + شے_ct1.get_local_id(2);

        اگر (ستون_x >= ncols_x) {
            توڑو;
        }

        // x is transposed and permuted
        ثابت صحیح ix = صف_x*nchannels_x*ncols_x + channel_x*ncols_x + ستون_x;
        ثابت اعشاری xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<اعشاری, sycl::rounding_منوال::automatic>()[0];

        ثابت صحیح صف_y = ستون_x;


        // y is not transposed but permuted
        ثابت صحیح iy = channel*nrows_y + صف_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ثابت صحیح idst = channel*nrows_dst + صف_dst;

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ساکن خالی mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ثابت خالی * __restrict__ vx, ثابت اعشاری * __restrict__ y, اعشاری * __restrict__ dst, ثابت صحیح ncols_x, ثابت صحیح nrows_x,
    ثابت صحیح صف_stride_x, ثابت صحیح channel_stride_x,ثابت صحیح channel_stride_y, ثابت صحیح channel_x_مقسوم_علیہ,
    ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    ثابت صحیح صف_x = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                      شے_ct1.get_local_id(1);
    ثابت صحیح channel = شے_ct1.get_local_range(0) * شے_ct1.get_گروہ(0) +
                        شے_ct1.get_local_id(0);
    ثابت صحیح channel_x = channel / channel_x_مقسوم_علیہ;

    ثابت صحیح nrows_dst = nrows_x;
    ثابت صحیح صف_dst   = صف_x;

    ثابت صحیح idst = channel*nrows_dst + صف_dst;

    اعشاری tmp = 0.0f;

    برائے (صحیح ستون_x0 = 0; ستون_x0 < ncols_x;
         ستون_x0 += شے_ct1.get_local_range(2)) {
        ثابت صحیح ستون_x = ستون_x0 + شے_ct1.get_local_id(2);

        اگر (ستون_x >= ncols_x) {
            توڑو;
        }

        ثابت صحیح صف_y = ستون_x;

        ثابت صحیح ix = channel_x*channel_stride_x + صف_x*صف_stride_x + ستون_x;
        ثابت صحیح iy = channel * channel_stride_y + صف_y;

        ثابت اعشاری xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<اعشاری, sycl::rounding_منوال::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#ہدایت unroll
    برائے (صحیح نقاب = WARP_سائز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتیب_بدلو_sub_گروہ_by_xor(شے_ct1.get_sub_گروہ(), tmp, نقاب);
    }

    اگر (شے_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ساکن خالی k_مجموع_صفیں_f32(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols,
                           ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح صف = شے_ct1.get_گروہ(1);
    ثابت صحیح ستون = شے_ct1.get_local_id(2);

    اعشاری مجموع = 0.0f;
    برائے (صحیح i = ستون; i < ncols; i += شے_ct1.get_local_range(2)) {
        مجموع += x[صف * ncols + i];
    }

    مجموع = warp_reduce_مجموع(مجموع, شے_ct1);

    اگر (ستون == 0) {
        dst[صف] = مجموع;
    }
}


سانچہ<قسم_نام T>
ساکن اندرونی خالی ggml_sycl_تبادلہ(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

سانچہ <ggml_ترتیب_دو_ترتیب ترتیب>
__dpct_inline__ ساکن خالی
k_argsort_f32_i32(ثابت اعشاری *x, صحیح *dst, ثابت صحیح ncols, صحیح ncols_pad,
                  ثابت صحیح tasks_فی_thread, ثابت sycl::nd_شے<3> &شے_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    صحیح ستون_اشاریہ =  شے_ct1.get_local_id(2);
    صحیح صف = شے_ct1.get_گروہ(1);

    برائے (صحیح i = 0; i < tasks_فی_thread; i++) {
        صحیح ستون = ستون_اشاریہ * tasks_فی_thread + i;
        اگر (ستون >= ncols_pad) {
            واپس;
        }
    }

    ثابت اعشاری * x_صف = x + صف * ncols;
    خودکار dst_صف = (صحیح *)dpct_local;

    // initialize indices
    برائے (صحیح i=0;i<tasks_فی_thread;i++){
        صحیح ستون = ستون_اشاریہ*tasks_فی_thread+i;
        dst_صف[ستون] = ستون;
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    برائے (صحیح k = 2; k <= ncols_pad; k *= 2) {
        برائے (صحیح j = k / 2; j > 0; j /= 2) {
            برائے (صحیح i = 0; i < tasks_فی_thread; i++) {
                صحیح ستون = ستون_اشاریہ * tasks_فی_thread + i;
                صحیح ixj = ستون ^ j;
                اگر (ixj > ستون) {
                    اگر ((ستون & k) == 0) {
                        اگر (dst_صف[ستون] >= ncols ||
                            (dst_صف[ixj] < ncols &&
                             (ترتیب == GGML_ترتیب_دو_ترتیب_ASC
                                  ? x_صف[dst_صف[ستون]] > x_صف[dst_صف[ixj]]
                                  : x_صف[dst_صف[ستون]] <
                                        x_صف[dst_صف[ixj]]))) {
                            ggml_sycl_تبادلہ(dst_صف[ستون], dst_صف[ixj]);
                        }
                    } ورنہ {
                        اگر (dst_صف[ixj] >= ncols ||
                            (dst_صف[ستون] < ncols &&
                             (ترتیب == GGML_ترتیب_دو_ترتیب_ASC
                                  ? x_صف[dst_صف[ستون]] < x_صف[dst_صف[ixj]]
                                  : x_صف[dst_صف[ستون]] >
                                        x_صف[dst_صف[ixj]]))) {
                            ggml_sycl_تبادلہ(dst_صف[ستون], dst_صف[ixj]);
                        }
                    }
                }
                شے_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    برائے (صحیح i = 0; i < tasks_فی_thread; i++) {
        صحیح ستون = ستون_اشاریہ * tasks_فی_thread + i;
        اگر (ستون < ncols) {
            dst[صف * ncols + ستون] = dst_صف[ستون];
        }
    }
}

ساکن خالی diag_نقاب_inf_f32(ثابت اعشاری * x, اعشاری * dst, ثابت صحیح ncols, ثابت صحیح صفیں_فی_channel, ثابت صحیح n_past,
                              ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح ستون = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                    شے_ct1.get_local_id(1);
    ثابت صحیح صف = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                    شے_ct1.get_local_id(2);

    اگر (ستون >= ncols) {
        واپس;
    }

    ثابت صحیح i = صف*ncols + ستون;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (ستون > n_past + صف % صفیں_فی_channel) * FLT_MAX;
}

ساکن خالی scale_f32(ثابت اعشاری * x, اعشاری * dst, ثابت اعشاری scale, ثابت اعشاری bias, ثابت صحیح k,
                      ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت صحیح i = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ(2) +
                  شے_ct1.get_local_id(2);

    اگر (i >= k) {
        واپس;
    }

    dst[i] = scale * x[i] + bias;
}


ساکن خالی ggml_mul_mat_p021_f16_f32_sycl(ثابت خالی *vx, ثابت اعشاری *y,
                                           اعشاری *dst, ثابت صحیح ncols_x,
                                           ثابت صحیح nrows_x,
                                           ثابت صحیح nchannels_x,
                                           ثابت صحیح nchannels_y,
                                           قطار_ptr stream) {

    ثابت sycl::range<3> بلاک_nums(nchannels_y, nrows_x, 1);
    ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, شے_ct1);
            });
    }
}

ساکن خالی ggml_mul_mat_vec_nc_f16_f32_sycl(
    ثابت خالی *vx, ثابت اعشاری *y, اعشاری *dst, ثابت صحیح ncols_x,
    ثابت صحیح nrows_x, ثابت صحیح صف_stride_x, ثابت صحیح nchannels_x,
    ثابت صحیح nchannels_y, ثابت صحیح channel_stride_x, ثابت صحیح channel_stride_y, قطار_ptr stream) {

    ثابت sycl::range<3> بلاک_nums(nchannels_y, nrows_x, 1);
    ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       صف_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, شے_ct1);
            });
    }
}



ساکن خالی scale_f32_sycl(ثابت اعشاری *x, اعشاری *dst, ثابت اعشاری scale, ثابت اعشاری bias,
                           ثابت صحیح k, قطار_ptr stream) {
    ثابت صحیح عدد_بلاکیں = (k + SYCL_SCALE_بلاک_سائز - 1) / SYCL_SCALE_بلاک_سائز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, عدد_بلاکیں) *
                              sycl::range<3>(1, 1, SYCL_SCALE_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_SCALE_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) {
            scale_f32(x, dst, scale, bias, k, شے_ct1);
        });
}


ساکن خالی مجموع_صفیں_f32_sycl(ثابت اعشاری *x, اعشاری *dst, ثابت صحیح ncols,
                              ثابت صحیح nrows, قطار_ptr stream) {
    ثابت sycl::range<3> بلاک_dims(1, 1, WARP_سائز);
    ثابت sycl::range<3> بلاک_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1)
                             [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
                                 k_مجموع_صفیں_f32(x, dst, ncols, شے_ct1);
                             });
}

ساکن صحیح اگلا_power_کا_2(صحیح x) {
    صحیح n = 1;
    جب (n < x) {
        n *= 2;
    }
    واپس n;
}

ساکن خالی init_argsort_indices_padded(
        صحیح * idx,
        ثابت صحیح nrows,
        ثابت صحیح ncols_pad,
        ثابت sycl::nd_شے<1> & شے_ct1) {
    ثابت size_t gid = شے_ct1.get_local_range(0) * شے_ct1.get_گروہ(0) + شے_ct1.get_local_id(0);
    ثابت size_t کل = (size_t) nrows * (size_t) ncols_pad;

    اگر (gid >= کل) {
        واپس;
    }

    idx[gid] = (صحیح) (gid % (size_t) ncols_pad);
}

سانچہ <ggml_ترتیب_دو_ترتیب ترتیب>
ساکن خالی argsort_f32_i32_global_pass(ثابت اعشاری *            x,
                                        صحیح *                    idx,
                                        ثابت صحیح                ncols,
                                        ثابت صحیح                nrows,
                                        ثابت صحیح                ncols_pad,
                                        ثابت صحیح                j,
                                        ثابت صحیح                k,
                                        ثابت sycl::nd_شے<1> & شے_ct1) {
    ثابت size_t gid   = شے_ct1.get_local_range(0) * شے_ct1.get_گروہ(0) + شے_ct1.get_local_id(0);
    ثابت size_t کل = (size_t) nrows * (size_t) ncols_pad;

    اگر (gid >= کل) {
        واپس;
    }

    ثابت صحیح صف = (صحیح) (gid / (size_t) ncols_pad);
    ثابت صحیح ستون = (صحیح) (gid % (size_t) ncols_pad);
    ثابت صحیح ixj = ستون ^ j;

    اگر (ixj <= ستون || ixj >= ncols_pad) {
        واپس;
    }

    ثابت size_t بنیاد  = (size_t) صف * (size_t) ncols_pad;
    ثابت size_t pos_a = بنیاد + (size_t) ستون;
    ثابت size_t pos_b = بنیاد + (size_t) ixj;

    ثابت صحیح a = idx[pos_a];
    ثابت صحیح b = idx[pos_b];

    منطقی do_تبادلہ = جھوٹ;

    اگر ((ستون & k) == 0) {
        اگر (a >= ncols ||
            (b < ncols &&
             (ترتیب == GGML_ترتیب_دو_ترتیب_ASC ?
                  x[(size_t) صف * (size_t) ncols + (size_t) a] > x[(size_t) صف * (size_t) ncols + (size_t) b] :
                  x[(size_t) صف * (size_t) ncols + (size_t) a] < x[(size_t) صف * (size_t) ncols + (size_t) b]))) {
            do_تبادلہ = سچ;
        }
    } ورنہ {
        اگر (b >= ncols ||
            (a < ncols &&
             (ترتیب == GGML_ترتیب_دو_ترتیب_ASC ?
                  x[(size_t) صف * (size_t) ncols + (size_t) a] < x[(size_t) صف * (size_t) ncols + (size_t) b] :
                  x[(size_t) صف * (size_t) ncols + (size_t) a] > x[(size_t) صف * (size_t) ncols + (size_t) b]))) {
            do_تبادلہ = سچ;
        }
    }

    اگر (do_تبادلہ) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

ساکن خالی copy_argsort_indices_unpadded(ثابت صحیح *              idx_padded,
                                          صحیح *                    dst,
                                          ثابت صحیح                nrows,
                                          ثابت صحیح                ncols,
                                          ثابت صحیح                ncols_pad,
                                          ثابت sycl::nd_شے<1> & شے_ct1) {
    ثابت size_t gid   = شے_ct1.get_local_range(0) * شے_ct1.get_گروہ(0) + شے_ct1.get_local_id(0);
    ثابت size_t کل = (size_t) nrows * (size_t) ncols;

    اگر (gid >= کل) {
        واپس;
    }

    ثابت صحیح صف = (صحیح) (gid / (size_t) ncols);
    ثابت صحیح ستون = (صحیح) (gid % (size_t) ncols);

    dst[(size_t) صف * (size_t) ncols + (size_t) ستون] = idx_padded[(size_t) صف * (size_t) ncols_pad + (size_t) ستون];
}

ساکن خالی argsort_f32_i32_sycl(ثابت اعشاری *x, صحیح *dst, ثابت صحیح ncols,
                                 ثابت صحیح nrows, ggml_ترتیب_دو_ترتیب ترتیب,
                                 قطار_ptr stream, صحیح device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ثابت صحیح ncols_pad = اگلا_power_کا_2(ncols);
    ثابت size_t shared_mem = (size_t) ncols_pad * حجم(صحیح);
    ثابت size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    اگر (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<صحیح> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        صحیح *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t بلاک_سائز_3     = 256;
        ثابت size_t     کل_padded   = (size_t) nrows * (size_t) ncols_pad;
        ثابت size_t     nblocks_padded = (کل_padded + بلاک_سائز_3 - 1) / بلاک_سائز_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
            [=](sycl::nd_شے<1> شے_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, شے_ct1); });

        برائے (صحیح k = 2; k <= ncols_pad; k *= 2) {
            برائے (صحیح j = k / 2; j > 0; j /= 2) {
                اگر (ترتیب == GGML_ترتیب_دو_ترتیب_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
                        [=](sycl::nd_شے<1> شے_ct1) {
                            argsort_f32_i32_global_pass<GGML_ترتیب_دو_ترتیب_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, شے_ct1);
                        });
                } ورنہ اگر (ترتیب == GGML_ترتیب_دو_ترتیب_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
                        [=](sycl::nd_شے<1> شے_ct1) {
                            argsort_f32_i32_global_pass<GGML_ترتیب_دو_ترتیب_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, شے_ct1);
                        });
                } ورنہ {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ثابت size_t کل   = (size_t) nrows * (size_t) ncols;
        ثابت size_t nblocks = (کل + بلاک_سائز_3 - 1) / بلاک_سائز_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
                             [=](sycl::nd_شے<1> شے_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, شے_ct1);
                             });

        واپس;
    }

    صحیح nth = 1;
    صحیح max_بلاک_سائز = ggml_sycl_info().max_work_گروہ_sizes[device];
    جب (nth < ncols_pad && nth < max_بلاک_سائز)
        nth *= 2;
    اگر (nth > max_بلاک_سائز)
        nth = max_بلاک_سائز;

    ثابت صحیح tasks_فی_thread = ncols_pad / nth;

    ثابت sycl::range<3> بلاک_dims(1, 1, nth);
    ثابت sycl::range<3> بلاک_nums(1, nrows, 1);

    اگر (ترتیب == GGML_ترتیب_دو_ترتیب_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    k_argsort_f32_i32<GGML_ترتیب_دو_ترتیب_ASC>(
                        x, dst, ncols, ncols_pad, tasks_فی_thread, شے_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ورنہ اگر (ترتیب == GGML_ترتیب_دو_ترتیب_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<3> شے_ct1) {
                    k_argsort_f32_i32<GGML_ترتیب_دو_ترتیب_DESC>(
                        x, dst, ncols, ncols_pad, tasks_فی_thread, شے_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } ورنہ {
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
ساکن خالی چوٹی_k_جائزہ_ملاؤ_f32(
    ثابت اعشاری *   src_vals,
    ثابت int32_t * src_map,
    ثابت صحیح       begin,
    ثابت صحیح       end,
    ثابت صحیح       k,
    ثابت صحیح       بلاک_سائز_3,
    اعشاری *         shared_vals,
    صحیح *           shared_idx,
    اعشاری *         برآمد_vals,
    int32_t *       برآمد_idx,
    ثابت منطقی      swap01,
    ثابت sycl::nd_شے<1> & شے_ct1
) {
    ثابت صحیح tid = شے_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    اعشاری * lv = shared_vals + tid;
    صحیح * li = shared_idx + tid;

    برائے (صحیح i = 0; i < k; i++) {
        lv[i * بلاک_سائز_3] = -FLT_MAX;
        li[i * بلاک_سائز_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    اعشاری kth = -FLT_MAX;

    برائے (صحیح ستون = begin + tid; ستون < end; ستون += بلاک_سائز_3) {
        اعشاری val = src_vals[ستون];

        اگر (val > kth) {
            صحیح pos = k - 1;
            جب (pos > 0 && val > lv[(pos - 1) * بلاک_سائز_3]) {
                pos--;
            }

            برائے (صحیح i = k - 1; i > pos; i--) {
                lv[i * بلاک_سائز_3] = lv[(i - 1) * بلاک_سائز_3];
                li[i * بلاک_سائز_3] = li[(i - 1) * بلاک_سائز_3];
            }
            lv[pos * بلاک_سائز_3] = val;
            li[pos * بلاک_سائز_3] = src_map ? src_map[ستون] : ستون;

            kth = lv[(k - 1) * بلاک_سائز_3];
        }
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    اگر (tid != 0) {
        واپس;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    اعشاری * fv = shared_vals + (size_t) k * بلاک_سائز_3;
    صحیح * fi = shared_idx + (size_t) k * بلاک_سائز_3;

    برائے (صحیح i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    اعشاری fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    برائے (صحیح t = 0; t < بلاک_سائز_3; t++) {
        برائے (صحیح i = 0; i < k; i++) {
            اعشاری val = shared_vals[i * بلاک_سائز_3 + t];

            اگر (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                توڑو;
            }

            صحیح idx = shared_idx[i * بلاک_سائز_3 + t];

            صحیح pos = k - 1;
            جب (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            برائے (صحیح j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    اگر (برآمد_vals) {
        برائے (صحیح i = 0; i < k; i++) {
            برآمد_vals[i] = fv[i];
        }
    }

    برائے (صحیح i = 0; i < k; i++) {
        برآمد_idx[i] = fi[i];
    }

    اگر (swap01 && k > 1) {
        int32_t عارضی = برآمد_idx[0];
        برآمد_idx[0] = برآمد_idx[1];
        برآمد_idx[1] = عارضی;
    }
}

ساکن خالی چوٹی_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت اعشاری * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحیح k,
    dpct::قطار_ptr main_دھارا
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
    constexpr صحیح بانٹو_بلاک = 128;
    constexpr صحیح max_splits  = 128;
    constexpr صحیح min_ستون_تعداد    = 8192;

    صحیح nsplit = 1;
    اگر (ncols >= min_ستون_تعداد) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ثابت int64_t want = ncols / بانٹو_بلاک;
        nsplit = (صحیح) (want > max_splits ? max_splits : want);
    }

    اگر (nsplit > 1) {
        ثابت صحیح nchunk = (صحیح) ((ncols + nsplit - 1) / nsplit);
        ثابت size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<اعشاری>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        اعشاری *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ثابت sycl::range<1> بلاک_dims(بانٹو_بلاک);

        main_دھارا->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<اعشاری, 1> shared_vals(sycl::range<1>((بانٹو_بلاک + 1) * k), cgh);
            sycl::local_accessor<صحیح, 1> shared_idx(sycl::range<1>((بانٹو_بلاک + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<1> شے_ct1) {
                    ثابت صحیح grp = شے_ct1.get_گروہ(0);
                    ثابت صحیح صف = grp / nsplit;
                    ثابت صحیح part = grp % nsplit;

                    ثابت صحیح begin = part * nchunk;
                    صحیح end = begin + nchunk;
                    اگر (end > (صحیح) ncols) {
                        end = (صحیح) ncols;
                    }

                    چوٹی_k_جائزہ_ملاؤ_f32(
                        src + (int64_t) صف * ncols, خالی_اشاریہ, begin, end, k, بانٹو_بلاک,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, جھوٹ, شے_ct1);
                });
        });

        main_دھارا->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<اعشاری, 1> shared_vals(sycl::range<1>((بانٹو_بلاک + 1) * k), cgh);
            sycl::local_accessor<صحیح, 1> shared_idx(sycl::range<1>((بانٹو_بلاک + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * بلاک_dims, بلاک_dims),
                [=](sycl::nd_شے<1> شے_ct1) {
                    ثابت صحیح صف = شے_ct1.get_گروہ(0);
                    ثابت size_t off = (size_t) صف * nsplit * k;

                    چوٹی_k_جائزہ_ملاؤ_f32(
                        pv + off, pi + off, 0, nsplit * k, k, بانٹو_بلاک,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        خالی_اشاریہ, dst_indices + (int64_t) صف * k, سچ, شے_ct1);
                });
        });

        واپس;
    }

    ثابت صحیح بلاک_سائز_3 = 128;

    ثابت sycl::range<1> بلاک_dims(بلاک_سائز_3);
    ثابت sycl::range<1> grid_dims(nrows);

    main_دھارا->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<اعشاری, 1> shared_vals(sycl::range<1>((بلاک_سائز_3 + 1) * k), cgh);
        sycl::local_accessor<صحیح, 1> shared_idx(sycl::range<1>((بلاک_سائز_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<1> شے_ct1) {
                ثابت صحیح صف = شے_ct1.get_گروہ(0);

                اگر (صف >= nrows) واپس;

                چوٹی_k_جائزہ_ملاؤ_f32(
                    src + (int64_t) صف * ncols, خالی_اشاریہ, 0, (صحیح) ncols, k, بلاک_سائز_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    خالی_اشاریہ, dst_indices + (int64_t) صف * k, سچ, شے_ct1);
            });
    });
}

ساکن خالی argmax_f32_i32_sycl(ثابت اعشاری *x, صحیح *dst, ثابت صحیح ncols,
                               ثابت صحیح nrows, قطار_ptr stream) {
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_ARGMAX_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(1, nrows, 1);
    ثابت size_t shared_mem = 256 * حجم(اعشاری);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<اعشاری, 1> shared_ڈیٹا(
            sycl::range<1>(shared_mem/حجم(اعشاری)), cgh);
        sycl::local_accessor<صحیح, 1> shared_indices(
            sycl::range<1>(shared_mem/حجم(اعشاری)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                ثابت صحیح tid = شے_ct1.get_local_id(2);
                ثابت صحیح صف = شے_ct1.get_global_id(1);

                اعشاری max_val = -INFINITY;
                صحیح max_idx = -1;

                برائے (صحیح ستون = tid; ستون < ncols; ستون += 256) {
                    اعشاری val = x[صف * ncols + ستون];
                    اگر (val > max_val) {
                        max_val = val;
                        max_idx = ستون;
                    }
                }

                shared_ڈیٹا[tid] = max_val;
                shared_indices[tid] = max_idx;
                شے_ct1.barrier(sycl::access::fence_space::local_space);

                برائے (صحیح stride = 256/2; stride > 0; stride >>= 1) {
                    اگر (tid < stride) {
                        اعشاری val1 = shared_ڈیٹا[tid];
                        اعشاری val2 = shared_ڈیٹا[tid + stride];
                        اگر (val2 > val1) {
                            shared_ڈیٹا[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    شے_ct1.barrier(sycl::access::fence_space::local_space);
                }


                اگر (tid == 0) {
                    dst[صف] = shared_indices[0];
                }
            });
    });
}
ساکن خالی diag_نقاب_inf_f32_sycl(ثابت اعشاری *x, اعشاری *dst,
                                   ثابت صحیح ncols_x, ثابت صحیح nrows_x,
                                   ثابت صحیح صفیں_فی_channel, ثابت صحیح n_past,
                                   قطار_ptr stream) {
    ثابت sycl::range<3> بلاک_dims(1, SYCL_DIAG_نقاب_INF_بلاک_سائز, 1);
    ثابت صحیح بلاک_عدد_x = (ncols_x + SYCL_DIAG_نقاب_INF_بلاک_سائز - 1) / SYCL_DIAG_نقاب_INF_بلاک_سائز;
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             diag_نقاب_inf_f32(x, dst, ncols_x,
                                               صفیں_فی_channel, n_past,
                                               شے_ct1);
                         });
}

ساکن dpct::err0 ggml_sycl_cpy_tensor_2d(خالی *dst,
                                          ثابت ساخت ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_کم, int64_t i1_بلند,
                                          قطار_ptr stream) کوشش {

    dpct::memcpy_direction قسم_2;
    حرف * src_ptr;
    اگر (ggml_backend_بفر_is_host(src->بفر)) {
        قسم_2 = dpct::host_مقصد_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (حرف *) src->ڈیٹا;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } ورنہ اگر (ggml_backend_بفر_is_sycl(src->بفر)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        قسم_2    = dpct::device_مقصد_device;
        src_ptr = (حرف *) src->ڈیٹا;
    } ورنہ اگر (ggml_backend_بفر_is_sycl_بانٹو(src->بفر)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_کم == 0 && i1_بلند == src->ne[1]);
        قسم_2 = dpct::device_مقصد_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        صحیح id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (حرف *) extra->ڈیٹا_device[id];
    } ورنہ {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    حرف * dst_ptr = (حرف *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ثابت شمار ggml_type type = src->type;
    ثابت int64_t ts = ggml_type_سائز(type);
    ثابت int64_t bs = ggml_blck_سائز(type);
    int64_t i1_diff = i1_بلند - i1_کم;

    ثابت حرف * x = src_ptr + i1_کم*nb1 + i2*nb2 + i3*nb3;
    اگر (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        واپس CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    قسم_2, *stream));

    } ورنہ اگر (nb0 == ts) {
        واپس CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, قسم_2, *stream));
    } ورنہ {
        برائے (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ثابت خالی * rx = (ثابت خالی *) ((ثابت حرف *) x + i1*nb1);
            خالی * rd = (خالی *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, قسم_2, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            اگر (r != 0) واپس r;
        }
        واپس 0;
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

اندرونی خالی ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت اعشاری *src1_ddf_i, ثابت حرف *src1_ddq_i,
    اعشاری *dst_dd_i, ثابت int64_t صف_کم, ثابت int64_t صف_بلند,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائز,
    ثابت قطار_ptr &stream) کوشش {

    GGML_ASSERT(src0_dd_i  != خالی_اشاریہ);
    GGML_ASSERT(src1_ddf_i != خالی_اشاریہ);
    GGML_ASSERT(dst_dd_i   != خالی_اشاریہ);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ثابت int64_t صف_diff = صف_بلند - صف_کم;

    صحیح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ثابت int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    صحیح ldc = id == ctx.device ? ne0 : صف_diff; // used by MKL only

#اگر_تعریف_شدہ GGML_SYCL_F16
    منطقی use_fp16 = سچ;  // TODO(Yu) SYCL capability check
#ورنہ
    منطقی use_fp16 = جھوٹ;
#اگر_ختم

#اگر GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    اگر (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        صف_diff == src0->ne[1]) {
        استعمال bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        اگر (src1->type != GGML_TYPE_BF16) {
            ثابت مقصد_bf16_sycl_t مقصد_bf16_sycl = ggml_get_مقصد_bf16_sycl(src1->type, dst);
            GGML_ASSERT(مقصد_bf16_sycl != خالی_اشاریہ);
            مقصد_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } ورنہ {
            stream->حافظہ_نقل(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*حجم(bf16_t));
        }
        DnnlGemmWrapper::صف_gemm(ctx, صف_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::مقصد_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::مقصد_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::مقصد_dt<اعشاری>(), stream);
        GGML_غیر_مستعمل(dst);
        GGML_غیر_مستعمل(src1_ddq_i);
        GGML_غیر_مستعمل(src1_padded_صف_سائز);
        واپس;
    }
#اگر_ختم

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    منطقی src0_reordered = جھوٹ;
    اگر (ثابت خودکار * src0_extra = static_cast<ثابت ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    اگر (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && صف_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     صف_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        واپس;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    اگر ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        صف_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        اگر (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ثابت مقصد_fp16_sycl_t مقصد_fp16_sycl = ggml_get_مقصد_fp16_sycl(src1->type, dst);
            GGML_ASSERT(مقصد_fp16_sycl != خالی_اشاریہ);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            مقصد_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ثابت sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ثابت sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        اگر (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ثابت مقصد_fp16_sycl_t مقصد_fp16_sycl = ggml_get_مقصد_fp16_sycl(src0->type, dst);
            GGML_ASSERT(مقصد_fp16_sycl != خالی_اشاریہ);
            size_t ne = صف_diff*ne00;
            src0_as_f16.alloc(ne);
            مقصد_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ثابت sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ثابت sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#اگر GGML_SYCL_DNNL
        اگر (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::صف_gemm(ctx,صف_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::مقصد_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::مقصد_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::مقصد_dt<اعشاری>(), stream);
        }
        ورنہ
#اگر_ختم
        {
            ثابت اعشاری alpha = 1.0f;
            ثابت اعشاری beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, صف_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_ڈیٹا_t::real_half, ne00,
                src1_ptr, dpct::library_ڈیٹا_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_ڈیٹا_t::real_float, ldc,
                dpct::library_ڈیٹا_t::real_float)));
        }
    } ورنہ {
        ggml_sycl_pool_alloc<اعشاری> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<اعشاری> src1_ddq_as_f32(ctx.pool());
        اگر (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ثابت مقصد_fp32_sycl_t مقصد_fp32_sycl = ggml_get_مقصد_fp32_sycl(src0->type, dst);
            GGML_ASSERT(مقصد_fp32_sycl != خالی_اشاریہ);
            src0_ddq_as_f32.alloc(صف_diff*ne00);
            مقصد_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), صف_diff*ne00, stream);
        }
        اگر (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ثابت مقصد_fp32_sycl_t مقصد_fp32_sycl = ggml_get_مقصد_fp32_sycl(src1->type, dst);
            GGML_ASSERT(مقصد_fp32_sycl != خالی_اشاریہ);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            مقصد_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ثابت اعشاری * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ثابت اعشاری *) src0_dd_i : src0_ddq_as_f32.get();
        ثابت اعشاری * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ثابت اعشاری *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#اگر GGML_SYCL_DNNL
            ثابت int64_t gemm_flops = (int64_t)صف_diff * src1_ncols * ne10;
            ثابت منطقی use_mkl_براہ_راست = gemm_flops < 256 * 256 * 256;
            اگر (g_ggml_sycl_enable_dnn && !use_mkl_براہ_راست && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::صف_gemm(ctx, صف_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::مقصد_dt<اعشاری>(), src1_ddf1_i, DnnlGemmWrapper::مقصد_dt<اعشاری>(),
                                          dst_dd_i, DnnlGemmWrapper::مقصد_dt<اعشاری>(), stream);
            }
            ورنہ
#اگر_ختم
            {
                ثابت اعشاری alpha = 1.0f;
                ثابت اعشاری beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::عمود_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, صف_diff,
                    src1_ncols, ne10, dpct::get_قدر(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_قدر(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(src1_ddq_i);
    GGML_غیر_مستعمل(src1_padded_صف_سائز);
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

اندرونی خالی ggml_sycl_op_مجموع(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    مجموع_صفیں_f32_sycl(src0_dd, dst_dd, ne, 1, main_دھارا);
}

اندرونی خالی ggml_sycl_op_مجموع_صفیں(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    مجموع_صفیں_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_دھارا);
}

اندرونی خالی ggml_sycl_op_اوسط(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    مجموع_صفیں_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_دھارا);

    main_دھارا->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> صف) {
            dst_dd[صف] /= ncols;
        }
    );
}


اندرونی خالی ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ڈیٹا);


    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);

    شمار ggml_ترتیب_دو_ترتیب ترتیب = (شمار ggml_ترتیب_دو_ترتیب) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (صحیح *)dst_dd, ncols, nrows, ترتیب,
                         main_دھارا, ctx.device, ctx.pool());
}

ساکن خالی ggml_sycl_op_چوٹی_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(src0->ڈیٹا);
    int32_t * dst_dd = static_cast<int32_t *>(dst->ڈیٹا);

    ثابت صحیح k = dst->ne[0];
    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    اگر (k <= SYCL_چوٹی_K_جائزہ_ملاؤ_MAX_K) {
        چوٹی_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_دھارا);
    } ورنہ {
        ggml_sycl_چوٹی_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_دھارا);
    }
}

اندرونی خالی ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->ڈیٹا);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_دھارا);
}

اندرونی خالی ggml_sycl_op_diag_نقاب_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت int64_t ne00 = dst->src[0]->ne[0];
    ثابت int64_t ne01 = dst->src[0]->ne[1];
    ثابت صحیح nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ثابت صحیح n_past = ((int32_t *) dst->op_params)[0];

    diag_نقاب_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_دھارا);
}

ساکن خالی tri_f32_sycl(
    ثابت اعشاری * src,
    اعشاری * dst,
    ثابت int64_t ne0,
    ثابت int64_t ne1,
    ثابت int64_t ne2,
    ثابت int64_t ne3,
    ثابت ggml_tri_type ttype,
    dpct::قطار_ptr main_دھارا
) {
    ثابت size_t کل = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_دھارا->parallel_for(sycl::range<1>(کل), [=](sycl::id<1> tid) {
        ثابت int64_t idx = (int64_t) tid[0];

        ثابت int64_t i0 = idx % ne0;
        ثابت int64_t t1 = idx / ne0;
        ثابت int64_t i1 = t1 % ne1;

        منطقی keep = جھوٹ;
        انتخاب (ttype) {
            حالت GGML_TRI_TYPE_زیریں:      keep = (i0 <  i1); توڑو;
            حالت GGML_TRI_TYPE_زیریں_DIAG: keep = (i0 <= i1); توڑو;
            حالت GGML_TRI_TYPE_بالائی:      keep = (i0 >  i1); توڑو;
            حالت GGML_TRI_TYPE_بالائی_DIAG: keep = (i0 >= i1); توڑو;
            طےشدہ: keep = جھوٹ; توڑو;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

ساکن خالی ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(src0->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ثابت int64_t ne0 = src0->ne[0];
    ثابت int64_t ne1 = src0->ne[1];
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_دھارا);
}


اندرونی خالی ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_دھارا = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    ثابت اعشاری * src0_dd = static_cast<ثابت اعشاری *>(dst->src[0]->ڈیٹا);
    اعشاری *       dst_dd  = static_cast<اعشاری *>(dst->ڈیٹا);

    اعشاری scale;
    اعشاری bias;
    حافظہ_نقل(&scale, (اعشاری *) dst->op_params + 0, حجم(اعشاری));
    حافظہ_نقل(&bias,  (اعشاری *) dst->op_params + 1, حجم(اعشاری));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_دھارا);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

ساکن خالی ggml_sycl_سیٹ_peer_access(ثابت صحیح n_علامات, [[maybe_unused]] صحیح main_device) {
    ساکن منطقی peer_access_enabled = جھوٹ;

    ثابت منطقی enable_peer_access = n_علامات <= GGML_SYCL_PEER_MAX_BATCH_سائز;

    اگر (peer_access_enabled == enable_peer_access) {
        واپس;
    }

#اگر_تعریف_شدہ NDEBUG
    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        SYCL_CHECK(ggml_sycl_سیٹ_device(i));
    }

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        SYCL_CHECK(ggml_sycl_سیٹ_device(i));

        برائے (صحیح id_other = 0; id_other < ggml_sycl_info().device_گنتی; ++id_other) {
            اگر (i == id_other) {
                جاری;
            }
            اگر (i != main_device && id_other != main_device) {
                جاری;
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
#اگر_ختم // NDEBUG

    peer_access_enabled = enable_peer_access;
}

سانچہ <سانچہ <صحیح> قسم_نام quantize_f>
ساکن خالی ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                 ثابت ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) کوشش {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ثابت int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];

    ثابت صحیح nb2 = dst->nb[2];
    ثابت صحیح nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(dst->بفر));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(src1->بفر));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ثابت int64_t i02_مقسوم_علیہ = ne12 / ne02;

    ثابت size_t src0_ts = ggml_type_سائز(src0->type);
    ثابت size_t src0_bs = ggml_blck_سائز(src0->type);
    ثابت size_t q8_1_ts = حجم(بلاک_q8_1);
    ثابت size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ثابت منطقی src0_is_contiguous = ggml_is_contiguous(src0);
    ثابت منطقی src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_ستون_سائز = GGML_PAD(ne10, میٹرکس_صف_PADDING);

    ثابت منطقی بانٹو = ggml_backend_بفر_is_sycl_بانٹو(src0->بفر);
    GGML_ASSERT(!(بانٹو && ne02 > 1));
    GGML_ASSERT(!(بانٹو && ne03 > 1));
    GGML_ASSERT(!(بانٹو && ne02 < ne12));

    std::array<اعشاری, GGML_SYCL_MAX_DEVICES> tensor_بانٹو;
    اگر (بانٹو) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_بانٹو_بفر_type_context * buft_ctx = (ggml_backend_sycl_بانٹو_بفر_type_context *) src0->بفر->buft->context;
        tensor_بانٹو = buft_ctx->tensor_بانٹو;
    }

    ساخت dev_ڈیٹا {
        ggml_sycl_pool_alloc<حرف> src0_dd_alloc;
        ggml_sycl_pool_alloc<اعشاری> src1_ddf_alloc;
        ggml_sycl_pool_alloc<حرف> src1_ddq_alloc;
        ggml_sycl_pool_alloc<اعشاری> dst_dd_alloc;

        حرف *src0_dd = خالی_اشاریہ;
        اعشاری *src1_ddf = خالی_اشاریہ; // float
        حرف *src1_ddq = خالی_اشاریہ;  // q8_1
        اعشاری *dst_dd = خالی_اشاریہ;

        int64_t صف_کم;
        int64_t صف_بلند;
    };

    dev_ڈیٹا dev[GGML_SYCL_MAX_DEVICES];

    صحیح مستعمل_devices = 0;
    قطار_ptr main_دھارا = ctx.stream();

    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        // by default, use all rows
        dev[i].صف_کم  = 0;
        dev[i].صف_بلند = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        اگر (بانٹو) {
            ثابت int64_t rounding = get_صف_rounding(src0->type, tensor_بانٹو);

            اگر (i != 0) {
                dev[i].صف_کم  = ne01*tensor_بانٹو[i];
                اگر (dev[i].صف_کم < ne01) {
                    dev[i].صف_کم -= dev[i].صف_کم % rounding;
                }
            }

            اگر (i != ggml_sycl_info().device_گنتی - 1) {
                dev[i].صف_بلند  = ne01*tensor_بانٹو[i + 1];
                اگر (dev[i].صف_بلند < ne01) {
                    dev[i].صف_بلند -= dev[i].صف_بلند % rounding;
                }
            }
        }
    }

    constexpr منطقی quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_سائز>,
                                                      no_quantize_q8_1<QK8_1 / WARP_سائز>>;
    برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
        اگر ((!بانٹو && i != ctx.device) || dev[i].صف_کم == dev[i].صف_بلند) {
            جاری;
        }

        مستعمل_devices++;

        ثابت منطقی src1_on_device = i == ctx.device;
        ثابت منطقی  dst_on_device = i == ctx.device;

        ggml_sycl_سیٹ_device(i);
        قطار_ptr stream = ctx.stream(i, 0);

        اگر (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = بانٹو ? (حرف *) src0_extra->ڈیٹا_device[i] : (حرف *) src0->ڈیٹا;
        } ورنہ {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        اگر (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (اعشاری *) src1->ڈیٹا;
        } ورنہ {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        اگر constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_ستون_سائز*q8_1_ts/q8_1_bs);

            اگر (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                کوشش {
                    quantize_صف_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_ستون_سائز, stream);
                } پکڑو (sycl::exception ثابت &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::سطر_ختم;
                    std::باہر_نکلو(1);
                }
            }
        }

        اگر (dst_on_device) {
            dev[i].dst_dd = (اعشاری *) dst->ڈیٹا;
        } ورنہ {
            ثابت size_t سائز_dst_ddf = بانٹو ? (dev[i].صف_بلند - dev[i].صف_کم)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), سائز_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    اگر (بانٹو && مستعمل_devices > 1) {
        ggml_sycl_سیٹ_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ثابت int64_t src1_ستون_stride = بانٹو && مستعمل_devices > 1 ? MUL_MAT_SRC1_ستون_STRIDE : ne11;
    برائے (int64_t src1_ستون_0 = 0; src1_ستون_0 < ne11; src1_ستون_0 += src1_ستون_stride) {
        ثابت int64_t is = بانٹو ? (src1_ستون_0/src1_ستون_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ثابت int64_t src1_ncols = src1_ستون_0 + src1_ستون_stride > ne11 ? ne11 - src1_ستون_0 : src1_ستون_stride;
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
            اگر ((!بانٹو && i != ctx.device) || dev[i].صف_کم == dev[i].صف_بلند) {
                جاری;
            }

            ثابت منطقی src1_on_device = i == ctx.device;
            ثابت منطقی  dst_on_device = i == ctx.device;
            ثابت int64_t صف_diff = dev[i].صف_بلند - dev[i].صف_کم;

            ggml_sycl_سیٹ_device(i);
            قطار_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            اگر (بانٹو && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            برائے (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ثابت int64_t i03 = i0 / ne12;
                ثابت int64_t i02 = i0 % ne12;

                ثابت size_t src1_ddq_i_انحراف = (i0*ne11 + src1_ستون_0) * src1_padded_ستون_سائز*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                حرف  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_مقسوم_علیہ) * (ne01*ne00*src0_ts)/src0_bs;
                اعشاری * src1_ddf_i = (اعشاری *) ((حرف *) dev[i].src1_ddf +
                    (i0*ne11 + src1_ستون_0) * ne10 * ggml_type_سائز(src1->type));
                حرف  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_انحراف;
                اعشاری *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_ستون_0) * (dst_on_device ? ne0 : صف_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                اگر (i == ctx.device) {
                    dst_dd_i += dev[i].صف_کم; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                اگر (src1_is_contiguous) {
                    اگر (i != ctx.device) {
                        اگر constexpr (quantize_enabled) {
                            حرف * src1_ddq_i_ماخذ = dev[ctx.device].src1_ddq + src1_ddq_i_انحراف;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->حافظہ_نقل(src1_ddq_i, src1_ddq_i_ماخذ,
                                                             src1_ncols * src1_padded_ستون_سائز * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } ورنہ {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ثابت حرف * src1_ddf_i_ماخذ = (ثابت حرف *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_ستون_0) * ne10 * ggml_type_سائز(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_دھارا, src1_ddf_i, src1_ddf_i_ماخذ,
                                                               src1_ncols * ne10 * ggml_type_سائز(src1->type))));
                        }
                    }
                } ورنہ {
                    اگر (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_ستون_0,
                                                           src1_ستون_0 + src1_ncols, stream));
                    } ورنہ {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    اگر constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        کوشش {
                            quantize_صف_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_ستون_سائز, stream);
                        } پکڑو (ثابت sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
                            std::باہر_نکلو(1);
                        }
                    }
                }

                اگر (src1_ستون_0 == 0 && !src0_is_contiguous && i02 % i02_مقسوم_علیہ == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_مقسوم_علیہ, dev[i].صف_کم, dev[i].صف_بلند, stream));
                }
                اگر (src1->type == GGML_TYPE_F16) {
                    src1_padded_ستون_سائز = (i0 * ne11 + src1_ستون_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].صف_کم, dev[i].صف_بلند, src1_ncols, src1_padded_ستون_سائز, stream)));

                // copy dst to host or other device if necessary
                اگر (!dst_on_device) {
                    خالی * dst_off_device = dst->ڈیٹا;
                    اگر (بانٹو) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        اعشاری * dhf_dst_i = (اعشاری *) ((حرف *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*حجم(اعشاری));
                        dhf_dst_i += src1_ستون_0*ne0 + dev[i].صف_کم;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * حجم(اعشاری), dst_dd_i,
                            صف_diff * حجم(اعشاری), صف_diff * حجم(اعشاری),
                            src1_ncols, dpct::device_مقصد_device, *stream)));
                    } ورنہ {
                        اعشاری * dhf_dst_i = (اعشاری *) ((حرف *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*حجم(اعشاری));
                        dhf_dst_i += src1_ستون_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->حافظہ_نقل(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * حجم(اعشاری)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                اگر (بانٹو && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    اگر (بانٹو && ggml_sycl_info().device_گنتی > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_ستون_STRIDE - 1) / MUL_MAT_SRC1_ستون_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_سیٹ_device(ctx.device);
        برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; ++i) {
            اگر (dev[i].صف_کم == dev[i].صف_بلند) {
                جاری;
            }
            برائے (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_sycl_repeat_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_عقب(ctx, dst);
}

ساکن خالی ggml_sycl_get_صفیں(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_صفیں(ctx, dst);
}

ساکن خالی ggml_sycl_get_صفیں_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_صفیں_عقب(ctx, dst);
}

ساکن خالی ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

ساکن خالی ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

ساکن خالی ggml_sycl_rms_norm_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_عقب(ctx, dst);
}

ساکن خالی ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

ساکن خالی ggml_sycl_گروہ_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_گروہ_norm(ctx, dst);
}

ساکن خالی ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                       ثابت ggml_tensor *src1,
                                       ggml_tensor *dst) کوشش {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(src0->بفر));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];

    ثابت int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    قطار_ptr main_دھارا = ctx.stream();

    خالی  * src0_ddq = src0->ڈیٹا;
    اعشاری * src1_ddf = (اعشاری *) src1->ڈیٹا;
    اعشاری * dst_ddf  = (اعشاری *) dst->ڈیٹا;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_دھارا);
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                     ثابت ggml_tensor *src1,
                                     ggml_tensor *dst) کوشش {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(src0->بفر));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];

    ثابت int64_t nb01 = src0->nb[1];
    ثابت int64_t nb02 = src0->nb[2];

    ثابت int64_t ne12 = src1->ne[2];
    ثابت int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    قطار_ptr main_دھارا = ctx.stream();

    خالی  * src0_ddq = src0->ڈیٹا;
    اعشاری * src1_ddf = (اعشاری *) src1->ڈیٹا;
    اعشاری * dst_ddf  = (اعشاری *) dst->ڈیٹا;

    ثابت int64_t صف_stride_x = nb01 / حجم(sycl::half);
    ثابت int64_t channel_stride_x = nb02 / حجم(sycl::half);
    ثابت int64_t channel_stride_y = nb11 / حجم(اعشاری);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, صف_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_دھارا);
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی k_compute_batched_ptrs(ثابت sycl::half * src0_as_f16, ثابت sycl::half * src1_as_f16, خالی * dst,
                                   ثابت خالی ** ptrs_src, خالی ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت int64_t i13 = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) + شے_ct1.get_local_id(2);
    ثابت int64_t i12 = شے_ct1.get_گروہ(1) * شے_ct1.get_local_range(1) + شے_ct1.get_local_id(1);

    اگر (i13 >= ne13 || i12 >= ne12) {
        واپس;
    }

    ثابت int64_t i03 = i13 / r3;
    ثابت int64_t i02 = i12 / r2;

    ثابت uint8_t * src0_بائٹیں = reinterpret_cast<ثابت uint8_t *>(src0_as_f16);
    ثابت uint8_t * src1_بائٹیں = reinterpret_cast<ثابت uint8_t *>(src1_as_f16);
    uint8_t *       dst_بائٹیں  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_بائٹیں + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_بائٹیں + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_بائٹیں + i12 * nbd2 + i13 * nbd3;
}

ساکن خالی ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                           ثابت ggml_tensor * src1, ggml_tensor * dst) کوشش {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(src0->بفر));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ثنائی_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));
    قطار_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ثابت sycl::half * src0_f16 = static_cast<ثابت sycl::half *>(src0->ڈیٹا);
    اعشاری *            dst_ddf  = static_cast<اعشاری *>(dst->ڈیٹا);

    ثابت sycl::half * src1_f16       = static_cast<ثابت sycl::half *>(src1->ڈیٹا);
#اگر GGML_SYCL_DNNL
    ثابت size_t       type_سائز_src0 = ggml_type_سائز(src0->type);
#اگر_ختم
    ثابت size_t       type_سائز_src1 = ggml_type_سائز(src1->type);

    منطقی is_src0_cont_2 = ggml_is_contiguous_2(src0);
    منطقی is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_سائز_src1;
    int64_t                          s12 = nb12 / type_سائز_src1;
    int64_t                          s13 = nb13 / type_سائز_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    اگر (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#اگر GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        صحیح آخری_dim=0;
        صحیح آخری_str=0;
        size_t سب_سے_بڑا_str=0;
        برائے(صحیح i = 0; i< 4; i++){
            // last stride is always the largest
            اگر(src1->nb[i] == سب_سے_بڑا_str){
                اگر(src1->ne[آخری_dim] == 1){
                    آخری_str = i;
                    آخری_dim = i;
                }
            }
            اگر(src1->nb[i] > سب_سے_بڑا_str){
                سب_سے_بڑا_str = src1->nb[i];
                آخری_str = i;
                آخری_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ثابت int64_t ne_src1 = src1->nb[آخری_str] * src1->ne[آخری_dim] / type_سائز_src1;
        src1_f16_alloc.alloc(ne_src1);
        ثابت مقصد_fp16_sycl_t مقصد_fp16_sycl = ggml_get_مقصد_fp16_sycl(src1->type, dst);
        GGML_ASSERT(مقصد_fp16_sycl != خالی_اشاریہ);
        مقصد_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# ورنہ
        ثابت int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ثابت مقصد_fp16_nc_sycl_t مقصد_fp16_nc_sycl = ggml_get_مقصد_fp16_nc_sycl(src1->type);
        GGML_ASSERT(مقصد_fp16_nc_sycl != خالی_اشاریہ);
        مقصد_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#اگر_ختم

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = سچ;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_ڈیٹا_t mkl_compute_type = dpct::library_ڈیٹا_t::real_float;
    dpct::library_ڈیٹا_t mkl_ڈیٹا_type    = dpct::library_ڈیٹا_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ثابت اعشاری alpha_f32 = 1.0f;
    ثابت اعشاری beta_f32  = 0.0f;

    ثابت خالی * alpha = &alpha_f32;
    ثابت خالی * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

#اگر GGML_SYCL_DNNL
    اگر (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_سائز_src0;
            int64_t str_a1 = nb01 / type_سائز_src0;
            int64_t str_a2 = nb02 / type_سائز_src0;

            int64_t str_b0 = nb10 / type_سائز_src1;
            int64_t str_b1 = nb11 / type_سائز_src1;
            int64_t str_b2 = nb12 / type_سائز_src1;

            خودکار launch_gemm_for_batches = [&ctx, queue](ثابت sycl::half *src0,
                                                ثابت sycl::half *src1, اعشاری *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                منطقی supported_نشر = batchb == batcha ? سچ
                        : batchb == 1 || batcha == 1        ? سچ
                                                            : جھوٹ;
                اگر (supported_نشر) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::مقصد_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::مقصد_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::مقصد_dt<اعشاری>(), queue, batcha, batchb);
                } ورنہ {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    اگر (batches0 > batches1) {
                        int64_t عدد_mul_mats = batches1;
                        int64_t sub_batch = batches0 / عدد_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        برائے (int64_t i0 = 0; i0 < عدد_mul_mats; i0++) {
                            ثابت sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ثابت sycl::half *src1_shifted = src1 + (sb2 * i0);
                            اعشاری *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::مقصد_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::مقصد_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::مقصد_dt<اعشاری>(),
                                    queue, sub_batch, 1);
                        }
                    } ورنہ {
                        int64_t عدد_mul_mats = batches0;
                        int64_t sub_batch = batches1 / عدد_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        برائے (int64_t i1 = 0; i1 < عدد_mul_mats; i1++) {
                            ثابت sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ثابت sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            اعشاری *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::مقصد_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::مقصد_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::مقصد_dt<اعشاری>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ثابت منطقی cont_batches_dim2_a = nb02 * ne02 == nb03;
            ثابت منطقی cont_batches_dim2_b = nb12 * ne12 == nb13;
            ثابت منطقی cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ثابت منطقی cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            اگر (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / حجم(اعشاری));
            } ورنہ اگر (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_سائز_src0;
                int64_t str_b3 = nb13 / type_سائز_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / حجم(اعشاری));
            } ورنہ {
                برائے (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ثابت sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_سائز_src0);
                    ثابت sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_سائز_src1);
                    اعشاری *dst_shifted = dst_ddf + (nb3 * b_a / حجم(اعشاری));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / حجم(اعشاری));
                }
            }

    }
    ورنہ
#اگر_ختم
    {
        اگر (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ثابت int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ثابت int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_ڈیٹا_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_ڈیٹا_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_ڈیٹا_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } ورنہ {
            ثابت صحیح ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ثابت خالی *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<خالی *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<میٹرکس_info_t<اعشاری>> میٹرکس_info(ctx.host_pool(), 1);

            sycl::range<3> بلاک_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ثابت خالی ** ptrs_src_get = ptrs_src.get();
                خالی **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * حجم(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * حجم(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(بلاک_dims, بلاک_dims), [=](sycl::nd_شے<3> شے_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, شے_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ثابت خالی **) (ptrs_src.get() + 0 * ne23), dpct::library_ڈیٹا_t::real_half, nb01 / nb00,
                (ثابت خالی **) (ptrs_src.get() + 1 * ne23), dpct::library_ڈیٹا_t::real_half, s11, beta,
                (خالی **) (ptrs_dst.get() + 0 * ne23), mkl_ڈیٹا_type, ne0, ne23, mkl_compute_type, میٹرکس_info.get())));
        }
    }
} پکڑو (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

شمار جماعت mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

اندرونی منطقی ggml_sycl_supports_mmq(شمار ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_غیر_مستعمل(type);
    واپس جھوٹ;
}

اندرونی منطقی ggml_sycl_supports_reorder_mul_mat_sycl(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q8_0:
            واپس سچ;
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
            واپس !g_ggml_sycl_prioritize_dmmv;
        طےشدہ:
            واپس جھوٹ;
    }
}

اندرونی منطقی ggml_sycl_supports_reorder_dmmv(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

اندرونی منطقی ggml_sycl_supports_reorder_mmvq(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_IQ3_XXS:
        حالت GGML_TYPE_IQ3_S:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

ساکن منطقی ggml_sycl_supports_reorder_esimd(شمار ggml_type type) {
#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
    انتخاب (type) {
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_Q8_0:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
#ورنہ
    GGML_غیر_مستعمل(type);
    واپس جھوٹ;
#اگر_ختم
}

ساکن منطقی ggml_sycl_supports_dmmv(شمار ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_Q1_0:
        حالت GGML_TYPE_Q4_0:
        حالت GGML_TYPE_Q4_1:
        حالت GGML_TYPE_Q5_0:
        حالت GGML_TYPE_Q5_1:
        حالت GGML_TYPE_Q8_0:
        حالت GGML_TYPE_Q2_K:
        حالت GGML_TYPE_Q3_K:
        حالت GGML_TYPE_Q4_K:
        حالت GGML_TYPE_Q5_K:
        حالت GGML_TYPE_Q6_K:
        حالت GGML_TYPE_F16:
        حالت GGML_TYPE_BF16:
            واپس سچ;
        طےشدہ:
            واپس جھوٹ;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
ساکن اندرونی خالی * sycl_ext_malloc_device(dpct::قطار_ptr stream, size_t size) {
    منطقی use_async = g_ggml_sycl_use_async_mem_op;
#اگر defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_حافظہ_ALLOC
    اگر (use_async) {
        خالی * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_جوڑو(GGML_SYCL_MEM_ASYNC, ptr, size);
        واپس ptr;
    }
#ورنہ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#اگر_ختم
    واپس ggml_sycl_malloc_device(size, *stream);
}

ساکن اندرونی خالی sycl_ext_free(dpct::قطار_ptr stream, خالی * ptr) {
    منطقی use_async = g_ggml_sycl_use_async_mem_op;
#اگر defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_حافظہ_ALLOC
    اگر (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        واپس;
    }
#ورنہ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#اگر_ختم
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
ساخت sycl_reorder_عارضی_بفر {
    خالی *          ptr  = خالی_اشاریہ;
    dpct::قطار_ptr stream;

    sycl_reorder_عارضی_بفر(dpct::قطار_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#اگر_تعریف_شدہ GGML_SYCL_HOST_MEM_FALLBACK
        اگر (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            اگر (ptr) {
                host_fallback = سچ;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#اگر_ختم
    }

    ~sycl_reorder_عارضی_بفر() {
        اگر (!ptr) {
            واپس;
        }
        اگر (host_fallback) {
            sycl::آزاد(ptr, *stream);
        } ورنہ {
            sycl_ext_free(stream, ptr);
        }
    }

    صریح عامل منطقی() ثابت { واپس ptr != خالی_اشاریہ; }

    sycl_reorder_عارضی_بفر(ثابت sycl_reorder_عارضی_بفر &)            = حذف;
    sycl_reorder_عارضی_بفر & عامل=(ثابت sycl_reorder_عارضی_بفر &) = حذف;

نجی:
    منطقی host_fallback = جھوٹ;
};

ساکن منطقی reorder_qw_q4_0(uint8_t * ڈیٹا_device, ثابت صحیح ncols, ثابت صحیح nrows, size_t size, size_t انحراف,
                            dpct::قطار_ptr stream) {
    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % حجم(بلاک_q4_0) == 0));
    GGML_ASSERT((انحراف % حجم(بلاک_q4_0) == 0));
    صحیح انحراف_blks = انحراف / حجم(بلاک_q4_0);
    خودکار qs_ptr      = ڈیٹا_device + انحراف_blks * QK4_0 / 2;
    خودکار d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + انحراف_blks;

    خودکار reorder_event = stream->parallel_for(
        size / حجم(بلاک_q4_0),
            [=](خودکار i) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            ثابت بلاک_q4_0* x = (ثابت بلاک_q4_0*)tmp_buf;
            ثابت صحیح ib = i;

            برائے (صحیح j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q8_0(uint8_t * ڈیٹا_device, ثابت صحیح ncols, ثابت صحیح nrows, size_t size, size_t انحراف,
                            dpct::قطار_ptr stream) {
    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % حجم(بلاک_q8_0) == 0));
    GGML_ASSERT((انحراف % حجم(بلاک_q8_0) == 0));
    صحیح انحراف_blks = انحراف / حجم(بلاک_q8_0);
    خودکار qs_ptr = ڈیٹا_device + انحراف_blks * QK8_0;
    خودکار d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + انحراف_blks;

    خودکار reorder_event = stream->parallel_for(
        size / حجم(بلاک_q8_0),
            [=](خودکار i) [[sycl::reqd_sub_گروہ_سائز(WARP_سائز)]] {
            ثابت بلاک_q8_0* x = (ثابت بلاک_q8_0*)tmp_buf;
            ثابت صحیح ib = i;

            برائے (صحیح j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
ساکن منطقی reorder_qw_q8_0_moe(uint8_t * ڈیٹا_device, size_t expert_بائٹیں, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائٹیں % حجم(بلاک_q8_0) == 0);
    ثابت صحیح    بلاکیں_فی_expert = (صحیح) (expert_بائٹیں / حجم(بلاک_q8_0));
    ثابت size_t کل_بائٹیں       = expert_بائٹیں * (size_t) n_expert;

    sycl_reorder_عارضی_بفر tmp(stream, کل_بائٹیں);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, کل_بائٹیں);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, کل_بائٹیں)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحیح کل_بلاکیں = بلاکیں_فی_expert * (صحیح) n_expert;
    خودکار reorder_event = stream->parallel_for(کل_بلاکیں, [=](خودکار gb_) {
        ثابت صحیح          gb   = gb_;
        ثابت صحیح          e    = gb / بلاکیں_فی_expert;
        ثابت صحیح          ib   = gb % بلاکیں_فی_expert;
        ثابت بلاک_q8_0 * x    = (ثابت بلاک_q8_0 *) (tmp_buf + (size_t) e * expert_بائٹیں);
        uint8_t *          بنیاد = ڈیٹا_device + (size_t) e * expert_بائٹیں;

        خودکار * qs_ptr = بنیاد;
        خودکار * d_ptr  = (sycl::half *) (qs_ptr + (size_t) بلاکیں_فی_expert * QK8_0);

        برائے (صحیح j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q4_k(uint8_t * ڈیٹا_device, size_t size, size_t انحراف, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_q4_K) == 0);
    GGML_ASSERT(انحراف % حجم(بلاک_q4_K) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_q4_K);

    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار * qs_ptr     = ڈیٹا_device;
    خودکار * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    خودکار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائز * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_q4_K * x  = (ثابت بلاک_q4_K *) tmp_buf;
        ثابت صحیح          ib = i;

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        برائے (صحیح j = 0; j < K_SCALE_سائز; ++j) {
            scales_ptr[ib * K_SCALE_سائز + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

// Reorder each expert slice into a self-contained SoA layout.
ساکن منطقی reorder_qw_q4_k_moe(uint8_t * ڈیٹا_device, size_t expert_بائٹیں, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائٹیں % حجم(بلاک_q4_K) == 0);
    ثابت صحیح    بلاکیں_فی_expert = (صحیح) (expert_بائٹیں / حجم(بلاک_q4_K));
    ثابت size_t کل_بائٹیں       = expert_بائٹیں * (size_t) n_expert;

    sycl_reorder_عارضی_بفر tmp(stream, کل_بائٹیں);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, کل_بائٹیں);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, کل_بائٹیں)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحیح کل_بلاکیں = بلاکیں_فی_expert * (صحیح) n_expert;
    خودکار reorder_event = stream->parallel_for(کل_بلاکیں, [=](خودکار gb_) {
        ثابت صحیح          gb   = gb_;
        ثابت صحیح          e    = gb / بلاکیں_فی_expert;
        ثابت صحیح          ib   = gb % بلاکیں_فی_expert;
        ثابت بلاک_q4_K * x    = (ثابت بلاک_q4_K *) (tmp_buf + (size_t) e * expert_بائٹیں);
        uint8_t *          بنیاد = ڈیٹا_device + (size_t) e * expert_بائٹیں;

        خودکار * qs_ptr     = بنیاد;
        خودکار * scales_ptr = qs_ptr + QK_K / 2 * بلاکیں_فی_expert;
        خودکار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائز * بلاکیں_فی_expert);

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        برائے (صحیح j = 0; j < K_SCALE_سائز; ++j) {
            scales_ptr[ib * K_SCALE_سائز + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
ساکن منطقی reorder_qw_q5_k_moe(uint8_t * ڈیٹا_device, size_t expert_بائٹیں, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائٹیں % حجم(بلاک_q5_K) == 0);
    ثابت صحیح    بلاکیں_فی_expert = (صحیح) (expert_بائٹیں / حجم(بلاک_q5_K));
    ثابت size_t کل_بائٹیں       = expert_بائٹیں * (size_t) n_expert;

    sycl_reorder_عارضی_بفر tmp(stream, کل_بائٹیں);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, کل_بائٹیں);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, کل_بائٹیں)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحیح کل_بلاکیں = بلاکیں_فی_expert * (صحیح) n_expert;
    خودکار reorder_event = stream->parallel_for(کل_بلاکیں, [=](خودکار gb_) {
        ثابت صحیح          gb   = gb_;
        ثابت صحیح          e    = gb / بلاکیں_فی_expert;
        ثابت صحیح          ib   = gb % بلاکیں_فی_expert;
        ثابت بلاک_q5_K * x    = (ثابت بلاک_q5_K *) (tmp_buf + (size_t) e * expert_بائٹیں);
        uint8_t *          بنیاد = ڈیٹا_device + (size_t) e * expert_بائٹیں;

        خودکار * qs_ptr     = بنیاد;
        خودکار * qh_ptr     = qs_ptr + (QK_K / 2) * بلاکیں_فی_expert;
        خودکار * scales_ptr = qh_ptr + (QK_K / 8) * بلاکیں_فی_expert;
        خودکار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائز * بلاکیں_فی_expert);

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        برائے (صحیح j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        برائے (صحیح j = 0; j < K_SCALE_سائز; ++j) {
            scales_ptr[ib * K_SCALE_سائز + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
ساکن منطقی reorder_qw_q6_k_moe(uint8_t * ڈیٹا_device, size_t expert_بائٹیں, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائٹیں % حجم(بلاک_q6_K) == 0);
    ثابت صحیح    بلاکیں_فی_expert = (صحیح) (expert_بائٹیں / حجم(بلاک_q6_K));
    ثابت size_t کل_بائٹیں       = expert_بائٹیں * (size_t) n_expert;

    sycl_reorder_عارضی_بفر tmp(stream, کل_بائٹیں);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, کل_بائٹیں);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, کل_بائٹیں)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحیح کل_بلاکیں = بلاکیں_فی_expert * (صحیح) n_expert;
    خودکار reorder_event = stream->parallel_for(کل_بلاکیں, [=](خودکار gb_) {
        ثابت صحیح          gb   = gb_;
        ثابت صحیح          e    = gb / بلاکیں_فی_expert;
        ثابت صحیح          ib   = gb % بلاکیں_فی_expert;
        ثابت بلاک_q6_K * x    = (ثابت بلاک_q6_K *) (tmp_buf + (size_t) e * expert_بائٹیں);
        uint8_t *          بنیاد = ڈیٹا_device + (size_t) e * expert_بائٹیں;

        خودکار * ql_ptr     = بنیاد;
        خودکار * qh_ptr     = ql_ptr + (QK_K / 2) * بلاکیں_فی_expert;
        خودکار * scales_ptr = qh_ptr + (QK_K / 4) * بلاکیں_فی_expert;
        خودکار * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * بلاکیں_فی_expert);

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        برائے (صحیح j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        برائے (صحیح j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
ساکن منطقی reorder_qw_mxfp4_moe(uint8_t * ڈیٹا_device, size_t expert_بائٹیں, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائٹیں % حجم(بلاک_mxfp4) == 0);
    ثابت صحیح     بلاکیں_فی_expert = (صحیح) (expert_بائٹیں / حجم(بلاک_mxfp4));
    ثابت size_t  max_chunk_بائٹیں   = 32u << 20;
    ثابت int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_بائٹیں / expert_بائٹیں)));

    sycl_reorder_عارضی_بفر tmp(stream, (size_t) chunk_experts * expert_بائٹیں);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_بائٹیں);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    برائے (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ثابت int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = ڈیٹا_device + (size_t) e0 * expert_بائٹیں;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, chunk, (size_t) n_chunk * expert_بائٹیں)));
        اگر (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        ثابت صحیح کل_بلاکیں = بلاکیں_فی_expert * (صحیح) n_chunk;
        خودکار reorder_event = stream->parallel_for(کل_بلاکیں, [=](خودکار gb_) {
            ثابت صحیح           gb   = gb_;
            ثابت صحیح           e    = gb / بلاکیں_فی_expert;
            ثابت صحیح           ib   = gb % بلاکیں_فی_expert;
            ثابت بلاک_mxfp4 * x    = (ثابت بلاک_mxfp4 *) (tmp_buf + (size_t) e * expert_بائٹیں);
            uint8_t *           بنیاد = chunk + (size_t) e * expert_بائٹیں;

            uint8_t * qs_ptr = بنیاد;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) بلاکیں_فی_expert;

            برائے (صحیح j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        اگر (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q2_k(uint8_t * ڈیٹا_device, size_t size, size_t انحراف, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_q2_K) == 0);
    GGML_ASSERT(انحراف % حجم(بلاک_q2_K) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_q2_K);

    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار *        qs_ptr     = ڈیٹا_device;
    خودکار *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_q2_K * x  = (ثابت بلاک_q2_K *) tmp_buf;
        ثابت صحیح          ib = i;

        برائے (صحیح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        برائے (صحیح j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q3_k(uint8_t * ڈیٹا_device, size_t size, size_t انحراف, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_q3_K) == 0);
    GGML_ASSERT(انحراف % حجم(بلاک_q3_K) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_q3_K);

    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار *       qs_ptr     = ڈیٹا_device;
    خودکار *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    خودکار *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_q3_K * x  = (ثابت بلاک_q3_K *) tmp_buf;
        ثابت صحیح          ib = i;

        برائے (صحیح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        برائے (صحیح j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        برائے (صحیح j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q5_k(uint8_t * ڈیٹا_device, size_t size, size_t انحراف, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_q5_K) == 0);
    GGML_ASSERT(انحراف % حجم(بلاک_q5_K) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_q5_K);

    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار * qs_ptr     = ڈیٹا_device;
    خودکار * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    خودکار * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    خودکار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائز * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_q5_K * x  = (ثابت بلاک_q5_K *) tmp_buf;
        ثابت صحیح          ib = i;

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        برائے (صحیح j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        برائے (صحیح j = 0; j < K_SCALE_سائز; ++j) {
            scales_ptr[ib * K_SCALE_سائز + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_q6_k(uint8_t * ڈیٹا_device, size_t size, size_t انحراف, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_q6_K) == 0);
    GGML_ASSERT(انحراف % حجم(بلاک_q6_K) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_q6_K);

    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار *       ql_ptr     = ڈیٹا_device;
    خودکار *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    خودکار *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_q6_K * x  = (ثابت بلاک_q6_K *) tmp_buf;
        ثابت صحیح          ib = i;

        ثابت uint8_t * ql              = x[ib].ql;
        ثابت uint8_t * qh              = x[ib].qh;
        uint8_t *       بنیاد_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       بنیاد_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       بنیاد_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        برائے (صحیح j = 0; j < QK_K / 2; ++j) {
            بنیاد_ql_ptr[j] = ql[j];
        }
        برائے (صحیح j = 0; j < QK_K / 4; ++j) {
            بنیاد_qh_ptr[j] = qh[j];
        }

        برائے (صحیح j = 0; j < QK_K / 16; ++j) {
            بنیاد_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_iq3_s(uint8_t * ڈیٹا_device, size_t size, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_iq3_s) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_iq3_s);
    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار * qs_ptr       = ڈیٹا_device;
    خودکار * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    خودکار * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    خودکار * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_iq3_s * x = (ثابت بلاک_iq3_s *) tmp_buf;
        ثابت صحیح ib = i;

        برائے (صحیح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        برائے (صحیح j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        برائے (صحیح j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (حجم(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        برائے (صحیح j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[حجم(ggml_half) + j] = x[ib].scales[j];
        }
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw_iq3_xxs(uint8_t * ڈیٹا_device, size_t size, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاک_iq3_xxs) == 0);

    ثابت صحیح nblocks = size / حجم(بلاک_iq3_xxs);
    sycl_reorder_عارضی_بفر tmp(stream, size);
    اگر (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        واپس جھوٹ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->حافظہ_نقل(tmp_buf, ڈیٹا_device, size)));
    اگر (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودکار * qs_ptr = ڈیٹا_device;
    خودکار * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    خودکار reorder_event = stream->parallel_for(nblocks, [=](خودکار i) {
        ثابت بلاک_iq3_xxs * x = (ثابت بلاک_iq3_xxs *) tmp_buf;
        ثابت صحیح ib = i;

        برائے (صحیح j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    اگر (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    واپس سچ;
}

ساکن منطقی reorder_qw(ثابت ggml_tensor * src0, dpct::قطار_ptr stream) {
    uint8_t * ڈیٹا_device = (uint8_t *) src0->ڈیٹا;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    اگر (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        انتخاب (src0->type) {
            حالت GGML_TYPE_Q8_0:
                واپس reorder_qw_q8_0_moe(ڈیٹا_device, src0->nb[2], src0->ne[2], stream);
            حالت GGML_TYPE_Q4_K:
                واپس reorder_qw_q4_k_moe(ڈیٹا_device, src0->nb[2], src0->ne[2], stream);
            حالت GGML_TYPE_Q5_K:
                واپس reorder_qw_q5_k_moe(ڈیٹا_device, src0->nb[2], src0->ne[2], stream);
            حالت GGML_TYPE_Q6_K:
                واپس reorder_qw_q6_k_moe(ڈیٹا_device, src0->nb[2], src0->ne[2], stream);
            حالت GGML_TYPE_MXFP4:
                واپس reorder_qw_mxfp4_moe(ڈیٹا_device, src0->nb[2], src0->ne[2], stream);
            طےشدہ:
                واپس جھوٹ;
        }
    }

    انتخاب (src0->type) {
        حالت GGML_TYPE_Q4_0:
            واپس reorder_qw_q4_0(ڈیٹا_device, ncols, nrows, size, 0, stream);
        حالت GGML_TYPE_Q8_0:
            واپس reorder_qw_q8_0(ڈیٹا_device, ncols, nrows, size, 0, stream);
        حالت GGML_TYPE_Q2_K:
            واپس reorder_qw_q2_k(ڈیٹا_device, size, 0, stream);
        حالت GGML_TYPE_Q3_K:
            واپس reorder_qw_q3_k(ڈیٹا_device, size, 0, stream);
        حالت GGML_TYPE_Q4_K:
            واپس reorder_qw_q4_k(ڈیٹا_device, size, 0, stream);
        حالت GGML_TYPE_Q5_K:
            واپس reorder_qw_q5_k(ڈیٹا_device, size, 0, stream);
        حالت GGML_TYPE_Q6_K:
            واپس reorder_qw_q6_k(ڈیٹا_device, size, 0, stream);
        حالت GGML_TYPE_IQ3_XXS:
            واپس reorder_qw_iq3_xxs(ڈیٹا_device, size, stream);
        حالت GGML_TYPE_IQ3_S:
            واپس reorder_qw_iq3_s(ڈیٹا_device, size, stream);
        طےشدہ:
            واپس جھوٹ;
    }
}

ساکن منطقی should_reorder_tensor(ggml_backend_sycl_context& ctx, ثابت ggml_tensor * dst) {
    واپس g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

ساکن خالی opt_for_reorder(ggml_backend_sycl_context * ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    اگر (!should_reorder_tensor(*ctx, dst)) {
        واپس;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    اگر (!extra || extra->optimized_feature.reorder) {
        واپس;  // Skip permutations and already reordered tensors
    }

    انتخاب (mm_algorithm) {
        حالت mul_mat_algo::DMMV:
            اگر (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                واپس;
            }
            توڑو;
        حالت mul_mat_algo::MMVQ:
            اگر (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                واپس;
            }
            توڑو;
        حالت mul_mat_algo::MUL_MAT_SYCL:
            اگر (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                واپس;
            }
            توڑو;
    }

    اگر (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = سچ;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
ساکن خالی opt_for_reorder_id(ggml_backend_sycl_context * ctx, ثابت ggml_tensor * src0) {
    اگر (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        واپس;
    }
    اگر (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        واپس;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    اگر (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->ڈیٹا % 16 != 0)) {
        واپس;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    اگر (!extra || extra->optimized_feature.reorder) {
        واپس;
    }
    اگر (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = سچ;
    }
}


ساکن منطقی can_use_dequantize_mul_mat_vec(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ثابت int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    واپس ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

ساکن منطقی can_use_mul_mat_vec_q(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    واپس ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_سائز;
}

ساکن خالی ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    اگر (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        واپس;
    }

    ثابت منطقی بانٹو = ggml_backend_بفر_is_sycl_بانٹو(src0->بفر);
    int64_t min_compute_capability = زیادہ_سے_زیادہ_عدد;

    اگر (بانٹو) {
        ggml_backend_sycl_بانٹو_بفر_type_context * buft_ctx =
            (ggml_backend_sycl_بانٹو_بفر_type_context *) src0->بفر->buft->context;
        خودکار & tensor_بانٹو = buft_ctx->tensor_بانٹو;
        برائے (صحیح id = 0; id < ggml_sycl_info().device_گنتی; ++id) {
            // skip devices that are not going to do any work:
            اگر (tensor_بانٹو[id] >= (id + 1 < ggml_sycl_info().device_گنتی ? tensor_بانٹو[id + 1] : 1.0f)) {
                جاری;
            }

            اگر (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } ورنہ {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    منطقی use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#اگر_تعریف_شدہ GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#اگر_ختم

    منطقی use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    منطقی use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#اگر_تعریف_شدہ SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_سائز);
#اگر_ختم // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    اگر (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        منطقی use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        اگر (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    اگر (!بانٹو && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        اگر (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } ورنہ {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } ورنہ اگر (!بانٹو && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } ورنہ اگر (!بانٹو && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } ورنہ اگر (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } ورنہ اگر (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        اگر (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } ورنہ {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } ورنہ اگر (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } ورنہ {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
ساکن منطقی ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * اوپر, ثابت ggml_tensor * wu,
                                             ثابت ggml_tensor * wg, ثابت ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ثابت خودکار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودکار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    اگر ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        واپس جھوٹ;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ثابت خودکار plain_type = [](ggml_type t) { واپس t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    اگر (!plain_type(wu->type) || !plain_type(wg->type)) {
        واپس جھوٹ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, اوپر, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ثابت int64_t ne00 = wu->ne[0];
    ثابت int64_t ne11 = act->ne[1];

    ثابت قطار_ptr stream = ctx.stream();
    ثابت صحیح src1_padded_ستون_تعداد = GGML_PAD((صحیح) ne00, میٹرکس_صف_PADDING);

    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ستون_تعداد * حجم(بلاک_q8_1) / QK8_1);
    حرف * src1_ddq = src1_q8_alloc.get();

    quantize_صف_q8_1_sycl<quantize_q8_1>((ثابت اعشاری *) act->ڈیٹا, src1_ddq, (صحیح) ne00, (صحیح) ne11,
                                          src1_padded_ستون_تعداد, stream);

    واپس ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->ڈیٹا, wu->ڈیٹا,
                                             src1_ddq, (اعشاری *) glu->ڈیٹا, (صحیح) ne00, (صحیح) wu->ne[1],
                                             (صحیح) ne11,
                                             /*stride_col_y=*/src1_padded_ستون_تعداد / QK8_1,
                                             /*stride_col_dst=*/(صحیح) glu->ne[0], stream);
    GGML_غیر_مستعمل(gate);
}

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
ساکن منطقی ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * اوپر, ثابت ggml_tensor * wu,
                                             ثابت ggml_tensor * wg, ثابت ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ثابت حرف * act_begin = (ثابت حرف *) act->ڈیٹا;
    ثابت حرف * glu_begin = (ثابت حرف *) glu->ڈیٹا;
    اگر (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        واپس جھوٹ;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, اوپر, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ثابت خودکار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودکار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    اگر (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        واپس جھوٹ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, اوپر, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    واپس ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->ڈیٹا, wg->ڈیٹا,
                                                              (ثابت اعشاری *) act->ڈیٹا, (اعشاری *) glu->ڈیٹا,
                                                              (صحیح) wu->ne[0], (صحیح) wu->ne[1], ctx.stream());
}
#اگر_ختم

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
ساکن منطقی ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح گرہ_idx) {
    اگر (!ggml_sycl_can_fuse(cgraph, گرہ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        واپس جھوٹ;
    }

    ggml_tensor *       glu  = cgraph->nodes[گرہ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       اوپر   = glu->src[1];
    ثابت ggml_tensor * wu   = اوپر->src[0];
    ثابت ggml_tensor * wg   = gate->src[0];
    ثابت ggml_tensor * act  = اوپر->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    اگر (ggml_backend_بفر_is_sycl_بانٹو(wu->بفر) || ggml_backend_بفر_is_sycl_بانٹو(wg->بفر)) {
        واپس جھوٹ;
    }

#اگر_تعریف_شدہ GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    اگر (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, اوپر, wu, wg, act)) {
        واپس سچ;
    }
#اگر_ختم

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    اگر (g_ggml_sycl_prioritize_dmmv) {
        واپس جھوٹ;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ثابت منطقی reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    اگر (!reorder_pair) {
        واپس ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, اوپر, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    اگر (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        واپس جھوٹ;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, اوپر, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ثابت خودکار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودکار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    اگر (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        واپس جھوٹ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, اوپر, /*num_src=*/2, " : fused with gate + GLU");

    ثابت int64_t ne00 = wu->ne[0];
    ثابت int64_t ne11 = act->ne[1];

    ثابت قطار_ptr stream           = ctx.stream();
    ثابت صحیح       src1_padded_ستون_تعداد = GGML_PAD((صحیح) ne00, میٹرکس_صف_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_ستون_تعداد * حجم(بلاک_q8_1) / QK8_1);
    حرف *                     src1_ddq = src1_q8_alloc.get();

    quantize_صف_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ثابت اعشاری *) act->ڈیٹا, src1_ddq, (صحیح) ne00, (صحیح) ne11,
                                                          src1_padded_ستون_تعداد, stream);

    واپس ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->ڈیٹا, wg->ڈیٹا, src1_ddq,
                                               (اعشاری *) glu->ڈیٹا, (صحیح) ne00, (صحیح) wu->ne[1], (صحیح) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_ستون_تعداد * (صحیح) حجم(بلاک_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(صحیح) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
ساکن صحیح ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحیح گرہ_idx) {
    ثابت ggml_tensor * گرہ = cgraph->nodes[گرہ_idx];
    اگر (ggml_sycl_info().device_گنتی != 1 || گرہ->type != GGML_TYPE_F32 ||
        گرہ->src[0]->type != GGML_TYPE_F32 || گرہ->src[0]->ne[0] >= 1024) {
        واپس 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    صحیح           count = 0;
    صحیح           آخری  = گرہ_idx;
    اعشاری         eps0;
    حافظہ_نقل(&eps0, گرہ->op_params, حجم(اعشاری));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    خودکار overlaps = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت حرف * ab = (ثابت حرف *) a->ڈیٹا;
        ثابت حرف * bb = (ثابت حرف *) b->ڈیٹا;
        واپس ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    برائے (صحیح j = گرہ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        اگر (ggml_is_خالی_خانے(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_ترتیب_بدلو || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            جاری;  // not a launch; cannot break a run of adjacent norms
        }
        اگر (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, گرہ) ||
            !ggml_are_same_shape(nj->src[0], گرہ->src[0])) {
            توڑو;  // any other launch ends the run
        }
        منطقی same_nb = سچ;
        برائے (صحیح d = 0; d < GGML_MAX_DIMS; ++d) {
            اگر (nj->nb[d] != گرہ->nb[d] || nj->src[0]->nb[d] != گرہ->src[0]->nb[d]) {
                same_nb = جھوٹ;
                توڑو;
            }
        }
        اگر (!same_nb) {
            توڑو;  // one nb[] stride set is shared by the whole batch
        }
        اعشاری epsj;
        حافظہ_نقل(&epsj, nj->op_params, حجم(اعشاری));
        اگر (epsj != eps0) {
            توڑو;  // eps mismatch ends the run
        }
        منطقی indep = سچ;
        برائے (صحیح k = 0; k < count; ++k) {
            اگر (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = جھوٹ;
                توڑو;
            }
        }
        اگر (!indep) {
            توڑو;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        آخری           = j;
    }
    اگر (count < 2) {
        واپس 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    واپس آخری - گرہ_idx;
}


__dpct_inline__ ساکن خالی k_copy_src1_مقصد_contiguous(
    ثابت حرف *__restrict__ src1_original, حرف *__restrict__ src1_contiguous,
    ثابت mmid_صف_mapping *__restrict__ صف_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت int32_t src1_صف = شے_ct1.get_گروہ(2);

    ثابت int32_t iid1 = صف_mapping[src1_صف].i2;
    ثابت int32_t id   = صف_mapping[src1_صف].i1;

    ثابت int64_t i11 = id % ne11;
    ثابت int64_t i12 = iid1;

    ثابت اعشاری * src1_صف_original = (ثابت اعشاری *)(src1_original + i11*nb11 + i12*nb12);
    اعشاری * src1_صف_contiguous = (اعشاری *)(src1_contiguous + src1_صف*nb11);

#ہدایت unroll
    برائے (صحیح i = شے_ct1.get_local_id(2); i < ne10;
         i += شے_ct1.get_local_range(2)) {
        src1_صف_contiguous[i] = src1_صف_original[i];
    }
}

__dpct_inline__ ساکن خالی k_copy_dst_from_contiguous(
    حرف *__restrict__ dst_original, ثابت حرف *__restrict__ dst_contiguous,
    ثابت mmid_صف_mapping *__restrict__ صف_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ثابت sycl::nd_شے<3> &شے_ct1) {
    int32_t i = شے_ct1.get_گروہ(2);

    ثابت int32_t i1 = صف_mapping[i].i1;
    ثابت int32_t i2 = صف_mapping[i].i2;

    ثابت اعشاری * dst_صف_contiguous = (ثابت اعشاری *)(dst_contiguous + i*nb1);
    اعشاری * dst_صف_original = (اعشاری *)(dst_original + i1*nb1 + i2*nb2);

#ہدایت unroll
    برائے (صحیح j = شے_ct1.get_local_id(2); j < ne0;
         j += شے_ct1.get_local_range(2)) {
        dst_صف_original[j] = dst_صف_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
ساکن منطقی ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
    ثابت ggml_tensor * src1, ثابت ggml_tensor * ids, ggml_tensor * dst)
{
    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    اگر (ne12 != 1) واپس جھوٹ;
    اگر (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) واپس جھوٹ;
    اگر (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) واپس جھوٹ;
    اگر (!ggml_is_contiguous(src1)) واپس جھوٹ;

    ثابت int64_t n_ids_فی_گروہ = ids->ne[0];
    اگر (ids->ne[1] != 1) واپس جھوٹ;
    اگر (ne11 != 1 && ne11 != n_ids_فی_گروہ) واپس جھوٹ;

    ثابت قطار_ptr stream           = ctx.stream();
    ثابت صحیح       src1_padded_ستون_تعداد = GGML_PAD((صحیح) ne10, میٹرکس_صف_PADDING);
    ثابت صحیح       n_experts_مستعمل   = (صحیح) n_ids_فی_گروہ;
    ثابت صحیح       nrows            = (صحیح) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ثابت ggml_tensor_extra_gpu * src0_extra =
        static_cast<ثابت ggml_tensor_extra_gpu *>(src0->extra);
    ثابت منطقی use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_ستون_تعداد * حجم(بلاک_q8_1) / QK8_1);
    حرف * src1_ddq = src1_q8_alloc.get();
    اگر (use_reorder) {
        quantize_صف_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ثابت اعشاری *) src1->ڈیٹا, src1_ddq, (صحیح) ne10, (صحیح) ne11,
            src1_padded_ستون_تعداد, stream);
    } ورنہ {
        quantize_صف_q8_1_sycl<quantize_q8_1>(
            (ثابت اعشاری *) src1->ڈیٹا, src1_ddq, (صحیح) ne10, (صحیح) ne11,
            src1_padded_ستون_تعداد, stream);
    }

    ثابت size_t بائٹیں_فی_qrow = (size_t) src1_padded_ستون_تعداد * حجم(بلاک_q8_1) / QK8_1;
    ثابت size_t src1_صف_stride = (ne11 == 1) ? 0 : بائٹیں_فی_qrow;

    اگر (use_reorder) {
        واپس ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->ڈیٹا, src1_ddq, (ثابت int32_t *) ids->ڈیٹا,
            (اعشاری *) dst->ڈیٹا, (صحیح) ne10, nrows, n_experts_مستعمل,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_صف_stride, stream);
    }
    واپس ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->ڈیٹا, src1_ddq, (ثابت int32_t *) ids->ڈیٹا,
        (اعشاری *) dst->ڈیٹا, (صحیح) ne10, nrows, n_experts_مستعمل,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_صف_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
ساکن خالی mmid_counting_ترتیب_دو_صفیں(
        ثابت ggml_tensor * ids, ثابت حرف * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_صفیں,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_صف_offsets,
        std::vector<mmid_صف_mapping> & routed_صف_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.تفویض(n_as, 0);
    برائے (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        برائے (int64_t id = 0; id < n_ids; id++) {
            ثابت int32_t صف_id_i = *(ثابت int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(صف_id_i >= 0 && صف_id_i < n_as);
            expert_counts[صف_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_صف_offsets.تفویض(n_as + 1, 0);
    برائے (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_صف_offsets[i02 + 1] = expert_صف_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_صف_اگلا = expert_صف_offsets;
    routed_صف_src.resize(n_routed_صفیں);
    برائے (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        برائے (int64_t id = 0; id < n_ids; id++) {
            ثابت int32_t صف_id_i = *(ثابت int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(صف_id_i >= 0 && صف_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ثابت int64_t routed_صف = expert_صف_اگلا[صف_id_i]++;
            GGML_ASSERT(routed_صف >= expert_صف_offsets[صف_id_i]);
            GGML_ASSERT(routed_صف < expert_صف_offsets[صف_id_i + 1]);
            routed_صف_src[routed_صف] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

ساکن خالی ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) کوشش {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_بانٹو(src0->بفر) && "mul_mat_id does not support split buffers");

    ثابت ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت قطار_ptr stream = ctx.stream();

    ثابت int64_t n_as = ne02;
    ثابت int64_t n_ids = ids->ne[0];

    اگر (ne12 == 1) {
        اگر (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            واپس;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    اگر (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<حرف> ids_host(ggml_nbytes(ids));
    ثابت حرف * ids_dev = (ثابت حرف *) ids->ڈیٹا;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->حافظہ_نقل(ids_host.ڈیٹا(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_صف = *src0;
    ggml_tensor src1_صف = *src1;
    ggml_tensor dst_صف = *dst;

    حرف *src0_original = (حرف *)src0->ڈیٹا;
    حرف *src1_original = (حرف *)src1->ڈیٹا;
    حرف *dst_original = (حرف *)dst->ڈیٹا;

    src0_صف.ne[2] = 1;
    src0_صف.ne[3] = 1;
    src0_صف.nb[3] = nb02;

    src1_صف.ne[1] = 1;
    src1_صف.ne[2] = 1;
    src1_صف.ne[3] = 1;
    src1_صف.nb[2] = nb11;
    src1_صف.nb[3] = nb11;

    dst_صف.ne[1] = 1;
    dst_صف.ne[2] = 1;
    dst_صف.ne[3] = 1;
    dst_صف.nb[2] = nb1;
    dst_صف.nb[3] = nb1;
    اگر (ne12 == 1) {
        برائے (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            برائے (int64_t id = 0; id < n_ids; id++) {
                ثابت int32_t i02 = *(ثابت int32_t *) (ids_host.ڈیٹا() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ثابت int64_t i11 = id % ne11;
                ثابت int64_t i12 = iid1;

                ثابت int64_t i1 = id;
                ثابت int64_t i2 = i12;

            src0_صف.ڈیٹا = src0_original + i02*nb02;
            src1_صف.ڈیٹا = src1_original + i11*nb11 + i12*nb12;
            dst_صف.ڈیٹا = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_صف, &src1_صف, &dst_صف);
            }
        }
    } ورنہ {
        ثابت int64_t n_routed_صفیں = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<حرف> src1_contiguous(ctx.pool(), حجم(اعشاری)*n_routed_صفیں*ne10);
        ggml_sycl_pool_alloc<حرف>  dst_contiguous(ctx.pool(), حجم(اعشاری)*n_routed_صفیں*ne0);

        src1_صف.ڈیٹا = src1_contiguous.get();
        dst_صف.ڈیٹا  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_صف_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_صف_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_صف_mapping> & routed_صف_src = ctx.mmid_صف_mapping_host;

        mmid_counting_ترتیب_دو_صفیں(ids, ids_host.ڈیٹا(), n_ids, n_as, n_routed_صفیں,
                                expert_صف_counts, expert_صف_offsets, routed_صف_src);

        ggml_sycl_pool_alloc<mmid_صف_mapping> dev_صف_mapping(ctx.pool(), n_routed_صفیں);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->حافظہ_نقل(dev_صف_mapping.get(), routed_صف_src.ڈیٹا(), n_routed_صفیں*حجم(mmid_صف_mapping))));

        ثابت غیرعلامتی صحیح max_work_گروہ_سائز = ggml_sycl_info().max_work_گروہ_sizes[ctx.device];
        أكد(max_work_گروہ_سائز % (WARP_سائز * WARP_سائز) == 0);

        {
            sycl::range<3> بلاک_dims(1, 1, std::min((غیرعلامتی صحیح)ne10, max_work_گروہ_سائز));
            sycl::range<3> grid_dims(1, 1, n_routed_صفیں);
            stream->submit([&](sycl::handler &cgh) {
                حرف *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_صف_mapping *__restrict dev_صف_mapping_get =
                    dev_صف_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                    [=](sycl::nd_شے<3> شے_ct1) {
                        k_copy_src1_مقصد_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_صف_mapping_get,
                            ne11, ne10, nb11, nb12,
                            شے_ct1);
                    });
            });
        }

        منطقی grouped = جھوٹ;
        اگر (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == حجم(اعشاری)*ne10 && nb1 == حجم(اعشاری)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            اگر (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ثابت منطقی src0_reordered =
                src0->extra && ((ثابت ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ثابت اعشاری *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (اعشاری *) dst_contiguous.get(),
                                                     expert_صف_offsets.ڈیٹا(), n_as, ne01, ne10, n_routed_صفیں,
                                                     ctx.mmid_tile_جدول_وقت_host, ctx.pool(), stream);
        }

        برائے (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ثابت int64_t عدد_src1_صفیں = expert_صف_counts[i02];

            اگر (عدد_src1_صفیں == 0) {
                جاری;
            }

            ثابت int64_t expert_صف_انحراف = expert_صف_offsets[i02];

            src0_صف.ڈیٹا = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == حجم(اعشاری)*ne10);
            GGML_ASSERT(nb1 == حجم(اعشاری)*ne0);
            src1_صف.ڈیٹا = src1_contiguous.get() + expert_صف_انحراف*nb11;
            src1_صف.ne[1] = عدد_src1_صفیں;

            src1_صف.nb[1] = nb11;
            src1_صف.nb[2] = عدد_src1_صفیں*nb11;
            src1_صف.nb[3] = عدد_src1_صفیں*nb11;

            dst_صف.ڈیٹا = dst_contiguous.get() + expert_صف_انحراف*nb1;
            dst_صف.ne[1] = عدد_src1_صفیں;
            dst_صف.nb[1] = nb1;
            dst_صف.nb[2] = عدد_src1_صفیں*nb1;
            dst_صف.nb[3] = عدد_src1_صفیں*nb1;

            ggml_sycl_mul_mat(ctx, &src0_صف, &src1_صف, &dst_صف);
        }

        {
            sycl::range<3> بلاک_dims(1, 1, std::min((غیرعلامتی صحیح)ne0, max_work_گروہ_سائز));
            sycl::range<3> grid_dims(1, 1, n_routed_صفیں);
            stream->submit([&](sycl::handler &cgh) {
                ثابت حرف *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ثابت mmid_صف_mapping *__restrict dev_صف_mapping_get =
                    dev_صف_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * بلاک_dims, بلاک_dims),
                    [=](sycl::nd_شے<3> شے_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_صف_mapping_get,
                                                   ne0, nb1, nb2, شے_ct1);
                    });
            });
        }
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

ساکن خالی ggml_sycl_diag_نقاب_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_نقاب_inf(ctx, dst);
}

ساکن خالی ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

ساکن خالی ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

ساکن خالی ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

ساکن خالی ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

ساکن خالی ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

ساکن خالی ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

ساکن خالی ggml_sycl_مجموع(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_مجموع(ctx, dst);
}

ساکن خالی ggml_sycl_مجموع_صفیں(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_مجموع_صفیں(ctx, dst);
}

ساکن خالی ggml_sycl_اوسط(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_اوسط(ctx, dst);
}

ساکن خالی ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

ساکن خالی ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


ساکن خالی ggml_sycl_سیٹ_main_device(ثابت صحیح main_device) کوشش {
    اگر (dpct::get_current_device_id() == static_cast<غیرعلامتی صحیح> (main_device)) {
        واپس;
    }
    check_allow_gpu_اشاریہ(main_device);
    dpct::select_device(main_device);

    اگر (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن منطقی ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ساخت ggml_tensor * dst) کوشش {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    اگر (!g_sycl_loaded) واپس جھوٹ;

    اگر (dst->src[0] != خالی_اشاریہ && ggml_backend_بفر_is_sycl_بانٹو(dst->src[0]->بفر)) {
        ggml_sycl_سیٹ_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    انتخاب (dst->op) {
        حالت GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            توڑو;
        حالت GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            توڑو;
        حالت GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            توڑو;
        حالت GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            توڑو;
        حالت GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            توڑو;
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            توڑو;
        حالت GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            توڑو;
        حالت GGML_OP_REPEAT_عقب:
            ggml_sycl_repeat_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_GET_صفیں:
            ggml_sycl_get_صفیں(ctx, dst);
            توڑو;
        حالت GGML_OP_GET_صفیں_عقب:
            ggml_sycl_get_صفیں_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_سیٹ:
            ggml_sycl_op_سیٹ(ctx, dst);
            توڑو;
        حالت GGML_OP_سیٹ_صفیں:
            ggml_sycl_op_سیٹ_صفیں(ctx, dst);
            توڑو;
        حالت GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            توڑو;
        حالت GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            توڑو;
        حالت GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            توڑو;
        حالت GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            توڑو;
        حالت GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            توڑو;
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_جوڑو(ctx, dst);
            توڑو;
        حالت GGML_OP_جوڑو_ID:
            ggml_sycl_جوڑو_id(ctx, dst);
            توڑو;
        حالت GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            توڑو;
        حالت GGML_OP_گنتی_EQUAL:
            ggml_sycl_گنتی_equal(ctx, dst);
            توڑو;
        حالت GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            توڑو;
        حالت GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            توڑو;
        حالت GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            توڑو;
        حالت GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            توڑو;
        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(dst)) {
                حالت GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_قدم:
                    ggml_sycl_قدم(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_GELU_تیز:
                    ggml_sycl_gelu_تیز(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_باری:
                    ggml_sycl_باری(ctx, dst);
                    توڑو;
                حالت GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    توڑو;
                طےشدہ:
                    واپس جھوٹ;
            }
            توڑو;
        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(dst)) {
                حالت GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    توڑو;
                حالت GGML_GLU_OP_GEGLU_تیز:
                    ggml_sycl_geglu_تیز(ctx, dst);
                    توڑو;
                طےشدہ:
                    واپس جھوٹ;
            }
            توڑو;
        حالت GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            توڑو;
        حالت GGML_OP_گروہ_NORM:
            ggml_sycl_گروہ_norm(ctx, dst);
            توڑو;
        حالت GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            توڑو;
        حالت GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            توڑو;
        حالت GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            توڑو;
        حالت GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            توڑو;
        حالت GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            توڑو;
        حالت GGML_OP_RMS_NORM_عقب:
            ggml_sycl_rms_norm_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            توڑو;
        حالت GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            توڑو;
        حالت GGML_OP_MUL_MAT:
            اگر (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                واپس جھوٹ;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            توڑو;
        حالت GGML_OP_MUL_MAT_ID:
            اگر (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                واپس جھوٹ;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            توڑو;
        حالت GGML_OP_برآمد_PROD:
            ggml_sycl_op_برآمد_prod(ctx, dst);
            توڑو;
        حالت GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            توڑو;
        حالت GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            توڑو;
        حالت GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            توڑو;
        حالت GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            توڑو;
        حالت GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            توڑو;
        حالت GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            توڑو;
        حالت GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            توڑو;
        حالت GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            توڑو;
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            توڑو;
        حالت GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            توڑو;
        حالت GGML_OP_DIAG_نقاب_INF:
            ggml_sycl_diag_نقاب_inf(ctx, dst);
            توڑو;
        حالت GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            توڑو;
        حالت GGML_OP_SOFT_MAX_عقب:
            ggml_sycl_op_soft_max_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            توڑو;
        حالت GGML_OP_CROSS_ENTROPY_LOSS_عقب:
            ggml_sycl_cross_entropy_loss_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            توڑو;
        حالت GGML_OP_ROPE_عقب:
            ggml_sycl_rope_عقب(ctx, dst);
            توڑو;
        حالت GGML_OP_IM2_ستون:
            ggml_sycl_im2col(ctx, dst);
            توڑو;
        حالت GGML_OP_IM2_ستون_3_D:
            ggml_sycl_im2col_3d(ctx, dst);
            توڑو;
        حالت GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            توڑو;
        حالت GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            توڑو;
        حالت GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            توڑو;
        حالت GGML_OP_مجموع:
            ggml_sycl_مجموع(ctx, dst);
            توڑو;
        حالت GGML_OP_مجموع_صفیں:
            ggml_sycl_مجموع_صفیں(ctx, dst);
            توڑو;
        حالت GGML_OP_اوسط:
            ggml_sycl_اوسط(ctx, dst);
            توڑو;
        حالت GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            توڑو;
        حالت GGML_OP_چوٹی_K:
            ggml_sycl_op_چوٹی_k(ctx, dst);
            توڑو;
        حالت GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            توڑو;
        حالت GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            توڑو;
        حالت GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            توڑو;
        حالت GGML_OP_GATED_خطی_ATTN:
            ggml_sycl_op_gated_خطی_attn(ctx, dst);
            توڑو;
        حالت GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            توڑو;
        حالت GGML_OP_OPT_قدم_ADAMW:
            ggml_sycl_opt_قدم_adamw(ctx, dst);
            توڑو;
        حالت GGML_OP_OPT_قدم_SGD:
            ggml_sycl_opt_قدم_sgd(ctx, dst);
            توڑو;
        حالت GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            توڑو;
        حالت GGML_OP_SSM_جائزہ:
            ggml_sycl_ssm_جائزہ(ctx, dst);
            توڑو;
        حالت GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            توڑو;
        حالت GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            توڑو;
        حالت GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            توڑو;
        حالت GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            توڑو;
        حالت GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            توڑو;
        حالت GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            توڑو;
        طےشدہ:
            واپس جھوٹ;
    }

    واپس سچ;
} پکڑو (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::سطر_ختم;
    std::باہر_نکلو(1);
}

GGML_API خالی ggml_backend_sycl_get_device_description(صحیح device, حرف *description,
                                      size_t description_سائز) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_سائز, "%s", prop.get_name());
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن منطقی sycl_get_mem_info(صحیح device, size_t * آزاد, size_t * کل) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    حافظہ_APIType mem_api_type = حافظہ_APIType::حافظہ_API_TYPE_SYCL;

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_نتیجہ_SUCCESS ?
        (حافظہ_APIType) g_ggml_sycl_get_mem_api : حافظہ_APIType::حافظہ_API_TYPE_SYCL;
#ورنہ
    mem_api_type = حافظہ_APIType::حافظہ_API_TYPE_SYCL;
#اگر_ختم
    منطقی res = get_حافظہ_سائز(dpct::dev_mgr::instance().get_device(device),
        *آزاد, *کل, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *کل, *آزاد);
    واپس res;
}

خالی ggml_backend_sycl_get_device_حافظہ(صحیح device, size_t * آزاد, size_t * کل) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    اگر (!sycl_get_mem_info(device, آزاد, کل)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *آزاد, *کل);
} پکڑو (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

ساکن ثابت حرف * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    واپس sycl_ctx->name.c_str();
}

ساکن خالی ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    حذف sycl_ctx;
    حذف backend;
}

ساکن خالی ggml_backend_sycl_سیٹ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ثابت خالی *ڈیٹا, size_t انحراف,
                                               size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) && "unsupported buffer type");
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->حافظہ_نقل((حرف *)tensor->ڈیٹا + انحراف, ڈیٹا, size)));
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ثابت ggml_tensor *tensor,
                                               خالی *ڈیٹا, size_t انحراف,
                                               size_t size) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, انحراف);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) && "unsupported buffer type");
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->حافظہ_نقل(
        ڈیٹا, (ثابت حرف *)tensor->ڈیٹا + انحراف, size)));
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن منطقی ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ثابت ggml_tensor *src,
                                               ggml_tensor *dst) کوشش {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    منطقی is_cpy_supported                = dst->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) &&
                            ggml_backend_بفر_is_sycl(src->بفر);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    اگر (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->حافظہ_نقل(
            dst->ڈیٹا, src->ڈیٹا, ggml_nbytes(dst))));
        واپس سچ;
    }

    واپس جھوٹ;
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_synchronize(ggml_backend_t backend) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_غیر_مستعمل(backend);
}
پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن منطقی ggml_sycl_is_view_or_noop(ثابت ggml_tensor * t) {
    واپس ggml_is_خالی_خانے(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_ترتیب_بدلو || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
ساکن صحیح ggml_sycl_try_gdn_cache_fusion(ثابت ggml_cgraph * cgraph, صحیح گرہ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    اگر (!g_ggml_sycl_enable_fusion) {
        واپس 0;
    }

    ثابت ggml_tensor * gdn = cgraph->nodes[گرہ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    اگر (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        واپس 0;
    }

    ثابت ggml_tensor * src_v     = gdn->src[2];
    ثابت int64_t       S_v       = src_v->ne[0];
    ثابت int64_t       H         = src_v->ne[1];
    ثابت int64_t       n_علامات  = src_v->ne[2];
    ثابت int64_t       n_seqs    = src_v->ne[3];
    ثابت int64_t       D         = S_v * S_v * H;
    ثابت int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ثابت int64_t       n_written = std::min<int64_t>(n_علامات, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ثابت size_t دم_off = ggml_صف_سائز(GGML_TYPE_F32, S_v * H * n_علامات * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ثابت ggml_tensor * cpy  = خالی_اشاریہ;
    صحیح                 skip = 0;
    برائے (صحیح j = گرہ_idx + 1; j < cgraph->n_nodes && cpy == خالی_اشاریہ; ++j) {
        ثابت ggml_tensor * n = cgraph->nodes[j];
        اگر (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            جاری;
        }
        اگر (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            واپس 0;
        }
        cpy  = n;
        skip = j - گرہ_idx;
    }
    اگر (cpy == خالی_اشاریہ) {
        واپس 0;
    }

    ثابت ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ثابت ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    اگر (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != دم_off ||
        !ggml_is_contiguous(src)) {
        واپس 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ثابت std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    اگر (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->ڈیٹا == خالی_اشاریہ ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_سائز(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_صف_سائز(GGML_TYPE_F32, D)) {
        واپس 0;
    }

    fused_state_cpy.ڈیٹا        = (اعشاری *) dst->ڈیٹا; // rollback group 0 (newest)
    fused_state_cpy.خانہ_stride = K > 1 ? (int64_t) (dst->nb[2] / حجم(اعشاری)) : 0;
    واپس skip;
}

ساکن خالی ggml_backend_sycl_گراف_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_سیٹ_main_device(sycl_ctx->device);

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];
        اگر (ggml_sycl_is_view_or_noop(گرہ)) {
            جاری;
        }
        اگر ((گرہ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            جاری;
        }

        ثابت صحیح nodes_مقصد_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        اگر (nodes_مقصد_skip != 0) {
            i += nodes_مقصد_skip;
            جاری;
        }
#اگر_غیر_تعریف_شدہ NDEBUG
        أكد(گرہ->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device));
        برائے (صحیح j = 0; j < GGML_MAX_SRC; j++) {
            اگر (گرہ->src[j] != خالی_اشاریہ) {
                أكد(گرہ->src[j]->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device));
            }
        }
#اگر_ختم
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        اگر (گرہ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ثابت صحیح gdn_nodes_مقصد_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            اگر (gdn_nodes_مقصد_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, گرہ, fused_state_cpy);
                i += gdn_nodes_مقصد_skip;
                جاری;
            }
        }
        اگر (گرہ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڑو }, {})) {
            ggml_sycl_op_rms_norm_fused_جوڑو(*sycl_ctx, گرہ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            جاری;
        }
        اگر (گرہ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;
            جاری;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        اگر (گرہ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;
            جاری;
        }
        اگر (گرہ->op == GGML_OP_جوڑو &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_جوڑو, GGML_OP_جوڑو }, {})) {
            ggml_sycl_op_جوڑو_جوڑو_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;
            جاری;
        }
        اگر (گرہ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(گرہ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;
            جاری;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        اگر (گرہ->op == GGML_OP_جوڑو && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_جوڑو, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_جوڑو_unary_mul_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            جاری;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        اگر (گرہ->op == GGML_OP_L2_NORM) {
            ثابت صحیح l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            اگر (l2_batch_skip > 0) {
                i += l2_batch_skip;
                جاری;
            }
        }

        اگر (گرہ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_جوڑو, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, گرہ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            جاری;
        }

        اگر (گرہ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, گرہ, خالی_اشاریہ, cgraph->nodes[i + 1]);
            i++;
            جاری;
        }

        اگر (گرہ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            جاری;
        }

        منطقی ok = ggml_sycl_compute_forward(*sycl_ctx, گرہ);
        اگر (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, گرہ->name, ggml_op_name(گرہ->op));
        }
        GGML_ASSERT(ok);
    }
}

#اگر_تعریف_شدہ GGML_SYCL_گراف
ساکن منطقی check_گراف_compatibility(ggml_cgraph * cgraph) {
    اگر (ggml_sycl_info().device_گنتی > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        واپس جھوٹ;
    }

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ثابت ggml_op گرہ_op = cgraph->nodes[i]->op;
        انتخاب (گرہ_op) {
            طےشدہ:
                توڑو;
            حالت GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            حالت GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(گرہ_op));
                واپس جھوٹ;
            حالت GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                اگر (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(گرہ_op));
                    واپس جھوٹ;
                }
        }
    }
    واپس سچ;
}
#اگر_ختم

ساکن ggml_حالت ggml_backend_sycl_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    خودکار * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#اگر_تعریف_شدہ GGML_SYCL_گراف
    منطقی use_sycl_گراف = جھوٹ;
    اگر (g_ggml_sycl_enable_گراف) {
        use_sycl_گراف = check_گراف_compatibility(cgraph);
    }
    اگر (use_sycl_گراف) {
        ثابت منطقی گراف_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_گراف);
        اگر (!گراف_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
            واپس GGML_حالت_SUCCESS;
        }

        sycl_ex::command_گراف model_sycl_گراف(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_بفر_outlives_گراف{}});

        model_sycl_گراف.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
        model_sycl_گراف.انتہا_recording();

        ثابت منطقی گراف_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_گراف);
        اگر (!sycl_ctx->exec_گراف || !گراف_update_support) {
            خودکار exec_گراف = گراف_update_support ? model_sycl_گراف.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_گراف.finalize();
            sycl_ctx->exec_گراف = std::make_unique<
                sycl_ex::command_گراف<sycl_ex::گراف_state::executable>>(exec_گراف);
        } ورنہ {
            کوشش {
                sycl_ctx->exec_گراف->update(model_sycl_گراف);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } پکڑو (sycl::exception ثابت & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                خودکار exec_گراف = model_sycl_گراف.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_گراف = std::make_unique<
                    sycl_ex::command_گراف<sycl_ex::گراف_state::executable>>(exec_گراف);
            }
        }

        sycl_ctx->stream()->ext_oneapi_گراف(*(sycl_ctx->exec_گراف));
    } ورنہ
#اگر_ختم
    {
        ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
    }
    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_sycl_event_ریکارڈ(ggml_backend_t backend, ggml_backend_event_t event)
کوشش
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ثابت قطار_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
پکڑو (sycl::exception ثابت &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

ساکن خالی ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) کوشش {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    اگر (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } ورنہ
        GGML_ABORT("fatal error");
} پکڑو (sycl::exception ثابت& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::سطر_ختم;
    std::باہر_نکلو(1);
}

ساکن ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_sycl_گراف_compute,
    /* .event_record            = */ ggml_backend_sycl_event_ریکارڈ,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ تہی,
};

ساکن ggml_guid_t ggml_backend_sycl_guid() {
    ساکن ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    واپس &guid;
}

منطقی ggml_backend_is_sycl(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

صحیح ggml_backend_sycl_get_device_گنتی(خالی) {
    واپس ggml_sycl_info().device_گنتی;
}


// backend device

ساکن ثابت حرف * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس ctx->name.c_str();
}

ساکن ثابت حرف * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس ctx->description.c_str();
}

ساکن خالی ggml_backend_sycl_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    اگر (!sycl_get_mem_info(ctx->device, آزاد, کل)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *کل, *آزاد);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *آزاد, *کل);
}

ساکن شمار ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ثابت sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    اگر (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        واپس GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساکن خالی ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);

    منطقی host_بفر = getenv("GGML_SYCL_NO_PINNED") == خالی_اشاریہ;
#اگر_تعریف_شدہ GGML_SYCL_NO_PEER_COPY
    منطقی events = جھوٹ;
#ورنہ
    منطقی events = سچ;
#اگر_ختم

    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ host_بفر,
        /* .buffer_from_host_ptr  = */ جھوٹ,
        /* .events                = */ events,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_غیر_مستعمل(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس ggml_backend_sycl_init(ctx->device);
}

ساکن ggml_backend_بفر_type_t ggml_backend_sycl_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس ggml_backend_sycl_بفر_type(ctx->device);
}

ساکن ggml_backend_بفر_type_t ggml_backend_sycl_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    واپس ggml_backend_sycl_host_بفر_type_for_device(ctx->device);
}

ساکن ggml_backend_بفر_t ggml_backend_sycl_device_بفر_from_host_ptr(ggml_backend_dev_t dev, خالی * ptr, size_t size, size_t max_tensor_سائز) {
    GGML_غیر_مستعمل(dev);
    GGML_غیر_مستعمل(ptr);
    GGML_غیر_مستعمل(size);
    GGML_غیر_مستعمل(max_tensor_سائز);
    واپس خالی_اشاریہ;
}

ساکن منطقی do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    صحیح device = sycl_ctx->device;
    انتخاب (op->op) {
        حالت GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                اگر (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    واپس سچ;
                }
                واپس جھوٹ;
            }
        حالت GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            واپس (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_سائز(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_سائز(op->src[0]->type));
        حالت GGML_OP_CONV_2D_DW:
        حالت GGML_OP_CONV_TRANSPOSE_2D:
            واپس سچ;
        حالت GGML_OP_UNARY:
            انتخاب (ggml_get_unary_op(op)) {
                حالت GGML_UNARY_OP_SGN:
                حالت GGML_UNARY_OP_ABS:
                حالت GGML_UNARY_OP_NEG:
                حالت GGML_UNARY_OP_قدم:
                حالت GGML_UNARY_OP_RELU:
                حالت GGML_UNARY_OP_HARDSIGMOID:
                حالت GGML_UNARY_OP_TANH:
                حالت GGML_UNARY_OP_GELU:
                حالت GGML_UNARY_OP_SILU:
                حالت GGML_UNARY_OP_SIGMOID:
                حالت GGML_UNARY_OP_HARDSWISH:
                حالت GGML_UNARY_OP_GELU_تیز:
                حالت GGML_UNARY_OP_GELU_ERF:
                حالت GGML_UNARY_OP_EXP:
                حالت GGML_UNARY_OP_EXPM1:
                حالت GGML_UNARY_OP_SOFTPLUS:
                حالت GGML_UNARY_OP_ELU:
                حالت GGML_UNARY_OP_XIELU:
                حالت GGML_UNARY_OP_CEIL:
                    واپس سچ;
                حالت GGML_UNARY_OP_FLOOR:
                حالت GGML_UNARY_OP_باری:
                حالت GGML_UNARY_OP_TRUNC:
                    واپس سچ;
                طےشدہ:
                    واپس جھوٹ;
            }
        حالت GGML_OP_GLU:
            انتخاب (ggml_get_glu_op(op)) {
                حالت GGML_GLU_OP_REGLU:
                حالت GGML_GLU_OP_GEGLU:
                حالت GGML_GLU_OP_SWIGLU:
                حالت GGML_GLU_OP_SWIGLU_OAI:
                حالت GGML_GLU_OP_GEGLU_ERF:
                حالت GGML_GLU_OP_GEGLU_تیز:
                حالت GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    واپس ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                طےشدہ:
                    واپس جھوٹ;
            }
        حالت GGML_OP_MUL_MAT:
        حالت GGML_OP_MUL_MAT_ID:
            {
                ساخت ggml_tensor * a = op->src[0];
                ساخت ggml_tensor * b = op->src[1];

                اگر (a->ne[3] != b->ne[3]) {
                    واپس جھوٹ;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                اگر (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  واپس جھوٹ;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                اگر (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    واپس جھوٹ;
                }

                اگر (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    واپس جھوٹ;
                }

                واپس سچ;
            }
        حالت GGML_OP_برآمد_PROD:
            واپس op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        حالت GGML_OP_GET_صفیں:
            {
                انتخاب (op->src[0]->type) {
                    حالت GGML_TYPE_I32:
                    حالت GGML_TYPE_F16:
                    حالت GGML_TYPE_BF16:
                    حالت GGML_TYPE_F32:
                    حالت GGML_TYPE_Q1_0:
                    حالت GGML_TYPE_MXFP4:
                    حالت GGML_TYPE_NVFP4:
                    حالت GGML_TYPE_IQ2_XXS:
                    حالت GGML_TYPE_IQ2_XS:
                    حالت GGML_TYPE_IQ2_S:
                    حالت GGML_TYPE_IQ3_XXS:
                    حالت GGML_TYPE_IQ1_S:
                    حالت GGML_TYPE_IQ1_M:
                    حالت GGML_TYPE_IQ3_S:
                    حالت GGML_TYPE_IQ4_NL:
                    حالت GGML_TYPE_IQ4_XS:
                    حالت GGML_TYPE_Q2_K:
                    حالت GGML_TYPE_Q3_K:
                    حالت GGML_TYPE_Q4_0:
                    حالت GGML_TYPE_Q4_1:
                    حالت GGML_TYPE_Q4_K:
                    حالت GGML_TYPE_Q5_0:
                    حالت GGML_TYPE_Q5_1:
                    حالت GGML_TYPE_Q5_K:
                    حالت GGML_TYPE_Q6_K:
                    حالت GGML_TYPE_Q8_0:
                        واپس سچ;
                    طےشدہ:
                        واپس جھوٹ;
                }
            }
        حالت GGML_OP_GET_صفیں_عقب:
            // return true;
             واپس op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         حالت GGML_OP_سیٹ:
               واپس (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        حالت GGML_OP_سیٹ_صفیں:
            {
                اگر (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    واپس جھوٹ;
                }
                خودکار res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                واپس res;
            }
        حالت GGML_OP_DSV4_HC_PRE:
            واپس op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        حالت GGML_OP_DSV4_HC_COMB:
            واپس op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالت GGML_OP_DSV4_HC_POST:
            واپس op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == خالی_اشاریہ || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        حالت GGML_OP_LIGHTNING_INDEXER:
            واپس op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_سائز * 8;
        حالت GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                اگر (src0_type == GGML_TYPE_F16) {
                    اگر (src1_type == GGML_TYPE_Q2_K ||
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
                        واپس جھوٹ;
                    }
                }

                اگر (src0_type == GGML_TYPE_BF16) {
                    اگر (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        واپس جھوٹ;
                    }
                }

                اگر (src0_type == GGML_TYPE_F32) {
                    اگر (src1_type == GGML_TYPE_Q2_K ||
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
                        واپس جھوٹ;
                    }
                }

                اگر (src1_type == GGML_TYPE_F32) {
                    اگر (src0_type == GGML_TYPE_Q1_0 ||
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
                        واپس جھوٹ;
                    }
                }

                اگر (src0_type == src1_type) {
                    اگر (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        واپس جھوٹ;
                    }
                }

                اگر (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    واپس جھوٹ;
                }

                واپس سچ;
            }
        حالت GGML_OP_REPEAT_عقب:
            {
                ggml_type src0_type = op->src[0]->type;
                واپس src0_type == GGML_TYPE_F32;
            }
        حالت GGML_OP_CONCAT:
        حالت GGML_OP_DUP:
        حالت GGML_OP_ARGMAX:
        حالت GGML_OP_NONE:
        حالت GGML_OP_RESHAPE:
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_ADD1:
        حالت GGML_OP_جوڑو_ID:
        حالت GGML_OP_SUB:
        حالت GGML_OP_گنتی_EQUAL:
        حالت GGML_OP_MUL:
        حالت GGML_OP_DIV:
        حالت GGML_OP_REPEAT:
            واپس سچ;
        حالت GGML_OP_PAD_REFLECT_1D:
            واپس ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالت GGML_OP_SQR:
        حالت GGML_OP_SQRT:
        حالت GGML_OP_SIN:
        حالت GGML_OP_COS:
        حالت GGML_OP_CLAMP:
        حالت GGML_OP_LOG:
        حالت GGML_OP_NORM:
        حالت GGML_OP_L2_NORM:
        حالت GGML_OP_گروہ_NORM:
        حالت GGML_OP_RMS_NORM:
            واپس سچ;
        حالت GGML_OP_RMS_NORM_عقب:
            واپس ggml_is_contiguous(op->src[0]);
        حالت GGML_OP_SCALE:
            // the scale kernel is f32 only
            واپس op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالت GGML_OP_CONT:
            واپس سچ;
        حالت GGML_OP_TRI:
            {
                ثابت ggml_tensor * src0 = op->src[0];
                واپس src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        حالت GGML_OP_DIAG_نقاب_INF:
            واپس سچ;
        حالت GGML_OP_SOFT_MAX:
            واپس سچ;
        حالت GGML_OP_SOFT_MAX_عقب: {
            اعشاری max_bias = 0.0f;
            حافظہ_نقل(&max_bias, (ثابت اعشاری *) op->op_params + 1, حجم(اعشاری));
            واپس max_bias == 0.0f;
        }
        حالت GGML_OP_ROPE:
        حالت GGML_OP_ROPE_عقب:
        حالت GGML_OP_IM2_ستون:
        حالت GGML_OP_IM2_ستون_3_D:
        حالت GGML_OP_UPSCALE:
            واپس سچ;
        حالت GGML_OP_COL2IM_1D:
            واپس ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#اگر_تعریف_شدہ GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#اگر_ختم
                   ) &&
                   op->src[0]->type == op->type;
        حالت GGML_OP_CONV_3D: {
            ثابت int32_t * opts = (ثابت int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ثابت int64_t knl_n_کل = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ثابت int64_t patch_کل = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            واپس op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_کل > 0 && patch_کل > 0;
        }
        حالت GGML_OP_مجموع:
        حالت GGML_OP_مجموع_صفیں:
        حالت GGML_OP_اوسط:
            واپس ggml_is_contiguous(op->src[0]);
        حالت GGML_OP_ARGSORT:
            واپس سچ;
        حالت GGML_OP_چوٹی_K: {
            ثابت ggml_tensor * src0 = op->src[0];
            ثابت صحیح k = op->ne[0];
            واپس src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        حالت GGML_OP_POOL_2D:
        حالت GGML_OP_POOL_1D:
        حالت GGML_OP_ACC:
            واپس سچ;
        حالت GGML_OP_PAD:
            اگر (ggml_get_op_params_i32(op, 8) != 0) {
                واپس جھوٹ;
            }
            واپس سچ;
        حالت GGML_OP_LEAKY_RELU:
        حالت GGML_OP_TIMESTEP_EMBEDDING:
        حالت GGML_OP_RWKV_WKV6:
        حالت GGML_OP_RWKV_WKV7:
        حالت GGML_OP_GATED_خطی_ATTN:
        حالت GGML_OP_GATED_DELTA_NET:
        حالت GGML_OP_OPT_قدم_ADAMW:
        حالت GGML_OP_OPT_قدم_SGD:
            واپس سچ;
        حالت GGML_OP_SSM_CONV:
            واپس op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        حالت GGML_OP_ROLL:
            واپس op->type == GGML_TYPE_F32;
        حالت GGML_OP_ARANGE:
            واپس op->type == GGML_TYPE_F32;
        حالت GGML_OP_SSM_جائزہ:
            اگر (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                واپس (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_سائز == 0;
            } ورنہ {
                // TODO Mamba-1 not yet ported to SYCL
                واپس جھوٹ;
            }
        حالت GGML_OP_FILL:
        حالت GGML_OP_CUMSUM:
        حالت GGML_OP_DIAG:
        حالت GGML_OP_CROSS_ENTROPY_LOSS:
        حالت GGML_OP_CROSS_ENTROPY_LOSS_عقب:
            واپس سچ;
        حالت GGML_OP_SOLVE_TRI:
            واپس op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        حالت GGML_OP_FLASH_ATTN_EXT:
            واپس ggml_sycl_flash_attn_ext_supported(device, op);
        طےشدہ:
            واپس جھوٹ;
    }

    GGML_غیر_مستعمل(dev);
}

ساکن منطقی ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    منطقی res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    واپس res;
}

ساکن منطقی ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    اگر (buft->iface.get_name != ggml_backend_sycl_بفر_type_get_name) {
        واپس جھوٹ;
    }
    ggml_backend_sycl_بفر_type_context * buft_ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس buft_ctx->device == sycl_ctx->device;
}

ساکن int64_t get_op_batch_سائز(ثابت ggml_tensor * op) {
    انتخاب (op->op) {
        حالت GGML_OP_GET_صفیں:
            واپس 0;
        حالت GGML_OP_MUL_MAT:
            واپس op->ne[1];
        حالت GGML_OP_MUL_MAT_ID:
        حالت GGML_OP_ROPE:
            واپس op->ne[2];
        طےشدہ:
            واپس ggml_nrows(op);
    }
}

ساکن منطقی ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    واپس get_op_batch_سائز(op) >= sycl_ctx->op_offload_min_batch_سائز;
}

ساکن ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#اگر_تعریف_شدہ GGML_SYCL_NO_PEER_COPY
    واپس خالی_اشاریہ;
#ورنہ
  sycl::event *event_ptr = نیا sycl::event();

  واپس نیا ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#اگر_ختم
}

ساکن خالی ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) کوشش {
  GGML_غیر_مستعمل(dev);
  اگر (event == خالی_اشاریہ) {
    واپس;
  }

  اگر (event->context != خالی_اشاریہ) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    حذف sycl_event;
    event->context = خالی_اشاریہ;
  }

  حذف event;
} پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}


ساکن خالی ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) کوشش {
  GGML_غیر_مستعمل(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} پکڑو (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سطر_ختم;
  std::باہر_نکلو(1);
}

ساکن ثابت ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_حافظہ,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_بفر_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_بفر_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_بفر_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

ساخت ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ساکن ثابت حرف * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_غیر_مستعمل(reg);
    واپس GGML_SYCL_NAME;
}

ساکن size_t ggml_backend_sycl_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    واپس ctx->devices.size();
}

ساکن ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t اشاریہ) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(اشاریہ < ctx->devices.size());
    واپس ctx->devices[اشاریہ];
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

ساخت ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_برآمد[2] = { خالی_اشاریہ, خالی_اشاریہ };
    uint8_t * host_اندر[2]  = { خالی_اشاریہ, خالی_اشاریہ };
    size_t    host_بائٹیں  = 0;

    خالی free_host() {
        برائے (صحیح i = 0; i < 2; ++i) {
            ثابت sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            اگر (host_برآمد[i]) { sycl::آزاد(host_برآمد[i], sctx); host_برآمد[i] = خالی_اشاریہ; }
            اگر (host_اندر[i])  { sycl::آزاد(host_اندر[i],  sctx); host_اندر[i]  = خالی_اشاریہ; }
        }
        host_بائٹیں = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
ساکن خالی ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, قطار_ptr q0, قطار_ptr q1,
                                    خالی * dst0, ثابت خالی * src1, خالی * dst1, ثابت خالی * src0,
                                    size_t nbytes) {
    اگر (comm->host_بائٹیں < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_برآمد[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_اندر[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_برآمد[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_اندر[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_برآمد[0] && comm->host_اندر[0] && comm->host_برآمد[1] && comm->host_اندر[1]);
        comm->host_بائٹیں = nbytes;
    }

    sycl::event e0 = q0->حافظہ_نقل(comm->host_برآمد[0], src0, nbytes);
    sycl::event e1 = q1->حافظہ_نقل(comm->host_برآمد[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::حافظہ_نقل(comm->host_اندر[0], comm->host_برآمد[1], nbytes);
    std::حافظہ_نقل(comm->host_اندر[1], comm->host_برآمد[0], nbytes);
    q0->حافظہ_نقل(dst0, comm->host_اندر[0], nbytes);
    q1->حافظہ_نقل(dst1, comm->host_اندر[1], nbytes);
}

خالی * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) کوشش {
    برائے (size_t i = 0; i < n_backends; ++i) {
        اگر (!ggml_backend_is_sycl(backends[i])) {
            واپس خالی_اشاریہ;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    اگر (n_backends != 2) {
        واپس خالی_اشاریہ;
    }

    خودکار * ctx = نیا ggml_backend_sycl_comm_context;
    ctx->backends.تفویض(backends, backends + n_backends);
    خودکار * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    خودکار * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    واپس ctx;
}
پکڑو (ثابت sycl::exception &) { واپس خالی_اشاریہ; }
پکڑو (...)                     { واپس خالی_اشاریہ; }

خالی ggml_backend_sycl_comm_free(خالی * comm_ctx_v) {
    خودکار * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    اگر (comm_ctx == خالی_اشاریہ) {
        واپس;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    اگر (comm_ctx->backends.size() == 2) {
        خودکار * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        خودکار * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        کوشش {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } پکڑو (...) { /* best effort during shutdown */ }
    }

    حذف comm_ctx;
}

منطقی ggml_backend_sycl_comm_allreduce_tensor(خالی * comm_ctx_v, ساخت ggml_tensor ** tensors) کوشش {
    اگر (comm_ctx_v == خالی_اشاریہ) {
        واپس جھوٹ;
    }

    خودکار * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ثابت size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    اگر (n_backends != 2) {
        واپس جھوٹ;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    اگر (tensors[0]->type != tensors[1]->type) {
        واپس جھوٹ;
    }
    اگر (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        واپس جھوٹ;
    }
    اگر (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        واپس جھوٹ;
    }
    اگر (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        واپس جھوٹ;
    }

    ثابت int64_t nelem  = ggml_nelements(tensors[0]);
    ثابت size_t  nbytes = ggml_nbytes(tensors[0]);
    اگر (nelem == 0) {
        واپس سچ;
    }

    خودکار * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    خودکار * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    قطار_ptr q0 = ctx0->stream();
    قطار_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    اگر (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    اگر (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->ڈیٹا;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->ڈیٹا;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->ڈیٹا, f16_tmp1, tensors[0]->ڈیٹا, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((اعشاری) f16_out0[i] + (اعشاری) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((اعشاری) f16_out1[i] + (اعشاری) f16_tmp1[i]);
            });
        });
        واپس سچ;
    }

    اعشاری * out0 = (اعشاری *) tensors[0]->ڈیٹا;
    اعشاری * out1 = (اعشاری *) tensors[1]->ڈیٹا;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    ساکن constexpr int64_t BF16_حد = 32768;

    اگر (nelem < BF16_حد) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        اعشاری * tmp0 = (اعشاری *) buf0;
        اعشاری * tmp1 = (اعشاری *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->ڈیٹا, tmp1, tensors[0]->ڈیٹا, nbytes);

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
        واپس سچ;
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
        outbox0[i] = (uint16_t) (sycl::ثنائی_ہندسہ_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::ثنائی_ہندسہ_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    ثابت size_t bf16_بائٹیں = nelem * حجم(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_بائٹیں);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::ثنائی_ہندسہ_cast<اعشاری>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::ثنائی_ہندسہ_cast<اعشاری>(((uint32_t) inbox1[i]) << 16);
        });
    });

    واپس سچ;
}
پکڑو (ثابت sycl::exception &) { واپس جھوٹ; }
پکڑو (...)                     { واپس جھوٹ; }

ساکن خالی *ggml_backend_sycl_reg_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف *name) {
    GGML_غیر_مستعمل(reg);

    اگر (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        واپس (خالی *)ggml_backend_sycl_بانٹو_بفر_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    اگر (strcmp(name, "ggml_backend_comm_init") == 0) {
        واپس (خالی *)ggml_backend_sycl_comm_init;
    }
    اگر (strcmp(name, "ggml_backend_comm_free") == 0) {
        واپس (خالی *)ggml_backend_sycl_comm_free;
    }
    اگر (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        واپس (خالی *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_غیر_مستعمل(name);
    واپس خالی_اشاریہ;
}

ساکن ثابت ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_گنتی,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_پتہ,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    ساکن ggml_backend_reg reg;
    ساکن منطقی initialized = جھوٹ;

    {
        ساکن std::mutex mutex;
        std::lock_guard<std::mutex> تالا(mutex);
        اگر (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = نیا ggml_backend_sycl_reg_context;
            ثابت صحیح min_batch_سائز = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            برائے (صحیح i = 0; i < ggml_sycl_info().device_گنتی; i++) {
                ggml_backend_sycl_device_context * dev_ctx = نیا ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_سیٹ_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_سائز = min_batch_سائز;

                ggml_backend_dev_t dev = نیا ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_نسخہ,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = سچ;
    }

    واپس &reg;
}

ggml_backend_t ggml_backend_sycl_init(صحیح device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_اشاریہ(device);

    ggml_backend_sycl_context * ctx = نیا ggml_backend_sycl_context(device);
    اگر (ctx == خالی_اشاریہ) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        واپس خالی_اشاریہ;
    };

    ggml_backend_t sycl_backend = نیا ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    واپس sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
