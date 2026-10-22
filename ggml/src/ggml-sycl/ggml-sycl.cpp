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

#ضمن <algorithm>
#ضمن <array>
#ضمن <assert.h>
#ضمن <atomic>
#ضمن <cctype>
#ضمن <cinttypes>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <cstdlib>
#ضمن <float.h>
#ضمن <limits>
#ضمن <optional>
#ضمن <stdint.h>
#ضمن <stdio.h>
#ضمن <vector>
#ضمن <cmath>
#ضمن <iostream>
#ضمن <fstream>
#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <regex>

#ضمن <sycl/sycl.hpp>
#ضمن <sycl/backend.hpp>
#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#ضمن <level_zero/zes_api.h>
#جيڪڏهن_پورو
#إذا defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_يادگيري_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#جيڪڏهن_پورو
#إذا SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#جيڪڏهن_پورو
#ضمن <sycl/half_type.hpp>

#ضمن "ggml.h"
#ضمن "ggml-sycl.h"
#ضمن "ggml-impl.h"
#ضمن "ggml-backend-impl.h"

#ضمن "ggml-sycl/add-id.hpp"
#ضمن "ggml-sycl/backend.hpp"
#ضمن "ggml-sycl/common.hpp"
#ضمن "ggml-sycl/element_wise.hpp"
#ضمن "ggml-sycl/fwht.hpp"
#ضمن "ggml-sycl/fused-gemm.hpp"
#ضمن "ggml-sycl/gemm.hpp"
#ضمن "ggml-sycl/getrows.hpp"
#ضمن "ggml-sycl/mem.hpp"
#ضمن "ggml-sycl/norm.hpp"
#ضمن "ggml-sycl/presets.hpp"
#ضمن "ggml-sycl/quantize.hpp"
#ضمن "ggml-sycl/repeat_back.hpp"
#ضمن "ggml-sycl/set_rows.hpp"
#ضمن "ggml-sycl/set.hpp"
#ضمن "ggml-sycl/dsv4-hc.hpp"
#ضمن "ggml-sycl/lightning-indexer.hpp"
#ضمن "ggml-sycl/conv2d.hpp"
#ضمن "ggml-sycl/conv2d-dw.hpp"
#ضمن "ggml-sycl/conv2d-transpose.hpp"
#ضمن "ggml-sycl/ssm_conv.hpp"
#ضمن "ggml-sycl/sycl_hw.hpp"
#ضمن "ggml-sycl/ssm_scan.hpp"
#ضمن "ggml-sycl/fill.hpp"
#ضمن "ggml-sycl/cumsum.hpp"
#ضمن "ggml-sycl/diag.hpp"
#ضمن "ggml-sycl/opt-step.hpp"
#ضمن "ggml-sycl/solve_tri.hpp"
#ضمن "ggml-sycl/gated_delta_net.hpp"
#ضمن "ggml-sycl/pool.hpp"
#ضمن "ggml-sycl/cross_entropy_loss.hpp"

#وصف_ڪريو MEM_سائيز_2_M	0x00200000
#وصف_ڪريو MEM_سائيز_1_G	0x40000000

ساكن منطقي g_sycl_loaded = خطأ;
صحيح g_ggml_sycl_debug = 0;
صحيح g_ggml_sycl_dev_debug = 0;
صحيح g_ggml_sycl_enable_optimize = 1;
صحيح g_ggml_sycl_enable_گراف = 0;
صحيح g_ggml_sycl_enable_dnn = 1;
صحيح g_ggml_sycl_fa_onednn = 1;
صحيح g_ggml_sycl_fa_onednn_max_kv = 0;
صحيح g_ggml_sycl_enable_mkl_fa = 1;
صحيح g_ggml_sycl_memtrace = 0;
صحيح g_ggml_sycl_memtrace_قدم = 64;
صحيح g_ggml_sycl_enable_vmm = 1;
صحيح g_ggml_sycl_enable_fusion = 1;
صحيح g_ggml_sycl_enable_esimd = 1;
صحيح g_ggml_sycl_mmvq_wide = 1;
صحيح g_ggml_sycl_prioritize_dmmv = 0;
صحيح g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
صحيح g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
صحيح g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
صحيح g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
ساكن ثابت حرف * ggml_sycl_dynamic_precision_نالا[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
ساكن صحيح ggml_sycl_get_env_precision(ثابت حرف * name, صحيح def) {
    ثابت حرف * env = getenv(name);
    إذا (!env) {
        أرجع def;
    }
    std::string mode(env);
    لكل (حرف & c : mode) {
        c = (حرف) std::وڏو_اکر((بنانشان حرف) c);
    }
    لكل (صحيح i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        إذا (mode == ggml_sycl_dynamic_precision_نالا[i]) {
            أرجع i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_نالا[def]);
    أرجع def;
}
صحيح g_ggml_sycl_use_async_mem_op = 0;
صحيح g_ggml_sycl_use_async_mem_op_requested = 1;
صحيح g_ggml_sycl_use_level_zero_api = 0;
صحيح g_ggml_sycl_enable_flash_attention = 1;
صحيح g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
صحيح g_ggml_sycl_usm_system = 0;
صحيح g_ggml_sycl_enable_host_pinned_mem = 1;
صحيح g_ggml_sycl_host_pinned_mem_2g = 0;
صحيح g_ggml_sycl_upload_staging_slots = 4;
صحيح g_ggml_sycl_get_mem_api = يادگيري_API_TYPE_LEVEL_ZERO;
صحيح g_ggml_sycl_enable_sparse_fa = 0;
صحيح g_ggml_sycl_debug_sparse_fa = 0;
صحيح g_ggml_sycl_sparse_fa_margin = 256;

#إذا GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
ساكن منطقي ggml_sycl_dnnl_detect_optimized_gemm(صحيح device) {
    استعمال dt = dnnl::يادگيري::معلومات_type;

    حاول {
        ثابت dnnl::يادگيري::dims dims    = { 1, 64, 64 };
        ثابت dnnl::يادگيري::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.سيٽ_scratchpad_گهڻائي(dnnl::scratchpad_گهڻائي::user);

        ثابت خودڪار & q   = dpct::dev_mgr::instance().get_device(device).default_قطار();
        ثابت خودڪار   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ثابت خودڪار   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::يادگيري::desc(dims, dt::f16, strides),
                                                        dnnl::يادگيري::desc(dims, dt::f16, strides),
                                                        dnnl::يادگيري::desc(dims, dt::f32, strides), attr);

        ثابت std::string impl = pd.impl_info_str();
        إذا (impl.find("ref") == std::string::npos) {
            أرجع سچ;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } امسك (ثابت std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    أرجع خطأ;
}
#جيڪڏهن_پورو

ساكن ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    حاول {
        info.device_ڳڻپ = dpct::dev_mgr::instance().device_ڳڻپ();
    } امسك (sycl::exception ثابت &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_ڳڻپ = 0;
    }
    إذا (info.device_ڳڻپ == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        أرجع info;
    }

    GGML_ASSERT(info.device_ڳڻپ <= GGML_SYCL_MAX_DEVICES);

    int64_t ڪل_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    لكل (صحيح i = 0; i < info.device_ڳڻپ; ++i) {
        dpct::device_info prop;
        خودڪار & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#إذا !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#وإلا
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        إذا (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ثابت size_t physical_صفحو = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_صفحو);
        }
#جيڪڏهن_پورو

        info.default_tensor_ورهايو[i] = ڪل_vram;
        ڪل_vram += prop.get_global_mem_سائيز();

        info.devices[i].cc =
            100 * prop.get_major_نسخو() + 10 * prop.get_minor_نسخو();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#إذا GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#جيڪڏهن_پورو
        info.devices[i].smpbo = prop.get_local_mem_سائيز();
        info.devices[i].warp_سائيز = WARP_سائيز;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_گروهه_sizes[i] = prop.get_max_work_گروهه_سائيز();
        info.devices[i].max_wg_في_cu = info.max_work_گروهه_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        إذا (device.is_gpu() && device.default_قطار().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = خطأ;
        }

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        إذا (info.ext_oneapi_level_zero && device.is_gpu() && device.default_قطار().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_قطار().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_نتيجو_t r = zeDeviceGetProperties(ze_dev, &props);
            إذا (r == ZE_نتيجو_SUCCESS) {
                info.devices[i].l0_device_type_valid = سچ;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#جيڪڏهن_پورو
    }

    لكل (صحيح id = 0; id < info.device_ڳڻپ; ++id) {
        info.default_tensor_ورهايو[id] /= ڪل_vram;
    }

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#جيڪڏهن_پورو

    أرجع info;
}

ثابت ggml_sycl_device_info & ggml_sycl_info() {
    ساكن ggml_sycl_device_info info = ggml_sycl_init();
    أرجع info;
}

ساكن فراغ print_device_detail(صحيح id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string نسخو_2;
    نسخو_2 += std::to_string(prop.get_major_نسخو());
    نسخو_2 += ".";
    نسخو_2 += std::to_string(prop.get_minor_نسخو());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    خودڪار global_mem_سائيز = prop.get_global_mem_سائيز()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), نسخو_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_گروهه_سائيز(), prop.get_max_sub_گروهه_سائيز(),
            global_mem_سائيز, device.get_info<sycl::info::device::driver_نسخو>().c_str());
}

ساكن فراغ print_device_opt_feature(صحيح device_ڳڻپ) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    لكل (صحيح id = 0; id < device_ڳڻپ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      صحيح type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
فراغ ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    صحيح device_ڳڻپ = dpct::dev_mgr::instance().device_ڳڻپ();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_ڳڻپ);

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

    لكل (صحيح id = 0; id < device_ڳڻپ; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      صحيح type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_ڳڻپ);
}

ساكن ثابت حرف* dev2dev_int2str(صحيح dev2dev) {
    إذا (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        أرجع "SYCL API";
    } وإلا إذا (dev2dev == DEV2DEV_MEMCPY_L0) {
        أرجع "Level Zero API";
    } وإلا إذا (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        أرجع "Host Forward";
    } وإلا {
        أرجع "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساكن ze_نتيجو_t init_zes() {
    ze_نتيجو_t res = zesInit(0);
    إذا (res != ZE_نتيجو_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (صحيح) res);
    }
    أرجع res;
}

ساكن ze_نتيجو_t get_zes_init_res() {
    ساكن ze_نتيجو_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (صحيح) zes_init_res);
    أرجع zes_init_res;
}
#جيڪڏهن_پورو

ساكن فراغ initialize_sycl_begining() {
#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#جيڪڏهن_پورو
}

ساكن فراغ ggml_check_sycl() حاول {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    ساكن منطقي initialized = خطأ;

    إذا (!initialized) {
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

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#وإلا
        g_ggml_sycl_use_level_zero_api = 0;
#جيڪڏهن_پورو
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", يادگيري_API_TYPE_LEVEL_ZERO);
        إذا (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = يادگيري_API_TYPE_SYCL;
        }

#جيڪڏهن_وصف_ٿيل SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#وإلا
        g_ggml_sycl_enable_flash_attention = 0;
#جيڪڏهن_پورو

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
#إذا defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#جيڪڏهن_پورو

#إذا defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#جيڪڏهن_پورو

#إذا defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#جيڪڏهن_پورو

#إذا defined(GGML_SYCL_گراف)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#جيڪڏهن_پورو

#إذا defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#جيڪڏهن_پورو
#إذا defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#جيڪڏهن_پورو

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#جيڪڏهن_پورو

#إذا defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#جيڪڏهن_پورو
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_قدم);
#جيڪڏهن_وصف_ٿيل SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_SYCL_گراف
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_گراف);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#جيڪڏهن_پورو

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_نالا[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_نالا[g_ggml_sycl_dynamic_required_precision]);

#إذا defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#جيڪڏهن_پورو

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#إذا defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#جيڪڏهن_پورو
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#وإلا
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#جيڪڏهن_پورو

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
#إذا defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_يادگيري_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_گراف;
        إذا (g_ggml_sycl_use_async_mem_op) {
            لكل (بنانشان صحيح i = 0; i < dpct::dev_mgr::instance().device_ڳڻپ(); ++i) {
                إذا (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_يادگيري_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    اكسر;
                }
            }
        }
#جيڪڏهن_پورو
        إذا (CHECK_TRY_ERROR(g_all_sycl_device_ڳڻپ =
                            dpct::dev_mgr::instance().device_ڳڻپ()) != 0) {
            initialized = سچ;
            g_sycl_loaded = خطأ;
            أرجع;
        }
        GGML_ASSERT(g_all_sycl_device_ڳڻپ <= GGML_SYCL_MAX_DEVICES);

        initialized = سچ;
        g_sycl_loaded = سچ;
        ggml_backend_sycl_print_sycl_devices();
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
مضمن فراغ check_allow_gpu_اشاريو(ثابت صحيح device_اشاريو) {
  إذا (device_اشاريو >= ggml_sycl_info().device_ڳڻپ) {
    حرف error_buf[256];
    snprintf(
        error_buf,
        حجم(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_اشاريو,
        ggml_sycl_info().device_ڳڻپ - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    أكد(خطأ);
  }
}

GGML_API فراغ ggml_backend_sycl_get_gpu_list(صحيح *id_list, صحيح max_len) حاول {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    لكل(صحيح i=0;i<max_len;i++) id_list[i] = -1;

    لكل (صحيح i=0;i< ggml_sycl_info().device_ڳڻپ;i++){
        إذا (i>=max_len) اكسر;
        id_list[i] = i;
    }
    أرجع;
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

مضمن فراغ free_aligned_mem_host(فراغ * memblock) {
#جيڪڏهن_وصف_ٿيل _WIN32
    _aligned_free(memblock);
#وإلا
    حرر(memblock);
#جيڪڏهن_پورو
}

// sycl buffer

بنية ggml_backend_sycl_بفر_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    ساكن constexpr size_t staging_جڳهه_سائيز = 8*1024*1024;

    بنية host_staging {
        فراغ * معلومات = خالي_اشارو;
        std::vector<sycl::event> events;
        std::vector<منطقي> submitted;
        صحيح ايندڙ = 0;
    };

    صحيح device;
    فراغ * dev_ptr = خالي_اشارو;
    قطار_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    منطقي is_usm_system;

    ggml_backend_sycl_بفر_context(صحيح device, فراغ * dev_ptr, قطار_ptr stream, منطقي is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_اشاريو(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    فراغ drop_host_staging() {
        لكل (size_t i = 0; i < staging.submitted.size(); ++i) {
            إذا (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = خطأ;
            }
        }
        إذا (staging.معلومات != خالي_اشارو) {
            sycl::حرر(staging.معلومات, *stream);
            staging.معلومات = خالي_اشارو;
        }
    }

    ~ggml_backend_sycl_بفر_context() {
        drop_host_staging();
        إذا (dev_ptr != خالي_اشارو) {
            ggml_sycl_سيٽ_device(device);
            إذا (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            وإلا
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        لكل (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

ساكن ثابت حرف * ggml_backend_sycl_بفر_type_get_name(ggml_backend_بفر_type_t buft);

ساكن منطقي ggml_backend_بفر_is_sycl(ggml_backend_بفر_t بفر) {
    أرجع بفر->buft->iface.get_name == ggml_backend_sycl_بفر_type_get_name;
}

ساكن فراغ
ggml_backend_sycl_بفر_free_بفر(ggml_backend_بفر_t بفر) حاول {
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    ggml_sycl_سيٽ_device(ctx->device);

    ميٽيو ctx;
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ * ggml_backend_sycl_بفر_get_بنياد(ggml_backend_بفر_t بفر) {
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    أرجع ctx->dev_ptr;
}

ساكن تعداد ggml_حالت
ggml_backend_sycl_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                     ggml_tensor *tensor) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *)بفر->context;

    إذا (tensor->view_src != عدم) {
        أكد(tensor->view_src->بفر->buft == بفر->buft);
        أرجع GGML_حالت_SUCCESS;
    }

    إذا (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        بدل (tensor->type) {
            حالة GGML_TYPE_Q4_0:
            حالة GGML_TYPE_Q8_0:
            حالة GGML_TYPE_Q2_K:
            حالة GGML_TYPE_Q3_K:
            حالة GGML_TYPE_Q4_K:
            حالة GGML_TYPE_Q5_K:
            حالة GGML_TYPE_Q6_K:
            حالة GGML_TYPE_IQ3_XXS:
            حالة GGML_TYPE_IQ3_S:
            حالة GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = جديد ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                اكسر;
            }
            افتراضي:
                اكسر;
        }
    }

    إذا (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_سائيز = ggml_nbytes(tensor);
        size_t padded_سائيز = ggml_backend_buft_get_alloc_سائيز(بفر->buft, tensor);

        إذا (padded_سائيز > original_سائيز && tensor->view_src == خالي_اشارو) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (حرف *)tensor->معلومات + original_سائيز, 0,
                padded_سائيز - original_سائيز).wait()));
        }
    }
    أرجع GGML_حالت_SUCCESS;
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر,
                                                ggml_tensor *tensor,
                                                ثابت فراغ *معلومات, size_t وٿي,
                                                size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;
    ggml_sycl_سيٽ_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ثابت صحيح n_slots = g_ggml_sycl_upload_staging_slots;
    إذا (n_slots > 0 && ctx->staging.معلومات == خالي_اشارو) {
        ctx->staging.معلومات = sycl::malloc_host(n_slots * ctx->staging_جڳهه_سائيز, *ctx->stream);
        إذا (ctx->staging.معلومات != خالي_اشارو) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.سونپيو(n_slots, خطأ);
        }
    }
    إذا (ctx->staging.معلومات != خالي_اشارو) {
        قطار_ptr    stream    = ctx->stream;
        حرف *       dst       = (حرف *) tensor->معلومات + وٿي;
        ثابت حرف * src       = (ثابت حرف *) معلومات;
        size_t       بچيل = size;
        طالما (بچيل > 0) {
            ثابت size_t chunk = std::min(بچيل, ctx->staging_جڳهه_سائيز);
            ثابت صحيح    جڳهه  = ctx->staging.ايندڙ;
            ctx->staging.ايندڙ = (ctx->staging.ايندڙ + 1) % (صحيح) ctx->staging.submitted.size();
            إذا (ctx->staging.submitted[جڳهه]) {
                ctx->staging.events[جڳهه].wait_and_throw();
            }
            فراغ * stage = (حرف *) ctx->staging.معلومات + جڳهه * ctx->staging_جڳهه_سائيز;
            يادگيري_نقل(stage, src, chunk);
            ctx->staging.events[جڳهه] = stream->يادگيري_نقل(dst, stage, chunk);
            ctx->staging.submitted[جڳهه] = سچ;
            src       += chunk;
            dst       += chunk;
            بچيل -= chunk;
        }
        أرجع;
    }

    خودڪار stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_قطار());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#جيڪڏهن_وصف_نه_ٿيل _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    حرف * host_buf = (حرف *) يادگيري_ڏيو(size);
    يادگيري_نقل(host_buf, معلومات, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).يادگيري_نقل((حرف *) tensor->معلومات + وٿي, host_buf, size).wait()));
    حرر(host_buf);
#وإلا
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).يادگيري_نقل((حرف *) tensor->معلومات + وٿي, معلومات, size).wait()));
#جيڪڏهن_پورو
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                                ثابت ggml_tensor *tensor,
                                                فراغ *معلومات, size_t وٿي,
                                                size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    ggml_backend_sycl_بفر_context * ctx = ( ggml_backend_sycl_بفر_context *)بفر->context;

    ggml_sycl_سيٽ_device(ctx->device);
    خودڪار stream = dpct::dev_mgr::instance().get_device(ctx->device).default_قطار();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.يادگيري_نقل(معلومات, (ثابت حرف *)tensor->معلومات + وٿي, size)
            .wait()));
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ساكن منطقي ggml_sycl_is_l0_discrete_gpu(صحيح device) {
    أرجع ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#جيڪڏهن_پورو

ساكن فراغ memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, فراغ *ptr_dst,
                         ثابت فراغ *ptr_src, size_t size) {
    حرف *host_buf = (حرف *)يادگيري_ڏيو(size);
    إذا (host_buf == خالي_اشارو) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.يادگيري_نقل(host_buf, (ثابت حرف *)ptr_src, size).wait();
    q_dst.يادگيري_نقل((حرف *)ptr_dst, host_buf, size).wait();
    حرر(host_buf);
}

ساكن فراغ dev2dev_memcpy(صحيح device_dst, sycl::queue &q_dst, صحيح device_src, sycl::queue &q_src, فراغ *ptr_dst,
                    ثابت فراغ *ptr_src, size_t size) {

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    إذا (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ثابت منطقي l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        إذا (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            خودڪار ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            خودڪار ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_قطار_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_قطار_DESC, خالي_اشارو, 0, 0,
                                            0, ZE_COMMAND_قطار_گهڻائي_SYNCHRONOUS, ZE_COMMAND_قطار_ترجيح_عام};
            ze_command_list_handle_t cl;
            ze_نتيجو_t r = ze_Command_List_Create_فوري(ze_ctx, ze_dev, &cq_desc, &cl);
            إذا (r == ZE_نتيجو_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_يادگيري_Copy(cl, ptr_dst, ptr_src, size, خالي_اشارو, 0, خالي_اشارو);
                zeCommandListDestroy(cl);
                إذا (r == ZE_نتيجو_SUCCESS) {
                    أرجع;
                }
            }
        }
    }
#جيڪڏهن_پورو

    إذا (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        إذا (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.يادگيري_نقل(ptr_dst, ptr_src, size).wait()));
            أرجع;
        }
    }

    // Host-staged copy
    إذا(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } وإلا {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

ساكن منطقي
ggml_backend_sycl_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                    ثابت ggml_tensor *src,
                                    ggml_tensor *dst) حاول {
    منطقي is_cpy_supported = ggml_backend_بفر_is_sycl(src->بفر);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    إذا (is_cpy_supported) {
        ggml_backend_sycl_بفر_context * src_ctx = (ggml_backend_sycl_بفر_context *)src->بفر->context;
        ggml_backend_sycl_بفر_context * dst_ctx = (ggml_backend_sycl_بفر_context *)dst->بفر->context;

        ggml_sycl_سيٽ_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_سيٽ_device(dst_ctx->device);
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

        قطار_ptr وهڪرو_dst = dst_ctx->stream;
        قطار_ptr وهڪرو_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *وهڪرو_dst, src_ctx->device, *وهڪرو_src, dst->معلومات, src->معلومات, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#إذا 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).يادگيري_نقل(
            (حرف *)dst->معلومات, (ثابت حرف *)src->معلومات, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#جيڪڏهن_پورو
        أرجع سچ;
    }
    أرجع خطأ;
    GGML_اڻ_استعمال_ٿيل(بفر);
} امسك (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_بفر_clear(ggml_backend_بفر_t بفر,
                                           uint8_t value) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, بفر->size);
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;

    ggml_sycl_سيٽ_device(ctx->device);
    قطار_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    لكل (size_t off = 0; off < بفر->size; off += MAX_CHUNK) {
        size_t chunk = std::min(بفر->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<حرف*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_بفر_memset_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor, uint8_t value,
                                                   size_t وٿي, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, وٿي, value);
    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx->device));
    خودڪار stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_قطار());
    إذا (size == 0) {
        أرجع;  // Nothing to do
    }
    إذا (tensor->معلومات == خالي_اشارو) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    فراغ * هدف_ptr = static_cast<حرف *>(tensor->معلومات) + وٿي;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(هدف_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

ساكن فراغ ggml_backend_sycl_بفر_reset(ggml_backend_بفر_t بفر) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    إذا (بفر == خالي_اشارو) {
        أرجع;
    }

    ggml_backend_sycl_بفر_context * ctx = (ggml_backend_sycl_بفر_context *) بفر->context;

    إذا (ctx != خالي_اشارو) {
        لكل (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

ساكن ثابت ggml_backend_بفر_i ggml_backend_sycl_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_sycl_بفر_get_بنياد,
    /* .init_tensor     = */ ggml_backend_sycl_بفر_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_بفر_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_بفر_سيٽ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_sycl_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_بفر_clear,
    /* .reset           = */ ggml_backend_sycl_بفر_reset,
};

// sycl buffer type
بنية ggml_backend_sycl_بفر_type_context {
    صحيح device;
    std::string name;

    // each buffer type has its own stream
    قطار_ptr stream = خالي_اشارو;
};

ساكن ثابت حرف * ggml_backend_sycl_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    ggml_backend_sycl_بفر_type_context * ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;

    أرجع ctx->name.c_str();
}

ساكن منطقي check_usm_system(صحيح device, size_t size) {
    منطقي use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_سائيز_1_G);

    إذا (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = خطأ;
    }

    أرجع use_usm_system;
}

مضمن فراغ * aligned_malloc_host(size_t alignment, size_t size) {
#جيڪڏهن_وصف_ٿيل _WIN32
    أرجع _aligned_malloc(size, alignment);
#وإلا
    أرجع aligned_alloc(alignment, size);
#جيڪڏهن_پورو
}

ساكن ggml_backend_بفر_t
ggml_backend_sycl_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft,
                                           size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_بفر_type_context * buft_ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    ggml_sycl_سيٽ_device(buft_ctx->device);
    ثابت قطار_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_سائيز_2_M;
    size_t aligned_سائيز = ((size + alignment - 1) / alignment) * alignment;
    منطقي use_usm_system = check_usm_system(buft_ctx->device, aligned_سائيز);

    فراغ * dev_ptr;
    إذا (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (فراغ *)aligned_malloc_host(alignment, aligned_سائيز);
        إذا (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            أرجع خالي_اشارو;
        }
    } وإلا {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (فراغ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_بفر)));
        إذا (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          أرجع خالي_اشارو;
        }
    }
    ggml_backend_sycl_بفر_context * ctx = جديد  ggml_backend_sycl_بفر_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    أرجع ggml_backend_بفر_init(buft, ggml_backend_sycl_بفر_interface, ctx, size);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن size_t ggml_backend_sycl_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع SYCL_بفر_ALIGNMENT;
    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن منطقي is_bmg_g31_arch(صحيح device) {
    أرجع ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

ساكن size_t ggml_backend_sycl_بفر_type_get_max_سائيز(ggml_backend_بفر_type_t buft) {
    ggml_backend_sycl_بفر_type_context * ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    صحيح device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_سائيز = dpct::get_device(device).get_max_mem_alloc_سائيز();
    إذا (g_ggml_sycl_host_pinned_mem_2g) {
        أرجع std::min(max_alloc_سائيز, (size_t) 2LL*1024*1024*1024);
    } وإلا {
        إذا(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_سائيز = max_alloc_سائيز*0.6;
        }
        أرجع max_alloc_سائيز;

    }
    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن size_t ggml_backend_sycl_بفر_type_get_alloc_سائيز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_سائيز(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    إذا (ggml_is_quantized(tensor->type)) {
        إذا (ne0 % ميٽرڪس_صف_PADDING != 0) {
            size += ggml_صف_سائيز(tensor->type, ميٽرڪس_صف_PADDING - ne0 % ميٽرڪس_صف_PADDING);
        }
    }

    أرجع size;

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن ثابت ggml_backend_بفر_type_i ggml_backend_sycl_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_sycl_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_بفر_type_get_max_سائيز,
    /* .get_alloc_size      = */ ggml_backend_sycl_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ عدم,
};

ggml_backend_بفر_type_t ggml_backend_sycl_بفر_type(صحيح device) {
    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> تالو(mutex);


    خودڪار dev_ڳڻپ = ggml_backend_sycl_get_device_ڳڻپ();

    إذا (device>=dev_ڳڻپ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_ڳڻپ-1);
        GGML_ASSERT(device<dev_ڳڻپ);
    }
    ساكن بنية ggml_backend_بفر_type ggml_backend_sycl_بفر_types[GGML_SYCL_MAX_DEVICES];

    ساكن منطقي ggml_backend_sycl_بفر_type_initialized = خطأ;

    إذا (!ggml_backend_sycl_بفر_type_initialized) {
        لكل (صحيح i = 0; i < dev_ڳڻپ; i++) {
            خودڪار & device_i = dpct::dev_mgr::instance().get_device(i);
            قطار_ptr stream = &(device_i.default_قطار());
            ggml_backend_sycl_بفر_types[i] = {
                /* .iface    = */ ggml_backend_sycl_بفر_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ جديد ggml_backend_sycl_بفر_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_بفر_type_initialized = سچ;
    }
    أرجع &ggml_backend_sycl_بفر_types[device];
}

ساكن ggml_backend_بفر_type_t ggml_backend_sycl_بفر_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    صحيح device = ctx->device;
    إذا (device>=ggml_sycl_info().device_ڳڻپ or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_ڳڻپ-1);
        GGML_ASSERT(device<ggml_sycl_info().device_ڳڻپ);
    }
    ساكن بنية ggml_backend_بفر_type ggml_backend_sycl_بفر_types[GGML_SYCL_MAX_DEVICES];

    ساكن منطقي ggml_backend_sycl_بفر_type_initialized = خطأ;

    إذا (!ggml_backend_sycl_بفر_type_initialized) {
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; i++) {
            ggml_backend_sycl_بفر_types[i] = {
                /* .iface    = */ ggml_backend_sycl_بفر_type_interface,
                /* .device   = */ خالي_اشارو,
                /* .context  = */ جديد ggml_backend_sycl_بفر_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_بفر_type_initialized = سچ;
    }
    أرجع &ggml_backend_sycl_بفر_types[device];
}

// sycl split buffer

ساكن int64_t get_صف_rounding(ggml_type type, ثابت std::array<عائم, GGML_SYCL_MAX_DEVICES> & tensor_ورهايو) {
    int64_t min_compute_capability = وڌ_کان_وڌ_عدد;
    int64_t max_compute_capability = INT_MIN;
    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        إذا (tensor_ورهايو[i] < (i + 1 < ggml_sycl_info().device_ڳڻپ ? tensor_ورهايو[i + 1] : 1.0f)) {
            إذا (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            إذا (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    بدل(type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
            أرجع max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
            أرجع 64;
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_F32:
            أرجع 1;
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_IQ2_XXS:
        حالة GGML_TYPE_IQ2_XS:
        حالة GGML_TYPE_IQ2_S:
        حالة GGML_TYPE_IQ1_S:
        حالة GGML_TYPE_IQ1_M:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ4_XS:
        حالة GGML_TYPE_IQ4_NL:
            أرجع max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالة GGML_TYPE_IQ3_S:
            أرجع max_compute_capability >= VER_GEN9 ? 128 : 64;
        حالة GGML_TYPE_Q6_K:
            أرجع 64;
        افتراضي:
            GGML_ABORT("fatal error");
    }
}

ساكن فراغ get_صف_ورهايو(int64_t * صف_هيٺيون, int64_t * صف_مٿاهون, ثابت ggml_tensor * tensor, ثابت std::array<عائم, GGML_SYCL_MAX_DEVICES> & tensor_ورهايو, صحيح id) {
    ثابت int64_t nrows = ggml_nrows(tensor);
    ثابت int64_t rounding = get_صف_rounding(tensor->type, tensor_ورهايو);

    *صف_هيٺيون = id == 0 ? 0 : nrows*tensor_ورهايو[id];
    *صف_هيٺيون -= *صف_هيٺيون % rounding;
    إذا (id == ggml_sycl_info().device_ڳڻپ - 1) {
        *صف_مٿاهون = nrows;
    } وإلا {
        *صف_مٿاهون = nrows*tensor_ورهايو[id + 1];
        *صف_مٿاهون -= *صف_مٿاهون % rounding;
    }
}

ساكن size_t ggml_nbytes_ورهايو(ثابت بنية ggml_tensor * tensor, صحيح nrows_ورهايو) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    أرجع nrows_ورهايو*ggml_صف_سائيز(tensor->type, tensor->ne[0]);
}

بنية ggml_backend_sycl_ورهايو_بفر_type_context {
    std::array<عائم, GGML_SYCL_MAX_DEVICES> tensor_ورهايو;
};

بنية ggml_backend_sycl_ورهايو_بفر_context {
    ~ggml_backend_sycl_ورهايو_بفر_context() حاول {
        لكل (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    امسك (sycl::exception ثابت &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
      std::ٻاهر_نڪرو(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<قطار_ptr> streams;
};

ساكن فراغ ggml_backend_sycl_ورهايو_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_sycl_ورهايو_بفر_context * ctx = (ggml_backend_sycl_ورهايو_بفر_context *)بفر->context;
    ميٽيو ctx;
}

ساكن فراغ * ggml_backend_sycl_ورهايو_بفر_get_بنياد(ggml_backend_بفر_t بفر) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    أرجع (فراغ *)0x1000;

    GGML_اڻ_استعمال_ٿيل(بفر);
}

ساكن تعداد ggml_حالت
ggml_backend_sycl_ورهايو_بفر_init_tensor(ggml_backend_بفر_t بفر,
                                           ggml_tensor *tensor) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == خالي_اشارو); // views of split tensors are not supported

    ggml_backend_sycl_ورهايو_بفر_context * ctx = (ggml_backend_sycl_ورهايو_بفر_context *)بفر->context;
    ggml_backend_sycl_ورهايو_بفر_type_context * buft_ctx = (ggml_backend_sycl_ورهايو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = جديد ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    إذا (ctx->streams.خالي_جايون()) {
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_قطار()));
        }
    }

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        int64_t صف_هيٺيون, صف_مٿاهون;
        get_صف_ورهايو(&صف_هيٺيون, &صف_مٿاهون, tensor, buft_ctx->tensor_ورهايو, i);

        int64_t nrows_ورهايو = صف_مٿاهون - صف_هيٺيون;
        إذا (nrows_ورهايو == 0) {
            استمر;
        }

        size_t size = ggml_nbytes_ورهايو(tensor, nrows_ورهايو);
        ثابت size_t original_سائيز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        إذا (ne0 % ميٽرڪس_صف_PADDING != 0) {
            size += ggml_صف_سائيز(tensor->type, ميٽرڪس_صف_PADDING - ne0 % ميٽرڪس_صف_PADDING);
        }

        ggml_sycl_سيٽ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        حرف * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (حرف *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_بفر)));
        إذا (!buf) {
            حرف err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            ارم std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        إذا (size > original_سائيز) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_سائيز, 0, size - original_سائيز)
                    .wait()));
        }

        extra->معلومات_device[i] = buf;

        لكل (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = جديد sycl::event()));
        }
    }
    tensor->extra = extra;
    أرجع GGML_حالت_SUCCESS;
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ
ggml_backend_sycl_ورهايو_بفر_سيٽ_tensor(ggml_backend_بفر_t بفر,
                                          ggml_tensor *tensor, ثابت فراغ *معلومات,
                                          size_t وٿي, size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(وٿي == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ورهايو_بفر_context * ctx = (ggml_backend_sycl_ورهايو_بفر_context *)بفر->context;
    ggml_backend_sycl_ورهايو_بفر_type_context * buft_ctx = (ggml_backend_sycl_ورهايو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];
    ثابت size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        int64_t صف_هيٺيون, صف_مٿاهون;
        get_صف_ورهايو(&صف_هيٺيون, &صف_مٿاهون, tensor, buft_ctx->tensor_ورهايو, i);

        int64_t nrows_ورهايو = صف_مٿاهون - صف_هيٺيون;
        إذا (nrows_ورهايو == 0) {
            استمر;
        }

        ثابت size_t وٿي_ورهايو = صف_هيٺيون*nb1;
        size_t size = ggml_nbytes_ورهايو(tensor, nrows_ورهايو);
        ثابت size_t original_سائيز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        إذا (ne0 % ميٽرڪس_صف_PADDING != 0) {
            size += ggml_صف_سائيز(tensor->type, ميٽرڪس_صف_PADDING - ne0 % ميٽرڪس_صف_PADDING);
        }

        ثابت حرف * buf_host = (ثابت حرف *)معلومات + وٿي_ورهايو;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_سيٽ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .يادگيري_نقل(extra->معلومات_device[i], buf_host, original_سائيز)
                .wait()));
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ
ggml_backend_sycl_ورهايو_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                          ثابت ggml_tensor *tensor, فراغ *معلومات,
                                          size_t وٿي, size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(وٿي == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_ورهايو_بفر_context * ctx = (ggml_backend_sycl_ورهايو_بفر_context *)بفر->context;
    ggml_backend_sycl_ورهايو_بفر_type_context * buft_ctx = (ggml_backend_sycl_ورهايو_بفر_type_context *)بفر->buft->context;

    ثابت int64_t ne0 = tensor->ne[0];
    ثابت size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        int64_t صف_هيٺيون, صف_مٿاهون;
        get_صف_ورهايو(&صف_هيٺيون, &صف_مٿاهون, tensor, buft_ctx->tensor_ورهايو, i);

        int64_t nrows_ورهايو = صف_مٿاهون - صف_هيٺيون;
        إذا (nrows_ورهايو == 0) {
            استمر;
        }

        ثابت size_t وٿي_ورهايو = صف_هيٺيون*nb1;
        size_t size = ggml_nbytes_ورهايو(tensor, nrows_ورهايو);
        ثابت size_t original_سائيز = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        إذا (ne0 % ميٽرڪس_صف_PADDING != 0) {
            size += ggml_صف_سائيز(tensor->type, ميٽرڪس_صف_PADDING - ne0 % ميٽرڪس_صف_PADDING);
        }

        حرف * buf_host = (حرف *)معلومات + وٿي_ورهايو;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_سيٽ_device(i);
        ثابت قطار_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .يادگيري_نقل(buf_host, extra->معلومات_device[i], original_سائيز)
                .wait()));
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_ورهايو_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    GGML_اڻ_استعمال_ٿيل(بفر);
    GGML_اڻ_استعمال_ٿيل(value);
}

ساكن بنية ggml_backend_بفر_i ggml_backend_sycl_ورهايو_بفر_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_ورهايو_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_sycl_ورهايو_بفر_get_بنياد,
    /* .init_tensor     = */ ggml_backend_sycl_ورهايو_بفر_init_tensor,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_sycl_ورهايو_بفر_سيٽ_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_ورهايو_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ عدم,
    /* .clear           = */ ggml_backend_sycl_ورهايو_بفر_clear,
    /* .reset           = */ عدم,
};

// sycl split buffer type

ساكن ثابت حرف * ggml_backend_sycl_ورهايو_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    أرجع GGML_SYCL_NAME "_Split";

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن منطقي ggml_backend_بفر_is_sycl_ورهايو(ggml_backend_بفر_t بفر) {
   أرجع بفر->buft->iface.get_name == ggml_backend_sycl_ورهايو_بفر_type_get_name;
}

ساكن ggml_backend_بفر_t ggml_backend_sycl_ورهايو_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_ورهايو_بفر_context * ctx = جديد ggml_backend_sycl_ورهايو_بفر_context();

    أرجع ggml_backend_بفر_init(buft, ggml_backend_sycl_ورهايو_بفر_interface, ctx, size);
}

ساكن size_t ggml_backend_sycl_ورهايو_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    أرجع SYCL_بفر_ALIGNMENT;
    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن size_t ggml_backend_sycl_ورهايو_بفر_type_get_alloc_سائيز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    ggml_backend_sycl_ورهايو_بفر_type_context * ctx = (ggml_backend_sycl_ورهايو_بفر_type_context *)buft->context;

    size_t ڪل_سائيز = 0;

    ثابت int64_t ne0 = tensor->ne[0];

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        int64_t صف_هيٺيون, صف_مٿاهون;
        get_صف_ورهايو(&صف_هيٺيون, &صف_مٿاهون, tensor, ctx->tensor_ورهايو, i);

        int64_t nrows_ورهايو = صف_مٿاهون - صف_هيٺيون;
        إذا (nrows_ورهايو == 0) {
            استمر;
        }

        ڪل_سائيز += ggml_nbytes_ورهايو(tensor, nrows_ورهايو);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        إذا (ne0 % ميٽرڪس_صف_PADDING != 0) {
            ڪل_سائيز += ggml_صف_سائيز(tensor->type, ميٽرڪس_صف_PADDING - ne0 % ميٽرڪس_صف_PADDING);
        }
    }

    أرجع ڪل_سائيز;
}

ساكن منطقي ggml_backend_sycl_ورهايو_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    أرجع خطأ;

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن ggml_backend_بفر_type_i ggml_backend_sycl_ورهايو_بفر_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_ورهايو_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_ورهايو_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_sycl_ورهايو_بفر_type_get_alignment,
    /* .get_max_size        = */ عدم, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_ورهايو_بفر_type_get_alloc_سائيز,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_sycl_ورهايو_بفر_type_is_host,
};

ggml_backend_بفر_type_t ggml_backend_sycl_ورهايو_بفر_type(صحيح main_device, ثابت عائم * tensor_ورهايو) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_اڻ_استعمال_ٿيل(main_device);

    ساكن std::mutex mutex;
    std::lock_guard<std::mutex> تالو(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    ساكن std::map<std::array<عائم, GGML_SYCL_MAX_DEVICES>, بنية ggml_backend_بفر_type> buft_map;

    std::array<عائم, GGML_SYCL_MAX_DEVICES> tensor_ورهايو_arr = {};

    منطقي all_zero = tensor_ورهايو == خالي_اشارو || std::all_of(tensor_ورهايو, tensor_ورهايو + GGML_SYCL_MAX_DEVICES, [](عائم x) { أرجع x == 0.0f; });
    إذا (all_zero) {
        tensor_ورهايو_arr = ggml_sycl_info().default_tensor_ورهايو;
    } وإلا {
        عائم ورهايو_جوڙ = 0.0f;
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
            tensor_ورهايو_arr[i] = ورهايو_جوڙ;
            ورهايو_جوڙ += tensor_ورهايو[i];
        }
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
            tensor_ورهايو_arr[i] /= ورهايو_جوڙ;
        }
    }

    خودڪار it = buft_map.find(tensor_ورهايو_arr);
    إذا (it != buft_map.end()) {
        أرجع &it->second;
    }

    بنية ggml_backend_بفر_type buft {
        /* .iface   = */ ggml_backend_sycl_ورهايو_بفر_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ جديد ggml_backend_sycl_ورهايو_بفر_type_context{tensor_ورهايو_arr},
    };

    خودڪار نتيجو = buft_map.emplace(tensor_ورهايو_arr, buft);
    أرجع &نتيجو.پهريون->second;
}

// host buffer type

بنية ggml_backend_sycl_device_context {
    صحيح device;
    std::string name;
    std::string description;
    صحيح op_offload_min_batch_سائيز;
};

ساكن ثابت حرف * ggml_backend_sycl_host_بفر_type_name(ggml_backend_بفر_type_t buft) {
    أرجع GGML_SYCL_NAME "_Host";

    GGML_اڻ_استعمال_ٿيل(buft);
}

ساكن صحيح ggml_backend_sycl_host_بفر_type_device(ggml_backend_بفر_type_t buft) {
    أرجع static_cast<ثابت ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
ساكن فراغ * ggml_backend_sycl_host_malloc(صحيح device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    فراغ * ptr = خالي_اشارو;
    حاول {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        خودڪار & q = dpct::dev_mgr::instance().get_device(device).default_قطار();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } امسك (...) {
        ptr = خالي_اشارو;
    }
    إذا (ptr == خالي_اشارو) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    أرجع ptr;
}

ساكن فراغ ggml_backend_sycl_host_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    إذا (بفر->context == خالي_اشارو) {
        أرجع;
    }
    إذا (g_ggml_sycl_enable_host_pinned_mem) {
        ثابت صحيح device = ggml_backend_sycl_host_بفر_type_device(بفر->buft);
        خودڪار & q = dpct::dev_mgr::instance().get_device(device).default_قطار();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(بفر->context, q)));
    } وإلا {
        free_aligned_mem_host((فراغ *) بفر->context);
    }
}

ساكن ggml_backend_بفر_t ggml_backend_sycl_host_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    فراغ * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_بفر_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    إذا (ptr == خالي_اشارو) {
        // fallback to cpu buffer
        أرجع ggml_backend_buft_alloc_بفر(ggml_backend_cpu_بفر_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_بفر_t بفر = ggml_backend_cpu_بفر_from_ptr(ptr, size);
    بفر->buft = buft;
    بفر->iface.free_بفر = ggml_backend_sycl_host_بفر_free_بفر;

    أرجع بفر;
}

ساكن size_t ggml_backend_sycl_host_بفر_type_get_max_سائيز(ggml_backend_بفر_type_t buft) {

    إذا (g_ggml_sycl_enable_host_pinned_mem) {
        ثابت صحيح device = ggml_backend_sycl_host_بفر_type_device(buft);
        size_t max_alloc_سائيز = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_سائيز();
        إذا (g_ggml_sycl_host_pinned_mem_2g) {
            أرجع std::min(max_alloc_سائيز, (size_t) 2LL*1024*1024*1024);
        } وإلا {
            أرجع max_alloc_سائيز;
        }
    } وإلا {
        أرجع SIZE_MAX;
    }
}

ساكن ggml_backend_بفر_type_t ggml_backend_sycl_host_بفر_type_for_device(صحيح device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    ساكن std::vector<ggml_backend_بفر_type> بفر_types_host = [] {
        std::vector<ggml_backend_بفر_type> bufts(ggml_backend_sycl_get_device_ڳڻپ());
        لكل (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_بفر_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_بفر_type_alloc_بفر,
                    /* .alloc_buffer_n      = */ عدم,
                    /* .get_alignment       = */ ggml_backend_cpu_بفر_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_بفر_type_get_max_سائيز,
                    /* .get_alloc_size      = */ ggml_backend_cpu_بفر_type()->iface.get_alloc_سائيز,
                    /* .get_alloc_size_n    = */ عدم,
                    /* .is_host             = */ ggml_backend_cpu_بفر_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ خالي_اشارو,
            };
        }
        أرجع bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (صحيح) بفر_types_host.size());

    أرجع &بفر_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_بفر_type_t ggml_backend_sycl_host_بفر_type() {
    أرجع ggml_backend_sycl_host_بفر_type_for_device(0);
}

// buffer pool for sycl (legacy)
بنية ggml_sycl_pool_leg : عام ggml_sycl_pool {
    ساكن ثابت صحيح MAX_SYCL_BUFFERS = 256;

    صحيح device;
    قطار_ptr qptr;
    بنية ggml_sycl_بفر {
        فراغ * ptr = خالي_اشارو;
        size_t size = 0;
    };

    ggml_sycl_بفر بفر_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_سائيز = 0;

    واضح ggml_sycl_pool_leg(قطار_ptr qptr_, صحيح device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_POOL
        صحيح    n_cached    = 0;
        size_t بائيٽون_cached = 0;
        لكل (صحيح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            إذا (بفر_pool[i].ptr != خالي_اشارو) {
                ++n_cached;
                بائيٽون_cached += بفر_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, بائيٽون_cached / 1024.0 / 1024.0);
        ثابت خودڪار slots = format_slots_اندريون_alloc_ترتيب();
        إذا (!slots.خالي_جايون()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#جيڪڏهن_پورو

        لكل (صحيح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            إذا (b.ptr != خالي_اشارو) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_سائيز -= b.size;
            }
        }
        GGML_ASSERT(pool_سائيز == 0);
    }

#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_POOL
    std::string format_slots_اندريون_alloc_ترتيب() ثابت {
        std::string سٽ;
        حرف buf[32];
        منطقي پهريون = سچ;
        لكل (صحيح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            إذا (بفر_pool[i].ptr == خالي_اشارو) {
                استمر;
            }
            إذا (!پهريون) {
                سٽ += '/';
            }
            پهريون = خطأ;
            snprintf(buf, حجم(buf), "%.2f", بفر_pool[i].size / 1024.0 / 1024.0);
            سٽ += buf;
        }
        أرجع سٽ;
    }
#جيڪڏهن_پورو

    فراغ * alloc(size_t size, size_t * actual_سائيز) override {
#جيڪڏهن_وصف_ٿيل DEBUG_sycl_MALLOC
        صحيح nnz = 0;
        size_t max_size = 0;
#جيڪڏهن_پورو
        size_t بهترين_diff = 1ull << 36;
        صحيح ibest = -1;
        لكل (صحيح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر& b = بفر_pool[i];
            إذا (b.ptr != خالي_اشارو) {
#جيڪڏهن_وصف_ٿيل DEBUG_sycl_MALLOC
                ++nnz;
                إذا (b.size > max_size) max_size = b.size;
#جيڪڏهن_پورو
                إذا (b.size >= size) {
                    size_t diff = b.size - size;
                    إذا (diff < بهترين_diff) {
                        بهترين_diff = diff;
                        ibest = i;
                        إذا (!بهترين_diff) {
                            فراغ * ptr = b.ptr;
                            *actual_سائيز = b.size;
                            b.ptr = خالي_اشارو;
                            b.size = 0;
                            أرجع ptr;
                        }
                    }
                }
            }
        }
        إذا (ibest >= 0) {
            ggml_sycl_بفر& b = بفر_pool[ibest];
            فراغ * ptr = b.ptr;
            *actual_سائيز = b.size;
            b.ptr = خالي_اشارو;
            b.size = 0;
            أرجع ptr;
        }
        فراغ * ptr;
        size_t look_ahead_سائيز = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (فراغ *)ggml_sycl_malloc_device(look_ahead_سائيز, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        إذا (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_سائيز);
            أرجع خالي_اشارو;
        }

        *actual_سائيز = look_ahead_سائيز;
        pool_سائيز += look_ahead_سائيز;

#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_سائيز[id]/1024/1024), (uint32_t)(size/1024/1024));
#جيڪڏهن_پورو

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        أرجع ptr;
    }

    فراغ حرر(فراغ * ptr, size_t size) override {
        لكل (صحيح i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_بفر& b = بفر_pool[i];
            إذا (b.ptr == خالي_اشارو) {
                b.ptr = ptr;
                b.size = size;
                أرجع;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_سائيز -= size;
    }
};

// pool with virtual memory management
#إذا defined(GGML_SYCL_SUPPORT_VMM)
بنية ggml_sycl_pool_vmm : عام ggml_sycl_pool {
    ساكن ثابت size_t SYCL_POOL_VMM_MAX_سائيز = 1ull << 35; // 32 GB

    صحيح           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_استعمال_ٿيل = 0;
    size_t    pool_سائيز = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    بنية mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        فراغ * map_ptr;
    };
    std::vector<mapping> mappings;

    واضح ggml_sycl_pool_vmm(قطار_ptr qptr_, صحيح device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        إذا (pool_addr == 0) {
            أرجع;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        لكل (خودڪار & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_سائيز, ctx)));
    }

    فراغ * alloc(size_t size, size_t * actual_سائيز) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_بفر_ALIGNMENT);

        size_t avail = pool_سائيز - pool_استعمال_ٿيل;

        إذا (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_سائيز = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_سائيز + reserve_سائيز <= SYCL_POOL_VMM_MAX_سائيز);

            إذا (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_ٽيگ " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_سائيز / (1024 * 1024), pool_سائيز / (1024 * 1024),
                              (pool_سائيز + reserve_سائيز) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_سائيز)));

            // reserve virtual address space (if not already reserved)
            إذا (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_سائيز, ctx)));
            }

            // map at the end of the pool
            فراغ * map_ptr = خالي_اشارو;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_سائيز, reserve_سائيز,
                                    sycl::ext::oneapi::experimental::پتو_access_گهڻائي::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_سائيز += reserve_سائيز;
            ggml_sycl_memtrace_جوڙيو(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_سائيز);

#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (بنانشان ڊگهو ڊگهو) (pool_سائيز/1024/1024),
                          (بنانشان ڊگهو ڊگهو) (reserve_سائيز/1024/1024));
#جيڪڏهن_پورو
        }

        GGML_ASSERT(pool_addr != 0);

        فراغ * ptr = reinterpret_cast<فراغ *>(pool_addr + pool_استعمال_ٿيل);
        *actual_سائيز = size;
        pool_استعمال_ٿيل += size;

#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (بنانشان ڊگهو ڊگهو) size, ptr);
#جيڪڏهن_پورو

        أرجع ptr;
    }

    فراغ حرر(فراغ * ptr, size_t size) override {
#جيڪڏهن_وصف_ٿيل DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (بنانشان ڊگهو ڊگهو) size, ptr);
#جيڪڏهن_پورو

        pool_استعمال_ٿيل -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<فراغ *>(pool_addr + pool_استعمال_ٿيل));
    }
};
#جيڪڏهن_پورو // defined(GGML_SYCL_SUPPORT_VMM)

بنية ggml_sycl_pool_host : عام ggml_sycl_pool {
    قطار_ptr qptr;
    صحيح       device;

    بنية ggml_sycl_بفر {
        فراغ * ptr  = خالي_اشارو;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    ساكن constexpr صحيح          MAX_POOL_سائيز{ 64 };
    std::vector<ggml_sycl_بفر> بفر_pool = std::vector<ggml_sycl_بفر>(MAX_POOL_سائيز);
    size_t                        pool_سائيز   = 0;
    صحيح                           counter     = 0;

    واضح ggml_sycl_pool_host(قطار_ptr qptr_, صحيح device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        لكل (صحيح i = 0; i < MAX_POOL_سائيز; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            إذا (b.ptr != خالي_اشارو) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(b.ptr, *qptr)));
                b.ptr = خالي_اشارو;
                pool_سائيز -= b.size;
                b.size = 0;
            }
        }
    }

    فراغ * alloc(size_t size, size_t * actual_سائيز) override {
        إذا (counter == MAX_POOL_سائيز) {
            // the pool wrapped around, no slot is reserved for this allocation
            فراغ * ptr = خالي_اشارو;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (فراغ *) sycl::malloc_host(size, *qptr)));
            إذا (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                أرجع خالي_اشارو;
            }
            pool_سائيز    += size;
            *actual_سائيز = size;
            counter      = 0;
            أرجع ptr;
        }
        ggml_sycl_بفر & b = بفر_pool[counter];
        ++counter;

        إذا (b.ptr != خالي_اشارو) {
            إذا (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                فراغ * ptr = b.ptr;
                *actual_سائيز = b.size;
                b.ptr = خالي_اشارو;
                b.size = 0;
                أرجع ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(b.ptr, *qptr)));
            pool_سائيز -= b.size;
            b.ptr = خالي_اشارو;
            b.size = 0;
        }

        فراغ * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (فراغ *) sycl::malloc_host(size, *qptr)));
        إذا (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            أرجع خالي_اشارو;
        }
        pool_سائيز    += size;
        *actual_سائيز = size;
        أرجع ptr;
    }

    فراغ حرر(فراغ * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        لكل (صحيح i = 0; i < MAX_POOL_سائيز; ++i) {
            ggml_sycl_بفر & b = بفر_pool[i];
            إذا (b.ptr == خالي_اشارو) {
                b.ptr  = ptr;
                b.size = size;
                أرجع;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::حرر(ptr, *qptr)));
        pool_سائيز -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(قطار_ptr qptr, صحيح device) {
    // return pool for the host to speed up memory management
    أرجع std::unique_ptr<ggml_sycl_pool>(جديد ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(قطار_ptr qptr, صحيح device) {
#إذا defined(GGML_SYCL_SUPPORT_VMM)
    إذا (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        أرجع std::unique_ptr<ggml_sycl_pool>(جديد ggml_sycl_pool_vmm(qptr, device));
    }
#جيڪڏهن_پورو // defined(GGML_SYCL_SUPPORT_VMM)
    أرجع std::unique_ptr<ggml_sycl_pool>(جديد ggml_sycl_pool_leg(qptr, device));
}

/// kernels
تعريفنوع فراغ (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t صف_هيٺيون, ثابت int64_t صف_مٿاهون,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائيز,
    ثابت قطار_ptr &stream);



ساكن فراغ mul_mat_p021_f16_f32(
    ثابت فراغ * __restrict__ vx, ثابت عائم * __restrict__ y, عائم * __restrict__ dst,
    ثابت صحيح ncols_x, ثابت صحيح nrows_x, ثابت صحيح nchannels_x, ثابت صحيح nchannels_y,
    ثابت sycl::nd_شي<3> &شي_ct1) {

    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    ثابت صحيح صف_x = شي_ct1.get_local_range(1) * شي_ct1.get_گروهه(1) +
                      شي_ct1.get_local_id(1);
    ثابت صحيح channel = شي_ct1.get_local_range(0) * شي_ct1.get_گروهه(0) +
                        شي_ct1.get_local_id(0);
    ثابت صحيح channel_x = channel / (nchannels_y / nchannels_x);

    ثابت صحيح nrows_y = ncols_x;
    ثابت صحيح nrows_dst = nrows_x;
    ثابت صحيح صف_dst = صف_x;

    عائم tmp = 0.0f;

    لكل (صحيح کانو_x0 = 0; کانو_x0 < ncols_x;
         کانو_x0 += شي_ct1.get_local_range(2)) {
        ثابت صحيح کانو_x = کانو_x0 + شي_ct1.get_local_id(2);

        إذا (کانو_x >= ncols_x) {
            اكسر;
        }

        // x is transposed and permuted
        ثابت صحيح ix = صف_x*nchannels_x*ncols_x + channel_x*ncols_x + کانو_x;
        ثابت عائم xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<عائم, sycl::rounding_گهڻائي::automatic>()[0];

        ثابت صحيح صف_y = کانو_x;


        // y is not transposed but permuted
        ثابت صحيح iy = channel*nrows_y + صف_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ثابت صحيح idst = channel*nrows_dst + صف_dst;

    // sum up partial sums and write back result
#هدايت unroll
    لكل (صحيح نقاب = WARP_سائيز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتيب_بدلايو_sub_گروهه_by_xor(شي_ct1.get_sub_گروهه(), tmp, نقاب);
    }

    إذا (شي_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ساكن فراغ mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ثابت فراغ * __restrict__ vx, ثابت عائم * __restrict__ y, عائم * __restrict__ dst, ثابت صحيح ncols_x, ثابت صحيح nrows_x,
    ثابت صحيح صف_stride_x, ثابت صحيح channel_stride_x,ثابت صحيح channel_stride_y, ثابت صحيح channel_x_ورهائيندڙ,
    ثابت sycl::nd_شي<3> &شي_ct1) {

    ثابت sycl::half *x = (ثابت sycl::half *)vx;

    ثابت صحيح صف_x = شي_ct1.get_local_range(1) * شي_ct1.get_گروهه(1) +
                      شي_ct1.get_local_id(1);
    ثابت صحيح channel = شي_ct1.get_local_range(0) * شي_ct1.get_گروهه(0) +
                        شي_ct1.get_local_id(0);
    ثابت صحيح channel_x = channel / channel_x_ورهائيندڙ;

    ثابت صحيح nrows_dst = nrows_x;
    ثابت صحيح صف_dst   = صف_x;

    ثابت صحيح idst = channel*nrows_dst + صف_dst;

    عائم tmp = 0.0f;

    لكل (صحيح کانو_x0 = 0; کانو_x0 < ncols_x;
         کانو_x0 += شي_ct1.get_local_range(2)) {
        ثابت صحيح کانو_x = کانو_x0 + شي_ct1.get_local_id(2);

        إذا (کانو_x >= ncols_x) {
            اكسر;
        }

        ثابت صحيح صف_y = کانو_x;

        ثابت صحيح ix = channel_x*channel_stride_x + صف_x*صف_stride_x + کانو_x;
        ثابت صحيح iy = channel * channel_stride_y + صف_y;

        ثابت عائم xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<عائم, sycl::rounding_گهڻائي::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#هدايت unroll
    لكل (صحيح نقاب = WARP_سائيز / 2; نقاب > 0; نقاب >>= 1) {
        tmp +=
            dpct::ترتيب_بدلايو_sub_گروهه_by_xor(شي_ct1.get_sub_گروهه(), tmp, نقاب);
    }

    إذا (شي_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ساكن فراغ k_جوڙ_صفون_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols,
                           ثابت sycl::nd_شي<3> &شي_ct1) {
    ثابت صحيح صف = شي_ct1.get_گروهه(1);
    ثابت صحيح کانو = شي_ct1.get_local_id(2);

    عائم جوڙ = 0.0f;
    لكل (صحيح i = کانو; i < ncols; i += شي_ct1.get_local_range(2)) {
        جوڙ += x[صف * ncols + i];
    }

    جوڙ = warp_reduce_جوڙ(جوڙ, شي_ct1);

    إذا (کانو == 0) {
        dst[صف] = جوڙ;
    }
}


سانچو<قسم_نالو T>
ساكن مضمن فراغ ggml_sycl_مٽا_سٽا(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

سانچو <ggml_ترتيب_ڏيو_ترتيب ترتيب>
__dpct_inline__ ساكن فراغ
k_argsort_f32_i32(ثابت عائم *x, صحيح *dst, ثابت صحيح ncols, صحيح ncols_pad,
                  ثابت صحيح tasks_في_thread, ثابت sycl::nd_شي<3> &شي_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    صحيح کانو_اشاريو =  شي_ct1.get_local_id(2);
    صحيح صف = شي_ct1.get_گروهه(1);

    لكل (صحيح i = 0; i < tasks_في_thread; i++) {
        صحيح کانو = کانو_اشاريو * tasks_في_thread + i;
        إذا (کانو >= ncols_pad) {
            أرجع;
        }
    }

    ثابت عائم * x_صف = x + صف * ncols;
    خودڪار dst_صف = (صحيح *)dpct_local;

    // initialize indices
    لكل (صحيح i=0;i<tasks_في_thread;i++){
        صحيح کانو = کانو_اشاريو*tasks_في_thread+i;
        dst_صف[کانو] = کانو;
    }

    شي_ct1.barrier(sycl::access::fence_space::local_space);

    لكل (صحيح k = 2; k <= ncols_pad; k *= 2) {
        لكل (صحيح j = k / 2; j > 0; j /= 2) {
            لكل (صحيح i = 0; i < tasks_في_thread; i++) {
                صحيح کانو = کانو_اشاريو * tasks_في_thread + i;
                صحيح ixj = کانو ^ j;
                إذا (ixj > کانو) {
                    إذا ((کانو & k) == 0) {
                        إذا (dst_صف[کانو] >= ncols ||
                            (dst_صف[ixj] < ncols &&
                             (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC
                                  ? x_صف[dst_صف[کانو]] > x_صف[dst_صف[ixj]]
                                  : x_صف[dst_صف[کانو]] <
                                        x_صف[dst_صف[ixj]]))) {
                            ggml_sycl_مٽا_سٽا(dst_صف[کانو], dst_صف[ixj]);
                        }
                    } وإلا {
                        إذا (dst_صف[ixj] >= ncols ||
                            (dst_صف[کانو] < ncols &&
                             (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC
                                  ? x_صف[dst_صف[کانو]] < x_صف[dst_صف[ixj]]
                                  : x_صف[dst_صف[کانو]] >
                                        x_صف[dst_صف[ixj]]))) {
                            ggml_sycl_مٽا_سٽا(dst_صف[کانو], dst_صف[ixj]);
                        }
                    }
                }
                شي_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    لكل (صحيح i = 0; i < tasks_في_thread; i++) {
        صحيح کانو = کانو_اشاريو * tasks_في_thread + i;
        إذا (کانو < ncols) {
            dst[صف * ncols + کانو] = dst_صف[کانو];
        }
    }
}

ساكن فراغ diag_نقاب_inf_f32(ثابت عائم * x, عائم * dst, ثابت صحيح ncols, ثابت صحيح صفون_في_channel, ثابت صحيح n_past,
                              ثابت sycl::nd_شي<3> &شي_ct1) {
    ثابت صحيح کانو = شي_ct1.get_local_range(1) * شي_ct1.get_گروهه(1) +
                    شي_ct1.get_local_id(1);
    ثابت صحيح صف = شي_ct1.get_local_range(2) * شي_ct1.get_گروهه(2) +
                    شي_ct1.get_local_id(2);

    إذا (کانو >= ncols) {
        أرجع;
    }

    ثابت صحيح i = صف*ncols + کانو;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (کانو > n_past + صف % صفون_في_channel) * FLT_MAX;
}

ساكن فراغ scale_f32(ثابت عائم * x, عائم * dst, ثابت عائم scale, ثابت عائم bias, ثابت صحيح k,
                      ثابت sycl::nd_شي<3> &شي_ct1) {
    ثابت صحيح i = شي_ct1.get_local_range(2) * شي_ct1.get_گروهه(2) +
                  شي_ct1.get_local_id(2);

    إذا (i >= k) {
        أرجع;
    }

    dst[i] = scale * x[i] + bias;
}


ساكن فراغ ggml_mul_mat_p021_f16_f32_sycl(ثابت فراغ *vx, ثابت عائم *y,
                                           عائم *dst, ثابت صحيح ncols_x,
                                           ثابت صحيح nrows_x,
                                           ثابت صحيح nchannels_x,
                                           ثابت صحيح nchannels_y,
                                           قطار_ptr stream) {

    ثابت sycl::range<3> بلاڪ_nums(nchannels_y, nrows_x, 1);
    ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<3> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, شي_ct1);
            });
    }
}

ساكن فراغ ggml_mul_mat_vec_nc_f16_f32_sycl(
    ثابت فراغ *vx, ثابت عائم *y, عائم *dst, ثابت صحيح ncols_x,
    ثابت صحيح nrows_x, ثابت صحيح صف_stride_x, ثابت صحيح nchannels_x,
    ثابت صحيح nchannels_y, ثابت صحيح channel_stride_x, ثابت صحيح channel_stride_y, قطار_ptr stream) {

    ثابت sycl::range<3> بلاڪ_nums(nchannels_y, nrows_x, 1);
    ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<3> شي_ct1) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       صف_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, شي_ct1);
            });
    }
}



ساكن فراغ scale_f32_sycl(ثابت عائم *x, عائم *dst, ثابت عائم scale, ثابت عائم bias,
                           ثابت صحيح k, قطار_ptr stream) {
    ثابت صحيح انگ_بلاڪون = (k + SYCL_SCALE_بلاڪ_سائيز - 1) / SYCL_SCALE_بلاڪ_سائيز;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, انگ_بلاڪون) *
                              sycl::range<3>(1, 1, SYCL_SCALE_بلاڪ_سائيز),
                          sycl::range<3>(1, 1, SYCL_SCALE_بلاڪ_سائيز)),
        [=](sycl::nd_شي<3> شي_ct1) {
            scale_f32(x, dst, scale, bias, k, شي_ct1);
        });
}


ساكن فراغ جوڙ_صفون_f32_sycl(ثابت عائم *x, عائم *dst, ثابت صحيح ncols,
                              ثابت صحيح nrows, قطار_ptr stream) {
    ثابت sycl::range<3> بلاڪ_dims(1, 1, WARP_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                         [=](sycl::nd_شي<3> شي_ct1)
                             [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
                                 k_جوڙ_صفون_f32(x, dst, ncols, شي_ct1);
                             });
}

ساكن صحيح ايندڙ_power_جو_2(صحيح x) {
    صحيح n = 1;
    طالما (n < x) {
        n *= 2;
    }
    أرجع n;
}

ساكن فراغ init_argsort_indices_padded(
        صحيح * idx,
        ثابت صحيح nrows,
        ثابت صحيح ncols_pad,
        ثابت sycl::nd_شي<1> & شي_ct1) {
    ثابت size_t gid = شي_ct1.get_local_range(0) * شي_ct1.get_گروهه(0) + شي_ct1.get_local_id(0);
    ثابت size_t ڪل = (size_t) nrows * (size_t) ncols_pad;

    إذا (gid >= ڪل) {
        أرجع;
    }

    idx[gid] = (صحيح) (gid % (size_t) ncols_pad);
}

سانچو <ggml_ترتيب_ڏيو_ترتيب ترتيب>
ساكن فراغ argsort_f32_i32_global_pass(ثابت عائم *            x,
                                        صحيح *                    idx,
                                        ثابت صحيح                ncols,
                                        ثابت صحيح                nrows,
                                        ثابت صحيح                ncols_pad,
                                        ثابت صحيح                j,
                                        ثابت صحيح                k,
                                        ثابت sycl::nd_شي<1> & شي_ct1) {
    ثابت size_t gid   = شي_ct1.get_local_range(0) * شي_ct1.get_گروهه(0) + شي_ct1.get_local_id(0);
    ثابت size_t ڪل = (size_t) nrows * (size_t) ncols_pad;

    إذا (gid >= ڪل) {
        أرجع;
    }

    ثابت صحيح صف = (صحيح) (gid / (size_t) ncols_pad);
    ثابت صحيح کانو = (صحيح) (gid % (size_t) ncols_pad);
    ثابت صحيح ixj = کانو ^ j;

    إذا (ixj <= کانو || ixj >= ncols_pad) {
        أرجع;
    }

    ثابت size_t بنياد  = (size_t) صف * (size_t) ncols_pad;
    ثابت size_t pos_a = بنياد + (size_t) کانو;
    ثابت size_t pos_b = بنياد + (size_t) ixj;

    ثابت صحيح a = idx[pos_a];
    ثابت صحيح b = idx[pos_b];

    منطقي do_مٽا_سٽا = خطأ;

    إذا ((کانو & k) == 0) {
        إذا (a >= ncols ||
            (b < ncols &&
             (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC ?
                  x[(size_t) صف * (size_t) ncols + (size_t) a] > x[(size_t) صف * (size_t) ncols + (size_t) b] :
                  x[(size_t) صف * (size_t) ncols + (size_t) a] < x[(size_t) صف * (size_t) ncols + (size_t) b]))) {
            do_مٽا_سٽا = سچ;
        }
    } وإلا {
        إذا (b >= ncols ||
            (a < ncols &&
             (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC ?
                  x[(size_t) صف * (size_t) ncols + (size_t) a] < x[(size_t) صف * (size_t) ncols + (size_t) b] :
                  x[(size_t) صف * (size_t) ncols + (size_t) a] > x[(size_t) صف * (size_t) ncols + (size_t) b]))) {
            do_مٽا_سٽا = سچ;
        }
    }

    إذا (do_مٽا_سٽا) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

ساكن فراغ copy_argsort_indices_unpadded(ثابت صحيح *              idx_padded,
                                          صحيح *                    dst,
                                          ثابت صحيح                nrows,
                                          ثابت صحيح                ncols,
                                          ثابت صحيح                ncols_pad,
                                          ثابت sycl::nd_شي<1> & شي_ct1) {
    ثابت size_t gid   = شي_ct1.get_local_range(0) * شي_ct1.get_گروهه(0) + شي_ct1.get_local_id(0);
    ثابت size_t ڪل = (size_t) nrows * (size_t) ncols;

    إذا (gid >= ڪل) {
        أرجع;
    }

    ثابت صحيح صف = (صحيح) (gid / (size_t) ncols);
    ثابت صحيح کانو = (صحيح) (gid % (size_t) ncols);

    dst[(size_t) صف * (size_t) ncols + (size_t) کانو] = idx_padded[(size_t) صف * (size_t) ncols_pad + (size_t) کانو];
}

ساكن فراغ argsort_f32_i32_sycl(ثابت عائم *x, صحيح *dst, ثابت صحيح ncols,
                                 ثابت صحيح nrows, ggml_ترتيب_ڏيو_ترتيب ترتيب,
                                 قطار_ptr stream, صحيح device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ثابت صحيح ncols_pad = ايندڙ_power_جو_2(ncols);
    ثابت size_t shared_mem = (size_t) ncols_pad * حجم(صحيح);
    ثابت size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    إذا (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<صحيح> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        صحيح *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t بلاڪ_سائيز_3     = 256;
        ثابت size_t     ڪل_padded   = (size_t) nrows * (size_t) ncols_pad;
        ثابت size_t     nblocks_padded = (ڪل_padded + بلاڪ_سائيز_3 - 1) / بلاڪ_سائيز_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاڪ_سائيز_3), sycl::range<1>(بلاڪ_سائيز_3)),
            [=](sycl::nd_شي<1> شي_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, شي_ct1); });

        لكل (صحيح k = 2; k <= ncols_pad; k *= 2) {
            لكل (صحيح j = k / 2; j > 0; j /= 2) {
                إذا (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاڪ_سائيز_3), sycl::range<1>(بلاڪ_سائيز_3)),
                        [=](sycl::nd_شي<1> شي_ct1) {
                            argsort_f32_i32_global_pass<GGML_ترتيب_ڏيو_ترتيب_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, شي_ct1);
                        });
                } وإلا إذا (ترتيب == GGML_ترتيب_ڏيو_ترتيب_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * بلاڪ_سائيز_3), sycl::range<1>(بلاڪ_سائيز_3)),
                        [=](sycl::nd_شي<1> شي_ct1) {
                            argsort_f32_i32_global_pass<GGML_ترتيب_ڏيو_ترتيب_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, شي_ct1);
                        });
                } وإلا {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ثابت size_t ڪل   = (size_t) nrows * (size_t) ncols;
        ثابت size_t nblocks = (ڪل + بلاڪ_سائيز_3 - 1) / بلاڪ_سائيز_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * بلاڪ_سائيز_3), sycl::range<1>(بلاڪ_سائيز_3)),
                             [=](sycl::nd_شي<1> شي_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, شي_ct1);
                             });

        أرجع;
    }

    صحيح nth = 1;
    صحيح max_بلاڪ_سائيز = ggml_sycl_info().max_work_گروهه_sizes[device];
    طالما (nth < ncols_pad && nth < max_بلاڪ_سائيز)
        nth *= 2;
    إذا (nth > max_بلاڪ_سائيز)
        nth = max_بلاڪ_سائيز;

    ثابت صحيح tasks_في_thread = ncols_pad / nth;

    ثابت sycl::range<3> بلاڪ_dims(1, 1, nth);
    ثابت sycl::range<3> بلاڪ_nums(1, nrows, 1);

    إذا (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1) {
                    k_argsort_f32_i32<GGML_ترتيب_ڏيو_ترتيب_ASC>(
                        x, dst, ncols, ncols_pad, tasks_في_thread, شي_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } وإلا إذا (ترتيب == GGML_ترتيب_ڏيو_ترتيب_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<3> شي_ct1) {
                    k_argsort_f32_i32<GGML_ترتيب_ڏيو_ترتيب_DESC>(
                        x, dst, ncols, ncols_pad, tasks_في_thread, شي_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } وإلا {
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
ساكن فراغ چوٽي_k_جاچ_ملايو_f32(
    ثابت عائم *   src_vals,
    ثابت int32_t * src_map,
    ثابت صحيح       begin,
    ثابت صحيح       end,
    ثابت صحيح       k,
    ثابت صحيح       بلاڪ_سائيز_3,
    عائم *         shared_vals,
    صحيح *           shared_idx,
    عائم *         ٻاهريون_vals,
    int32_t *       ٻاهريون_idx,
    ثابت منطقي      swap01,
    ثابت sycl::nd_شي<1> & شي_ct1
) {
    ثابت صحيح tid = شي_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    عائم * lv = shared_vals + tid;
    صحيح * li = shared_idx + tid;

    لكل (صحيح i = 0; i < k; i++) {
        lv[i * بلاڪ_سائيز_3] = -FLT_MAX;
        li[i * بلاڪ_سائيز_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    عائم kth = -FLT_MAX;

    لكل (صحيح کانو = begin + tid; کانو < end; کانو += بلاڪ_سائيز_3) {
        عائم val = src_vals[کانو];

        إذا (val > kth) {
            صحيح pos = k - 1;
            طالما (pos > 0 && val > lv[(pos - 1) * بلاڪ_سائيز_3]) {
                pos--;
            }

            لكل (صحيح i = k - 1; i > pos; i--) {
                lv[i * بلاڪ_سائيز_3] = lv[(i - 1) * بلاڪ_سائيز_3];
                li[i * بلاڪ_سائيز_3] = li[(i - 1) * بلاڪ_سائيز_3];
            }
            lv[pos * بلاڪ_سائيز_3] = val;
            li[pos * بلاڪ_سائيز_3] = src_map ? src_map[کانو] : کانو;

            kth = lv[(k - 1) * بلاڪ_سائيز_3];
        }
    }

    شي_ct1.barrier(sycl::access::fence_space::local_space);

    إذا (tid != 0) {
        أرجع;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    عائم * fv = shared_vals + (size_t) k * بلاڪ_سائيز_3;
    صحيح * fi = shared_idx + (size_t) k * بلاڪ_سائيز_3;

    لكل (صحيح i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    عائم fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    لكل (صحيح t = 0; t < بلاڪ_سائيز_3; t++) {
        لكل (صحيح i = 0; i < k; i++) {
            عائم val = shared_vals[i * بلاڪ_سائيز_3 + t];

            إذا (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                اكسر;
            }

            صحيح idx = shared_idx[i * بلاڪ_سائيز_3 + t];

            صحيح pos = k - 1;
            طالما (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            لكل (صحيح j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    إذا (ٻاهريون_vals) {
        لكل (صحيح i = 0; i < k; i++) {
            ٻاهريون_vals[i] = fv[i];
        }
    }

    لكل (صحيح i = 0; i < k; i++) {
        ٻاهريون_idx[i] = fi[i];
    }

    إذا (swap01 && k > 1) {
        int32_t عارضي = ٻاهريون_idx[0];
        ٻاهريون_idx[0] = ٻاهريون_idx[1];
        ٻاهريون_idx[1] = عارضي;
    }
}

ساكن فراغ چوٽي_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت عائم * src,
    int32_t * dst_indices,
    ثابت int64_t ncols,
    ثابت int64_t nrows,
    ثابت صحيح k,
    dpct::قطار_ptr main_وهڪرو
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
    constexpr صحيح ورهايو_بلاڪ = 128;
    constexpr صحيح max_splits  = 128;
    constexpr صحيح min_کانن_ڳڻپ    = 8192;

    صحيح nsplit = 1;
    إذا (ncols >= min_کانن_ڳڻپ) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ثابت int64_t want = ncols / ورهايو_بلاڪ;
        nsplit = (صحيح) (want > max_splits ? max_splits : want);
    }

    إذا (nsplit > 1) {
        ثابت صحيح nchunk = (صحيح) ((ncols + nsplit - 1) / nsplit);
        ثابت size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<عائم>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        عائم *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ثابت sycl::range<1> بلاڪ_dims(ورهايو_بلاڪ);

        main_وهڪرو->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<عائم, 1> shared_vals(sycl::range<1>((ورهايو_بلاڪ + 1) * k), cgh);
            sycl::local_accessor<صحيح, 1> shared_idx(sycl::range<1>((ورهايو_بلاڪ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<1> شي_ct1) {
                    ثابت صحيح grp = شي_ct1.get_گروهه(0);
                    ثابت صحيح صف = grp / nsplit;
                    ثابت صحيح part = grp % nsplit;

                    ثابت صحيح begin = part * nchunk;
                    صحيح end = begin + nchunk;
                    إذا (end > (صحيح) ncols) {
                        end = (صحيح) ncols;
                    }

                    چوٽي_k_جاچ_ملايو_f32(
                        src + (int64_t) صف * ncols, خالي_اشارو, begin, end, k, ورهايو_بلاڪ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, خطأ, شي_ct1);
                });
        });

        main_وهڪرو->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<عائم, 1> shared_vals(sycl::range<1>((ورهايو_بلاڪ + 1) * k), cgh);
            sycl::local_accessor<صحيح, 1> shared_idx(sycl::range<1>((ورهايو_بلاڪ + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * بلاڪ_dims, بلاڪ_dims),
                [=](sycl::nd_شي<1> شي_ct1) {
                    ثابت صحيح صف = شي_ct1.get_گروهه(0);
                    ثابت size_t off = (size_t) صف * nsplit * k;

                    چوٽي_k_جاچ_ملايو_f32(
                        pv + off, pi + off, 0, nsplit * k, k, ورهايو_بلاڪ,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        خالي_اشارو, dst_indices + (int64_t) صف * k, سچ, شي_ct1);
                });
        });

        أرجع;
    }

    ثابت صحيح بلاڪ_سائيز_3 = 128;

    ثابت sycl::range<1> بلاڪ_dims(بلاڪ_سائيز_3);
    ثابت sycl::range<1> grid_dims(nrows);

    main_وهڪرو->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<عائم, 1> shared_vals(sycl::range<1>((بلاڪ_سائيز_3 + 1) * k), cgh);
        sycl::local_accessor<صحيح, 1> shared_idx(sycl::range<1>((بلاڪ_سائيز_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<1> شي_ct1) {
                ثابت صحيح صف = شي_ct1.get_گروهه(0);

                إذا (صف >= nrows) أرجع;

                چوٽي_k_جاچ_ملايو_f32(
                    src + (int64_t) صف * ncols, خالي_اشارو, 0, (صحيح) ncols, k, بلاڪ_سائيز_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    خالي_اشارو, dst_indices + (int64_t) صف * k, سچ, شي_ct1);
            });
    });
}

ساكن فراغ argmax_f32_i32_sycl(ثابت عائم *x, صحيح *dst, ثابت صحيح ncols,
                               ثابت صحيح nrows, قطار_ptr stream) {
    ثابت sycl::range<3> بلاڪ_dims(1, 1, SYCL_ARGMAX_بلاڪ_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, nrows, 1);
    ثابت size_t shared_mem = 256 * حجم(عائم);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<عائم, 1> shared_معلومات(
            sycl::range<1>(shared_mem/حجم(عائم)), cgh);
        sycl::local_accessor<صحيح, 1> shared_indices(
            sycl::range<1>(shared_mem/حجم(عائم)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
            [=](sycl::nd_شي<3> شي_ct1) {
                ثابت صحيح tid = شي_ct1.get_local_id(2);
                ثابت صحيح صف = شي_ct1.get_global_id(1);

                عائم max_val = -INFINITY;
                صحيح max_idx = -1;

                لكل (صحيح کانو = tid; کانو < ncols; کانو += 256) {
                    عائم val = x[صف * ncols + کانو];
                    إذا (val > max_val) {
                        max_val = val;
                        max_idx = کانو;
                    }
                }

                shared_معلومات[tid] = max_val;
                shared_indices[tid] = max_idx;
                شي_ct1.barrier(sycl::access::fence_space::local_space);

                لكل (صحيح stride = 256/2; stride > 0; stride >>= 1) {
                    إذا (tid < stride) {
                        عائم val1 = shared_معلومات[tid];
                        عائم val2 = shared_معلومات[tid + stride];
                        إذا (val2 > val1) {
                            shared_معلومات[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    شي_ct1.barrier(sycl::access::fence_space::local_space);
                }


                إذا (tid == 0) {
                    dst[صف] = shared_indices[0];
                }
            });
    });
}
ساكن فراغ diag_نقاب_inf_f32_sycl(ثابت عائم *x, عائم *dst,
                                   ثابت صحيح ncols_x, ثابت صحيح nrows_x,
                                   ثابت صحيح صفون_في_channel, ثابت صحيح n_past,
                                   قطار_ptr stream) {
    ثابت sycl::range<3> بلاڪ_dims(1, SYCL_DIAG_نقاب_INF_بلاڪ_سائيز, 1);
    ثابت صحيح بلاڪ_انگ_x = (ncols_x + SYCL_DIAG_نقاب_INF_بلاڪ_سائيز - 1) / SYCL_DIAG_نقاب_INF_بلاڪ_سائيز;
    ثابت sycl::range<3> بلاڪ_nums(1, بلاڪ_انگ_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
                         [=](sycl::nd_شي<3> شي_ct1) {
                             diag_نقاب_inf_f32(x, dst, ncols_x,
                                               صفون_في_channel, n_past,
                                               شي_ct1);
                         });
}

ساكن dpct::err0 ggml_sycl_cpy_tensor_2d(فراغ *dst,
                                          ثابت بنية ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_هيٺيون, int64_t i1_مٿاهون,
                                          قطار_ptr stream) حاول {

    dpct::memcpy_direction قسم;
    حرف * src_ptr;
    إذا (ggml_backend_بفر_is_host(src->بفر)) {
        قسم = dpct::host_منزل_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (حرف *) src->معلومات;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } وإلا إذا (ggml_backend_بفر_is_sycl(src->بفر)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        قسم    = dpct::device_منزل_device;
        src_ptr = (حرف *) src->معلومات;
    } وإلا إذا (ggml_backend_بفر_is_sycl_ورهايو(src->بفر)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_هيٺيون == 0 && i1_مٿاهون == src->ne[1]);
        قسم = dpct::device_منزل_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        صحيح id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (حرف *) extra->معلومات_device[id];
    } وإلا {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    حرف * dst_ptr = (حرف *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ثابت تعداد ggml_type type = src->type;
    ثابت int64_t ts = ggml_type_سائيز(type);
    ثابت int64_t bs = ggml_blck_سائيز(type);
    int64_t i1_diff = i1_مٿاهون - i1_هيٺيون;

    ثابت حرف * x = src_ptr + i1_هيٺيون*nb1 + i2*nb2 + i3*nb3;
    إذا (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        أرجع CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    قسم, *stream));

    } وإلا إذا (nb0 == ts) {
        أرجع CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, قسم, *stream));
    } وإلا {
        لكل (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ثابت فراغ * rx = (ثابت فراغ *) ((ثابت حرف *) x + i1*nb1);
            فراغ * rd = (فراغ *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, قسم, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            إذا (r != 0) أرجع r;
        }
        أرجع 0;
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

مضمن فراغ ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ثابت ggml_tensor *src0, ثابت ggml_tensor *src1, ggml_tensor *dst,
    ثابت حرف *src0_dd_i, ثابت عائم *src1_ddf_i, ثابت حرف *src1_ddq_i,
    عائم *dst_dd_i, ثابت int64_t صف_هيٺيون, ثابت int64_t صف_مٿاهون,
    ثابت int64_t src1_ncols, ثابت int64_t src1_padded_صف_سائيز,
    ثابت قطار_ptr &stream) حاول {

    GGML_ASSERT(src0_dd_i  != خالي_اشارو);
    GGML_ASSERT(src1_ddf_i != خالي_اشارو);
    GGML_ASSERT(dst_dd_i   != خالي_اشارو);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ثابت int64_t صف_diff = صف_مٿاهون - صف_هيٺيون;

    صحيح id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ثابت int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    صحيح ldc = id == ctx.device ? ne0 : صف_diff; // used by MKL only

#جيڪڏهن_وصف_ٿيل GGML_SYCL_F16
    منطقي use_fp16 = سچ;  // TODO(Yu) SYCL capability check
#وإلا
    منطقي use_fp16 = خطأ;
#جيڪڏهن_پورو

#إذا GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    إذا (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        صف_diff == src0->ne[1]) {
        استعمال bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        إذا (src1->type != GGML_TYPE_BF16) {
            ثابت منزل_bf16_sycl_t منزل_bf16_sycl = ggml_get_منزل_bf16_sycl(src1->type, dst);
            GGML_ASSERT(منزل_bf16_sycl != خالي_اشارو);
            منزل_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } وإلا {
            stream->يادگيري_نقل(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*حجم(bf16_t));
        }
        DnnlGemmWrapper::صف_gemm(ctx, صف_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::منزل_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::منزل_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::منزل_dt<عائم>(), stream);
        GGML_اڻ_استعمال_ٿيل(dst);
        GGML_اڻ_استعمال_ٿيل(src1_ddq_i);
        GGML_اڻ_استعمال_ٿيل(src1_padded_صف_سائيز);
        أرجع;
    }
#جيڪڏهن_پورو

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    منطقي src0_reordered = خطأ;
    إذا (ثابت خودڪار * src0_extra = static_cast<ثابت ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    إذا (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && صف_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     صف_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        أرجع;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    إذا ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        صف_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        إذا (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ثابت منزل_fp16_sycl_t منزل_fp16_sycl = ggml_get_منزل_fp16_sycl(src1->type, dst);
            GGML_ASSERT(منزل_fp16_sycl != خالي_اشارو);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            منزل_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ثابت sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ثابت sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        إذا (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ثابت منزل_fp16_sycl_t منزل_fp16_sycl = ggml_get_منزل_fp16_sycl(src0->type, dst);
            GGML_ASSERT(منزل_fp16_sycl != خالي_اشارو);
            size_t ne = صف_diff*ne00;
            src0_as_f16.alloc(ne);
            منزل_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ثابت sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ثابت sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#إذا GGML_SYCL_DNNL
        إذا (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::صف_gemm(ctx,صف_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::منزل_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::منزل_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::منزل_dt<عائم>(), stream);
        }
        وإلا
#جيڪڏهن_پورو
        {
            ثابت عائم alpha = 1.0f;
            ثابت عائم beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, صف_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_معلومات_t::real_half, ne00,
                src1_ptr, dpct::library_معلومات_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_معلومات_t::real_float, ldc,
                dpct::library_معلومات_t::real_float)));
        }
    } وإلا {
        ggml_sycl_pool_alloc<عائم> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<عائم> src1_ddq_as_f32(ctx.pool());
        إذا (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ثابت منزل_fp32_sycl_t منزل_fp32_sycl = ggml_get_منزل_fp32_sycl(src0->type, dst);
            GGML_ASSERT(منزل_fp32_sycl != خالي_اشارو);
            src0_ddq_as_f32.alloc(صف_diff*ne00);
            منزل_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), صف_diff*ne00, stream);
        }
        إذا (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ثابت منزل_fp32_sycl_t منزل_fp32_sycl = ggml_get_منزل_fp32_sycl(src1->type, dst);
            GGML_ASSERT(منزل_fp32_sycl != خالي_اشارو);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            منزل_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ثابت عائم * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ثابت عائم *) src0_dd_i : src0_ddq_as_f32.get();
        ثابت عائم * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ثابت عائم *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#إذا GGML_SYCL_DNNL
            ثابت int64_t gemm_flops = (int64_t)صف_diff * src1_ncols * ne10;
            ثابت منطقي use_mkl_سڌو = gemm_flops < 256 * 256 * 256;
            إذا (g_ggml_sycl_enable_dnn && !use_mkl_سڌو && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::صف_gemm(ctx, صف_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::منزل_dt<عائم>(), src1_ddf1_i, DnnlGemmWrapper::منزل_dt<عائم>(),
                                          dst_dd_i, DnnlGemmWrapper::منزل_dt<عائم>(), stream);
            }
            وإلا
#جيڪڏهن_پورو
            {
                ثابت عائم alpha = 1.0f;
                ثابت عائم beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::ڪالم_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, صف_diff,
                    src1_ncols, ne10, dpct::get_قدر(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_قدر(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_اڻ_استعمال_ٿيل(dst);
    GGML_اڻ_استعمال_ٿيل(src1_ddq_i);
    GGML_اڻ_استعمال_ٿيل(src1_padded_صف_سائيز);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

مضمن فراغ ggml_sycl_op_جوڙ(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    جوڙ_صفون_f32_sycl(src0_dd, dst_dd, ne, 1, main_وهڪرو);
}

مضمن فراغ ggml_sycl_op_جوڙ_صفون(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    جوڙ_صفون_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_وهڪرو);
}

مضمن فراغ ggml_sycl_op_سراسري(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    جوڙ_صفون_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_وهڪرو);

    main_وهڪرو->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> صف) {
            dst_dd[صف] /= ncols;
        }
    );
}


مضمن فراغ ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->معلومات);


    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);

    تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب = (تعداد ggml_ترتيب_ڏيو_ترتيب) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (صحيح *)dst_dd, ncols, nrows, ترتيب,
                         main_وهڪرو, ctx.device, ctx.pool());
}

ساكن فراغ ggml_sycl_op_چوٽي_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->معلومات);
    int32_t * dst_dd = static_cast<int32_t *>(dst->معلومات);

    ثابت صحيح k = dst->ne[0];
    ثابت int64_t ncols = src0->ne[0];
    ثابت int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    إذا (k <= SYCL_چوٽي_K_جاچ_ملايو_MAX_K) {
        چوٽي_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_وهڪرو);
    } وإلا {
        ggml_sycl_چوٽي_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_وهڪرو);
    }
}

مضمن فراغ ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->معلومات);

    ثابت int64_t ncols = dst->src[0]->ne[0];
    ثابت int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_وهڪرو);
}

مضمن فراغ ggml_sycl_op_diag_نقاب_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت int64_t ne00 = dst->src[0]->ne[0];
    ثابت int64_t ne01 = dst->src[0]->ne[1];
    ثابت صحيح nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ثابت صحيح n_past = ((int32_t *) dst->op_params)[0];

    diag_نقاب_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_وهڪرو);
}

ساكن فراغ tri_f32_sycl(
    ثابت عائم * src,
    عائم * dst,
    ثابت int64_t ne0,
    ثابت int64_t ne1,
    ثابت int64_t ne2,
    ثابت int64_t ne3,
    ثابت ggml_tri_type ttype,
    dpct::قطار_ptr main_وهڪرو
) {
    ثابت size_t ڪل = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_وهڪرو->parallel_for(sycl::range<1>(ڪل), [=](sycl::id<1> tid) {
        ثابت int64_t idx = (int64_t) tid[0];

        ثابت int64_t i0 = idx % ne0;
        ثابت int64_t t1 = idx / ne0;
        ثابت int64_t i1 = t1 % ne1;

        منطقي keep = خطأ;
        بدل (ttype) {
            حالة GGML_TRI_TYPE_هيٺاهون:      keep = (i0 <  i1); اكسر;
            حالة GGML_TRI_TYPE_هيٺاهون_DIAG: keep = (i0 <= i1); اكسر;
            حالة GGML_TRI_TYPE_مٿانهون:      keep = (i0 >  i1); اكسر;
            حالة GGML_TRI_TYPE_مٿانهون_DIAG: keep = (i0 >= i1); اكسر;
            افتراضي: keep = خطأ; اكسر;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

ساكن فراغ ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(src0->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    ثابت ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ثابت int64_t ne0 = src0->ne[0];
    ثابت int64_t ne1 = src0->ne[1];
    ثابت int64_t ne2 = src0->ne[2];
    ثابت int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_وهڪرو);
}


مضمن فراغ ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::قطار_ptr main_وهڪرو = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    ثابت عائم * src0_dd = static_cast<ثابت عائم *>(dst->src[0]->معلومات);
    عائم *       dst_dd  = static_cast<عائم *>(dst->معلومات);

    عائم scale;
    عائم bias;
    يادگيري_نقل(&scale, (عائم *) dst->op_params + 0, حجم(عائم));
    يادگيري_نقل(&bias,  (عائم *) dst->op_params + 1, حجم(عائم));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_وهڪرو);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

ساكن فراغ ggml_sycl_سيٽ_peer_access(ثابت صحيح n_ٽوڪن, [[maybe_unused]] صحيح main_device) {
    ساكن منطقي peer_access_enabled = خطأ;

    ثابت منطقي enable_peer_access = n_ٽوڪن <= GGML_SYCL_PEER_MAX_BATCH_سائيز;

    إذا (peer_access_enabled == enable_peer_access) {
        أرجع;
    }

#جيڪڏهن_وصف_ٿيل NDEBUG
    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        SYCL_CHECK(ggml_sycl_سيٽ_device(i));
    }

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        SYCL_CHECK(ggml_sycl_سيٽ_device(i));

        لكل (صحيح id_other = 0; id_other < ggml_sycl_info().device_ڳڻپ; ++id_other) {
            إذا (i == id_other) {
                استمر;
            }
            إذا (i != main_device && id_other != main_device) {
                استمر;
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
#جيڪڏهن_پورو // NDEBUG

    peer_access_enabled = enable_peer_access;
}

سانچو <سانچو <صحيح> قسم_نالو quantize_f>
ساكن فراغ ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                 ثابت ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) حاول {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ثابت int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ثابت int64_t ne0 = dst->ne[0];
    ثابت int64_t ne1 = dst->ne[1];

    ثابت صحيح nb2 = dst->nb[2];
    ثابت صحيح nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(dst->بفر));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(src1->بفر));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ثابت int64_t i02_ورهائيندڙ = ne12 / ne02;

    ثابت size_t src0_ts = ggml_type_سائيز(src0->type);
    ثابت size_t src0_bs = ggml_blck_سائيز(src0->type);
    ثابت size_t q8_1_ts = حجم(بلاڪ_q8_1);
    ثابت size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ثابت منطقي src0_is_contiguous = ggml_is_contiguous(src0);
    ثابت منطقي src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_کانو_سائيز = GGML_PAD(ne10, ميٽرڪس_صف_PADDING);

    ثابت منطقي ورهايو = ggml_backend_بفر_is_sycl_ورهايو(src0->بفر);
    GGML_ASSERT(!(ورهايو && ne02 > 1));
    GGML_ASSERT(!(ورهايو && ne03 > 1));
    GGML_ASSERT(!(ورهايو && ne02 < ne12));

    std::array<عائم, GGML_SYCL_MAX_DEVICES> tensor_ورهايو;
    إذا (ورهايو) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_ورهايو_بفر_type_context * buft_ctx = (ggml_backend_sycl_ورهايو_بفر_type_context *) src0->بفر->buft->context;
        tensor_ورهايو = buft_ctx->tensor_ورهايو;
    }

    بنية dev_معلومات {
        ggml_sycl_pool_alloc<حرف> src0_dd_alloc;
        ggml_sycl_pool_alloc<عائم> src1_ddf_alloc;
        ggml_sycl_pool_alloc<حرف> src1_ddq_alloc;
        ggml_sycl_pool_alloc<عائم> dst_dd_alloc;

        حرف *src0_dd = خالي_اشارو;
        عائم *src1_ddf = خالي_اشارو; // float
        حرف *src1_ddq = خالي_اشارو;  // q8_1
        عائم *dst_dd = خالي_اشارو;

        int64_t صف_هيٺيون;
        int64_t صف_مٿاهون;
    };

    dev_معلومات dev[GGML_SYCL_MAX_DEVICES];

    صحيح استعمال_ٿيل_devices = 0;
    قطار_ptr main_وهڪرو = ctx.stream();

    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        // by default, use all rows
        dev[i].صف_هيٺيون  = 0;
        dev[i].صف_مٿاهون = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        إذا (ورهايو) {
            ثابت int64_t rounding = get_صف_rounding(src0->type, tensor_ورهايو);

            إذا (i != 0) {
                dev[i].صف_هيٺيون  = ne01*tensor_ورهايو[i];
                إذا (dev[i].صف_هيٺيون < ne01) {
                    dev[i].صف_هيٺيون -= dev[i].صف_هيٺيون % rounding;
                }
            }

            إذا (i != ggml_sycl_info().device_ڳڻپ - 1) {
                dev[i].صف_مٿاهون  = ne01*tensor_ورهايو[i + 1];
                إذا (dev[i].صف_مٿاهون < ne01) {
                    dev[i].صف_مٿاهون -= dev[i].صف_مٿاهون % rounding;
                }
            }
        }
    }

    constexpr منطقي quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_سائيز>,
                                                      no_quantize_q8_1<QK8_1 / WARP_سائيز>>;
    لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
        إذا ((!ورهايو && i != ctx.device) || dev[i].صف_هيٺيون == dev[i].صف_مٿاهون) {
            استمر;
        }

        استعمال_ٿيل_devices++;

        ثابت منطقي src1_on_device = i == ctx.device;
        ثابت منطقي  dst_on_device = i == ctx.device;

        ggml_sycl_سيٽ_device(i);
        قطار_ptr stream = ctx.stream(i, 0);

        إذا (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = ورهايو ? (حرف *) src0_extra->معلومات_device[i] : (حرف *) src0->معلومات;
        } وإلا {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        إذا (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (عائم *) src1->معلومات;
        } وإلا {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        إذا constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_کانو_سائيز*q8_1_ts/q8_1_bs);

            إذا (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                حاول {
                    quantize_صف_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_کانو_سائيز, stream);
                } امسك (sycl::exception ثابت &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
                    std::ٻاهر_نڪرو(1);
                }
            }
        }

        إذا (dst_on_device) {
            dev[i].dst_dd = (عائم *) dst->معلومات;
        } وإلا {
            ثابت size_t سائيز_dst_ddf = ورهايو ? (dev[i].صف_مٿاهون - dev[i].صف_هيٺيون)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), سائيز_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    إذا (ورهايو && استعمال_ٿيل_devices > 1) {
        ggml_sycl_سيٽ_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ثابت int64_t src1_کانو_stride = ورهايو && استعمال_ٿيل_devices > 1 ? MUL_MAT_SRC1_کانو_STRIDE : ne11;
    لكل (int64_t src1_کانو_0 = 0; src1_کانو_0 < ne11; src1_کانو_0 += src1_کانو_stride) {
        ثابت int64_t is = ورهايو ? (src1_کانو_0/src1_کانو_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ثابت int64_t src1_ncols = src1_کانو_0 + src1_کانو_stride > ne11 ? ne11 - src1_کانو_0 : src1_کانو_stride;
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
            إذا ((!ورهايو && i != ctx.device) || dev[i].صف_هيٺيون == dev[i].صف_مٿاهون) {
                استمر;
            }

            ثابت منطقي src1_on_device = i == ctx.device;
            ثابت منطقي  dst_on_device = i == ctx.device;
            ثابت int64_t صف_diff = dev[i].صف_مٿاهون - dev[i].صف_هيٺيون;

            ggml_sycl_سيٽ_device(i);
            قطار_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            إذا (ورهايو && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            لكل (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ثابت int64_t i03 = i0 / ne12;
                ثابت int64_t i02 = i0 % ne12;

                ثابت size_t src1_ddq_i_وٿي = (i0*ne11 + src1_کانو_0) * src1_padded_کانو_سائيز*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                حرف  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ورهائيندڙ) * (ne01*ne00*src0_ts)/src0_bs;
                عائم * src1_ddf_i = (عائم *) ((حرف *) dev[i].src1_ddf +
                    (i0*ne11 + src1_کانو_0) * ne10 * ggml_type_سائيز(src1->type));
                حرف  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_وٿي;
                عائم *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_کانو_0) * (dst_on_device ? ne0 : صف_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                إذا (i == ctx.device) {
                    dst_dd_i += dev[i].صف_هيٺيون; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                إذا (src1_is_contiguous) {
                    إذا (i != ctx.device) {
                        إذا constexpr (quantize_enabled) {
                            حرف * src1_ddq_i_ذريعو = dev[ctx.device].src1_ddq + src1_ddq_i_وٿي;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->يادگيري_نقل(src1_ddq_i, src1_ddq_i_ذريعو,
                                                             src1_ncols * src1_padded_کانو_سائيز * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } وإلا {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ثابت حرف * src1_ddf_i_ذريعو = (ثابت حرف *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_کانو_0) * ne10 * ggml_type_سائيز(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_وهڪرو, src1_ddf_i, src1_ddf_i_ذريعو,
                                                               src1_ncols * ne10 * ggml_type_سائيز(src1->type))));
                        }
                    }
                } وإلا {
                    إذا (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_کانو_0,
                                                           src1_کانو_0 + src1_ncols, stream));
                    } وإلا {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    إذا constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        حاول {
                            quantize_صف_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_کانو_سائيز, stream);
                        } امسك (ثابت sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
                            std::ٻاهر_نڪرو(1);
                        }
                    }
                }

                إذا (src1_کانو_0 == 0 && !src0_is_contiguous && i02 % i02_ورهائيندڙ == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ورهائيندڙ, dev[i].صف_هيٺيون, dev[i].صف_مٿاهون, stream));
                }
                إذا (src1->type == GGML_TYPE_F16) {
                    src1_padded_کانو_سائيز = (i0 * ne11 + src1_کانو_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].صف_هيٺيون, dev[i].صف_مٿاهون, src1_ncols, src1_padded_کانو_سائيز, stream)));

                // copy dst to host or other device if necessary
                إذا (!dst_on_device) {
                    فراغ * dst_off_device = dst->معلومات;
                    إذا (ورهايو) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        عائم * dhf_dst_i = (عائم *) ((حرف *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*حجم(عائم));
                        dhf_dst_i += src1_کانو_0*ne0 + dev[i].صف_هيٺيون;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * حجم(عائم), dst_dd_i,
                            صف_diff * حجم(عائم), صف_diff * حجم(عائم),
                            src1_ncols, dpct::device_منزل_device, *stream)));
                    } وإلا {
                        عائم * dhf_dst_i = (عائم *) ((حرف *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*حجم(عائم));
                        dhf_dst_i += src1_کانو_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->يادگيري_نقل(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * حجم(عائم)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                إذا (ورهايو && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    إذا (ورهايو && ggml_sycl_info().device_ڳڻپ > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_کانو_STRIDE - 1) / MUL_MAT_SRC1_کانو_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_سيٽ_device(ctx.device);
        لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; ++i) {
            إذا (dev[i].صف_هيٺيون == dev[i].صف_مٿاهون) {
                استمر;
            }
            لكل (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_sycl_repeat_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_پٺ(ctx, dst);
}

ساكن فراغ ggml_sycl_get_صفون(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_صفون(ctx, dst);
}

ساكن فراغ ggml_sycl_get_صفون_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_صفون_پٺ(ctx, dst);
}

ساكن فراغ ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

ساكن فراغ ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

ساكن فراغ ggml_sycl_rms_norm_پٺ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_پٺ(ctx, dst);
}

ساكن فراغ ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

ساكن فراغ ggml_sycl_گروهه_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_گروهه_norm(ctx, dst);
}

ساكن فراغ ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                       ثابت ggml_tensor *src1,
                                       ggml_tensor *dst) حاول {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(src0->بفر));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ثابت int64_t ne00 = src0->ne[0];
    ثابت int64_t ne01 = src0->ne[1];
    ثابت int64_t ne02 = src0->ne[2];

    ثابت int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    قطار_ptr main_وهڪرو = ctx.stream();

    فراغ  * src0_ddq = src0->معلومات;
    عائم * src1_ddf = (عائم *) src1->معلومات;
    عائم * dst_ddf  = (عائم *) dst->معلومات;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_وهڪرو);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                     ثابت ggml_tensor *src1,
                                     ggml_tensor *dst) حاول {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(src0->بفر));
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

    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    قطار_ptr main_وهڪرو = ctx.stream();

    فراغ  * src0_ddq = src0->معلومات;
    عائم * src1_ddf = (عائم *) src1->معلومات;
    عائم * dst_ddf  = (عائم *) dst->معلومات;

    ثابت int64_t صف_stride_x = nb01 / حجم(sycl::half);
    ثابت int64_t channel_stride_x = nb02 / حجم(sycl::half);
    ثابت int64_t channel_stride_y = nb11 / حجم(عائم);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, صف_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_وهڪرو);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ k_compute_batched_ptrs(ثابت sycl::half * src0_as_f16, ثابت sycl::half * src1_as_f16, فراغ * dst,
                                   ثابت فراغ ** ptrs_src, فراغ ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ثابت sycl::nd_شي<3> & شي_ct1) {
    ثابت int64_t i13 = شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2) + شي_ct1.get_local_id(2);
    ثابت int64_t i12 = شي_ct1.get_گروهه(1) * شي_ct1.get_local_range(1) + شي_ct1.get_local_id(1);

    إذا (i13 >= ne13 || i12 >= ne12) {
        أرجع;
    }

    ثابت int64_t i03 = i13 / r3;
    ثابت int64_t i02 = i12 / r2;

    ثابت uint8_t * src0_بائيٽون = reinterpret_cast<ثابت uint8_t *>(src0_as_f16);
    ثابت uint8_t * src1_بائيٽون = reinterpret_cast<ثابت uint8_t *>(src1_as_f16);
    uint8_t *       dst_بائيٽون  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_بائيٽون + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_بائيٽون + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_بائيٽون + i12 * nbd2 + i13 * nbd3;
}

ساكن فراغ ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                           ثابت ggml_tensor * src1, ggml_tensor * dst) حاول {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(src0->بفر));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ٻٽو_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));
    قطار_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ثابت sycl::half * src0_f16 = static_cast<ثابت sycl::half *>(src0->معلومات);
    عائم *            dst_ddf  = static_cast<عائم *>(dst->معلومات);

    ثابت sycl::half * src1_f16       = static_cast<ثابت sycl::half *>(src1->معلومات);
#إذا GGML_SYCL_DNNL
    ثابت size_t       type_سائيز_src0 = ggml_type_سائيز(src0->type);
#جيڪڏهن_پورو
    ثابت size_t       type_سائيز_src1 = ggml_type_سائيز(src1->type);

    منطقي is_src0_cont_2 = ggml_is_contiguous_2(src0);
    منطقي is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_سائيز_src1;
    int64_t                          s12 = nb12 / type_سائيز_src1;
    int64_t                          s13 = nb13 / type_سائيز_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    إذا (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#إذا GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        صحيح آخري_dim=0;
        صحيح آخري_str=0;
        size_t سڀ_کان_وڏو_str=0;
        لكل(صحيح i = 0; i< 4; i++){
            // last stride is always the largest
            إذا(src1->nb[i] == سڀ_کان_وڏو_str){
                إذا(src1->ne[آخري_dim] == 1){
                    آخري_str = i;
                    آخري_dim = i;
                }
            }
            إذا(src1->nb[i] > سڀ_کان_وڏو_str){
                سڀ_کان_وڏو_str = src1->nb[i];
                آخري_str = i;
                آخري_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ثابت int64_t ne_src1 = src1->nb[آخري_str] * src1->ne[آخري_dim] / type_سائيز_src1;
        src1_f16_alloc.alloc(ne_src1);
        ثابت منزل_fp16_sycl_t منزل_fp16_sycl = ggml_get_منزل_fp16_sycl(src1->type, dst);
        GGML_ASSERT(منزل_fp16_sycl != خالي_اشارو);
        منزل_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# وإلا
        ثابت int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ثابت منزل_fp16_nc_sycl_t منزل_fp16_nc_sycl = ggml_get_منزل_fp16_nc_sycl(src1->type);
        GGML_ASSERT(منزل_fp16_nc_sycl != خالي_اشارو);
        منزل_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#جيڪڏهن_پورو

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = سچ;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_معلومات_t mkl_compute_type = dpct::library_معلومات_t::real_float;
    dpct::library_معلومات_t mkl_معلومات_type    = dpct::library_معلومات_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ثابت عائم alpha_f32 = 1.0f;
    ثابت عائم beta_f32  = 0.0f;

    ثابت فراغ * alpha = &alpha_f32;
    ثابت فراغ * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

#إذا GGML_SYCL_DNNL
    إذا (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_سائيز_src0;
            int64_t str_a1 = nb01 / type_سائيز_src0;
            int64_t str_a2 = nb02 / type_سائيز_src0;

            int64_t str_b0 = nb10 / type_سائيز_src1;
            int64_t str_b1 = nb11 / type_سائيز_src1;
            int64_t str_b2 = nb12 / type_سائيز_src1;

            خودڪار launch_gemm_for_batches = [&ctx, queue](ثابت sycl::half *src0,
                                                ثابت sycl::half *src1, عائم *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                منطقي supported_نشر = batchb == batcha ? سچ
                        : batchb == 1 || batcha == 1        ? سچ
                                                            : خطأ;
                إذا (supported_نشر) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::منزل_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::منزل_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::منزل_dt<عائم>(), queue, batcha, batchb);
                } وإلا {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    إذا (batches0 > batches1) {
                        int64_t انگ_mul_mats = batches1;
                        int64_t sub_batch = batches0 / انگ_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        لكل (int64_t i0 = 0; i0 < انگ_mul_mats; i0++) {
                            ثابت sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ثابت sycl::half *src1_shifted = src1 + (sb2 * i0);
                            عائم *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::منزل_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::منزل_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::منزل_dt<عائم>(),
                                    queue, sub_batch, 1);
                        }
                    } وإلا {
                        int64_t انگ_mul_mats = batches0;
                        int64_t sub_batch = batches1 / انگ_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        لكل (int64_t i1 = 0; i1 < انگ_mul_mats; i1++) {
                            ثابت sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ثابت sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            عائم *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::منزل_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::منزل_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::منزل_dt<عائم>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ثابت منطقي cont_batches_dim2_a = nb02 * ne02 == nb03;
            ثابت منطقي cont_batches_dim2_b = nb12 * ne12 == nb13;
            ثابت منطقي cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ثابت منطقي cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            إذا (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / حجم(عائم));
            } وإلا إذا (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_سائيز_src0;
                int64_t str_b3 = nb13 / type_سائيز_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / حجم(عائم));
            } وإلا {
                لكل (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ثابت sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_سائيز_src0);
                    ثابت sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_سائيز_src1);
                    عائم *dst_shifted = dst_ddf + (nb3 * b_a / حجم(عائم));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / حجم(عائم));
                }
            }

    }
    وإلا
#جيڪڏهن_پورو
    {
        إذا (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ثابت int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ثابت int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_معلومات_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_معلومات_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_معلومات_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } وإلا {
            ثابت صحيح ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ثابت فراغ *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<فراغ *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<ميٽرڪس_info_t<عائم>> ميٽرڪس_info(ctx.host_pool(), 1);

            sycl::range<3> بلاڪ_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ثابت فراغ ** ptrs_src_get = ptrs_src.get();
                فراغ **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * حجم(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * حجم(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(بلاڪ_dims, بلاڪ_dims), [=](sycl::nd_شي<3> شي_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, شي_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ثابت فراغ **) (ptrs_src.get() + 0 * ne23), dpct::library_معلومات_t::real_half, nb01 / nb00,
                (ثابت فراغ **) (ptrs_src.get() + 1 * ne23), dpct::library_معلومات_t::real_half, s11, beta,
                (فراغ **) (ptrs_dst.get() + 0 * ne23), mkl_معلومات_type, ne0, ne23, mkl_compute_type, ميٽرڪس_info.get())));
        }
    }
} امسك (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

تعداد صنف mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

مضمن منطقي ggml_sycl_supports_mmq(تعداد ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_اڻ_استعمال_ٿيل(type);
    أرجع خطأ;
}

مضمن منطقي ggml_sycl_supports_reorder_mul_mat_sycl(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
            أرجع سچ;
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
            أرجع !g_ggml_sycl_prioritize_dmmv;
        افتراضي:
            أرجع خطأ;
    }
}

مضمن منطقي ggml_sycl_supports_reorder_dmmv(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

مضمن منطقي ggml_sycl_supports_reorder_mmvq(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_IQ3_XXS:
        حالة GGML_TYPE_IQ3_S:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

ساكن منطقي ggml_sycl_supports_reorder_esimd(تعداد ggml_type type) {
#جيڪڏهن_وصف_ٿيل GGML_SYCL_DMMV_HAS_ESIMD
    بدل (type) {
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_Q8_0:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
#وإلا
    GGML_اڻ_استعمال_ٿيل(type);
    أرجع خطأ;
#جيڪڏهن_پورو
}

ساكن منطقي ggml_sycl_supports_dmmv(تعداد ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_Q1_0:
        حالة GGML_TYPE_Q4_0:
        حالة GGML_TYPE_Q4_1:
        حالة GGML_TYPE_Q5_0:
        حالة GGML_TYPE_Q5_1:
        حالة GGML_TYPE_Q8_0:
        حالة GGML_TYPE_Q2_K:
        حالة GGML_TYPE_Q3_K:
        حالة GGML_TYPE_Q4_K:
        حالة GGML_TYPE_Q5_K:
        حالة GGML_TYPE_Q6_K:
        حالة GGML_TYPE_F16:
        حالة GGML_TYPE_BF16:
            أرجع سچ;
        افتراضي:
            أرجع خطأ;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
ساكن مضمن فراغ * sycl_ext_malloc_device(dpct::قطار_ptr stream, size_t size) {
    منطقي use_async = g_ggml_sycl_use_async_mem_op;
#إذا defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_يادگيري_ALLOC
    إذا (use_async) {
        فراغ * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_جوڙيو(GGML_SYCL_MEM_ASYNC, ptr, size);
        أرجع ptr;
    }
#وإلا
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#جيڪڏهن_پورو
    أرجع ggml_sycl_malloc_device(size, *stream);
}

ساكن مضمن فراغ sycl_ext_free(dpct::قطار_ptr stream, فراغ * ptr) {
    منطقي use_async = g_ggml_sycl_use_async_mem_op;
#إذا defined(GGML_SYCL_گراف) && SYCL_EXT_ONEAPI_ASYNC_يادگيري_ALLOC
    إذا (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        أرجع;
    }
#وإلا
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#جيڪڏهن_پورو
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
بنية sycl_reorder_عارضي_بفر {
    فراغ *          ptr  = خالي_اشارو;
    dpct::قطار_ptr stream;

    sycl_reorder_عارضي_بفر(dpct::قطار_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HOST_MEM_FALLBACK
        إذا (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            إذا (ptr) {
                host_fallback = سچ;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#جيڪڏهن_پورو
    }

    ~sycl_reorder_عارضي_بفر() {
        إذا (!ptr) {
            أرجع;
        }
        إذا (host_fallback) {
            sycl::حرر(ptr, *stream);
        } وإلا {
            sycl_ext_free(stream, ptr);
        }
    }

    واضح عامل منطقي() ثابت { أرجع ptr != خالي_اشارو; }

    sycl_reorder_عارضي_بفر(ثابت sycl_reorder_عارضي_بفر &)            = ميٽيو;
    sycl_reorder_عارضي_بفر & عامل=(ثابت sycl_reorder_عارضي_بفر &) = ميٽيو;

خاص:
    منطقي host_fallback = خطأ;
};

ساكن منطقي reorder_qw_q4_0(uint8_t * معلومات_device, ثابت صحيح ncols, ثابت صحيح nrows, size_t size, size_t وٿي,
                            dpct::قطار_ptr stream) {
    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % حجم(بلاڪ_q4_0) == 0));
    GGML_ASSERT((وٿي % حجم(بلاڪ_q4_0) == 0));
    صحيح وٿي_blks = وٿي / حجم(بلاڪ_q4_0);
    خودڪار qs_ptr      = معلومات_device + وٿي_blks * QK4_0 / 2;
    خودڪار d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + وٿي_blks;

    خودڪار reorder_event = stream->parallel_for(
        size / حجم(بلاڪ_q4_0),
            [=](خودڪار i) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
            ثابت بلاڪ_q4_0* x = (ثابت بلاڪ_q4_0*)tmp_buf;
            ثابت صحيح ib = i;

            لكل (صحيح j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q8_0(uint8_t * معلومات_device, ثابت صحيح ncols, ثابت صحيح nrows, size_t size, size_t وٿي,
                            dpct::قطار_ptr stream) {
    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % حجم(بلاڪ_q8_0) == 0));
    GGML_ASSERT((وٿي % حجم(بلاڪ_q8_0) == 0));
    صحيح وٿي_blks = وٿي / حجم(بلاڪ_q8_0);
    خودڪار qs_ptr = معلومات_device + وٿي_blks * QK8_0;
    خودڪار d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + وٿي_blks;

    خودڪار reorder_event = stream->parallel_for(
        size / حجم(بلاڪ_q8_0),
            [=](خودڪار i) [[sycl::reqd_sub_گروهه_سائيز(WARP_سائيز)]] {
            ثابت بلاڪ_q8_0* x = (ثابت بلاڪ_q8_0*)tmp_buf;
            ثابت صحيح ib = i;

            لكل (صحيح j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
ساكن منطقي reorder_qw_q8_0_moe(uint8_t * معلومات_device, size_t expert_بائيٽون, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائيٽون % حجم(بلاڪ_q8_0) == 0);
    ثابت صحيح    بلاڪون_في_expert = (صحيح) (expert_بائيٽون / حجم(بلاڪ_q8_0));
    ثابت size_t ڪل_بائيٽون       = expert_بائيٽون * (size_t) n_expert;

    sycl_reorder_عارضي_بفر tmp(stream, ڪل_بائيٽون);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ڪل_بائيٽون);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, ڪل_بائيٽون)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحيح ڪل_بلاڪون = بلاڪون_في_expert * (صحيح) n_expert;
    خودڪار reorder_event = stream->parallel_for(ڪل_بلاڪون, [=](خودڪار gb_) {
        ثابت صحيح          gb   = gb_;
        ثابت صحيح          e    = gb / بلاڪون_في_expert;
        ثابت صحيح          ib   = gb % بلاڪون_في_expert;
        ثابت بلاڪ_q8_0 * x    = (ثابت بلاڪ_q8_0 *) (tmp_buf + (size_t) e * expert_بائيٽون);
        uint8_t *          بنياد = معلومات_device + (size_t) e * expert_بائيٽون;

        خودڪار * qs_ptr = بنياد;
        خودڪار * d_ptr  = (sycl::half *) (qs_ptr + (size_t) بلاڪون_في_expert * QK8_0);

        لكل (صحيح j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q4_k(uint8_t * معلومات_device, size_t size, size_t وٿي, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_q4_K) == 0);
    GGML_ASSERT(وٿي % حجم(بلاڪ_q4_K) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_q4_K);

    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار * qs_ptr     = معلومات_device;
    خودڪار * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    خودڪار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائيز * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_q4_K * x  = (ثابت بلاڪ_q4_K *) tmp_buf;
        ثابت صحيح          ib = i;

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        لكل (صحيح j = 0; j < K_SCALE_سائيز; ++j) {
            scales_ptr[ib * K_SCALE_سائيز + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

// Reorder each expert slice into a self-contained SoA layout.
ساكن منطقي reorder_qw_q4_k_moe(uint8_t * معلومات_device, size_t expert_بائيٽون, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائيٽون % حجم(بلاڪ_q4_K) == 0);
    ثابت صحيح    بلاڪون_في_expert = (صحيح) (expert_بائيٽون / حجم(بلاڪ_q4_K));
    ثابت size_t ڪل_بائيٽون       = expert_بائيٽون * (size_t) n_expert;

    sycl_reorder_عارضي_بفر tmp(stream, ڪل_بائيٽون);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ڪل_بائيٽون);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, ڪل_بائيٽون)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحيح ڪل_بلاڪون = بلاڪون_في_expert * (صحيح) n_expert;
    خودڪار reorder_event = stream->parallel_for(ڪل_بلاڪون, [=](خودڪار gb_) {
        ثابت صحيح          gb   = gb_;
        ثابت صحيح          e    = gb / بلاڪون_في_expert;
        ثابت صحيح          ib   = gb % بلاڪون_في_expert;
        ثابت بلاڪ_q4_K * x    = (ثابت بلاڪ_q4_K *) (tmp_buf + (size_t) e * expert_بائيٽون);
        uint8_t *          بنياد = معلومات_device + (size_t) e * expert_بائيٽون;

        خودڪار * qs_ptr     = بنياد;
        خودڪار * scales_ptr = qs_ptr + QK_K / 2 * بلاڪون_في_expert;
        خودڪار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائيز * بلاڪون_في_expert);

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        لكل (صحيح j = 0; j < K_SCALE_سائيز; ++j) {
            scales_ptr[ib * K_SCALE_سائيز + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
ساكن منطقي reorder_qw_q5_k_moe(uint8_t * معلومات_device, size_t expert_بائيٽون, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائيٽون % حجم(بلاڪ_q5_K) == 0);
    ثابت صحيح    بلاڪون_في_expert = (صحيح) (expert_بائيٽون / حجم(بلاڪ_q5_K));
    ثابت size_t ڪل_بائيٽون       = expert_بائيٽون * (size_t) n_expert;

    sycl_reorder_عارضي_بفر tmp(stream, ڪل_بائيٽون);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ڪل_بائيٽون);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, ڪل_بائيٽون)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحيح ڪل_بلاڪون = بلاڪون_في_expert * (صحيح) n_expert;
    خودڪار reorder_event = stream->parallel_for(ڪل_بلاڪون, [=](خودڪار gb_) {
        ثابت صحيح          gb   = gb_;
        ثابت صحيح          e    = gb / بلاڪون_في_expert;
        ثابت صحيح          ib   = gb % بلاڪون_في_expert;
        ثابت بلاڪ_q5_K * x    = (ثابت بلاڪ_q5_K *) (tmp_buf + (size_t) e * expert_بائيٽون);
        uint8_t *          بنياد = معلومات_device + (size_t) e * expert_بائيٽون;

        خودڪار * qs_ptr     = بنياد;
        خودڪار * qh_ptr     = qs_ptr + (QK_K / 2) * بلاڪون_في_expert;
        خودڪار * scales_ptr = qh_ptr + (QK_K / 8) * بلاڪون_في_expert;
        خودڪار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائيز * بلاڪون_في_expert);

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        لكل (صحيح j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        لكل (صحيح j = 0; j < K_SCALE_سائيز; ++j) {
            scales_ptr[ib * K_SCALE_سائيز + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
ساكن منطقي reorder_qw_q6_k_moe(uint8_t * معلومات_device, size_t expert_بائيٽون, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائيٽون % حجم(بلاڪ_q6_K) == 0);
    ثابت صحيح    بلاڪون_في_expert = (صحيح) (expert_بائيٽون / حجم(بلاڪ_q6_K));
    ثابت size_t ڪل_بائيٽون       = expert_بائيٽون * (size_t) n_expert;

    sycl_reorder_عارضي_بفر tmp(stream, ڪل_بائيٽون);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, ڪل_بائيٽون);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, ڪل_بائيٽون)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ثابت صحيح ڪل_بلاڪون = بلاڪون_في_expert * (صحيح) n_expert;
    خودڪار reorder_event = stream->parallel_for(ڪل_بلاڪون, [=](خودڪار gb_) {
        ثابت صحيح          gb   = gb_;
        ثابت صحيح          e    = gb / بلاڪون_في_expert;
        ثابت صحيح          ib   = gb % بلاڪون_في_expert;
        ثابت بلاڪ_q6_K * x    = (ثابت بلاڪ_q6_K *) (tmp_buf + (size_t) e * expert_بائيٽون);
        uint8_t *          بنياد = معلومات_device + (size_t) e * expert_بائيٽون;

        خودڪار * ql_ptr     = بنياد;
        خودڪار * qh_ptr     = ql_ptr + (QK_K / 2) * بلاڪون_في_expert;
        خودڪار * scales_ptr = qh_ptr + (QK_K / 4) * بلاڪون_في_expert;
        خودڪار * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * بلاڪون_في_expert);

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        لكل (صحيح j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        لكل (صحيح j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
ساكن منطقي reorder_qw_mxfp4_moe(uint8_t * معلومات_device, size_t expert_بائيٽون, int64_t n_expert, dpct::قطار_ptr stream) {
    GGML_ASSERT(expert_بائيٽون % حجم(بلاڪ_mxfp4) == 0);
    ثابت صحيح     بلاڪون_في_expert = (صحيح) (expert_بائيٽون / حجم(بلاڪ_mxfp4));
    ثابت size_t  max_chunk_بائيٽون   = 32u << 20;
    ثابت int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_بائيٽون / expert_بائيٽون)));

    sycl_reorder_عارضي_بفر tmp(stream, (size_t) chunk_experts * expert_بائيٽون);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_بائيٽون);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    لكل (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ثابت int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = معلومات_device + (size_t) e0 * expert_بائيٽون;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, chunk, (size_t) n_chunk * expert_بائيٽون)));
        إذا (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        ثابت صحيح ڪل_بلاڪون = بلاڪون_في_expert * (صحيح) n_chunk;
        خودڪار reorder_event = stream->parallel_for(ڪل_بلاڪون, [=](خودڪار gb_) {
            ثابت صحيح           gb   = gb_;
            ثابت صحيح           e    = gb / بلاڪون_في_expert;
            ثابت صحيح           ib   = gb % بلاڪون_في_expert;
            ثابت بلاڪ_mxfp4 * x    = (ثابت بلاڪ_mxfp4 *) (tmp_buf + (size_t) e * expert_بائيٽون);
            uint8_t *           بنياد = chunk + (size_t) e * expert_بائيٽون;

            uint8_t * qs_ptr = بنياد;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) بلاڪون_في_expert;

            لكل (صحيح j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        إذا (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q2_k(uint8_t * معلومات_device, size_t size, size_t وٿي, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_q2_K) == 0);
    GGML_ASSERT(وٿي % حجم(بلاڪ_q2_K) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_q2_K);

    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار *        qs_ptr     = معلومات_device;
    خودڪار *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_q2_K * x  = (ثابت بلاڪ_q2_K *) tmp_buf;
        ثابت صحيح          ib = i;

        لكل (صحيح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        لكل (صحيح j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q3_k(uint8_t * معلومات_device, size_t size, size_t وٿي, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_q3_K) == 0);
    GGML_ASSERT(وٿي % حجم(بلاڪ_q3_K) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_q3_K);

    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار *       qs_ptr     = معلومات_device;
    خودڪار *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    خودڪار *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_q3_K * x  = (ثابت بلاڪ_q3_K *) tmp_buf;
        ثابت صحيح          ib = i;

        لكل (صحيح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        لكل (صحيح j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        لكل (صحيح j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q5_k(uint8_t * معلومات_device, size_t size, size_t وٿي, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_q5_K) == 0);
    GGML_ASSERT(وٿي % حجم(بلاڪ_q5_K) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_q5_K);

    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار * qs_ptr     = معلومات_device;
    خودڪار * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    خودڪار * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    خودڪار * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_سائيز * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_q5_K * x  = (ثابت بلاڪ_q5_K *) tmp_buf;
        ثابت صحيح          ib = i;

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        لكل (صحيح j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        لكل (صحيح j = 0; j < K_SCALE_سائيز; ++j) {
            scales_ptr[ib * K_SCALE_سائيز + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_q6_k(uint8_t * معلومات_device, size_t size, size_t وٿي, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_q6_K) == 0);
    GGML_ASSERT(وٿي % حجم(بلاڪ_q6_K) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_q6_K);

    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار *       ql_ptr     = معلومات_device;
    خودڪار *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    خودڪار *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_q6_K * x  = (ثابت بلاڪ_q6_K *) tmp_buf;
        ثابت صحيح          ib = i;

        ثابت uint8_t * ql              = x[ib].ql;
        ثابت uint8_t * qh              = x[ib].qh;
        uint8_t *       بنياد_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       بنياد_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       بنياد_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        لكل (صحيح j = 0; j < QK_K / 2; ++j) {
            بنياد_ql_ptr[j] = ql[j];
        }
        لكل (صحيح j = 0; j < QK_K / 4; ++j) {
            بنياد_qh_ptr[j] = qh[j];
        }

        لكل (صحيح j = 0; j < QK_K / 16; ++j) {
            بنياد_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_iq3_s(uint8_t * معلومات_device, size_t size, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_iq3_s) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_iq3_s);
    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار * qs_ptr       = معلومات_device;
    خودڪار * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    خودڪار * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    خودڪار * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_iq3_s * x = (ثابت بلاڪ_iq3_s *) tmp_buf;
        ثابت صحيح ib = i;

        لكل (صحيح j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        لكل (صحيح j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        لكل (صحيح j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (حجم(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        لكل (صحيح j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[حجم(ggml_half) + j] = x[ib].scales[j];
        }
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw_iq3_xxs(uint8_t * معلومات_device, size_t size, dpct::قطار_ptr stream) {
    GGML_ASSERT(size % حجم(بلاڪ_iq3_xxs) == 0);

    ثابت صحيح nblocks = size / حجم(بلاڪ_iq3_xxs);
    sycl_reorder_عارضي_بفر tmp(stream, size);
    إذا (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        أرجع خطأ;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->يادگيري_نقل(tmp_buf, معلومات_device, size)));
    إذا (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    خودڪار * qs_ptr = معلومات_device;
    خودڪار * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    خودڪار reorder_event = stream->parallel_for(nblocks, [=](خودڪار i) {
        ثابت بلاڪ_iq3_xxs * x = (ثابت بلاڪ_iq3_xxs *) tmp_buf;
        ثابت صحيح ib = i;

        لكل (صحيح j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    إذا (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    أرجع سچ;
}

ساكن منطقي reorder_qw(ثابت ggml_tensor * src0, dpct::قطار_ptr stream) {
    uint8_t * معلومات_device = (uint8_t *) src0->معلومات;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    إذا (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        بدل (src0->type) {
            حالة GGML_TYPE_Q8_0:
                أرجع reorder_qw_q8_0_moe(معلومات_device, src0->nb[2], src0->ne[2], stream);
            حالة GGML_TYPE_Q4_K:
                أرجع reorder_qw_q4_k_moe(معلومات_device, src0->nb[2], src0->ne[2], stream);
            حالة GGML_TYPE_Q5_K:
                أرجع reorder_qw_q5_k_moe(معلومات_device, src0->nb[2], src0->ne[2], stream);
            حالة GGML_TYPE_Q6_K:
                أرجع reorder_qw_q6_k_moe(معلومات_device, src0->nb[2], src0->ne[2], stream);
            حالة GGML_TYPE_MXFP4:
                أرجع reorder_qw_mxfp4_moe(معلومات_device, src0->nb[2], src0->ne[2], stream);
            افتراضي:
                أرجع خطأ;
        }
    }

    بدل (src0->type) {
        حالة GGML_TYPE_Q4_0:
            أرجع reorder_qw_q4_0(معلومات_device, ncols, nrows, size, 0, stream);
        حالة GGML_TYPE_Q8_0:
            أرجع reorder_qw_q8_0(معلومات_device, ncols, nrows, size, 0, stream);
        حالة GGML_TYPE_Q2_K:
            أرجع reorder_qw_q2_k(معلومات_device, size, 0, stream);
        حالة GGML_TYPE_Q3_K:
            أرجع reorder_qw_q3_k(معلومات_device, size, 0, stream);
        حالة GGML_TYPE_Q4_K:
            أرجع reorder_qw_q4_k(معلومات_device, size, 0, stream);
        حالة GGML_TYPE_Q5_K:
            أرجع reorder_qw_q5_k(معلومات_device, size, 0, stream);
        حالة GGML_TYPE_Q6_K:
            أرجع reorder_qw_q6_k(معلومات_device, size, 0, stream);
        حالة GGML_TYPE_IQ3_XXS:
            أرجع reorder_qw_iq3_xxs(معلومات_device, size, stream);
        حالة GGML_TYPE_IQ3_S:
            أرجع reorder_qw_iq3_s(معلومات_device, size, stream);
        افتراضي:
            أرجع خطأ;
    }
}

ساكن منطقي should_reorder_tensor(ggml_backend_sycl_context& ctx, ثابت ggml_tensor * dst) {
    أرجع g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

ساكن فراغ opt_for_reorder(ggml_backend_sycl_context * ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    إذا (!should_reorder_tensor(*ctx, dst)) {
        أرجع;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    إذا (!extra || extra->optimized_feature.reorder) {
        أرجع;  // Skip permutations and already reordered tensors
    }

    بدل (mm_algorithm) {
        حالة mul_mat_algo::DMMV:
            إذا (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                أرجع;
            }
            اكسر;
        حالة mul_mat_algo::MMVQ:
            إذا (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                أرجع;
            }
            اكسر;
        حالة mul_mat_algo::MUL_MAT_SYCL:
            إذا (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                أرجع;
            }
            اكسر;
    }

    إذا (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = سچ;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
ساكن فراغ opt_for_reorder_id(ggml_backend_sycl_context * ctx, ثابت ggml_tensor * src0) {
    إذا (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        أرجع;
    }
    إذا (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        أرجع;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    إذا (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->معلومات % 16 != 0)) {
        أرجع;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    إذا (!extra || extra->optimized_feature.reorder) {
        أرجع;
    }
    إذا (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = سچ;
    }
}


ساكن منطقي can_use_dequantize_mul_mat_vec(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ثابت int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    أرجع ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

ساكن منطقي can_use_mul_mat_vec_q(ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    أرجع ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_سائيز;
}

ساكن فراغ ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    إذا (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        أرجع;
    }

    ثابت منطقي ورهايو = ggml_backend_بفر_is_sycl_ورهايو(src0->بفر);
    int64_t min_compute_capability = وڌ_کان_وڌ_عدد;

    إذا (ورهايو) {
        ggml_backend_sycl_ورهايو_بفر_type_context * buft_ctx =
            (ggml_backend_sycl_ورهايو_بفر_type_context *) src0->بفر->buft->context;
        خودڪار & tensor_ورهايو = buft_ctx->tensor_ورهايو;
        لكل (صحيح id = 0; id < ggml_sycl_info().device_ڳڻپ; ++id) {
            // skip devices that are not going to do any work:
            إذا (tensor_ورهايو[id] >= (id + 1 < ggml_sycl_info().device_ڳڻپ ? tensor_ورهايو[id + 1] : 1.0f)) {
                استمر;
            }

            إذا (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } وإلا {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    منطقي use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#جيڪڏهن_وصف_ٿيل GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#جيڪڏهن_پورو

    منطقي use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    منطقي use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#جيڪڏهن_وصف_ٿيل SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_سائيز);
#جيڪڏهن_پورو // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    إذا (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        منطقي use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        إذا (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    إذا (!ورهايو && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        إذا (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } وإلا {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } وإلا إذا (!ورهايو && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } وإلا إذا (!ورهايو && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } وإلا إذا (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } وإلا إذا (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        إذا (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } وإلا {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } وإلا إذا (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } وإلا {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
ساكن منطقي ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * مٿي, ثابت ggml_tensor * wu,
                                             ثابت ggml_tensor * wg, ثابت ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ثابت خودڪار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودڪار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    إذا ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        أرجع خطأ;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ثابت خودڪار plain_type = [](ggml_type t) { أرجع t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    إذا (!plain_type(wu->type) || !plain_type(wg->type)) {
        أرجع خطأ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, مٿي, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ثابت int64_t ne00 = wu->ne[0];
    ثابت int64_t ne11 = act->ne[1];

    ثابت قطار_ptr stream = ctx.stream();
    ثابت صحيح src1_padded_کانن_ڳڻپ = GGML_PAD((صحيح) ne00, ميٽرڪس_صف_PADDING);

    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_کانن_ڳڻپ * حجم(بلاڪ_q8_1) / QK8_1);
    حرف * src1_ddq = src1_q8_alloc.get();

    quantize_صف_q8_1_sycl<quantize_q8_1>((ثابت عائم *) act->معلومات, src1_ddq, (صحيح) ne00, (صحيح) ne11,
                                          src1_padded_کانن_ڳڻپ, stream);

    أرجع ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->معلومات, wu->معلومات,
                                             src1_ddq, (عائم *) glu->معلومات, (صحيح) ne00, (صحيح) wu->ne[1],
                                             (صحيح) ne11,
                                             /*stride_col_y=*/src1_padded_کانن_ڳڻپ / QK8_1,
                                             /*stride_col_dst=*/(صحيح) glu->ne[0], stream);
    GGML_اڻ_استعمال_ٿيل(gate);
}

#جيڪڏهن_وصف_ٿيل GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
ساكن منطقي ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * مٿي, ثابت ggml_tensor * wu,
                                             ثابت ggml_tensor * wg, ثابت ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ثابت حرف * act_begin = (ثابت حرف *) act->معلومات;
    ثابت حرف * glu_begin = (ثابت حرف *) glu->معلومات;
    إذا (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        أرجع خطأ;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, مٿي, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ثابت خودڪار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودڪار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    إذا (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        أرجع خطأ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, مٿي, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    أرجع ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->معلومات, wg->معلومات,
                                                              (ثابت عائم *) act->معلومات, (عائم *) glu->معلومات,
                                                              (صحيح) wu->ne[0], (صحيح) wu->ne[1], ctx.stream());
}
#جيڪڏهن_پورو

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
ساكن منطقي ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx) {
    إذا (!ggml_sycl_can_fuse(cgraph, ڳنڍ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        أرجع خطأ;
    }

    ggml_tensor *       glu  = cgraph->nodes[ڳنڍ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       مٿي   = glu->src[1];
    ثابت ggml_tensor * wu   = مٿي->src[0];
    ثابت ggml_tensor * wg   = gate->src[0];
    ثابت ggml_tensor * act  = مٿي->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    إذا (ggml_backend_بفر_is_sycl_ورهايو(wu->بفر) || ggml_backend_بفر_is_sycl_ورهايو(wg->بفر)) {
        أرجع خطأ;
    }

#جيڪڏهن_وصف_ٿيل GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    إذا (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, مٿي, wu, wg, act)) {
        أرجع سچ;
    }
#جيڪڏهن_پورو

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    إذا (g_ggml_sycl_prioritize_dmmv) {
        أرجع خطأ;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ثابت منطقي reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    إذا (!reorder_pair) {
        أرجع ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, مٿي, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    إذا (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        أرجع خطأ;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, مٿي, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ثابت خودڪار * extra_u = static_cast<ثابت ggml_tensor_extra_gpu *>(wu->extra);
    ثابت خودڪار * extra_g = static_cast<ثابت ggml_tensor_extra_gpu *>(wg->extra);
    إذا (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        أرجع خطأ;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, مٿي, /*num_src=*/2, " : fused with gate + GLU");

    ثابت int64_t ne00 = wu->ne[0];
    ثابت int64_t ne11 = act->ne[1];

    ثابت قطار_ptr stream           = ctx.stream();
    ثابت صحيح       src1_padded_کانن_ڳڻپ = GGML_PAD((صحيح) ne00, ميٽرڪس_صف_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_کانن_ڳڻپ * حجم(بلاڪ_q8_1) / QK8_1);
    حرف *                     src1_ddq = src1_q8_alloc.get();

    quantize_صف_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ثابت عائم *) act->معلومات, src1_ddq, (صحيح) ne00, (صحيح) ne11,
                                                          src1_padded_کانن_ڳڻپ, stream);

    أرجع ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->معلومات, wg->معلومات, src1_ddq,
                                               (عائم *) glu->معلومات, (صحيح) ne00, (صحيح) wu->ne[1], (صحيح) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_کانن_ڳڻپ * (صحيح) حجم(بلاڪ_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(صحيح) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
ساكن صحيح ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, صحيح ڳنڍ_idx) {
    ثابت ggml_tensor * ڳنڍ = cgraph->nodes[ڳنڍ_idx];
    إذا (ggml_sycl_info().device_ڳڻپ != 1 || ڳنڍ->type != GGML_TYPE_F32 ||
        ڳنڍ->src[0]->type != GGML_TYPE_F32 || ڳنڍ->src[0]->ne[0] >= 1024) {
        أرجع 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    صحيح           count = 0;
    صحيح           آخري  = ڳنڍ_idx;
    عائم         eps0;
    يادگيري_نقل(&eps0, ڳنڍ->op_params, حجم(عائم));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    خودڪار overlaps = [](ثابت ggml_tensor * a, ثابت ggml_tensor * b) {
        ثابت حرف * ab = (ثابت حرف *) a->معلومات;
        ثابت حرف * bb = (ثابت حرف *) b->معلومات;
        أرجع ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    لكل (صحيح j = ڳنڍ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        إذا (ggml_is_خالي_جايون(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_ترتيب_بدلايو || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;  // not a launch; cannot break a run of adjacent norms
        }
        إذا (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ڳنڍ) ||
            !ggml_are_same_shape(nj->src[0], ڳنڍ->src[0])) {
            اكسر;  // any other launch ends the run
        }
        منطقي same_nb = سچ;
        لكل (صحيح d = 0; d < GGML_MAX_DIMS; ++d) {
            إذا (nj->nb[d] != ڳنڍ->nb[d] || nj->src[0]->nb[d] != ڳنڍ->src[0]->nb[d]) {
                same_nb = خطأ;
                اكسر;
            }
        }
        إذا (!same_nb) {
            اكسر;  // one nb[] stride set is shared by the whole batch
        }
        عائم epsj;
        يادگيري_نقل(&epsj, nj->op_params, حجم(عائم));
        إذا (epsj != eps0) {
            اكسر;  // eps mismatch ends the run
        }
        منطقي indep = سچ;
        لكل (صحيح k = 0; k < count; ++k) {
            إذا (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = خطأ;
                اكسر;
            }
        }
        إذا (!indep) {
            اكسر;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        آخري           = j;
    }
    إذا (count < 2) {
        أرجع 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    أرجع آخري - ڳنڍ_idx;
}


__dpct_inline__ ساكن فراغ k_copy_src1_منزل_contiguous(
    ثابت حرف *__restrict__ src1_original, حرف *__restrict__ src1_contiguous,
    ثابت mmid_صف_mapping *__restrict__ صف_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ثابت sycl::nd_شي<3> &شي_ct1) {
    ثابت int32_t src1_صف = شي_ct1.get_گروهه(2);

    ثابت int32_t iid1 = صف_mapping[src1_صف].i2;
    ثابت int32_t id   = صف_mapping[src1_صف].i1;

    ثابت int64_t i11 = id % ne11;
    ثابت int64_t i12 = iid1;

    ثابت عائم * src1_صف_original = (ثابت عائم *)(src1_original + i11*nb11 + i12*nb12);
    عائم * src1_صف_contiguous = (عائم *)(src1_contiguous + src1_صف*nb11);

#هدايت unroll
    لكل (صحيح i = شي_ct1.get_local_id(2); i < ne10;
         i += شي_ct1.get_local_range(2)) {
        src1_صف_contiguous[i] = src1_صف_original[i];
    }
}

__dpct_inline__ ساكن فراغ k_copy_dst_from_contiguous(
    حرف *__restrict__ dst_original, ثابت حرف *__restrict__ dst_contiguous,
    ثابت mmid_صف_mapping *__restrict__ صف_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ثابت sycl::nd_شي<3> &شي_ct1) {
    int32_t i = شي_ct1.get_گروهه(2);

    ثابت int32_t i1 = صف_mapping[i].i1;
    ثابت int32_t i2 = صف_mapping[i].i2;

    ثابت عائم * dst_صف_contiguous = (ثابت عائم *)(dst_contiguous + i*nb1);
    عائم * dst_صف_original = (عائم *)(dst_original + i1*nb1 + i2*nb2);

#هدايت unroll
    لكل (صحيح j = شي_ct1.get_local_id(2); j < ne0;
         j += شي_ct1.get_local_range(2)) {
        dst_صف_original[j] = dst_صف_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
ساكن منطقي ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
    ثابت ggml_tensor * src1, ثابت ggml_tensor * ids, ggml_tensor * dst)
{
    ثابت int64_t ne10 = src1->ne[0];
    ثابت int64_t ne11 = src1->ne[1];
    ثابت int64_t ne12 = src1->ne[2];
    إذا (ne12 != 1) أرجع خطأ;
    إذا (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) أرجع خطأ;
    إذا (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) أرجع خطأ;
    إذا (!ggml_is_contiguous(src1)) أرجع خطأ;

    ثابت int64_t n_ids_في_گروهه = ids->ne[0];
    إذا (ids->ne[1] != 1) أرجع خطأ;
    إذا (ne11 != 1 && ne11 != n_ids_في_گروهه) أرجع خطأ;

    ثابت قطار_ptr stream           = ctx.stream();
    ثابت صحيح       src1_padded_کانن_ڳڻپ = GGML_PAD((صحيح) ne10, ميٽرڪس_صف_PADDING);
    ثابت صحيح       n_experts_استعمال_ٿيل   = (صحيح) n_ids_في_گروهه;
    ثابت صحيح       nrows            = (صحيح) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ثابت ggml_tensor_extra_gpu * src0_extra =
        static_cast<ثابت ggml_tensor_extra_gpu *>(src0->extra);
    ثابت منطقي use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<حرف> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_کانن_ڳڻپ * حجم(بلاڪ_q8_1) / QK8_1);
    حرف * src1_ddq = src1_q8_alloc.get();
    إذا (use_reorder) {
        quantize_صف_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ثابت عائم *) src1->معلومات, src1_ddq, (صحيح) ne10, (صحيح) ne11,
            src1_padded_کانن_ڳڻپ, stream);
    } وإلا {
        quantize_صف_q8_1_sycl<quantize_q8_1>(
            (ثابت عائم *) src1->معلومات, src1_ddq, (صحيح) ne10, (صحيح) ne11,
            src1_padded_کانن_ڳڻپ, stream);
    }

    ثابت size_t بائيٽون_في_qrow = (size_t) src1_padded_کانن_ڳڻپ * حجم(بلاڪ_q8_1) / QK8_1;
    ثابت size_t src1_صف_stride = (ne11 == 1) ? 0 : بائيٽون_في_qrow;

    إذا (use_reorder) {
        أرجع ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->معلومات, src1_ddq, (ثابت int32_t *) ids->معلومات,
            (عائم *) dst->معلومات, (صحيح) ne10, nrows, n_experts_استعمال_ٿيل,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_صف_stride, stream);
    }
    أرجع ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->معلومات, src1_ddq, (ثابت int32_t *) ids->معلومات,
        (عائم *) dst->معلومات, (صحيح) ne10, nrows, n_experts_استعمال_ٿيل,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_صف_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
ساكن فراغ mmid_counting_ترتيب_ڏيو_صفون(
        ثابت ggml_tensor * ids, ثابت حرف * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_صفون,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_صف_offsets,
        std::vector<mmid_صف_mapping> & routed_صف_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.سونپيو(n_as, 0);
    لكل (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        لكل (int64_t id = 0; id < n_ids; id++) {
            ثابت int32_t صف_id_i = *(ثابت int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(صف_id_i >= 0 && صف_id_i < n_as);
            expert_counts[صف_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_صف_offsets.سونپيو(n_as + 1, 0);
    لكل (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_صف_offsets[i02 + 1] = expert_صف_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_صف_ايندڙ = expert_صف_offsets;
    routed_صف_src.resize(n_routed_صفون);
    لكل (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        لكل (int64_t id = 0; id < n_ids; id++) {
            ثابت int32_t صف_id_i = *(ثابت int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(صف_id_i >= 0 && صف_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ثابت int64_t routed_صف = expert_صف_ايندڙ[صف_id_i]++;
            GGML_ASSERT(routed_صف >= expert_صف_offsets[صف_id_i]);
            GGML_ASSERT(routed_صف < expert_صف_offsets[صف_id_i + 1]);
            routed_صف_src[routed_صف] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

ساكن فراغ ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) حاول {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_بفر_is_sycl_ورهايو(src0->بفر) && "mul_mat_id does not support split buffers");

    ثابت ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ٻٽو_OP_LOCALS

    ثابت قطار_ptr stream = ctx.stream();

    ثابت int64_t n_as = ne02;
    ثابت int64_t n_ids = ids->ne[0];

    إذا (ne12 == 1) {
        إذا (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            أرجع;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    إذا (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<حرف> ids_host(ggml_nbytes(ids));
    ثابت حرف * ids_dev = (ثابت حرف *) ids->معلومات;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->يادگيري_نقل(ids_host.معلومات(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_صف = *src0;
    ggml_tensor src1_صف = *src1;
    ggml_tensor dst_صف = *dst;

    حرف *src0_original = (حرف *)src0->معلومات;
    حرف *src1_original = (حرف *)src1->معلومات;
    حرف *dst_original = (حرف *)dst->معلومات;

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
    إذا (ne12 == 1) {
        لكل (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            لكل (int64_t id = 0; id < n_ids; id++) {
                ثابت int32_t i02 = *(ثابت int32_t *) (ids_host.معلومات() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ثابت int64_t i11 = id % ne11;
                ثابت int64_t i12 = iid1;

                ثابت int64_t i1 = id;
                ثابت int64_t i2 = i12;

            src0_صف.معلومات = src0_original + i02*nb02;
            src1_صف.معلومات = src1_original + i11*nb11 + i12*nb12;
            dst_صف.معلومات = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_صف, &src1_صف, &dst_صف);
            }
        }
    } وإلا {
        ثابت int64_t n_routed_صفون = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<حرف> src1_contiguous(ctx.pool(), حجم(عائم)*n_routed_صفون*ne10);
        ggml_sycl_pool_alloc<حرف>  dst_contiguous(ctx.pool(), حجم(عائم)*n_routed_صفون*ne0);

        src1_صف.معلومات = src1_contiguous.get();
        dst_صف.معلومات  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_صف_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_صف_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_صف_mapping> & routed_صف_src = ctx.mmid_صف_mapping_host;

        mmid_counting_ترتيب_ڏيو_صفون(ids, ids_host.معلومات(), n_ids, n_as, n_routed_صفون,
                                expert_صف_counts, expert_صف_offsets, routed_صف_src);

        ggml_sycl_pool_alloc<mmid_صف_mapping> dev_صف_mapping(ctx.pool(), n_routed_صفون);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->يادگيري_نقل(dev_صف_mapping.get(), routed_صف_src.معلومات(), n_routed_صفون*حجم(mmid_صف_mapping))));

        ثابت بنانشان صحيح max_work_گروهه_سائيز = ggml_sycl_info().max_work_گروهه_sizes[ctx.device];
        أكد(max_work_گروهه_سائيز % (WARP_سائيز * WARP_سائيز) == 0);

        {
            sycl::range<3> بلاڪ_dims(1, 1, std::min((بنانشان صحيح)ne10, max_work_گروهه_سائيز));
            sycl::range<3> grid_dims(1, 1, n_routed_صفون);
            stream->submit([&](sycl::handler &cgh) {
                حرف *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_صف_mapping *__restrict dev_صف_mapping_get =
                    dev_صف_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                    [=](sycl::nd_شي<3> شي_ct1) {
                        k_copy_src1_منزل_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_صف_mapping_get,
                            ne11, ne10, nb11, nb12,
                            شي_ct1);
                    });
            });
        }

        منطقي grouped = خطأ;
        إذا (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == حجم(عائم)*ne10 && nb1 == حجم(عائم)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            إذا (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ثابت منطقي src0_reordered =
                src0->extra && ((ثابت ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ثابت عائم *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (عائم *) dst_contiguous.get(),
                                                     expert_صف_offsets.معلومات(), n_as, ne01, ne10, n_routed_صفون,
                                                     ctx.mmid_tile_وقت_جدول_host, ctx.pool(), stream);
        }

        لكل (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ثابت int64_t انگ_src1_صفون = expert_صف_counts[i02];

            إذا (انگ_src1_صفون == 0) {
                استمر;
            }

            ثابت int64_t expert_صف_وٿي = expert_صف_offsets[i02];

            src0_صف.معلومات = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == حجم(عائم)*ne10);
            GGML_ASSERT(nb1 == حجم(عائم)*ne0);
            src1_صف.معلومات = src1_contiguous.get() + expert_صف_وٿي*nb11;
            src1_صف.ne[1] = انگ_src1_صفون;

            src1_صف.nb[1] = nb11;
            src1_صف.nb[2] = انگ_src1_صفون*nb11;
            src1_صف.nb[3] = انگ_src1_صفون*nb11;

            dst_صف.معلومات = dst_contiguous.get() + expert_صف_وٿي*nb1;
            dst_صف.ne[1] = انگ_src1_صفون;
            dst_صف.nb[1] = nb1;
            dst_صف.nb[2] = انگ_src1_صفون*nb1;
            dst_صف.nb[3] = انگ_src1_صفون*nb1;

            ggml_sycl_mul_mat(ctx, &src0_صف, &src1_صف, &dst_صف);
        }

        {
            sycl::range<3> بلاڪ_dims(1, 1, std::min((بنانشان صحيح)ne0, max_work_گروهه_سائيز));
            sycl::range<3> grid_dims(1, 1, n_routed_صفون);
            stream->submit([&](sycl::handler &cgh) {
                ثابت حرف *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ثابت mmid_صف_mapping *__restrict dev_صف_mapping_get =
                    dev_صف_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * بلاڪ_dims, بلاڪ_dims),
                    [=](sycl::nd_شي<3> شي_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_صف_mapping_get,
                                                   ne0, nb1, nb2, شي_ct1);
                    });
            });
        }
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

ساكن فراغ ggml_sycl_diag_نقاب_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_نقاب_inf(ctx, dst);
}

ساكن فراغ ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

ساكن فراغ ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

ساكن فراغ ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

ساكن فراغ ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

ساكن فراغ ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

ساكن فراغ ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

ساكن فراغ ggml_sycl_جوڙ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_جوڙ(ctx, dst);
}

ساكن فراغ ggml_sycl_جوڙ_صفون(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_جوڙ_صفون(ctx, dst);
}

ساكن فراغ ggml_sycl_سراسري(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_سراسري(ctx, dst);
}

ساكن فراغ ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

ساكن فراغ ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


ساكن فراغ ggml_sycl_سيٽ_main_device(ثابت صحيح main_device) حاول {
    إذا (dpct::get_current_device_id() == static_cast<بنانشان صحيح> (main_device)) {
        أرجع;
    }
    check_allow_gpu_اشاريو(main_device);
    dpct::select_device(main_device);

    إذا (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن منطقي ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, بنية ggml_tensor * dst) حاول {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    إذا (!g_sycl_loaded) أرجع خطأ;

    إذا (dst->src[0] != خالي_اشارو && ggml_backend_بفر_is_sycl_ورهايو(dst->src[0]->بفر)) {
        ggml_sycl_سيٽ_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    بدل (dst->op) {
        حالة GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            اكسر;
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            اكسر;
        حالة GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            اكسر;
        حالة GGML_OP_REPEAT_پٺ:
            ggml_sycl_repeat_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_GET_صفون:
            ggml_sycl_get_صفون(ctx, dst);
            اكسر;
        حالة GGML_OP_GET_صفون_پٺ:
            ggml_sycl_get_صفون_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_سيٽ:
            ggml_sycl_op_سيٽ(ctx, dst);
            اكسر;
        حالة GGML_OP_سيٽ_صفون:
            ggml_sycl_op_سيٽ_صفون(ctx, dst);
            اكسر;
        حالة GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            اكسر;
        حالة GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            اكسر;
        حالة GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            اكسر;
        حالة GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            اكسر;
        حالة GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڙيو:
        حالة GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_جوڙيو(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڙيو_ID:
            ggml_sycl_جوڙيو_id(ctx, dst);
            اكسر;
        حالة GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            اكسر;
        حالة GGML_OP_ڳڻپ_EQUAL:
            ggml_sycl_ڳڻپ_equal(ctx, dst);
            اكسر;
        حالة GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            اكسر;
        حالة GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            اكسر;
        حالة GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            اكسر;
        حالة GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            اكسر;
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(dst)) {
                حالة GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_قدم:
                    ggml_sycl_قدم(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_GELU_تيز:
                    ggml_sycl_gelu_تيز(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_ڦيرو:
                    ggml_sycl_ڦيرو(ctx, dst);
                    اكسر;
                حالة GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    اكسر;
                افتراضي:
                    أرجع خطأ;
            }
            اكسر;
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(dst)) {
                حالة GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    اكسر;
                حالة GGML_GLU_OP_GEGLU_تيز:
                    ggml_sycl_geglu_تيز(ctx, dst);
                    اكسر;
                افتراضي:
                    أرجع خطأ;
            }
            اكسر;
        حالة GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_گروهه_NORM:
            ggml_sycl_گروهه_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            اكسر;
        حالة GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            اكسر;
        حالة GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            اكسر;
        حالة GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            اكسر;
        حالة GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            اكسر;
        حالة GGML_OP_RMS_NORM_پٺ:
            ggml_sycl_rms_norm_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            اكسر;
        حالة GGML_OP_MUL_MAT:
            إذا (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                أرجع خطأ;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            إذا (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                أرجع خطأ;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            اكسر;
        حالة GGML_OP_ٻاهريون_PROD:
            ggml_sycl_op_ٻاهريون_prod(ctx, dst);
            اكسر;
        حالة GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            اكسر;
        حالة GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            اكسر;
        حالة GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            اكسر;
        حالة GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            اكسر;
        حالة GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            اكسر;
        حالة GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            اكسر;
        حالة GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            اكسر;
        حالة GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            اكسر;
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتيب_بدلايو:
        حالة GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            اكسر;
        حالة GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            اكسر;
        حالة GGML_OP_DIAG_نقاب_INF:
            ggml_sycl_diag_نقاب_inf(ctx, dst);
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            اكسر;
        حالة GGML_OP_SOFT_MAX_پٺ:
            ggml_sycl_op_soft_max_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            اكسر;
        حالة GGML_OP_CROSS_ENTROPY_LOSS_پٺ:
            ggml_sycl_cross_entropy_loss_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            اكسر;
        حالة GGML_OP_ROPE_پٺ:
            ggml_sycl_rope_پٺ(ctx, dst);
            اكسر;
        حالة GGML_OP_IM2_کانو:
            ggml_sycl_im2col(ctx, dst);
            اكسر;
        حالة GGML_OP_IM2_کانو_3_D:
            ggml_sycl_im2col_3d(ctx, dst);
            اكسر;
        حالة GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            اكسر;
        حالة GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            اكسر;
        حالة GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڙ:
            ggml_sycl_جوڙ(ctx, dst);
            اكسر;
        حالة GGML_OP_جوڙ_صفون:
            ggml_sycl_جوڙ_صفون(ctx, dst);
            اكسر;
        حالة GGML_OP_سراسري:
            ggml_sycl_سراسري(ctx, dst);
            اكسر;
        حالة GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            اكسر;
        حالة GGML_OP_چوٽي_K:
            ggml_sycl_op_چوٽي_k(ctx, dst);
            اكسر;
        حالة GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            اكسر;
        حالة GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            اكسر;
        حالة GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            اكسر;
        حالة GGML_OP_GATED_لڪيري_ATTN:
            ggml_sycl_op_gated_لڪيري_attn(ctx, dst);
            اكسر;
        حالة GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            اكسر;
        حالة GGML_OP_OPT_قدم_ADAMW:
            ggml_sycl_opt_قدم_adamw(ctx, dst);
            اكسر;
        حالة GGML_OP_OPT_قدم_SGD:
            ggml_sycl_opt_قدم_sgd(ctx, dst);
            اكسر;
        حالة GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            اكسر;
        حالة GGML_OP_SSM_جاچ:
            ggml_sycl_ssm_جاچ(ctx, dst);
            اكسر;
        حالة GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            اكسر;
        حالة GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            اكسر;
        حالة GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            اكسر;
        حالة GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            اكسر;
        حالة GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            اكسر;
        حالة GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            اكسر;
        افتراضي:
            أرجع خطأ;
    }

    أرجع سچ;
} امسك (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

GGML_API فراغ ggml_backend_sycl_get_device_description(صحيح device, حرف *description,
                                      size_t description_سائيز) حاول {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_سائيز, "%s", prop.get_name());
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن منطقي sycl_get_mem_info(صحيح device, size_t * حرر, size_t * ڪل) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    يادگيري_APIType mem_api_type = يادگيري_APIType::يادگيري_API_TYPE_SYCL;

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_نتيجو_SUCCESS ?
        (يادگيري_APIType) g_ggml_sycl_get_mem_api : يادگيري_APIType::يادگيري_API_TYPE_SYCL;
#وإلا
    mem_api_type = يادگيري_APIType::يادگيري_API_TYPE_SYCL;
#جيڪڏهن_پورو
    منطقي res = get_يادگيري_سائيز(dpct::dev_mgr::instance().get_device(device),
        *حرر, *ڪل, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *ڪل, *حرر);
    أرجع res;
}

فراغ ggml_backend_sycl_get_device_يادگيري(صحيح device, size_t * حرر, size_t * ڪل) حاول {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    إذا (!sycl_get_mem_info(device, حرر, ڪل)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *حرر, *ڪل);
} امسك (ثابت sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

ساكن ثابت حرف * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    أرجع sycl_ctx->name.c_str();
}

ساكن فراغ ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ميٽيو sycl_ctx;
    ميٽيو backend;
}

ساكن فراغ ggml_backend_sycl_سيٽ_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ثابت فراغ *معلومات, size_t وٿي,
                                               size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) && "unsupported buffer type");
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->يادگيري_نقل((حرف *)tensor->معلومات + وٿي, معلومات, size)));
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ثابت ggml_tensor *tensor,
                                               فراغ *معلومات, size_t وٿي,
                                               size_t size) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, وٿي);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_بفر_t buf = tensor->view_src ? tensor->view_src->بفر : tensor->بفر;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) && "unsupported buffer type");
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->يادگيري_نقل(
        معلومات, (ثابت حرف *)tensor->معلومات + وٿي, size)));
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن منطقي ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ثابت ggml_tensor *src,
                                               ggml_tensor *dst) حاول {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    منطقي is_cpy_supported                = dst->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device) &&
                            ggml_backend_بفر_is_sycl(src->بفر);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    إذا (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->يادگيري_نقل(
            dst->معلومات, src->معلومات, ggml_nbytes(dst))));
        أرجع سچ;
    }

    أرجع خطأ;
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_synchronize(ggml_backend_t backend) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ثابت قطار_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_اڻ_استعمال_ٿيل(backend);
}
امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن منطقي ggml_sycl_is_view_or_noop(ثابت ggml_tensor * t) {
    أرجع ggml_is_خالي_جايون(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_ترتيب_بدلايو || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
ساكن صحيح ggml_sycl_try_gdn_cache_fusion(ثابت ggml_cgraph * cgraph, صحيح ڳنڍ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    إذا (!g_ggml_sycl_enable_fusion) {
        أرجع 0;
    }

    ثابت ggml_tensor * gdn = cgraph->nodes[ڳنڍ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    إذا (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        أرجع 0;
    }

    ثابت ggml_tensor * src_v     = gdn->src[2];
    ثابت int64_t       S_v       = src_v->ne[0];
    ثابت int64_t       H         = src_v->ne[1];
    ثابت int64_t       n_ٽوڪن  = src_v->ne[2];
    ثابت int64_t       n_seqs    = src_v->ne[3];
    ثابت int64_t       D         = S_v * S_v * H;
    ثابت int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ثابت int64_t       n_written = std::min<int64_t>(n_ٽوڪن, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ثابت size_t پڇ_off = ggml_صف_سائيز(GGML_TYPE_F32, S_v * H * n_ٽوڪن * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ثابت ggml_tensor * cpy  = خالي_اشارو;
    صحيح                 skip = 0;
    لكل (صحيح j = ڳنڍ_idx + 1; j < cgraph->n_nodes && cpy == خالي_اشارو; ++j) {
        ثابت ggml_tensor * n = cgraph->nodes[j];
        إذا (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }
        إذا (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            أرجع 0;
        }
        cpy  = n;
        skip = j - ڳنڍ_idx;
    }
    إذا (cpy == خالي_اشارو) {
        أرجع 0;
    }

    ثابت ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ثابت ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    إذا (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != پڇ_off ||
        !ggml_is_contiguous(src)) {
        أرجع 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ثابت std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    إذا (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->معلومات == خالي_اشارو ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_سائيز(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_صف_سائيز(GGML_TYPE_F32, D)) {
        أرجع 0;
    }

    fused_state_cpy.معلومات        = (عائم *) dst->معلومات; // rollback group 0 (newest)
    fused_state_cpy.جڳهه_stride = K > 1 ? (int64_t) (dst->nb[2] / حجم(عائم)) : 0;
    أرجع skip;
}

ساكن فراغ ggml_backend_sycl_گراف_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_سيٽ_main_device(sycl_ctx->device);

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ڳنڍ = cgraph->nodes[i];
        إذا (ggml_sycl_is_view_or_noop(ڳنڍ)) {
            استمر;
        }
        إذا ((ڳنڍ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            استمر;
        }

        ثابت صحيح nodes_منزل_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        إذا (nodes_منزل_skip != 0) {
            i += nodes_منزل_skip;
            استمر;
        }
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        أكد(ڳنڍ->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device));
        لكل (صحيح j = 0; j < GGML_MAX_SRC; j++) {
            إذا (ڳنڍ->src[j] != خالي_اشارو) {
                أكد(ڳنڍ->src[j]->بفر->buft == ggml_backend_sycl_بفر_type(sycl_ctx->device));
            }
        }
#جيڪڏهن_پورو
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        إذا (ڳنڍ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ثابت صحيح gdn_nodes_منزل_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            إذا (gdn_nodes_منزل_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ڳنڍ, fused_state_cpy);
                i += gdn_nodes_منزل_skip;
                استمر;
            }
        }
        إذا (ڳنڍ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_جوڙيو }, {})) {
            ggml_sycl_op_rms_norm_fused_جوڙيو(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            استمر;
        }
        إذا (ڳنڍ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1]);
            i++;
            استمر;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        إذا (ڳنڍ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1]);
            i++;
            استمر;
        }
        إذا (ڳنڍ->op == GGML_OP_جوڙيو &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_جوڙيو, GGML_OP_جوڙيو }, {})) {
            ggml_sycl_op_جوڙيو_جوڙيو_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1]);
            i++;
            استمر;
        }
        إذا (ڳنڍ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ڳنڍ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1]);
            i++;
            استمر;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        إذا (ڳنڍ->op == GGML_OP_جوڙيو && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_جوڙيو, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_جوڙيو_unary_mul_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            استمر;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        إذا (ڳنڍ->op == GGML_OP_L2_NORM) {
            ثابت صحيح l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            إذا (l2_batch_skip > 0) {
                i += l2_batch_skip;
                استمر;
            }
        }

        إذا (ڳنڍ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_جوڙيو, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ڳنڍ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            استمر;
        }

        إذا (ڳنڍ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ڳنڍ, خالي_اشارو, cgraph->nodes[i + 1]);
            i++;
            استمر;
        }

        إذا (ڳنڍ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            استمر;
        }

        منطقي ok = ggml_sycl_compute_forward(*sycl_ctx, ڳنڍ);
        إذا (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ڳنڍ->name, ggml_op_name(ڳنڍ->op));
        }
        GGML_ASSERT(ok);
    }
}

#جيڪڏهن_وصف_ٿيل GGML_SYCL_گراف
ساكن منطقي check_گراف_compatibility(ggml_cgraph * cgraph) {
    إذا (ggml_sycl_info().device_ڳڻپ > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        أرجع خطأ;
    }

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ثابت ggml_op ڳنڍ_op = cgraph->nodes[i]->op;
        بدل (ڳنڍ_op) {
            افتراضي:
                اكسر;
            حالة GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            حالة GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ڳنڍ_op));
                أرجع خطأ;
            حالة GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                إذا (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ڳنڍ_op));
                    أرجع خطأ;
                }
        }
    }
    أرجع سچ;
}
#جيڪڏهن_پورو

ساكن ggml_حالت ggml_backend_sycl_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    خودڪار * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#جيڪڏهن_وصف_ٿيل GGML_SYCL_گراف
    منطقي use_sycl_گراف = خطأ;
    إذا (g_ggml_sycl_enable_گراف) {
        use_sycl_گراف = check_گراف_compatibility(cgraph);
    }
    إذا (use_sycl_گراف) {
        ثابت منطقي گراف_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_گراف);
        إذا (!گراف_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
            أرجع GGML_حالت_SUCCESS;
        }

        sycl_ex::command_گراف model_sycl_گراف(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_بفر_outlives_گراف{}});

        model_sycl_گراف.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
        model_sycl_گراف.پڇاڙي_recording();

        ثابت منطقي گراف_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_گراف);
        إذا (!sycl_ctx->exec_گراف || !گراف_update_support) {
            خودڪار exec_گراف = گراف_update_support ? model_sycl_گراف.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_گراف.finalize();
            sycl_ctx->exec_گراف = std::make_unique<
                sycl_ex::command_گراف<sycl_ex::گراف_state::executable>>(exec_گراف);
        } وإلا {
            حاول {
                sycl_ctx->exec_گراف->update(model_sycl_گراف);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } امسك (sycl::exception ثابت & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                خودڪار exec_گراف = model_sycl_گراف.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_گراف = std::make_unique<
                    sycl_ex::command_گراف<sycl_ex::گراف_state::executable>>(exec_گراف);
            }
        }

        sycl_ctx->stream()->ext_oneapi_گراف(*(sycl_ctx->exec_گراف));
    } وإلا
#جيڪڏهن_پورو
    {
        ggml_backend_sycl_گراف_compute_impl(sycl_ctx, cgraph);
    }
    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_sycl_event_رڪارڊ(ggml_backend_t backend, ggml_backend_event_t event)
حاول
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ثابت قطار_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
امسك (sycl::exception ثابت &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

ساكن فراغ ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) حاول {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    إذا (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } وإلا
        GGML_ABORT("fatal error");
} امسك (sycl::exception ثابت& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
    std::ٻاهر_نڪرو(1);
}

ساكن ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_سيٽ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_sycl_گراف_compute,
    /* .event_record            = */ ggml_backend_sycl_event_رڪارڊ,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ عدم,
};

ساكن ggml_guid_t ggml_backend_sycl_guid() {
    ساكن ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    أرجع &guid;
}

منطقي ggml_backend_is_sycl(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

صحيح ggml_backend_sycl_get_device_ڳڻپ(فراغ) {
    أرجع ggml_sycl_info().device_ڳڻپ;
}


// backend device

ساكن ثابت حرف * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع ctx->description.c_str();
}

ساكن فراغ ggml_backend_sycl_device_get_يادگيري(ggml_backend_dev_t dev, size_t * حرر, size_t * ڪل) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    إذا (!sycl_get_mem_info(ctx->device, حرر, ڪل)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *ڪل, *حرر);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *حرر, *ڪل);
}

ساكن تعداد ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ثابت sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    إذا (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        أرجع GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_يادگيري(dev, &props->يادگيري_free, &props->يادگيري_ڪل);

    منطقي host_بفر = getenv("GGML_SYCL_NO_PINNED") == خالي_اشارو;
#جيڪڏهن_وصف_ٿيل GGML_SYCL_NO_PEER_COPY
    منطقي events = خطأ;
#وإلا
    منطقي events = سچ;
#جيڪڏهن_پورو

    props->caps = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ host_بفر,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ events,
        /* .mmap_support          = */ سچ,
    };
}

ساكن ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_اڻ_استعمال_ٿيل(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع ggml_backend_sycl_init(ctx->device);
}

ساكن ggml_backend_بفر_type_t ggml_backend_sycl_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع ggml_backend_sycl_بفر_type(ctx->device);
}

ساكن ggml_backend_بفر_type_t ggml_backend_sycl_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    أرجع ggml_backend_sycl_host_بفر_type_for_device(ctx->device);
}

ساكن ggml_backend_بفر_t ggml_backend_sycl_device_بفر_from_host_ptr(ggml_backend_dev_t dev, فراغ * ptr, size_t size, size_t max_tensor_سائيز) {
    GGML_اڻ_استعمال_ٿيل(dev);
    GGML_اڻ_استعمال_ٿيل(ptr);
    GGML_اڻ_استعمال_ٿيل(size);
    GGML_اڻ_استعمال_ٿيل(max_tensor_سائيز);
    أرجع خالي_اشارو;
}

ساكن منطقي do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    صحيح device = sycl_ctx->device;
    بدل (op->op) {
        حالة GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                إذا (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    أرجع سچ;
                }
                أرجع خطأ;
            }
        حالة GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            أرجع (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_سائيز(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_سائيز(op->src[0]->type));
        حالة GGML_OP_CONV_2D_DW:
        حالة GGML_OP_CONV_TRANSPOSE_2D:
            أرجع سچ;
        حالة GGML_OP_UNARY:
            بدل (ggml_get_unary_op(op)) {
                حالة GGML_UNARY_OP_SGN:
                حالة GGML_UNARY_OP_ABS:
                حالة GGML_UNARY_OP_NEG:
                حالة GGML_UNARY_OP_قدم:
                حالة GGML_UNARY_OP_RELU:
                حالة GGML_UNARY_OP_HARDSIGMOID:
                حالة GGML_UNARY_OP_TANH:
                حالة GGML_UNARY_OP_GELU:
                حالة GGML_UNARY_OP_SILU:
                حالة GGML_UNARY_OP_SIGMOID:
                حالة GGML_UNARY_OP_HARDSWISH:
                حالة GGML_UNARY_OP_GELU_تيز:
                حالة GGML_UNARY_OP_GELU_ERF:
                حالة GGML_UNARY_OP_EXP:
                حالة GGML_UNARY_OP_EXPM1:
                حالة GGML_UNARY_OP_SOFTPLUS:
                حالة GGML_UNARY_OP_ELU:
                حالة GGML_UNARY_OP_XIELU:
                حالة GGML_UNARY_OP_CEIL:
                    أرجع سچ;
                حالة GGML_UNARY_OP_FLOOR:
                حالة GGML_UNARY_OP_ڦيرو:
                حالة GGML_UNARY_OP_TRUNC:
                    أرجع سچ;
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_GLU:
            بدل (ggml_get_glu_op(op)) {
                حالة GGML_GLU_OP_REGLU:
                حالة GGML_GLU_OP_GEGLU:
                حالة GGML_GLU_OP_SWIGLU:
                حالة GGML_GLU_OP_SWIGLU_OAI:
                حالة GGML_GLU_OP_GEGLU_ERF:
                حالة GGML_GLU_OP_GEGLU_تيز:
                حالة GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    أرجع ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                افتراضي:
                    أرجع خطأ;
            }
        حالة GGML_OP_MUL_MAT:
        حالة GGML_OP_MUL_MAT_ID:
            {
                بنية ggml_tensor * a = op->src[0];
                بنية ggml_tensor * b = op->src[1];

                إذا (a->ne[3] != b->ne[3]) {
                    أرجع خطأ;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                إذا (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  أرجع خطأ;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                إذا (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    أرجع خطأ;
                }

                إذا (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    أرجع خطأ;
                }

                أرجع سچ;
            }
        حالة GGML_OP_ٻاهريون_PROD:
            أرجع op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_GET_صفون:
            {
                بدل (op->src[0]->type) {
                    حالة GGML_TYPE_I32:
                    حالة GGML_TYPE_F16:
                    حالة GGML_TYPE_BF16:
                    حالة GGML_TYPE_F32:
                    حالة GGML_TYPE_Q1_0:
                    حالة GGML_TYPE_MXFP4:
                    حالة GGML_TYPE_NVFP4:
                    حالة GGML_TYPE_IQ2_XXS:
                    حالة GGML_TYPE_IQ2_XS:
                    حالة GGML_TYPE_IQ2_S:
                    حالة GGML_TYPE_IQ3_XXS:
                    حالة GGML_TYPE_IQ1_S:
                    حالة GGML_TYPE_IQ1_M:
                    حالة GGML_TYPE_IQ3_S:
                    حالة GGML_TYPE_IQ4_NL:
                    حالة GGML_TYPE_IQ4_XS:
                    حالة GGML_TYPE_Q2_K:
                    حالة GGML_TYPE_Q3_K:
                    حالة GGML_TYPE_Q4_0:
                    حالة GGML_TYPE_Q4_1:
                    حالة GGML_TYPE_Q4_K:
                    حالة GGML_TYPE_Q5_0:
                    حالة GGML_TYPE_Q5_1:
                    حالة GGML_TYPE_Q5_K:
                    حالة GGML_TYPE_Q6_K:
                    حالة GGML_TYPE_Q8_0:
                        أرجع سچ;
                    افتراضي:
                        أرجع خطأ;
                }
            }
        حالة GGML_OP_GET_صفون_پٺ:
            // return true;
             أرجع op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         حالة GGML_OP_سيٽ:
               أرجع (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        حالة GGML_OP_سيٽ_صفون:
            {
                إذا (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    أرجع خطأ;
                }
                خودڪار res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                أرجع res;
            }
        حالة GGML_OP_DSV4_HC_PRE:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        حالة GGML_OP_DSV4_HC_COMB:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        حالة GGML_OP_DSV4_HC_POST:
            أرجع op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == خالي_اشارو || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        حالة GGML_OP_LIGHTNING_INDEXER:
            أرجع op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_سائيز * 8;
        حالة GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                إذا (src0_type == GGML_TYPE_F16) {
                    إذا (src1_type == GGML_TYPE_Q2_K ||
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
                        أرجع خطأ;
                    }
                }

                إذا (src0_type == GGML_TYPE_BF16) {
                    إذا (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        أرجع خطأ;
                    }
                }

                إذا (src0_type == GGML_TYPE_F32) {
                    إذا (src1_type == GGML_TYPE_Q2_K ||
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
                        أرجع خطأ;
                    }
                }

                إذا (src1_type == GGML_TYPE_F32) {
                    إذا (src0_type == GGML_TYPE_Q1_0 ||
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
                        أرجع خطأ;
                    }
                }

                إذا (src0_type == src1_type) {
                    إذا (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        أرجع خطأ;
                    }
                }

                إذا (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    أرجع خطأ;
                }

                أرجع سچ;
            }
        حالة GGML_OP_REPEAT_پٺ:
            {
                ggml_type src0_type = op->src[0]->type;
                أرجع src0_type == GGML_TYPE_F32;
            }
        حالة GGML_OP_CONCAT:
        حالة GGML_OP_DUP:
        حالة GGML_OP_ARGMAX:
        حالة GGML_OP_NONE:
        حالة GGML_OP_RESHAPE:
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتيب_بدلايو:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_جوڙيو:
        حالة GGML_OP_ADD1:
        حالة GGML_OP_جوڙيو_ID:
        حالة GGML_OP_SUB:
        حالة GGML_OP_ڳڻپ_EQUAL:
        حالة GGML_OP_MUL:
        حالة GGML_OP_DIV:
        حالة GGML_OP_REPEAT:
            أرجع سچ;
        حالة GGML_OP_PAD_REFLECT_1D:
            أرجع ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_SQR:
        حالة GGML_OP_SQRT:
        حالة GGML_OP_SIN:
        حالة GGML_OP_COS:
        حالة GGML_OP_CLAMP:
        حالة GGML_OP_LOG:
        حالة GGML_OP_NORM:
        حالة GGML_OP_L2_NORM:
        حالة GGML_OP_گروهه_NORM:
        حالة GGML_OP_RMS_NORM:
            أرجع سچ;
        حالة GGML_OP_RMS_NORM_پٺ:
            أرجع ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_SCALE:
            // the scale kernel is f32 only
            أرجع op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        حالة GGML_OP_CONT:
            أرجع سچ;
        حالة GGML_OP_TRI:
            {
                ثابت ggml_tensor * src0 = op->src[0];
                أرجع src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        حالة GGML_OP_DIAG_نقاب_INF:
            أرجع سچ;
        حالة GGML_OP_SOFT_MAX:
            أرجع سچ;
        حالة GGML_OP_SOFT_MAX_پٺ: {
            عائم max_bias = 0.0f;
            يادگيري_نقل(&max_bias, (ثابت عائم *) op->op_params + 1, حجم(عائم));
            أرجع max_bias == 0.0f;
        }
        حالة GGML_OP_ROPE:
        حالة GGML_OP_ROPE_پٺ:
        حالة GGML_OP_IM2_کانو:
        حالة GGML_OP_IM2_کانو_3_D:
        حالة GGML_OP_UPSCALE:
            أرجع سچ;
        حالة GGML_OP_COL2IM_1D:
            أرجع ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#جيڪڏهن_وصف_ٿيل GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#جيڪڏهن_پورو
                   ) &&
                   op->src[0]->type == op->type;
        حالة GGML_OP_CONV_3D: {
            ثابت int32_t * opts = (ثابت int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ثابت int64_t knl_n_ڪل = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ثابت int64_t patch_ڪل = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            أرجع op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_ڪل > 0 && patch_ڪل > 0;
        }
        حالة GGML_OP_جوڙ:
        حالة GGML_OP_جوڙ_صفون:
        حالة GGML_OP_سراسري:
            أرجع ggml_is_contiguous(op->src[0]);
        حالة GGML_OP_ARGSORT:
            أرجع سچ;
        حالة GGML_OP_چوٽي_K: {
            ثابت ggml_tensor * src0 = op->src[0];
            ثابت صحيح k = op->ne[0];
            أرجع src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        حالة GGML_OP_POOL_2D:
        حالة GGML_OP_POOL_1D:
        حالة GGML_OP_ACC:
            أرجع سچ;
        حالة GGML_OP_PAD:
            إذا (ggml_get_op_params_i32(op, 8) != 0) {
                أرجع خطأ;
            }
            أرجع سچ;
        حالة GGML_OP_LEAKY_RELU:
        حالة GGML_OP_TIMESTEP_EMBEDDING:
        حالة GGML_OP_RWKV_WKV6:
        حالة GGML_OP_RWKV_WKV7:
        حالة GGML_OP_GATED_لڪيري_ATTN:
        حالة GGML_OP_GATED_DELTA_NET:
        حالة GGML_OP_OPT_قدم_ADAMW:
        حالة GGML_OP_OPT_قدم_SGD:
            أرجع سچ;
        حالة GGML_OP_SSM_CONV:
            أرجع op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        حالة GGML_OP_ROLL:
            أرجع op->type == GGML_TYPE_F32;
        حالة GGML_OP_ARANGE:
            أرجع op->type == GGML_TYPE_F32;
        حالة GGML_OP_SSM_جاچ:
            إذا (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                أرجع (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_سائيز == 0;
            } وإلا {
                // TODO Mamba-1 not yet ported to SYCL
                أرجع خطأ;
            }
        حالة GGML_OP_FILL:
        حالة GGML_OP_CUMSUM:
        حالة GGML_OP_DIAG:
        حالة GGML_OP_CROSS_ENTROPY_LOSS:
        حالة GGML_OP_CROSS_ENTROPY_LOSS_پٺ:
            أرجع سچ;
        حالة GGML_OP_SOLVE_TRI:
            أرجع op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        حالة GGML_OP_FLASH_ATTN_EXT:
            أرجع ggml_sycl_flash_attn_ext_supported(device, op);
        افتراضي:
            أرجع خطأ;
    }

    GGML_اڻ_استعمال_ٿيل(dev);
}

ساكن منطقي ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    منطقي res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    أرجع res;
}

ساكن منطقي ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    إذا (buft->iface.get_name != ggml_backend_sycl_بفر_type_get_name) {
        أرجع خطأ;
    }
    ggml_backend_sycl_بفر_type_context * buft_ctx = (ggml_backend_sycl_بفر_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع buft_ctx->device == sycl_ctx->device;
}

ساكن int64_t get_op_batch_سائيز(ثابت ggml_tensor * op) {
    بدل (op->op) {
        حالة GGML_OP_GET_صفون:
            أرجع 0;
        حالة GGML_OP_MUL_MAT:
            أرجع op->ne[1];
        حالة GGML_OP_MUL_MAT_ID:
        حالة GGML_OP_ROPE:
            أرجع op->ne[2];
        افتراضي:
            أرجع ggml_nrows(op);
    }
}

ساكن منطقي ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    أرجع get_op_batch_سائيز(op) >= sycl_ctx->op_offload_min_batch_سائيز;
}

ساكن ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#جيڪڏهن_وصف_ٿيل GGML_SYCL_NO_PEER_COPY
    أرجع خالي_اشارو;
#وإلا
  sycl::event *event_ptr = جديد sycl::event();

  أرجع جديد ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#جيڪڏهن_پورو
}

ساكن فراغ ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) حاول {
  GGML_اڻ_استعمال_ٿيل(dev);
  إذا (event == خالي_اشارو) {
    أرجع;
  }

  إذا (event->context != خالي_اشارو) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    ميٽيو sycl_event;
    event->context = خالي_اشارو;
  }

  ميٽيو event;
} امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}


ساكن فراغ ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) حاول {
  GGML_اڻ_استعمال_ٿيل(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} امسك (sycl::exception ثابت &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::سٽ_پڄاڻي;
  std::ٻاهر_نڪرو(1);
}

ساكن ثابت ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_يادگيري,
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

بنية ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ساكن ثابت حرف * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_اڻ_استعمال_ٿيل(reg);
    أرجع GGML_SYCL_NAME;
}

ساكن size_t ggml_backend_sycl_reg_get_device_ڳڻپ(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    أرجع ctx->devices.size();
}

ساكن ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t اشاريو) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(اشاريو < ctx->devices.size());
    أرجع ctx->devices[اشاريو];
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

بنية ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_ٻاهريون[2] = { خالي_اشارو, خالي_اشارو };
    uint8_t * host_اندريون[2]  = { خالي_اشارو, خالي_اشارو };
    size_t    host_بائيٽون  = 0;

    فراغ free_host() {
        لكل (صحيح i = 0; i < 2; ++i) {
            ثابت sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            إذا (host_ٻاهريون[i]) { sycl::حرر(host_ٻاهريون[i], sctx); host_ٻاهريون[i] = خالي_اشارو; }
            إذا (host_اندريون[i])  { sycl::حرر(host_اندريون[i],  sctx); host_اندريون[i]  = خالي_اشارو; }
        }
        host_بائيٽون = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
ساكن فراغ ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, قطار_ptr q0, قطار_ptr q1,
                                    فراغ * dst0, ثابت فراغ * src1, فراغ * dst1, ثابت فراغ * src0,
                                    size_t nbytes) {
    إذا (comm->host_بائيٽون < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_ٻاهريون[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_اندريون[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_ٻاهريون[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_اندريون[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_ٻاهريون[0] && comm->host_اندريون[0] && comm->host_ٻاهريون[1] && comm->host_اندريون[1]);
        comm->host_بائيٽون = nbytes;
    }

    sycl::event e0 = q0->يادگيري_نقل(comm->host_ٻاهريون[0], src0, nbytes);
    sycl::event e1 = q1->يادگيري_نقل(comm->host_ٻاهريون[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::يادگيري_نقل(comm->host_اندريون[0], comm->host_ٻاهريون[1], nbytes);
    std::يادگيري_نقل(comm->host_اندريون[1], comm->host_ٻاهريون[0], nbytes);
    q0->يادگيري_نقل(dst0, comm->host_اندريون[0], nbytes);
    q1->يادگيري_نقل(dst1, comm->host_اندريون[1], nbytes);
}

فراغ * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) حاول {
    لكل (size_t i = 0; i < n_backends; ++i) {
        إذا (!ggml_backend_is_sycl(backends[i])) {
            أرجع خالي_اشارو;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    إذا (n_backends != 2) {
        أرجع خالي_اشارو;
    }

    خودڪار * ctx = جديد ggml_backend_sycl_comm_context;
    ctx->backends.سونپيو(backends, backends + n_backends);
    خودڪار * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    خودڪار * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    أرجع ctx;
}
امسك (ثابت sycl::exception &) { أرجع خالي_اشارو; }
امسك (...)                     { أرجع خالي_اشارو; }

فراغ ggml_backend_sycl_comm_free(فراغ * comm_ctx_v) {
    خودڪار * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    إذا (comm_ctx == خالي_اشارو) {
        أرجع;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    إذا (comm_ctx->backends.size() == 2) {
        خودڪار * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        خودڪار * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        حاول {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } امسك (...) { /* best effort during shutdown */ }
    }

    ميٽيو comm_ctx;
}

منطقي ggml_backend_sycl_comm_allreduce_tensor(فراغ * comm_ctx_v, بنية ggml_tensor ** tensors) حاول {
    إذا (comm_ctx_v == خالي_اشارو) {
        أرجع خطأ;
    }

    خودڪار * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ثابت size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    إذا (n_backends != 2) {
        أرجع خطأ;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    إذا (tensors[0]->type != tensors[1]->type) {
        أرجع خطأ;
    }
    إذا (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        أرجع خطأ;
    }
    إذا (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        أرجع خطأ;
    }
    إذا (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        أرجع خطأ;
    }

    ثابت int64_t nelem  = ggml_nelements(tensors[0]);
    ثابت size_t  nbytes = ggml_nbytes(tensors[0]);
    إذا (nelem == 0) {
        أرجع سچ;
    }

    خودڪار * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    خودڪار * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    قطار_ptr q0 = ctx0->stream();
    قطار_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    إذا (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    إذا (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->معلومات;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->معلومات;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->معلومات, f16_tmp1, tensors[0]->معلومات, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((عائم) f16_out0[i] + (عائم) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((عائم) f16_out1[i] + (عائم) f16_tmp1[i]);
            });
        });
        أرجع سچ;
    }

    عائم * out0 = (عائم *) tensors[0]->معلومات;
    عائم * out1 = (عائم *) tensors[1]->معلومات;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    ساكن constexpr int64_t BF16_حد = 32768;

    إذا (nelem < BF16_حد) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        عائم * tmp0 = (عائم *) buf0;
        عائم * tmp1 = (عائم *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->معلومات, tmp1, tensors[0]->معلومات, nbytes);

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
        أرجع سچ;
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
        outbox0[i] = (uint16_t) (sycl::ٻٽو_انگ_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::ٻٽو_انگ_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    ثابت size_t bf16_بائيٽون = nelem * حجم(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_بائيٽون);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::ٻٽو_انگ_cast<عائم>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::ٻٽو_انگ_cast<عائم>(((uint32_t) inbox1[i]) << 16);
        });
    });

    أرجع سچ;
}
امسك (ثابت sycl::exception &) { أرجع خطأ; }
امسك (...)                     { أرجع خطأ; }

ساكن فراغ *ggml_backend_sycl_reg_get_proc_پتو(ggml_backend_reg_t reg, ثابت حرف *name) {
    GGML_اڻ_استعمال_ٿيل(reg);

    إذا (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        أرجع (فراغ *)ggml_backend_sycl_ورهايو_بفر_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    إذا (strcmp(name, "ggml_backend_comm_init") == 0) {
        أرجع (فراغ *)ggml_backend_sycl_comm_init;
    }
    إذا (strcmp(name, "ggml_backend_comm_free") == 0) {
        أرجع (فراغ *)ggml_backend_sycl_comm_free;
    }
    إذا (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        أرجع (فراغ *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_اڻ_استعمال_ٿيل(name);
    أرجع خالي_اشارو;
}

ساكن ثابت ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_ڳڻپ,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_پتو,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    ساكن ggml_backend_reg reg;
    ساكن منطقي initialized = خطأ;

    {
        ساكن std::mutex mutex;
        std::lock_guard<std::mutex> تالو(mutex);
        إذا (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = جديد ggml_backend_sycl_reg_context;
            ثابت صحيح min_batch_سائيز = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            لكل (صحيح i = 0; i < ggml_sycl_info().device_ڳڻپ; i++) {
                ggml_backend_sycl_device_context * dev_ctx = جديد ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_سيٽ_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_سائيز = min_batch_سائيز;

                ggml_backend_dev_t dev = جديد ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_نسخو,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = سچ;
    }

    أرجع &reg;
}

ggml_backend_t ggml_backend_sycl_init(صحيح device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_اشاريو(device);

    ggml_backend_sycl_context * ctx = جديد ggml_backend_sycl_context(device);
    إذا (ctx == خالي_اشارو) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        أرجع خالي_اشارو;
    };

    ggml_backend_t sycl_backend = جديد ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    أرجع sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
