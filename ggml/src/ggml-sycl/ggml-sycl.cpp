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

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <array>
#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <atomic>
#અન્તર્ભાવ <cctype>
#અન્તર્ભાવ <cinttypes>
#અન્તર્ભાવ <cstddef>
#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <float.h>
#અન્તર્ભાવ <limits>
#અન્તર્ભાવ <optional>
#અન્તર્ભાવ <stdint.h>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <fstream>
#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <regex>

#અન્તર્ભાવ <sycl/sycl.hpp>
#અન્તર્ભાવ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#અન્તર્ભાવ <level_zero/ze_api.h>
#અન્તર્ભાવ <level_zero/zes_api.h>
#endif
#જો defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#જો SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#અન્તર્ભાવ <sycl/half_type.hpp>

#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "ggml-sycl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "ggml-backend-impl.h"

#અન્તર્ભાવ "ggml-sycl/add-id.hpp"
#અન્તર્ભાવ "ggml-sycl/backend.hpp"
#અન્તર્ભાવ "ggml-sycl/common.hpp"
#અન્તર્ભાવ "ggml-sycl/element_wise.hpp"
#અન્તર્ભાવ "ggml-sycl/fwht.hpp"
#અન્તર્ભાવ "ggml-sycl/fused-gemm.hpp"
#અન્તર્ભાવ "ggml-sycl/gemm.hpp"
#અન્તર્ભાવ "ggml-sycl/getrows.hpp"
#અન્તર્ભાવ "ggml-sycl/mem.hpp"
#અન્તર્ભાવ "ggml-sycl/norm.hpp"
#અન્તર્ભાવ "ggml-sycl/presets.hpp"
#અન્તર્ભાવ "ggml-sycl/quantize.hpp"
#અન્તર્ભાવ "ggml-sycl/repeat_back.hpp"
#અન્તર્ભાવ "ggml-sycl/set_rows.hpp"
#અન્તર્ભાવ "ggml-sycl/set.hpp"
#અન્તર્ભાવ "ggml-sycl/dsv4-hc.hpp"
#અન્તર્ભાવ "ggml-sycl/lightning-indexer.hpp"
#અન્તર્ભાવ "ggml-sycl/conv2d.hpp"
#અન્તર્ભાવ "ggml-sycl/conv2d-dw.hpp"
#અન્તર્ભાવ "ggml-sycl/conv2d-transpose.hpp"
#અન્તર્ભાવ "ggml-sycl/ssm_conv.hpp"
#અન્તર્ભાવ "ggml-sycl/sycl_hw.hpp"
#અન્તર્ભાવ "ggml-sycl/ssm_scan.hpp"
#અન્તર્ભાવ "ggml-sycl/fill.hpp"
#અન્તર્ભાવ "ggml-sycl/cumsum.hpp"
#અન્તર્ભાવ "ggml-sycl/diag.hpp"
#અન્તર્ભાવ "ggml-sycl/opt-step.hpp"
#અન્તર્ભાવ "ggml-sycl/solve_tri.hpp"
#અન્તર્ભાવ "ggml-sycl/gated_delta_net.hpp"
#અન્તર્ભાવ "ggml-sycl/pool.hpp"
#અન્તર્ભાવ "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

સ્થિર તાર્કિક g_sycl_loaded = ખોટું;
પૂર્ણાંક g_ggml_sycl_debug = 0;
પૂર્ણાંક g_ggml_sycl_dev_debug = 0;
પૂર્ણાંક g_ggml_sycl_enable_optimize = 1;
પૂર્ણાંક g_ggml_sycl_enable_graph = 0;
પૂર્ણાંક g_ggml_sycl_enable_dnn = 1;
પૂર્ણાંક g_ggml_sycl_fa_onednn = 1;
પૂર્ણાંક g_ggml_sycl_fa_onednn_max_kv = 0;
પૂર્ણાંક g_ggml_sycl_enable_mkl_fa = 1;
પૂર્ણાંક g_ggml_sycl_memtrace = 0;
પૂર્ણાંક g_ggml_sycl_memtrace_step = 64;
પૂર્ણાંક g_ggml_sycl_enable_vmm = 1;
પૂર્ણાંક g_ggml_sycl_enable_fusion = 1;
પૂર્ણાંક g_ggml_sycl_enable_esimd = 1;
પૂર્ણાંક g_ggml_sycl_mmvq_wide = 1;
પૂર્ણાંક g_ggml_sycl_prioritize_dmmv = 0;
પૂર્ણાંક g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
પૂર્ણાંક g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
પૂર્ણાંક g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
પૂર્ણાંક g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
સ્થિર અચલ અક્ષર * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
સ્થિર પૂર્ણાંક ggml_sycl_get_env_precision(અચલ અક્ષર * name, પૂર્ણાંક def) {
    અચલ અક્ષર * env = getenv(name);
    જો (!env) {
        પાછા def;
    }
    std::string mode(env);
    માટે (અક્ષર & c : mode) {
        c = (અક્ષર) std::toupper((unsigned અક્ષર) c);
    }
    માટે (પૂર્ણાંક i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        જો (mode == ggml_sycl_dynamic_precision_names[i]) {
            પાછા i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    પાછા def;
}
પૂર્ણાંક g_ggml_sycl_use_async_mem_op = 0;
પૂર્ણાંક g_ggml_sycl_use_async_mem_op_requested = 1;
પૂર્ણાંક g_ggml_sycl_use_level_zero_api = 0;
પૂર્ણાંક g_ggml_sycl_enable_flash_attention = 1;
પૂર્ણાંક g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
પૂર્ણાંક g_ggml_sycl_usm_system = 0;
પૂર્ણાંક g_ggml_sycl_enable_host_pinned_mem = 1;
પૂર્ણાંક g_ggml_sycl_host_pinned_mem_2g = 0;
પૂર્ણાંક g_ggml_sycl_upload_staging_slots = 4;
પૂર્ણાંક g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
પૂર્ણાંક g_ggml_sycl_enable_sparse_fa = 0;
પૂર્ણાંક g_ggml_sycl_debug_sparse_fa = 0;
પૂર્ણાંક g_ggml_sycl_sparse_fa_margin = 256;

#જો GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
સ્થિર તાર્કિક ggml_sycl_dnnl_detect_optimized_gemm(પૂર્ણાંક device) {
    using dt = dnnl::memory::data_type;

    પ્રયાસ {
        અચલ dnnl::memory::dims dims    = { 1, 64, 64 };
        અચલ dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        અચલ auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        અચલ auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        અચલ auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        અચલ std::string impl = pd.impl_info_str();
        જો (impl.find("ref") == std::string::npos) {
            પાછા સાચું;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } પકડો (અચલ std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    પાછા ખોટું;
}
#endif

સ્થિર ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    પ્રયાસ {
        info.device_ગણતરી = dpct::dev_mgr::instance().device_ગણતરી();
    } પકડો (sycl::exception અચલ &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_ગણતરી = 0;
    }
    જો (info.device_ગણતરી == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        પાછા info;
    }

    GGML_ASSERT(info.device_ગણતરી <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    માટે (પૂર્ણાંક i = 0; i < info.device_ગણતરી; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#જો !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#નહીંતર
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        જો (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            અચલ size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#endif

        info.default_tensor_split[i] = total_vram;
        total_vram += prop.get_global_mem_size();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#જો GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#endif
        info.devices[i].smpbo = prop.get_local_mem_size();
        info.devices[i].warp_size = WARP_SIZE;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_size();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        જો (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = ખોટું;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        જો (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            જો (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = સાચું;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    માટે (પૂર્ણાંક id = 0; id < info.device_ગણતરી; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    પાછા info;
}

અચલ ggml_sycl_device_info & ggml_sycl_info() {
    સ્થિર ggml_sycl_device_info info = ggml_sycl_init();
    પાછા info;
}

સ્થિર શૂન્ય print_device_detail(પૂર્ણાંક id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string version;
    version += std::to_string(prop.get_major_version());
    version += ".";
    version += std::to_string(prop.get_minor_version());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    auto global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

સ્થિર શૂન્ય print_device_opt_feature(પૂર્ણાંક device_ગણતરી) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    માટે (પૂર્ણાંક id = 0; id < device_ગણતરી; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      પૂર્ણાંક type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
શૂન્ય ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    પૂર્ણાંક device_ગણતરી = dpct::dev_mgr::instance().device_ગણતરી();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_ગણતરી);

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

    માટે (પૂર્ણાંક id = 0; id < device_ગણતરી; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      પૂર્ણાંક type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_ગણતરી);
}

સ્થિર અચલ અક્ષર* dev2dev_int2str(પૂર્ણાંક dev2dev) {
    જો (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        પાછા "SYCL API";
    } નહીંતર જો (dev2dev == DEV2DEV_MEMCPY_L0) {
        પાછા "Level Zero API";
    } નહીંતર જો (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        પાછા "Host Forward";
    } નહીંતર {
        પાછા "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
સ્થિર ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    જો (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (પૂર્ણાંક) res);
    }
    પાછા res;
}

સ્થિર ze_result_t get_zes_init_res() {
    સ્થિર ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (પૂર્ણાંક) zes_init_res);
    પાછા zes_init_res;
}
#endif

સ્થિર શૂન્ય initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

સ્થિર શૂન્ય ggml_check_sycl() પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    સ્થિર તાર્કિક initialized = ખોટું;

    જો (!initialized) {
        initialize_sycl_begining();

        g_ggml_sycl_debug = ggml_sycl_get_env("GGML_SYCL_DEBUG", 0);
        g_ggml_sycl_dev_debug = ggml_sycl_get_env("GGML_SYCL_DEV_DEBUG", 0);
        g_ggml_sycl_enable_optimize = ggml_sycl_get_env("GGML_SYCL_ENABLE_OPT", 1);
        g_ggml_sycl_enable_graph = ggml_sycl_get_env("GGML_SYCL_ENABLE_GRAPH", 0);
        g_ggml_sycl_enable_dnn = ggml_sycl_get_env("GGML_SYCL_ENABLE_DNN", 1);
        g_ggml_sycl_fa_onednn = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN", 1);
        g_ggml_sycl_fa_onednn_max_kv = ggml_sycl_get_env("GGML_SYCL_FA_ONEDNN_MAX_KV", 0);
        g_ggml_sycl_enable_mkl_fa = ggml_sycl_get_env("GGML_SYCL_ENABLE_MKL_FA", 1);
        g_ggml_sycl_memtrace = ggml_sycl_get_env("GGML_SYCL_MEMTRACE", 0);
        g_ggml_sycl_memtrace_step = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
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

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#નહીંતર
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        જો (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#નહીંતર
        g_ggml_sycl_enable_flash_attention = 0;
#endif

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
#જો defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#જો defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#જો defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#જો defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#જો defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#જો defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#જો defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#જો defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#જો defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#નહીંતર
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#endif

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
#જો defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        જો (g_ggml_sycl_use_async_mem_op) {
            માટે (unsigned પૂર્ણાંક i = 0; i < dpct::dev_mgr::instance().device_ગણતરી(); ++i) {
                જો (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    તોડો;
                }
            }
        }
#endif
        જો (CHECK_TRY_ERROR(g_all_sycl_device_ગણતરી =
                            dpct::dev_mgr::instance().device_ગણતરી()) != 0) {
            initialized = સાચું;
            g_sycl_loaded = ખોટું;
            પાછા;
        }
        GGML_ASSERT(g_all_sycl_device_ગણતરી <= GGML_SYCL_MAX_DEVICES);

        initialized = સાચું;
        g_sycl_loaded = સાચું;
        ggml_backend_sycl_print_sycl_devices();
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
અંતઃરેખ શૂન્ય check_allow_gpu_index(અચલ પૂર્ણાંક device_index) {
  જો (device_index >= ggml_sycl_info().device_ગણતરી) {
    અક્ષર error_buf[256];
    snprintf(
        error_buf,
        કદ(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_ગણતરી - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    પ્રતિજ્ઞા(ખોટું);
  }
}

GGML_API શૂન્ય ggml_backend_sycl_get_gpu_list(પૂર્ણાંક *id_list, પૂર્ણાંક max_len) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    માટે(પૂર્ણાંક i=0;i<max_len;i++) id_list[i] = -1;

    માટે (પૂર્ણાંક i=0;i< ggml_sycl_info().device_ગણતરી;i++){
        જો (i>=max_len) તોડો;
        id_list[i] = i;
    }
    પાછા;
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

અંતઃરેખ શૂન્ય free_aligned_mem_host(શૂન્ય * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#નહીંતર
    મુક્ત(memblock);
#endif
}

// sycl buffer

સંરચના ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    સ્થિર constexpr size_t staging_slot_size = 8*1024*1024;

    સંરચના host_staging {
        શૂન્ય * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<તાર્કિક> submitted;
        પૂર્ણાંક next = 0;
    };

    પૂર્ણાંક device;
    શૂન્ય * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    તાર્કિક is_usm_system;

    ggml_backend_sycl_buffer_context(પૂર્ણાંક device, શૂન્ય * dev_ptr, queue_ptr stream, તાર્કિક is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    શૂન્ય drop_host_staging() {
        માટે (size_t i = 0; i < staging.submitted.size(); ++i) {
            જો (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = ખોટું;
            }
        }
        જો (staging.data != nullptr) {
            sycl::મુક્ત(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        જો (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            જો (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            નહીંતર
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        માટે (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

સ્થિર અચલ અક્ષર * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

સ્થિર તાર્કિક ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    પાછા buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

સ્થિર શૂન્ય
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) પ્રયાસ {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    પાછા ctx->dev_ptr;
}

સ્થિર ગણના ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    જો (tensor->view_src != ખાલી) {
        પ્રતિજ્ઞા(tensor->view_src->buffer->buft == buffer->buft);
        પાછા GGML_STATUS_SUCCESS;
    }

    જો (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        પસંદગી (tensor->type) {
            સ્થિતિ GGML_TYPE_Q4_0:
            સ્થિતિ GGML_TYPE_Q8_0:
            સ્થિતિ GGML_TYPE_Q2_K:
            સ્થિતિ GGML_TYPE_Q3_K:
            સ્થિતિ GGML_TYPE_Q4_K:
            સ્થિતિ GGML_TYPE_Q5_K:
            સ્થિતિ GGML_TYPE_Q6_K:
            સ્થિતિ GGML_TYPE_IQ3_XXS:
            સ્થિતિ GGML_TYPE_IQ3_S:
            સ્થિતિ GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = નવું ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                તોડો;
            }
            મૂળભૂત:
                તોડો;
        }
    }

    જો (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        જો (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (અક્ષર *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    પાછા GGML_STATUS_SUCCESS;
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                અચલ શૂન્ય *data, size_t offset,
                                                size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    અચલ પૂર્ણાંક n_slots = g_ggml_sycl_upload_staging_slots;
    જો (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        જો (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, ખોટું);
        }
    }
    જો (ctx->staging.data != nullptr) {
        queue_ptr    stream    = ctx->stream;
        અક્ષર *       dst       = (અક્ષર *) tensor->data + offset;
        અચલ અક્ષર * src       = (અચલ અક્ષર *) data;
        size_t       remaining = size;
        જ્યારે (remaining > 0) {
            અચલ size_t chunk = std::min(remaining, ctx->staging_slot_size);
            અચલ પૂર્ણાંક    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (પૂર્ણાંક) ctx->staging.submitted.size();
            જો (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            શૂન્ય * stage = (અક્ષર *) ctx->staging.data + slot * ctx->staging_slot_size;
            memcpy(stage, src, chunk);
            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
            ctx->staging.submitted[slot] = સાચું;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        પાછા;
    }

    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    અક્ષર * host_buf = (અક્ષર *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((અક્ષર *) tensor->data + offset, host_buf, size).wait()));
    મુક્ત(host_buf);
#નહીંતર
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((અક્ષર *) tensor->data + offset, data, size).wait()));
#endif
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                અચલ ggml_tensor *tensor,
                                                શૂન્ય *data, size_t offset,
                                                size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (અચલ અક્ષર *)tensor->data + offset, size)
            .wait()));
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
સ્થિર તાર્કિક ggml_sycl_is_l0_discrete_gpu(પૂર્ણાંક device) {
    પાછા ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

સ્થિર શૂન્ય memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, શૂન્ય *ptr_dst,
                         અચલ શૂન્ય *ptr_src, size_t size) {
    અક્ષર *host_buf = (અક્ષર *)malloc(size);
    જો (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.memcpy(host_buf, (અચલ અક્ષર *)ptr_src, size).wait();
    q_dst.memcpy((અક્ષર *)ptr_dst, host_buf, size).wait();
    મુક્ત(host_buf);
}

સ્થિર શૂન્ય dev2dev_memcpy(પૂર્ણાંક device_dst, sycl::queue &q_dst, પૂર્ણાંક device_src, sycl::queue &q_src, શૂન્ય *ptr_dst,
                    અચલ શૂન્ય *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    જો (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        અચલ તાર્કિક l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        જો (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            જો (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                જો (r == ZE_RESULT_SUCCESS) {
                    પાછા;
                }
            }
        }
    }
#endif

    જો (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        જો (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            પાછા;
        }
    }

    // Host-staged copy
    જો(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } નહીંતર {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

સ્થિર તાર્કિક
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    અચલ ggml_tensor *src,
                                    ggml_tensor *dst) પ્રયાસ {
    તાર્કિક is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    જો (is_cpy_supported) {
        ggml_backend_sycl_buffer_context * src_ctx = (ggml_backend_sycl_buffer_context *)src->buffer->context;
        ggml_backend_sycl_buffer_context * dst_ctx = (ggml_backend_sycl_buffer_context *)dst->buffer->context;

        ggml_sycl_set_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_set_device(dst_ctx->device);
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

        queue_ptr stream_dst = dst_ctx->stream;
        queue_ptr stream_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#જો 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (અક્ષર *)dst->data, (અચલ અક્ષર *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        પાછા સાચું;
    }
    પાછા ખોટું;
    GGML_UNUSED(buffer);
} પકડો (અચલ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    માટે (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<અક્ષર*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    જો (size == 0) {
        પાછા;  // Nothing to do
    }
    જો (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    શૂન્ય * target_ptr = static_cast<અક્ષર *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

સ્થિર શૂન્ય ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    જો (buffer == nullptr) {
        પાછા;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    જો (ctx != nullptr) {
        માટે (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

સ્થિર અચલ ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
સંરચના ggml_backend_sycl_buffer_type_context {
    પૂર્ણાંક device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

સ્થિર અચલ અક્ષર * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    પાછા ctx->name.c_str();
}

સ્થિર તાર્કિક check_usm_system(પૂર્ણાંક device, size_t size) {
    તાર્કિક use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    જો (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = ખોટું;
    }

    પાછા use_usm_system;
}

અંતઃરેખ શૂન્ય * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    પાછા _aligned_malloc(size, alignment);
#નહીંતર
    પાછા aligned_alloc(alignment, size);
#endif
}

સ્થિર ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    અચલ queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    તાર્કિક use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    શૂન્ય * dev_ptr;
    જો (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (શૂન્ય *)aligned_malloc_host(alignment, aligned_size);
        જો (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            પાછા nullptr;
        }
    } નહીંતર {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (શૂન્ય *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        જો (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          પાછા nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = નવું  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    પાછા ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક is_bmg_g31_arch(પૂર્ણાંક device) {
    પાછા ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

સ્થિર size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    પૂર્ણાંક device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    જો (g_ggml_sycl_host_pinned_mem_2g) {
        પાછા std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } નહીંતર {
        જો(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        પાછા max_alloc_size;

    }
    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    જો (ggml_is_quantized(tensor->type)) {
        જો (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    પાછા size;

    GGML_UNUSED(buft);
}

સ્થિર અચલ ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ખાલી,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ખાલી,
    /* .is_host             = */ ખાલી,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(પૂર્ણાંક device) {
    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_ગણતરી = ggml_backend_sycl_get_device_ગણતરી();

    જો (device>=dev_ગણતરી or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_ગણતરી-1);
        GGML_ASSERT(device<dev_ગણતરી);
    }
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    સ્થિર તાર્કિક ggml_backend_sycl_buffer_type_initialized = ખોટું;

    જો (!ggml_backend_sycl_buffer_type_initialized) {
        માટે (પૂર્ણાંક i = 0; i < dev_ગણતરી; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ નવું ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = સાચું;
    }
    પાછા &ggml_backend_sycl_buffer_types[device];
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    પૂર્ણાંક device = ctx->device;
    જો (device>=ggml_sycl_info().device_ગણતરી or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_ગણતરી-1);
        GGML_ASSERT(device<ggml_sycl_info().device_ગણતરી);
    }
    સ્થિર સંરચના ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    સ્થિર તાર્કિક ggml_backend_sycl_buffer_type_initialized = ખોટું;

    જો (!ggml_backend_sycl_buffer_type_initialized) {
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ નવું ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = સાચું;
    }
    પાછા &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

સ્થિર int64_t get_row_rounding(ggml_type type, અચલ std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        જો (tensor_split[i] < (i + 1 < ggml_sycl_info().device_ગણતરી ? tensor_split[i + 1] : 1.0f)) {
            જો (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            જો (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    પસંદગી(type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
            પાછા max_compute_capability >= VER_GEN9 ? 128 : 64;
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા 64;
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_F32:
            પાછા 1;
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ4_NL:
            પાછા max_compute_capability >= VER_GEN9 ? 128 : 64;
        સ્થિતિ GGML_TYPE_IQ3_S:
            પાછા max_compute_capability >= VER_GEN9 ? 128 : 64;
        સ્થિતિ GGML_TYPE_Q6_K:
            પાછા 64;
        મૂળભૂત:
            GGML_ABORT("fatal error");
    }
}

સ્થિર શૂન્ય get_row_split(int64_t * row_નીચલું, int64_t * row_ઉપલું, અચલ ggml_tensor * tensor, અચલ std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> & tensor_split, પૂર્ણાંક id) {
    અચલ int64_t nrows = ggml_nrows(tensor);
    અચલ int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_નીચલું = id == 0 ? 0 : nrows*tensor_split[id];
    *row_નીચલું -= *row_નીચલું % rounding;
    જો (id == ggml_sycl_info().device_ગણતરી - 1) {
        *row_ઉપલું = nrows;
    } નહીંતર {
        *row_ઉપલું = nrows*tensor_split[id + 1];
        *row_ઉપલું -= *row_ઉપલું % rounding;
    }
}

સ્થિર size_t ggml_nbytes_split(અચલ સંરચના ggml_tensor * tensor, પૂર્ણાંક nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    પાછા nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

સંરચના ggml_backend_sycl_split_buffer_type_context {
    std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> tensor_split;
};

સંરચના ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() પ્રયાસ {
        માટે (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    પકડો (sycl::exception અચલ &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

સ્થિર શૂન્ય ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

સ્થિર શૂન્ય * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    પાછા (શૂન્ય *)0x1000;

    GGML_UNUSED(buffer);
}

સ્થિર ગણના ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    અચલ int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = નવું ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    જો (ctx->streams.empty()) {
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        int64_t row_નીચલું, row_ઉપલું;
        get_row_split(&row_નીચલું, &row_ઉપલું, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ઉપલું - row_નીચલું;
        જો (nrows_split == 0) {
            ચાલુ;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        અચલ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        જો (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        અચલ queue_ptr stream = ctx->streams[i];
        અક્ષર * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (અક્ષર *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        જો (!buf) {
            અક્ષર err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            ફેંકો std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        જો (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .memset(buf + original_size, 0, size - original_size)
                    .wait()));
        }

        extra->data_device[i] = buf;

        માટે (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = નવું sycl::event()));
        }
    }
    tensor->extra = extra;
    પાછા GGML_STATUS_SUCCESS;
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, અચલ શૂન્ય *data,
                                          size_t offset, size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    અચલ int64_t ne0 = tensor->ne[0];
    અચલ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        int64_t row_નીચલું, row_ઉપલું;
        get_row_split(&row_નીચલું, &row_ઉપલું, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ઉપલું - row_નીચલું;
        જો (nrows_split == 0) {
            ચાલુ;
        }

        અચલ size_t offset_split = row_નીચલું*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        અચલ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        જો (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        અચલ અક્ષર * buf_host = (અચલ અક્ષર *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        અચલ queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          અચલ ggml_tensor *tensor, શૂન્ય *data,
                                          size_t offset, size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    અચલ int64_t ne0 = tensor->ne[0];
    અચલ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        int64_t row_નીચલું, row_ઉપલું;
        get_row_split(&row_નીચલું, &row_ઉપલું, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ઉપલું - row_નીચલું;
        જો (nrows_split == 0) {
            ચાલુ;
        }

        અચલ size_t offset_split = row_નીચલું*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        અચલ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        જો (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        અક્ષર * buf_host = (અક્ષર *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        અચલ queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

સ્થિર સંરચના ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ ખાલી,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ ખાલી,
    /* .get_tensor_2d   = */ ખાલી,
    /* .cpy_tensor      = */ ખાલી,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ ખાલી,
};

// sycl split buffer type

સ્થિર અચલ અક્ષર * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    પાછા GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

સ્થિર તાર્કિક ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   પાછા buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

સ્થિર ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = નવું ggml_backend_sycl_split_buffer_context();

    પાછા ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

સ્થિર size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    પાછા SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

સ્થિર size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, અચલ ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    અચલ int64_t ne0 = tensor->ne[0];

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        int64_t row_નીચલું, row_ઉપલું;
        get_row_split(&row_નીચલું, &row_ઉપલું, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_ઉપલું - row_નીચલું;
        જો (nrows_split == 0) {
            ચાલુ;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        જો (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    પાછા total_size;
}

સ્થિર તાર્કિક ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    પાછા ખોટું;

    GGML_UNUSED(buft);
}

સ્થિર ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ખાલી,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ ખાલી, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ખાલી,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(પૂર્ણાંક main_device, અચલ દશાંશ * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    સ્થિર std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    સ્થિર std::map<std::array<દશાંશ, GGML_SYCL_MAX_DEVICES>, સંરચના ggml_backend_buffer_type> buft_map;

    std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    તાર્કિક all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](દશાંશ x) { પાછા x == 0.0f; });
    જો (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } નહીંતર {
        દશાંશ split_sum = 0.0f;
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    જો (it != buft_map.end()) {
        પાછા &it->second;
    }

    સંરચના ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ નવું ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    પાછા &result.first->second;
}

// host buffer type

સંરચના ggml_backend_sycl_device_context {
    પૂર્ણાંક device;
    std::string name;
    std::string description;
    પૂર્ણાંક op_offload_min_batch_size;
};

સ્થિર અચલ અક્ષર * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    પાછા GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

સ્થિર પૂર્ણાંક ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    પાછા static_cast<અચલ ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
સ્થિર શૂન્ય * ggml_backend_sycl_host_malloc(પૂર્ણાંક device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    શૂન્ય * ptr = nullptr;
    પ્રયાસ {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } પકડો (...) {
        ptr = nullptr;
    }
    જો (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    પાછા ptr;
}

સ્થિર શૂન્ય ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    જો (buffer->context == nullptr) {
        પાછા;
    }
    જો (g_ggml_sycl_enable_host_pinned_mem) {
        અચલ પૂર્ણાંક device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(buffer->context, q)));
    } નહીંતર {
        free_aligned_mem_host((શૂન્ય *) buffer->context);
    }
}

સ્થિર ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    શૂન્ય * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    જો (ptr == nullptr) {
        // fallback to cpu buffer
        પાછા ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    પાછા buffer;
}

સ્થિર size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    જો (g_ggml_sycl_enable_host_pinned_mem) {
        અચલ પૂર્ણાંક device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        જો (g_ggml_sycl_host_pinned_mem_2g) {
            પાછા std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } નહીંતર {
            પાછા max_alloc_size;
        }
    } નહીંતર {
        પાછા SIZE_MAX;
    }
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(પૂર્ણાંક device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    સ્થિર std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_ગણતરી());
        માટે (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ ખાલી,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ ખાલી,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        પાછા bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (પૂર્ણાંક) buffer_types_host.size());

    પાછા &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    પાછા ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
સંરચના ggml_sycl_pool_leg : જાહેર ggml_sycl_pool {
    સ્થિર અચલ પૂર્ણાંક MAX_SYCL_BUFFERS = 256;

    પૂર્ણાંક device;
    queue_ptr qptr;
    સંરચના ggml_sycl_buffer {
        શૂન્ય * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, પૂર્ણાંક device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        પૂર્ણાંક    n_cached    = 0;
        size_t bytes_cached = 0;
        માટે (પૂર્ણાંક i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            જો (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        અચલ auto slots = format_slots_in_alloc_order();
        જો (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        માટે (પૂર્ણાંક i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            જો (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() અચલ {
        std::string લીટી;
        અક્ષર buf[32];
        તાર્કિક first = સાચું;
        માટે (પૂર્ણાંક i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            જો (buffer_pool[i].ptr == nullptr) {
                ચાલુ;
            }
            જો (!first) {
                લીટી += '/';
            }
            first = ખોટું;
            snprintf(buf, કદ(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            લીટી += buf;
        }
        પાછા લીટી;
    }
#endif

    શૂન્ય * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        પૂર્ણાંક nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        પૂર્ણાંક ibest = -1;
        માટે (પૂર્ણાંક i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            જો (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                જો (b.size > max_size) max_size = b.size;
#endif
                જો (b.size >= size) {
                    size_t diff = b.size - size;
                    જો (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        જો (!best_diff) {
                            શૂન્ય * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            પાછા ptr;
                        }
                    }
                }
            }
        }
        જો (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            શૂન્ય * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            પાછા ptr;
        }
        શૂન્ય * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (શૂન્ય *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        જો (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            પાછા nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        પાછા ptr;
    }

    શૂન્ય મુક્ત(શૂન્ય * ptr, size_t size) override {
        માટે (પૂર્ણાંક i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            જો (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                પાછા;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#જો defined(GGML_SYCL_SUPPORT_VMM)
સંરચના ggml_sycl_pool_vmm : જાહેર ggml_sycl_pool {
    સ્થિર અચલ size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    પૂર્ણાંક           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    સંરચના mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        શૂન્ય * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, પૂર્ણાંક device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        જો (pool_addr == 0) {
            પાછા;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        માટે (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    શૂન્ય * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        જો (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            જો (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            જો (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            શૂન્ય * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_size, reserve_size,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_size += reserve_size;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_size);

#ifdef DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (unsigned long long) (pool_size/1024/1024),
                          (unsigned long long) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        શૂન્ય * ptr = reinterpret_cast<શૂન્ય *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        પાછા ptr;
    }

    શૂન્ય મુક્ત(શૂન્ય * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<શૂન્ય *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

સંરચના ggml_sycl_pool_host : જાહેર ggml_sycl_pool {
    queue_ptr qptr;
    પૂર્ણાંક       device;

    સંરચના ggml_sycl_buffer {
        શૂન્ય * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    સ્થિર constexpr પૂર્ણાંક          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;
    પૂર્ણાંક                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, પૂર્ણાંક device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        માટે (પૂર્ણાંક i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            જો (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    શૂન્ય * alloc(size_t size, size_t * actual_size) override {
        જો (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            શૂન્ય * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (શૂન્ય *) sycl::malloc_host(size, *qptr)));
            જો (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                પાછા nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            પાછા ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        જો (b.ptr != nullptr) {
            જો (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                શૂન્ય * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                પાછા ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        શૂન્ય * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (શૂન્ય *) sycl::malloc_host(size, *qptr)));
        જો (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            પાછા nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        પાછા ptr;
    }

    શૂન્ય મુક્ત(શૂન્ય * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        માટે (પૂર્ણાંક i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            જો (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                પાછા;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::મુક્ત(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, પૂર્ણાંક device) {
    // return pool for the host to speed up memory management
    પાછા std::unique_ptr<ggml_sycl_pool>(નવું ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, પૂર્ણાંક device) {
#જો defined(GGML_SYCL_SUPPORT_VMM)
    જો (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        પાછા std::unique_ptr<ggml_sycl_pool>(નવું ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    પાછા std::unique_ptr<ggml_sycl_pool>(નવું ggml_sycl_pool_leg(qptr, device));
}

/// kernels
પ્રકારવ્યાખ્યા શૂન્ય (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    અચલ ggml_tensor *src0, અચલ ggml_tensor *src1, ggml_tensor *dst,
    અચલ અક્ષર *src0_dd_i, અચલ દશાંશ *src1_ddf_i, અચલ અક્ષર *src1_ddq_i,
    દશાંશ *dst_dd_i, અચલ int64_t row_નીચલું, અચલ int64_t row_ઉપલું,
    અચલ int64_t src1_ncols, અચલ int64_t src1_padded_row_size,
    અચલ queue_ptr &stream);



સ્થિર શૂન્ય mul_mat_p021_f16_f32(
    અચલ શૂન્ય * __restrict__ vx, અચલ દશાંશ * __restrict__ y, દશાંશ * __restrict__ dst,
    અચલ પૂર્ણાંક ncols_x, અચલ પૂર્ણાંક nrows_x, અચલ પૂર્ણાંક nchannels_x, અચલ પૂર્ણાંક nchannels_y,
    અચલ sycl::nd_item<3> &item_ct1) {

    અચલ sycl::half *x = (અચલ sycl::half *)vx;

    અચલ પૂર્ણાંક row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    અચલ પૂર્ણાંક channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    અચલ પૂર્ણાંક channel_x = channel / (nchannels_y / nchannels_x);

    અચલ પૂર્ણાંક nrows_y = ncols_x;
    અચલ પૂર્ણાંક nrows_dst = nrows_x;
    અચલ પૂર્ણાંક row_dst = row_x;

    દશાંશ tmp = 0.0f;

    માટે (પૂર્ણાંક col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        અચલ પૂર્ણાંક col_x = col_x0 + item_ct1.get_local_id(2);

        જો (col_x >= ncols_x) {
            તોડો;
        }

        // x is transposed and permuted
        અચલ પૂર્ણાંક ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        અચલ દશાંશ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<દશાંશ, sycl::rounding_mode::automatic>()[0];

        અચલ પૂર્ણાંક row_y = col_x;


        // y is not transposed but permuted
        અચલ પૂર્ણાંક iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    અચલ પૂર્ણાંક idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

સ્થિર શૂન્ય mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    અચલ શૂન્ય * __restrict__ vx, અચલ દશાંશ * __restrict__ y, દશાંશ * __restrict__ dst, અચલ પૂર્ણાંક ncols_x, અચલ પૂર્ણાંક nrows_x,
    અચલ પૂર્ણાંક row_stride_x, અચલ પૂર્ણાંક channel_stride_x,અચલ પૂર્ણાંક channel_stride_y, અચલ પૂર્ણાંક channel_x_ભાજક,
    અચલ sycl::nd_item<3> &item_ct1) {

    અચલ sycl::half *x = (અચલ sycl::half *)vx;

    અચલ પૂર્ણાંક row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    અચલ પૂર્ણાંક channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    અચલ પૂર્ણાંક channel_x = channel / channel_x_ભાજક;

    અચલ પૂર્ણાંક nrows_dst = nrows_x;
    અચલ પૂર્ણાંક row_dst   = row_x;

    અચલ પૂર્ણાંક idst = channel*nrows_dst + row_dst;

    દશાંશ tmp = 0.0f;

    માટે (પૂર્ણાંક col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        અચલ પૂર્ણાંક col_x = col_x0 + item_ct1.get_local_id(2);

        જો (col_x >= ncols_x) {
            તોડો;
        }

        અચલ પૂર્ણાંક row_y = col_x;

        અચલ પૂર્ણાંક ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        અચલ પૂર્ણાંક iy = channel * channel_stride_y + row_y;

        અચલ દશાંશ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<દશાંશ, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    માટે (પૂર્ણાંક mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    જો (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

સ્થિર શૂન્ય k_sum_rows_f32(અચલ દશાંશ * x, દશાંશ * dst, અચલ પૂર્ણાંક ncols,
                           અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક row = item_ct1.get_group(1);
    અચલ પૂર્ણાંક col = item_ct1.get_local_id(2);

    દશાંશ sum = 0.0f;
    માટે (પૂર્ણાંક i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    જો (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
સ્થિર અંતઃરેખ શૂન્ય ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ સ્થિર શૂન્ય
k_argsort_f32_i32(અચલ દશાંશ *x, પૂર્ણાંક *dst, અચલ પૂર્ણાંક ncols, પૂર્ણાંક ncols_pad,
                  અચલ પૂર્ણાંક tasks_per_thread, અચલ sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    પૂર્ણાંક col_index =  item_ct1.get_local_id(2);
    પૂર્ણાંક row = item_ct1.get_group(1);

    માટે (પૂર્ણાંક i = 0; i < tasks_per_thread; i++) {
        પૂર્ણાંક col = col_index * tasks_per_thread + i;
        જો (col >= ncols_pad) {
            પાછા;
        }
    }

    અચલ દશાંશ * x_row = x + row * ncols;
    auto dst_row = (પૂર્ણાંક *)dpct_local;

    // initialize indices
    માટે (પૂર્ણાંક i=0;i<tasks_per_thread;i++){
        પૂર્ણાંક col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    માટે (પૂર્ણાંક k = 2; k <= ncols_pad; k *= 2) {
        માટે (પૂર્ણાંક j = k / 2; j > 0; j /= 2) {
            માટે (પૂર્ણાંક i = 0; i < tasks_per_thread; i++) {
                પૂર્ણાંક col = col_index * tasks_per_thread + i;
                પૂર્ણાંક ixj = col ^ j;
                જો (ixj > col) {
                    જો ((col & k) == 0) {
                        જો (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } નહીંતર {
                        જો (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] >
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
                item_ct1.barrier(sycl::access::fence_space::local_space);
            }
        }
    }

    // copy the result to dst without the padding
    માટે (પૂર્ણાંક i = 0; i < tasks_per_thread; i++) {
        પૂર્ણાંક col = col_index * tasks_per_thread + i;
        જો (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

સ્થિર શૂન્ય diag_mask_inf_f32(અચલ દશાંશ * x, દશાંશ * dst, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક rows_per_channel, અચલ પૂર્ણાંક n_past,
                              અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    અચલ પૂર્ણાંક row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    જો (col >= ncols) {
        પાછા;
    }

    અચલ પૂર્ણાંક i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

સ્થિર શૂન્ય scale_f32(અચલ દશાંશ * x, દશાંશ * dst, અચલ દશાંશ scale, અચલ દશાંશ bias, અચલ પૂર્ણાંક k,
                      અચલ sycl::nd_item<3> &item_ct1) {
    અચલ પૂર્ણાંક i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    જો (i >= k) {
        પાછા;
    }

    dst[i] = scale * x[i] + bias;
}


સ્થિર શૂન્ય ggml_mul_mat_p021_f16_f32_sycl(અચલ શૂન્ય *vx, અચલ દશાંશ *y,
                                           દશાંશ *dst, અચલ પૂર્ણાંક ncols_x,
                                           અચલ પૂર્ણાંક nrows_x,
                                           અચલ પૂર્ણાંક nchannels_x,
                                           અચલ પૂર્ણાંક nchannels_y,
                                           queue_ptr stream) {

    અચલ sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    અચલ sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

સ્થિર શૂન્ય ggml_mul_mat_vec_nc_f16_f32_sycl(
    અચલ શૂન્ય *vx, અચલ દશાંશ *y, દશાંશ *dst, અચલ પૂર્ણાંક ncols_x,
    અચલ પૂર્ણાંક nrows_x, અચલ પૂર્ણાંક row_stride_x, અચલ પૂર્ણાંક nchannels_x,
    અચલ પૂર્ણાંક nchannels_y, અચલ પૂર્ણાંક channel_stride_x, અચલ પૂર્ણાંક channel_stride_y, queue_ptr stream) {

    અચલ sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    અચલ sycl::range<3> block_dims(1, 1, WARP_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



સ્થિર શૂન્ય scale_f32_sycl(અચલ દશાંશ *x, દશાંશ *dst, અચલ દશાંશ scale, અચલ દશાંશ bias,
                           અચલ પૂર્ણાંક k, queue_ptr stream) {
    અચલ પૂર્ણાંક અંક_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, અંક_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


સ્થિર શૂન્ય sum_rows_f32_sycl(અચલ દશાંશ *x, દશાંશ *dst, અચલ પૂર્ણાંક ncols,
                              અચલ પૂર્ણાંક nrows, queue_ptr stream) {
    અચલ sycl::range<3> block_dims(1, 1, WARP_SIZE);
    અચલ sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

સ્થિર પૂર્ણાંક next_power_of_2(પૂર્ણાંક x) {
    પૂર્ણાંક n = 1;
    જ્યારે (n < x) {
        n *= 2;
    }
    પાછા n;
}

સ્થિર શૂન્ય init_argsort_indices_padded(
        પૂર્ણાંક * idx,
        અચલ પૂર્ણાંક nrows,
        અચલ પૂર્ણાંક ncols_pad,
        અચલ sycl::nd_item<1> & item_ct1) {
    અચલ size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    અચલ size_t total = (size_t) nrows * (size_t) ncols_pad;

    જો (gid >= total) {
        પાછા;
    }

    idx[gid] = (પૂર્ણાંક) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
સ્થિર શૂન્ય argsort_f32_i32_global_pass(અચલ દશાંશ *            x,
                                        પૂર્ણાંક *                    idx,
                                        અચલ પૂર્ણાંક                ncols,
                                        અચલ પૂર્ણાંક                nrows,
                                        અચલ પૂર્ણાંક                ncols_pad,
                                        અચલ પૂર્ણાંક                j,
                                        અચલ પૂર્ણાંક                k,
                                        અચલ sycl::nd_item<1> & item_ct1) {
    અચલ size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    અચલ size_t total = (size_t) nrows * (size_t) ncols_pad;

    જો (gid >= total) {
        પાછા;
    }

    અચલ પૂર્ણાંક row = (પૂર્ણાંક) (gid / (size_t) ncols_pad);
    અચલ પૂર્ણાંક col = (પૂર્ણાંક) (gid % (size_t) ncols_pad);
    અચલ પૂર્ણાંક ixj = col ^ j;

    જો (ixj <= col || ixj >= ncols_pad) {
        પાછા;
    }

    અચલ size_t base  = (size_t) row * (size_t) ncols_pad;
    અચલ size_t pos_a = base + (size_t) col;
    અચલ size_t pos_b = base + (size_t) ixj;

    અચલ પૂર્ણાંક a = idx[pos_a];
    અચલ પૂર્ણાંક b = idx[pos_b];

    તાર્કિક do_swap = ખોટું;

    જો ((col & k) == 0) {
        જો (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = સાચું;
        }
    } નહીંતર {
        જો (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = સાચું;
        }
    }

    જો (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

સ્થિર શૂન્ય copy_argsort_indices_unpadded(અચલ પૂર્ણાંક *              idx_padded,
                                          પૂર્ણાંક *                    dst,
                                          અચલ પૂર્ણાંક                nrows,
                                          અચલ પૂર્ણાંક                ncols,
                                          અચલ પૂર્ણાંક                ncols_pad,
                                          અચલ sycl::nd_item<1> & item_ct1) {
    અચલ size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    અચલ size_t total = (size_t) nrows * (size_t) ncols;

    જો (gid >= total) {
        પાછા;
    }

    અચલ પૂર્ણાંક row = (પૂર્ણાંક) (gid / (size_t) ncols);
    અચલ પૂર્ણાંક col = (પૂર્ણાંક) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

સ્થિર શૂન્ય argsort_f32_i32_sycl(અચલ દશાંશ *x, પૂર્ણાંક *dst, અચલ પૂર્ણાંક ncols,
                                 અચલ પૂર્ણાંક nrows, ggml_sort_order order,
                                 queue_ptr stream, પૂર્ણાંક device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    અચલ પૂર્ણાંક ncols_pad = next_power_of_2(ncols);
    અચલ size_t shared_mem = (size_t) ncols_pad * કદ(પૂર્ણાંક);
    અચલ size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    જો (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<પૂર્ણાંક> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        પૂર્ણાંક *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        અચલ size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        અચલ size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        માટે (પૂર્ણાંક k = 2; k <= ncols_pad; k *= 2) {
            માટે (પૂર્ણાંક j = k / 2; j > 0; j /= 2) {
                જો (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } નહીંતર જો (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } નહીંતર {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        અચલ size_t total   = (size_t) nrows * (size_t) ncols;
        અચલ size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        પાછા;
    }

    પૂર્ણાંક nth = 1;
    પૂર્ણાંક max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    જ્યારે (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    જો (nth > max_block_size)
        nth = max_block_size;

    અચલ પૂર્ણાંક tasks_per_thread = ncols_pad / nth;

    અચલ sycl::range<3> block_dims(1, 1, nth);
    અચલ sycl::range<3> block_nums(1, nrows, 1);

    જો (order == GGML_SORT_ORDER_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_ASC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } નહીંતર જો (order == GGML_SORT_ORDER_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(block_nums * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_SORT_ORDER_DESC>(
                        x, dst, ncols, ncols_pad, tasks_per_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } નહીંતર {
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
સ્થિર શૂન્ય ટોચ_k_scan_merge_f32(
    અચલ દશાંશ *   src_vals,
    અચલ int32_t * src_map,
    અચલ પૂર્ણાંક       begin,
    અચલ પૂર્ણાંક       end,
    અચલ પૂર્ણાંક       k,
    અચલ પૂર્ણાંક       block_size,
    દશાંશ *         shared_vals,
    પૂર્ણાંક *           shared_idx,
    દશાંશ *         out_vals,
    int32_t *       out_idx,
    અચલ તાર્કિક      swap01,
    અચલ sycl::nd_item<1> & item_ct1
) {
    અચલ પૂર્ણાંક tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    દશાંશ * lv = shared_vals + tid;
    પૂર્ણાંક * li = shared_idx + tid;

    માટે (પૂર્ણાંક i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    દશાંશ kth = -FLT_MAX;

    માટે (પૂર્ણાંક col = begin + tid; col < end; col += block_size) {
        દશાંશ val = src_vals[col];

        જો (val > kth) {
            પૂર્ણાંક pos = k - 1;
            જ્યારે (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            માટે (પૂર્ણાંક i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    જો (tid != 0) {
        પાછા;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    દશાંશ * fv = shared_vals + (size_t) k * block_size;
    પૂર્ણાંક * fi = shared_idx + (size_t) k * block_size;

    માટે (પૂર્ણાંક i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    દશાંશ fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    માટે (પૂર્ણાંક t = 0; t < block_size; t++) {
        માટે (પૂર્ણાંક i = 0; i < k; i++) {
            દશાંશ val = shared_vals[i * block_size + t];

            જો (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                તોડો;
            }

            પૂર્ણાંક idx = shared_idx[i * block_size + t];

            પૂર્ણાંક pos = k - 1;
            જ્યારે (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            માટે (પૂર્ણાંક j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    જો (out_vals) {
        માટે (પૂર્ણાંક i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    માટે (પૂર્ણાંક i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    જો (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

સ્થિર શૂન્ય ટોચ_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    અચલ દશાંશ * src,
    int32_t * dst_indices,
    અચલ int64_t ncols,
    અચલ int64_t nrows,
    અચલ પૂર્ણાંક k,
    dpct::queue_ptr main_stream
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
    constexpr પૂર્ણાંક split_block = 128;
    constexpr પૂર્ણાંક max_splits  = 128;
    constexpr પૂર્ણાંક min_cols    = 8192;

    પૂર્ણાંક nsplit = 1;
    જો (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        અચલ int64_t want = ncols / split_block;
        nsplit = (પૂર્ણાંક) (want > max_splits ? max_splits : want);
    }

    જો (nsplit > 1) {
        અચલ પૂર્ણાંક nchunk = (પૂર્ણાંક) ((ncols + nsplit - 1) / nsplit);
        અચલ size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<દશાંશ>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        દશાંશ *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        અચલ sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<પૂર્ણાંક, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    અચલ પૂર્ણાંક grp = item_ct1.get_group(0);
                    અચલ પૂર્ણાંક row = grp / nsplit;
                    અચલ પૂર્ણાંક part = grp % nsplit;

                    અચલ પૂર્ણાંક begin = part * nchunk;
                    પૂર્ણાંક end = begin + nchunk;
                    જો (end > (પૂર્ણાંક) ncols) {
                        end = (પૂર્ણાંક) ncols;
                    }

                    ટોચ_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, ખોટું, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<દશાંશ, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<પૂર્ણાંક, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    અચલ પૂર્ણાંક row = item_ct1.get_group(0);
                    અચલ size_t off = (size_t) row * nsplit * k;

                    ટોચ_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, સાચું, item_ct1);
                });
        });

        પાછા;
    }

    અચલ પૂર્ણાંક block_size = 128;

    અચલ sycl::range<1> block_dims(block_size);
    અચલ sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<દશાંશ, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<પૂર્ણાંક, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                અચલ પૂર્ણાંક row = item_ct1.get_group(0);

                જો (row >= nrows) પાછા;

                ટોચ_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (પૂર્ણાંક) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, સાચું, item_ct1);
            });
    });
}

સ્થિર શૂન્ય argmax_f32_i32_sycl(અચલ દશાંશ *x, પૂર્ણાંક *dst, અચલ પૂર્ણાંક ncols,
                               અચલ પૂર્ણાંક nrows, queue_ptr stream) {
    અચલ sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    અચલ sycl::range<3> block_nums(1, nrows, 1);
    અચલ size_t shared_mem = 256 * કદ(દશાંશ);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<દશાંશ, 1> shared_data(
            sycl::range<1>(shared_mem/કદ(દશાંશ)), cgh);
        sycl::local_accessor<પૂર્ણાંક, 1> shared_indices(
            sycl::range<1>(shared_mem/કદ(દશાંશ)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                અચલ પૂર્ણાંક tid = item_ct1.get_local_id(2);
                અચલ પૂર્ણાંક row = item_ct1.get_global_id(1);

                દશાંશ max_val = -INFINITY;
                પૂર્ણાંક max_idx = -1;

                માટે (પૂર્ણાંક col = tid; col < ncols; col += 256) {
                    દશાંશ val = x[row * ncols + col];
                    જો (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                માટે (પૂર્ણાંક stride = 256/2; stride > 0; stride >>= 1) {
                    જો (tid < stride) {
                        દશાંશ val1 = shared_data[tid];
                        દશાંશ val2 = shared_data[tid + stride];
                        જો (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                જો (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
સ્થિર શૂન્ય diag_mask_inf_f32_sycl(અચલ દશાંશ *x, દશાંશ *dst,
                                   અચલ પૂર્ણાંક ncols_x, અચલ પૂર્ણાંક nrows_x,
                                   અચલ પૂર્ણાંક rows_per_channel, અચલ પૂર્ણાંક n_past,
                                   queue_ptr stream) {
    અચલ sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    અચલ પૂર્ણાંક block_અંક_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    અચલ sycl::range<3> block_nums(1, block_અંક_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

સ્થિર dpct::err0 ggml_sycl_cpy_tensor_2d(શૂન્ય *dst,
                                          અચલ સંરચના ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_નીચલું, int64_t i1_ઉપલું,
                                          queue_ptr stream) પ્રયાસ {

    dpct::memcpy_direction kind;
    અક્ષર * src_ptr;
    જો (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (અક્ષર *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } નહીંતર જો (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (અક્ષર *) src->data;
    } નહીંતર જો (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_નીચલું == 0 && i1_ઉપલું == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        પૂર્ણાંક id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (અક્ષર *) extra->data_device[id];
    } નહીંતર {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    અક્ષર * dst_ptr = (અક્ષર *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    અચલ ગણના ggml_type type = src->type;
    અચલ int64_t ts = ggml_type_size(type);
    અચલ int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_ઉપલું - i1_નીચલું;

    અચલ અક્ષર * x = src_ptr + i1_નીચલું*nb1 + i2*nb2 + i3*nb3;
    જો (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        પાછા CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } નહીંતર જો (nb0 == ts) {
        પાછા CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } નહીંતર {
        માટે (int64_t i1 = 0; i1 < i1_diff; i1++) {
            અચલ શૂન્ય * rx = (અચલ શૂન્ય *) ((અચલ અક્ષર *) x + i1*nb1);
            શૂન્ય * rd = (શૂન્ય *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, kind, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            જો (r != 0) પાછા r;
        }
        પાછા 0;
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

અંતઃરેખ શૂન્ય ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    અચલ ggml_tensor *src0, અચલ ggml_tensor *src1, ggml_tensor *dst,
    અચલ અક્ષર *src0_dd_i, અચલ દશાંશ *src1_ddf_i, અચલ અક્ષર *src1_ddq_i,
    દશાંશ *dst_dd_i, અચલ int64_t row_નીચલું, અચલ int64_t row_ઉપલું,
    અચલ int64_t src1_ncols, અચલ int64_t src1_padded_row_size,
    અચલ queue_ptr &stream) પ્રયાસ {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    અચલ int64_t row_diff = row_ઉપલું - row_નીચલું;

    પૂર્ણાંક id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    અચલ int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    પૂર્ણાંક ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    તાર્કિક use_fp16 = સાચું;  // TODO(Yu) SYCL capability check
#નહીંતર
    તાર્કિક use_fp16 = ખોટું;
#endif

#જો GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    જો (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        જો (src1->type != GGML_TYPE_BF16) {
            અચલ to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } નહીંતર {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*કદ(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<દશાંશ>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        પાછા;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    તાર્કિક src0_reordered = ખોટું;
    જો (અચલ auto * src0_extra = static_cast<અચલ ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    જો (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        પાછા;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    જો ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        જો (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            અચલ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        અચલ sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (અચલ sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        જો (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            અચલ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        અચલ sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (અચલ sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#જો GGML_SYCL_DNNL
        જો (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<દશાંશ>(), stream);
        }
        નહીંતર
#endif
        {
            અચલ દશાંશ alpha = 1.0f;
            અચલ દશાંશ beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } નહીંતર {
        ggml_sycl_pool_alloc<દશાંશ> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<દશાંશ> src1_ddq_as_f32(ctx.pool());
        જો (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            અચલ to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        જો (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            અચલ to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        અચલ દશાંશ * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (અચલ દશાંશ *) src0_dd_i : src0_ddq_as_f32.get();
        અચલ દશાંશ * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (અચલ દશાંશ *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#જો GGML_SYCL_DNNL
            અચલ int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            અચલ તાર્કિક use_mkl_direct = gemm_flops < 256 * 256 * 256;
            જો (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<દશાંશ>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<દશાંશ>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<દશાંશ>(), stream);
            }
            નહીંતર
#endif
            {
                અચલ દશાંશ alpha = 1.0f;
                અચલ દશાંશ beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_કિંમત(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_કિંમત(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

અંતઃરેખ શૂન્ય ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

અંતઃરેખ શૂન્ય ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int64_t ncols = dst->src[0]->ne[0];
    અચલ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

અંતઃરેખ શૂન્ય ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int64_t ncols = dst->src[0]->ne[0];
    અચલ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


અંતઃરેખ શૂન્ય ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    અચલ int64_t ncols = dst->src[0]->ne[0];
    અચલ int64_t nrows = ggml_nrows(dst->src[0]);

    ગણના ggml_sort_order order = (ગણના ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (પૂર્ણાંક *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

સ્થિર શૂન્ય ggml_sycl_op_ટોચ_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    અચલ પૂર્ણાંક k = dst->ne[0];
    અચલ int64_t ncols = src0->ne[0];
    અચલ int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    જો (k <= SYCL_ટોચ_K_SCAN_MERGE_MAX_K) {
        ટોચ_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } નહીંતર {
        ggml_sycl_ટોચ_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

અંતઃરેખ શૂન્ય ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    અચલ int64_t ncols = dst->src[0]->ne[0];
    અચલ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

અંતઃરેખ શૂન્ય ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ int64_t ne00 = dst->src[0]->ne[0];
    અચલ int64_t ne01 = dst->src[0]->ne[1];
    અચલ પૂર્ણાંક nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    અચલ પૂર્ણાંક n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

સ્થિર શૂન્ય tri_f32_sycl(
    અચલ દશાંશ * src,
    દશાંશ * dst,
    અચલ int64_t ne0,
    અચલ int64_t ne1,
    અચલ int64_t ne2,
    અચલ int64_t ne3,
    અચલ ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    અચલ size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        અચલ int64_t idx = (int64_t) tid[0];

        અચલ int64_t i0 = idx % ne0;
        અચલ int64_t t1 = idx / ne0;
        અચલ int64_t i1 = t1 % ne1;

        તાર્કિક keep = ખોટું;
        પસંદગી (ttype) {
            સ્થિતિ GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); તોડો;
            સ્થિતિ GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); તોડો;
            સ્થિતિ GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); તોડો;
            સ્થિતિ GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); તોડો;
            મૂળભૂત: keep = ખોટું; તોડો;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

સ્થિર શૂન્ય ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(src0->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    અચલ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    અચલ int64_t ne0 = src0->ne[0];
    અચલ int64_t ne1 = src0->ne[1];
    અચલ int64_t ne2 = src0->ne[2];
    અચલ int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


અંતઃરેખ શૂન્ય ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    અચલ દશાંશ * src0_dd = static_cast<અચલ દશાંશ *>(dst->src[0]->data);
    દશાંશ *       dst_dd  = static_cast<દશાંશ *>(dst->data);

    દશાંશ scale;
    દશાંશ bias;
    memcpy(&scale, (દશાંશ *) dst->op_params + 0, કદ(દશાંશ));
    memcpy(&bias,  (દશાંશ *) dst->op_params + 1, કદ(દશાંશ));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

સ્થિર શૂન્ય ggml_sycl_set_peer_access(અચલ પૂર્ણાંક n_tokens, [[maybe_unused]] પૂર્ણાંક main_device) {
    સ્થિર તાર્કિક peer_access_enabled = ખોટું;

    અચલ તાર્કિક enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    જો (peer_access_enabled == enable_peer_access) {
        પાછા;
    }

#ifdef NDEBUG
    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        માટે (પૂર્ણાંક id_other = 0; id_other < ggml_sycl_info().device_ગણતરી; ++id_other) {
            જો (i == id_other) {
                ચાલુ;
            }
            જો (i != main_device && id_other != main_device) {
                ચાલુ;
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
#endif // NDEBUG

    peer_access_enabled = enable_peer_access;
}

template <template <પૂર્ણાંક> typename quantize_f>
સ્થિર શૂન્ય ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, અચલ ggml_tensor *src0,
                                 અચલ ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) પ્રયાસ {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    અચલ int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    અચલ int64_t ne0 = dst->ne[0];
    અચલ int64_t ne1 = dst->ne[1];

    અચલ પૂર્ણાંક nb2 = dst->nb[2];
    અચલ પૂર્ણાંક nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    અચલ int64_t i02_ભાજક = ne12 / ne02;

    અચલ size_t src0_ts = ggml_type_size(src0->type);
    અચલ size_t src0_bs = ggml_blck_size(src0->type);
    અચલ size_t q8_1_ts = કદ(block_q8_1);
    અચલ size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    અચલ તાર્કિક src0_is_contiguous = ggml_is_contiguous(src0);
    અચલ તાર્કિક src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    અચલ તાર્કિક split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<દશાંશ, GGML_SYCL_MAX_DEVICES> tensor_split;
    જો (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    સંરચના dev_data {
        ggml_sycl_pool_alloc<અક્ષર> src0_dd_alloc;
        ggml_sycl_pool_alloc<દશાંશ> src1_ddf_alloc;
        ggml_sycl_pool_alloc<અક્ષર> src1_ddq_alloc;
        ggml_sycl_pool_alloc<દશાંશ> dst_dd_alloc;

        અક્ષર *src0_dd = nullptr;
        દશાંશ *src1_ddf = nullptr; // float
        અક્ષર *src1_ddq = nullptr;  // q8_1
        દશાંશ *dst_dd = nullptr;

        int64_t row_નીચલું;
        int64_t row_ઉપલું;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    પૂર્ણાંક used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        // by default, use all rows
        dev[i].row_નીચલું  = 0;
        dev[i].row_ઉપલું = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        જો (split) {
            અચલ int64_t rounding = get_row_rounding(src0->type, tensor_split);

            જો (i != 0) {
                dev[i].row_નીચલું  = ne01*tensor_split[i];
                જો (dev[i].row_નીચલું < ne01) {
                    dev[i].row_નીચલું -= dev[i].row_નીચલું % rounding;
                }
            }

            જો (i != ggml_sycl_info().device_ગણતરી - 1) {
                dev[i].row_ઉપલું  = ne01*tensor_split[i + 1];
                જો (dev[i].row_ઉપલું < ne01) {
                    dev[i].row_ઉપલું -= dev[i].row_ઉપલું % rounding;
                }
            }
        }
    }

    constexpr તાર્કિક quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
        જો ((!split && i != ctx.device) || dev[i].row_નીચલું == dev[i].row_ઉપલું) {
            ચાલુ;
        }

        used_devices++;

        અચલ તાર્કિક src1_on_device = i == ctx.device;
        અચલ તાર્કિક  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        જો (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (અક્ષર *) src0_extra->data_device[i] : (અક્ષર *) src0->data;
        } નહીંતર {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        જો (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (દશાંશ *) src1->data;
        } નહીંતર {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        જો constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            જો (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                પ્રયાસ {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } પકડો (sycl::exception અચલ &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        જો (dst_on_device) {
            dev[i].dst_dd = (દશાંશ *) dst->data;
        } નહીંતર {
            અચલ size_t size_dst_ddf = split ? (dev[i].row_ઉપલું - dev[i].row_નીચલું)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    જો (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    અચલ int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    માટે (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        અચલ int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        અચલ int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
            જો ((!split && i != ctx.device) || dev[i].row_નીચલું == dev[i].row_ઉપલું) {
                ચાલુ;
            }

            અચલ તાર્કિક src1_on_device = i == ctx.device;
            અચલ તાર્કિક  dst_on_device = i == ctx.device;
            અચલ int64_t row_diff = dev[i].row_ઉપલું - dev[i].row_નીચલું;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            જો (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            માટે (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                અચલ int64_t i03 = i0 / ne12;
                અચલ int64_t i02 = i0 % ne12;

                અચલ size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                અક્ષર  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ભાજક) * (ne01*ne00*src0_ts)/src0_bs;
                દશાંશ * src1_ddf_i = (દશાંશ *) ((અક્ષર *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                અક્ષર  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                દશાંશ *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                જો (i == ctx.device) {
                    dst_dd_i += dev[i].row_નીચલું; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                જો (src1_is_contiguous) {
                    જો (i != ctx.device) {
                        જો constexpr (quantize_enabled) {
                            અક્ષર * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } નહીંતર {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            અચલ અક્ષર * src1_ddf_i_source = (અચલ અક્ષર *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } નહીંતર {
                    જો (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } નહીંતર {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    જો constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        પ્રયાસ {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } પકડો (અચલ sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                જો (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_ભાજક == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ભાજક, dev[i].row_નીચલું, dev[i].row_ઉપલું, stream));
                }
                જો (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_નીચલું, dev[i].row_ઉપલું, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                જો (!dst_on_device) {
                    શૂન્ય * dst_off_device = dst->data;
                    જો (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        દશાંશ * dhf_dst_i = (દશાંશ *) ((અક્ષર *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*કદ(દશાંશ));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_નીચલું;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * કદ(દશાંશ), dst_dd_i,
                            row_diff * કદ(દશાંશ), row_diff * કદ(દશાંશ),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } નહીંતર {
                        દશાંશ * dhf_dst_i = (દશાંશ *) ((અક્ષર *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*કદ(દશાંશ));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * કદ(દશાંશ)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                જો (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    જો (split && ggml_sycl_info().device_ગણતરી > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; ++i) {
            જો (dev[i].row_નીચલું == dev[i].row_ઉપલું) {
                ચાલુ;
            }
            માટે (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, અચલ ggml_tensor *src0,
                                       અચલ ggml_tensor *src1,
                                       ggml_tensor *dst) પ્રયાસ {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];

    અચલ int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    શૂન્ય  * src0_ddq = src0->data;
    દશાંશ * src1_ddf = (દશાંશ *) src1->data;
    દશાંશ * dst_ddf  = (દશાંશ *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, અચલ ggml_tensor *src0,
                                     અચલ ggml_tensor *src1,
                                     ggml_tensor *dst) પ્રયાસ {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];
    અચલ int64_t ne02 = src0->ne[2];

    અચલ int64_t nb01 = src0->nb[1];
    અચલ int64_t nb02 = src0->nb[2];

    અચલ int64_t ne12 = src1->ne[2];
    અચલ int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    શૂન્ય  * src0_ddq = src0->data;
    દશાંશ * src1_ddf = (દશાંશ *) src1->data;
    દશાંશ * dst_ddf  = (દશાંશ *) dst->data;

    અચલ int64_t row_stride_x = nb01 / કદ(sycl::half);
    અચલ int64_t channel_stride_x = nb02 / કદ(sycl::half);
    અચલ int64_t channel_stride_y = nb11 / કદ(દશાંશ);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય k_compute_batched_ptrs(અચલ sycl::half * src0_as_f16, અચલ sycl::half * src1_as_f16, શૂન્ય * dst,
                                   અચલ શૂન્ય ** ptrs_src, શૂન્ય ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, અચલ sycl::nd_item<3> & item_ct1) {
    અચલ int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    અચલ int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    જો (i13 >= ne13 || i12 >= ne12) {
        પાછા;
    }

    અચલ int64_t i03 = i13 / r3;
    અચલ int64_t i02 = i12 / r2;

    અચલ uint8_t * src0_bytes = reinterpret_cast<અચલ uint8_t *>(src0_as_f16);
    અચલ uint8_t * src1_bytes = reinterpret_cast<અચલ uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

સ્થિર શૂન્ય ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src0,
                                           અચલ ggml_tensor * src1, ggml_tensor * dst) પ્રયાસ {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    અચલ sycl::half * src0_f16 = static_cast<અચલ sycl::half *>(src0->data);
    દશાંશ *            dst_ddf  = static_cast<દશાંશ *>(dst->data);

    અચલ sycl::half * src1_f16       = static_cast<અચલ sycl::half *>(src1->data);
#જો GGML_SYCL_DNNL
    અચલ size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    અચલ size_t       type_size_src1 = ggml_type_size(src1->type);

    તાર્કિક is_src0_cont_2 = ggml_is_contiguous_2(src0);
    તાર્કિક is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    જો (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#જો GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        પૂર્ણાંક last_dim=0;
        પૂર્ણાંક last_str=0;
        size_t largest_str=0;
        માટે(પૂર્ણાંક i = 0; i< 4; i++){
            // last stride is always the largest
            જો(src1->nb[i] == largest_str){
                જો(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            જો(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        અચલ int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        અચલ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# નહીંતર
        અચલ int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        અચલ to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = સાચું;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    અચલ દશાંશ alpha_f32 = 1.0f;
    અચલ દશાંશ beta_f32  = 0.0f;

    અચલ શૂન્ય * alpha = &alpha_f32;
    અચલ શૂન્ય * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

#જો GGML_SYCL_DNNL
    જો (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](અચલ sycl::half *src0,
                                                અચલ sycl::half *src1, દશાંશ *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                તાર્કિક supported_broadcast = batchb == batcha ? સાચું
                        : batchb == 1 || batcha == 1        ? સાચું
                                                            : ખોટું;
                જો (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<દશાંશ>(), queue, batcha, batchb);
                } નહીંતર {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    જો (batches0 > batches1) {
                        int64_t અંક_mul_mats = batches1;
                        int64_t sub_batch = batches0 / અંક_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        માટે (int64_t i0 = 0; i0 < અંક_mul_mats; i0++) {
                            અચલ sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            અચલ sycl::half *src1_shifted = src1 + (sb2 * i0);
                            દશાંશ *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<દશાંશ>(),
                                    queue, sub_batch, 1);
                        }
                    } નહીંતર {
                        int64_t અંક_mul_mats = batches0;
                        int64_t sub_batch = batches1 / અંક_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        માટે (int64_t i1 = 0; i1 < અંક_mul_mats; i1++) {
                            અચલ sycl::half *src0_shifted = src0 + (sa2 * i1);
                            અચલ sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            દશાંશ *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<દશાંશ>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            અચલ તાર્કિક cont_batches_dim2_a = nb02 * ne02 == nb03;
            અચલ તાર્કિક cont_batches_dim2_b = nb12 * ne12 == nb13;
            અચલ તાર્કિક cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            અચલ તાર્કિક cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            જો (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / કદ(દશાંશ));
            } નહીંતર જો (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / કદ(દશાંશ));
            } નહીંતર {
                માટે (int64_t b_a = 0; b_a < ne03; b_a++) {
                    અચલ sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    અચલ sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    દશાંશ *dst_shifted = dst_ddf + (nb3 * b_a / કદ(દશાંશ));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / કદ(દશાંશ));
                }
            }

    }
    નહીંતર
#endif
    {
        જો (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            અચલ int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            અચલ int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } નહીંતર {
            અચલ પૂર્ણાંક ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<અચલ શૂન્ય *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<શૂન્ય *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<દશાંશ>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                અચલ શૂન્ય ** ptrs_src_get = ptrs_src.get();
                શૂન્ય **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * કદ(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * કદ(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (અચલ શૂન્ય **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (અચલ શૂન્ય **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (શૂન્ય **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} પકડો (અચલ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ગણના વર્ગ mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

અંતઃરેખ તાર્કિક ggml_sycl_supports_mmq(ગણના ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    પાછા ખોટું;
}

અંતઃરેખ તાર્કિક ggml_sycl_supports_reorder_mul_mat_sycl(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા સાચું;
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
            પાછા !g_ggml_sycl_prioritize_dmmv;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

અંતઃરેખ તાર્કિક ggml_sycl_supports_reorder_dmmv(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

અંતઃરેખ તાર્કિક ggml_sycl_supports_reorder_mmvq(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ3_S:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

સ્થિર તાર્કિક ggml_sycl_supports_reorder_esimd(ગણના ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
#નહીંતર
    GGML_UNUSED(type);
    પાછા ખોટું;
#endif
}

સ્થિર તાર્કિક ggml_sycl_supports_dmmv(ગણના ggml_type type) {
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q1_0:
        સ્થિતિ GGML_TYPE_Q4_0:
        સ્થિતિ GGML_TYPE_Q4_1:
        સ્થિતિ GGML_TYPE_Q5_0:
        સ્થિતિ GGML_TYPE_Q5_1:
        સ્થિતિ GGML_TYPE_Q8_0:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_F16:
        સ્થિતિ GGML_TYPE_BF16:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
સ્થિર અંતઃરેખ શૂન્ય * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    તાર્કિક use_async = g_ggml_sycl_use_async_mem_op;
#જો defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    જો (use_async) {
        શૂન્ય * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        પાછા ptr;
    }
#નહીંતર
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    પાછા ggml_sycl_malloc_device(size, *stream);
}

સ્થિર અંતઃરેખ શૂન્ય sycl_ext_free(dpct::queue_ptr stream, શૂન્ય * ptr) {
    તાર્કિક use_async = g_ggml_sycl_use_async_mem_op;
#જો defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    જો (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        પાછા;
    }
#નહીંતર
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
સંરચના sycl_reorder_temp_buffer {
    શૂન્ય *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        જો (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            જો (ptr) {
                host_fallback = સાચું;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        જો (!ptr) {
            પાછા;
        }
        જો (host_fallback) {
            sycl::મુક્ત(ptr, *stream);
        } નહીંતર {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator તાર્કિક() અચલ { પાછા ptr != nullptr; }

    sycl_reorder_temp_buffer(અચલ sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(અચલ sycl_reorder_temp_buffer &) = delete;

ખાનગી:
    તાર્કિક host_fallback = ખોટું;
};

સ્થિર તાર્કિક reorder_qw_q4_0(uint8_t * data_device, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % કદ(block_q4_0) == 0));
    GGML_ASSERT((offset % કદ(block_q4_0) == 0));
    પૂર્ણાંક offset_blks = offset / કદ(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / કદ(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            અચલ block_q4_0* x = (અચલ block_q4_0*)tmp_buf;
            અચલ પૂર્ણાંક ib = i;

            માટે (પૂર્ણાંક j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q8_0(uint8_t * data_device, અચલ પૂર્ણાંક ncols, અચલ પૂર્ણાંક nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % કદ(block_q8_0) == 0));
    GGML_ASSERT((offset % કદ(block_q8_0) == 0));
    પૂર્ણાંક offset_blks = offset / કદ(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / કદ(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            અચલ block_q8_0* x = (અચલ block_q8_0*)tmp_buf;
            અચલ પૂર્ણાંક ib = i;

            માટે (પૂર્ણાંક j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
સ્થિર તાર્કિક reorder_qw_q8_0_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % કદ(block_q8_0) == 0);
    અચલ પૂર્ણાંક    blocks_per_expert = (પૂર્ણાંક) (expert_bytes / કદ(block_q8_0));
    અચલ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    અચલ પૂર્ણાંક total_blocks = blocks_per_expert * (પૂર્ણાંક) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        અચલ પૂર્ણાંક          gb   = gb_;
        અચલ પૂર્ણાંક          e    = gb / blocks_per_expert;
        અચલ પૂર્ણાંક          ib   = gb % blocks_per_expert;
        અચલ block_q8_0 * x    = (અચલ block_q8_0 *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr = base;
        auto * d_ptr  = (sycl::half *) (qs_ptr + (size_t) blocks_per_expert * QK8_0);

        માટે (પૂર્ણાંક j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_q4_K) == 0);
    GGML_ASSERT(offset % કદ(block_q4_K) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_q4_K * x  = (અચલ block_q4_K *) tmp_buf;
        અચલ પૂર્ણાંક          ib = i;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        માટે (પૂર્ણાંક j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

// Reorder each expert slice into a self-contained SoA layout.
સ્થિર તાર્કિક reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % કદ(block_q4_K) == 0);
    અચલ પૂર્ણાંક    blocks_per_expert = (પૂર્ણાંક) (expert_bytes / કદ(block_q4_K));
    અચલ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    અચલ પૂર્ણાંક total_blocks = blocks_per_expert * (પૂર્ણાંક) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        અચલ પૂર્ણાંક          gb   = gb_;
        અચલ પૂર્ણાંક          e    = gb / blocks_per_expert;
        અચલ પૂર્ણાંક          ib   = gb % blocks_per_expert;
        અચલ block_q4_K * x    = (અચલ block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        માટે (પૂર્ણાંક j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
સ્થિર તાર્કિક reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % કદ(block_q5_K) == 0);
    અચલ પૂર્ણાંક    blocks_per_expert = (પૂર્ણાંક) (expert_bytes / કદ(block_q5_K));
    અચલ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    અચલ પૂર્ણાંક total_blocks = blocks_per_expert * (પૂર્ણાંક) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        અચલ પૂર્ણાંક          gb   = gb_;
        અચલ પૂર્ણાંક          e    = gb / blocks_per_expert;
        અચલ પૂર્ણાંક          ib   = gb % blocks_per_expert;
        અચલ block_q5_K * x    = (અચલ block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        માટે (પૂર્ણાંક j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
સ્થિર તાર્કિક reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % કદ(block_q6_K) == 0);
    અચલ પૂર્ણાંક    blocks_per_expert = (પૂર્ણાંક) (expert_bytes / કદ(block_q6_K));
    અચલ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    અચલ પૂર્ણાંક total_blocks = blocks_per_expert * (પૂર્ણાંક) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        અચલ પૂર્ણાંક          gb   = gb_;
        અચલ પૂર્ણાંક          e    = gb / blocks_per_expert;
        અચલ પૂર્ણાંક          ib   = gb % blocks_per_expert;
        અચલ block_q6_K * x    = (અચલ block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
સ્થિર તાર્કિક reorder_qw_mxfp4_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % કદ(block_mxfp4) == 0);
    અચલ પૂર્ણાંક     blocks_per_expert = (પૂર્ણાંક) (expert_bytes / કદ(block_mxfp4));
    અચલ size_t  max_chunk_bytes   = 32u << 20;
    અચલ int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    માટે (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        અચલ int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        જો (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        અચલ પૂર્ણાંક total_blocks = blocks_per_expert * (પૂર્ણાંક) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            અચલ પૂર્ણાંક           gb   = gb_;
            અચલ પૂર્ણાંક           e    = gb / blocks_per_expert;
            અચલ પૂર્ણાંક           ib   = gb % blocks_per_expert;
            અચલ block_mxfp4 * x    = (અચલ block_mxfp4 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *           base = chunk + (size_t) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) blocks_per_expert;

            માટે (પૂર્ણાંક j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        જો (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_q2_K) == 0);
    GGML_ASSERT(offset % કદ(block_q2_K) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_q2_K * x  = (અચલ block_q2_K *) tmp_buf;
        અચલ પૂર્ણાંક          ib = i;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        માટે (પૂર્ણાંક j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_q3_K) == 0);
    GGML_ASSERT(offset % કદ(block_q3_K) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_q3_K * x  = (અચલ block_q3_K *) tmp_buf;
        અચલ પૂર્ણાંક          ib = i;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        માટે (પૂર્ણાંક j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        માટે (પૂર્ણાંક j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_q5_K) == 0);
    GGML_ASSERT(offset % કદ(block_q5_K) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_q5_K * x  = (અચલ block_q5_K *) tmp_buf;
        અચલ પૂર્ણાંક          ib = i;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        માટે (પૂર્ણાંક j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        માટે (પૂર્ણાંક j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_q6_K) == 0);
    GGML_ASSERT(offset % કદ(block_q6_K) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_q6_K * x  = (અચલ block_q6_K *) tmp_buf;
        અચલ પૂર્ણાંક          ib = i;

        અચલ uint8_t * ql              = x[ib].ql;
        અચલ uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        માટે (પૂર્ણાંક j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_iq3_s(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_iq3_s) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_iq3_s);
    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr       = data_device;
    auto * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    auto * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    auto * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_iq3_s * x = (અચલ block_iq3_s *) tmp_buf;
        અચલ પૂર્ણાંક ib = i;

        માટે (પૂર્ણાંક j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        માટે (પૂર્ણાંક j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (કદ(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        માટે (પૂર્ણાંક j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[કદ(ggml_half) + j] = x[ib].scales[j];
        }
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw_iq3_xxs(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % કદ(block_iq3_xxs) == 0);

    અચલ પૂર્ણાંક nblocks = size / કદ(block_iq3_xxs);
    sycl_reorder_temp_buffer tmp(stream, size);
    જો (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        પાછા ખોટું;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    જો (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr = data_device;
    auto * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        અચલ block_iq3_xxs * x = (અચલ block_iq3_xxs *) tmp_buf;
        અચલ પૂર્ણાંક ib = i;

        માટે (પૂર્ણાંક j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    જો (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    પાછા સાચું;
}

સ્થિર તાર્કિક reorder_qw(અચલ ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    જો (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        પસંદગી (src0->type) {
            સ્થિતિ GGML_TYPE_Q8_0:
                પાછા reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            સ્થિતિ GGML_TYPE_Q4_K:
                પાછા reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            સ્થિતિ GGML_TYPE_Q5_K:
                પાછા reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            સ્થિતિ GGML_TYPE_Q6_K:
                પાછા reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            સ્થિતિ GGML_TYPE_MXFP4:
                પાછા reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            મૂળભૂત:
                પાછા ખોટું;
        }
    }

    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q4_0:
            પાછા reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q2_K:
            પાછા reorder_qw_q2_k(data_device, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q3_K:
            પાછા reorder_qw_q3_k(data_device, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q4_K:
            પાછા reorder_qw_q4_k(data_device, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q5_K:
            પાછા reorder_qw_q5_k(data_device, size, 0, stream);
        સ્થિતિ GGML_TYPE_Q6_K:
            પાછા reorder_qw_q6_k(data_device, size, 0, stream);
        સ્થિતિ GGML_TYPE_IQ3_XXS:
            પાછા reorder_qw_iq3_xxs(data_device, size, stream);
        સ્થિતિ GGML_TYPE_IQ3_S:
            પાછા reorder_qw_iq3_s(data_device, size, stream);
        મૂળભૂત:
            પાછા ખોટું;
    }
}

સ્થિર તાર્કિક should_reorder_tensor(ggml_backend_sycl_context& ctx, અચલ ggml_tensor * dst) {
    પાછા g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

સ્થિર શૂન્ય opt_for_reorder(ggml_backend_sycl_context * ctx, અચલ ggml_tensor * src0, અચલ ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    જો (!should_reorder_tensor(*ctx, dst)) {
        પાછા;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    જો (!extra || extra->optimized_feature.reorder) {
        પાછા;  // Skip permutations and already reordered tensors
    }

    પસંદગી (mm_algorithm) {
        સ્થિતિ mul_mat_algo::DMMV:
            જો (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                પાછા;
            }
            તોડો;
        સ્થિતિ mul_mat_algo::MMVQ:
            જો (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                પાછા;
            }
            તોડો;
        સ્થિતિ mul_mat_algo::MUL_MAT_SYCL:
            જો (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                પાછા;
            }
            તોડો;
    }

    જો (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = સાચું;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
સ્થિર શૂન્ય opt_for_reorder_id(ggml_backend_sycl_context * ctx, અચલ ggml_tensor * src0) {
    જો (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        પાછા;
    }
    જો (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        પાછા;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    જો (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        પાછા;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    જો (!extra || extra->optimized_feature.reorder) {
        પાછા;
    }
    જો (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = સાચું;
    }
}


સ્થિર તાર્કિક can_use_dequantize_mul_mat_vec(અચલ ggml_tensor * src0, અચલ ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    અચલ int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    પાછા ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

સ્થિર તાર્કિક can_use_mul_mat_vec_q(અચલ ggml_tensor * src0, અચલ ggml_tensor * src1, ggml_tensor * dst) {
    પાછા ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

સ્થિર શૂન્ય ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src0, અચલ ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    જો (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        પાછા;
    }

    અચલ તાર્કિક split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    જો (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        માટે (પૂર્ણાંક id = 0; id < ggml_sycl_info().device_ગણતરી; ++id) {
            // skip devices that are not going to do any work:
            જો (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_ગણતરી ? tensor_split[id + 1] : 1.0f)) {
                ચાલુ;
            }

            જો (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } નહીંતર {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    તાર્કિક use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    તાર્કિક use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    તાર્કિક use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#ifdef SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_SIZE);
#endif // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    જો (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        તાર્કિક use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        જો (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    જો (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        જો (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } નહીંતર {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } નહીંતર જો (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } નહીંતર જો (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } નહીંતર જો (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } નહીંતર જો (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        જો (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } નહીંતર {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } નહીંતર જો (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } નહીંતર {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
સ્થિર તાર્કિક ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, અચલ ggml_tensor * wu,
                                             અચલ ggml_tensor * wg, અચલ ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    અચલ auto * extra_u = static_cast<અચલ ggml_tensor_extra_gpu *>(wu->extra);
    અચલ auto * extra_g = static_cast<અચલ ggml_tensor_extra_gpu *>(wg->extra);
    જો ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        પાછા ખોટું;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    અચલ auto plain_type = [](ggml_type t) { પાછા t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    જો (!plain_type(wu->type) || !plain_type(wg->type)) {
        પાછા ખોટું;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    અચલ int64_t ne00 = wu->ne[0];
    અચલ int64_t ne11 = act->ne[1];

    અચલ queue_ptr stream = ctx.stream();
    અચલ પૂર્ણાંક src1_padded_cols = GGML_PAD((પૂર્ણાંક) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<અક્ષર> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * કદ(block_q8_1) / QK8_1);
    અક્ષર * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((અચલ દશાંશ *) act->data, src1_ddq, (પૂર્ણાંક) ne00, (પૂર્ણાંક) ne11,
                                          src1_padded_cols, stream);

    પાછા ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (દશાંશ *) glu->data, (પૂર્ણાંક) ne00, (પૂર્ણાંક) wu->ne[1],
                                             (પૂર્ણાંક) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(પૂર્ણાંક) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
સ્થિર તાર્કિક ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, અચલ ggml_tensor * wu,
                                             અચલ ggml_tensor * wg, અચલ ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    અચલ અક્ષર * act_begin = (અચલ અક્ષર *) act->data;
    અચલ અક્ષર * glu_begin = (અચલ અક્ષર *) glu->data;
    જો (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        પાછા ખોટું;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    અચલ auto * extra_u = static_cast<અચલ ggml_tensor_extra_gpu *>(wu->extra);
    અચલ auto * extra_g = static_cast<અચલ ggml_tensor_extra_gpu *>(wg->extra);
    જો (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        પાછા ખોટું;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    પાછા ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (અચલ દશાંશ *) act->data, (દશાંશ *) glu->data,
                                                              (પૂર્ણાંક) wu->ne[0], (પૂર્ણાંક) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
સ્થિર તાર્કિક ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx) {
    જો (!ggml_sycl_can_fuse(cgraph, ગાંઠ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        પાછા ખોટું;
    }

    ggml_tensor *       glu  = cgraph->nodes[ગાંઠ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    અચલ ggml_tensor * wu   = up->src[0];
    અચલ ggml_tensor * wg   = gate->src[0];
    અચલ ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    જો (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        પાછા ખોટું;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    જો (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        પાછા સાચું;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    જો (g_ggml_sycl_prioritize_dmmv) {
        પાછા ખોટું;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    અચલ તાર્કિક reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    જો (!reorder_pair) {
        પાછા ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    જો (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        પાછા ખોટું;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    અચલ auto * extra_u = static_cast<અચલ ggml_tensor_extra_gpu *>(wu->extra);
    અચલ auto * extra_g = static_cast<અચલ ggml_tensor_extra_gpu *>(wg->extra);
    જો (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        પાછા ખોટું;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    અચલ int64_t ne00 = wu->ne[0];
    અચલ int64_t ne11 = act->ne[1];

    અચલ queue_ptr stream           = ctx.stream();
    અચલ પૂર્ણાંક       src1_padded_cols = GGML_PAD((પૂર્ણાંક) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<અક્ષર> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * કદ(block_q8_1) / QK8_1);
    અક્ષર *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((અચલ દશાંશ *) act->data, src1_ddq, (પૂર્ણાંક) ne00, (પૂર્ણાંક) ne11,
                                                          src1_padded_cols, stream);

    પાછા ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (દશાંશ *) glu->data, (પૂર્ણાંક) ne00, (પૂર્ણાંક) wu->ne[1], (પૂર્ણાંક) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (પૂર્ણાંક) કદ(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(પૂર્ણાંક) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
સ્થિર પૂર્ણાંક ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx) {
    અચલ ggml_tensor * ગાંઠ = cgraph->nodes[ગાંઠ_idx];
    જો (ggml_sycl_info().device_ગણતરી != 1 || ગાંઠ->type != GGML_TYPE_F32 ||
        ગાંઠ->src[0]->type != GGML_TYPE_F32 || ગાંઠ->src[0]->ne[0] >= 1024) {
        પાછા 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    પૂર્ણાંક           count = 0;
    પૂર્ણાંક           last  = ગાંઠ_idx;
    દશાંશ         eps0;
    memcpy(&eps0, ગાંઠ->op_params, કદ(દશાંશ));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](અચલ ggml_tensor * a, અચલ ggml_tensor * b) {
        અચલ અક્ષર * ab = (અચલ અક્ષર *) a->data;
        અચલ અક્ષર * bb = (અચલ અક્ષર *) b->data;
        પાછા ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    માટે (પૂર્ણાંક j = ગાંઠ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        જો (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;  // not a launch; cannot break a run of adjacent norms
        }
        જો (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ગાંઠ) ||
            !ggml_are_same_shape(nj->src[0], ગાંઠ->src[0])) {
            તોડો;  // any other launch ends the run
        }
        તાર્કિક same_nb = સાચું;
        માટે (પૂર્ણાંક d = 0; d < GGML_MAX_DIMS; ++d) {
            જો (nj->nb[d] != ગાંઠ->nb[d] || nj->src[0]->nb[d] != ગાંઠ->src[0]->nb[d]) {
                same_nb = ખોટું;
                તોડો;
            }
        }
        જો (!same_nb) {
            તોડો;  // one nb[] stride set is shared by the whole batch
        }
        દશાંશ epsj;
        memcpy(&epsj, nj->op_params, કદ(દશાંશ));
        જો (epsj != eps0) {
            તોડો;  // eps mismatch ends the run
        }
        તાર્કિક indep = સાચું;
        માટે (પૂર્ણાંક k = 0; k < count; ++k) {
            જો (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = ખોટું;
                તોડો;
            }
        }
        જો (!indep) {
            તોડો;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    જો (count < 2) {
        પાછા 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    પાછા last - ગાંઠ_idx;
}


__dpct_inline__ સ્થિર શૂન્ય k_copy_src1_to_contiguous(
    અચલ અક્ષર *__restrict__ src1_original, અક્ષર *__restrict__ src1_contiguous,
    અચલ mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    અચલ sycl::nd_item<3> &item_ct1) {
    અચલ int32_t src1_row = item_ct1.get_group(2);

    અચલ int32_t iid1 = row_mapping[src1_row].i2;
    અચલ int32_t id   = row_mapping[src1_row].i1;

    અચલ int64_t i11 = id % ne11;
    અચલ int64_t i12 = iid1;

    અચલ દશાંશ * src1_row_original = (અચલ દશાંશ *)(src1_original + i11*nb11 + i12*nb12);
    દશાંશ * src1_row_contiguous = (દશાંશ *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    માટે (પૂર્ણાંક i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ સ્થિર શૂન્ય k_copy_dst_from_contiguous(
    અક્ષર *__restrict__ dst_original, અચલ અક્ષર *__restrict__ dst_contiguous,
    અચલ mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, અચલ sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    અચલ int32_t i1 = row_mapping[i].i1;
    અચલ int32_t i2 = row_mapping[i].i2;

    અચલ દશાંશ * dst_row_contiguous = (અચલ દશાંશ *)(dst_contiguous + i*nb1);
    દશાંશ * dst_row_original = (દશાંશ *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    માટે (પૂર્ણાંક j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
સ્થિર તાર્કિક ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, અચલ ggml_tensor * src0,
    અચલ ggml_tensor * src1, અચલ ggml_tensor * ids, ggml_tensor * dst)
{
    અચલ int64_t ne10 = src1->ne[0];
    અચલ int64_t ne11 = src1->ne[1];
    અચલ int64_t ne12 = src1->ne[2];
    જો (ne12 != 1) પાછા ખોટું;
    જો (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) પાછા ખોટું;
    જો (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) પાછા ખોટું;
    જો (!ggml_is_contiguous(src1)) પાછા ખોટું;

    અચલ int64_t n_ids_per_group = ids->ne[0];
    જો (ids->ne[1] != 1) પાછા ખોટું;
    જો (ne11 != 1 && ne11 != n_ids_per_group) પાછા ખોટું;

    અચલ queue_ptr stream           = ctx.stream();
    અચલ પૂર્ણાંક       src1_padded_cols = GGML_PAD((પૂર્ણાંક) ne10, MATRIX_ROW_PADDING);
    અચલ પૂર્ણાંક       n_experts_used   = (પૂર્ણાંક) n_ids_per_group;
    અચલ પૂર્ણાંક       nrows            = (પૂર્ણાંક) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    અચલ ggml_tensor_extra_gpu * src0_extra =
        static_cast<અચલ ggml_tensor_extra_gpu *>(src0->extra);
    અચલ તાર્કિક use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<અક્ષર> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * કદ(block_q8_1) / QK8_1);
    અક્ષર * src1_ddq = src1_q8_alloc.get();
    જો (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (અચલ દશાંશ *) src1->data, src1_ddq, (પૂર્ણાંક) ne10, (પૂર્ણાંક) ne11,
            src1_padded_cols, stream);
    } નહીંતર {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (અચલ દશાંશ *) src1->data, src1_ddq, (પૂર્ણાંક) ne10, (પૂર્ણાંક) ne11,
            src1_padded_cols, stream);
    }

    અચલ size_t bytes_per_qrow = (size_t) src1_padded_cols * કદ(block_q8_1) / QK8_1;
    અચલ size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    જો (use_reorder) {
        પાછા ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (અચલ int32_t *) ids->data,
            (દશાંશ *) dst->data, (પૂર્ણાંક) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    પાછા ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (અચલ int32_t *) ids->data,
        (દશાંશ *) dst->data, (પૂર્ણાંક) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
સ્થિર શૂન્ય mmid_counting_sort_rows(
        અચલ ggml_tensor * ids, અચલ અક્ષર * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    માટે (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        માટે (int64_t id = 0; id < n_ids; id++) {
            અચલ int32_t row_id_i = *(અચલ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    માટે (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    માટે (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        માટે (int64_t id = 0; id < n_ids; id++) {
            અચલ int32_t row_id_i = *(અચલ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            અચલ int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

સ્થિર શૂન્ય ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) પ્રયાસ {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    અચલ ggml_tensor *src0 = dst->src[0];
    અચલ ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    અચલ ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ queue_ptr stream = ctx.stream();

    અચલ int64_t n_as = ne02;
    અચલ int64_t n_ids = ids->ne[0];

    જો (ne12 == 1) {
        જો (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            પાછા;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    જો (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<અક્ષર> ids_host(ggml_nbytes(ids));
    અચલ અક્ષર * ids_dev = (અચલ અક્ષર *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    અક્ષર *src0_original = (અક્ષર *)src0->data;
    અક્ષર *src1_original = (અક્ષર *)src1->data;
    અક્ષર *dst_original = (અક્ષર *)dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;
    જો (ne12 == 1) {
        માટે (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            માટે (int64_t id = 0; id < n_ids; id++) {
                અચલ int32_t i02 = *(અચલ int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                અચલ int64_t i11 = id % ne11;
                અચલ int64_t i12 = iid1;

                અચલ int64_t i1 = id;
                અચલ int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } નહીંતર {
        અચલ int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<અક્ષર> src1_contiguous(ctx.pool(), કદ(દશાંશ)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<અક્ષર>  dst_contiguous(ctx.pool(), કદ(દશાંશ)*n_routed_rows*ne0);

        src1_row.data = src1_contiguous.get();
        dst_row.data  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

        mmid_counting_sort_rows(ids, ids_host.data(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*કદ(mmid_row_mapping))));

        અચલ unsigned પૂર્ણાંક max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        પ્રતિજ્ઞા(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned પૂર્ણાંક)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                અક્ષર *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_to_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        તાર્કિક grouped = ખોટું;
        જો (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == કદ(દશાંશ)*ne10 && nb1 == કદ(દશાંશ)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            જો (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            અચલ તાર્કિક src0_reordered =
                src0->extra && ((અચલ ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (અચલ દશાંશ *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (દશાંશ *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        માટે (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            અચલ int64_t અંક_src1_rows = expert_row_counts[i02];

            જો (અંક_src1_rows == 0) {
                ચાલુ;
            }

            અચલ int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == કદ(દશાંશ)*ne10);
            GGML_ASSERT(nb1 == કદ(દશાંશ)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = અંક_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = અંક_src1_rows*nb11;
            src1_row.nb[3] = અંક_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = અંક_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = અંક_src1_rows*nb1;
            dst_row.nb[3] = અંક_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned પૂર્ણાંક)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                અચલ અક્ષર *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                અચલ mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_dst_from_contiguous(dst_original,
                                                   dst_contiguous_get,
                                                   dev_row_mapping_get,
                                                   ne0, nb1, nb2, item_ct1);
                    });
            });
        }
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

સ્થિર શૂન્ય ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


સ્થિર શૂન્ય ggml_sycl_set_main_device(અચલ પૂર્ણાંક main_device) પ્રયાસ {
    જો (dpct::get_current_device_id() == static_cast<unsigned પૂર્ણાંક> (main_device)) {
        પાછા;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    જો (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર તાર્કિક ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, સંરચના ggml_tensor * dst) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    જો (!g_sycl_loaded) પાછા ખોટું;

    જો (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    પસંદગી (dst->op) {
        સ્થિતિ GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ગણતરી_EQUAL:
            ggml_sycl_ગણતરી_equal(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_UNARY:
            પસંદગી (ggml_get_unary_op(dst)) {
                સ્થિતિ GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    તોડો;
                મૂળભૂત:
                    પાછા ખોટું;
            }
            તોડો;
        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(dst)) {
                સ્થિતિ GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    તોડો;
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    તોડો;
                મૂળભૂત:
                    પાછા ખોટું;
            }
            તોડો;
        સ્થિતિ GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            તોડો;
        સ્થિતિ GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT:
            જો (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                પાછા ખોટું;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            તોડો;
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            જો (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                પાછા ખોટું;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            તોડો;
        સ્થિતિ GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            તોડો;
        સ્થિતિ GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ટોચ_K:
            ggml_sycl_op_ટોચ_k(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            તોડો;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            તોડો;
        મૂળભૂત:
            પાછા ખોટું;
    }

    પાછા સાચું;
} પકડો (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API શૂન્ય ggml_backend_sycl_get_device_description(પૂર્ણાંક device, અક્ષર *description,
                                      size_t description_size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર તાર્કિક sycl_get_mem_info(પૂર્ણાંક device, size_t * મુક્ત, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#નહીંતર
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    તાર્કિક res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *મુક્ત, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *મુક્ત);
    પાછા res;
}

શૂન્ય ggml_backend_sycl_get_device_memory(પૂર્ણાંક device, size_t * મુક્ત, size_t * total) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    જો (!sycl_get_mem_info(device, મુક્ત, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *મુક્ત, *total);
} પકડો (અચલ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

સ્થિર અચલ અક્ષર * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    પાછા sycl_ctx->name.c_str();
}

સ્થિર શૂન્ય ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

સ્થિર શૂન્ય ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               અચલ શૂન્ય *data, size_t offset,
                                               size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    અચલ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((અક્ષર *)tensor->data + offset, data, size)));
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               અચલ ggml_tensor *tensor,
                                               શૂન્ય *data, size_t offset,
                                               size_t size) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    અચલ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (અચલ અક્ષર *)tensor->data + offset, size)));
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર તાર્કિક ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               અચલ ggml_tensor *src,
                                               ggml_tensor *dst) પ્રયાસ {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    તાર્કિક is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    જો (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        અચલ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        પાછા સાચું;
    }

    પાછા ખોટું;
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_synchronize(ggml_backend_t backend) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    અચલ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર તાર્કિક ggml_sycl_is_view_or_noop(અચલ ggml_tensor * t) {
    પાછા ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
સ્થિર પૂર્ણાંક ggml_sycl_try_gdn_cache_fusion(અચલ ggml_cgraph * cgraph, પૂર્ણાંક ગાંઠ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    જો (!g_ggml_sycl_enable_fusion) {
        પાછા 0;
    }

    અચલ ggml_tensor * gdn = cgraph->nodes[ગાંઠ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    જો (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        પાછા 0;
    }

    અચલ ggml_tensor * src_v     = gdn->src[2];
    અચલ int64_t       S_v       = src_v->ne[0];
    અચલ int64_t       H         = src_v->ne[1];
    અચલ int64_t       n_tokens  = src_v->ne[2];
    અચલ int64_t       n_seqs    = src_v->ne[3];
    અચલ int64_t       D         = S_v * S_v * H;
    અચલ int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    અચલ int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    અચલ size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    અચલ ggml_tensor * cpy  = nullptr;
    પૂર્ણાંક                 skip = 0;
    માટે (પૂર્ણાંક j = ગાંઠ_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        અચલ ggml_tensor * n = cgraph->nodes[j];
        જો (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }
        જો (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            પાછા 0;
        }
        cpy  = n;
        skip = j - ગાંઠ_idx;
    }
    જો (cpy == nullptr) {
        પાછા 0;
    }

    અચલ ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    અચલ ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    જો (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        પાછા 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    અચલ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    જો (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        પાછા 0;
    }

    fused_state_cpy.data        = (દશાંશ *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / કદ(દશાંશ)) : 0;
    પાછા skip;
}

સ્થિર શૂન્ય ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ગાંઠ = cgraph->nodes[i];
        જો (ggml_sycl_is_view_or_noop(ગાંઠ)) {
            ચાલુ;
        }
        જો ((ગાંઠ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ચાલુ;
        }

        અચલ પૂર્ણાંક nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        જો (nodes_to_skip != 0) {
            i += nodes_to_skip;
            ચાલુ;
        }
#ifndef NDEBUG
        પ્રતિજ્ઞા(ગાંઠ->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        માટે (પૂર્ણાંક j = 0; j < GGML_MAX_SRC; j++) {
            જો (ગાંઠ->src[j] != nullptr) {
                પ્રતિજ્ઞા(ગાંઠ->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        જો (ગાંઠ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            અચલ પૂર્ણાંક gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            જો (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ગાંઠ, fused_state_cpy);
                i += gdn_nodes_to_skip;
                ચાલુ;
            }
        }
        જો (ગાંઠ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ચાલુ;
        }
        જો (ગાંઠ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1]);
            i++;
            ચાલુ;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        જો (ગાંઠ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1]);
            i++;
            ચાલુ;
        }
        જો (ગાંઠ->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1]);
            i++;
            ચાલુ;
        }
        જો (ગાંઠ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ગાંઠ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1]);
            i++;
            ચાલુ;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        જો (ગાંઠ->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ચાલુ;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        જો (ગાંઠ->op == GGML_OP_L2_NORM) {
            અચલ પૂર્ણાંક l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            જો (l2_batch_skip > 0) {
                i += l2_batch_skip;
                ચાલુ;
            }
        }

        જો (ગાંઠ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ગાંઠ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ચાલુ;
        }

        જો (ગાંઠ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ગાંઠ, nullptr, cgraph->nodes[i + 1]);
            i++;
            ચાલુ;
        }

        જો (ગાંઠ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            ચાલુ;
        }

        તાર્કિક ok = ggml_sycl_compute_forward(*sycl_ctx, ગાંઠ);
        જો (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ગાંઠ->name, ggml_op_name(ગાંઠ->op));
        }
        GGML_ASSERT(ok);
    }
}

#ifdef GGML_SYCL_GRAPH
સ્થિર તાર્કિક check_graph_compatibility(ggml_cgraph * cgraph) {
    જો (ggml_sycl_info().device_ગણતરી > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        પાછા ખોટું;
    }

    માટે (પૂર્ણાંક i = 0; i < cgraph->n_nodes; i++) {
        અચલ ggml_op ગાંઠ_op = cgraph->nodes[i]->op;
        પસંદગી (ગાંઠ_op) {
            મૂળભૂત:
                તોડો;
            સ્થિતિ GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            સ્થિતિ GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ગાંઠ_op));
                પાછા ખોટું;
            સ્થિતિ GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                જો (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ગાંઠ_op));
                    પાછા ખોટું;
                }
        }
    }
    પાછા સાચું;
}
#endif

સ્થિર ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    તાર્કિક use_sycl_graph = ખોટું;
    જો (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    જો (use_sycl_graph) {
        અચલ તાર્કિક graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        જો (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            પાછા GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        અચલ તાર્કિક graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        જો (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } નહીંતર {
            પ્રયાસ {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } પકડો (sycl::exception અચલ & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } નહીંતર
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    પાછા GGML_STATUS_SUCCESS;
}

સ્થિર શૂન્ય ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
પ્રયાસ
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    અચલ queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
પકડો (sycl::exception અચલ &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

સ્થિર શૂન્ય ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) પ્રયાસ {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    જો (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } નહીંતર
        GGML_ABORT("fatal error");
} પકડો (sycl::exception અચલ& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

સ્થિર ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ખાલી,
    /* .get_tensor_2d_async     = */ ખાલી,
    /* .cpy_tensor_async        = */ ખાલી, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ખાલી,
    /* .graph_plan_free         = */ ખાલી,
    /* .graph_plan_update       = */ ખાલી,
    /* .graph_plan_compute      = */ ખાલી,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ખાલી,
};

સ્થિર ggml_guid_t ggml_backend_sycl_guid() {
    સ્થિર ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    પાછા &guid;
}

તાર્કિક ggml_backend_is_sycl(ggml_backend_t backend) {
    પાછા backend != ખાલી && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

પૂર્ણાંક ggml_backend_sycl_get_device_ગણતરી(શૂન્ય) {
    પાછા ggml_sycl_info().device_ગણતરી;
}


// backend device

સ્થિર અચલ અક્ષર * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા ctx->name.c_str();
}

સ્થિર અચલ અક્ષર * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા ctx->description.c_str();
}

સ્થિર શૂન્ય ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * મુક્ત, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    જો (!sycl_get_mem_info(ctx->device, મુક્ત, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *મુક્ત);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *મુક્ત, *total);
}

સ્થિર ગણના ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    અચલ sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    જો (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        પાછા GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    પાછા GGML_BACKEND_DEVICE_TYPE_GPU;
}

સ્થિર શૂન્ય ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    તાર્કિક host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    તાર્કિક events = ખોટું;
#નહીંતર
    તાર્કિક events = સાચું;
#endif

    props->caps = {
        /* .async                 = */ સાચું,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ ખોટું,
        /* .events                = */ events,
        /* .mmap_support          = */ સાચું,
    };
}

સ્થિર ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, અચલ અક્ષર * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા ggml_backend_sycl_init(ctx->device);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા ggml_backend_sycl_buffer_type(ctx->device);
}

સ્થિર ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    પાછા ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

સ્થિર ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, શૂન્ય * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    પાછા nullptr;
}

સ્થિર તાર્કિક do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    પૂર્ણાંક device = sycl_ctx->device;
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                જો (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    પાછા સાચું;
                }
                પાછા ખોટું;
            }
        સ્થિતિ GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            પાછા (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        સ્થિતિ GGML_OP_CONV_2D_DW:
        સ્થિતિ GGML_OP_CONV_TRANSPOSE_2D:
            પાછા સાચું;
        સ્થિતિ GGML_OP_UNARY:
            પસંદગી (ggml_get_unary_op(op)) {
                સ્થિતિ GGML_UNARY_OP_SGN:
                સ્થિતિ GGML_UNARY_OP_ABS:
                સ્થિતિ GGML_UNARY_OP_NEG:
                સ્થિતિ GGML_UNARY_OP_STEP:
                સ્થિતિ GGML_UNARY_OP_RELU:
                સ્થિતિ GGML_UNARY_OP_HARDSIGMOID:
                સ્થિતિ GGML_UNARY_OP_TANH:
                સ્થિતિ GGML_UNARY_OP_GELU:
                સ્થિતિ GGML_UNARY_OP_SILU:
                સ્થિતિ GGML_UNARY_OP_SIGMOID:
                સ્થિતિ GGML_UNARY_OP_HARDSWISH:
                સ્થિતિ GGML_UNARY_OP_GELU_QUICK:
                સ્થિતિ GGML_UNARY_OP_GELU_ERF:
                સ્થિતિ GGML_UNARY_OP_EXP:
                સ્થિતિ GGML_UNARY_OP_EXPM1:
                સ્થિતિ GGML_UNARY_OP_SOFTPLUS:
                સ્થિતિ GGML_UNARY_OP_ELU:
                સ્થિતિ GGML_UNARY_OP_XIELU:
                સ્થિતિ GGML_UNARY_OP_CEIL:
                    પાછા સાચું;
                સ્થિતિ GGML_UNARY_OP_FLOOR:
                સ્થિતિ GGML_UNARY_OP_ROUND:
                સ્થિતિ GGML_UNARY_OP_TRUNC:
                    પાછા સાચું;
                મૂળભૂત:
                    પાછા ખોટું;
            }
        સ્થિતિ GGML_OP_GLU:
            પસંદગી (ggml_get_glu_op(op)) {
                સ્થિતિ GGML_GLU_OP_REGLU:
                સ્થિતિ GGML_GLU_OP_GEGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU:
                સ્થિતિ GGML_GLU_OP_SWIGLU_OAI:
                સ્થિતિ GGML_GLU_OP_GEGLU_ERF:
                સ્થિતિ GGML_GLU_OP_GEGLU_QUICK:
                સ્થિતિ GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    પાછા ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                મૂળભૂત:
                    પાછા ખોટું;
            }
        સ્થિતિ GGML_OP_MUL_MAT:
        સ્થિતિ GGML_OP_MUL_MAT_ID:
            {
                સંરચના ggml_tensor * a = op->src[0];
                સંરચના ggml_tensor * b = op->src[1];

                જો (a->ne[3] != b->ne[3]) {
                    પાછા ખોટું;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                જો (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  પાછા ખોટું;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                જો (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    પાછા ખોટું;
                }

                જો (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    પાછા ખોટું;
                }

                પાછા સાચું;
            }
        સ્થિતિ GGML_OP_OUT_PROD:
            પાછા op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_GET_ROWS:
            {
                પસંદગી (op->src[0]->type) {
                    સ્થિતિ GGML_TYPE_I32:
                    સ્થિતિ GGML_TYPE_F16:
                    સ્થિતિ GGML_TYPE_BF16:
                    સ્થિતિ GGML_TYPE_F32:
                    સ્થિતિ GGML_TYPE_Q1_0:
                    સ્થિતિ GGML_TYPE_MXFP4:
                    સ્થિતિ GGML_TYPE_NVFP4:
                    સ્થિતિ GGML_TYPE_IQ2_XXS:
                    સ્થિતિ GGML_TYPE_IQ2_XS:
                    સ્થિતિ GGML_TYPE_IQ2_S:
                    સ્થિતિ GGML_TYPE_IQ3_XXS:
                    સ્થિતિ GGML_TYPE_IQ1_S:
                    સ્થિતિ GGML_TYPE_IQ1_M:
                    સ્થિતિ GGML_TYPE_IQ3_S:
                    સ્થિતિ GGML_TYPE_IQ4_NL:
                    સ્થિતિ GGML_TYPE_IQ4_XS:
                    સ્થિતિ GGML_TYPE_Q2_K:
                    સ્થિતિ GGML_TYPE_Q3_K:
                    સ્થિતિ GGML_TYPE_Q4_0:
                    સ્થિતિ GGML_TYPE_Q4_1:
                    સ્થિતિ GGML_TYPE_Q4_K:
                    સ્થિતિ GGML_TYPE_Q5_0:
                    સ્થિતિ GGML_TYPE_Q5_1:
                    સ્થિતિ GGML_TYPE_Q5_K:
                    સ્થિતિ GGML_TYPE_Q6_K:
                    સ્થિતિ GGML_TYPE_Q8_0:
                        પાછા સાચું;
                    મૂળભૂત:
                        પાછા ખોટું;
                }
            }
        સ્થિતિ GGML_OP_GET_ROWS_BACK:
            // return true;
             પાછા op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         સ્થિતિ GGML_OP_SET:
               પાછા (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        સ્થિતિ GGML_OP_SET_ROWS:
            {
                જો (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    પાછા ખોટું;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                પાછા res;
            }
        સ્થિતિ GGML_OP_DSV4_HC_PRE:
            પાછા op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_DSV4_HC_COMB:
            પાછા op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_DSV4_HC_POST:
            પાછા op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_LIGHTNING_INDEXER:
            પાછા op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        સ્થિતિ GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                જો (src0_type == GGML_TYPE_F16) {
                    જો (src1_type == GGML_TYPE_Q2_K ||
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
                        પાછા ખોટું;
                    }
                }

                જો (src0_type == GGML_TYPE_BF16) {
                    જો (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        પાછા ખોટું;
                    }
                }

                જો (src0_type == GGML_TYPE_F32) {
                    જો (src1_type == GGML_TYPE_Q2_K ||
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
                        પાછા ખોટું;
                    }
                }

                જો (src1_type == GGML_TYPE_F32) {
                    જો (src0_type == GGML_TYPE_Q1_0 ||
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
                        પાછા ખોટું;
                    }
                }

                જો (src0_type == src1_type) {
                    જો (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        પાછા ખોટું;
                    }
                }

                જો (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    પાછા ખોટું;
                }

                પાછા સાચું;
            }
        સ્થિતિ GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                પાછા src0_type == GGML_TYPE_F32;
            }
        સ્થિતિ GGML_OP_CONCAT:
        સ્થિતિ GGML_OP_DUP:
        સ્થિતિ GGML_OP_ARGMAX:
        સ્થિતિ GGML_OP_NONE:
        સ્થિતિ GGML_OP_RESHAPE:
        સ્થિતિ GGML_OP_VIEW:
        સ્થિતિ GGML_OP_PERMUTE:
        સ્થિતિ GGML_OP_TRANSPOSE:
        સ્થિતિ GGML_OP_ADD:
        સ્થિતિ GGML_OP_ADD1:
        સ્થિતિ GGML_OP_ADD_ID:
        સ્થિતિ GGML_OP_SUB:
        સ્થિતિ GGML_OP_ગણતરી_EQUAL:
        સ્થિતિ GGML_OP_MUL:
        સ્થિતિ GGML_OP_DIV:
        સ્થિતિ GGML_OP_REPEAT:
            પાછા સાચું;
        સ્થિતિ GGML_OP_PAD_REFLECT_1D:
            પાછા ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_SQR:
        સ્થિતિ GGML_OP_SQRT:
        સ્થિતિ GGML_OP_SIN:
        સ્થિતિ GGML_OP_COS:
        સ્થિતિ GGML_OP_CLAMP:
        સ્થિતિ GGML_OP_LOG:
        સ્થિતિ GGML_OP_NORM:
        સ્થિતિ GGML_OP_L2_NORM:
        સ્થિતિ GGML_OP_GROUP_NORM:
        સ્થિતિ GGML_OP_RMS_NORM:
            પાછા સાચું;
        સ્થિતિ GGML_OP_RMS_NORM_BACK:
            પાછા ggml_is_contiguous(op->src[0]);
        સ્થિતિ GGML_OP_SCALE:
            // the scale kernel is f32 only
            પાછા op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_CONT:
            પાછા સાચું;
        સ્થિતિ GGML_OP_TRI:
            {
                અચલ ggml_tensor * src0 = op->src[0];
                પાછા src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        સ્થિતિ GGML_OP_DIAG_MASK_INF:
            પાછા સાચું;
        સ્થિતિ GGML_OP_SOFT_MAX:
            પાછા સાચું;
        સ્થિતિ GGML_OP_SOFT_MAX_BACK: {
            દશાંશ max_bias = 0.0f;
            memcpy(&max_bias, (અચલ દશાંશ *) op->op_params + 1, કદ(દશાંશ));
            પાછા max_bias == 0.0f;
        }
        સ્થિતિ GGML_OP_ROPE:
        સ્થિતિ GGML_OP_ROPE_BACK:
        સ્થિતિ GGML_OP_IM2COL:
        સ્થિતિ GGML_OP_IM2COL_3D:
        સ્થિતિ GGML_OP_UPSCALE:
            પાછા સાચું;
        સ્થિતિ GGML_OP_COL2IM_1D:
            પાછા ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        સ્થિતિ GGML_OP_CONV_3D: {
            અચલ int32_t * opts = (અચલ int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            અચલ int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            અચલ int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            પાછા op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        સ્થિતિ GGML_OP_SUM:
        સ્થિતિ GGML_OP_SUM_ROWS:
        સ્થિતિ GGML_OP_MEAN:
            પાછા ggml_is_contiguous(op->src[0]);
        સ્થિતિ GGML_OP_ARGSORT:
            પાછા સાચું;
        સ્થિતિ GGML_OP_ટોચ_K: {
            અચલ ggml_tensor * src0 = op->src[0];
            અચલ પૂર્ણાંક k = op->ne[0];
            પાછા src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        સ્થિતિ GGML_OP_POOL_2D:
        સ્થિતિ GGML_OP_POOL_1D:
        સ્થિતિ GGML_OP_ACC:
            પાછા સાચું;
        સ્થિતિ GGML_OP_PAD:
            જો (ggml_get_op_params_i32(op, 8) != 0) {
                પાછા ખોટું;
            }
            પાછા સાચું;
        સ્થિતિ GGML_OP_LEAKY_RELU:
        સ્થિતિ GGML_OP_TIMESTEP_EMBEDDING:
        સ્થિતિ GGML_OP_RWKV_WKV6:
        સ્થિતિ GGML_OP_RWKV_WKV7:
        સ્થિતિ GGML_OP_GATED_LINEAR_ATTN:
        સ્થિતિ GGML_OP_GATED_DELTA_NET:
        સ્થિતિ GGML_OP_OPT_STEP_ADAMW:
        સ્થિતિ GGML_OP_OPT_STEP_SGD:
            પાછા સાચું;
        સ્થિતિ GGML_OP_SSM_CONV:
            પાછા op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_ROLL:
            પાછા op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_ARANGE:
            પાછા op->type == GGML_TYPE_F32;
        સ્થિતિ GGML_OP_SSM_SCAN:
            જો (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                પાછા (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } નહીંતર {
                // TODO Mamba-1 not yet ported to SYCL
                પાછા ખોટું;
            }
        સ્થિતિ GGML_OP_FILL:
        સ્થિતિ GGML_OP_CUMSUM:
        સ્થિતિ GGML_OP_DIAG:
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS:
        સ્થિતિ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            પાછા સાચું;
        સ્થિતિ GGML_OP_SOLVE_TRI:
            પાછા op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        સ્થિતિ GGML_OP_FLASH_ATTN_EXT:
            પાછા ggml_sycl_flash_attn_ext_supported(device, op);
        મૂળભૂત:
            પાછા ખોટું;
    }

    GGML_UNUSED(dev);
}

સ્થિર તાર્કિક ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    તાર્કિક res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    પાછા res;
}

સ્થિર તાર્કિક ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    જો (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        પાછા ખોટું;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા buft_ctx->device == sycl_ctx->device;
}

સ્થિર int64_t get_op_batch_size(અચલ ggml_tensor * op) {
    પસંદગી (op->op) {
        સ્થિતિ GGML_OP_GET_ROWS:
            પાછા 0;
        સ્થિતિ GGML_OP_MUL_MAT:
            પાછા op->ne[1];
        સ્થિતિ GGML_OP_MUL_MAT_ID:
        સ્થિતિ GGML_OP_ROPE:
            પાછા op->ne[2];
        મૂળભૂત:
            પાછા ggml_nrows(op);
    }
}

સ્થિર તાર્કિક ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, અચલ ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    પાછા get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

સ્થિર ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    પાછા nullptr;
#નહીંતર
  sycl::event *event_ptr = નવું sycl::event();

  પાછા નવું ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

સ્થિર શૂન્ય ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) પ્રયાસ {
  GGML_UNUSED(dev);
  જો (event == nullptr) {
    પાછા;
  }

  જો (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


સ્થિર શૂન્ય ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) પ્રયાસ {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} પકડો (sycl::exception અચલ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

સ્થિર અચલ ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_buffer_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

સંરચના ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

સ્થિર અચલ અક્ષર * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    પાછા GGML_SYCL_NAME;
}

સ્થિર size_t ggml_backend_sycl_reg_get_device_ગણતરી(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    પાછા ctx->devices.size();
}

સ્થિર ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    પાછા ctx->devices[index];
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

સંરચના ggml_backend_sycl_comm_context {
    std::vector<ggml_backend_t> backends;
    // ONE persistent per-device byte buffer, 4*nelem bytes.  Both the
    // FP32 small-tensor path and the BF16 large-tensor path share it
    // by reinterpreting.
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf0;
    std::unique_ptr<ggml_sycl_pool_alloc<uint8_t>> buf1;
    int64_t buf_nelem = 0;

    // pinned host staging per device context, reused across calls (queues are in-order)
    uint8_t * host_out[2] = { nullptr, nullptr };
    uint8_t * host_in[2]  = { nullptr, nullptr };
    size_t    host_bytes  = 0;

    શૂન્ય free_host() {
        માટે (પૂર્ણાંક i = 0; i < 2; ++i) {
            અચલ sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            જો (host_out[i]) { sycl::મુક્ત(host_out[i], sctx); host_out[i] = nullptr; }
            જો (host_in[i])  { sycl::મુક્ત(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
સ્થિર શૂન્ય ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    શૂન્ય * dst0, અચલ શૂન્ય * src1, શૂન્ય * dst1, અચલ શૂન્ય * src0,
                                    size_t nbytes) {
    જો (comm->host_bytes < nbytes) {
        q0->wait();
        q1->wait();
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_in[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_in[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_out[0] && comm->host_in[0] && comm->host_out[1] && comm->host_in[1]);
        comm->host_bytes = nbytes;
    }

    sycl::event e0 = q0->memcpy(comm->host_out[0], src0, nbytes);
    sycl::event e1 = q1->memcpy(comm->host_out[1], src1, nbytes);
    e0.wait();
    e1.wait();
    std::memcpy(comm->host_in[0], comm->host_out[1], nbytes);
    std::memcpy(comm->host_in[1], comm->host_out[0], nbytes);
    q0->memcpy(dst0, comm->host_in[0], nbytes);
    q1->memcpy(dst1, comm->host_in[1], nbytes);
}

શૂન્ય * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) પ્રયાસ {
    માટે (size_t i = 0; i < n_backends; ++i) {
        જો (!ggml_backend_is_sycl(backends[i])) {
            પાછા nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    જો (n_backends != 2) {
        પાછા nullptr;
    }

    auto * ctx = નવું ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    પાછા ctx;
}
પકડો (અચલ sycl::exception &) { પાછા nullptr; }
પકડો (...)                     { પાછા nullptr; }

શૂન્ય ggml_backend_sycl_comm_free(શૂન્ય * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    જો (comm_ctx == nullptr) {
        પાછા;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    જો (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        પ્રયાસ {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } પકડો (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

તાર્કિક ggml_backend_sycl_comm_allreduce_tensor(શૂન્ય * comm_ctx_v, સંરચના ggml_tensor ** tensors) પ્રયાસ {
    જો (comm_ctx_v == nullptr) {
        પાછા ખોટું;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    અચલ size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    જો (n_backends != 2) {
        પાછા ખોટું;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    જો (tensors[0]->type != tensors[1]->type) {
        પાછા ખોટું;
    }
    જો (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        પાછા ખોટું;
    }
    જો (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        પાછા ખોટું;
    }
    જો (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        પાછા ખોટું;
    }

    અચલ int64_t nelem  = ggml_nelements(tensors[0]);
    અચલ size_t  nbytes = ggml_nbytes(tensors[0]);
    જો (nelem == 0) {
        પાછા સાચું;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    જો (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    જો (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((દશાંશ) f16_out0[i] + (દશાંશ) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((દશાંશ) f16_out1[i] + (દશાંશ) f16_tmp1[i]);
            });
        });
        પાછા સાચું;
    }

    દશાંશ * out0 = (દશાંશ *) tensors[0]->data;
    દશાંશ * out1 = (દશાંશ *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    સ્થિર constexpr int64_t BF16_THRESHOLD = 32768;

    જો (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        દશાંશ * tmp0 = (દશાંશ *) buf0;
        દશાંશ * tmp1 = (દશાંશ *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->data, tmp1, tensors[0]->data, nbytes);

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
        પાછા સાચું;
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
        outbox0[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out0[i]) >> 16);
    });

    q1->parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
        outbox1[i] = (uint16_t) (sycl::bit_cast<uint32_t>(out1[i]) >> 16);
    });

    // Phase B: exchange the compressed bytes (in-order queues keep this after the compress kernels).
    અચલ size_t bf16_bytes = nelem * કદ(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<દશાંશ>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<દશાંશ>(((uint32_t) inbox1[i]) << 16);
        });
    });

    પાછા સાચું;
}
પકડો (અચલ sycl::exception &) { પાછા ખોટું; }
પકડો (...)                     { પાછા ખોટું; }

સ્થિર શૂન્ય *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, અચલ અક્ષર *name) {
    GGML_UNUSED(reg);

    જો (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        પાછા (શૂન્ય *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    જો (strcmp(name, "ggml_backend_comm_init") == 0) {
        પાછા (શૂન્ય *)ggml_backend_sycl_comm_init;
    }
    જો (strcmp(name, "ggml_backend_comm_free") == 0) {
        પાછા (શૂન્ય *)ggml_backend_sycl_comm_free;
    }
    જો (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        પાછા (શૂન્ય *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    પાછા nullptr;
}

સ્થિર અચલ ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_ગણતરી,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    સ્થિર ggml_backend_reg reg;
    સ્થિર તાર્કિક initialized = ખોટું;

    {
        સ્થિર std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        જો (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = નવું ggml_backend_sycl_reg_context;
            અચલ પૂર્ણાંક min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            માટે (પૂર્ણાંક i = 0; i < ggml_sycl_info().device_ગણતરી; i++) {
                ggml_backend_sycl_device_context * dev_ctx = નવું ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = નવું ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = સાચું;
    }

    પાછા &reg;
}

ggml_backend_t ggml_backend_sycl_init(પૂર્ણાંક device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = નવું ggml_backend_sycl_context(device);
    જો (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        પાછા nullptr;
    };

    ggml_backend_t sycl_backend = નવું ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    પાછા sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
