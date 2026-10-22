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

#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <array>
#അന്തര്ഭാവ <assert.h>
#അന്തര്ഭാവ <atomic>
#അന്തര്ഭാവ <cctype>
#അന്തര്ഭാവ <cinttypes>
#അന്തര്ഭാവ <cstddef>
#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <float.h>
#അന്തര്ഭാവ <limits>
#അന്തര്ഭാവ <optional>
#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <fstream>
#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <regex>

#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#അന്തര്ഭാവ <level_zero/ze_api.h>
#അന്തര്ഭാവ <level_zero/zes_api.h>
#endif
#എങ്കിൽ defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#എങ്കിൽ SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#അന്തര്ഭാവ <sycl/half_type.hpp>

#അന്തര്ഭാവ "ggml.h"
#അന്തര്ഭാവ "ggml-sycl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "ggml-backend-impl.h"

#അന്തര്ഭാവ "ggml-sycl/add-id.hpp"
#അന്തര്ഭാവ "ggml-sycl/backend.hpp"
#അന്തര്ഭാവ "ggml-sycl/common.hpp"
#അന്തര്ഭാവ "ggml-sycl/element_wise.hpp"
#അന്തര്ഭാവ "ggml-sycl/fwht.hpp"
#അന്തര്ഭാവ "ggml-sycl/fused-gemm.hpp"
#അന്തര്ഭാവ "ggml-sycl/gemm.hpp"
#അന്തര്ഭാവ "ggml-sycl/getrows.hpp"
#അന്തര്ഭാവ "ggml-sycl/mem.hpp"
#അന്തര്ഭാവ "ggml-sycl/norm.hpp"
#അന്തര്ഭാവ "ggml-sycl/presets.hpp"
#അന്തര്ഭാവ "ggml-sycl/quantize.hpp"
#അന്തര്ഭാവ "ggml-sycl/repeat_back.hpp"
#അന്തര്ഭാവ "ggml-sycl/set_rows.hpp"
#അന്തര്ഭാവ "ggml-sycl/set.hpp"
#അന്തര്ഭാവ "ggml-sycl/dsv4-hc.hpp"
#അന്തര്ഭാവ "ggml-sycl/lightning-indexer.hpp"
#അന്തര്ഭാവ "ggml-sycl/conv2d.hpp"
#അന്തര്ഭാവ "ggml-sycl/conv2d-dw.hpp"
#അന്തര്ഭാവ "ggml-sycl/conv2d-transpose.hpp"
#അന്തര്ഭാവ "ggml-sycl/ssm_conv.hpp"
#അന്തര്ഭാവ "ggml-sycl/sycl_hw.hpp"
#അന്തര്ഭാവ "ggml-sycl/ssm_scan.hpp"
#അന്തര്ഭാവ "ggml-sycl/fill.hpp"
#അന്തര്ഭാവ "ggml-sycl/cumsum.hpp"
#അന്തര്ഭാവ "ggml-sycl/diag.hpp"
#അന്തര്ഭാവ "ggml-sycl/opt-step.hpp"
#അന്തര്ഭാവ "ggml-sycl/solve_tri.hpp"
#അന്തര്ഭാവ "ggml-sycl/gated_delta_net.hpp"
#അന്തര്ഭാവ "ggml-sycl/pool.hpp"
#അന്തര്ഭാവ "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

സ്ഥിതം തർക്കം g_sycl_loaded = അസത്യം;
പൂർണ്ണസംഖ്യ g_ggml_sycl_debug = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_dev_debug = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_optimize = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_graph = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_dnn = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_fa_onednn = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_fa_onednn_max_kv = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_mkl_fa = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_memtrace = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_memtrace_step = 64;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_vmm = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_fusion = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_esimd = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_mmvq_wide = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_prioritize_dmmv = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
പൂർണ്ണസംഖ്യ g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
പൂർണ്ണസംഖ്യ g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
പൂർണ്ണസംഖ്യ g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_sycl_get_env_precision(സ്ഥിരം അക്ഷരം * name, പൂർണ്ണസംഖ്യ def) {
    സ്ഥിരം അക്ഷരം * env = getenv(name);
    എങ്കിൽ (!env) {
        തിരികെ def;
    }
    std::string mode(env);
    വേണ്ടി (അക്ഷരം & c : mode) {
        c = (അക്ഷരം) std::toupper((unsigned അക്ഷരം) c);
    }
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        എങ്കിൽ (mode == ggml_sycl_dynamic_precision_names[i]) {
            തിരികെ i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    തിരികെ def;
}
പൂർണ്ണസംഖ്യ g_ggml_sycl_use_async_mem_op = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_use_async_mem_op_requested = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_use_level_zero_api = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_flash_attention = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
പൂർണ്ണസംഖ്യ g_ggml_sycl_usm_system = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_host_pinned_mem = 1;
പൂർണ്ണസംഖ്യ g_ggml_sycl_host_pinned_mem_2g = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_upload_staging_slots = 4;
പൂർണ്ണസംഖ്യ g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
പൂർണ്ണസംഖ്യ g_ggml_sycl_enable_sparse_fa = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_debug_sparse_fa = 0;
പൂർണ്ണസംഖ്യ g_ggml_sycl_sparse_fa_margin = 256;

#എങ്കിൽ GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
സ്ഥിതം തർക്കം ggml_sycl_dnnl_detect_optimized_gemm(പൂർണ്ണസംഖ്യ device) {
    using dt = dnnl::memory::data_type;

    ശ്രമിക്കുക {
        സ്ഥിരം dnnl::memory::dims dims    = { 1, 64, 64 };
        സ്ഥിരം dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        സ്ഥിരം auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        സ്ഥിരം auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        സ്ഥിരം auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        സ്ഥിരം std::string impl = pd.impl_info_str();
        എങ്കിൽ (impl.find("ref") == std::string::npos) {
            തിരികെ സത്യം;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } പിടിക്കുക (സ്ഥിരം std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    തിരികെ അസത്യം;
}
#endif

സ്ഥിതം ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    ശ്രമിക്കുക {
        info.device_എണ്ണം = dpct::dev_mgr::instance().device_എണ്ണം();
    } പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_എണ്ണം = 0;
    }
    എങ്കിൽ (info.device_എണ്ണം == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        തിരികെ info;
    }

    GGML_ASSERT(info.device_എണ്ണം <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < info.device_എണ്ണം; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#എങ്കിൽ !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#അല്ലെങ്കിൽ
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        എങ്കിൽ (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            സ്ഥിരം size_t physical_page = 2ull << 20; // 2 MiB
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
#എങ്കിൽ GGML_SYCL_DNNL
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
        എങ്കിൽ (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = അസത്യം;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        എങ്കിൽ (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            എങ്കിൽ (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = സത്യം;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < info.device_എണ്ണം; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    തിരികെ info;
}

സ്ഥിരം ggml_sycl_device_info & ggml_sycl_info() {
    സ്ഥിതം ggml_sycl_device_info info = ggml_sycl_init();
    തിരികെ info;
}

സ്ഥിതം ശൂന്യം print_device_detail(പൂർണ്ണസംഖ്യ id, sycl::device &device, std::string device_type) {

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

സ്ഥിതം ശൂന്യം print_device_opt_feature(പൂർണ്ണസംഖ്യ device_എണ്ണം) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < device_എണ്ണം; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      പൂർണ്ണസംഖ്യ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
ശൂന്യം ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    പൂർണ്ണസംഖ്യ device_എണ്ണം = dpct::dev_mgr::instance().device_എണ്ണം();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_എണ്ണം);

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

    വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < device_എണ്ണം; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      പൂർണ്ണസംഖ്യ type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_എണ്ണം);
}

സ്ഥിതം സ്ഥിരം അക്ഷരം* dev2dev_int2str(പൂർണ്ണസംഖ്യ dev2dev) {
    എങ്കിൽ (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        തിരികെ "SYCL API";
    } അല്ലെങ്കിൽ എങ്കിൽ (dev2dev == DEV2DEV_MEMCPY_L0) {
        തിരികെ "Level Zero API";
    } അല്ലെങ്കിൽ എങ്കിൽ (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        തിരികെ "Host Forward";
    } അല്ലെങ്കിൽ {
        തിരികെ "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
സ്ഥിതം ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    എങ്കിൽ (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (പൂർണ്ണസംഖ്യ) res);
    }
    തിരികെ res;
}

സ്ഥിതം ze_result_t get_zes_init_res() {
    സ്ഥിതം ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (പൂർണ്ണസംഖ്യ) zes_init_res);
    തിരികെ zes_init_res;
}
#endif

സ്ഥിതം ശൂന്യം initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

സ്ഥിതം ശൂന്യം ggml_check_sycl() ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    സ്ഥിതം തർക്കം initialized = അസത്യം;

    എങ്കിൽ (!initialized) {
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
#അല്ലെങ്കിൽ
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        എങ്കിൽ (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#അല്ലെങ്കിൽ
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
#എങ്കിൽ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#എങ്കിൽ defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#എങ്കിൽ defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#എങ്കിൽ defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#എങ്കിൽ defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#എങ്കിൽ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#എങ്കിൽ defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#എങ്കിൽ defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#എങ്കിൽ defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#അല്ലെങ്കിൽ
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#അല്ലെങ്കിൽ
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
#എങ്കിൽ defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        എങ്കിൽ (g_ggml_sycl_use_async_mem_op) {
            വേണ്ടി (unsigned പൂർണ്ണസംഖ്യ i = 0; i < dpct::dev_mgr::instance().device_എണ്ണം(); ++i) {
                എങ്കിൽ (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    നിർത്തുക;
                }
            }
        }
#endif
        എങ്കിൽ (CHECK_TRY_ERROR(g_all_sycl_device_എണ്ണം =
                            dpct::dev_mgr::instance().device_എണ്ണം()) != 0) {
            initialized = സത്യം;
            g_sycl_loaded = അസത്യം;
            തിരികെ;
        }
        GGML_ASSERT(g_all_sycl_device_എണ്ണം <= GGML_SYCL_MAX_DEVICES);

        initialized = സത്യം;
        g_sycl_loaded = സത്യം;
        ggml_backend_sycl_print_sycl_devices();
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
വരിയിൽ ശൂന്യം check_allow_gpu_index(സ്ഥിരം പൂർണ്ണസംഖ്യ device_index) {
  എങ്കിൽ (device_index >= ggml_sycl_info().device_എണ്ണം) {
    അക്ഷരം error_buf[256];
    snprintf(
        error_buf,
        വലുപ്പം(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_എണ്ണം - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    പ്രതിജ്ഞാ(അസത്യം);
  }
}

GGML_API ശൂന്യം ggml_backend_sycl_get_gpu_list(പൂർണ്ണസംഖ്യ *id_list, പൂർണ്ണസംഖ്യ max_len) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    വേണ്ടി(പൂർണ്ണസംഖ്യ i=0;i<max_len;i++) id_list[i] = -1;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i=0;i< ggml_sycl_info().device_എണ്ണം;i++){
        എങ്കിൽ (i>=max_len) നിർത്തുക;
        id_list[i] = i;
    }
    തിരികെ;
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

വരിയിൽ ശൂന്യം free_aligned_mem_host(ശൂന്യം * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#അല്ലെങ്കിൽ
    സ്വതന്ത്രം(memblock);
#endif
}

// sycl buffer

ഘടന ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    സ്ഥിതം constexpr size_t staging_slot_size = 8*1024*1024;

    ഘടന host_staging {
        ശൂന്യം * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<തർക്കം> submitted;
        പൂർണ്ണസംഖ്യ next = 0;
    };

    പൂർണ്ണസംഖ്യ device;
    ശൂന്യം * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    തർക്കം is_usm_system;

    ggml_backend_sycl_buffer_context(പൂർണ്ണസംഖ്യ device, ശൂന്യം * dev_ptr, queue_ptr stream, തർക്കം is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    ശൂന്യം drop_host_staging() {
        വേണ്ടി (size_t i = 0; i < staging.submitted.size(); ++i) {
            എങ്കിൽ (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = അസത്യം;
            }
        }
        എങ്കിൽ (staging.data != nullptr) {
            sycl::സ്വതന്ത്രം(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        എങ്കിൽ (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            എങ്കിൽ (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            അല്ലെങ്കിൽ
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        വേണ്ടി (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

സ്ഥിതം തർക്കം ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    തിരികെ buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

സ്ഥിതം ശൂന്യം
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) ശ്രമിക്കുക {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    തിരികെ ctx->dev_ptr;
}

സ്ഥിതം എണ്ണൽ ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    എങ്കിൽ (tensor->view_src != ശൂന്യ) {
        പ്രതിജ്ഞാ(tensor->view_src->buffer->buft == buffer->buft);
        തിരികെ GGML_STATUS_SUCCESS;
    }

    എങ്കിൽ (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        തിരഞ്ഞെടുപ്പ് (tensor->type) {
            അവസ്ഥ GGML_TYPE_Q4_0:
            അവസ്ഥ GGML_TYPE_Q8_0:
            അവസ്ഥ GGML_TYPE_Q2_K:
            അവസ്ഥ GGML_TYPE_Q3_K:
            അവസ്ഥ GGML_TYPE_Q4_K:
            അവസ്ഥ GGML_TYPE_Q5_K:
            അവസ്ഥ GGML_TYPE_Q6_K:
            അവസ്ഥ GGML_TYPE_IQ3_XXS:
            അവസ്ഥ GGML_TYPE_IQ3_S:
            അവസ്ഥ GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = പുതിയ ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                നിർത്തുക;
            }
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
        }
    }

    എങ്കിൽ (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        എങ്കിൽ (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (അക്ഷരം *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    തിരികെ GGML_STATUS_SUCCESS;
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                സ്ഥിരം ശൂന്യം *data, size_t offset,
                                                size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_slots = g_ggml_sycl_upload_staging_slots;
    എങ്കിൽ (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        എങ്കിൽ (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, അസത്യം);
        }
    }
    എങ്കിൽ (ctx->staging.data != nullptr) {
        queue_ptr    stream    = ctx->stream;
        അക്ഷരം *       dst       = (അക്ഷരം *) tensor->data + offset;
        സ്ഥിരം അക്ഷരം * src       = (സ്ഥിരം അക്ഷരം *) data;
        size_t       remaining = size;
        ആകുമ്പോൾ (remaining > 0) {
            സ്ഥിരം size_t chunk = std::min(remaining, ctx->staging_slot_size);
            സ്ഥിരം പൂർണ്ണസംഖ്യ    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (പൂർണ്ണസംഖ്യ) ctx->staging.submitted.size();
            എങ്കിൽ (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            ശൂന്യം * stage = (അക്ഷരം *) ctx->staging.data + slot * ctx->staging_slot_size;
            memcpy(stage, src, chunk);
            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
            ctx->staging.submitted[slot] = സത്യം;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        തിരികെ;
    }

    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    അക്ഷരം * host_buf = (അക്ഷരം *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((അക്ഷരം *) tensor->data + offset, host_buf, size).wait()));
    സ്വതന്ത്രം(host_buf);
#അല്ലെങ്കിൽ
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((അക്ഷരം *) tensor->data + offset, data, size).wait()));
#endif
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                സ്ഥിരം ggml_tensor *tensor,
                                                ശൂന്യം *data, size_t offset,
                                                size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (സ്ഥിരം അക്ഷരം *)tensor->data + offset, size)
            .wait()));
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
സ്ഥിതം തർക്കം ggml_sycl_is_l0_discrete_gpu(പൂർണ്ണസംഖ്യ device) {
    തിരികെ ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

സ്ഥിതം ശൂന്യം memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, ശൂന്യം *ptr_dst,
                         സ്ഥിരം ശൂന്യം *ptr_src, size_t size) {
    അക്ഷരം *host_buf = (അക്ഷരം *)malloc(size);
    എങ്കിൽ (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.memcpy(host_buf, (സ്ഥിരം അക്ഷരം *)ptr_src, size).wait();
    q_dst.memcpy((അക്ഷരം *)ptr_dst, host_buf, size).wait();
    സ്വതന്ത്രം(host_buf);
}

സ്ഥിതം ശൂന്യം dev2dev_memcpy(പൂർണ്ണസംഖ്യ device_dst, sycl::queue &q_dst, പൂർണ്ണസംഖ്യ device_src, sycl::queue &q_src, ശൂന്യം *ptr_dst,
                    സ്ഥിരം ശൂന്യം *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    എങ്കിൽ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        സ്ഥിരം തർക്കം l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        എങ്കിൽ (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            എങ്കിൽ (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                എങ്കിൽ (r == ZE_RESULT_SUCCESS) {
                    തിരികെ;
                }
            }
        }
    }
#endif

    എങ്കിൽ (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        എങ്കിൽ (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            തിരികെ;
        }
    }

    // Host-staged copy
    എങ്കിൽ(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } അല്ലെങ്കിൽ {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

സ്ഥിതം തർക്കം
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    സ്ഥിരം ggml_tensor *src,
                                    ggml_tensor *dst) ശ്രമിക്കുക {
    തർക്കം is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    എങ്കിൽ (is_cpy_supported) {
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
#എങ്കിൽ 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (അക്ഷരം *)dst->data, (സ്ഥിരം അക്ഷരം *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        തിരികെ സത്യം;
    }
    തിരികെ അസത്യം;
    GGML_UNUSED(buffer);
} പിടിക്കുക (സ്ഥിരം sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    വേണ്ടി (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<അക്ഷരം*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    എങ്കിൽ (size == 0) {
        തിരികെ;  // Nothing to do
    }
    എങ്കിൽ (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    ശൂന്യം * target_ptr = static_cast<അക്ഷരം *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    എങ്കിൽ (buffer == nullptr) {
        തിരികെ;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    എങ്കിൽ (ctx != nullptr) {
        വേണ്ടി (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

സ്ഥിതം സ്ഥിരം ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
ഘടന ggml_backend_sycl_buffer_type_context {
    പൂർണ്ണസംഖ്യ device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    തിരികെ ctx->name.c_str();
}

സ്ഥിതം തർക്കം check_usm_system(പൂർണ്ണസംഖ്യ device, size_t size) {
    തർക്കം use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    എങ്കിൽ (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = അസത്യം;
    }

    തിരികെ use_usm_system;
}

വരിയിൽ ശൂന്യം * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    തിരികെ _aligned_malloc(size, alignment);
#അല്ലെങ്കിൽ
    തിരികെ aligned_alloc(alignment, size);
#endif
}

സ്ഥിതം ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    സ്ഥിരം queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    തർക്കം use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    ശൂന്യം * dev_ptr;
    എങ്കിൽ (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (ശൂന്യം *)aligned_malloc_host(alignment, aligned_size);
        എങ്കിൽ (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            തിരികെ nullptr;
        }
    } അല്ലെങ്കിൽ {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (ശൂന്യം *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        എങ്കിൽ (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          തിരികെ nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = പുതിയ  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    തിരികെ ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം is_bmg_g31_arch(പൂർണ്ണസംഖ്യ device) {
    തിരികെ ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

സ്ഥിതം size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    പൂർണ്ണസംഖ്യ device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    എങ്കിൽ (g_ggml_sycl_host_pinned_mem_2g) {
        തിരികെ std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } അല്ലെങ്കിൽ {
        എങ്കിൽ(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        തിരികെ max_alloc_size;

    }
    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    എങ്കിൽ (ggml_is_quantized(tensor->type)) {
        എങ്കിൽ (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    തിരികെ size;

    GGML_UNUSED(buft);
}

സ്ഥിതം സ്ഥിരം ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ശൂന്യ,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ശൂന്യ,
    /* .is_host             = */ ശൂന്യ,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(പൂർണ്ണസംഖ്യ device) {
    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_എണ്ണം = ggml_backend_sycl_get_device_എണ്ണം();

    എങ്കിൽ (device>=dev_എണ്ണം or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_എണ്ണം-1);
        GGML_ASSERT(device<dev_എണ്ണം);
    }
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    സ്ഥിതം തർക്കം ggml_backend_sycl_buffer_type_initialized = അസത്യം;

    എങ്കിൽ (!ggml_backend_sycl_buffer_type_initialized) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < dev_എണ്ണം; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ പുതിയ ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = സത്യം;
    }
    തിരികെ &ggml_backend_sycl_buffer_types[device];
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    പൂർണ്ണസംഖ്യ device = ctx->device;
    എങ്കിൽ (device>=ggml_sycl_info().device_എണ്ണം or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_എണ്ണം-1);
        GGML_ASSERT(device<ggml_sycl_info().device_എണ്ണം);
    }
    സ്ഥിതം ഘടന ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    സ്ഥിതം തർക്കം ggml_backend_sycl_buffer_type_initialized = അസത്യം;

    എങ്കിൽ (!ggml_backend_sycl_buffer_type_initialized) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ പുതിയ ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = സത്യം;
    }
    തിരികെ &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

സ്ഥിതം int64_t get_row_rounding(ggml_type type, സ്ഥിരം std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        എങ്കിൽ (tensor_split[i] < (i + 1 < ggml_sycl_info().device_എണ്ണം ? tensor_split[i + 1] : 1.0f)) {
            എങ്കിൽ (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            എങ്കിൽ (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    തിരഞ്ഞെടുപ്പ്(type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
            തിരികെ max_compute_capability >= VER_GEN9 ? 128 : 64;
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
            തിരികെ 64;
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_F32:
            തിരികെ 1;
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_IQ2_XXS:
        അവസ്ഥ GGML_TYPE_IQ2_XS:
        അവസ്ഥ GGML_TYPE_IQ2_S:
        അവസ്ഥ GGML_TYPE_IQ1_S:
        അവസ്ഥ GGML_TYPE_IQ1_M:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ4_XS:
        അവസ്ഥ GGML_TYPE_IQ4_NL:
            തിരികെ max_compute_capability >= VER_GEN9 ? 128 : 64;
        അവസ്ഥ GGML_TYPE_IQ3_S:
            തിരികെ max_compute_capability >= VER_GEN9 ? 128 : 64;
        അവസ്ഥ GGML_TYPE_Q6_K:
            തിരികെ 64;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("fatal error");
    }
}

സ്ഥിതം ശൂന്യം get_row_split(int64_t * row_താഴെയുള്ള, int64_t * row_മുകളിലെ, സ്ഥിരം ggml_tensor * tensor, സ്ഥിരം std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> & tensor_split, പൂർണ്ണസംഖ്യ id) {
    സ്ഥിരം int64_t nrows = ggml_nrows(tensor);
    സ്ഥിരം int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_താഴെയുള്ള = id == 0 ? 0 : nrows*tensor_split[id];
    *row_താഴെയുള്ള -= *row_താഴെയുള്ള % rounding;
    എങ്കിൽ (id == ggml_sycl_info().device_എണ്ണം - 1) {
        *row_മുകളിലെ = nrows;
    } അല്ലെങ്കിൽ {
        *row_മുകളിലെ = nrows*tensor_split[id + 1];
        *row_മുകളിലെ -= *row_മുകളിലെ % rounding;
    }
}

സ്ഥിതം size_t ggml_nbytes_split(സ്ഥിരം ഘടന ggml_tensor * tensor, പൂർണ്ണസംഖ്യ nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    തിരികെ nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

ഘടന ggml_backend_sycl_split_buffer_type_context {
    std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> tensor_split;
};

ഘടന ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() ശ്രമിക്കുക {
        വേണ്ടി (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

സ്ഥിതം ശൂന്യം ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

സ്ഥിതം ശൂന്യം * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    തിരികെ (ശൂന്യം *)0x1000;

    GGML_UNUSED(buffer);
}

സ്ഥിതം എണ്ണൽ ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    സ്ഥിരം int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = പുതിയ ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    എങ്കിൽ (ctx->streams.empty()) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        int64_t row_താഴെയുള്ള, row_മുകളിലെ;
        get_row_split(&row_താഴെയുള്ള, &row_മുകളിലെ, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_മുകളിലെ - row_താഴെയുള്ള;
        എങ്കിൽ (nrows_split == 0) {
            തുടരുക;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        സ്ഥിരം size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        എങ്കിൽ (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        സ്ഥിരം queue_ptr stream = ctx->streams[i];
        അക്ഷരം * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (അക്ഷരം *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        എങ്കിൽ (!buf) {
            അക്ഷരം err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            എറിയുക std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        എങ്കിൽ (size > original_size) {
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

        വേണ്ടി (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = പുതിയ sycl::event()));
        }
    }
    tensor->extra = extra;
    തിരികെ GGML_STATUS_SUCCESS;
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, സ്ഥിരം ശൂന്യം *data,
                                          size_t offset, size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    സ്ഥിരം int64_t ne0 = tensor->ne[0];
    സ്ഥിരം size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        int64_t row_താഴെയുള്ള, row_മുകളിലെ;
        get_row_split(&row_താഴെയുള്ള, &row_മുകളിലെ, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_മുകളിലെ - row_താഴെയുള്ള;
        എങ്കിൽ (nrows_split == 0) {
            തുടരുക;
        }

        സ്ഥിരം size_t offset_split = row_താഴെയുള്ള*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        സ്ഥിരം size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        എങ്കിൽ (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        സ്ഥിരം അക്ഷരം * buf_host = (സ്ഥിരം അക്ഷരം *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        സ്ഥിരം queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          സ്ഥിരം ggml_tensor *tensor, ശൂന്യം *data,
                                          size_t offset, size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    സ്ഥിരം int64_t ne0 = tensor->ne[0];
    സ്ഥിരം size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        int64_t row_താഴെയുള്ള, row_മുകളിലെ;
        get_row_split(&row_താഴെയുള്ള, &row_മുകളിലെ, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_മുകളിലെ - row_താഴെയുള്ള;
        എങ്കിൽ (nrows_split == 0) {
            തുടരുക;
        }

        സ്ഥിരം size_t offset_split = row_താഴെയുള്ള*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        സ്ഥിരം size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        എങ്കിൽ (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        അക്ഷരം * buf_host = (അക്ഷരം *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        സ്ഥിരം queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

സ്ഥിതം ഘടന ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ ശൂന്യ,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ ശൂന്യ,
    /* .get_tensor_2d   = */ ശൂന്യ,
    /* .cpy_tensor      = */ ശൂന്യ,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ ശൂന്യ,
};

// sycl split buffer type

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    തിരികെ GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

സ്ഥിതം തർക്കം ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   തിരികെ buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = പുതിയ ggml_backend_sycl_split_buffer_context();

    തിരികെ ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

സ്ഥിതം size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    തിരികെ SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

സ്ഥിതം size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, സ്ഥിരം ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    സ്ഥിരം int64_t ne0 = tensor->ne[0];

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        int64_t row_താഴെയുള്ള, row_മുകളിലെ;
        get_row_split(&row_താഴെയുള്ള, &row_മുകളിലെ, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_മുകളിലെ - row_താഴെയുള്ള;
        എങ്കിൽ (nrows_split == 0) {
            തുടരുക;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        എങ്കിൽ (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    തിരികെ total_size;
}

സ്ഥിതം തർക്കം ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    തിരികെ അസത്യം;

    GGML_UNUSED(buft);
}

സ്ഥിതം ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ശൂന്യ,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ ശൂന്യ, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ശൂന്യ,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(പൂർണ്ണസംഖ്യ main_device, സ്ഥിരം ദശാംശം * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    സ്ഥിതം std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    സ്ഥിതം std::map<std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES>, ഘടന ggml_backend_buffer_type> buft_map;

    std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    തർക്കം all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](ദശാംശം x) { തിരികെ x == 0.0f; });
    എങ്കിൽ (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } അല്ലെങ്കിൽ {
        ദശാംശം split_sum = 0.0f;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    എങ്കിൽ (it != buft_map.end()) {
        തിരികെ &it->second;
    }

    ഘടന ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ പുതിയ ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    തിരികെ &result.first->second;
}

// host buffer type

ഘടന ggml_backend_sycl_device_context {
    പൂർണ്ണസംഖ്യ device;
    std::string name;
    std::string description;
    പൂർണ്ണസംഖ്യ op_offload_min_batch_size;
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    തിരികെ GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    തിരികെ static_cast<സ്ഥിരം ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
സ്ഥിതം ശൂന്യം * ggml_backend_sycl_host_malloc(പൂർണ്ണസംഖ്യ device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    ശൂന്യം * ptr = nullptr;
    ശ്രമിക്കുക {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } പിടിക്കുക (...) {
        ptr = nullptr;
    }
    എങ്കിൽ (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    തിരികെ ptr;
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    എങ്കിൽ (buffer->context == nullptr) {
        തിരികെ;
    }
    എങ്കിൽ (g_ggml_sycl_enable_host_pinned_mem) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(buffer->context, q)));
    } അല്ലെങ്കിൽ {
        free_aligned_mem_host((ശൂന്യം *) buffer->context);
    }
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ശൂന്യം * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    എങ്കിൽ (ptr == nullptr) {
        // fallback to cpu buffer
        തിരികെ ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    തിരികെ buffer;
}

സ്ഥിതം size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    എങ്കിൽ (g_ggml_sycl_enable_host_pinned_mem) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        എങ്കിൽ (g_ggml_sycl_host_pinned_mem_2g) {
            തിരികെ std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } അല്ലെങ്കിൽ {
            തിരികെ max_alloc_size;
        }
    } അല്ലെങ്കിൽ {
        തിരികെ SIZE_MAX;
    }
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(പൂർണ്ണസംഖ്യ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    സ്ഥിതം std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_എണ്ണം());
        വേണ്ടി (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ ശൂന്യ,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ ശൂന്യ,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        തിരികെ bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (പൂർണ്ണസംഖ്യ) buffer_types_host.size());

    തിരികെ &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    തിരികെ ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
ഘടന ggml_sycl_pool_leg : പൊതു ggml_sycl_pool {
    സ്ഥിതം സ്ഥിരം പൂർണ്ണസംഖ്യ MAX_SYCL_BUFFERS = 256;

    പൂർണ്ണസംഖ്യ device;
    queue_ptr qptr;
    ഘടന ggml_sycl_buffer {
        ശൂന്യം * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, പൂർണ്ണസംഖ്യ device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        പൂർണ്ണസംഖ്യ    n_cached    = 0;
        size_t bytes_cached = 0;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            എങ്കിൽ (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        സ്ഥിരം auto slots = format_slots_in_alloc_order();
        എങ്കിൽ (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            എങ്കിൽ (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() സ്ഥിരം {
        std::string വരി;
        അക്ഷരം buf[32];
        തർക്കം first = സത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            എങ്കിൽ (buffer_pool[i].ptr == nullptr) {
                തുടരുക;
            }
            എങ്കിൽ (!first) {
                വരി += '/';
            }
            first = അസത്യം;
            snprintf(buf, വലുപ്പം(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            വരി += buf;
        }
        തിരികെ വരി;
    }
#endif

    ശൂന്യം * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        പൂർണ്ണസംഖ്യ nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        പൂർണ്ണസംഖ്യ ibest = -1;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            എങ്കിൽ (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                എങ്കിൽ (b.size > max_size) max_size = b.size;
#endif
                എങ്കിൽ (b.size >= size) {
                    size_t diff = b.size - size;
                    എങ്കിൽ (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        എങ്കിൽ (!best_diff) {
                            ശൂന്യം * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            തിരികെ ptr;
                        }
                    }
                }
            }
        }
        എങ്കിൽ (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            ശൂന്യം * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            തിരികെ ptr;
        }
        ശൂന്യം * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ശൂന്യം *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        എങ്കിൽ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            തിരികെ nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        തിരികെ ptr;
    }

    ശൂന്യം സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size) override {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            എങ്കിൽ (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                തിരികെ;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#എങ്കിൽ defined(GGML_SYCL_SUPPORT_VMM)
ഘടന ggml_sycl_pool_vmm : പൊതു ggml_sycl_pool {
    സ്ഥിതം സ്ഥിരം size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    പൂർണ്ണസംഖ്യ           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    ഘടന mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        ശൂന്യം * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, പൂർണ്ണസംഖ്യ device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        എങ്കിൽ (pool_addr == 0) {
            തിരികെ;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        വേണ്ടി (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    ശൂന്യം * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        എങ്കിൽ (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            എങ്കിൽ (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            എങ്കിൽ (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            ശൂന്യം * map_ptr = nullptr;
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

        ശൂന്യം * ptr = reinterpret_cast<ശൂന്യം *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        തിരികെ ptr;
    }

    ശൂന്യം സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<ശൂന്യം *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

ഘടന ggml_sycl_pool_host : പൊതു ggml_sycl_pool {
    queue_ptr qptr;
    പൂർണ്ണസംഖ്യ       device;

    ഘടന ggml_sycl_buffer {
        ശൂന്യം * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;
    പൂർണ്ണസംഖ്യ                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, പൂർണ്ണസംഖ്യ device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            എങ്കിൽ (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    ശൂന്യം * alloc(size_t size, size_t * actual_size) override {
        എങ്കിൽ (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            ശൂന്യം * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ശൂന്യം *) sycl::malloc_host(size, *qptr)));
            എങ്കിൽ (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                തിരികെ nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            തിരികെ ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        എങ്കിൽ (b.ptr != nullptr) {
            എങ്കിൽ (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                ശൂന്യം * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                തിരികെ ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        ശൂന്യം * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ശൂന്യം *) sycl::malloc_host(size, *qptr)));
        എങ്കിൽ (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            തിരികെ nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        തിരികെ ptr;
    }

    ശൂന്യം സ്വതന്ത്രം(ശൂന്യം * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            എങ്കിൽ (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                തിരികെ;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::സ്വതന്ത്രം(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, പൂർണ്ണസംഖ്യ device) {
    // return pool for the host to speed up memory management
    തിരികെ std::unique_ptr<ggml_sycl_pool>(പുതിയ ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, പൂർണ്ണസംഖ്യ device) {
#എങ്കിൽ defined(GGML_SYCL_SUPPORT_VMM)
    എങ്കിൽ (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        തിരികെ std::unique_ptr<ggml_sycl_pool>(പുതിയ ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    തിരികെ std::unique_ptr<ggml_sycl_pool>(പുതിയ ggml_sycl_pool_leg(qptr, device));
}

/// kernels
തരനിർവചനം ശൂന്യം (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം queue_ptr &stream);



സ്ഥിതം ശൂന്യം mul_mat_p021_f16_f32(
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ദശാംശം * __restrict__ y, ദശാംശം * __restrict__ dst,
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_y,
    സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം sycl::half *x = (സ്ഥിരം sycl::half *)vx;

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ channel_x = channel / (nchannels_y / nchannels_x);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_y = ncols_x;
    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst = nrows_x;
    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst = row_x;

    ദശാംശം tmp = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col_x = col_x0 + item_ct1.get_local_id(2);

        എങ്കിൽ (col_x >= ncols_x) {
            നിർത്തുക;
        }

        // x is transposed and permuted
        സ്ഥിരം പൂർണ്ണസംഖ്യ ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        സ്ഥിരം ദശാംശം xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ദശാംശം, sycl::rounding_mode::automatic>()[0];

        സ്ഥിരം പൂർണ്ണസംഖ്യ row_y = col_x;


        // y is not transposed but permuted
        സ്ഥിരം പൂർണ്ണസംഖ്യ iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    സ്ഥിരം പൂർണ്ണസംഖ്യ idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

സ്ഥിതം ശൂന്യം mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    സ്ഥിരം ശൂന്യം * __restrict__ vx, സ്ഥിരം ദശാംശം * __restrict__ y, ദശാംശം * __restrict__ dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x,
    സ്ഥിരം പൂർണ്ണസംഖ്യ row_stride_x, സ്ഥിരം പൂർണ്ണസംഖ്യ channel_stride_x,സ്ഥിരം പൂർണ്ണസംഖ്യ channel_stride_y, സ്ഥിരം പൂർണ്ണസംഖ്യ channel_x_ഹാരകം,
    സ്ഥിരം sycl::nd_item<3> &item_ct1) {

    സ്ഥിരം sycl::half *x = (സ്ഥിരം sycl::half *)vx;

    സ്ഥിരം പൂർണ്ണസംഖ്യ row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ channel_x = channel / channel_x_ഹാരകം;

    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_dst = nrows_x;
    സ്ഥിരം പൂർണ്ണസംഖ്യ row_dst   = row_x;

    സ്ഥിരം പൂർണ്ണസംഖ്യ idst = channel*nrows_dst + row_dst;

    ദശാംശം tmp = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ col_x = col_x0 + item_ct1.get_local_id(2);

        എങ്കിൽ (col_x >= ncols_x) {
            നിർത്തുക;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ row_y = col_x;

        സ്ഥിരം പൂർണ്ണസംഖ്യ ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        സ്ഥിരം പൂർണ്ണസംഖ്യ iy = channel * channel_stride_y + row_y;

        സ്ഥിരം ദശാംശം xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ദശാംശം, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    എങ്കിൽ (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

സ്ഥിതം ശൂന്യം k_sum_rows_f32(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                           സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ col = item_ct1.get_local_id(2);

    ദശാംശം sum = 0.0f;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    എങ്കിൽ (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
സ്ഥിതം വരിയിൽ ശൂന്യം ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ സ്ഥിതം ശൂന്യം
k_argsort_f32_i32(സ്ഥിരം ദശാംശം *x, പൂർണ്ണസംഖ്യ *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, പൂർണ്ണസംഖ്യ ncols_pad,
                  സ്ഥിരം പൂർണ്ണസംഖ്യ tasks_per_thread, സ്ഥിരം sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    പൂർണ്ണസംഖ്യ col_index =  item_ct1.get_local_id(2);
    പൂർണ്ണസംഖ്യ row = item_ct1.get_group(1);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < tasks_per_thread; i++) {
        പൂർണ്ണസംഖ്യ col = col_index * tasks_per_thread + i;
        എങ്കിൽ (col >= ncols_pad) {
            തിരികെ;
        }
    }

    സ്ഥിരം ദശാംശം * x_row = x + row * ncols;
    auto dst_row = (പൂർണ്ണസംഖ്യ *)dpct_local;

    // initialize indices
    വേണ്ടി (പൂർണ്ണസംഖ്യ i=0;i<tasks_per_thread;i++){
        പൂർണ്ണസംഖ്യ col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    വേണ്ടി (പൂർണ്ണസംഖ്യ k = 2; k <= ncols_pad; k *= 2) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = k / 2; j > 0; j /= 2) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < tasks_per_thread; i++) {
                പൂർണ്ണസംഖ്യ col = col_index * tasks_per_thread + i;
                പൂർണ്ണസംഖ്യ ixj = col ^ j;
                എങ്കിൽ (ixj > col) {
                    എങ്കിൽ ((col & k) == 0) {
                        എങ്കിൽ (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } അല്ലെങ്കിൽ {
                        എങ്കിൽ (dst_row[ixj] >= ncols ||
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
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < tasks_per_thread; i++) {
        പൂർണ്ണസംഖ്യ col = col_index * tasks_per_thread + i;
        എങ്കിൽ (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

സ്ഥിതം ശൂന്യം diag_mask_inf_f32(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ rows_per_channel, സ്ഥിരം പൂർണ്ണസംഖ്യ n_past,
                              സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    എങ്കിൽ (col >= ncols) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

സ്ഥിതം ശൂന്യം scale_f32(സ്ഥിരം ദശാംശം * x, ദശാംശം * dst, സ്ഥിരം ദശാംശം scale, സ്ഥിരം ദശാംശം bias, സ്ഥിരം പൂർണ്ണസംഖ്യ k,
                      സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    എങ്കിൽ (i >= k) {
        തിരികെ;
    }

    dst[i] = scale * x[i] + bias;
}


സ്ഥിതം ശൂന്യം ggml_mul_mat_p021_f16_f32_sycl(സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y,
                                           ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_x,
                                           സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_y,
                                           queue_ptr stream) {

    സ്ഥിരം sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    സ്ഥിരം sycl::range<3> block_dims(1, 1, WARP_SIZE);
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

സ്ഥിതം ശൂന്യം ggml_mul_mat_vec_nc_f16_f32_sycl(
    സ്ഥിരം ശൂന്യം *vx, സ്ഥിരം ദശാംശം *y, ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x,
    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x, സ്ഥിരം പൂർണ്ണസംഖ്യ row_stride_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_x,
    സ്ഥിരം പൂർണ്ണസംഖ്യ nchannels_y, സ്ഥിരം പൂർണ്ണസംഖ്യ channel_stride_x, സ്ഥിരം പൂർണ്ണസംഖ്യ channel_stride_y, queue_ptr stream) {

    സ്ഥിരം sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    സ്ഥിരം sycl::range<3> block_dims(1, 1, WARP_SIZE);
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



സ്ഥിതം ശൂന്യം scale_f32_sycl(സ്ഥിരം ദശാംശം *x, ദശാംശം *dst, സ്ഥിരം ദശാംശം scale, സ്ഥിരം ദശാംശം bias,
                           സ്ഥിരം പൂർണ്ണസംഖ്യ k, queue_ptr stream) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, അക്കം_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


സ്ഥിതം ശൂന്യം sum_rows_f32_sycl(സ്ഥിരം ദശാംശം *x, ദശാംശം *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                              സ്ഥിരം പൂർണ്ണസംഖ്യ nrows, queue_ptr stream) {
    സ്ഥിരം sycl::range<3> block_dims(1, 1, WARP_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

സ്ഥിതം പൂർണ്ണസംഖ്യ next_power_of_2(പൂർണ്ണസംഖ്യ x) {
    പൂർണ്ണസംഖ്യ n = 1;
    ആകുമ്പോൾ (n < x) {
        n *= 2;
    }
    തിരികെ n;
}

സ്ഥിതം ശൂന്യം init_argsort_indices_padded(
        പൂർണ്ണസംഖ്യ * idx,
        സ്ഥിരം പൂർണ്ണസംഖ്യ nrows,
        സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_pad,
        സ്ഥിരം sycl::nd_item<1> & item_ct1) {
    സ്ഥിരം size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    സ്ഥിരം size_t total = (size_t) nrows * (size_t) ncols_pad;

    എങ്കിൽ (gid >= total) {
        തിരികെ;
    }

    idx[gid] = (പൂർണ്ണസംഖ്യ) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
സ്ഥിതം ശൂന്യം argsort_f32_i32_global_pass(സ്ഥിരം ദശാംശം *            x,
                                        പൂർണ്ണസംഖ്യ *                    idx,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ                ncols,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ                nrows,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ                ncols_pad,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ                j,
                                        സ്ഥിരം പൂർണ്ണസംഖ്യ                k,
                                        സ്ഥിരം sycl::nd_item<1> & item_ct1) {
    സ്ഥിരം size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    സ്ഥിരം size_t total = (size_t) nrows * (size_t) ncols_pad;

    എങ്കിൽ (gid >= total) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = (പൂർണ്ണസംഖ്യ) (gid / (size_t) ncols_pad);
    സ്ഥിരം പൂർണ്ണസംഖ്യ col = (പൂർണ്ണസംഖ്യ) (gid % (size_t) ncols_pad);
    സ്ഥിരം പൂർണ്ണസംഖ്യ ixj = col ^ j;

    എങ്കിൽ (ixj <= col || ixj >= ncols_pad) {
        തിരികെ;
    }

    സ്ഥിരം size_t base  = (size_t) row * (size_t) ncols_pad;
    സ്ഥിരം size_t pos_a = base + (size_t) col;
    സ്ഥിരം size_t pos_b = base + (size_t) ixj;

    സ്ഥിരം പൂർണ്ണസംഖ്യ a = idx[pos_a];
    സ്ഥിരം പൂർണ്ണസംഖ്യ b = idx[pos_b];

    തർക്കം do_swap = അസത്യം;

    എങ്കിൽ ((col & k) == 0) {
        എങ്കിൽ (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = സത്യം;
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = സത്യം;
        }
    }

    എങ്കിൽ (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

സ്ഥിതം ശൂന്യം copy_argsort_indices_unpadded(സ്ഥിരം പൂർണ്ണസംഖ്യ *              idx_padded,
                                          പൂർണ്ണസംഖ്യ *                    dst,
                                          സ്ഥിരം പൂർണ്ണസംഖ്യ                nrows,
                                          സ്ഥിരം പൂർണ്ണസംഖ്യ                ncols,
                                          സ്ഥിരം പൂർണ്ണസംഖ്യ                ncols_pad,
                                          സ്ഥിരം sycl::nd_item<1> & item_ct1) {
    സ്ഥിരം size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    സ്ഥിരം size_t total = (size_t) nrows * (size_t) ncols;

    എങ്കിൽ (gid >= total) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ row = (പൂർണ്ണസംഖ്യ) (gid / (size_t) ncols);
    സ്ഥിരം പൂർണ്ണസംഖ്യ col = (പൂർണ്ണസംഖ്യ) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

സ്ഥിതം ശൂന്യം argsort_f32_i32_sycl(സ്ഥിരം ദശാംശം *x, പൂർണ്ണസംഖ്യ *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                                 സ്ഥിരം പൂർണ്ണസംഖ്യ nrows, ggml_sort_order order,
                                 queue_ptr stream, പൂർണ്ണസംഖ്യ device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_pad = next_power_of_2(ncols);
    സ്ഥിരം size_t shared_mem = (size_t) ncols_pad * വലുപ്പം(പൂർണ്ണസംഖ്യ);
    സ്ഥിരം size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    എങ്കിൽ (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<പൂർണ്ണസംഖ്യ> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        പൂർണ്ണസംഖ്യ *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        സ്ഥിരം size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        സ്ഥിരം size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 2; k <= ncols_pad; k *= 2) {
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = k / 2; j > 0; j /= 2) {
                എങ്കിൽ (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } അല്ലെങ്കിൽ എങ്കിൽ (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } അല്ലെങ്കിൽ {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        സ്ഥിരം size_t total   = (size_t) nrows * (size_t) ncols;
        സ്ഥിരം size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        തിരികെ;
    }

    പൂർണ്ണസംഖ്യ nth = 1;
    പൂർണ്ണസംഖ്യ max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ആകുമ്പോൾ (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    എങ്കിൽ (nth > max_block_size)
        nth = max_block_size;

    സ്ഥിരം പൂർണ്ണസംഖ്യ tasks_per_thread = ncols_pad / nth;

    സ്ഥിരം sycl::range<3> block_dims(1, 1, nth);
    സ്ഥിരം sycl::range<3> block_nums(1, nrows, 1);

    എങ്കിൽ (order == GGML_SORT_ORDER_ASC) {
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
    } അല്ലെങ്കിൽ എങ്കിൽ (order == GGML_SORT_ORDER_DESC) {
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
    } അല്ലെങ്കിൽ {
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
സ്ഥിതം ശൂന്യം മുകൾ_k_scan_merge_f32(
    സ്ഥിരം ദശാംശം *   src_vals,
    സ്ഥിരം int32_t * src_map,
    സ്ഥിരം പൂർണ്ണസംഖ്യ       begin,
    സ്ഥിരം പൂർണ്ണസംഖ്യ       end,
    സ്ഥിരം പൂർണ്ണസംഖ്യ       k,
    സ്ഥിരം പൂർണ്ണസംഖ്യ       block_size,
    ദശാംശം *         shared_vals,
    പൂർണ്ണസംഖ്യ *           shared_idx,
    ദശാംശം *         out_vals,
    int32_t *       out_idx,
    സ്ഥിരം തർക്കം      swap01,
    സ്ഥിരം sycl::nd_item<1> & item_ct1
) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    ദശാംശം * lv = shared_vals + tid;
    പൂർണ്ണസംഖ്യ * li = shared_idx + tid;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    ദശാംശം kth = -FLT_MAX;

    വേണ്ടി (പൂർണ്ണസംഖ്യ col = begin + tid; col < end; col += block_size) {
        ദശാംശം val = src_vals[col];

        എങ്കിൽ (val > kth) {
            പൂർണ്ണസംഖ്യ pos = k - 1;
            ആകുമ്പോൾ (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    എങ്കിൽ (tid != 0) {
        തിരികെ;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    ദശാംശം * fv = shared_vals + (size_t) k * block_size;
    പൂർണ്ണസംഖ്യ * fi = shared_idx + (size_t) k * block_size;

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    ദശാംശം fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    വേണ്ടി (പൂർണ്ണസംഖ്യ t = 0; t < block_size; t++) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < k; i++) {
            ദശാംശം val = shared_vals[i * block_size + t];

            എങ്കിൽ (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                നിർത്തുക;
            }

            പൂർണ്ണസംഖ്യ idx = shared_idx[i * block_size + t];

            പൂർണ്ണസംഖ്യ pos = k - 1;
            ആകുമ്പോൾ (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    എങ്കിൽ (out_vals) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    എങ്കിൽ (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

സ്ഥിതം ശൂന്യം മുകൾ_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ദശാംശം * src,
    int32_t * dst_indices,
    സ്ഥിരം int64_t ncols,
    സ്ഥിരം int64_t nrows,
    സ്ഥിരം പൂർണ്ണസംഖ്യ k,
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
    constexpr പൂർണ്ണസംഖ്യ split_block = 128;
    constexpr പൂർണ്ണസംഖ്യ max_splits  = 128;
    constexpr പൂർണ്ണസംഖ്യ min_cols    = 8192;

    പൂർണ്ണസംഖ്യ nsplit = 1;
    എങ്കിൽ (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        സ്ഥിരം int64_t want = ncols / split_block;
        nsplit = (പൂർണ്ണസംഖ്യ) (want > max_splits ? max_splits : want);
    }

    എങ്കിൽ (nsplit > 1) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ nchunk = (പൂർണ്ണസംഖ്യ) ((ncols + nsplit - 1) / nsplit);
        സ്ഥിരം size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<ദശാംശം>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        ദശാംശം *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        സ്ഥിരം sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ദശാംശം, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ grp = item_ct1.get_group(0);
                    സ്ഥിരം പൂർണ്ണസംഖ്യ row = grp / nsplit;
                    സ്ഥിരം പൂർണ്ണസംഖ്യ part = grp % nsplit;

                    സ്ഥിരം പൂർണ്ണസംഖ്യ begin = part * nchunk;
                    പൂർണ്ണസംഖ്യ end = begin + nchunk;
                    എങ്കിൽ (end > (പൂർണ്ണസംഖ്യ) ncols) {
                        end = (പൂർണ്ണസംഖ്യ) ncols;
                    }

                    മുകൾ_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, അസത്യം, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ദശാംശം, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(0);
                    സ്ഥിരം size_t off = (size_t) row * nsplit * k;

                    മുകൾ_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, സത്യം, item_ct1);
                });
        });

        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ block_size = 128;

    സ്ഥിരം sycl::range<1> block_dims(block_size);
    സ്ഥിരം sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ദശാംശം, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_group(0);

                എങ്കിൽ (row >= nrows) തിരികെ;

                മുകൾ_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (പൂർണ്ണസംഖ്യ) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, സത്യം, item_ct1);
            });
    });
}

സ്ഥിതം ശൂന്യം argmax_f32_i32_sycl(സ്ഥിരം ദശാംശം *x, പൂർണ്ണസംഖ്യ *dst, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols,
                               സ്ഥിരം പൂർണ്ണസംഖ്യ nrows, queue_ptr stream) {
    സ്ഥിരം sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, nrows, 1);
    സ്ഥിരം size_t shared_mem = 256 * വലുപ്പം(ദശാംശം);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ദശാംശം, 1> shared_data(
            sycl::range<1>(shared_mem/വലുപ്പം(ദശാംശം)), cgh);
        sycl::local_accessor<പൂർണ്ണസംഖ്യ, 1> shared_indices(
            sycl::range<1>(shared_mem/വലുപ്പം(ദശാംശം)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                സ്ഥിരം പൂർണ്ണസംഖ്യ tid = item_ct1.get_local_id(2);
                സ്ഥിരം പൂർണ്ണസംഖ്യ row = item_ct1.get_global_id(1);

                ദശാംശം max_val = -INFINITY;
                പൂർണ്ണസംഖ്യ max_idx = -1;

                വേണ്ടി (പൂർണ്ണസംഖ്യ col = tid; col < ncols; col += 256) {
                    ദശാംശം val = x[row * ncols + col];
                    എങ്കിൽ (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                വേണ്ടി (പൂർണ്ണസംഖ്യ stride = 256/2; stride > 0; stride >>= 1) {
                    എങ്കിൽ (tid < stride) {
                        ദശാംശം val1 = shared_data[tid];
                        ദശാംശം val2 = shared_data[tid + stride];
                        എങ്കിൽ (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                എങ്കിൽ (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
സ്ഥിതം ശൂന്യം diag_mask_inf_f32_sycl(സ്ഥിരം ദശാംശം *x, ദശാംശം *dst,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ ncols_x, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows_x,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ rows_per_channel, സ്ഥിരം പൂർണ്ണസംഖ്യ n_past,
                                   queue_ptr stream) {
    സ്ഥിരം sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    സ്ഥിരം പൂർണ്ണസംഖ്യ block_അക്കം_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    സ്ഥിരം sycl::range<3> block_nums(1, block_അക്കം_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

സ്ഥിതം dpct::err0 ggml_sycl_cpy_tensor_2d(ശൂന്യം *dst,
                                          സ്ഥിരം ഘടന ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_താഴെയുള്ള, int64_t i1_മുകളിലെ,
                                          queue_ptr stream) ശ്രമിക്കുക {

    dpct::memcpy_direction kind;
    അക്ഷരം * src_ptr;
    എങ്കിൽ (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (അക്ഷരം *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (അക്ഷരം *) src->data;
    } അല്ലെങ്കിൽ എങ്കിൽ (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_താഴെയുള്ള == 0 && i1_മുകളിലെ == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        പൂർണ്ണസംഖ്യ id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (അക്ഷരം *) extra->data_device[id];
    } അല്ലെങ്കിൽ {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    അക്ഷരം * dst_ptr = (അക്ഷരം *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    സ്ഥിരം എണ്ണൽ ggml_type type = src->type;
    സ്ഥിരം int64_t ts = ggml_type_size(type);
    സ്ഥിരം int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_മുകളിലെ - i1_താഴെയുള്ള;

    സ്ഥിരം അക്ഷരം * x = src_ptr + i1_താഴെയുള്ള*nb1 + i2*nb2 + i3*nb3;
    എങ്കിൽ (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        തിരികെ CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } അല്ലെങ്കിൽ എങ്കിൽ (nb0 == ts) {
        തിരികെ CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } അല്ലെങ്കിൽ {
        വേണ്ടി (int64_t i1 = 0; i1 < i1_diff; i1++) {
            സ്ഥിരം ശൂന്യം * rx = (സ്ഥിരം ശൂന്യം *) ((സ്ഥിരം അക്ഷരം *) x + i1*nb1);
            ശൂന്യം * rd = (ശൂന്യം *) (dst_ptr + i1*ts*ne0/bs);
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
            എങ്കിൽ (r != 0) തിരികെ r;
        }
        തിരികെ 0;
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

വരിയിൽ ശൂന്യം ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    സ്ഥിരം ggml_tensor *src0, സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
    സ്ഥിരം അക്ഷരം *src0_dd_i, സ്ഥിരം ദശാംശം *src1_ddf_i, സ്ഥിരം അക്ഷരം *src1_ddq_i,
    ദശാംശം *dst_dd_i, സ്ഥിരം int64_t row_താഴെയുള്ള, സ്ഥിരം int64_t row_മുകളിലെ,
    സ്ഥിരം int64_t src1_ncols, സ്ഥിരം int64_t src1_padded_row_size,
    സ്ഥിരം queue_ptr &stream) ശ്രമിക്കുക {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    സ്ഥിരം int64_t row_diff = row_മുകളിലെ - row_താഴെയുള്ള;

    പൂർണ്ണസംഖ്യ id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    സ്ഥിരം int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    പൂർണ്ണസംഖ്യ ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    തർക്കം use_fp16 = സത്യം;  // TODO(Yu) SYCL capability check
#അല്ലെങ്കിൽ
    തർക്കം use_fp16 = അസത്യം;
#endif

#എങ്കിൽ GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    എങ്കിൽ (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        എങ്കിൽ (src1->type != GGML_TYPE_BF16) {
            സ്ഥിരം to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } അല്ലെങ്കിൽ {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*വലുപ്പം(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<ദശാംശം>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        തിരികെ;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    തർക്കം src0_reordered = അസത്യം;
    എങ്കിൽ (സ്ഥിരം auto * src0_extra = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    എങ്കിൽ (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        തിരികെ;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    എങ്കിൽ ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        എങ്കിൽ (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            സ്ഥിരം to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        സ്ഥിരം sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (സ്ഥിരം sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        എങ്കിൽ (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            സ്ഥിരം to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        സ്ഥിരം sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (സ്ഥിരം sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#എങ്കിൽ GGML_SYCL_DNNL
        എങ്കിൽ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<ദശാംശം>(), stream);
        }
        അല്ലെങ്കിൽ
#endif
        {
            സ്ഥിരം ദശാംശം alpha = 1.0f;
            സ്ഥിരം ദശാംശം beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } അല്ലെങ്കിൽ {
        ggml_sycl_pool_alloc<ദശാംശം> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<ദശാംശം> src1_ddq_as_f32(ctx.pool());
        എങ്കിൽ (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            സ്ഥിരം to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        എങ്കിൽ (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            സ്ഥിരം to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        സ്ഥിരം ദശാംശം * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (സ്ഥിരം ദശാംശം *) src0_dd_i : src0_ddq_as_f32.get();
        സ്ഥിരം ദശാംശം * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (സ്ഥിരം ദശാംശം *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#എങ്കിൽ GGML_SYCL_DNNL
            സ്ഥിരം int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            സ്ഥിരം തർക്കം use_mkl_direct = gemm_flops < 256 * 256 * 256;
            എങ്കിൽ (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<ദശാംശം>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<ദശാംശം>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<ദശാംശം>(), stream);
            }
            അല്ലെങ്കിൽ
#endif
            {
                സ്ഥിരം ദശാംശം alpha = 1.0f;
                സ്ഥിരം ദശാംശം beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_മൂല്യം(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_മൂല്യം(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

വരിയിൽ ശൂന്യം ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

വരിയിൽ ശൂന്യം ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int64_t ncols = dst->src[0]->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

വരിയിൽ ശൂന്യം ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int64_t ncols = dst->src[0]->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(dst->src[0]);
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


വരിയിൽ ശൂന്യം ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    സ്ഥിരം int64_t ncols = dst->src[0]->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(dst->src[0]);

    എണ്ണൽ ggml_sort_order order = (എണ്ണൽ ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (പൂർണ്ണസംഖ്യ *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_മുകൾ_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    സ്ഥിരം പൂർണ്ണസംഖ്യ k = dst->ne[0];
    സ്ഥിരം int64_t ncols = src0->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    എങ്കിൽ (k <= SYCL_മുകൾ_K_SCAN_MERGE_MAX_K) {
        മുകൾ_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } അല്ലെങ്കിൽ {
        ggml_sycl_മുകൾ_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

വരിയിൽ ശൂന്യം ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    സ്ഥിരം int64_t ncols = dst->src[0]->ne[0];
    സ്ഥിരം int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

വരിയിൽ ശൂന്യം ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം int64_t ne00 = dst->src[0]->ne[0];
    സ്ഥിരം int64_t ne01 = dst->src[0]->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    സ്ഥിരം പൂർണ്ണസംഖ്യ n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

സ്ഥിതം ശൂന്യം tri_f32_sycl(
    സ്ഥിരം ദശാംശം * src,
    ദശാംശം * dst,
    സ്ഥിരം int64_t ne0,
    സ്ഥിരം int64_t ne1,
    സ്ഥിരം int64_t ne2,
    സ്ഥിരം int64_t ne3,
    സ്ഥിരം ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    സ്ഥിരം size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        സ്ഥിരം int64_t idx = (int64_t) tid[0];

        സ്ഥിരം int64_t i0 = idx % ne0;
        സ്ഥിരം int64_t t1 = idx / ne0;
        സ്ഥിരം int64_t i1 = t1 % ne1;

        തർക്കം keep = അസത്യം;
        തിരഞ്ഞെടുപ്പ് (ttype) {
            അവസ്ഥ GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); നിർത്തുക;
            അവസ്ഥ GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); നിർത്തുക;
            അവസ്ഥ GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); നിർത്തുക;
            അവസ്ഥ GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); നിർത്തുക;
            സ്ഥിരസ്ഥിതി: keep = അസത്യം; നിർത്തുക;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

സ്ഥിതം ശൂന്യം ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(src0->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    സ്ഥിരം int64_t ne0 = src0->ne[0];
    സ്ഥിരം int64_t ne1 = src0->ne[1];
    സ്ഥിരം int64_t ne2 = src0->ne[2];
    സ്ഥിരം int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


വരിയിൽ ശൂന്യം ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    സ്ഥിരം ദശാംശം * src0_dd = static_cast<സ്ഥിരം ദശാംശം *>(dst->src[0]->data);
    ദശാംശം *       dst_dd  = static_cast<ദശാംശം *>(dst->data);

    ദശാംശം scale;
    ദശാംശം bias;
    memcpy(&scale, (ദശാംശം *) dst->op_params + 0, വലുപ്പം(ദശാംശം));
    memcpy(&bias,  (ദശാംശം *) dst->op_params + 1, വലുപ്പം(ദശാംശം));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

സ്ഥിതം ശൂന്യം ggml_sycl_set_peer_access(സ്ഥിരം പൂർണ്ണസംഖ്യ n_tokens, [[maybe_unused]] പൂർണ്ണസംഖ്യ main_device) {
    സ്ഥിതം തർക്കം peer_access_enabled = അസത്യം;

    സ്ഥിരം തർക്കം enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    എങ്കിൽ (peer_access_enabled == enable_peer_access) {
        തിരികെ;
    }

#ifdef NDEBUG
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        വേണ്ടി (പൂർണ്ണസംഖ്യ id_other = 0; id_other < ggml_sycl_info().device_എണ്ണം; ++id_other) {
            എങ്കിൽ (i == id_other) {
                തുടരുക;
            }
            എങ്കിൽ (i != main_device && id_other != main_device) {
                തുടരുക;
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

template <template <പൂർണ്ണസംഖ്യ> typename quantize_f>
സ്ഥിതം ശൂന്യം ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor *src0,
                                 സ്ഥിരം ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) ശ്രമിക്കുക {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    സ്ഥിരം int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    സ്ഥിരം int64_t ne0 = dst->ne[0];
    സ്ഥിരം int64_t ne1 = dst->ne[1];

    സ്ഥിരം പൂർണ്ണസംഖ്യ nb2 = dst->nb[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    സ്ഥിരം int64_t i02_ഹാരകം = ne12 / ne02;

    സ്ഥിരം size_t src0_ts = ggml_type_size(src0->type);
    സ്ഥിരം size_t src0_bs = ggml_blck_size(src0->type);
    സ്ഥിരം size_t q8_1_ts = വലുപ്പം(block_q8_1);
    സ്ഥിരം size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    സ്ഥിരം തർക്കം src0_is_contiguous = ggml_is_contiguous(src0);
    സ്ഥിരം തർക്കം src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    സ്ഥിരം തർക്കം split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<ദശാംശം, GGML_SYCL_MAX_DEVICES> tensor_split;
    എങ്കിൽ (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    ഘടന dev_data {
        ggml_sycl_pool_alloc<അക്ഷരം> src0_dd_alloc;
        ggml_sycl_pool_alloc<ദശാംശം> src1_ddf_alloc;
        ggml_sycl_pool_alloc<അക്ഷരം> src1_ddq_alloc;
        ggml_sycl_pool_alloc<ദശാംശം> dst_dd_alloc;

        അക്ഷരം *src0_dd = nullptr;
        ദശാംശം *src1_ddf = nullptr; // float
        അക്ഷരം *src1_ddq = nullptr;  // q8_1
        ദശാംശം *dst_dd = nullptr;

        int64_t row_താഴെയുള്ള;
        int64_t row_മുകളിലെ;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    പൂർണ്ണസംഖ്യ used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        // by default, use all rows
        dev[i].row_താഴെയുള്ള  = 0;
        dev[i].row_മുകളിലെ = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        എങ്കിൽ (split) {
            സ്ഥിരം int64_t rounding = get_row_rounding(src0->type, tensor_split);

            എങ്കിൽ (i != 0) {
                dev[i].row_താഴെയുള്ള  = ne01*tensor_split[i];
                എങ്കിൽ (dev[i].row_താഴെയുള്ള < ne01) {
                    dev[i].row_താഴെയുള്ള -= dev[i].row_താഴെയുള്ള % rounding;
                }
            }

            എങ്കിൽ (i != ggml_sycl_info().device_എണ്ണം - 1) {
                dev[i].row_മുകളിലെ  = ne01*tensor_split[i + 1];
                എങ്കിൽ (dev[i].row_മുകളിലെ < ne01) {
                    dev[i].row_മുകളിലെ -= dev[i].row_മുകളിലെ % rounding;
                }
            }
        }
    }

    constexpr തർക്കം quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
        എങ്കിൽ ((!split && i != ctx.device) || dev[i].row_താഴെയുള്ള == dev[i].row_മുകളിലെ) {
            തുടരുക;
        }

        used_devices++;

        സ്ഥിരം തർക്കം src1_on_device = i == ctx.device;
        സ്ഥിരം തർക്കം  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        എങ്കിൽ (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (അക്ഷരം *) src0_extra->data_device[i] : (അക്ഷരം *) src0->data;
        } അല്ലെങ്കിൽ {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        എങ്കിൽ (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (ദശാംശം *) src1->data;
        } അല്ലെങ്കിൽ {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        എങ്കിൽ constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            എങ്കിൽ (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                ശ്രമിക്കുക {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        എങ്കിൽ (dst_on_device) {
            dev[i].dst_dd = (ദശാംശം *) dst->data;
        } അല്ലെങ്കിൽ {
            സ്ഥിരം size_t size_dst_ddf = split ? (dev[i].row_മുകളിലെ - dev[i].row_താഴെയുള്ള)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    എങ്കിൽ (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    സ്ഥിരം int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    വേണ്ടി (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        സ്ഥിരം int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        സ്ഥിരം int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
            എങ്കിൽ ((!split && i != ctx.device) || dev[i].row_താഴെയുള്ള == dev[i].row_മുകളിലെ) {
                തുടരുക;
            }

            സ്ഥിരം തർക്കം src1_on_device = i == ctx.device;
            സ്ഥിരം തർക്കം  dst_on_device = i == ctx.device;
            സ്ഥിരം int64_t row_diff = dev[i].row_മുകളിലെ - dev[i].row_താഴെയുള്ള;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            എങ്കിൽ (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            വേണ്ടി (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                സ്ഥിരം int64_t i03 = i0 / ne12;
                സ്ഥിരം int64_t i02 = i0 % ne12;

                സ്ഥിരം size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                അക്ഷരം  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ഹാരകം) * (ne01*ne00*src0_ts)/src0_bs;
                ദശാംശം * src1_ddf_i = (ദശാംശം *) ((അക്ഷരം *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                അക്ഷരം  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                ദശാംശം *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                എങ്കിൽ (i == ctx.device) {
                    dst_dd_i += dev[i].row_താഴെയുള്ള; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                എങ്കിൽ (src1_is_contiguous) {
                    എങ്കിൽ (i != ctx.device) {
                        എങ്കിൽ constexpr (quantize_enabled) {
                            അക്ഷരം * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } അല്ലെങ്കിൽ {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            സ്ഥിരം അക്ഷരം * src1_ddf_i_source = (സ്ഥിരം അക്ഷരം *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } അല്ലെങ്കിൽ {
                    എങ്കിൽ (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } അല്ലെങ്കിൽ {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    എങ്കിൽ constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        ശ്രമിക്കുക {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } പിടിക്കുക (സ്ഥിരം sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                എങ്കിൽ (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_ഹാരകം == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ഹാരകം, dev[i].row_താഴെയുള്ള, dev[i].row_മുകളിലെ, stream));
                }
                എങ്കിൽ (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_താഴെയുള്ള, dev[i].row_മുകളിലെ, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                എങ്കിൽ (!dst_on_device) {
                    ശൂന്യം * dst_off_device = dst->data;
                    എങ്കിൽ (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        ദശാംശം * dhf_dst_i = (ദശാംശം *) ((അക്ഷരം *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*വലുപ്പം(ദശാംശം));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_താഴെയുള്ള;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * വലുപ്പം(ദശാംശം), dst_dd_i,
                            row_diff * വലുപ്പം(ദശാംശം), row_diff * വലുപ്പം(ദശാംശം),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } അല്ലെങ്കിൽ {
                        ദശാംശം * dhf_dst_i = (ദശാംശം *) ((അക്ഷരം *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*വലുപ്പം(ദശാംശം));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * വലുപ്പം(ദശാംശം)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                എങ്കിൽ (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    എങ്കിൽ (split && ggml_sycl_info().device_എണ്ണം > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; ++i) {
            എങ്കിൽ (dev[i].row_താഴെയുള്ള == dev[i].row_മുകളിലെ) {
                തുടരുക;
            }
            വേണ്ടി (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor *src0,
                                       സ്ഥിരം ggml_tensor *src1,
                                       ggml_tensor *dst) ശ്രമിക്കുക {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];

    സ്ഥിരം int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ശൂന്യം  * src0_ddq = src0->data;
    ദശാംശം * src1_ddf = (ദശാംശം *) src1->data;
    ദശാംശം * dst_ddf  = (ദശാംശം *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor *src0,
                                     സ്ഥിരം ggml_tensor *src1,
                                     ggml_tensor *dst) ശ്രമിക്കുക {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    സ്ഥിരം int64_t ne00 = src0->ne[0];
    സ്ഥിരം int64_t ne01 = src0->ne[1];
    സ്ഥിരം int64_t ne02 = src0->ne[2];

    സ്ഥിരം int64_t nb01 = src0->nb[1];
    സ്ഥിരം int64_t nb02 = src0->nb[2];

    സ്ഥിരം int64_t ne12 = src1->ne[2];
    സ്ഥിരം int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ശൂന്യം  * src0_ddq = src0->data;
    ദശാംശം * src1_ddf = (ദശാംശം *) src1->data;
    ദശാംശം * dst_ddf  = (ദശാംശം *) dst->data;

    സ്ഥിരം int64_t row_stride_x = nb01 / വലുപ്പം(sycl::half);
    സ്ഥിരം int64_t channel_stride_x = nb02 / വലുപ്പം(sycl::half);
    സ്ഥിരം int64_t channel_stride_y = nb11 / വലുപ്പം(ദശാംശം);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം k_compute_batched_ptrs(സ്ഥിരം sycl::half * src0_as_f16, സ്ഥിരം sycl::half * src1_as_f16, ശൂന്യം * dst,
                                   സ്ഥിരം ശൂന്യം ** ptrs_src, ശൂന്യം ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, സ്ഥിരം sycl::nd_item<3> & item_ct1) {
    സ്ഥിരം int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    സ്ഥിരം int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    എങ്കിൽ (i13 >= ne13 || i12 >= ne12) {
        തിരികെ;
    }

    സ്ഥിരം int64_t i03 = i13 / r3;
    സ്ഥിരം int64_t i02 = i12 / r2;

    സ്ഥിരം uint8_t * src0_bytes = reinterpret_cast<സ്ഥിരം uint8_t *>(src0_as_f16);
    സ്ഥിരം uint8_t * src1_bytes = reinterpret_cast<സ്ഥിരം uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

സ്ഥിതം ശൂന്യം ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src0,
                                           സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst) ശ്രമിക്കുക {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    സ്ഥിരം sycl::half * src0_f16 = static_cast<സ്ഥിരം sycl::half *>(src0->data);
    ദശാംശം *            dst_ddf  = static_cast<ദശാംശം *>(dst->data);

    സ്ഥിരം sycl::half * src1_f16       = static_cast<സ്ഥിരം sycl::half *>(src1->data);
#എങ്കിൽ GGML_SYCL_DNNL
    സ്ഥിരം size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    സ്ഥിരം size_t       type_size_src1 = ggml_type_size(src1->type);

    തർക്കം is_src0_cont_2 = ggml_is_contiguous_2(src0);
    തർക്കം is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    എങ്കിൽ (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#എങ്കിൽ GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        പൂർണ്ണസംഖ്യ last_dim=0;
        പൂർണ്ണസംഖ്യ last_str=0;
        size_t largest_str=0;
        വേണ്ടി(പൂർണ്ണസംഖ്യ i = 0; i< 4; i++){
            // last stride is always the largest
            എങ്കിൽ(src1->nb[i] == largest_str){
                എങ്കിൽ(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            എങ്കിൽ(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        സ്ഥിരം int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        സ്ഥിരം to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# അല്ലെങ്കിൽ
        സ്ഥിരം int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        സ്ഥിരം to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = സത്യം;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    സ്ഥിരം ദശാംശം alpha_f32 = 1.0f;
    സ്ഥിരം ദശാംശം beta_f32  = 0.0f;

    സ്ഥിരം ശൂന്യം * alpha = &alpha_f32;
    സ്ഥിരം ശൂന്യം * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    സ്ഥിരം int64_t r2 = ne12 / ne02;
    സ്ഥിരം int64_t r3 = ne13 / ne03;

#എങ്കിൽ GGML_SYCL_DNNL
    എങ്കിൽ (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](സ്ഥിരം sycl::half *src0,
                                                സ്ഥിരം sycl::half *src1, ദശാംശം *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                തർക്കം supported_broadcast = batchb == batcha ? സത്യം
                        : batchb == 1 || batcha == 1        ? സത്യം
                                                            : അസത്യം;
                എങ്കിൽ (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<ദശാംശം>(), queue, batcha, batchb);
                } അല്ലെങ്കിൽ {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    എങ്കിൽ (batches0 > batches1) {
                        int64_t അക്കം_mul_mats = batches1;
                        int64_t sub_batch = batches0 / അക്കം_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        വേണ്ടി (int64_t i0 = 0; i0 < അക്കം_mul_mats; i0++) {
                            സ്ഥിരം sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            സ്ഥിരം sycl::half *src1_shifted = src1 + (sb2 * i0);
                            ദശാംശം *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<ദശാംശം>(),
                                    queue, sub_batch, 1);
                        }
                    } അല്ലെങ്കിൽ {
                        int64_t അക്കം_mul_mats = batches0;
                        int64_t sub_batch = batches1 / അക്കം_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        വേണ്ടി (int64_t i1 = 0; i1 < അക്കം_mul_mats; i1++) {
                            സ്ഥിരം sycl::half *src0_shifted = src0 + (sa2 * i1);
                            സ്ഥിരം sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            ദശാംശം *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<ദശാംശം>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            സ്ഥിരം തർക്കം cont_batches_dim2_a = nb02 * ne02 == nb03;
            സ്ഥിരം തർക്കം cont_batches_dim2_b = nb12 * ne12 == nb13;
            സ്ഥിരം തർക്കം cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            സ്ഥിരം തർക്കം cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            എങ്കിൽ (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / വലുപ്പം(ദശാംശം));
            } അല്ലെങ്കിൽ എങ്കിൽ (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / വലുപ്പം(ദശാംശം));
            } അല്ലെങ്കിൽ {
                വേണ്ടി (int64_t b_a = 0; b_a < ne03; b_a++) {
                    സ്ഥിരം sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    സ്ഥിരം sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    ദശാംശം *dst_shifted = dst_ddf + (nb3 * b_a / വലുപ്പം(ദശാംശം));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / വലുപ്പം(ദശാംശം));
                }
            }

    }
    അല്ലെങ്കിൽ
#endif
    {
        എങ്കിൽ (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            സ്ഥിരം int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            സ്ഥിരം int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } അല്ലെങ്കിൽ {
            സ്ഥിരം പൂർണ്ണസംഖ്യ ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<സ്ഥിരം ശൂന്യം *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<ശൂന്യം *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<ദശാംശം>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                സ്ഥിരം ശൂന്യം ** ptrs_src_get = ptrs_src.get();
                ശൂന്യം **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * വലുപ്പം(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * വലുപ്പം(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (സ്ഥിരം ശൂന്യം **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (സ്ഥിരം ശൂന്യം **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (ശൂന്യം **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} പിടിക്കുക (സ്ഥിരം sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

എണ്ണൽ വർഗ്ഗം mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

വരിയിൽ തർക്കം ggml_sycl_supports_mmq(എണ്ണൽ ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    തിരികെ അസത്യം;
}

വരിയിൽ തർക്കം ggml_sycl_supports_reorder_mul_mat_sycl(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q8_0:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
            തിരികെ !g_ggml_sycl_prioritize_dmmv;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

വരിയിൽ തർക്കം ggml_sycl_supports_reorder_dmmv(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

വരിയിൽ തർക്കം ggml_sycl_supports_reorder_mmvq(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
        അവസ്ഥ GGML_TYPE_IQ3_S:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

സ്ഥിതം തർക്കം ggml_sycl_supports_reorder_esimd(എണ്ണൽ ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_Q8_0:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
#അല്ലെങ്കിൽ
    GGML_UNUSED(type);
    തിരികെ അസത്യം;
#endif
}

സ്ഥിതം തർക്കം ggml_sycl_supports_dmmv(എണ്ണൽ ggml_type type) {
    തിരഞ്ഞെടുപ്പ് (type) {
        അവസ്ഥ GGML_TYPE_Q1_0:
        അവസ്ഥ GGML_TYPE_Q4_0:
        അവസ്ഥ GGML_TYPE_Q4_1:
        അവസ്ഥ GGML_TYPE_Q5_0:
        അവസ്ഥ GGML_TYPE_Q5_1:
        അവസ്ഥ GGML_TYPE_Q8_0:
        അവസ്ഥ GGML_TYPE_Q2_K:
        അവസ്ഥ GGML_TYPE_Q3_K:
        അവസ്ഥ GGML_TYPE_Q4_K:
        അവസ്ഥ GGML_TYPE_Q5_K:
        അവസ്ഥ GGML_TYPE_Q6_K:
        അവസ്ഥ GGML_TYPE_F16:
        അവസ്ഥ GGML_TYPE_BF16:
            തിരികെ സത്യം;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
സ്ഥിതം വരിയിൽ ശൂന്യം * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    തർക്കം use_async = g_ggml_sycl_use_async_mem_op;
#എങ്കിൽ defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    എങ്കിൽ (use_async) {
        ശൂന്യം * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        തിരികെ ptr;
    }
#അല്ലെങ്കിൽ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    തിരികെ ggml_sycl_malloc_device(size, *stream);
}

സ്ഥിതം വരിയിൽ ശൂന്യം sycl_ext_free(dpct::queue_ptr stream, ശൂന്യം * ptr) {
    തർക്കം use_async = g_ggml_sycl_use_async_mem_op;
#എങ്കിൽ defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    എങ്കിൽ (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        തിരികെ;
    }
#അല്ലെങ്കിൽ
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
ഘടന sycl_reorder_temp_buffer {
    ശൂന്യം *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        എങ്കിൽ (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            എങ്കിൽ (ptr) {
                host_fallback = സത്യം;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        എങ്കിൽ (!ptr) {
            തിരികെ;
        }
        എങ്കിൽ (host_fallback) {
            sycl::സ്വതന്ത്രം(ptr, *stream);
        } അല്ലെങ്കിൽ {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator തർക്കം() സ്ഥിരം { തിരികെ ptr != nullptr; }

    sycl_reorder_temp_buffer(സ്ഥിരം sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(സ്ഥിരം sycl_reorder_temp_buffer &) = delete;

സ്വകാര്യ:
    തർക്കം host_fallback = അസത്യം;
};

സ്ഥിതം തർക്കം reorder_qw_q4_0(uint8_t * data_device, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % വലുപ്പം(block_q4_0) == 0));
    GGML_ASSERT((offset % വലുപ്പം(block_q4_0) == 0));
    പൂർണ്ണസംഖ്യ offset_blks = offset / വലുപ്പം(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / വലുപ്പം(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            സ്ഥിരം block_q4_0* x = (സ്ഥിരം block_q4_0*)tmp_buf;
            സ്ഥിരം പൂർണ്ണസംഖ്യ ib = i;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q8_0(uint8_t * data_device, സ്ഥിരം പൂർണ്ണസംഖ്യ ncols, സ്ഥിരം പൂർണ്ണസംഖ്യ nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % വലുപ്പം(block_q8_0) == 0));
    GGML_ASSERT((offset % വലുപ്പം(block_q8_0) == 0));
    പൂർണ്ണസംഖ്യ offset_blks = offset / വലുപ്പം(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / വലുപ്പം(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            സ്ഥിരം block_q8_0* x = (സ്ഥിരം block_q8_0*)tmp_buf;
            സ്ഥിരം പൂർണ്ണസംഖ്യ ib = i;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
സ്ഥിതം തർക്കം reorder_qw_q8_0_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % വലുപ്പം(block_q8_0) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    blocks_per_expert = (പൂർണ്ണസംഖ്യ) (expert_bytes / വലുപ്പം(block_q8_0));
    സ്ഥിരം size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ total_blocks = blocks_per_expert * (പൂർണ്ണസംഖ്യ) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ          gb   = gb_;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          e    = gb / blocks_per_expert;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib   = gb % blocks_per_expert;
        സ്ഥിരം block_q8_0 * x    = (സ്ഥിരം block_q8_0 *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr = base;
        auto * d_ptr  = (sycl::half *) (qs_ptr + (size_t) blocks_per_expert * QK8_0);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_q4_K) == 0);
    GGML_ASSERT(offset % വലുപ്പം(block_q4_K) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_q4_K * x  = (സ്ഥിരം block_q4_K *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

// Reorder each expert slice into a self-contained SoA layout.
സ്ഥിതം തർക്കം reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % വലുപ്പം(block_q4_K) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    blocks_per_expert = (പൂർണ്ണസംഖ്യ) (expert_bytes / വലുപ്പം(block_q4_K));
    സ്ഥിരം size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ total_blocks = blocks_per_expert * (പൂർണ്ണസംഖ്യ) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ          gb   = gb_;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          e    = gb / blocks_per_expert;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib   = gb % blocks_per_expert;
        സ്ഥിരം block_q4_K * x    = (സ്ഥിരം block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
സ്ഥിതം തർക്കം reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % വലുപ്പം(block_q5_K) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    blocks_per_expert = (പൂർണ്ണസംഖ്യ) (expert_bytes / വലുപ്പം(block_q5_K));
    സ്ഥിരം size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ total_blocks = blocks_per_expert * (പൂർണ്ണസംഖ്യ) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ          gb   = gb_;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          e    = gb / blocks_per_expert;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib   = gb % blocks_per_expert;
        സ്ഥിരം block_q5_K * x    = (സ്ഥിരം block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
സ്ഥിതം തർക്കം reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % വലുപ്പം(block_q6_K) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ    blocks_per_expert = (പൂർണ്ണസംഖ്യ) (expert_bytes / വലുപ്പം(block_q6_K));
    സ്ഥിരം size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ total_blocks = blocks_per_expert * (പൂർണ്ണസംഖ്യ) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ          gb   = gb_;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          e    = gb / blocks_per_expert;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib   = gb % blocks_per_expert;
        സ്ഥിരം block_q6_K * x    = (സ്ഥിരം block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
സ്ഥിതം തർക്കം reorder_qw_mxfp4_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % വലുപ്പം(block_mxfp4) == 0);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     blocks_per_expert = (പൂർണ്ണസംഖ്യ) (expert_bytes / വലുപ്പം(block_mxfp4));
    സ്ഥിരം size_t  max_chunk_bytes   = 32u << 20;
    സ്ഥിരം int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    വേണ്ടി (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        സ്ഥിരം int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ total_blocks = blocks_per_expert * (പൂർണ്ണസംഖ്യ) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ           gb   = gb_;
            സ്ഥിരം പൂർണ്ണസംഖ്യ           e    = gb / blocks_per_expert;
            സ്ഥിരം പൂർണ്ണസംഖ്യ           ib   = gb % blocks_per_expert;
            സ്ഥിരം block_mxfp4 * x    = (സ്ഥിരം block_mxfp4 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *           base = chunk + (size_t) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) blocks_per_expert;

            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_q2_K) == 0);
    GGML_ASSERT(offset % വലുപ്പം(block_q2_K) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_q2_K * x  = (സ്ഥിരം block_q2_K *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_q3_K) == 0);
    GGML_ASSERT(offset % വലുപ്പം(block_q3_K) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_q3_K * x  = (സ്ഥിരം block_q3_K *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_q5_K) == 0);
    GGML_ASSERT(offset % വലുപ്പം(block_q5_K) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_q5_K * x  = (സ്ഥിരം block_q5_K *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_q6_K) == 0);
    GGML_ASSERT(offset % വലുപ്പം(block_q6_K) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_q6_K * x  = (സ്ഥിരം block_q6_K *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ          ib = i;

        സ്ഥിരം uint8_t * ql              = x[ib].ql;
        സ്ഥിരം uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_iq3_s(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_iq3_s) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_iq3_s);
    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr       = data_device;
    auto * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    auto * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    auto * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_iq3_s * x = (സ്ഥിരം block_iq3_s *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (വലുപ്പം(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[വലുപ്പം(ggml_half) + j] = x[ib].scales[j];
        }
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw_iq3_xxs(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % വലുപ്പം(block_iq3_xxs) == 0);

    സ്ഥിരം പൂർണ്ണസംഖ്യ nblocks = size / വലുപ്പം(block_iq3_xxs);
    sycl_reorder_temp_buffer tmp(stream, size);
    എങ്കിൽ (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        തിരികെ അസത്യം;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr = data_device;
    auto * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        സ്ഥിരം block_iq3_xxs * x = (സ്ഥിരം block_iq3_xxs *) tmp_buf;
        സ്ഥിരം പൂർണ്ണസംഖ്യ ib = i;

        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    തിരികെ സത്യം;
}

സ്ഥിതം തർക്കം reorder_qw(സ്ഥിരം ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    എങ്കിൽ (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        തിരഞ്ഞെടുപ്പ് (src0->type) {
            അവസ്ഥ GGML_TYPE_Q8_0:
                തിരികെ reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            അവസ്ഥ GGML_TYPE_Q4_K:
                തിരികെ reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            അവസ്ഥ GGML_TYPE_Q5_K:
                തിരികെ reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            അവസ്ഥ GGML_TYPE_Q6_K:
                തിരികെ reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            അവസ്ഥ GGML_TYPE_MXFP4:
                തിരികെ reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            സ്ഥിരസ്ഥിതി:
                തിരികെ അസത്യം;
        }
    }

    തിരഞ്ഞെടുപ്പ് (src0->type) {
        അവസ്ഥ GGML_TYPE_Q4_0:
            തിരികെ reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q8_0:
            തിരികെ reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q2_K:
            തിരികെ reorder_qw_q2_k(data_device, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q3_K:
            തിരികെ reorder_qw_q3_k(data_device, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q4_K:
            തിരികെ reorder_qw_q4_k(data_device, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q5_K:
            തിരികെ reorder_qw_q5_k(data_device, size, 0, stream);
        അവസ്ഥ GGML_TYPE_Q6_K:
            തിരികെ reorder_qw_q6_k(data_device, size, 0, stream);
        അവസ്ഥ GGML_TYPE_IQ3_XXS:
            തിരികെ reorder_qw_iq3_xxs(data_device, size, stream);
        അവസ്ഥ GGML_TYPE_IQ3_S:
            തിരികെ reorder_qw_iq3_s(data_device, size, stream);
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }
}

സ്ഥിതം തർക്കം should_reorder_tensor(ggml_backend_sycl_context& ctx, സ്ഥിരം ggml_tensor * dst) {
    തിരികെ g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

സ്ഥിതം ശൂന്യം opt_for_reorder(ggml_backend_sycl_context * ctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    എങ്കിൽ (!should_reorder_tensor(*ctx, dst)) {
        തിരികെ;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    എങ്കിൽ (!extra || extra->optimized_feature.reorder) {
        തിരികെ;  // Skip permutations and already reordered tensors
    }

    തിരഞ്ഞെടുപ്പ് (mm_algorithm) {
        അവസ്ഥ mul_mat_algo::DMMV:
            എങ്കിൽ (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                തിരികെ;
            }
            നിർത്തുക;
        അവസ്ഥ mul_mat_algo::MMVQ:
            എങ്കിൽ (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                തിരികെ;
            }
            നിർത്തുക;
        അവസ്ഥ mul_mat_algo::MUL_MAT_SYCL:
            എങ്കിൽ (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                തിരികെ;
            }
            നിർത്തുക;
    }

    എങ്കിൽ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = സത്യം;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
സ്ഥിതം ശൂന്യം opt_for_reorder_id(ggml_backend_sycl_context * ctx, സ്ഥിരം ggml_tensor * src0) {
    എങ്കിൽ (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        തിരികെ;
    }
    എങ്കിൽ (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        തിരികെ;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    എങ്കിൽ (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        തിരികെ;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    എങ്കിൽ (!extra || extra->optimized_feature.reorder) {
        തിരികെ;
    }
    എങ്കിൽ (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = സത്യം;
    }
}


സ്ഥിതം തർക്കം can_use_dequantize_mul_mat_vec(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    സ്ഥിരം int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    തിരികെ ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

സ്ഥിതം തർക്കം can_use_mul_mat_vec_q(സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst) {
    തിരികെ ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

സ്ഥിതം ശൂന്യം ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src0, സ്ഥിരം ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    എങ്കിൽ (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        തിരികെ;
    }

    സ്ഥിരം തർക്കം split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    എങ്കിൽ (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        വേണ്ടി (പൂർണ്ണസംഖ്യ id = 0; id < ggml_sycl_info().device_എണ്ണം; ++id) {
            // skip devices that are not going to do any work:
            എങ്കിൽ (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_എണ്ണം ? tensor_split[id + 1] : 1.0f)) {
                തുടരുക;
            }

            എങ്കിൽ (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } അല്ലെങ്കിൽ {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    തർക്കം use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    തർക്കം use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    തർക്കം use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
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

    എങ്കിൽ (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        തർക്കം use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        എങ്കിൽ (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    എങ്കിൽ (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        എങ്കിൽ (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } അല്ലെങ്കിൽ {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } അല്ലെങ്കിൽ എങ്കിൽ (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } അല്ലെങ്കിൽ എങ്കിൽ (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        എങ്കിൽ (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } അല്ലെങ്കിൽ {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } അല്ലെങ്കിൽ {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
സ്ഥിതം തർക്കം ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, സ്ഥിരം ggml_tensor * wu,
                                             സ്ഥിരം ggml_tensor * wg, സ്ഥിരം ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    സ്ഥിരം auto * extra_u = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wu->extra);
    സ്ഥിരം auto * extra_g = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wg->extra);
    എങ്കിൽ ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        തിരികെ അസത്യം;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    സ്ഥിരം auto plain_type = [](ggml_type t) { തിരികെ t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    എങ്കിൽ (!plain_type(wu->type) || !plain_type(wg->type)) {
        തിരികെ അസത്യം;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    സ്ഥിരം int64_t ne00 = wu->ne[0];
    സ്ഥിരം int64_t ne11 = act->ne[1];

    സ്ഥിരം queue_ptr stream = ctx.stream();
    സ്ഥിരം പൂർണ്ണസംഖ്യ src1_padded_cols = GGML_PAD((പൂർണ്ണസംഖ്യ) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<അക്ഷരം> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * വലുപ്പം(block_q8_1) / QK8_1);
    അക്ഷരം * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((സ്ഥിരം ദശാംശം *) act->data, src1_ddq, (പൂർണ്ണസംഖ്യ) ne00, (പൂർണ്ണസംഖ്യ) ne11,
                                          src1_padded_cols, stream);

    തിരികെ ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (ദശാംശം *) glu->data, (പൂർണ്ണസംഖ്യ) ne00, (പൂർണ്ണസംഖ്യ) wu->ne[1],
                                             (പൂർണ്ണസംഖ്യ) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(പൂർണ്ണസംഖ്യ) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
സ്ഥിതം തർക്കം ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, സ്ഥിരം ggml_tensor * wu,
                                             സ്ഥിരം ggml_tensor * wg, സ്ഥിരം ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    സ്ഥിരം അക്ഷരം * act_begin = (സ്ഥിരം അക്ഷരം *) act->data;
    സ്ഥിരം അക്ഷരം * glu_begin = (സ്ഥിരം അക്ഷരം *) glu->data;
    എങ്കിൽ (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        തിരികെ അസത്യം;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    സ്ഥിരം auto * extra_u = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wu->extra);
    സ്ഥിരം auto * extra_g = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wg->extra);
    എങ്കിൽ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        തിരികെ അസത്യം;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    തിരികെ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (സ്ഥിരം ദശാംശം *) act->data, (ദശാംശം *) glu->data,
                                                              (പൂർണ്ണസംഖ്യ) wu->ne[0], (പൂർണ്ണസംഖ്യ) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
സ്ഥിതം തർക്കം ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx) {
    എങ്കിൽ (!ggml_sycl_can_fuse(cgraph, കണ്ണി_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        തിരികെ അസത്യം;
    }

    ggml_tensor *       glu  = cgraph->nodes[കണ്ണി_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    സ്ഥിരം ggml_tensor * wu   = up->src[0];
    സ്ഥിരം ggml_tensor * wg   = gate->src[0];
    സ്ഥിരം ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    എങ്കിൽ (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        തിരികെ അസത്യം;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    എങ്കിൽ (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        തിരികെ സത്യം;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    എങ്കിൽ (g_ggml_sycl_prioritize_dmmv) {
        തിരികെ അസത്യം;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    സ്ഥിരം തർക്കം reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    എങ്കിൽ (!reorder_pair) {
        തിരികെ ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    എങ്കിൽ (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        തിരികെ അസത്യം;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    സ്ഥിരം auto * extra_u = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wu->extra);
    സ്ഥിരം auto * extra_g = static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(wg->extra);
    എങ്കിൽ (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        തിരികെ അസത്യം;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    സ്ഥിരം int64_t ne00 = wu->ne[0];
    സ്ഥിരം int64_t ne11 = act->ne[1];

    സ്ഥിരം queue_ptr stream           = ctx.stream();
    സ്ഥിരം പൂർണ്ണസംഖ്യ       src1_padded_cols = GGML_PAD((പൂർണ്ണസംഖ്യ) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<അക്ഷരം> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * വലുപ്പം(block_q8_1) / QK8_1);
    അക്ഷരം *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((സ്ഥിരം ദശാംശം *) act->data, src1_ddq, (പൂർണ്ണസംഖ്യ) ne00, (പൂർണ്ണസംഖ്യ) ne11,
                                                          src1_padded_cols, stream);

    തിരികെ ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (ദശാംശം *) glu->data, (പൂർണ്ണസംഖ്യ) ne00, (പൂർണ്ണസംഖ്യ) wu->ne[1], (പൂർണ്ണസംഖ്യ) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (പൂർണ്ണസംഖ്യ) വലുപ്പം(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(പൂർണ്ണസംഖ്യ) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx) {
    സ്ഥിരം ggml_tensor * കണ്ണി = cgraph->nodes[കണ്ണി_idx];
    എങ്കിൽ (ggml_sycl_info().device_എണ്ണം != 1 || കണ്ണി->type != GGML_TYPE_F32 ||
        കണ്ണി->src[0]->type != GGML_TYPE_F32 || കണ്ണി->src[0]->ne[0] >= 1024) {
        തിരികെ 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    പൂർണ്ണസംഖ്യ           count = 0;
    പൂർണ്ണസംഖ്യ           last  = കണ്ണി_idx;
    ദശാംശം         eps0;
    memcpy(&eps0, കണ്ണി->op_params, വലുപ്പം(ദശാംശം));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](സ്ഥിരം ggml_tensor * a, സ്ഥിരം ggml_tensor * b) {
        സ്ഥിരം അക്ഷരം * ab = (സ്ഥിരം അക്ഷരം *) a->data;
        സ്ഥിരം അക്ഷരം * bb = (സ്ഥിരം അക്ഷരം *) b->data;
        തിരികെ ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    വേണ്ടി (പൂർണ്ണസംഖ്യ j = കണ്ണി_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        എങ്കിൽ (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;  // not a launch; cannot break a run of adjacent norms
        }
        എങ്കിൽ (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, കണ്ണി) ||
            !ggml_are_same_shape(nj->src[0], കണ്ണി->src[0])) {
            നിർത്തുക;  // any other launch ends the run
        }
        തർക്കം same_nb = സത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ d = 0; d < GGML_MAX_DIMS; ++d) {
            എങ്കിൽ (nj->nb[d] != കണ്ണി->nb[d] || nj->src[0]->nb[d] != കണ്ണി->src[0]->nb[d]) {
                same_nb = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (!same_nb) {
            നിർത്തുക;  // one nb[] stride set is shared by the whole batch
        }
        ദശാംശം epsj;
        memcpy(&epsj, nj->op_params, വലുപ്പം(ദശാംശം));
        എങ്കിൽ (epsj != eps0) {
            നിർത്തുക;  // eps mismatch ends the run
        }
        തർക്കം indep = സത്യം;
        വേണ്ടി (പൂർണ്ണസംഖ്യ k = 0; k < count; ++k) {
            എങ്കിൽ (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = അസത്യം;
                നിർത്തുക;
            }
        }
        എങ്കിൽ (!indep) {
            നിർത്തുക;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    എങ്കിൽ (count < 2) {
        തിരികെ 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    തിരികെ last - കണ്ണി_idx;
}


__dpct_inline__ സ്ഥിതം ശൂന്യം k_copy_src1_to_contiguous(
    സ്ഥിരം അക്ഷരം *__restrict__ src1_original, അക്ഷരം *__restrict__ src1_contiguous,
    സ്ഥിരം mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    സ്ഥിരം int32_t src1_row = item_ct1.get_group(2);

    സ്ഥിരം int32_t iid1 = row_mapping[src1_row].i2;
    സ്ഥിരം int32_t id   = row_mapping[src1_row].i1;

    സ്ഥിരം int64_t i11 = id % ne11;
    സ്ഥിരം int64_t i12 = iid1;

    സ്ഥിരം ദശാംശം * src1_row_original = (സ്ഥിരം ദശാംശം *)(src1_original + i11*nb11 + i12*nb12);
    ദശാംശം * src1_row_contiguous = (ദശാംശം *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ സ്ഥിതം ശൂന്യം k_copy_dst_from_contiguous(
    അക്ഷരം *__restrict__ dst_original, സ്ഥിരം അക്ഷരം *__restrict__ dst_contiguous,
    സ്ഥിരം mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, സ്ഥിരം sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    സ്ഥിരം int32_t i1 = row_mapping[i].i1;
    സ്ഥിരം int32_t i2 = row_mapping[i].i2;

    സ്ഥിരം ദശാംശം * dst_row_contiguous = (സ്ഥിരം ദശാംശം *)(dst_contiguous + i*nb1);
    ദശാംശം * dst_row_original = (ദശാംശം *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
സ്ഥിതം തർക്കം ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, സ്ഥിരം ggml_tensor * src0,
    സ്ഥിരം ggml_tensor * src1, സ്ഥിരം ggml_tensor * ids, ggml_tensor * dst)
{
    സ്ഥിരം int64_t ne10 = src1->ne[0];
    സ്ഥിരം int64_t ne11 = src1->ne[1];
    സ്ഥിരം int64_t ne12 = src1->ne[2];
    എങ്കിൽ (ne12 != 1) തിരികെ അസത്യം;
    എങ്കിൽ (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) തിരികെ അസത്യം;
    എങ്കിൽ (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) തിരികെ അസത്യം;
    എങ്കിൽ (!ggml_is_contiguous(src1)) തിരികെ അസത്യം;

    സ്ഥിരം int64_t n_ids_per_group = ids->ne[0];
    എങ്കിൽ (ids->ne[1] != 1) തിരികെ അസത്യം;
    എങ്കിൽ (ne11 != 1 && ne11 != n_ids_per_group) തിരികെ അസത്യം;

    സ്ഥിരം queue_ptr stream           = ctx.stream();
    സ്ഥിരം പൂർണ്ണസംഖ്യ       src1_padded_cols = GGML_PAD((പൂർണ്ണസംഖ്യ) ne10, MATRIX_ROW_PADDING);
    സ്ഥിരം പൂർണ്ണസംഖ്യ       n_experts_used   = (പൂർണ്ണസംഖ്യ) n_ids_per_group;
    സ്ഥിരം പൂർണ്ണസംഖ്യ       nrows            = (പൂർണ്ണസംഖ്യ) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    സ്ഥിരം ggml_tensor_extra_gpu * src0_extra =
        static_cast<സ്ഥിരം ggml_tensor_extra_gpu *>(src0->extra);
    സ്ഥിരം തർക്കം use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<അക്ഷരം> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * വലുപ്പം(block_q8_1) / QK8_1);
    അക്ഷരം * src1_ddq = src1_q8_alloc.get();
    എങ്കിൽ (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (സ്ഥിരം ദശാംശം *) src1->data, src1_ddq, (പൂർണ്ണസംഖ്യ) ne10, (പൂർണ്ണസംഖ്യ) ne11,
            src1_padded_cols, stream);
    } അല്ലെങ്കിൽ {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (സ്ഥിരം ദശാംശം *) src1->data, src1_ddq, (പൂർണ്ണസംഖ്യ) ne10, (പൂർണ്ണസംഖ്യ) ne11,
            src1_padded_cols, stream);
    }

    സ്ഥിരം size_t bytes_per_qrow = (size_t) src1_padded_cols * വലുപ്പം(block_q8_1) / QK8_1;
    സ്ഥിരം size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    എങ്കിൽ (use_reorder) {
        തിരികെ ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (സ്ഥിരം int32_t *) ids->data,
            (ദശാംശം *) dst->data, (പൂർണ്ണസംഖ്യ) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    തിരികെ ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (സ്ഥിരം int32_t *) ids->data,
        (ദശാംശം *) dst->data, (പൂർണ്ണസംഖ്യ) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
സ്ഥിതം ശൂന്യം mmid_counting_sort_rows(
        സ്ഥിരം ggml_tensor * ids, സ്ഥിരം അക്ഷരം * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    വേണ്ടി (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        വേണ്ടി (int64_t id = 0; id < n_ids; id++) {
            സ്ഥിരം int32_t row_id_i = *(സ്ഥിരം int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    വേണ്ടി (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    വേണ്ടി (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        വേണ്ടി (int64_t id = 0; id < n_ids; id++) {
            സ്ഥിരം int32_t row_id_i = *(സ്ഥിരം int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            സ്ഥിരം int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

സ്ഥിതം ശൂന്യം ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) ശ്രമിക്കുക {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    സ്ഥിരം ggml_tensor *src0 = dst->src[0];
    സ്ഥിരം ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    സ്ഥിരം ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ദ്വിമാന_OP_LOCALS

    സ്ഥിരം queue_ptr stream = ctx.stream();

    സ്ഥിരം int64_t n_as = ne02;
    സ്ഥിരം int64_t n_ids = ids->ne[0];

    എങ്കിൽ (ne12 == 1) {
        എങ്കിൽ (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            തിരികെ;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    എങ്കിൽ (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<അക്ഷരം> ids_host(ggml_nbytes(ids));
    സ്ഥിരം അക്ഷരം * ids_dev = (സ്ഥിരം അക്ഷരം *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    അക്ഷരം *src0_original = (അക്ഷരം *)src0->data;
    അക്ഷരം *src1_original = (അക്ഷരം *)src1->data;
    അക്ഷരം *dst_original = (അക്ഷരം *)dst->data;

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
    എങ്കിൽ (ne12 == 1) {
        വേണ്ടി (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            വേണ്ടി (int64_t id = 0; id < n_ids; id++) {
                സ്ഥിരം int32_t i02 = *(സ്ഥിരം int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                സ്ഥിരം int64_t i11 = id % ne11;
                സ്ഥിരം int64_t i12 = iid1;

                സ്ഥിരം int64_t i1 = id;
                സ്ഥിരം int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } അല്ലെങ്കിൽ {
        സ്ഥിരം int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<അക്ഷരം> src1_contiguous(ctx.pool(), വലുപ്പം(ദശാംശം)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<അക്ഷരം>  dst_contiguous(ctx.pool(), വലുപ്പം(ദശാംശം)*n_routed_rows*ne0);

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
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*വലുപ്പം(mmid_row_mapping))));

        സ്ഥിരം unsigned പൂർണ്ണസംഖ്യ max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        പ്രതിജ്ഞാ(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned പൂർണ്ണസംഖ്യ)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                അക്ഷരം *__restrict src1_contiguous_get =
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

        തർക്കം grouped = അസത്യം;
        എങ്കിൽ (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == വലുപ്പം(ദശാംശം)*ne10 && nb1 == വലുപ്പം(ദശാംശം)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            എങ്കിൽ (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            സ്ഥിരം തർക്കം src0_reordered =
                src0->extra && ((സ്ഥിരം ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (സ്ഥിരം ദശാംശം *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (ദശാംശം *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        വേണ്ടി (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            സ്ഥിരം int64_t അക്കം_src1_rows = expert_row_counts[i02];

            എങ്കിൽ (അക്കം_src1_rows == 0) {
                തുടരുക;
            }

            സ്ഥിരം int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == വലുപ്പം(ദശാംശം)*ne10);
            GGML_ASSERT(nb1 == വലുപ്പം(ദശാംശം)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = അക്കം_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = അക്കം_src1_rows*nb11;
            src1_row.nb[3] = അക്കം_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = അക്കം_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = അക്കം_src1_rows*nb1;
            dst_row.nb[3] = അക്കം_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned പൂർണ്ണസംഖ്യ)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                സ്ഥിരം അക്ഷരം *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                സ്ഥിരം mmid_row_mapping *__restrict dev_row_mapping_get =
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
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

സ്ഥിതം ശൂന്യം ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


സ്ഥിതം ശൂന്യം ggml_sycl_set_main_device(സ്ഥിരം പൂർണ്ണസംഖ്യ main_device) ശ്രമിക്കുക {
    എങ്കിൽ (dpct::get_current_device_id() == static_cast<unsigned പൂർണ്ണസംഖ്യ> (main_device)) {
        തിരികെ;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    എങ്കിൽ (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം തർക്കം ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ഘടന ggml_tensor * dst) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    എങ്കിൽ (!g_sycl_loaded) തിരികെ അസത്യം;

    എങ്കിൽ (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    തിരഞ്ഞെടുപ്പ് (dst->op) {
        അവസ്ഥ GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_എണ്ണം_EQUAL:
            ggml_sycl_എണ്ണം_equal(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_UNARY:
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(dst)) {
                അവസ്ഥ GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ അസത്യം;
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GLU:
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(dst)) {
                അവസ്ഥ GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    നിർത്തുക;
                അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    നിർത്തുക;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ അസത്യം;
            }
            നിർത്തുക;
        അവസ്ഥ GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT:
            എങ്കിൽ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                തിരികെ അസത്യം;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            എങ്കിൽ (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                തിരികെ അസത്യം;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_മുകൾ_K:
            ggml_sycl_op_മുകൾ_k(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            നിർത്തുക;
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    തിരികെ സത്യം;
} പിടിക്കുക (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API ശൂന്യം ggml_backend_sycl_get_device_description(പൂർണ്ണസംഖ്യ device, അക്ഷരം *description,
                                      size_t description_size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം തർക്കം sycl_get_mem_info(പൂർണ്ണസംഖ്യ device, size_t * സ്വതന്ത്രം, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#അല്ലെങ്കിൽ
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    തർക്കം res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *സ്വതന്ത്രം, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *സ്വതന്ത്രം);
    തിരികെ res;
}

ശൂന്യം ggml_backend_sycl_get_device_memory(പൂർണ്ണസംഖ്യ device, size_t * സ്വതന്ത്രം, size_t * total) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    എങ്കിൽ (!sycl_get_mem_info(device, സ്വതന്ത്രം, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *സ്വതന്ത്രം, *total);
} പിടിക്കുക (സ്ഥിരം sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    തിരികെ sycl_ctx->name.c_str();
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               സ്ഥിരം ശൂന്യം *data, size_t offset,
                                               size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    സ്ഥിരം queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((അക്ഷരം *)tensor->data + offset, data, size)));
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               സ്ഥിരം ggml_tensor *tensor,
                                               ശൂന്യം *data, size_t offset,
                                               size_t size) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    സ്ഥിരം queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (സ്ഥിരം അക്ഷരം *)tensor->data + offset, size)));
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം തർക്കം ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               സ്ഥിരം ggml_tensor *src,
                                               ggml_tensor *dst) ശ്രമിക്കുക {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    തർക്കം is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    എങ്കിൽ (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        സ്ഥിരം queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        തിരികെ സത്യം;
    }

    തിരികെ അസത്യം;
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_synchronize(ggml_backend_t backend) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    സ്ഥിരം queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം തർക്കം ggml_sycl_is_view_or_noop(സ്ഥിരം ggml_tensor * t) {
    തിരികെ ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
സ്ഥിതം പൂർണ്ണസംഖ്യ ggml_sycl_try_gdn_cache_fusion(സ്ഥിരം ggml_cgraph * cgraph, പൂർണ്ണസംഖ്യ കണ്ണി_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    എങ്കിൽ (!g_ggml_sycl_enable_fusion) {
        തിരികെ 0;
    }

    സ്ഥിരം ggml_tensor * gdn = cgraph->nodes[കണ്ണി_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    എങ്കിൽ (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        തിരികെ 0;
    }

    സ്ഥിരം ggml_tensor * src_v     = gdn->src[2];
    സ്ഥിരം int64_t       S_v       = src_v->ne[0];
    സ്ഥിരം int64_t       H         = src_v->ne[1];
    സ്ഥിരം int64_t       n_tokens  = src_v->ne[2];
    സ്ഥിരം int64_t       n_seqs    = src_v->ne[3];
    സ്ഥിരം int64_t       D         = S_v * S_v * H;
    സ്ഥിരം int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    സ്ഥിരം int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    സ്ഥിരം size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    സ്ഥിരം ggml_tensor * cpy  = nullptr;
    പൂർണ്ണസംഖ്യ                 skip = 0;
    വേണ്ടി (പൂർണ്ണസംഖ്യ j = കണ്ണി_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        സ്ഥിരം ggml_tensor * n = cgraph->nodes[j];
        എങ്കിൽ (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }
        എങ്കിൽ (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            തിരികെ 0;
        }
        cpy  = n;
        skip = j - കണ്ണി_idx;
    }
    എങ്കിൽ (cpy == nullptr) {
        തിരികെ 0;
    }

    സ്ഥിരം ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    സ്ഥിരം ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    എങ്കിൽ (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        തിരികെ 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    സ്ഥിരം std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    എങ്കിൽ (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        തിരികെ 0;
    }

    fused_state_cpy.data        = (ദശാംശം *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / വലുപ്പം(ദശാംശം)) : 0;
    തിരികെ skip;
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * കണ്ണി = cgraph->nodes[i];
        എങ്കിൽ (ggml_sycl_is_view_or_noop(കണ്ണി)) {
            തുടരുക;
        }
        എങ്കിൽ ((കണ്ണി->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            തുടരുക;
        }

        സ്ഥിരം പൂർണ്ണസംഖ്യ nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        എങ്കിൽ (nodes_to_skip != 0) {
            i += nodes_to_skip;
            തുടരുക;
        }
#ifndef NDEBUG
        പ്രതിജ്ഞാ(കണ്ണി->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < GGML_MAX_SRC; j++) {
            എങ്കിൽ (കണ്ണി->src[j] != nullptr) {
                പ്രതിജ്ഞാ(കണ്ണി->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        എങ്കിൽ (കണ്ണി->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            സ്ഥിരം പൂർണ്ണസംഖ്യ gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            എങ്കിൽ (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, കണ്ണി, fused_state_cpy);
                i += gdn_nodes_to_skip;
                തുടരുക;
            }
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            തുടരുക;
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1]);
            i++;
            തുടരുക;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        എങ്കിൽ (കണ്ണി->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1]);
            i++;
            തുടരുക;
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1]);
            i++;
            തുടരുക;
        }
        എങ്കിൽ (കണ്ണി->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(കണ്ണി) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1]);
            i++;
            തുടരുക;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        എങ്കിൽ (കണ്ണി->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            തുടരുക;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        എങ്കിൽ (കണ്ണി->op == GGML_OP_L2_NORM) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            എങ്കിൽ (l2_batch_skip > 0) {
                i += l2_batch_skip;
                തുടരുക;
            }
        }

        എങ്കിൽ (കണ്ണി->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, കണ്ണി, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            തുടരുക;
        }

        എങ്കിൽ (കണ്ണി->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, കണ്ണി, nullptr, cgraph->nodes[i + 1]);
            i++;
            തുടരുക;
        }

        എങ്കിൽ (കണ്ണി->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            തുടരുക;
        }

        തർക്കം ok = ggml_sycl_compute_forward(*sycl_ctx, കണ്ണി);
        എങ്കിൽ (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, കണ്ണി->name, ggml_op_name(കണ്ണി->op));
        }
        GGML_ASSERT(ok);
    }
}

#ifdef GGML_SYCL_GRAPH
സ്ഥിതം തർക്കം check_graph_compatibility(ggml_cgraph * cgraph) {
    എങ്കിൽ (ggml_sycl_info().device_എണ്ണം > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        തിരികെ അസത്യം;
    }

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < cgraph->n_nodes; i++) {
        സ്ഥിരം ggml_op കണ്ണി_op = cgraph->nodes[i]->op;
        തിരഞ്ഞെടുപ്പ് (കണ്ണി_op) {
            സ്ഥിരസ്ഥിതി:
                നിർത്തുക;
            അവസ്ഥ GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            അവസ്ഥ GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(കണ്ണി_op));
                തിരികെ അസത്യം;
            അവസ്ഥ GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                എങ്കിൽ (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(കണ്ണി_op));
                    തിരികെ അസത്യം;
                }
        }
    }
    തിരികെ സത്യം;
}
#endif

സ്ഥിതം ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    തർക്കം use_sycl_graph = അസത്യം;
    എങ്കിൽ (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    എങ്കിൽ (use_sycl_graph) {
        സ്ഥിരം തർക്കം graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        എങ്കിൽ (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            തിരികെ GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        സ്ഥിരം തർക്കം graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        എങ്കിൽ (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } അല്ലെങ്കിൽ {
            ശ്രമിക്കുക {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } പിടിക്കുക (sycl::exception സ്ഥിരം & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } അല്ലെങ്കിൽ
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    തിരികെ GGML_STATUS_SUCCESS;
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
ശ്രമിക്കുക
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    സ്ഥിരം queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
പിടിക്കുക (sycl::exception സ്ഥിരം &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) ശ്രമിക്കുക {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    എങ്കിൽ (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } അല്ലെങ്കിൽ
        GGML_ABORT("fatal error");
} പിടിക്കുക (sycl::exception സ്ഥിരം& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

സ്ഥിതം ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ശൂന്യ,
    /* .get_tensor_2d_async     = */ ശൂന്യ,
    /* .cpy_tensor_async        = */ ശൂന്യ, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ശൂന്യ,
    /* .graph_plan_free         = */ ശൂന്യ,
    /* .graph_plan_update       = */ ശൂന്യ,
    /* .graph_plan_compute      = */ ശൂന്യ,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ശൂന്യ,
};

സ്ഥിതം ggml_guid_t ggml_backend_sycl_guid() {
    സ്ഥിതം ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    തിരികെ &guid;
}

തർക്കം ggml_backend_is_sycl(ggml_backend_t backend) {
    തിരികെ backend != ശൂന്യ && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

പൂർണ്ണസംഖ്യ ggml_backend_sycl_get_device_എണ്ണം(ശൂന്യം) {
    തിരികെ ggml_sycl_info().device_എണ്ണം;
}


// backend device

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ ctx->name.c_str();
}

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ ctx->description.c_str();
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * സ്വതന്ത്രം, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    എങ്കിൽ (!sycl_get_mem_info(ctx->device, സ്വതന്ത്രം, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *സ്വതന്ത്രം);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *സ്വതന്ത്രം, *total);
}

സ്ഥിതം എണ്ണൽ ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    സ്ഥിരം sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    എങ്കിൽ (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        തിരികെ GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    തിരികെ GGML_BACKEND_DEVICE_TYPE_GPU;
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    തർക്കം host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    തർക്കം events = അസത്യം;
#അല്ലെങ്കിൽ
    തർക്കം events = സത്യം;
#endif

    props->caps = {
        /* .async                 = */ സത്യം,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ അസത്യം,
        /* .events                = */ events,
        /* .mmap_support          = */ സത്യം,
    };
}

സ്ഥിതം ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, സ്ഥിരം അക്ഷരം * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ ggml_backend_sycl_init(ctx->device);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ ggml_backend_sycl_buffer_type(ctx->device);
}

സ്ഥിതം ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    തിരികെ ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

സ്ഥിതം ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ശൂന്യം * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    തിരികെ nullptr;
}

സ്ഥിതം തർക്കം do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    പൂർണ്ണസംഖ്യ device = sycl_ctx->device;
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                എങ്കിൽ (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    തിരികെ സത്യം;
                }
                തിരികെ അസത്യം;
            }
        അവസ്ഥ GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            തിരികെ (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        അവസ്ഥ GGML_OP_CONV_2D_DW:
        അവസ്ഥ GGML_OP_CONV_TRANSPOSE_2D:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_UNARY:
            തിരഞ്ഞെടുപ്പ് (ggml_get_unary_op(op)) {
                അവസ്ഥ GGML_UNARY_OP_SGN:
                അവസ്ഥ GGML_UNARY_OP_ABS:
                അവസ്ഥ GGML_UNARY_OP_NEG:
                അവസ്ഥ GGML_UNARY_OP_STEP:
                അവസ്ഥ GGML_UNARY_OP_RELU:
                അവസ്ഥ GGML_UNARY_OP_HARDSIGMOID:
                അവസ്ഥ GGML_UNARY_OP_TANH:
                അവസ്ഥ GGML_UNARY_OP_GELU:
                അവസ്ഥ GGML_UNARY_OP_SILU:
                അവസ്ഥ GGML_UNARY_OP_SIGMOID:
                അവസ്ഥ GGML_UNARY_OP_HARDSWISH:
                അവസ്ഥ GGML_UNARY_OP_GELU_QUICK:
                അവസ്ഥ GGML_UNARY_OP_GELU_ERF:
                അവസ്ഥ GGML_UNARY_OP_EXP:
                അവസ്ഥ GGML_UNARY_OP_EXPM1:
                അവസ്ഥ GGML_UNARY_OP_SOFTPLUS:
                അവസ്ഥ GGML_UNARY_OP_ELU:
                അവസ്ഥ GGML_UNARY_OP_XIELU:
                അവസ്ഥ GGML_UNARY_OP_CEIL:
                    തിരികെ സത്യം;
                അവസ്ഥ GGML_UNARY_OP_FLOOR:
                അവസ്ഥ GGML_UNARY_OP_ROUND:
                അവസ്ഥ GGML_UNARY_OP_TRUNC:
                    തിരികെ സത്യം;
                സ്ഥിരസ്ഥിതി:
                    തിരികെ അസത്യം;
            }
        അവസ്ഥ GGML_OP_GLU:
            തിരഞ്ഞെടുപ്പ് (ggml_get_glu_op(op)) {
                അവസ്ഥ GGML_GLU_OP_REGLU:
                അവസ്ഥ GGML_GLU_OP_GEGLU:
                അവസ്ഥ GGML_GLU_OP_SWIGLU:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_OAI:
                അവസ്ഥ GGML_GLU_OP_GEGLU_ERF:
                അവസ്ഥ GGML_GLU_OP_GEGLU_QUICK:
                അവസ്ഥ GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    തിരികെ ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                സ്ഥിരസ്ഥിതി:
                    തിരികെ അസത്യം;
            }
        അവസ്ഥ GGML_OP_MUL_MAT:
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
            {
                ഘടന ggml_tensor * a = op->src[0];
                ഘടന ggml_tensor * b = op->src[1];

                എങ്കിൽ (a->ne[3] != b->ne[3]) {
                    തിരികെ അസത്യം;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                എങ്കിൽ (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  തിരികെ അസത്യം;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                എങ്കിൽ (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    തിരികെ അസത്യം;
                }

                എങ്കിൽ (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    തിരികെ അസത്യം;
                }

                തിരികെ സത്യം;
            }
        അവസ്ഥ GGML_OP_OUT_PROD:
            തിരികെ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_GET_ROWS:
            {
                തിരഞ്ഞെടുപ്പ് (op->src[0]->type) {
                    അവസ്ഥ GGML_TYPE_I32:
                    അവസ്ഥ GGML_TYPE_F16:
                    അവസ്ഥ GGML_TYPE_BF16:
                    അവസ്ഥ GGML_TYPE_F32:
                    അവസ്ഥ GGML_TYPE_Q1_0:
                    അവസ്ഥ GGML_TYPE_MXFP4:
                    അവസ്ഥ GGML_TYPE_NVFP4:
                    അവസ്ഥ GGML_TYPE_IQ2_XXS:
                    അവസ്ഥ GGML_TYPE_IQ2_XS:
                    അവസ്ഥ GGML_TYPE_IQ2_S:
                    അവസ്ഥ GGML_TYPE_IQ3_XXS:
                    അവസ്ഥ GGML_TYPE_IQ1_S:
                    അവസ്ഥ GGML_TYPE_IQ1_M:
                    അവസ്ഥ GGML_TYPE_IQ3_S:
                    അവസ്ഥ GGML_TYPE_IQ4_NL:
                    അവസ്ഥ GGML_TYPE_IQ4_XS:
                    അവസ്ഥ GGML_TYPE_Q2_K:
                    അവസ്ഥ GGML_TYPE_Q3_K:
                    അവസ്ഥ GGML_TYPE_Q4_0:
                    അവസ്ഥ GGML_TYPE_Q4_1:
                    അവസ്ഥ GGML_TYPE_Q4_K:
                    അവസ്ഥ GGML_TYPE_Q5_0:
                    അവസ്ഥ GGML_TYPE_Q5_1:
                    അവസ്ഥ GGML_TYPE_Q5_K:
                    അവസ്ഥ GGML_TYPE_Q6_K:
                    അവസ്ഥ GGML_TYPE_Q8_0:
                        തിരികെ സത്യം;
                    സ്ഥിരസ്ഥിതി:
                        തിരികെ അസത്യം;
                }
            }
        അവസ്ഥ GGML_OP_GET_ROWS_BACK:
            // return true;
             തിരികെ op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         അവസ്ഥ GGML_OP_SET:
               തിരികെ (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        അവസ്ഥ GGML_OP_SET_ROWS:
            {
                എങ്കിൽ (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    തിരികെ അസത്യം;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                തിരികെ res;
            }
        അവസ്ഥ GGML_OP_DSV4_HC_PRE:
            തിരികെ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_DSV4_HC_COMB:
            തിരികെ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_DSV4_HC_POST:
            തിരികെ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_LIGHTNING_INDEXER:
            തിരികെ op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        അവസ്ഥ GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                എങ്കിൽ (src0_type == GGML_TYPE_F16) {
                    എങ്കിൽ (src1_type == GGML_TYPE_Q2_K ||
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
                        തിരികെ അസത്യം;
                    }
                }

                എങ്കിൽ (src0_type == GGML_TYPE_BF16) {
                    എങ്കിൽ (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        തിരികെ അസത്യം;
                    }
                }

                എങ്കിൽ (src0_type == GGML_TYPE_F32) {
                    എങ്കിൽ (src1_type == GGML_TYPE_Q2_K ||
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
                        തിരികെ അസത്യം;
                    }
                }

                എങ്കിൽ (src1_type == GGML_TYPE_F32) {
                    എങ്കിൽ (src0_type == GGML_TYPE_Q1_0 ||
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
                        തിരികെ അസത്യം;
                    }
                }

                എങ്കിൽ (src0_type == src1_type) {
                    എങ്കിൽ (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        തിരികെ അസത്യം;
                    }
                }

                എങ്കിൽ (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    തിരികെ അസത്യം;
                }

                തിരികെ സത്യം;
            }
        അവസ്ഥ GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                തിരികെ src0_type == GGML_TYPE_F32;
            }
        അവസ്ഥ GGML_OP_CONCAT:
        അവസ്ഥ GGML_OP_DUP:
        അവസ്ഥ GGML_OP_ARGMAX:
        അവസ്ഥ GGML_OP_NONE:
        അവസ്ഥ GGML_OP_RESHAPE:
        അവസ്ഥ GGML_OP_VIEW:
        അവസ്ഥ GGML_OP_PERMUTE:
        അവസ്ഥ GGML_OP_TRANSPOSE:
        അവസ്ഥ GGML_OP_ADD:
        അവസ്ഥ GGML_OP_ADD1:
        അവസ്ഥ GGML_OP_ADD_ID:
        അവസ്ഥ GGML_OP_SUB:
        അവസ്ഥ GGML_OP_എണ്ണം_EQUAL:
        അവസ്ഥ GGML_OP_MUL:
        അവസ്ഥ GGML_OP_DIV:
        അവസ്ഥ GGML_OP_REPEAT:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_PAD_REFLECT_1D:
            തിരികെ ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_SQR:
        അവസ്ഥ GGML_OP_SQRT:
        അവസ്ഥ GGML_OP_SIN:
        അവസ്ഥ GGML_OP_COS:
        അവസ്ഥ GGML_OP_CLAMP:
        അവസ്ഥ GGML_OP_LOG:
        അവസ്ഥ GGML_OP_NORM:
        അവസ്ഥ GGML_OP_L2_NORM:
        അവസ്ഥ GGML_OP_GROUP_NORM:
        അവസ്ഥ GGML_OP_RMS_NORM:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_RMS_NORM_BACK:
            തിരികെ ggml_is_contiguous(op->src[0]);
        അവസ്ഥ GGML_OP_SCALE:
            // the scale kernel is f32 only
            തിരികെ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_CONT:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_TRI:
            {
                സ്ഥിരം ggml_tensor * src0 = op->src[0];
                തിരികെ src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        അവസ്ഥ GGML_OP_DIAG_MASK_INF:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_SOFT_MAX:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_SOFT_MAX_BACK: {
            ദശാംശം max_bias = 0.0f;
            memcpy(&max_bias, (സ്ഥിരം ദശാംശം *) op->op_params + 1, വലുപ്പം(ദശാംശം));
            തിരികെ max_bias == 0.0f;
        }
        അവസ്ഥ GGML_OP_ROPE:
        അവസ്ഥ GGML_OP_ROPE_BACK:
        അവസ്ഥ GGML_OP_IM2COL:
        അവസ്ഥ GGML_OP_IM2COL_3D:
        അവസ്ഥ GGML_OP_UPSCALE:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_COL2IM_1D:
            തിരികെ ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        അവസ്ഥ GGML_OP_CONV_3D: {
            സ്ഥിരം int32_t * opts = (സ്ഥിരം int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            സ്ഥിരം int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            സ്ഥിരം int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            തിരികെ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        അവസ്ഥ GGML_OP_SUM:
        അവസ്ഥ GGML_OP_SUM_ROWS:
        അവസ്ഥ GGML_OP_MEAN:
            തിരികെ ggml_is_contiguous(op->src[0]);
        അവസ്ഥ GGML_OP_ARGSORT:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_മുകൾ_K: {
            സ്ഥിരം ggml_tensor * src0 = op->src[0];
            സ്ഥിരം പൂർണ്ണസംഖ്യ k = op->ne[0];
            തിരികെ src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        അവസ്ഥ GGML_OP_POOL_2D:
        അവസ്ഥ GGML_OP_POOL_1D:
        അവസ്ഥ GGML_OP_ACC:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_PAD:
            എങ്കിൽ (ggml_get_op_params_i32(op, 8) != 0) {
                തിരികെ അസത്യം;
            }
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_LEAKY_RELU:
        അവസ്ഥ GGML_OP_TIMESTEP_EMBEDDING:
        അവസ്ഥ GGML_OP_RWKV_WKV6:
        അവസ്ഥ GGML_OP_RWKV_WKV7:
        അവസ്ഥ GGML_OP_GATED_LINEAR_ATTN:
        അവസ്ഥ GGML_OP_GATED_DELTA_NET:
        അവസ്ഥ GGML_OP_OPT_STEP_ADAMW:
        അവസ്ഥ GGML_OP_OPT_STEP_SGD:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_SSM_CONV:
            തിരികെ op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_ROLL:
            തിരികെ op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_ARANGE:
            തിരികെ op->type == GGML_TYPE_F32;
        അവസ്ഥ GGML_OP_SSM_SCAN:
            എങ്കിൽ (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                തിരികെ (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } അല്ലെങ്കിൽ {
                // TODO Mamba-1 not yet ported to SYCL
                തിരികെ അസത്യം;
            }
        അവസ്ഥ GGML_OP_FILL:
        അവസ്ഥ GGML_OP_CUMSUM:
        അവസ്ഥ GGML_OP_DIAG:
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS:
        അവസ്ഥ GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            തിരികെ സത്യം;
        അവസ്ഥ GGML_OP_SOLVE_TRI:
            തിരികെ op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        അവസ്ഥ GGML_OP_FLASH_ATTN_EXT:
            തിരികെ ggml_sycl_flash_attn_ext_supported(device, op);
        സ്ഥിരസ്ഥിതി:
            തിരികെ അസത്യം;
    }

    GGML_UNUSED(dev);
}

സ്ഥിതം തർക്കം ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    തർക്കം res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    തിരികെ res;
}

സ്ഥിതം തർക്കം ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    എങ്കിൽ (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        തിരികെ അസത്യം;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ buft_ctx->device == sycl_ctx->device;
}

സ്ഥിതം int64_t get_op_batch_size(സ്ഥിരം ggml_tensor * op) {
    തിരഞ്ഞെടുപ്പ് (op->op) {
        അവസ്ഥ GGML_OP_GET_ROWS:
            തിരികെ 0;
        അവസ്ഥ GGML_OP_MUL_MAT:
            തിരികെ op->ne[1];
        അവസ്ഥ GGML_OP_MUL_MAT_ID:
        അവസ്ഥ GGML_OP_ROPE:
            തിരികെ op->ne[2];
        സ്ഥിരസ്ഥിതി:
            തിരികെ ggml_nrows(op);
    }
}

സ്ഥിതം തർക്കം ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, സ്ഥിരം ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    തിരികെ get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

സ്ഥിതം ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    തിരികെ nullptr;
#അല്ലെങ്കിൽ
  sycl::event *event_ptr = പുതിയ sycl::event();

  തിരികെ പുതിയ ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

സ്ഥിതം ശൂന്യം ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) ശ്രമിക്കുക {
  GGML_UNUSED(dev);
  എങ്കിൽ (event == nullptr) {
    തിരികെ;
  }

  എങ്കിൽ (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


സ്ഥിതം ശൂന്യം ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) ശ്രമിക്കുക {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} പിടിക്കുക (sycl::exception സ്ഥിരം &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

സ്ഥിതം സ്ഥിരം ggml_backend_device_i ggml_backend_sycl_device_interface = {
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

ഘടന ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

സ്ഥിതം സ്ഥിരം അക്ഷരം * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    തിരികെ GGML_SYCL_NAME;
}

സ്ഥിതം size_t ggml_backend_sycl_reg_get_device_എണ്ണം(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    തിരികെ ctx->devices.size();
}

സ്ഥിതം ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    തിരികെ ctx->devices[index];
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

ഘടന ggml_backend_sycl_comm_context {
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

    ശൂന്യം free_host() {
        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 2; ++i) {
            സ്ഥിരം sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            എങ്കിൽ (host_out[i]) { sycl::സ്വതന്ത്രം(host_out[i], sctx); host_out[i] = nullptr; }
            എങ്കിൽ (host_in[i])  { sycl::സ്വതന്ത്രം(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
സ്ഥിതം ശൂന്യം ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    ശൂന്യം * dst0, സ്ഥിരം ശൂന്യം * src1, ശൂന്യം * dst1, സ്ഥിരം ശൂന്യം * src0,
                                    size_t nbytes) {
    എങ്കിൽ (comm->host_bytes < nbytes) {
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

ശൂന്യം * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) ശ്രമിക്കുക {
    വേണ്ടി (size_t i = 0; i < n_backends; ++i) {
        എങ്കിൽ (!ggml_backend_is_sycl(backends[i])) {
            തിരികെ nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    എങ്കിൽ (n_backends != 2) {
        തിരികെ nullptr;
    }

    auto * ctx = പുതിയ ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    തിരികെ ctx;
}
പിടിക്കുക (സ്ഥിരം sycl::exception &) { തിരികെ nullptr; }
പിടിക്കുക (...)                     { തിരികെ nullptr; }

ശൂന്യം ggml_backend_sycl_comm_free(ശൂന്യം * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    എങ്കിൽ (comm_ctx == nullptr) {
        തിരികെ;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    എങ്കിൽ (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        ശ്രമിക്കുക {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } പിടിക്കുക (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

തർക്കം ggml_backend_sycl_comm_allreduce_tensor(ശൂന്യം * comm_ctx_v, ഘടന ggml_tensor ** tensors) ശ്രമിക്കുക {
    എങ്കിൽ (comm_ctx_v == nullptr) {
        തിരികെ അസത്യം;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    സ്ഥിരം size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    എങ്കിൽ (n_backends != 2) {
        തിരികെ അസത്യം;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    എങ്കിൽ (tensors[0]->type != tensors[1]->type) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        തിരികെ അസത്യം;
    }
    എങ്കിൽ (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        തിരികെ അസത്യം;
    }

    സ്ഥിരം int64_t nelem  = ggml_nelements(tensors[0]);
    സ്ഥിരം size_t  nbytes = ggml_nbytes(tensors[0]);
    എങ്കിൽ (nelem == 0) {
        തിരികെ സത്യം;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    എങ്കിൽ (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    എങ്കിൽ (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((ദശാംശം) f16_out0[i] + (ദശാംശം) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((ദശാംശം) f16_out1[i] + (ദശാംശം) f16_tmp1[i]);
            });
        });
        തിരികെ സത്യം;
    }

    ദശാംശം * out0 = (ദശാംശം *) tensors[0]->data;
    ദശാംശം * out1 = (ദശാംശം *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    സ്ഥിതം constexpr int64_t BF16_THRESHOLD = 32768;

    എങ്കിൽ (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        ദശാംശം * tmp0 = (ദശാംശം *) buf0;
        ദശാംശം * tmp1 = (ദശാംശം *) buf1;

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
        തിരികെ സത്യം;
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
    സ്ഥിരം size_t bf16_bytes = nelem * വലുപ്പം(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<ദശാംശം>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<ദശാംശം>(((uint32_t) inbox1[i]) << 16);
        });
    });

    തിരികെ സത്യം;
}
പിടിക്കുക (സ്ഥിരം sycl::exception &) { തിരികെ അസത്യം; }
പിടിക്കുക (...)                     { തിരികെ അസത്യം; }

സ്ഥിതം ശൂന്യം *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, സ്ഥിരം അക്ഷരം *name) {
    GGML_UNUSED(reg);

    എങ്കിൽ (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_init") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_sycl_comm_init;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_free") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_sycl_comm_free;
    }
    എങ്കിൽ (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        തിരികെ (ശൂന്യം *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    തിരികെ nullptr;
}

സ്ഥിതം സ്ഥിരം ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_എണ്ണം,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    സ്ഥിതം ggml_backend_reg reg;
    സ്ഥിതം തർക്കം initialized = അസത്യം;

    {
        സ്ഥിതം std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        എങ്കിൽ (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = പുതിയ ggml_backend_sycl_reg_context;
            സ്ഥിരം പൂർണ്ണസംഖ്യ min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ggml_sycl_info().device_എണ്ണം; i++) {
                ggml_backend_sycl_device_context * dev_ctx = പുതിയ ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = പുതിയ ggml_backend_device {
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

        initialized = സത്യം;
    }

    തിരികെ &reg;
}

ggml_backend_t ggml_backend_sycl_init(പൂർണ്ണസംഖ്യ device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = പുതിയ ggml_backend_sycl_context(device);
    എങ്കിൽ (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        തിരികെ nullptr;
    };

    ggml_backend_t sycl_backend = പുതിയ ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    തിരികെ sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
