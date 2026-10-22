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

#अन्तर्भाव <algorithm>
#अन्तर्भाव <array>
#अन्तर्भाव <assert.h>
#अन्तर्भाव <atomic>
#अन्तर्भाव <cctype>
#अन्तर्भाव <cinttypes>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <float.h>
#अन्तर्भाव <limits>
#अन्तर्भाव <optional>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <vector>
#अन्तर्भाव <cmath>
#अन्तर्भाव <iostream>
#अन्तर्भाव <fstream>
#अन्तर्भाव <stdio.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <regex>

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#अन्तर्भाव <level_zero/ze_api.h>
#अन्तर्भाव <level_zero/zes_api.h>
#endif
#जर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#जर SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#अन्तर्भाव <sycl/half_type.hpp>

#अन्तर्भाव "ggml.h"
#अन्तर्भाव "ggml-sycl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml-backend-impl.h"

#अन्तर्भाव "ggml-sycl/add-id.hpp"
#अन्तर्भाव "ggml-sycl/backend.hpp"
#अन्तर्भाव "ggml-sycl/common.hpp"
#अन्तर्भाव "ggml-sycl/element_wise.hpp"
#अन्तर्भाव "ggml-sycl/fwht.hpp"
#अन्तर्भाव "ggml-sycl/fused-gemm.hpp"
#अन्तर्भाव "ggml-sycl/gemm.hpp"
#अन्तर्भाव "ggml-sycl/getrows.hpp"
#अन्तर्भाव "ggml-sycl/mem.hpp"
#अन्तर्भाव "ggml-sycl/norm.hpp"
#अन्तर्भाव "ggml-sycl/presets.hpp"
#अन्तर्भाव "ggml-sycl/quantize.hpp"
#अन्तर्भाव "ggml-sycl/repeat_back.hpp"
#अन्तर्भाव "ggml-sycl/set_rows.hpp"
#अन्तर्भाव "ggml-sycl/set.hpp"
#अन्तर्भाव "ggml-sycl/dsv4-hc.hpp"
#अन्तर्भाव "ggml-sycl/lightning-indexer.hpp"
#अन्तर्भाव "ggml-sycl/conv2d.hpp"
#अन्तर्भाव "ggml-sycl/conv2d-dw.hpp"
#अन्तर्भाव "ggml-sycl/conv2d-transpose.hpp"
#अन्तर्भाव "ggml-sycl/ssm_conv.hpp"
#अन्तर्भाव "ggml-sycl/sycl_hw.hpp"
#अन्तर्भाव "ggml-sycl/ssm_scan.hpp"
#अन्तर्भाव "ggml-sycl/fill.hpp"
#अन्तर्भाव "ggml-sycl/cumsum.hpp"
#अन्तर्भाव "ggml-sycl/diag.hpp"
#अन्तर्भाव "ggml-sycl/opt-step.hpp"
#अन्तर्भाव "ggml-sycl/solve_tri.hpp"
#अन्तर्भाव "ggml-sycl/gated_delta_net.hpp"
#अन्तर्भाव "ggml-sycl/pool.hpp"
#अन्तर्भाव "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

स्थिर तार्किक g_sycl_loaded = असत्य;
पूर्णांक g_ggml_sycl_debug = 0;
पूर्णांक g_ggml_sycl_dev_debug = 0;
पूर्णांक g_ggml_sycl_enable_optimize = 1;
पूर्णांक g_ggml_sycl_enable_graph = 0;
पूर्णांक g_ggml_sycl_enable_dnn = 1;
पूर्णांक g_ggml_sycl_fa_onednn = 1;
पूर्णांक g_ggml_sycl_fa_onednn_max_kv = 0;
पूर्णांक g_ggml_sycl_enable_mkl_fa = 1;
पूर्णांक g_ggml_sycl_memtrace = 0;
पूर्णांक g_ggml_sycl_memtrace_step = 64;
पूर्णांक g_ggml_sycl_enable_vmm = 1;
पूर्णांक g_ggml_sycl_enable_fusion = 1;
पूर्णांक g_ggml_sycl_enable_esimd = 1;
पूर्णांक g_ggml_sycl_mmvq_wide = 1;
पूर्णांक g_ggml_sycl_prioritize_dmmv = 0;
पूर्णांक g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
पूर्णांक g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
पूर्णांक g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
पूर्णांक g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
स्थिर स्थिरांक अक्षर * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
स्थिर पूर्णांक ggml_sycl_get_env_precision(स्थिरांक अक्षर * name, पूर्णांक def) {
    स्थिरांक अक्षर * env = दो_पर्या(name);
    जर (!env) {
        परत def;
    }
    std::string mode(env);
    साठी (अक्षर & c : mode) {
        c = (अक्षर) std::बड़े((अचिन्हित अक्षर) c);
    }
    साठी (पूर्णांक i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        जर (mode == ggml_sycl_dynamic_precision_names[i]) {
            परत i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    परत def;
}
पूर्णांक g_ggml_sycl_use_async_mem_op = 0;
पूर्णांक g_ggml_sycl_use_async_mem_op_requested = 1;
पूर्णांक g_ggml_sycl_use_level_zero_api = 0;
पूर्णांक g_ggml_sycl_enable_flash_attention = 1;
पूर्णांक g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
पूर्णांक g_ggml_sycl_usm_system = 0;
पूर्णांक g_ggml_sycl_enable_host_pinned_mem = 1;
पूर्णांक g_ggml_sycl_host_pinned_mem_2g = 0;
पूर्णांक g_ggml_sycl_upload_staging_slots = 4;
पूर्णांक g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
पूर्णांक g_ggml_sycl_enable_sparse_fa = 0;
पूर्णांक g_ggml_sycl_debug_sparse_fa = 0;
पूर्णांक g_ggml_sycl_sparse_fa_margin = 256;

#जर GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
स्थिर तार्किक ggml_sycl_dnnl_detect_optimized_gemm(पूर्णांक device) {
    using dt = dnnl::memory::data_type;

    प्रयत्न {
        स्थिरांक dnnl::memory::dims dims    = { 1, 64, 64 };
        स्थिरांक dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        स्थिरांक स्वतः & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        स्थिरांक स्वतः   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        स्थिरांक स्वतः   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        स्थिरांक std::string impl = pd.impl_info_str();
        जर (impl.find("ref") == std::string::npos) {
            परत सत्य;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } पकडा (स्थिरांक std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    परत असत्य;
}
#endif

स्थिर ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    प्रयत्न {
        info.device_मोजणी = dpct::dev_mgr::instance().device_मोजणी();
    } पकडा (sycl::exception स्थिरांक &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_मोजणी = 0;
    }
    जर (info.device_मोजणी == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        परत info;
    }

    GGML_ASSERT(info.device_मोजणी <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    साठी (पूर्णांक i = 0; i < info.device_मोजणी; ++i) {
        dpct::device_info prop;
        स्वतः & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#जर !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#अन्यथा
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        जर (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            स्थिरांक माप_प्रकार physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<माप_प्रकार>(
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
#जर GGML_SYCL_DNNL
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
        जर (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = असत्य;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        जर (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            जर (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = सत्य;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    साठी (पूर्णांक id = 0; id < info.device_मोजणी; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    परत info;
}

स्थिरांक ggml_sycl_device_info & ggml_sycl_info() {
    स्थिर ggml_sycl_device_info info = ggml_sycl_init();
    परत info;
}

स्थिर शून्य print_device_detail(पूर्णांक id, sycl::device &device, std::string device_type) {

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

    स्वतः global_mem_size = prop.get_global_mem_size()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_size(), prop.get_max_sub_group_size(),
            global_mem_size, device.get_info<sycl::info::device::driver_version>().c_str());
}

स्थिर शून्य print_device_opt_feature(पूर्णांक device_मोजणी) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, माप_प्रकार> DeviceNums;
    साठी (पूर्णांक id = 0; id < device_मोजणी; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      पूर्णांक type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
शून्य ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    पूर्णांक device_मोजणी = dpct::dev_mgr::instance().device_मोजणी();
    std::map<std::string, माप_प्रकार> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_मोजणी);

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

    साठी (पूर्णांक id = 0; id < device_मोजणी; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      पूर्णांक type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_मोजणी);
}

स्थिर स्थिरांक अक्षर* dev2dev_int2str(पूर्णांक dev2dev) {
    जर (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        परत "SYCL API";
    } अन्यथा जर (dev2dev == DEV2DEV_MEMCPY_L0) {
        परत "Level Zero API";
    } अन्यथा जर (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        परत "Host Forward";
    } अन्यथा {
        परत "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    जर (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (पूर्णांक) res);
    }
    परत res;
}

स्थिर ze_result_t get_zes_init_res() {
    स्थिर ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (पूर्णांक) zes_init_res);
    परत zes_init_res;
}
#endif

स्थिर शून्य initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

स्थिर शून्य ggml_check_sycl() प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    स्थिर तार्किक initialized = असत्य;

    जर (!initialized) {
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
#अन्यथा
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        जर (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#अन्यथा
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
#जर defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#जर defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#जर defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#जर defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#जर defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#जर defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#जर defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#जर defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#जर defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#अन्यथा
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
#जर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        जर (g_ggml_sycl_use_async_mem_op) {
            साठी (अचिन्हित पूर्णांक i = 0; i < dpct::dev_mgr::instance().device_मोजणी(); ++i) {
                जर (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    थांबा;
                }
            }
        }
#endif
        जर (CHECK_TRY_ERROR(g_all_sycl_device_मोजणी =
                            dpct::dev_mgr::instance().device_मोजणी()) != 0) {
            initialized = सत्य;
            g_sycl_loaded = असत्य;
            परत;
        }
        GGML_ASSERT(g_all_sycl_device_मोजणी <= GGML_SYCL_MAX_DEVICES);

        initialized = सत्य;
        g_sycl_loaded = सत्य;
        ggml_backend_sycl_print_sycl_devices();
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
ओळीत शून्य check_allow_gpu_index(स्थिरांक पूर्णांक device_index) {
  जर (device_index >= ggml_sycl_info().device_मोजणी) {
    अक्षर error_buf[256];
    snprintf(
        error_buf,
        आकार(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_मोजणी - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    प्रतिज्ञा(असत्य);
  }
}

GGML_API शून्य ggml_backend_sycl_get_gpu_list(पूर्णांक *id_list, पूर्णांक max_len) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    साठी(पूर्णांक i=0;i<max_len;i++) id_list[i] = -1;

    साठी (पूर्णांक i=0;i< ggml_sycl_info().device_मोजणी;i++){
        जर (i>=max_len) थांबा;
        id_list[i] = i;
    }
    परत;
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

ओळीत शून्य free_aligned_mem_host(शून्य * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#अन्यथा
    मुक्त(memblock);
#endif
}

// sycl buffer

संरचना ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    स्थिर constexpr माप_प्रकार staging_slot_size = 8*1024*1024;

    संरचना host_staging {
        शून्य * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<तार्किक> submitted;
        पूर्णांक next = 0;
    };

    पूर्णांक device;
    शून्य * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    तार्किक is_usm_system;

    ggml_backend_sycl_buffer_context(पूर्णांक device, शून्य * dev_ptr, queue_ptr stream, तार्किक is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    शून्य drop_host_staging() {
        साठी (माप_प्रकार i = 0; i < staging.submitted.size(); ++i) {
            जर (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = असत्य;
            }
        }
        जर (staging.data != nullptr) {
            sycl::मुक्त(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        जर (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            जर (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            अन्यथा
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        साठी (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

स्थिर तार्किक ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    परत buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

स्थिर शून्य
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) प्रयत्न {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    परत ctx->dev_ptr;
}

स्थिर गणना ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    जर (tensor->view_src != रिक्त) {
        प्रतिज्ञा(tensor->view_src->buffer->buft == buffer->buft);
        परत GGML_STATUS_SUCCESS;
    }

    जर (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        निवड (tensor->type) {
            स्थिती GGML_TYPE_Q4_0:
            स्थिती GGML_TYPE_Q8_0:
            स्थिती GGML_TYPE_Q2_K:
            स्थिती GGML_TYPE_Q3_K:
            स्थिती GGML_TYPE_Q4_K:
            स्थिती GGML_TYPE_Q5_K:
            स्थिती GGML_TYPE_Q6_K:
            स्थिती GGML_TYPE_IQ3_XXS:
            स्थिती GGML_TYPE_IQ3_S:
            स्थिती GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = नवीन ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                थांबा;
            }
            पूर्वनिर्धारित:
                थांबा;
        }
    }

    जर (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        माप_प्रकार original_size = ggml_nbytes(tensor);
        माप_प्रकार padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        जर (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->स_रखो(
                (अक्षर *)tensor->data + original_size, 0,
                padded_size - original_size).रुको()));
        }
    }
    परत GGML_STATUS_SUCCESS;
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                स्थिरांक शून्य *data, माप_प्रकार offset,
                                                माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    स्थिरांक पूर्णांक n_slots = g_ggml_sycl_upload_staging_slots;
    जर (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        जर (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, असत्य);
        }
    }
    जर (ctx->staging.data != nullptr) {
        queue_ptr    stream    = ctx->stream;
        अक्षर *       dst       = (अक्षर *) tensor->data + offset;
        स्थिरांक अक्षर * src       = (स्थिरांक अक्षर *) data;
        माप_प्रकार       remaining = size;
        पर्यंत (remaining > 0) {
            स्थिरांक माप_प्रकार chunk = std::min(remaining, ctx->staging_slot_size);
            स्थिरांक पूर्णांक    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (पूर्णांक) ctx->staging.submitted.size();
            जर (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            शून्य * stage = (अक्षर *) ctx->staging.data + slot * ctx->staging_slot_size;
            स_नकल(stage, src, chunk);
            ctx->staging.events[slot] = stream->स_नकल(dst, stage, chunk);
            ctx->staging.submitted[slot] = सत्य;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        परत;
    }

    स्वतः stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    अक्षर * host_buf = (अक्षर *) दो_स्मृति(size);
    स_नकल(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल((अक्षर *) tensor->data + offset, host_buf, size).रुको()));
    मुक्त(host_buf);
#अन्यथा
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल((अक्षर *) tensor->data + offset, data, size).रुको()));
#endif
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                स्थिरांक ggml_tensor *tensor,
                                                शून्य *data, माप_प्रकार offset,
                                                माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    स्वतः stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.स_नकल(data, (स्थिरांक अक्षर *)tensor->data + offset, size)
            .रुको()));
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर तार्किक ggml_sycl_is_l0_discrete_gpu(पूर्णांक device) {
    परत ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

स्थिर शून्य memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, शून्य *ptr_dst,
                         स्थिरांक शून्य *ptr_src, माप_प्रकार size) {
    अक्षर *host_buf = (अक्षर *)दो_स्मृति(size);
    जर (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.स_नकल(host_buf, (स्थिरांक अक्षर *)ptr_src, size).रुको();
    q_dst.स_नकल((अक्षर *)ptr_dst, host_buf, size).रुको();
    मुक्त(host_buf);
}

स्थिर शून्य dev2dev_memcpy(पूर्णांक device_dst, sycl::queue &q_dst, पूर्णांक device_src, sycl::queue &q_src, शून्य *ptr_dst,
                    स्थिरांक शून्य *ptr_src, माप_प्रकार size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    जर (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        स्थिरांक तार्किक l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        जर (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            स्वतः ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            जर (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                जर (r == ZE_RESULT_SUCCESS) {
                    परत;
                }
            }
        }
    }
#endif

    जर (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        जर (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.स_नकल(ptr_dst, ptr_src, size).रुको()));
            परत;
        }
    }

    // Host-staged copy
    जर(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } अन्यथा {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

स्थिर तार्किक
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    स्थिरांक ggml_tensor *src,
                                    ggml_tensor *dst) प्रयत्न {
    तार्किक is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    जर (is_cpy_supported) {
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
        माप_प्रकार size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->data, src->data, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#जर 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल(
            (अक्षर *)dst->data, (स्थिरांक अक्षर *)src->data, size).रुको()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        परत सत्य;
    }
    परत असत्य;
    GGML_UNUSED(buffer);
} पकडा (स्थिरांक sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr माप_प्रकार MAX_CHUNK = 2ULL << 30;  // 2 GiB
    साठी (माप_प्रकार off = 0; off < buffer->size; off += MAX_CHUNK) {
        माप_प्रकार chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_रखो(static_cast<अक्षर*>(ctx->dev_ptr) + off, value, chunk)
                .रुको()
        ));
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   माप_प्रकार offset, माप_प्रकार size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    स्वतः stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    जर (size == 0) {
        परत;  // Nothing to do
    }
    जर (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    शून्य * target_ptr = static_cast<अक्षर *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_रखो(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).रुको()));
}

स्थिर शून्य ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    जर (buffer == nullptr) {
        परत;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    जर (ctx != nullptr) {
        साठी (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

स्थिर स्थिरांक ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
संरचना ggml_backend_sycl_buffer_type_context {
    पूर्णांक device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    परत ctx->name.c_str();
}

स्थिर तार्किक check_usm_system(पूर्णांक device, माप_प्रकार size) {
    तार्किक use_usm_system = g_ggml_sycl_usm_system && size >= ((माप_प्रकार)4 * MEM_SIZE_1G);

    जर (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = असत्य;
    }

    परत use_usm_system;
}

ओळीत शून्य * aligned_malloc_host(माप_प्रकार alignment, माप_प्रकार size) {
#ifdef _WIN32
    परत _aligned_malloc(size, alignment);
#अन्यथा
    परत aligned_alloc(alignment, size);
#endif
}

स्थिर ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    स्थिरांक queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (माप_प्रकार)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    माप_प्रकार alignment = MEM_SIZE_2M;
    माप_प्रकार aligned_size = ((size + alignment - 1) / alignment) * alignment;
    तार्किक use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    शून्य * dev_ptr;
    जर (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (शून्य *)aligned_malloc_host(alignment, aligned_size);
        जर (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            परत nullptr;
        }
    } अन्यथा {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (शून्य *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        जर (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          परत nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = नवीन  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    परत ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर माप_प्रकार ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    परत SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

स्थिर तार्किक is_bmg_g31_arch(पूर्णांक device) {
    परत ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

स्थिर माप_प्रकार ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    पूर्णांक device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    माप_प्रकार max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    जर (g_ggml_sycl_host_pinned_mem_2g) {
        परत std::min(max_alloc_size, (माप_प्रकार) 2LL*1024*1024*1024);
    } अन्यथा {
        जर(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        परत max_alloc_size;

    }
    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    माप_प्रकार size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    जर (ggml_is_quantized(tensor->type)) {
        जर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    परत size;

    GGML_UNUSED(buft);
}

स्थिर स्थिरांक ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(पूर्णांक device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    स्वतः dev_मोजणी = ggml_backend_sycl_get_device_मोजणी();

    जर (device>=dev_मोजणी or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_मोजणी-1);
        GGML_ASSERT(device<dev_मोजणी);
    }
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    स्थिर तार्किक ggml_backend_sycl_buffer_type_initialized = असत्य;

    जर (!ggml_backend_sycl_buffer_type_initialized) {
        साठी (पूर्णांक i = 0; i < dev_मोजणी; i++) {
            स्वतः & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ नवीन ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = सत्य;
    }
    परत &ggml_backend_sycl_buffer_types[device];
}

स्थिर ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    पूर्णांक device = ctx->device;
    जर (device>=ggml_sycl_info().device_मोजणी or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_मोजणी-1);
        GGML_ASSERT(device<ggml_sycl_info().device_मोजणी);
    }
    स्थिर संरचना ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    स्थिर तार्किक ggml_backend_sycl_buffer_type_initialized = असत्य;

    जर (!ggml_backend_sycl_buffer_type_initialized) {
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ नवीन ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = सत्य;
    }
    परत &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

स्थिर int64_t get_row_rounding(ggml_type type, स्थिरांक std::array<दशांश, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = पूर्णांक_उच्च;
    int64_t max_compute_capability = पूर्णांक_न्यून;
    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        जर (tensor_split[i] < (i + 1 < ggml_sycl_info().device_मोजणी ? tensor_split[i + 1] : 1.0f)) {
            जर (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            जर (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    निवड(type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
            परत max_compute_capability >= VER_GEN9 ? 128 : 64;
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
            परत 64;
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_F32:
            परत 1;
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ2_S:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ4_NL:
            परत max_compute_capability >= VER_GEN9 ? 128 : 64;
        स्थिती GGML_TYPE_IQ3_S:
            परत max_compute_capability >= VER_GEN9 ? 128 : 64;
        स्थिती GGML_TYPE_Q6_K:
            परत 64;
        पूर्वनिर्धारित:
            GGML_ABORT("fatal error");
    }
}

स्थिर शून्य get_row_split(int64_t * row_खालचा, int64_t * row_वरचा, स्थिरांक ggml_tensor * tensor, स्थिरांक std::array<दशांश, GGML_SYCL_MAX_DEVICES> & tensor_split, पूर्णांक id) {
    स्थिरांक int64_t nrows = ggml_nrows(tensor);
    स्थिरांक int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_खालचा = id == 0 ? 0 : nrows*tensor_split[id];
    *row_खालचा -= *row_खालचा % rounding;
    जर (id == ggml_sycl_info().device_मोजणी - 1) {
        *row_वरचा = nrows;
    } अन्यथा {
        *row_वरचा = nrows*tensor_split[id + 1];
        *row_वरचा -= *row_वरचा % rounding;
    }
}

स्थिर माप_प्रकार ggml_nbytes_split(स्थिरांक संरचना ggml_tensor * tensor, पूर्णांक nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    परत nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

संरचना ggml_backend_sycl_split_buffer_type_context {
    std::array<दशांश, GGML_SYCL_MAX_DEVICES> tensor_split;
};

संरचना ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() प्रयत्न {
        साठी (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    पकडा (sycl::exception स्थिरांक &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::निकास(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

स्थिर शून्य ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

स्थिर शून्य * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    परत (शून्य *)0x1000;

    GGML_UNUSED(buffer);
}

स्थिर गणना ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    स्थिरांक int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = नवीन ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    जर (ctx->streams.empty()) {
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        int64_t row_खालचा, row_वरचा;
        get_row_split(&row_खालचा, &row_वरचा, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_वरचा - row_खालचा;
        जर (nrows_split == 0) {
            सुरूठेवा;
        }

        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिरांक माप_प्रकार original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        जर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        स्थिरांक queue_ptr stream = ctx->streams[i];
        अक्षर * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (अक्षर *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        जर (!buf) {
            अक्षर err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            फेका std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        जर (size > original_size) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .स_रखो(buf + original_size, 0, size - original_size)
                    .रुको()));
        }

        extra->data_device[i] = buf;

        साठी (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = नवीन sycl::event()));
        }
    }
    tensor->extra = extra;
    परत GGML_STATUS_SUCCESS;
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, स्थिरांक शून्य *data,
                                          माप_प्रकार offset, माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    स्थिरांक int64_t ne0 = tensor->ne[0];
    स्थिरांक माप_प्रकार nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        int64_t row_खालचा, row_वरचा;
        get_row_split(&row_खालचा, &row_वरचा, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_वरचा - row_खालचा;
        जर (nrows_split == 0) {
            सुरूठेवा;
        }

        स्थिरांक माप_प्रकार offset_split = row_खालचा*nb1;
        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिरांक माप_प्रकार original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        जर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        स्थिरांक अक्षर * buf_host = (स्थिरांक अक्षर *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        स्थिरांक queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_नकल(extra->data_device[i], buf_host, original_size)
                .रुको()));
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          स्थिरांक ggml_tensor *tensor, शून्य *data,
                                          माप_प्रकार offset, माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    स्थिरांक int64_t ne0 = tensor->ne[0];
    स्थिरांक माप_प्रकार nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        int64_t row_खालचा, row_वरचा;
        get_row_split(&row_खालचा, &row_वरचा, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_वरचा - row_खालचा;
        जर (nrows_split == 0) {
            सुरूठेवा;
        }

        स्थिरांक माप_प्रकार offset_split = row_खालचा*nb1;
        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिरांक माप_प्रकार original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        जर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        अक्षर * buf_host = (अक्षर *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        स्थिरांक queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_नकल(buf_host, extra->data_device[i], original_size)
                .रुको()));
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

स्थिर संरचना ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ रिक्त,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ रिक्त,
};

// sycl split buffer type

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    परत GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

स्थिर तार्किक ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   परत buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

स्थिर ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = नवीन ggml_backend_sycl_split_buffer_context();

    परत ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

स्थिर माप_प्रकार ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    परत SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

स्थिर माप_प्रकार ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, स्थिरांक ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    माप_प्रकार total_size = 0;

    स्थिरांक int64_t ne0 = tensor->ne[0];

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        int64_t row_खालचा, row_वरचा;
        get_row_split(&row_खालचा, &row_वरचा, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_वरचा - row_खालचा;
        जर (nrows_split == 0) {
            सुरूठेवा;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        जर (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    परत total_size;
}

स्थिर तार्किक ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    परत असत्य;

    GGML_UNUSED(buft);
}

स्थिर ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ रिक्त, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(पूर्णांक main_device, स्थिरांक दशांश * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    स्थिर std::map<std::array<दशांश, GGML_SYCL_MAX_DEVICES>, संरचना ggml_backend_buffer_type> buft_map;

    std::array<दशांश, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    तार्किक all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](दशांश x) { परत x == 0.0f; });
    जर (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } अन्यथा {
        दशांश split_sum = 0.0f;
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    स्वतः it = buft_map.find(tensor_split_arr);
    जर (it != buft_map.end()) {
        परत &it->second;
    }

    संरचना ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ नवीन ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    स्वतः result = buft_map.emplace(tensor_split_arr, buft);
    परत &result.first->second;
}

// host buffer type

संरचना ggml_backend_sycl_device_context {
    पूर्णांक device;
    std::string name;
    std::string description;
    पूर्णांक op_offload_min_batch_size;
};

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    परत GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

स्थिर पूर्णांक ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    परत static_cast<स्थिरांक ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
स्थिर शून्य * ggml_backend_sycl_host_malloc(पूर्णांक device, माप_प्रकार size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    शून्य * ptr = nullptr;
    प्रयत्न {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        स्वतः & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } पकडा (...) {
        ptr = nullptr;
    }
    जर (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    परत ptr;
}

स्थिर शून्य ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    जर (buffer->context == nullptr) {
        परत;
    }
    जर (g_ggml_sycl_enable_host_pinned_mem) {
        स्थिरांक पूर्णांक device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        स्वतः & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(buffer->context, q)));
    } अन्यथा {
        free_aligned_mem_host((शून्य *) buffer->context);
    }
}

स्थिर ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, माप_प्रकार size) {
    शून्य * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    जर (ptr == nullptr) {
        // fallback to cpu buffer
        परत ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    परत buffer;
}

स्थिर माप_प्रकार ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    जर (g_ggml_sycl_enable_host_pinned_mem) {
        स्थिरांक पूर्णांक device = ggml_backend_sycl_host_buffer_type_device(buft);
        माप_प्रकार max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        जर (g_ggml_sycl_host_pinned_mem_2g) {
            परत std::min(max_alloc_size, (माप_प्रकार) 2LL*1024*1024*1024);
        } अन्यथा {
            परत max_alloc_size;
        }
    } अन्यथा {
        परत SIZE_MAX;
    }
}

स्थिर ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(पूर्णांक device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    स्थिर std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_मोजणी());
        साठी (माप_प्रकार i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ रिक्त,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ रिक्त,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        परत bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (पूर्णांक) buffer_types_host.size());

    परत &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    परत ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
संरचना ggml_sycl_pool_leg : सार्वजनिक ggml_sycl_pool {
    स्थिर स्थिरांक पूर्णांक MAX_SYCL_BUFFERS = 256;

    पूर्णांक device;
    queue_ptr qptr;
    संरचना ggml_sycl_buffer {
        शून्य * ptr = nullptr;
        माप_प्रकार size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    माप_प्रकार pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, पूर्णांक device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        पूर्णांक    n_cached    = 0;
        माप_प्रकार bytes_cached = 0;
        साठी (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            जर (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        स्थिरांक स्वतः slots = format_slots_in_alloc_order();
        जर (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        साठी (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            जर (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() स्थिरांक {
        std::string ओळ;
        अक्षर buf[32];
        तार्किक first = सत्य;
        साठी (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            जर (buffer_pool[i].ptr == nullptr) {
                सुरूठेवा;
            }
            जर (!first) {
                ओळ += '/';
            }
            first = असत्य;
            snprintf(buf, आकार(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            ओळ += buf;
        }
        परत ओळ;
    }
#endif

    शून्य * alloc(माप_प्रकार size, माप_प्रकार * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        पूर्णांक nnz = 0;
        माप_प्रकार max_size = 0;
#endif
        माप_प्रकार best_diff = 1ull << 36;
        पूर्णांक ibest = -1;
        साठी (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            जर (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                जर (b.size > max_size) max_size = b.size;
#endif
                जर (b.size >= size) {
                    माप_प्रकार diff = b.size - size;
                    जर (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        जर (!best_diff) {
                            शून्य * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            परत ptr;
                        }
                    }
                }
            }
        }
        जर (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            शून्य * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            परत ptr;
        }
        शून्य * ptr;
        माप_प्रकार look_ahead_size = (माप_प्रकार) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        जर (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            परत nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        परत ptr;
    }

    शून्य मुक्त(शून्य * ptr, माप_प्रकार size) override {
        साठी (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            जर (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                परत;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#जर defined(GGML_SYCL_SUPPORT_VMM)
संरचना ggml_sycl_pool_vmm : सार्वजनिक ggml_sycl_pool {
    स्थिर स्थिरांक माप_प्रकार SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    पूर्णांक           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    माप_प्रकार    pool_used = 0;
    माप_प्रकार    pool_size = 0;
    माप_प्रकार    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    संरचना mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        शून्य * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, पूर्णांक device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        जर (pool_addr == 0) {
            परत;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        साठी (स्वतः & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    शून्य * alloc(माप_प्रकार size, माप_प्रकार * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        माप_प्रकार avail = pool_size - pool_used;

        जर (size > avail) {
            // round up to the next multiple of the granularity
            माप_प्रकार reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            जर (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            जर (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            शून्य * map_ptr = nullptr;
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
                          device, (अचिन्हित दीर्घ दीर्घ) (pool_size/1024/1024),
                          (अचिन्हित दीर्घ दीर्घ) (reserve_size/1024/1024));
#endif
        }

        GGML_ASSERT(pool_addr != 0);

        शून्य * ptr = reinterpret_cast<शून्य *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (अचिन्हित दीर्घ दीर्घ) size, ptr);
#endif

        परत ptr;
    }

    शून्य मुक्त(शून्य * ptr, माप_प्रकार size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (अचिन्हित दीर्घ दीर्घ) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<शून्य *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

संरचना ggml_sycl_pool_host : सार्वजनिक ggml_sycl_pool {
    queue_ptr qptr;
    पूर्णांक       device;

    संरचना ggml_sycl_buffer {
        शून्य * ptr  = nullptr;
        माप_प्रकार size = 0;
    };

    // Set arbitrarly to 64
    स्थिर constexpr पूर्णांक          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    माप_प्रकार                        pool_size   = 0;
    पूर्णांक                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, पूर्णांक device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        साठी (पूर्णांक i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            जर (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    शून्य * alloc(माप_प्रकार size, माप_प्रकार * actual_size) override {
        जर (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            शून्य * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *) sycl::malloc_host(size, *qptr)));
            जर (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                परत nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            परत ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        जर (b.ptr != nullptr) {
            जर (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                शून्य * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                परत ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        शून्य * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *) sycl::malloc_host(size, *qptr)));
        जर (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            परत nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        परत ptr;
    }

    शून्य मुक्त(शून्य * ptr, माप_प्रकार size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        साठी (पूर्णांक i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            जर (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                परत;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, पूर्णांक device) {
    // return pool for the host to speed up memory management
    परत std::unique_ptr<ggml_sycl_pool>(नवीन ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, पूर्णांक device) {
#जर defined(GGML_SYCL_SUPPORT_VMM)
    जर (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        परत std::unique_ptr<ggml_sycl_pool>(नवीन ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    परत std::unique_ptr<ggml_sycl_pool>(नवीन ggml_sycl_pool_leg(qptr, device));
}

/// kernels
प्रकारव्याख्या शून्य (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    स्थिरांक ggml_tensor *src0, स्थिरांक ggml_tensor *src1, ggml_tensor *dst,
    स्थिरांक अक्षर *src0_dd_i, स्थिरांक दशांश *src1_ddf_i, स्थिरांक अक्षर *src1_ddq_i,
    दशांश *dst_dd_i, स्थिरांक int64_t row_खालचा, स्थिरांक int64_t row_वरचा,
    स्थिरांक int64_t src1_ncols, स्थिरांक int64_t src1_padded_row_size,
    स्थिरांक queue_ptr &stream);



स्थिर शून्य mul_mat_p021_f16_f32(
    स्थिरांक शून्य * __restrict__ vx, स्थिरांक दशांश * __restrict__ y, दशांश * __restrict__ dst,
    स्थिरांक पूर्णांक ncols_x, स्थिरांक पूर्णांक nrows_x, स्थिरांक पूर्णांक nchannels_x, स्थिरांक पूर्णांक nchannels_y,
    स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक sycl::half *x = (स्थिरांक sycl::half *)vx;

    स्थिरांक पूर्णांक row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    स्थिरांक पूर्णांक channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    स्थिरांक पूर्णांक channel_x = channel / (nchannels_y / nchannels_x);

    स्थिरांक पूर्णांक nrows_y = ncols_x;
    स्थिरांक पूर्णांक nrows_dst = nrows_x;
    स्थिरांक पूर्णांक row_dst = row_x;

    दशांश tmp = 0.0f;

    साठी (पूर्णांक col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        स्थिरांक पूर्णांक col_x = col_x0 + item_ct1.get_local_id(2);

        जर (col_x >= ncols_x) {
            थांबा;
        }

        // x is transposed and permuted
        स्थिरांक पूर्णांक ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        स्थिरांक दशांश xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<दशांश, sycl::rounding_mode::automatic>()[0];

        स्थिरांक पूर्णांक row_y = col_x;


        // y is not transposed but permuted
        स्थिरांक पूर्णांक iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    स्थिरांक पूर्णांक idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    साठी (पूर्णांक mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    जर (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

स्थिर शून्य mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    स्थिरांक शून्य * __restrict__ vx, स्थिरांक दशांश * __restrict__ y, दशांश * __restrict__ dst, स्थिरांक पूर्णांक ncols_x, स्थिरांक पूर्णांक nrows_x,
    स्थिरांक पूर्णांक row_stride_x, स्थिरांक पूर्णांक channel_stride_x,स्थिरांक पूर्णांक channel_stride_y, स्थिरांक पूर्णांक channel_x_भाजक,
    स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक sycl::half *x = (स्थिरांक sycl::half *)vx;

    स्थिरांक पूर्णांक row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    स्थिरांक पूर्णांक channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    स्थिरांक पूर्णांक channel_x = channel / channel_x_भाजक;

    स्थिरांक पूर्णांक nrows_dst = nrows_x;
    स्थिरांक पूर्णांक row_dst   = row_x;

    स्थिरांक पूर्णांक idst = channel*nrows_dst + row_dst;

    दशांश tmp = 0.0f;

    साठी (पूर्णांक col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        स्थिरांक पूर्णांक col_x = col_x0 + item_ct1.get_local_id(2);

        जर (col_x >= ncols_x) {
            थांबा;
        }

        स्थिरांक पूर्णांक row_y = col_x;

        स्थिरांक पूर्णांक ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        स्थिरांक पूर्णांक iy = channel * channel_stride_y + row_y;

        स्थिरांक दशांश xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<दशांश, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    साठी (पूर्णांक mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    जर (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

स्थिर शून्य k_sum_rows_f32(स्थिरांक दशांश * x, दशांश * dst, स्थिरांक पूर्णांक ncols,
                           स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक पूर्णांक row = item_ct1.get_group(1);
    स्थिरांक पूर्णांक col = item_ct1.get_local_id(2);

    दशांश sum = 0.0f;
    साठी (पूर्णांक i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    जर (col == 0) {
        dst[row] = sum;
    }
}


ढाँचा<typename T>
स्थिर ओळीत शून्य ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

ढाँचा <ggml_sort_order order>
__dpct_inline__ स्थिर शून्य
k_argsort_f32_i32(स्थिरांक दशांश *x, पूर्णांक *dst, स्थिरांक पूर्णांक ncols, पूर्णांक ncols_pad,
                  स्थिरांक पूर्णांक tasks_per_thread, स्थिरांक sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    पूर्णांक col_index =  item_ct1.get_local_id(2);
    पूर्णांक row = item_ct1.get_group(1);

    साठी (पूर्णांक i = 0; i < tasks_per_thread; i++) {
        पूर्णांक col = col_index * tasks_per_thread + i;
        जर (col >= ncols_pad) {
            परत;
        }
    }

    स्थिरांक दशांश * x_row = x + row * ncols;
    स्वतः dst_row = (पूर्णांक *)dpct_local;

    // initialize indices
    साठी (पूर्णांक i=0;i<tasks_per_thread;i++){
        पूर्णांक col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    साठी (पूर्णांक k = 2; k <= ncols_pad; k *= 2) {
        साठी (पूर्णांक j = k / 2; j > 0; j /= 2) {
            साठी (पूर्णांक i = 0; i < tasks_per_thread; i++) {
                पूर्णांक col = col_index * tasks_per_thread + i;
                पूर्णांक ixj = col ^ j;
                जर (ixj > col) {
                    जर ((col & k) == 0) {
                        जर (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } अन्यथा {
                        जर (dst_row[ixj] >= ncols ||
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
    साठी (पूर्णांक i = 0; i < tasks_per_thread; i++) {
        पूर्णांक col = col_index * tasks_per_thread + i;
        जर (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

स्थिर शून्य diag_mask_inf_f32(स्थिरांक दशांश * x, दशांश * dst, स्थिरांक पूर्णांक ncols, स्थिरांक पूर्णांक rows_per_channel, स्थिरांक पूर्णांक n_past,
                              स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक पूर्णांक col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    स्थिरांक पूर्णांक row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    जर (col >= ncols) {
        परत;
    }

    स्थिरांक पूर्णांक i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * भग्न_उच्च;
}

स्थिर शून्य scale_f32(स्थिरांक दशांश * x, दशांश * dst, स्थिरांक दशांश scale, स्थिरांक दशांश bias, स्थिरांक पूर्णांक k,
                      स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक पूर्णांक i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    जर (i >= k) {
        परत;
    }

    dst[i] = scale * x[i] + bias;
}


स्थिर शून्य ggml_mul_mat_p021_f16_f32_sycl(स्थिरांक शून्य *vx, स्थिरांक दशांश *y,
                                           दशांश *dst, स्थिरांक पूर्णांक ncols_x,
                                           स्थिरांक पूर्णांक nrows_x,
                                           स्थिरांक पूर्णांक nchannels_x,
                                           स्थिरांक पूर्णांक nchannels_y,
                                           queue_ptr stream) {

    स्थिरांक sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    स्थिरांक sycl::range<3> block_dims(1, 1, WARP_SIZE);
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

स्थिर शून्य ggml_mul_mat_vec_nc_f16_f32_sycl(
    स्थिरांक शून्य *vx, स्थिरांक दशांश *y, दशांश *dst, स्थिरांक पूर्णांक ncols_x,
    स्थिरांक पूर्णांक nrows_x, स्थिरांक पूर्णांक row_stride_x, स्थिरांक पूर्णांक nchannels_x,
    स्थिरांक पूर्णांक nchannels_y, स्थिरांक पूर्णांक channel_stride_x, स्थिरांक पूर्णांक channel_stride_y, queue_ptr stream) {

    स्थिरांक sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    स्थिरांक sycl::range<3> block_dims(1, 1, WARP_SIZE);
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



स्थिर शून्य scale_f32_sycl(स्थिरांक दशांश *x, दशांश *dst, स्थिरांक दशांश scale, स्थिरांक दशांश bias,
                           स्थिरांक पूर्णांक k, queue_ptr stream) {
    स्थिरांक पूर्णांक अंक_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


स्थिर शून्य sum_rows_f32_sycl(स्थिरांक दशांश *x, दशांश *dst, स्थिरांक पूर्णांक ncols,
                              स्थिरांक पूर्णांक nrows, queue_ptr stream) {
    स्थिरांक sycl::range<3> block_dims(1, 1, WARP_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

स्थिर पूर्णांक next_power_of_2(पूर्णांक x) {
    पूर्णांक n = 1;
    पर्यंत (n < x) {
        n *= 2;
    }
    परत n;
}

स्थिर शून्य init_argsort_indices_padded(
        पूर्णांक * idx,
        स्थिरांक पूर्णांक nrows,
        स्थिरांक पूर्णांक ncols_pad,
        स्थिरांक sycl::nd_item<1> & item_ct1) {
    स्थिरांक माप_प्रकार gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिरांक माप_प्रकार total = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;

    जर (gid >= total) {
        परत;
    }

    idx[gid] = (पूर्णांक) (gid % (माप_प्रकार) ncols_pad);
}

ढाँचा <ggml_sort_order order>
स्थिर शून्य argsort_f32_i32_global_pass(स्थिरांक दशांश *            x,
                                        पूर्णांक *                    idx,
                                        स्थिरांक पूर्णांक                ncols,
                                        स्थिरांक पूर्णांक                nrows,
                                        स्थिरांक पूर्णांक                ncols_pad,
                                        स्थिरांक पूर्णांक                j,
                                        स्थिरांक पूर्णांक                k,
                                        स्थिरांक sycl::nd_item<1> & item_ct1) {
    स्थिरांक माप_प्रकार gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिरांक माप_प्रकार total = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;

    जर (gid >= total) {
        परत;
    }

    स्थिरांक पूर्णांक row = (पूर्णांक) (gid / (माप_प्रकार) ncols_pad);
    स्थिरांक पूर्णांक col = (पूर्णांक) (gid % (माप_प्रकार) ncols_pad);
    स्थिरांक पूर्णांक ixj = col ^ j;

    जर (ixj <= col || ixj >= ncols_pad) {
        परत;
    }

    स्थिरांक माप_प्रकार base  = (माप_प्रकार) row * (माप_प्रकार) ncols_pad;
    स्थिरांक माप_प्रकार pos_a = base + (माप_प्रकार) col;
    स्थिरांक माप_प्रकार pos_b = base + (माप_प्रकार) ixj;

    स्थिरांक पूर्णांक a = idx[pos_a];
    स्थिरांक पूर्णांक b = idx[pos_b];

    तार्किक do_swap = असत्य;

    जर ((col & k) == 0) {
        जर (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] > x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b] :
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] < x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b]))) {
            do_swap = सत्य;
        }
    } अन्यथा {
        जर (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] < x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b] :
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] > x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b]))) {
            do_swap = सत्य;
        }
    }

    जर (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

स्थिर शून्य copy_argsort_indices_unpadded(स्थिरांक पूर्णांक *              idx_padded,
                                          पूर्णांक *                    dst,
                                          स्थिरांक पूर्णांक                nrows,
                                          स्थिरांक पूर्णांक                ncols,
                                          स्थिरांक पूर्णांक                ncols_pad,
                                          स्थिरांक sycl::nd_item<1> & item_ct1) {
    स्थिरांक माप_प्रकार gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिरांक माप_प्रकार total = (माप_प्रकार) nrows * (माप_प्रकार) ncols;

    जर (gid >= total) {
        परत;
    }

    स्थिरांक पूर्णांक row = (पूर्णांक) (gid / (माप_प्रकार) ncols);
    स्थिरांक पूर्णांक col = (पूर्णांक) (gid % (माप_प्रकार) ncols);

    dst[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) col] = idx_padded[(माप_प्रकार) row * (माप_प्रकार) ncols_pad + (माप_प्रकार) col];
}

स्थिर शून्य argsort_f32_i32_sycl(स्थिरांक दशांश *x, पूर्णांक *dst, स्थिरांक पूर्णांक ncols,
                                 स्थिरांक पूर्णांक nrows, ggml_sort_order order,
                                 queue_ptr stream, पूर्णांक device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    स्थिरांक पूर्णांक ncols_pad = next_power_of_2(ncols);
    स्थिरांक माप_प्रकार shared_mem = (माप_प्रकार) ncols_pad * आकार(पूर्णांक);
    स्थिरांक माप_प्रकार smpbo = ggml_sycl_info().devices[device].smpbo;

    जर (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<पूर्णांक> idx_padded_alloc(pool, (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad);
        पूर्णांक *                     idx_padded = idx_padded_alloc.get();

        constexpr माप_प्रकार block_size     = 256;
        स्थिरांक माप_प्रकार     total_padded   = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;
        स्थिरांक माप_प्रकार     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        साठी (पूर्णांक k = 2; k <= ncols_pad; k *= 2) {
            साठी (पूर्णांक j = k / 2; j > 0; j /= 2) {
                जर (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } अन्यथा जर (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } अन्यथा {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        स्थिरांक माप_प्रकार total   = (माप_प्रकार) nrows * (माप_प्रकार) ncols;
        स्थिरांक माप_प्रकार nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        परत;
    }

    पूर्णांक nth = 1;
    पूर्णांक max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    पर्यंत (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    जर (nth > max_block_size)
        nth = max_block_size;

    स्थिरांक पूर्णांक tasks_per_thread = ncols_pad / nth;

    स्थिरांक sycl::range<3> block_dims(1, 1, nth);
    स्थिरांक sycl::range<3> block_nums(1, nrows, 1);

    जर (order == GGML_SORT_ORDER_ASC) {
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
    } अन्यथा जर (order == GGML_SORT_ORDER_DESC) {
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
    } अन्यथा {
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
स्थिर शून्य शीर्ष_k_scan_merge_f32(
    स्थिरांक दशांश *   src_vals,
    स्थिरांक int32_t * src_map,
    स्थिरांक पूर्णांक       begin,
    स्थिरांक पूर्णांक       end,
    स्थिरांक पूर्णांक       k,
    स्थिरांक पूर्णांक       block_size,
    दशांश *         shared_vals,
    पूर्णांक *           shared_idx,
    दशांश *         out_vals,
    int32_t *       out_idx,
    स्थिरांक तार्किक      swap01,
    स्थिरांक sycl::nd_item<1> & item_ct1
) {
    स्थिरांक पूर्णांक tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    दशांश * lv = shared_vals + tid;
    पूर्णांक * li = shared_idx + tid;

    साठी (पूर्णांक i = 0; i < k; i++) {
        lv[i * block_size] = -भग्न_उच्च;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    दशांश kth = -भग्न_उच्च;

    साठी (पूर्णांक col = begin + tid; col < end; col += block_size) {
        दशांश val = src_vals[col];

        जर (val > kth) {
            पूर्णांक pos = k - 1;
            पर्यंत (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            साठी (पूर्णांक i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    जर (tid != 0) {
        परत;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    दशांश * fv = shared_vals + (माप_प्रकार) k * block_size;
    पूर्णांक * fi = shared_idx + (माप_प्रकार) k * block_size;

    साठी (पूर्णांक i = 0; i < k; i++) {
        fv[i] = -भग्न_उच्च;
        fi[i] = -1;
    }

    दशांश fkth = -भग्न_उच्च;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    साठी (पूर्णांक t = 0; t < block_size; t++) {
        साठी (पूर्णांक i = 0; i < k; i++) {
            दशांश val = shared_vals[i * block_size + t];

            जर (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                थांबा;
            }

            पूर्णांक idx = shared_idx[i * block_size + t];

            पूर्णांक pos = k - 1;
            पर्यंत (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            साठी (पूर्णांक j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    जर (out_vals) {
        साठी (पूर्णांक i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    साठी (पूर्णांक i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    जर (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

स्थिर शून्य शीर्ष_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    स्थिरांक दशांश * src,
    int32_t * dst_indices,
    स्थिरांक int64_t ncols,
    स्थिरांक int64_t nrows,
    स्थिरांक पूर्णांक k,
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
    constexpr पूर्णांक split_block = 128;
    constexpr पूर्णांक max_splits  = 128;
    constexpr पूर्णांक min_cols    = 8192;

    पूर्णांक nsplit = 1;
    जर (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        स्थिरांक int64_t want = ncols / split_block;
        nsplit = (पूर्णांक) (want > max_splits ? max_splits : want);
    }

    जर (nsplit > 1) {
        स्थिरांक पूर्णांक nchunk = (पूर्णांक) ((ncols + nsplit - 1) / nsplit);
        स्थिरांक माप_प्रकार ncand = (माप_प्रकार) nrows * nsplit * k;

        ggml_sycl_pool_alloc<दशांश>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        दशांश *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        स्थिरांक sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<दशांश, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    स्थिरांक पूर्णांक grp = item_ct1.get_group(0);
                    स्थिरांक पूर्णांक row = grp / nsplit;
                    स्थिरांक पूर्णांक part = grp % nsplit;

                    स्थिरांक पूर्णांक begin = part * nchunk;
                    पूर्णांक end = begin + nchunk;
                    जर (end > (पूर्णांक) ncols) {
                        end = (पूर्णांक) ncols;
                    }

                    शीर्ष_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (माप_प्रकार) grp * k, pi + (माप_प्रकार) grp * k, असत्य, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<दशांश, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    स्थिरांक पूर्णांक row = item_ct1.get_group(0);
                    स्थिरांक माप_प्रकार off = (माप_प्रकार) row * nsplit * k;

                    शीर्ष_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, सत्य, item_ct1);
                });
        });

        परत;
    }

    स्थिरांक पूर्णांक block_size = 128;

    स्थिरांक sycl::range<1> block_dims(block_size);
    स्थिरांक sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<दशांश, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                स्थिरांक पूर्णांक row = item_ct1.get_group(0);

                जर (row >= nrows) परत;

                शीर्ष_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (पूर्णांक) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, सत्य, item_ct1);
            });
    });
}

स्थिर शून्य argmax_f32_i32_sycl(स्थिरांक दशांश *x, पूर्णांक *dst, स्थिरांक पूर्णांक ncols,
                               स्थिरांक पूर्णांक nrows, queue_ptr stream) {
    स्थिरांक sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, nrows, 1);
    स्थिरांक माप_प्रकार shared_mem = 256 * आकार(दशांश);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<दशांश, 1> shared_data(
            sycl::range<1>(shared_mem/आकार(दशांश)), cgh);
        sycl::local_accessor<पूर्णांक, 1> shared_indices(
            sycl::range<1>(shared_mem/आकार(दशांश)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                स्थिरांक पूर्णांक tid = item_ct1.get_local_id(2);
                स्थिरांक पूर्णांक row = item_ct1.get_global_id(1);

                दशांश max_val = -अनन्त;
                पूर्णांक max_idx = -1;

                साठी (पूर्णांक col = tid; col < ncols; col += 256) {
                    दशांश val = x[row * ncols + col];
                    जर (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                साठी (पूर्णांक stride = 256/2; stride > 0; stride >>= 1) {
                    जर (tid < stride) {
                        दशांश val1 = shared_data[tid];
                        दशांश val2 = shared_data[tid + stride];
                        जर (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                जर (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
स्थिर शून्य diag_mask_inf_f32_sycl(स्थिरांक दशांश *x, दशांश *dst,
                                   स्थिरांक पूर्णांक ncols_x, स्थिरांक पूर्णांक nrows_x,
                                   स्थिरांक पूर्णांक rows_per_channel, स्थिरांक पूर्णांक n_past,
                                   queue_ptr stream) {
    स्थिरांक sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    स्थिरांक पूर्णांक block_अंक_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    स्थिरांक sycl::range<3> block_nums(1, block_अंक_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

स्थिर dpct::err0 ggml_sycl_cpy_tensor_2d(शून्य *dst,
                                          स्थिरांक संरचना ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_खालचा, int64_t i1_वरचा,
                                          queue_ptr stream) प्रयत्न {

    dpct::memcpy_direction kind;
    अक्षर * src_ptr;
    जर (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (अक्षर *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } अन्यथा जर (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (अक्षर *) src->data;
    } अन्यथा जर (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_खालचा == 0 && i1_वरचा == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        पूर्णांक id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (अक्षर *) extra->data_device[id];
    } अन्यथा {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    अक्षर * dst_ptr = (अक्षर *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    स्थिरांक गणना ggml_type type = src->type;
    स्थिरांक int64_t ts = ggml_type_size(type);
    स्थिरांक int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_वरचा - i1_खालचा;

    स्थिरांक अक्षर * x = src_ptr + i1_खालचा*nb1 + i2*nb2 + i3*nb3;
    जर (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        परत CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } अन्यथा जर (nb0 == ts) {
        परत CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } अन्यथा {
        साठी (int64_t i1 = 0; i1 < i1_diff; i1++) {
            स्थिरांक शून्य * rx = (स्थिरांक शून्य *) ((स्थिरांक अक्षर *) x + i1*nb1);
            शून्य * rd = (शून्य *) (dst_ptr + i1*ts*ne0/bs);
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
            जर (r != 0) परत r;
        }
        परत 0;
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

ओळीत शून्य ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    स्थिरांक ggml_tensor *src0, स्थिरांक ggml_tensor *src1, ggml_tensor *dst,
    स्थिरांक अक्षर *src0_dd_i, स्थिरांक दशांश *src1_ddf_i, स्थिरांक अक्षर *src1_ddq_i,
    दशांश *dst_dd_i, स्थिरांक int64_t row_खालचा, स्थिरांक int64_t row_वरचा,
    स्थिरांक int64_t src1_ncols, स्थिरांक int64_t src1_padded_row_size,
    स्थिरांक queue_ptr &stream) प्रयत्न {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    स्थिरांक int64_t row_diff = row_वरचा - row_खालचा;

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    स्थिरांक int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    पूर्णांक ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    तार्किक use_fp16 = सत्य;  // TODO(Yu) SYCL capability check
#अन्यथा
    तार्किक use_fp16 = असत्य;
#endif

#जर GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    जर (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        जर (src1->type != GGML_TYPE_BF16) {
            स्थिरांक to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } अन्यथा {
            stream->स_नकल(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*आकार(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<दशांश>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        परत;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    तार्किक src0_reordered = असत्य;
    जर (स्थिरांक स्वतः * src0_extra = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    जर (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        परत;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    जर ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        जर (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            स्थिरांक to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            माप_प्रकार ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        स्थिरांक sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (स्थिरांक sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        जर (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            स्थिरांक to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            माप_प्रकार ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        स्थिरांक sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (स्थिरांक sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#जर GGML_SYCL_DNNL
        जर (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<दशांश>(), stream);
        }
        अन्यथा
#endif
        {
            स्थिरांक दशांश alpha = 1.0f;
            स्थिरांक दशांश beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } अन्यथा {
        ggml_sycl_pool_alloc<दशांश> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<दशांश> src1_ddq_as_f32(ctx.pool());
        जर (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            स्थिरांक to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        जर (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            स्थिरांक to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        स्थिरांक दशांश * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (स्थिरांक दशांश *) src0_dd_i : src0_ddq_as_f32.get();
        स्थिरांक दशांश * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (स्थिरांक दशांश *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#जर GGML_SYCL_DNNL
            स्थिरांक int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            स्थिरांक तार्किक use_mkl_direct = gemm_flops < 256 * 256 * 256;
            जर (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<दशांश>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<दशांश>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<दशांश>(), stream);
            }
            अन्यथा
#endif
            {
                स्थिरांक दशांश alpha = 1.0f;
                स्थिरांक दशांश beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_मूल्य(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_मूल्य(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

ओळीत शून्य ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    स्थिरांक int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

ओळीत शून्य ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    स्थिरांक int64_t ncols = dst->src[0]->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

ओळीत शून्य ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    स्थिरांक int64_t ncols = dst->src[0]->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(dst->src[0]);
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


ओळीत शून्य ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    स्थिरांक int64_t ncols = dst->src[0]->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(dst->src[0]);

    गणना ggml_sort_order order = (गणना ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (पूर्णांक *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

स्थिर शून्य ggml_sycl_op_शीर्ष_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    स्थिरांक पूर्णांक k = dst->ne[0];
    स्थिरांक int64_t ncols = src0->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    जर (k <= SYCL_शीर्ष_K_SCAN_MERGE_MAX_K) {
        शीर्ष_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } अन्यथा {
        ggml_sycl_शीर्ष_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

ओळीत शून्य ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    स्थिरांक int64_t ncols = dst->src[0]->ne[0];
    स्थिरांक int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

ओळीत शून्य ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    स्थिरांक int64_t ne00 = dst->src[0]->ne[0];
    स्थिरांक int64_t ne01 = dst->src[0]->ne[1];
    स्थिरांक पूर्णांक nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    स्थिरांक पूर्णांक n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

स्थिर शून्य tri_f32_sycl(
    स्थिरांक दशांश * src,
    दशांश * dst,
    स्थिरांक int64_t ne0,
    स्थिरांक int64_t ne1,
    स्थिरांक int64_t ne2,
    स्थिरांक int64_t ne3,
    स्थिरांक ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    स्थिरांक माप_प्रकार total = (माप_प्रकार) ne0 * (माप_प्रकार) ne1 * (माप_प्रकार) ne2 * (माप_प्रकार) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        स्थिरांक int64_t idx = (int64_t) tid[0];

        स्थिरांक int64_t i0 = idx % ne0;
        स्थिरांक int64_t t1 = idx / ne0;
        स्थिरांक int64_t i1 = t1 % ne1;

        तार्किक keep = असत्य;
        निवड (ttype) {
            स्थिती GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); थांबा;
            स्थिती GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); थांबा;
            स्थिती GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); थांबा;
            स्थिती GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); थांबा;
            पूर्वनिर्धारित: keep = असत्य; थांबा;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

स्थिर शून्य ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(src0->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    स्थिरांक ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    स्थिरांक int64_t ne0 = src0->ne[0];
    स्थिरांक int64_t ne1 = src0->ne[1];
    स्थिरांक int64_t ne2 = src0->ne[2];
    स्थिरांक int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


ओळीत शून्य ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिरांक दशांश * src0_dd = static_cast<स्थिरांक दशांश *>(dst->src[0]->data);
    दशांश *       dst_dd  = static_cast<दशांश *>(dst->data);

    दशांश scale;
    दशांश bias;
    स_नकल(&scale, (दशांश *) dst->op_params + 0, आकार(दशांश));
    स_नकल(&bias,  (दशांश *) dst->op_params + 1, आकार(दशांश));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

स्थिर शून्य ggml_sycl_set_peer_access(स्थिरांक पूर्णांक n_tokens, [[maybe_unused]] पूर्णांक main_device) {
    स्थिर तार्किक peer_access_enabled = असत्य;

    स्थिरांक तार्किक enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    जर (peer_access_enabled == enable_peer_access) {
        परत;
    }

#ifdef न_संशोधन
    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        साठी (पूर्णांक id_other = 0; id_other < ggml_sycl_info().device_मोजणी; ++id_other) {
            जर (i == id_other) {
                सुरूठेवा;
            }
            जर (i != main_device && id_other != main_device) {
                सुरूठेवा;
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

ढाँचा <ढाँचा <पूर्णांक> typename quantize_f>
स्थिर शून्य ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor *src0,
                                 स्थिरांक ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) प्रयत्न {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    स्थिरांक int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    स्थिरांक int64_t ne0 = dst->ne[0];
    स्थिरांक int64_t ne1 = dst->ne[1];

    स्थिरांक पूर्णांक nb2 = dst->nb[2];
    स्थिरांक पूर्णांक nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    स्थिरांक int64_t i02_भाजक = ne12 / ne02;

    स्थिरांक माप_प्रकार src0_ts = ggml_type_size(src0->type);
    स्थिरांक माप_प्रकार src0_bs = ggml_blck_size(src0->type);
    स्थिरांक माप_प्रकार q8_1_ts = आकार(block_q8_1);
    स्थिरांक माप_प्रकार q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    स्थिरांक तार्किक src0_is_contiguous = ggml_is_contiguous(src0);
    स्थिरांक तार्किक src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    स्थिरांक तार्किक split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<दशांश, GGML_SYCL_MAX_DEVICES> tensor_split;
    जर (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    संरचना dev_data {
        ggml_sycl_pool_alloc<अक्षर> src0_dd_alloc;
        ggml_sycl_pool_alloc<दशांश> src1_ddf_alloc;
        ggml_sycl_pool_alloc<अक्षर> src1_ddq_alloc;
        ggml_sycl_pool_alloc<दशांश> dst_dd_alloc;

        अक्षर *src0_dd = nullptr;
        दशांश *src1_ddf = nullptr; // float
        अक्षर *src1_ddq = nullptr;  // q8_1
        दशांश *dst_dd = nullptr;

        int64_t row_खालचा;
        int64_t row_वरचा;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    पूर्णांक used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        // by default, use all rows
        dev[i].row_खालचा  = 0;
        dev[i].row_वरचा = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        जर (split) {
            स्थिरांक int64_t rounding = get_row_rounding(src0->type, tensor_split);

            जर (i != 0) {
                dev[i].row_खालचा  = ne01*tensor_split[i];
                जर (dev[i].row_खालचा < ne01) {
                    dev[i].row_खालचा -= dev[i].row_खालचा % rounding;
                }
            }

            जर (i != ggml_sycl_info().device_मोजणी - 1) {
                dev[i].row_वरचा  = ne01*tensor_split[i + 1];
                जर (dev[i].row_वरचा < ne01) {
                    dev[i].row_वरचा -= dev[i].row_वरचा % rounding;
                }
            }
        }
    }

    constexpr तार्किक quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
        जर ((!split && i != ctx.device) || dev[i].row_खालचा == dev[i].row_वरचा) {
            सुरूठेवा;
        }

        used_devices++;

        स्थिरांक तार्किक src1_on_device = i == ctx.device;
        स्थिरांक तार्किक  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        जर (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (अक्षर *) src0_extra->data_device[i] : (अक्षर *) src0->data;
        } अन्यथा {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        जर (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (दशांश *) src1->data;
        } अन्यथा {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        जर constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            जर (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                प्रयत्न {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } पकडा (sycl::exception स्थिरांक &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::निकास(1);
                }
            }
        }

        जर (dst_on_device) {
            dev[i].dst_dd = (दशांश *) dst->data;
        } अन्यथा {
            स्थिरांक माप_प्रकार size_dst_ddf = split ? (dev[i].row_वरचा - dev[i].row_खालचा)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    जर (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    स्थिरांक int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    साठी (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        स्थिरांक int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        स्थिरांक int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
            जर ((!split && i != ctx.device) || dev[i].row_खालचा == dev[i].row_वरचा) {
                सुरूठेवा;
            }

            स्थिरांक तार्किक src1_on_device = i == ctx.device;
            स्थिरांक तार्किक  dst_on_device = i == ctx.device;
            स्थिरांक int64_t row_diff = dev[i].row_वरचा - dev[i].row_खालचा;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            जर (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            साठी (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                स्थिरांक int64_t i03 = i0 / ne12;
                स्थिरांक int64_t i02 = i0 % ne12;

                स्थिरांक माप_प्रकार src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                अक्षर  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_भाजक) * (ne01*ne00*src0_ts)/src0_bs;
                दशांश * src1_ddf_i = (दशांश *) ((अक्षर *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                अक्षर  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                दशांश *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                जर (i == ctx.device) {
                    dst_dd_i += dev[i].row_खालचा; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                जर (src1_is_contiguous) {
                    जर (i != ctx.device) {
                        जर constexpr (quantize_enabled) {
                            अक्षर * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->स_नकल(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .रुको()));
                        } अन्यथा {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            स्थिरांक अक्षर * src1_ddf_i_source = (स्थिरांक अक्षर *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } अन्यथा {
                    जर (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } अन्यथा {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    जर constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        प्रयत्न {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } पकडा (स्थिरांक sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::निकास(1);
                        }
                    }
                }

                जर (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_भाजक == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_भाजक, dev[i].row_खालचा, dev[i].row_वरचा, stream));
                }
                जर (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_खालचा, dev[i].row_वरचा, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                जर (!dst_on_device) {
                    शून्य * dst_off_device = dst->data;
                    जर (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        दशांश * dhf_dst_i = (दशांश *) ((अक्षर *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*आकार(दशांश));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_खालचा;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * आकार(दशांश), dst_dd_i,
                            row_diff * आकार(दशांश), row_diff * आकार(दशांश),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } अन्यथा {
                        दशांश * dhf_dst_i = (दशांश *) ((अक्षर *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*आकार(दशांश));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->स_नकल(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * आकार(दशांश)).रुको()));
                    }
                }

                // add event for the main device to wait on until other device is done
                जर (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    जर (split && ggml_sycl_info().device_मोजणी > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; ++i) {
            जर (dev[i].row_खालचा == dev[i].row_वरचा) {
                सुरूठेवा;
            }
            साठी (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

स्थिर शून्य ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

स्थिर शून्य ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

स्थिर शून्य ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

स्थिर शून्य ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor *src0,
                                       स्थिरांक ggml_tensor *src1,
                                       ggml_tensor *dst) प्रयत्न {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];
    स्थिरांक int64_t ne02 = src0->ne[2];

    स्थिरांक int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    शून्य  * src0_ddq = src0->data;
    दशांश * src1_ddf = (दशांश *) src1->data;
    दशांश * dst_ddf  = (दशांश *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor *src0,
                                     स्थिरांक ggml_tensor *src1,
                                     ggml_tensor *dst) प्रयत्न {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];
    स्थिरांक int64_t ne02 = src0->ne[2];

    स्थिरांक int64_t nb01 = src0->nb[1];
    स्थिरांक int64_t nb02 = src0->nb[2];

    स्थिरांक int64_t ne12 = src1->ne[2];
    स्थिरांक int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    शून्य  * src0_ddq = src0->data;
    दशांश * src1_ddf = (दशांश *) src1->data;
    दशांश * dst_ddf  = (दशांश *) dst->data;

    स्थिरांक int64_t row_stride_x = nb01 / आकार(sycl::half);
    स्थिरांक int64_t channel_stride_x = nb02 / आकार(sycl::half);
    स्थिरांक int64_t channel_stride_y = nb11 / आकार(दशांश);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य k_compute_batched_ptrs(स्थिरांक sycl::half * src0_as_f16, स्थिरांक sycl::half * src1_as_f16, शून्य * dst,
                                   स्थिरांक शून्य ** ptrs_src, शून्य ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   माप_प्रकार nb02, माप_प्रकार nb03, माप_प्रकार nb12, माप_प्रकार nb13, माप_प्रकार nbd2, माप_प्रकार nbd3,
                                   int64_t r2, int64_t r3, स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    स्थिरांक int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    जर (i13 >= ne13 || i12 >= ne12) {
        परत;
    }

    स्थिरांक int64_t i03 = i13 / r3;
    स्थिरांक int64_t i02 = i12 / r2;

    स्थिरांक uint8_t * src0_bytes = reinterpret_cast<स्थिरांक uint8_t *>(src0_as_f16);
    स्थिरांक uint8_t * src1_bytes = reinterpret_cast<स्थिरांक uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

स्थिर शून्य ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor * src0,
                                           स्थिरांक ggml_tensor * src1, ggml_tensor * dst) प्रयत्न {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    स्थिरांक sycl::half * src0_f16 = static_cast<स्थिरांक sycl::half *>(src0->data);
    दशांश *            dst_ddf  = static_cast<दशांश *>(dst->data);

    स्थिरांक sycl::half * src1_f16       = static_cast<स्थिरांक sycl::half *>(src1->data);
#जर GGML_SYCL_DNNL
    स्थिरांक माप_प्रकार       type_size_src0 = ggml_type_size(src0->type);
#endif
    स्थिरांक माप_प्रकार       type_size_src1 = ggml_type_size(src1->type);

    तार्किक is_src0_cont_2 = ggml_is_contiguous_2(src0);
    तार्किक is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    जर (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#जर GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        पूर्णांक last_dim=0;
        पूर्णांक last_str=0;
        माप_प्रकार largest_str=0;
        साठी(पूर्णांक i = 0; i< 4; i++){
            // last stride is always the largest
            जर(src1->nb[i] == largest_str){
                जर(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            जर(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        स्थिरांक int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        स्थिरांक to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# अन्यथा
        स्थिरांक int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        स्थिरांक to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = सत्य;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    माप_प्रकार nbd2 = dst->nb[2];
    माप_प्रकार nbd3 = dst->nb[3];

    स्थिरांक दशांश alpha_f32 = 1.0f;
    स्थिरांक दशांश beta_f32  = 0.0f;

    स्थिरांक शून्य * alpha = &alpha_f32;
    स्थिरांक शून्य * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

#जर GGML_SYCL_DNNL
    जर (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            स्वतः launch_gemm_for_batches = [&ctx, queue](स्थिरांक sycl::half *src0,
                                                स्थिरांक sycl::half *src1, दशांश *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                तार्किक supported_broadcast = batchb == batcha ? सत्य
                        : batchb == 1 || batcha == 1        ? सत्य
                                                            : असत्य;
                जर (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<दशांश>(), queue, batcha, batchb);
                } अन्यथा {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    जर (batches0 > batches1) {
                        int64_t अंक_mul_mats = batches1;
                        int64_t sub_batch = batches0 / अंक_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        साठी (int64_t i0 = 0; i0 < अंक_mul_mats; i0++) {
                            स्थिरांक sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            स्थिरांक sycl::half *src1_shifted = src1 + (sb2 * i0);
                            दशांश *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<दशांश>(),
                                    queue, sub_batch, 1);
                        }
                    } अन्यथा {
                        int64_t अंक_mul_mats = batches0;
                        int64_t sub_batch = batches1 / अंक_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        साठी (int64_t i1 = 0; i1 < अंक_mul_mats; i1++) {
                            स्थिरांक sycl::half *src0_shifted = src0 + (sa2 * i1);
                            स्थिरांक sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            दशांश *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<दशांश>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            स्थिरांक तार्किक cont_batches_dim2_a = nb02 * ne02 == nb03;
            स्थिरांक तार्किक cont_batches_dim2_b = nb12 * ne12 == nb13;
            स्थिरांक तार्किक cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            स्थिरांक तार्किक cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            जर (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / आकार(दशांश));
            } अन्यथा जर (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / आकार(दशांश));
            } अन्यथा {
                साठी (int64_t b_a = 0; b_a < ne03; b_a++) {
                    स्थिरांक sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    स्थिरांक sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    दशांश *dst_shifted = dst_ddf + (nb3 * b_a / आकार(दशांश));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / आकार(दशांश));
                }
            }

    }
    अन्यथा
#endif
    {
        जर (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            स्थिरांक int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            स्थिरांक int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } अन्यथा {
            स्थिरांक पूर्णांक ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<स्थिरांक शून्य *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<शून्य *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<दशांश>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                स्थिरांक शून्य ** ptrs_src_get = ptrs_src.get();
                शून्य **       ptrs_dst_get = ptrs_dst.get();
                माप_प्रकार        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * आकार(sycl::half);
                माप_प्रकार        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * आकार(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (स्थिरांक शून्य **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (स्थिरांक शून्य **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (शून्य **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} पकडा (स्थिरांक sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

गणना वर्ग mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

ओळीत तार्किक ggml_sycl_supports_mmq(गणना ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    परत असत्य;
}

ओळीत तार्किक ggml_sycl_supports_reorder_mul_mat_sycl(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q8_0:
            परत सत्य;
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
            परत !g_ggml_sycl_prioritize_dmmv;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

ओळीत तार्किक ggml_sycl_supports_reorder_dmmv(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

ओळीत तार्किक ggml_sycl_supports_reorder_mmvq(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ3_S:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

स्थिर तार्किक ggml_sycl_supports_reorder_esimd(गणना ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    निवड (type) {
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_Q8_0:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
#अन्यथा
    GGML_UNUSED(type);
    परत असत्य;
#endif
}

स्थिर तार्किक ggml_sycl_supports_dmmv(गणना ggml_type type) {
    निवड (type) {
        स्थिती GGML_TYPE_Q1_0:
        स्थिती GGML_TYPE_Q4_0:
        स्थिती GGML_TYPE_Q4_1:
        स्थिती GGML_TYPE_Q5_0:
        स्थिती GGML_TYPE_Q5_1:
        स्थिती GGML_TYPE_Q8_0:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_F16:
        स्थिती GGML_TYPE_BF16:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
स्थिर ओळीत शून्य * sycl_ext_malloc_device(dpct::queue_ptr stream, माप_प्रकार size) {
    तार्किक use_async = g_ggml_sycl_use_async_mem_op;
#जर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    जर (use_async) {
        शून्य * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        परत ptr;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    परत ggml_sycl_malloc_device(size, *stream);
}

स्थिर ओळीत शून्य sycl_ext_free(dpct::queue_ptr stream, शून्य * ptr) {
    तार्किक use_async = g_ggml_sycl_use_async_mem_op;
#जर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    जर (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        परत;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
संरचना sycl_reorder_temp_buffer {
    शून्य *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, माप_प्रकार size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        जर (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            जर (ptr) {
                host_fallback = सत्य;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        जर (!ptr) {
            परत;
        }
        जर (host_fallback) {
            sycl::मुक्त(ptr, *stream);
        } अन्यथा {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit चालक तार्किक() स्थिरांक { परत ptr != nullptr; }

    sycl_reorder_temp_buffer(स्थिरांक sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & चालक=(स्थिरांक sycl_reorder_temp_buffer &) = delete;

खाजगी:
    तार्किक host_fallback = असत्य;
};

स्थिर तार्किक reorder_qw_q4_0(uint8_t * data_device, स्थिरांक पूर्णांक ncols, स्थिरांक पूर्णांक nrows, माप_प्रकार size, माप_प्रकार offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    GGML_ASSERT((size % आकार(block_q4_0) == 0));
    GGML_ASSERT((offset % आकार(block_q4_0) == 0));
    पूर्णांक offset_blks = offset / आकार(block_q4_0);
    स्वतः qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    स्वतः d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    स्वतः reorder_event = stream->parallel_for(
        size / आकार(block_q4_0),
            [=](स्वतः i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            स्थिरांक block_q4_0* x = (स्थिरांक block_q4_0*)tmp_buf;
            स्थिरांक पूर्णांक ib = i;

            साठी (पूर्णांक j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q8_0(uint8_t * data_device, स्थिरांक पूर्णांक ncols, स्थिरांक पूर्णांक nrows, माप_प्रकार size, माप_प्रकार offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    GGML_ASSERT((size % आकार(block_q8_0) == 0));
    GGML_ASSERT((offset % आकार(block_q8_0) == 0));
    पूर्णांक offset_blks = offset / आकार(block_q8_0);
    स्वतः qs_ptr = data_device + offset_blks * QK8_0;
    स्वतः d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    स्वतः reorder_event = stream->parallel_for(
        size / आकार(block_q8_0),
            [=](स्वतः i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            स्थिरांक block_q8_0* x = (स्थिरांक block_q8_0*)tmp_buf;
            स्थिरांक पूर्णांक ib = i;

            साठी (पूर्णांक j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
स्थिर तार्किक reorder_qw_q8_0_moe(uint8_t * data_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % आकार(block_q8_0) == 0);
    स्थिरांक पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / आकार(block_q8_0));
    स्थिरांक माप_प्रकार total_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, total_bytes)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिरांक पूर्णांक total_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(total_blocks, [=](स्वतः gb_) {
        स्थिरांक पूर्णांक          gb   = gb_;
        स्थिरांक पूर्णांक          e    = gb / blocks_per_expert;
        स्थिरांक पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिरांक block_q8_0 * x    = (स्थिरांक block_q8_0 *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = data_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr = base;
        स्वतः * d_ptr  = (sycl::half *) (qs_ptr + (माप_प्रकार) blocks_per_expert * QK8_0);

        साठी (पूर्णांक j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q4_k(uint8_t * data_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_q4_K) == 0);
    GGML_ASSERT(offset % आकार(block_q4_K) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr     = data_device;
    स्वतः * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_q4_K * x  = (स्थिरांक block_q4_K *) tmp_buf;
        स्थिरांक पूर्णांक          ib = i;

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        साठी (पूर्णांक j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

// Reorder each expert slice into a self-contained SoA layout.
स्थिर तार्किक reorder_qw_q4_k_moe(uint8_t * data_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % आकार(block_q4_K) == 0);
    स्थिरांक पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / आकार(block_q4_K));
    स्थिरांक माप_प्रकार total_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, total_bytes)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिरांक पूर्णांक total_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(total_blocks, [=](स्वतः gb_) {
        स्थिरांक पूर्णांक          gb   = gb_;
        स्थिरांक पूर्णांक          e    = gb / blocks_per_expert;
        स्थिरांक पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिरांक block_q4_K * x    = (स्थिरांक block_q4_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = data_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr     = base;
        स्वतः * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        साठी (पूर्णांक j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
स्थिर तार्किक reorder_qw_q5_k_moe(uint8_t * data_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % आकार(block_q5_K) == 0);
    स्थिरांक पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / आकार(block_q5_K));
    स्थिरांक माप_प्रकार total_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, total_bytes)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिरांक पूर्णांक total_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(total_blocks, [=](स्वतः gb_) {
        स्थिरांक पूर्णांक          gb   = gb_;
        स्थिरांक पूर्णांक          e    = gb / blocks_per_expert;
        स्थिरांक पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिरांक block_q5_K * x    = (स्थिरांक block_q5_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = data_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr     = base;
        स्वतः * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        स्वतः * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        साठी (पूर्णांक j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
स्थिर तार्किक reorder_qw_q6_k_moe(uint8_t * data_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % आकार(block_q6_K) == 0);
    स्थिरांक पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / आकार(block_q6_K));
    स्थिरांक माप_प्रकार total_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, total_bytes)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिरांक पूर्णांक total_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(total_blocks, [=](स्वतः gb_) {
        स्थिरांक पूर्णांक          gb   = gb_;
        स्थिरांक पूर्णांक          e    = gb / blocks_per_expert;
        स्थिरांक पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिरांक block_q6_K * x    = (स्थिरांक block_q6_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = data_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * ql_ptr     = base;
        स्वतः * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        स्वतः * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        स्वतः * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
स्थिर तार्किक reorder_qw_mxfp4_moe(uint8_t * data_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % आकार(block_mxfp4) == 0);
    स्थिरांक पूर्णांक     blocks_per_expert = (पूर्णांक) (expert_bytes / आकार(block_mxfp4));
    स्थिरांक माप_प्रकार  max_chunk_bytes   = 32u << 20;
    स्थिरांक int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (माप_प्रकार) chunk_experts * expert_bytes);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (माप_प्रकार) chunk_experts * expert_bytes);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    साठी (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        स्थिरांक int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (माप_प्रकार) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, chunk, (माप_प्रकार) n_chunk * expert_bytes)));
        जर (!g_ggml_sycl_use_async_mem_op) {
            copy_event.रुको();
        }

        स्थिरांक पूर्णांक total_blocks = blocks_per_expert * (पूर्णांक) n_chunk;
        स्वतः reorder_event = stream->parallel_for(total_blocks, [=](स्वतः gb_) {
            स्थिरांक पूर्णांक           gb   = gb_;
            स्थिरांक पूर्णांक           e    = gb / blocks_per_expert;
            स्थिरांक पूर्णांक           ib   = gb % blocks_per_expert;
            स्थिरांक block_mxfp4 * x    = (स्थिरांक block_mxfp4 *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
            uint8_t *           base = chunk + (माप_प्रकार) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (माप_प्रकार) blocks_per_expert;

            साठी (पूर्णांक j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(माप_प्रकार) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        जर (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q2_k(uint8_t * data_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_q2_K) == 0);
    GGML_ASSERT(offset % आकार(block_q2_K) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *        qs_ptr     = data_device;
    स्वतः *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_q2_K * x  = (स्थिरांक block_q2_K *) tmp_buf;
        स्थिरांक पूर्णांक          ib = i;

        साठी (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        साठी (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q3_k(uint8_t * data_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_q3_K) == 0);
    GGML_ASSERT(offset % आकार(block_q3_K) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *       qs_ptr     = data_device;
    स्वतः *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    स्वतः *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_q3_K * x  = (स्थिरांक block_q3_K *) tmp_buf;
        स्थिरांक पूर्णांक          ib = i;

        साठी (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        साठी (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        साठी (पूर्णांक j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q5_k(uint8_t * data_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_q5_K) == 0);
    GGML_ASSERT(offset % आकार(block_q5_K) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr     = data_device;
    स्वतः * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    स्वतः * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_q5_K * x  = (स्थिरांक block_q5_K *) tmp_buf;
        स्थिरांक पूर्णांक          ib = i;

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        साठी (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        साठी (पूर्णांक j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_q6_k(uint8_t * data_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_q6_K) == 0);
    GGML_ASSERT(offset % आकार(block_q6_K) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *       ql_ptr     = data_device;
    स्वतः *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    स्वतः *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_q6_K * x  = (स्थिरांक block_q6_K *) tmp_buf;
        स्थिरांक पूर्णांक          ib = i;

        स्थिरांक uint8_t * ql              = x[ib].ql;
        स्थिरांक uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        साठी (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        साठी (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_iq3_s(uint8_t * data_device, माप_प्रकार size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_iq3_s) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_iq3_s);
    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr       = data_device;
    स्वतः * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    स्वतः * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    स्वतः * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_iq3_s * x = (स्थिरांक block_iq3_s *) tmp_buf;
        स्थिरांक पूर्णांक ib = i;

        साठी (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        साठी (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (आकार(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        साठी (पूर्णांक j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[आकार(ggml_half) + j] = x[ib].scales[j];
        }
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw_iq3_xxs(uint8_t * data_device, माप_प्रकार size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % आकार(block_iq3_xxs) == 0);

    स्थिरांक पूर्णांक nblocks = size / आकार(block_iq3_xxs);
    sycl_reorder_temp_buffer tmp(stream, size);
    जर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        परत असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, data_device, size)));
    जर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr = data_device;
    स्वतः * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिरांक block_iq3_xxs * x = (स्थिरांक block_iq3_xxs *) tmp_buf;
        स्थिरांक पूर्णांक ib = i;

        साठी (पूर्णांक j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    जर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    परत सत्य;
}

स्थिर तार्किक reorder_qw(स्थिरांक ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    माप_प्रकार ncols = src0->ne[0];
    माप_प्रकार nrows = src0->ne[1];
    माप_प्रकार size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    जर (src0->ne[2] > 1) {
        GGML_ASSERT((माप_प्रकार) size == (माप_प्रकार) src0->ne[2] * src0->nb[2]);
        निवड (src0->type) {
            स्थिती GGML_TYPE_Q8_0:
                परत reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            स्थिती GGML_TYPE_Q4_K:
                परत reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            स्थिती GGML_TYPE_Q5_K:
                परत reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            स्थिती GGML_TYPE_Q6_K:
                परत reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            स्थिती GGML_TYPE_MXFP4:
                परत reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            पूर्वनिर्धारित:
                परत असत्य;
        }
    }

    निवड (src0->type) {
        स्थिती GGML_TYPE_Q4_0:
            परत reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        स्थिती GGML_TYPE_Q8_0:
            परत reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        स्थिती GGML_TYPE_Q2_K:
            परत reorder_qw_q2_k(data_device, size, 0, stream);
        स्थिती GGML_TYPE_Q3_K:
            परत reorder_qw_q3_k(data_device, size, 0, stream);
        स्थिती GGML_TYPE_Q4_K:
            परत reorder_qw_q4_k(data_device, size, 0, stream);
        स्थिती GGML_TYPE_Q5_K:
            परत reorder_qw_q5_k(data_device, size, 0, stream);
        स्थिती GGML_TYPE_Q6_K:
            परत reorder_qw_q6_k(data_device, size, 0, stream);
        स्थिती GGML_TYPE_IQ3_XXS:
            परत reorder_qw_iq3_xxs(data_device, size, stream);
        स्थिती GGML_TYPE_IQ3_S:
            परत reorder_qw_iq3_s(data_device, size, stream);
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

स्थिर तार्किक should_reorder_tensor(ggml_backend_sycl_context& ctx, स्थिरांक ggml_tensor * dst) {
    परत g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

स्थिर शून्य opt_for_reorder(ggml_backend_sycl_context * ctx, स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    जर (!should_reorder_tensor(*ctx, dst)) {
        परत;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    जर (!extra || extra->optimized_feature.reorder) {
        परत;  // Skip permutations and already reordered tensors
    }

    निवड (mm_algorithm) {
        स्थिती mul_mat_algo::DMMV:
            जर (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                परत;
            }
            थांबा;
        स्थिती mul_mat_algo::MMVQ:
            जर (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                परत;
            }
            थांबा;
        स्थिती mul_mat_algo::MUL_MAT_SYCL:
            जर (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                परत;
            }
            थांबा;
    }

    जर (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = सत्य;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
स्थिर शून्य opt_for_reorder_id(ggml_backend_sycl_context * ctx, स्थिरांक ggml_tensor * src0) {
    जर (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        परत;
    }
    जर (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        परत;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    जर (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        परत;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    जर (!extra || extra->optimized_feature.reorder) {
        परत;
    }
    जर (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = सत्य;
    }
}


स्थिर तार्किक can_use_dequantize_mul_mat_vec(स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    स्थिरांक int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    परत ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

स्थिर तार्किक can_use_mul_mat_vec_q(स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1, ggml_tensor * dst) {
    परत ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

स्थिर शून्य ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor * src0, स्थिरांक ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    जर (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        परत;
    }

    स्थिरांक तार्किक split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = पूर्णांक_उच्च;

    जर (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        स्वतः & tensor_split = buft_ctx->tensor_split;
        साठी (पूर्णांक id = 0; id < ggml_sycl_info().device_मोजणी; ++id) {
            // skip devices that are not going to do any work:
            जर (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_मोजणी ? tensor_split[id + 1] : 1.0f)) {
                सुरूठेवा;
            }

            जर (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } अन्यथा {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    तार्किक use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    तार्किक use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    तार्किक use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
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

    जर (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        तार्किक use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        जर (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    जर (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        जर (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } अन्यथा {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } अन्यथा जर (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } अन्यथा जर (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } अन्यथा जर (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } अन्यथा जर (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        जर (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } अन्यथा {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } अन्यथा जर (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } अन्यथा {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
स्थिर तार्किक ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, स्थिरांक ggml_tensor * wu,
                                             स्थिरांक ggml_tensor * wg, स्थिरांक ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    स्थिरांक स्वतः * extra_u = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wu->extra);
    स्थिरांक स्वतः * extra_g = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wg->extra);
    जर ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        परत असत्य;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    स्थिरांक स्वतः plain_type = [](ggml_type t) { परत t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    जर (!plain_type(wu->type) || !plain_type(wg->type)) {
        परत असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    स्थिरांक int64_t ne00 = wu->ne[0];
    स्थिरांक int64_t ne11 = act->ne[1];

    स्थिरांक queue_ptr stream = ctx.stream();
    स्थिरांक पूर्णांक src1_padded_cols = GGML_PAD((पूर्णांक) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
        (माप_प्रकार) ne11 * src1_padded_cols * आकार(block_q8_1) / QK8_1);
    अक्षर * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((स्थिरांक दशांश *) act->data, src1_ddq, (पूर्णांक) ne00, (पूर्णांक) ne11,
                                          src1_padded_cols, stream);

    परत ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (दशांश *) glu->data, (पूर्णांक) ne00, (पूर्णांक) wu->ne[1],
                                             (पूर्णांक) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(पूर्णांक) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
स्थिर तार्किक ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, स्थिरांक ggml_tensor * wu,
                                             स्थिरांक ggml_tensor * wg, स्थिरांक ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    स्थिरांक अक्षर * act_begin = (स्थिरांक अक्षर *) act->data;
    स्थिरांक अक्षर * glu_begin = (स्थिरांक अक्षर *) glu->data;
    जर (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        परत असत्य;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    स्थिरांक स्वतः * extra_u = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wu->extra);
    स्थिरांक स्वतः * extra_g = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wg->extra);
    जर (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        परत असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    परत ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (स्थिरांक दशांश *) act->data, (दशांश *) glu->data,
                                                              (पूर्णांक) wu->ne[0], (पूर्णांक) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
स्थिर तार्किक ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक गाठ_idx) {
    जर (!ggml_sycl_can_fuse(cgraph, गाठ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        परत असत्य;
    }

    ggml_tensor *       glu  = cgraph->nodes[गाठ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    स्थिरांक ggml_tensor * wu   = up->src[0];
    स्थिरांक ggml_tensor * wg   = gate->src[0];
    स्थिरांक ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    जर (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        परत असत्य;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    जर (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        परत सत्य;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    जर (g_ggml_sycl_prioritize_dmmv) {
        परत असत्य;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    स्थिरांक तार्किक reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    जर (!reorder_pair) {
        परत ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    जर (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        परत असत्य;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    स्थिरांक स्वतः * extra_u = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wu->extra);
    स्थिरांक स्वतः * extra_g = static_cast<स्थिरांक ggml_tensor_extra_gpu *>(wg->extra);
    जर (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        परत असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    स्थिरांक int64_t ne00 = wu->ne[0];
    स्थिरांक int64_t ne11 = act->ne[1];

    स्थिरांक queue_ptr stream           = ctx.stream();
    स्थिरांक पूर्णांक       src1_padded_cols = GGML_PAD((पूर्णांक) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
                                             (माप_प्रकार) ne11 * src1_padded_cols * आकार(block_q8_1) / QK8_1);
    अक्षर *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((स्थिरांक दशांश *) act->data, src1_ddq, (पूर्णांक) ne00, (पूर्णांक) ne11,
                                                          src1_padded_cols, stream);

    परत ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (दशांश *) glu->data, (पूर्णांक) ne00, (पूर्णांक) wu->ne[1], (पूर्णांक) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (पूर्णांक) आकार(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(पूर्णांक) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
स्थिर पूर्णांक ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक गाठ_idx) {
    स्थिरांक ggml_tensor * गाठ = cgraph->nodes[गाठ_idx];
    जर (ggml_sycl_info().device_मोजणी != 1 || गाठ->type != GGML_TYPE_F32 ||
        गाठ->src[0]->type != GGML_TYPE_F32 || गाठ->src[0]->ne[0] >= 1024) {
        परत 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    पूर्णांक           count = 0;
    पूर्णांक           last  = गाठ_idx;
    दशांश         eps0;
    स_नकल(&eps0, गाठ->op_params, आकार(दशांश));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    स्वतः overlaps = [](स्थिरांक ggml_tensor * a, स्थिरांक ggml_tensor * b) {
        स्थिरांक अक्षर * ab = (स्थिरांक अक्षर *) a->data;
        स्थिरांक अक्षर * bb = (स्थिरांक अक्षर *) b->data;
        परत ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    साठी (पूर्णांक j = गाठ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        जर (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            सुरूठेवा;  // not a launch; cannot break a run of adjacent norms
        }
        जर (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, गाठ) ||
            !ggml_are_same_shape(nj->src[0], गाठ->src[0])) {
            थांबा;  // any other launch ends the run
        }
        तार्किक same_nb = सत्य;
        साठी (पूर्णांक d = 0; d < GGML_MAX_DIMS; ++d) {
            जर (nj->nb[d] != गाठ->nb[d] || nj->src[0]->nb[d] != गाठ->src[0]->nb[d]) {
                same_nb = असत्य;
                थांबा;
            }
        }
        जर (!same_nb) {
            थांबा;  // one nb[] stride set is shared by the whole batch
        }
        दशांश epsj;
        स_नकल(&epsj, nj->op_params, आकार(दशांश));
        जर (epsj != eps0) {
            थांबा;  // eps mismatch ends the run
        }
        तार्किक indep = सत्य;
        साठी (पूर्णांक k = 0; k < count; ++k) {
            जर (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = असत्य;
                थांबा;
            }
        }
        जर (!indep) {
            थांबा;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    जर (count < 2) {
        परत 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    परत last - गाठ_idx;
}


__dpct_inline__ स्थिर शून्य k_copy_src1_to_contiguous(
    स्थिरांक अक्षर *__restrict__ src1_original, अक्षर *__restrict__ src1_contiguous,
    स्थिरांक mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, माप_प्रकार nb11, माप_प्रकार nb12,
    स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक int32_t src1_row = item_ct1.get_group(2);

    स्थिरांक int32_t iid1 = row_mapping[src1_row].i2;
    स्थिरांक int32_t id   = row_mapping[src1_row].i1;

    स्थिरांक int64_t i11 = id % ne11;
    स्थिरांक int64_t i12 = iid1;

    स्थिरांक दशांश * src1_row_original = (स्थिरांक दशांश *)(src1_original + i11*nb11 + i12*nb12);
    दशांश * src1_row_contiguous = (दशांश *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    साठी (पूर्णांक i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ स्थिर शून्य k_copy_dst_from_contiguous(
    अक्षर *__restrict__ dst_original, स्थिरांक अक्षर *__restrict__ dst_contiguous,
    स्थिरांक mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, माप_प्रकार nb1,
    माप_प्रकार nb2, स्थिरांक sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    स्थिरांक int32_t i1 = row_mapping[i].i1;
    स्थिरांक int32_t i2 = row_mapping[i].i2;

    स्थिरांक दशांश * dst_row_contiguous = (स्थिरांक दशांश *)(dst_contiguous + i*nb1);
    दशांश * dst_row_original = (दशांश *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    साठी (पूर्णांक j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
स्थिर तार्किक ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, स्थिरांक ggml_tensor * src0,
    स्थिरांक ggml_tensor * src1, स्थिरांक ggml_tensor * ids, ggml_tensor * dst)
{
    स्थिरांक int64_t ne10 = src1->ne[0];
    स्थिरांक int64_t ne11 = src1->ne[1];
    स्थिरांक int64_t ne12 = src1->ne[2];
    जर (ne12 != 1) परत असत्य;
    जर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) परत असत्य;
    जर (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) परत असत्य;
    जर (!ggml_is_contiguous(src1)) परत असत्य;

    स्थिरांक int64_t n_ids_per_group = ids->ne[0];
    जर (ids->ne[1] != 1) परत असत्य;
    जर (ne11 != 1 && ne11 != n_ids_per_group) परत असत्य;

    स्थिरांक queue_ptr stream           = ctx.stream();
    स्थिरांक पूर्णांक       src1_padded_cols = GGML_PAD((पूर्णांक) ne10, MATRIX_ROW_PADDING);
    स्थिरांक पूर्णांक       n_experts_used   = (पूर्णांक) n_ids_per_group;
    स्थिरांक पूर्णांक       nrows            = (पूर्णांक) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    स्थिरांक ggml_tensor_extra_gpu * src0_extra =
        static_cast<स्थिरांक ggml_tensor_extra_gpu *>(src0->extra);
    स्थिरांक तार्किक use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
        (माप_प्रकार) ne11 * src1_padded_cols * आकार(block_q8_1) / QK8_1);
    अक्षर * src1_ddq = src1_q8_alloc.get();
    जर (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (स्थिरांक दशांश *) src1->data, src1_ddq, (पूर्णांक) ne10, (पूर्णांक) ne11,
            src1_padded_cols, stream);
    } अन्यथा {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (स्थिरांक दशांश *) src1->data, src1_ddq, (पूर्णांक) ne10, (पूर्णांक) ne11,
            src1_padded_cols, stream);
    }

    स्थिरांक माप_प्रकार bytes_per_qrow = (माप_प्रकार) src1_padded_cols * आकार(block_q8_1) / QK8_1;
    स्थिरांक माप_प्रकार src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    जर (use_reorder) {
        परत ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (स्थिरांक int32_t *) ids->data,
            (दशांश *) dst->data, (पूर्णांक) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    परत ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (स्थिरांक int32_t *) ids->data,
        (दशांश *) dst->data, (पूर्णांक) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
स्थिर शून्य mmid_counting_sort_rows(
        स्थिरांक ggml_tensor * ids, स्थिरांक अक्षर * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    साठी (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        साठी (int64_t id = 0; id < n_ids; id++) {
            स्थिरांक int32_t row_id_i = *(स्थिरांक int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    साठी (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    साठी (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        साठी (int64_t id = 0; id < n_ids; id++) {
            स्थिरांक int32_t row_id_i = *(स्थिरांक int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            स्थिरांक int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

स्थिर शून्य ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) प्रयत्न {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    स्थिरांक ggml_tensor *src0 = dst->src[0];
    स्थिरांक ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    स्थिरांक ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक queue_ptr stream = ctx.stream();

    स्थिरांक int64_t n_as = ne02;
    स्थिरांक int64_t n_ids = ids->ne[0];

    जर (ne12 == 1) {
        जर (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            परत;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    जर (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<अक्षर> ids_host(ggml_nbytes(ids));
    स्थिरांक अक्षर * ids_dev = (स्थिरांक अक्षर *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->स_नकल(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->रुको()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    अक्षर *src0_original = (अक्षर *)src0->data;
    अक्षर *src1_original = (अक्षर *)src1->data;
    अक्षर *dst_original = (अक्षर *)dst->data;

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
    जर (ne12 == 1) {
        साठी (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            साठी (int64_t id = 0; id < n_ids; id++) {
                स्थिरांक int32_t i02 = *(स्थिरांक int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                स्थिरांक int64_t i11 = id % ne11;
                स्थिरांक int64_t i12 = iid1;

                स्थिरांक int64_t i1 = id;
                स्थिरांक int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } अन्यथा {
        स्थिरांक int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<अक्षर> src1_contiguous(ctx.pool(), आकार(दशांश)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<अक्षर>  dst_contiguous(ctx.pool(), आकार(दशांश)*n_routed_rows*ne0);

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
                stream->स_नकल(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*आकार(mmid_row_mapping))));

        स्थिरांक अचिन्हित पूर्णांक max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        प्रतिज्ञा(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((अचिन्हित पूर्णांक)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                अक्षर *__restrict src1_contiguous_get =
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

        तार्किक grouped = असत्य;
        जर (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == आकार(दशांश)*ne10 && nb1 == आकार(दशांश)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            जर (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            स्थिरांक तार्किक src0_reordered =
                src0->extra && ((स्थिरांक ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (स्थिरांक दशांश *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (दशांश *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        साठी (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            स्थिरांक int64_t अंक_src1_rows = expert_row_counts[i02];

            जर (अंक_src1_rows == 0) {
                सुरूठेवा;
            }

            स्थिरांक int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == आकार(दशांश)*ne10);
            GGML_ASSERT(nb1 == आकार(दशांश)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = अंक_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = अंक_src1_rows*nb11;
            src1_row.nb[3] = अंक_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = अंक_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = अंक_src1_rows*nb1;
            dst_row.nb[3] = अंक_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((अचिन्हित पूर्णांक)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                स्थिरांक अक्षर *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                स्थिरांक mmid_row_mapping *__restrict dev_row_mapping_get =
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
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

स्थिर शून्य ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

स्थिर शून्य ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

स्थिर शून्य ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

स्थिर शून्य ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

स्थिर शून्य ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

स्थिर शून्य ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

स्थिर शून्य ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

स्थिर शून्य ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

स्थिर शून्य ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

स्थिर शून्य ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

स्थिर शून्य ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

स्थिर शून्य ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


स्थिर शून्य ggml_sycl_set_main_device(स्थिरांक पूर्णांक main_device) प्रयत्न {
    जर (dpct::get_current_device_id() == static_cast<अचिन्हित पूर्णांक> (main_device)) {
        परत;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    जर (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर तार्किक ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, संरचना ggml_tensor * dst) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    जर (!g_sycl_loaded) परत असत्य;

    जर (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    निवड (dst->op) {
        स्थिती GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            थांबा;
        स्थिती GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            थांबा;
        स्थिती GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            थांबा;
        स्थिती GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            थांबा;
        स्थिती GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ADD:
        स्थिती GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            थांबा;
        स्थिती GGML_OP_मोजणी_EQUAL:
            ggml_sycl_मोजणी_equal(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            थांबा;
        स्थिती GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            थांबा;
        स्थिती GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            थांबा;
        स्थिती GGML_OP_UNARY:
            निवड (ggml_get_unary_op(dst)) {
                स्थिती GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    थांबा;
                स्थिती GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    थांबा;
                पूर्वनिर्धारित:
                    परत असत्य;
            }
            थांबा;
        स्थिती GGML_OP_GLU:
            निवड (ggml_get_glu_op(dst)) {
                स्थिती GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    थांबा;
                स्थिती GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    थांबा;
                पूर्वनिर्धारित:
                    परत असत्य;
            }
            थांबा;
        स्थिती GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            थांबा;
        स्थिती GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            थांबा;
        स्थिती GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            थांबा;
        स्थिती GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            थांबा;
        स्थिती GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            थांबा;
        स्थिती GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            थांबा;
        स्थिती GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            थांबा;
        स्थिती GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            थांबा;
        स्थिती GGML_OP_MUL_MAT:
            जर (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                परत असत्य;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            थांबा;
        स्थिती GGML_OP_MUL_MAT_ID:
            जर (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                परत असत्य;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            थांबा;
        स्थिती GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            थांबा;
        स्थिती GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            थांबा;
        स्थिती GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            थांबा;
        स्थिती GGML_OP_NONE:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
        स्थिती GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            थांबा;
        स्थिती GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            थांबा;
        स्थिती GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            थांबा;
        स्थिती GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            थांबा;
        स्थिती GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            थांबा;
        स्थिती GGML_OP_शीर्ष_K:
            ggml_sycl_op_शीर्ष_k(ctx, dst);
            थांबा;
        स्थिती GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            थांबा;
        स्थिती GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            थांबा;
        स्थिती GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            थांबा;
        स्थिती GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            थांबा;
        स्थिती GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            थांबा;
        स्थिती GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            थांबा;
        स्थिती GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            थांबा;
        स्थिती GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            थांबा;
        स्थिती GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            थांबा;
        स्थिती GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            थांबा;
        स्थिती GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            थांबा;
        स्थिती GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            थांबा;
        स्थिती GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            थांबा;
        पूर्वनिर्धारित:
            परत असत्य;
    }

    परत सत्य;
} पकडा (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::निकास(1);
}

GGML_API शून्य ggml_backend_sycl_get_device_description(पूर्णांक device, अक्षर *description,
                                      माप_प्रकार description_size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर तार्किक sycl_get_mem_info(पूर्णांक device, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#अन्यथा
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    तार्किक res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *मुक्त, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *मुक्त);
    परत res;
}

शून्य ggml_backend_sycl_get_device_memory(पूर्णांक device, माप_प्रकार * मुक्त, माप_प्रकार * total) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    जर (!sycl_get_mem_info(device, मुक्त, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *मुक्त, *total);
} पकडा (स्थिरांक sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    परत sycl_ctx->name.c_str();
}

स्थिर शून्य ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

स्थिर शून्य ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               स्थिरांक शून्य *data, माप_प्रकार offset,
                                               माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    स्थिरांक queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->स_नकल((अक्षर *)tensor->data + offset, data, size)));
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               स्थिरांक ggml_tensor *tensor,
                                               शून्य *data, माप_प्रकार offset,
                                               माप_प्रकार size) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    स्थिरांक queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->स_नकल(
        data, (स्थिरांक अक्षर *)tensor->data + offset, size)));
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर तार्किक ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               स्थिरांक ggml_tensor *src,
                                               ggml_tensor *dst) प्रयत्न {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    तार्किक is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    जर (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        स्थिरांक queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->स_नकल(
            dst->data, src->data, ggml_nbytes(dst))));
        परत सत्य;
    }

    परत असत्य;
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_synchronize(ggml_backend_t backend) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    स्थिरांक queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->रुको()));

    GGML_UNUSED(backend);
}
पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर तार्किक ggml_sycl_is_view_or_noop(स्थिरांक ggml_tensor * t) {
    परत ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
स्थिर पूर्णांक ggml_sycl_try_gdn_cache_fusion(स्थिरांक ggml_cgraph * cgraph, पूर्णांक गाठ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    जर (!g_ggml_sycl_enable_fusion) {
        परत 0;
    }

    स्थिरांक ggml_tensor * gdn = cgraph->nodes[गाठ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    जर (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        परत 0;
    }

    स्थिरांक ggml_tensor * src_v     = gdn->src[2];
    स्थिरांक int64_t       S_v       = src_v->ne[0];
    स्थिरांक int64_t       H         = src_v->ne[1];
    स्थिरांक int64_t       n_tokens  = src_v->ne[2];
    स्थिरांक int64_t       n_seqs    = src_v->ne[3];
    स्थिरांक int64_t       D         = S_v * S_v * H;
    स्थिरांक int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    स्थिरांक int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    स्थिरांक माप_प्रकार tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    स्थिरांक ggml_tensor * cpy  = nullptr;
    पूर्णांक                 skip = 0;
    साठी (पूर्णांक j = गाठ_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        स्थिरांक ggml_tensor * n = cgraph->nodes[j];
        जर (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            सुरूठेवा;
        }
        जर (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            परत 0;
        }
        cpy  = n;
        skip = j - गाठ_idx;
    }
    जर (cpy == nullptr) {
        परत 0;
    }

    स्थिरांक ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    स्थिरांक ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    जर (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        परत 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    स्थिरांक std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    जर (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (माप_प्रकार) ggml_row_size(GGML_TYPE_F32, D)) {
        परत 0;
    }

    fused_state_cpy.data        = (दशांश *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / आकार(दशांश)) : 0;
    परत skip;
}

स्थिर शून्य ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाठ = cgraph->nodes[i];
        जर (ggml_sycl_is_view_or_noop(गाठ)) {
            सुरूठेवा;
        }
        जर ((गाठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            सुरूठेवा;
        }

        स्थिरांक पूर्णांक nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        जर (nodes_to_skip != 0) {
            i += nodes_to_skip;
            सुरूठेवा;
        }
#ifndef न_संशोधन
        प्रतिज्ञा(गाठ->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        साठी (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            जर (गाठ->src[j] != nullptr) {
                प्रतिज्ञा(गाठ->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        जर (गाठ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            स्थिरांक पूर्णांक gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            जर (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, गाठ, fused_state_cpy);
                i += gdn_nodes_to_skip;
                सुरूठेवा;
            }
        }
        जर (गाठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, गाठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            सुरूठेवा;
        }
        जर (गाठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1]);
            i++;
            सुरूठेवा;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        जर (गाठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1]);
            i++;
            सुरूठेवा;
        }
        जर (गाठ->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1]);
            i++;
            सुरूठेवा;
        }
        जर (गाठ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(गाठ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1]);
            i++;
            सुरूठेवा;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        जर (गाठ->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            सुरूठेवा;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        जर (गाठ->op == GGML_OP_L2_NORM) {
            स्थिरांक पूर्णांक l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            जर (l2_batch_skip > 0) {
                i += l2_batch_skip;
                सुरूठेवा;
            }
        }

        जर (गाठ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, गाठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            सुरूठेवा;
        }

        जर (गाठ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, गाठ, nullptr, cgraph->nodes[i + 1]);
            i++;
            सुरूठेवा;
        }

        जर (गाठ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            सुरूठेवा;
        }

        तार्किक ok = ggml_sycl_compute_forward(*sycl_ctx, गाठ);
        जर (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, गाठ->name, ggml_op_name(गाठ->op));
        }
        GGML_ASSERT(ok);
    }
}

#ifdef GGML_SYCL_GRAPH
स्थिर तार्किक check_graph_compatibility(ggml_cgraph * cgraph) {
    जर (ggml_sycl_info().device_मोजणी > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        परत असत्य;
    }

    साठी (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        स्थिरांक ggml_op गाठ_op = cgraph->nodes[i]->op;
        निवड (गाठ_op) {
            पूर्वनिर्धारित:
                थांबा;
            स्थिती GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            स्थिती GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(गाठ_op));
                परत असत्य;
            स्थिती GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                जर (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(गाठ_op));
                    परत असत्य;
                }
        }
    }
    परत सत्य;
}
#endif

स्थिर ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    स्वतः * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    तार्किक use_sycl_graph = असत्य;
    जर (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    जर (use_sycl_graph) {
        स्थिरांक तार्किक graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        जर (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            परत GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        स्थिरांक तार्किक graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        जर (!sycl_ctx->exec_graph || !graph_update_support) {
            स्वतः exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } अन्यथा {
            प्रयत्न {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } पकडा (sycl::exception स्थिरांक & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                स्वतः exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } अन्यथा
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    परत GGML_STATUS_SUCCESS;
}

स्थिर शून्य ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
प्रयत्न
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    स्थिरांक queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
पकडा (sycl::exception स्थिरांक &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

स्थिर शून्य ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) प्रयत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    जर (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->रुको()));
    } अन्यथा
        GGML_ABORT("fatal error");
} पकडा (sycl::exception स्थिरांक& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::निकास(1);
}

स्थिर ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_sycl_guid() {
    स्थिर ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    परत &guid;
}

तार्किक ggml_backend_is_sycl(ggml_backend_t backend) {
    परत backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

पूर्णांक ggml_backend_sycl_get_device_मोजणी(शून्य) {
    परत ggml_sycl_info().device_मोजणी;
}


// backend device

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत ctx->name.c_str();
}

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत ctx->description.c_str();
}

स्थिर शून्य ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    जर (!sycl_get_mem_info(ctx->device, मुक्त, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *मुक्त);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *मुक्त, *total);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    स्थिरांक sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    जर (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        परत GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    परत GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    तार्किक host_buffer = दो_पर्या("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    तार्किक events = असत्य;
#अन्यथा
    तार्किक events = सत्य;
#endif

    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ events,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, स्थिरांक अक्षर * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत ggml_backend_sycl_init(ctx->device);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत ggml_backend_sycl_buffer_type(ctx->device);
}

स्थिर ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    परत ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

स्थिर ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, माप_प्रकार size, माप_प्रकार max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    परत nullptr;
}

स्थिर तार्किक do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, स्थिरांक ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    पूर्णांक device = sycl_ctx->device;
    निवड (op->op) {
        स्थिती GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                जर (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    परत सत्य;
                }
                परत असत्य;
            }
        स्थिती GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            परत (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        स्थिती GGML_OP_CONV_2D_DW:
        स्थिती GGML_OP_CONV_TRANSPOSE_2D:
            परत सत्य;
        स्थिती GGML_OP_UNARY:
            निवड (ggml_get_unary_op(op)) {
                स्थिती GGML_UNARY_OP_SGN:
                स्थिती GGML_UNARY_OP_ABS:
                स्थिती GGML_UNARY_OP_NEG:
                स्थिती GGML_UNARY_OP_STEP:
                स्थिती GGML_UNARY_OP_RELU:
                स्थिती GGML_UNARY_OP_HARDSIGMOID:
                स्थिती GGML_UNARY_OP_TANH:
                स्थिती GGML_UNARY_OP_GELU:
                स्थिती GGML_UNARY_OP_SILU:
                स्थिती GGML_UNARY_OP_SIGMOID:
                स्थिती GGML_UNARY_OP_HARDSWISH:
                स्थिती GGML_UNARY_OP_GELU_QUICK:
                स्थिती GGML_UNARY_OP_GELU_ERF:
                स्थिती GGML_UNARY_OP_EXP:
                स्थिती GGML_UNARY_OP_EXPM1:
                स्थिती GGML_UNARY_OP_SOFTPLUS:
                स्थिती GGML_UNARY_OP_ELU:
                स्थिती GGML_UNARY_OP_XIELU:
                स्थिती GGML_UNARY_OP_CEIL:
                    परत सत्य;
                स्थिती GGML_UNARY_OP_FLOOR:
                स्थिती GGML_UNARY_OP_ROUND:
                स्थिती GGML_UNARY_OP_TRUNC:
                    परत सत्य;
                पूर्वनिर्धारित:
                    परत असत्य;
            }
        स्थिती GGML_OP_GLU:
            निवड (ggml_get_glu_op(op)) {
                स्थिती GGML_GLU_OP_REGLU:
                स्थिती GGML_GLU_OP_GEGLU:
                स्थिती GGML_GLU_OP_SWIGLU:
                स्थिती GGML_GLU_OP_SWIGLU_OAI:
                स्थिती GGML_GLU_OP_GEGLU_ERF:
                स्थिती GGML_GLU_OP_GEGLU_QUICK:
                स्थिती GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    परत ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                पूर्वनिर्धारित:
                    परत असत्य;
            }
        स्थिती GGML_OP_MUL_MAT:
        स्थिती GGML_OP_MUL_MAT_ID:
            {
                संरचना ggml_tensor * a = op->src[0];
                संरचना ggml_tensor * b = op->src[1];

                जर (a->ne[3] != b->ne[3]) {
                    परत असत्य;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                जर (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  परत असत्य;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                जर (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    परत असत्य;
                }

                जर (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    परत असत्य;
                }

                परत सत्य;
            }
        स्थिती GGML_OP_OUT_PROD:
            परत op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        स्थिती GGML_OP_GET_ROWS:
            {
                निवड (op->src[0]->type) {
                    स्थिती GGML_TYPE_I32:
                    स्थिती GGML_TYPE_F16:
                    स्थिती GGML_TYPE_BF16:
                    स्थिती GGML_TYPE_F32:
                    स्थिती GGML_TYPE_Q1_0:
                    स्थिती GGML_TYPE_MXFP4:
                    स्थिती GGML_TYPE_NVFP4:
                    स्थिती GGML_TYPE_IQ2_XXS:
                    स्थिती GGML_TYPE_IQ2_XS:
                    स्थिती GGML_TYPE_IQ2_S:
                    स्थिती GGML_TYPE_IQ3_XXS:
                    स्थिती GGML_TYPE_IQ1_S:
                    स्थिती GGML_TYPE_IQ1_M:
                    स्थिती GGML_TYPE_IQ3_S:
                    स्थिती GGML_TYPE_IQ4_NL:
                    स्थिती GGML_TYPE_IQ4_XS:
                    स्थिती GGML_TYPE_Q2_K:
                    स्थिती GGML_TYPE_Q3_K:
                    स्थिती GGML_TYPE_Q4_0:
                    स्थिती GGML_TYPE_Q4_1:
                    स्थिती GGML_TYPE_Q4_K:
                    स्थिती GGML_TYPE_Q5_0:
                    स्थिती GGML_TYPE_Q5_1:
                    स्थिती GGML_TYPE_Q5_K:
                    स्थिती GGML_TYPE_Q6_K:
                    स्थिती GGML_TYPE_Q8_0:
                        परत सत्य;
                    पूर्वनिर्धारित:
                        परत असत्य;
                }
            }
        स्थिती GGML_OP_GET_ROWS_BACK:
            // return true;
             परत op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         स्थिती GGML_OP_SET:
               परत (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        स्थिती GGML_OP_SET_ROWS:
            {
                जर (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    परत असत्य;
                }
                स्वतः res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                परत res;
            }
        स्थिती GGML_OP_DSV4_HC_PRE:
            परत op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        स्थिती GGML_OP_DSV4_HC_COMB:
            परत op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        स्थिती GGML_OP_DSV4_HC_POST:
            परत op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        स्थिती GGML_OP_LIGHTNING_INDEXER:
            परत op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        स्थिती GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                जर (src0_type == GGML_TYPE_F16) {
                    जर (src1_type == GGML_TYPE_Q2_K ||
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
                        परत असत्य;
                    }
                }

                जर (src0_type == GGML_TYPE_BF16) {
                    जर (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        परत असत्य;
                    }
                }

                जर (src0_type == GGML_TYPE_F32) {
                    जर (src1_type == GGML_TYPE_Q2_K ||
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
                        परत असत्य;
                    }
                }

                जर (src1_type == GGML_TYPE_F32) {
                    जर (src0_type == GGML_TYPE_Q1_0 ||
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
                        परत असत्य;
                    }
                }

                जर (src0_type == src1_type) {
                    जर (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        परत असत्य;
                    }
                }

                जर (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    परत असत्य;
                }

                परत सत्य;
            }
        स्थिती GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                परत src0_type == GGML_TYPE_F32;
            }
        स्थिती GGML_OP_CONCAT:
        स्थिती GGML_OP_DUP:
        स्थिती GGML_OP_ARGMAX:
        स्थिती GGML_OP_NONE:
        स्थिती GGML_OP_RESHAPE:
        स्थिती GGML_OP_VIEW:
        स्थिती GGML_OP_PERMUTE:
        स्थिती GGML_OP_TRANSPOSE:
        स्थिती GGML_OP_ADD:
        स्थिती GGML_OP_ADD1:
        स्थिती GGML_OP_ADD_ID:
        स्थिती GGML_OP_SUB:
        स्थिती GGML_OP_मोजणी_EQUAL:
        स्थिती GGML_OP_MUL:
        स्थिती GGML_OP_DIV:
        स्थिती GGML_OP_REPEAT:
            परत सत्य;
        स्थिती GGML_OP_PAD_REFLECT_1D:
            परत ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        स्थिती GGML_OP_SQR:
        स्थिती GGML_OP_SQRT:
        स्थिती GGML_OP_SIN:
        स्थिती GGML_OP_COS:
        स्थिती GGML_OP_CLAMP:
        स्थिती GGML_OP_LOG:
        स्थिती GGML_OP_NORM:
        स्थिती GGML_OP_L2_NORM:
        स्थिती GGML_OP_GROUP_NORM:
        स्थिती GGML_OP_RMS_NORM:
            परत सत्य;
        स्थिती GGML_OP_RMS_NORM_BACK:
            परत ggml_is_contiguous(op->src[0]);
        स्थिती GGML_OP_SCALE:
            // the scale kernel is f32 only
            परत op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        स्थिती GGML_OP_CONT:
            परत सत्य;
        स्थिती GGML_OP_TRI:
            {
                स्थिरांक ggml_tensor * src0 = op->src[0];
                परत src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        स्थिती GGML_OP_DIAG_MASK_INF:
            परत सत्य;
        स्थिती GGML_OP_SOFT_MAX:
            परत सत्य;
        स्थिती GGML_OP_SOFT_MAX_BACK: {
            दशांश max_bias = 0.0f;
            स_नकल(&max_bias, (स्थिरांक दशांश *) op->op_params + 1, आकार(दशांश));
            परत max_bias == 0.0f;
        }
        स्थिती GGML_OP_ROPE:
        स्थिती GGML_OP_ROPE_BACK:
        स्थिती GGML_OP_IM2COL:
        स्थिती GGML_OP_IM2COL_3D:
        स्थिती GGML_OP_UPSCALE:
            परत सत्य;
        स्थिती GGML_OP_COL2IM_1D:
            परत ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        स्थिती GGML_OP_CONV_3D: {
            स्थिरांक int32_t * opts = (स्थिरांक int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            स्थिरांक int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            स्थिरांक int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            परत op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        स्थिती GGML_OP_SUM:
        स्थिती GGML_OP_SUM_ROWS:
        स्थिती GGML_OP_MEAN:
            परत ggml_is_contiguous(op->src[0]);
        स्थिती GGML_OP_ARGSORT:
            परत सत्य;
        स्थिती GGML_OP_शीर्ष_K: {
            स्थिरांक ggml_tensor * src0 = op->src[0];
            स्थिरांक पूर्णांक k = op->ne[0];
            परत src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        स्थिती GGML_OP_POOL_2D:
        स्थिती GGML_OP_POOL_1D:
        स्थिती GGML_OP_ACC:
            परत सत्य;
        स्थिती GGML_OP_PAD:
            जर (ggml_get_op_params_i32(op, 8) != 0) {
                परत असत्य;
            }
            परत सत्य;
        स्थिती GGML_OP_LEAKY_RELU:
        स्थिती GGML_OP_TIMESTEP_EMBEDDING:
        स्थिती GGML_OP_RWKV_WKV6:
        स्थिती GGML_OP_RWKV_WKV7:
        स्थिती GGML_OP_GATED_LINEAR_ATTN:
        स्थिती GGML_OP_GATED_DELTA_NET:
        स्थिती GGML_OP_OPT_STEP_ADAMW:
        स्थिती GGML_OP_OPT_STEP_SGD:
            परत सत्य;
        स्थिती GGML_OP_SSM_CONV:
            परत op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        स्थिती GGML_OP_ROLL:
            परत op->type == GGML_TYPE_F32;
        स्थिती GGML_OP_ARANGE:
            परत op->type == GGML_TYPE_F32;
        स्थिती GGML_OP_SSM_SCAN:
            जर (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                परत (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } अन्यथा {
                // TODO Mamba-1 not yet ported to SYCL
                परत असत्य;
            }
        स्थिती GGML_OP_FILL:
        स्थिती GGML_OP_CUMSUM:
        स्थिती GGML_OP_DIAG:
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS:
        स्थिती GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            परत सत्य;
        स्थिती GGML_OP_SOLVE_TRI:
            परत op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        स्थिती GGML_OP_FLASH_ATTN_EXT:
            परत ggml_sycl_flash_attn_ext_supported(device, op);
        पूर्वनिर्धारित:
            परत असत्य;
    }

    GGML_UNUSED(dev);
}

स्थिर तार्किक ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, स्थिरांक ggml_tensor * op) {
    तार्किक res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    परत res;
}

स्थिर तार्किक ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    जर (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        परत असत्य;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत buft_ctx->device == sycl_ctx->device;
}

स्थिर int64_t get_op_batch_size(स्थिरांक ggml_tensor * op) {
    निवड (op->op) {
        स्थिती GGML_OP_GET_ROWS:
            परत 0;
        स्थिती GGML_OP_MUL_MAT:
            परत op->ne[1];
        स्थिती GGML_OP_MUL_MAT_ID:
        स्थिती GGML_OP_ROPE:
            परत op->ne[2];
        पूर्वनिर्धारित:
            परत ggml_nrows(op);
    }
}

स्थिर तार्किक ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, स्थिरांक ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    परत get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

स्थिर ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    परत nullptr;
#अन्यथा
  sycl::event *event_ptr = नवीन sycl::event();

  परत नवीन ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

स्थिर शून्य ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) प्रयत्न {
  GGML_UNUSED(dev);
  जर (event == nullptr) {
    परत;
  }

  जर (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}


स्थिर शून्य ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) प्रयत्न {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->रुको()));
} पकडा (sycl::exception स्थिरांक &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::निकास(1);
}

स्थिर स्थिरांक ggml_backend_device_i ggml_backend_sycl_device_interface = {
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

संरचना ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

स्थिर स्थिरांक अक्षर * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    परत GGML_SYCL_NAME;
}

स्थिर माप_प्रकार ggml_backend_sycl_reg_get_device_मोजणी(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    परत ctx->devices.size();
}

स्थिर ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    परत ctx->devices[index];
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

संरचना ggml_backend_sycl_comm_context {
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
    माप_प्रकार    host_bytes  = 0;

    शून्य free_host() {
        साठी (पूर्णांक i = 0; i < 2; ++i) {
            स्थिरांक sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            जर (host_out[i]) { sycl::मुक्त(host_out[i], sctx); host_out[i] = nullptr; }
            जर (host_in[i])  { sycl::मुक्त(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
स्थिर शून्य ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    शून्य * dst0, स्थिरांक शून्य * src1, शून्य * dst1, स्थिरांक शून्य * src0,
                                    माप_प्रकार nbytes) {
    जर (comm->host_bytes < nbytes) {
        q0->रुको();
        q1->रुको();
        comm->free_host();
        comm->host_out[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_in[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_out[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_in[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_out[0] && comm->host_in[0] && comm->host_out[1] && comm->host_in[1]);
        comm->host_bytes = nbytes;
    }

    sycl::event e0 = q0->स_नकल(comm->host_out[0], src0, nbytes);
    sycl::event e1 = q1->स_नकल(comm->host_out[1], src1, nbytes);
    e0.रुको();
    e1.रुको();
    std::स_नकल(comm->host_in[0], comm->host_out[1], nbytes);
    std::स_नकल(comm->host_in[1], comm->host_out[0], nbytes);
    q0->स_नकल(dst0, comm->host_in[0], nbytes);
    q1->स_नकल(dst1, comm->host_in[1], nbytes);
}

शून्य * ggml_backend_sycl_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) प्रयत्न {
    साठी (माप_प्रकार i = 0; i < n_backends; ++i) {
        जर (!ggml_backend_is_sycl(backends[i])) {
            परत nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    जर (n_backends != 2) {
        परत nullptr;
    }

    स्वतः * ctx = नवीन ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    स्वतः * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    स्वतः * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    परत ctx;
}
पकडा (स्थिरांक sycl::exception &) { परत nullptr; }
पकडा (...)                     { परत nullptr; }

शून्य ggml_backend_sycl_comm_free(शून्य * comm_ctx_v) {
    स्वतः * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    जर (comm_ctx == nullptr) {
        परत;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    जर (comm_ctx->backends.size() == 2) {
        स्वतः * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        स्वतः * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        प्रयत्न {
            sctx0->stream()->रुको();
            sctx1->stream()->रुको();
            comm_ctx->free_host();
        } पकडा (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

तार्किक ggml_backend_sycl_comm_allreduce_tensor(शून्य * comm_ctx_v, संरचना ggml_tensor ** tensors) प्रयत्न {
    जर (comm_ctx_v == nullptr) {
        परत असत्य;
    }

    स्वतः * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    स्थिरांक माप_प्रकार n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    जर (n_backends != 2) {
        परत असत्य;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    जर (tensors[0]->type != tensors[1]->type) {
        परत असत्य;
    }
    जर (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        परत असत्य;
    }
    जर (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        परत असत्य;
    }
    जर (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        परत असत्य;
    }

    स्थिरांक int64_t nelem  = ggml_nelements(tensors[0]);
    स्थिरांक माप_प्रकार  nbytes = ggml_nbytes(tensors[0]);
    जर (nelem == 0) {
        परत सत्य;
    }

    स्वतः * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    स्वतः * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    जर (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->पुनः_स्मृति(nelem * 4);
        comm_ctx->buf1->पुनः_स्मृति(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    जर (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((दशांश) f16_out0[i] + (दशांश) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((दशांश) f16_out1[i] + (दशांश) f16_tmp1[i]);
            });
        });
        परत सत्य;
    }

    दशांश * out0 = (दशांश *) tensors[0]->data;
    दशांश * out1 = (दशांश *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    स्थिर constexpr int64_t BF16_THRESHOLD = 32768;

    जर (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        दशांश * tmp0 = (दशांश *) buf0;
        दशांश * tmp1 = (दशांश *) buf1;

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
        परत सत्य;
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
    स्थिरांक माप_प्रकार bf16_bytes = nelem * आकार(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<दशांश>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<दशांश>(((uint32_t) inbox1[i]) << 16);
        });
    });

    परत सत्य;
}
पकडा (स्थिरांक sycl::exception &) { परत असत्य; }
पकडा (...)                     { परत असत्य; }

स्थिर शून्य *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, स्थिरांक अक्षर *name) {
    GGML_UNUSED(reg);

    जर (म_भेद(name, "ggml_backend_split_buffer_type") == 0) {
        परत (शून्य *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    जर (म_भेद(name, "ggml_backend_comm_init") == 0) {
        परत (शून्य *)ggml_backend_sycl_comm_init;
    }
    जर (म_भेद(name, "ggml_backend_comm_free") == 0) {
        परत (शून्य *)ggml_backend_sycl_comm_free;
    }
    जर (म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        परत (शून्य *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    परत nullptr;
}

स्थिर स्थिरांक ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_मोजणी,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    स्थिर ggml_backend_reg reg;
    स्थिर तार्किक initialized = असत्य;

    {
        स्थिर std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        जर (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = नवीन ggml_backend_sycl_reg_context;
            स्थिरांक पूर्णांक min_batch_size = दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH") ? म_से_प(दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            साठी (पूर्णांक i = 0; i < ggml_sycl_info().device_मोजणी; i++) {
                ggml_backend_sycl_device_context * dev_ctx = नवीन ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = नवीन ggml_backend_device {
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

        initialized = सत्य;
    }

    परत &reg;
}

ggml_backend_t ggml_backend_sycl_init(पूर्णांक device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = नवीन ggml_backend_sycl_context(device);
    जर (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        परत nullptr;
    };

    ggml_backend_t sycl_backend = नवीन ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    परत sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
