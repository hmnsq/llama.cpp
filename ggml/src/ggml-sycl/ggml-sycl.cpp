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

#समावेश <algorithm>
#समावेश <array>
#समावेश <assert.h>
#समावेश <atomic>
#समावेश <cctype>
#समावेश <cinttypes>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <cstdlib>
#समावेश <float.h>
#समावेश <limits>
#समावेश <optional>
#समावेश <stdint.h>
#समावेश <stdio.h>
#समावेश <vector>
#समावेश <cmath>
#समावेश <iostream>
#समावेश <fstream>
#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <regex>

#समावेश <sycl/sycl.hpp>
#समावेश <sycl/backend.hpp>
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#समावेश <level_zero/ze_api.h>
#समावेश <level_zero/zes_api.h>
#पूर्ण_अगर
#अगर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#पूर्ण_अगर
#अगर SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#पूर्ण_अगर
#समावेश <sycl/half_type.hpp>

#समावेश "ggml.h"
#समावेश "ggml-sycl.h"
#समावेश "ggml-impl.h"
#समावेश "ggml-backend-impl.h"

#समावेश "ggml-sycl/add-id.hpp"
#समावेश "ggml-sycl/backend.hpp"
#समावेश "ggml-sycl/common.hpp"
#समावेश "ggml-sycl/element_wise.hpp"
#समावेश "ggml-sycl/fwht.hpp"
#समावेश "ggml-sycl/fused-gemm.hpp"
#समावेश "ggml-sycl/gemm.hpp"
#समावेश "ggml-sycl/getrows.hpp"
#समावेश "ggml-sycl/mem.hpp"
#समावेश "ggml-sycl/norm.hpp"
#समावेश "ggml-sycl/presets.hpp"
#समावेश "ggml-sycl/quantize.hpp"
#समावेश "ggml-sycl/repeat_back.hpp"
#समावेश "ggml-sycl/set_rows.hpp"
#समावेश "ggml-sycl/set.hpp"
#समावेश "ggml-sycl/dsv4-hc.hpp"
#समावेश "ggml-sycl/lightning-indexer.hpp"
#समावेश "ggml-sycl/conv2d.hpp"
#समावेश "ggml-sycl/conv2d-dw.hpp"
#समावेश "ggml-sycl/conv2d-transpose.hpp"
#समावेश "ggml-sycl/ssm_conv.hpp"
#समावेश "ggml-sycl/sycl_hw.hpp"
#समावेश "ggml-sycl/ssm_scan.hpp"
#समावेश "ggml-sycl/fill.hpp"
#समावेश "ggml-sycl/cumsum.hpp"
#समावेश "ggml-sycl/diag.hpp"
#समावेश "ggml-sycl/opt-step.hpp"
#समावेश "ggml-sycl/solve_tri.hpp"
#समावेश "ggml-sycl/gated_delta_net.hpp"
#समावेश "ggml-sycl/pool.hpp"
#समावेश "ggml-sycl/cross_entropy_loss.hpp"

#घोषणा MEM_आकार_2_M	0x00200000
#घोषणा MEM_आकार_1_G	0x40000000

अटल bool g_sycl_loaded = false;
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
अटल स्थिर अक्षर * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
अटल पूर्णांक ggml_sycl_get_env_precision(स्थिर अक्षर * name, पूर्णांक def) {
    स्थिर अक्षर * env = दो_पर्या(name);
    अगर (!env) {
        वापस def;
    }
    std::string mode(env);
    क्रम (अक्षर & c : mode) {
        c = (अक्षर) std::बड़े((अचिन्हित अक्षर) c);
    }
    क्रम (पूर्णांक i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        अगर (mode == ggml_sycl_dynamic_precision_names[i]) {
            वापस i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    वापस def;
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

#अगर GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
अटल bool ggml_sycl_dnnl_detect_optimized_gemm(पूर्णांक device) {
    using dt = dnnl::memory::तथ्य_type;

    प्रयास {
        स्थिर dnnl::memory::dims dims    = { 1, 64, 64 };
        स्थिर dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        स्थिर स्वतः & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        स्थिर स्वतः   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        स्थिर स्वतः   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        स्थिर std::string impl = pd.impl_info_str();
        अगर (impl.find("ref") == std::string::npos) {
            वापस true;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } पकड़ो (स्थिर std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    वापस false;
}
#पूर्ण_अगर

अटल ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    प्रयास {
        info.device_गिनती = dpct::dev_mgr::instance().device_गिनती();
    } पकड़ो (sycl::exception स्थिर &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_गिनती = 0;
    }
    अगर (info.device_गिनती == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        वापस info;
    }

    GGML_ASSERT(info.device_गिनती <= GGML_SYCL_MAX_DEVICES);

    int64_t योग_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    क्रम (पूर्णांक i = 0; i < info.device_गिनती; ++i) {
        dpct::device_info prop;
        स्वतः & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#अगर !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#अन्यथा
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        अगर (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            स्थिर माप_प्रकार physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<माप_प्रकार>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#पूर्ण_अगर

        info.default_tensor_split[i] = योग_vram;
        योग_vram += prop.get_global_mem_आकार();

        info.devices[i].cc =
            100 * prop.get_major_version() + 10 * prop.get_minor_version();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#अगर GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#पूर्ण_अगर
        info.devices[i].smpbo = prop.get_local_mem_आकार();
        info.devices[i].warp_आकार = WARP_आकार;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_group_sizes[i] = prop.get_max_work_group_आकार();
        info.devices[i].max_wg_per_cu = info.max_work_group_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        अगर (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = false;
        }

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        अगर (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_परिणाम_t r = zeDeviceGetProperties(ze_dev, &props);
            अगर (r == ZE_परिणाम_SUCCESS) {
                info.devices[i].l0_device_type_valid = true;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#पूर्ण_अगर
    }

    क्रम (पूर्णांक id = 0; id < info.device_गिनती; ++id) {
        info.default_tensor_split[id] /= योग_vram;
    }

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#पूर्ण_अगर

    वापस info;
}

स्थिर ggml_sycl_device_info & ggml_sycl_info() {
    अटल ggml_sycl_device_info info = ggml_sycl_init();
    वापस info;
}

अटल व्योम print_device_detail(पूर्णांक id, sycl::device &device, std::string device_type) {

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

    स्वतः global_mem_आकार = prop.get_global_mem_आकार()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), version.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_group_आकार(), prop.get_max_sub_group_आकार(),
            global_mem_आकार, device.get_info<sycl::info::device::driver_version>().c_str());
}

अटल व्योम print_device_opt_feature(पूर्णांक device_गिनती) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, माप_प्रकार> DeviceNums;
    क्रम (पूर्णांक id = 0; id < device_गिनती; ++id) {
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
व्योम ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    पूर्णांक device_गिनती = dpct::dev_mgr::instance().device_गिनती();
    std::map<std::string, माप_प्रकार> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_गिनती);

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

    क्रम (पूर्णांक id = 0; id < device_गिनती; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      पूर्णांक type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_गिनती);
}

अटल स्थिर अक्षर* dev2dev_int2str(पूर्णांक dev2dev) {
    अगर (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        वापस "SYCL API";
    } अन्यथा अगर (dev2dev == DEV2DEV_MEMCPY_L0) {
        वापस "Level Zero API";
    } अन्यथा अगर (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        वापस "Host Forward";
    } अन्यथा {
        वापस "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
अटल ze_परिणाम_t init_zes() {
    ze_परिणाम_t res = zesInit(0);
    अगर (res != ZE_परिणाम_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (पूर्णांक) res);
    }
    वापस res;
}

अटल ze_परिणाम_t get_zes_init_res() {
    अटल ze_परिणाम_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (पूर्णांक) zes_init_res);
    वापस zes_init_res;
}
#पूर्ण_अगर

अटल व्योम initialize_sycl_begining() {
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#पूर्ण_अगर
}

अटल व्योम ggml_check_sycl() प्रयास {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    अटल bool initialized = false;

    अगर (!initialized) {
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

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#अन्यथा
        g_ggml_sycl_use_level_zero_api = 0;
#पूर्ण_अगर
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        अगर (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#अगर_घोषित SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#अन्यथा
        g_ggml_sycl_enable_flash_attention = 0;
#पूर्ण_अगर

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
#अगर defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#पूर्ण_अगर

#अगर defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#पूर्ण_अगर

#अगर defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#पूर्ण_अगर

#अगर defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#पूर्ण_अगर

#अगर defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#पूर्ण_अगर
#अगर defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#पूर्ण_अगर

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#पूर्ण_अगर

#अगर defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#पूर्ण_अगर
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#अगर_घोषित SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#पूर्ण_अगर

#अगर_घोषित GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#पूर्ण_अगर

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#अगर defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#पूर्ण_अगर

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#अगर defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#पूर्ण_अगर
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#पूर्ण_अगर

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
#अगर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        अगर (g_ggml_sycl_use_async_mem_op) {
            क्रम (अचिन्हित पूर्णांक i = 0; i < dpct::dev_mgr::instance().device_गिनती(); ++i) {
                अगर (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    अवरोध;
                }
            }
        }
#पूर्ण_अगर
        अगर (CHECK_TRY_ERROR(g_all_sycl_device_गिनती =
                            dpct::dev_mgr::instance().device_गिनती()) != 0) {
            initialized = true;
            g_sycl_loaded = false;
            वापस;
        }
        GGML_ASSERT(g_all_sycl_device_गिनती <= GGML_SYCL_MAX_DEVICES);

        initialized = true;
        g_sycl_loaded = true;
        ggml_backend_sycl_print_sycl_devices();
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
अंतरभूत व्योम check_allow_gpu_सूचकांक(स्थिर पूर्णांक device_सूचकांक) {
  अगर (device_सूचकांक >= ggml_sycl_info().device_गिनती) {
    अक्षर error_buf[256];
    snprintf(
        error_buf,
        माप(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_सूचकांक,
        ggml_sycl_info().device_गिनती - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    निश्चित(false);
  }
}

GGML_API व्योम ggml_backend_sycl_get_gpu_list(पूर्णांक *id_list, पूर्णांक max_len) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    क्रम(पूर्णांक i=0;i<max_len;i++) id_list[i] = -1;

    क्रम (पूर्णांक i=0;i< ggml_sycl_info().device_गिनती;i++){
        अगर (i>=max_len) अवरोध;
        id_list[i] = i;
    }
    वापस;
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अंतरभूत व्योम free_aligned_mem_host(व्योम * memblock) {
#अगर_घोषित _WIN32
    _aligned_free(memblock);
#अन्यथा
    मुक्त(memblock);
#पूर्ण_अगर
}

// sycl buffer

काष्ठा ggml_backend_sycl_बफर_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    अटल constexpr माप_प्रकार staging_slot_आकार = 8*1024*1024;

    काष्ठा host_staging {
        व्योम * तथ्य = nullptr;
        std::vector<sycl::event> events;
        std::vector<bool> submitted;
        पूर्णांक अग्र = 0;
    };

    पूर्णांक device;
    व्योम * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    bool is_usm_system;

    ggml_backend_sycl_बफर_context(पूर्णांक device, व्योम * dev_ptr, queue_ptr stream, bool is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_सूचकांक(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    व्योम drop_host_staging() {
        क्रम (माप_प्रकार i = 0; i < staging.submitted.size(); ++i) {
            अगर (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = false;
            }
        }
        अगर (staging.तथ्य != nullptr) {
            sycl::मुक्त(staging.तथ्य, *stream);
            staging.तथ्य = nullptr;
        }
    }

    ~ggml_backend_sycl_बफर_context() {
        drop_host_staging();
        अगर (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            अगर (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            अन्यथा
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        क्रम (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

अटल स्थिर अक्षर * ggml_backend_sycl_बफर_type_get_name(ggml_backend_बफर_type_t buft);

अटल bool ggml_backend_बफर_is_sycl(ggml_backend_बफर_t बफर) {
    वापस बफर->buft->iface.get_name == ggml_backend_sycl_बफर_type_get_name;
}

अटल व्योम
ggml_backend_sycl_बफर_free_बफर(ggml_backend_बफर_t बफर) प्रयास {
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    ggml_sycl_set_device(ctx->device);

    मिटाओ ctx;
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम * ggml_backend_sycl_बफर_get_base(ggml_backend_बफर_t बफर) {
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    वापस ctx->dev_ptr;
}

अटल क्रमागत ggml_status
ggml_backend_sycl_बफर_init_tensor(ggml_backend_बफर_t बफर,
                                     ggml_tensor *tensor) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *)बफर->context;

    अगर (tensor->view_src != शून्य) {
        निश्चित(tensor->view_src->बफर->buft == बफर->buft);
        वापस GGML_STATUS_SUCCESS;
    }

    अगर (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        चयन (tensor->type) {
            हाल GGML_TYPE_Q4_0:
            हाल GGML_TYPE_Q8_0:
            हाल GGML_TYPE_Q2_K:
            हाल GGML_TYPE_Q3_K:
            हाल GGML_TYPE_Q4_K:
            हाल GGML_TYPE_Q5_K:
            हाल GGML_TYPE_Q6_K:
            हाल GGML_TYPE_IQ3_XXS:
            हाल GGML_TYPE_IQ3_S:
            हाल GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = नया ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                अवरोध;
            }
            शेष:
                अवरोध;
        }
    }

    अगर (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        माप_प्रकार original_आकार = ggml_nbytes(tensor);
        माप_प्रकार padded_आकार = ggml_backend_buft_get_alloc_आकार(बफर->buft, tensor);

        अगर (padded_आकार > original_आकार && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->स_रखो(
                (अक्षर *)tensor->तथ्य + original_आकार, 0,
                padded_आकार - original_आकार).रुको()));
        }
    }
    वापस GGML_STATUS_SUCCESS;
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                                ggml_tensor *tensor,
                                                स्थिर व्योम *तथ्य, माप_प्रकार offset,
                                                माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    स्थिर पूर्णांक n_slots = g_ggml_sycl_upload_staging_slots;
    अगर (n_slots > 0 && ctx->staging.तथ्य == nullptr) {
        ctx->staging.तथ्य = sycl::malloc_host(n_slots * ctx->staging_slot_आकार, *ctx->stream);
        अगर (ctx->staging.तथ्य != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, false);
        }
    }
    अगर (ctx->staging.तथ्य != nullptr) {
        queue_ptr    stream    = ctx->stream;
        अक्षर *       dst       = (अक्षर *) tensor->तथ्य + offset;
        स्थिर अक्षर * src       = (स्थिर अक्षर *) तथ्य;
        माप_प्रकार       remaining = size;
        जबतक (remaining > 0) {
            स्थिर माप_प्रकार chunk = std::min(remaining, ctx->staging_slot_आकार);
            स्थिर पूर्णांक    slot  = ctx->staging.अग्र;
            ctx->staging.अग्र = (ctx->staging.अग्र + 1) % (पूर्णांक) ctx->staging.submitted.size();
            अगर (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            व्योम * stage = (अक्षर *) ctx->staging.तथ्य + slot * ctx->staging_slot_आकार;
            स_नकल(stage, src, chunk);
            ctx->staging.events[slot] = stream->स_नकल(dst, stage, chunk);
            ctx->staging.submitted[slot] = true;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        वापस;
    }

    स्वतः stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#अगर_अघोषित _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    अक्षर * host_buf = (अक्षर *) दो_स्मृति(size);
    स_नकल(host_buf, तथ्य, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल((अक्षर *) tensor->तथ्य + offset, host_buf, size).रुको()));
    मुक्त(host_buf);
#अन्यथा
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल((अक्षर *) tensor->तथ्य + offset, तथ्य, size).रुको()));
#पूर्ण_अगर
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                स्थिर ggml_tensor *tensor,
                                                व्योम *तथ्य, माप_प्रकार offset,
                                                माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;

    ggml_sycl_set_device(ctx->device);
    स्वतः stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.स_नकल(तथ्य, (स्थिर अक्षर *)tensor->तथ्य + offset, size)
            .रुको()));
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
अटल bool ggml_sycl_is_l0_discrete_gpu(पूर्णांक device) {
    वापस ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#पूर्ण_अगर

अटल व्योम memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, व्योम *ptr_dst,
                         स्थिर व्योम *ptr_src, माप_प्रकार size) {
    अक्षर *host_buf = (अक्षर *)दो_स्मृति(size);
    अगर (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.स_नकल(host_buf, (स्थिर अक्षर *)ptr_src, size).रुको();
    q_dst.स_नकल((अक्षर *)ptr_dst, host_buf, size).रुको();
    मुक्त(host_buf);
}

अटल व्योम dev2dev_memcpy(पूर्णांक device_dst, sycl::queue &q_dst, पूर्णांक device_src, sycl::queue &q_src, व्योम *ptr_dst,
                    स्थिर व्योम *ptr_src, माप_प्रकार size) {

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    अगर (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        स्थिर bool l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        अगर (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            स्वतः ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            स्वतः ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_परिणाम_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            अगर (r == ZE_परिणाम_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                अगर (r == ZE_परिणाम_SUCCESS) {
                    वापस;
                }
            }
        }
    }
#पूर्ण_अगर

    अगर (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        अगर (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.स_नकल(ptr_dst, ptr_src, size).रुको()));
            वापस;
        }
    }

    // Host-staged copy
    अगर(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } अन्यथा {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

अटल bool
ggml_backend_sycl_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                    स्थिर ggml_tensor *src,
                                    ggml_tensor *dst) प्रयास {
    bool is_cpy_supported = ggml_backend_बफर_is_sycl(src->बफर);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    अगर (is_cpy_supported) {
        ggml_backend_sycl_बफर_context * src_ctx = (ggml_backend_sycl_बफर_context *)src->बफर->context;
        ggml_backend_sycl_बफर_context * dst_ctx = (ggml_backend_sycl_बफर_context *)dst->बफर->context;

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
        dev2dev_memcpy(dst_ctx->device, *stream_dst, src_ctx->device, *stream_src, dst->तथ्य, src->तथ्य, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#अगर 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_नकल(
            (अक्षर *)dst->तथ्य, (स्थिर अक्षर *)src->तथ्य, size).रुको()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#पूर्ण_अगर
        वापस true;
    }
    वापस false;
    GGML_UNUSED(बफर);
} पकड़ो (स्थिर sycl::exception & exc) {
    std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__ << ", line:" << __LINE__ << std::प_पूर्ण;
    std::निकास(1);
}

अटल व्योम ggml_backend_sycl_बफर_clear(ggml_backend_बफर_t बफर,
                                           uint8_t value) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, बफर->size);
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr माप_प्रकार MAX_CHUNK = 2ULL << 30;  // 2 GiB
    क्रम (माप_प्रकार off = 0; off < बफर->size; off += MAX_CHUNK) {
        माप_प्रकार chunk = std::min(बफर->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_रखो(static_cast<अक्षर*>(ctx->dev_ptr) + off, value, chunk)
                .रुको()
        ));
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value,
                                                   माप_प्रकार offset, माप_प्रकार size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    स्वतः stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    अगर (size == 0) {
        वापस;  // Nothing to do
    }
    अगर (tensor->तथ्य == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    व्योम * target_ptr = static_cast<अक्षर *>(tensor->तथ्य) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स_रखो(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).रुको()));
}

अटल व्योम ggml_backend_sycl_बफर_reset(ggml_backend_बफर_t बफर) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    अगर (बफर == nullptr) {
        वापस;
    }

    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;

    अगर (ctx != nullptr) {
        क्रम (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.सफा();  // reset the tensor_extras vector
    }
}

अटल स्थिर ggml_backend_बफर_i ggml_backend_sycl_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_sycl_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ ggml_backend_sycl_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_बफर_clear,
    /* .reset           = */ ggml_backend_sycl_बफर_reset,
};

// sycl buffer type
काष्ठा ggml_backend_sycl_बफर_type_context {
    पूर्णांक device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

अटल स्थिर अक्षर * ggml_backend_sycl_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_sycl_बफर_type_context * ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;

    वापस ctx->name.c_str();
}

अटल bool check_usm_system(पूर्णांक device, माप_प्रकार size) {
    bool use_usm_system = g_ggml_sycl_usm_system && size >= ((माप_प्रकार)4 * MEM_आकार_1_G);

    अगर (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = false;
    }

    वापस use_usm_system;
}

अंतरभूत व्योम * aligned_malloc_host(माप_प्रकार alignment, माप_प्रकार size) {
#अगर_घोषित _WIN32
    वापस _aligned_malloc(size, alignment);
#अन्यथा
    वापस aligned_alloc(alignment, size);
#पूर्ण_अगर
}

अटल ggml_backend_बफर_t
ggml_backend_sycl_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                           माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_बफर_type_context * buft_ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    स्थिर queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (माप_प्रकार)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    माप_प्रकार alignment = MEM_आकार_2_M;
    माप_प्रकार aligned_आकार = ((size + alignment - 1) / alignment) * alignment;
    bool use_usm_system = check_usm_system(buft_ctx->device, aligned_आकार);

    व्योम * dev_ptr;
    अगर (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (व्योम *)aligned_malloc_host(alignment, aligned_आकार);
        अगर (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            वापस nullptr;
        }
    } अन्यथा {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (व्योम *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_बफर)));
        अगर (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          वापस nullptr;
        }
    }
    ggml_backend_sycl_बफर_context * ctx = नया  ggml_backend_sycl_बफर_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    वापस ggml_backend_बफर_init(buft, ggml_backend_sycl_बफर_interface, ctx, size);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल माप_प्रकार ggml_backend_sycl_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस SYCL_बफर_ALIGNMENT;
    GGML_UNUSED(buft);
}

अटल bool is_bmg_g31_arch(पूर्णांक device) {
    वापस ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

अटल माप_प्रकार ggml_backend_sycl_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_sycl_बफर_type_context * ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    पूर्णांक device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    माप_प्रकार max_alloc_आकार = dpct::get_device(device).get_max_mem_alloc_आकार();
    अगर (g_ggml_sycl_host_pinned_mem_2g) {
        वापस std::min(max_alloc_आकार, (माप_प्रकार) 2LL*1024*1024*1024);
    } अन्यथा {
        अगर(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_आकार = max_alloc_आकार*0.6;
        }
        वापस max_alloc_आकार;

    }
    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_sycl_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    माप_प्रकार size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_आकार(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    अगर (ggml_is_quantized(tensor->type)) {
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    वापस size;

    GGML_UNUSED(buft);
}

अटल स्थिर ggml_backend_बफर_type_i ggml_backend_sycl_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_sycl_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_sycl_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ शून्य,
};

ggml_backend_बफर_type_t ggml_backend_sycl_बफर_type(पूर्णांक device) {
    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    स्वतः dev_गिनती = ggml_backend_sycl_get_device_गिनती();

    अगर (device>=dev_गिनती or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_गिनती-1);
        GGML_ASSERT(device<dev_गिनती);
    }
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_sycl_बफर_types[GGML_SYCL_MAX_DEVICES];

    अटल bool ggml_backend_sycl_बफर_type_initialized = false;

    अगर (!ggml_backend_sycl_बफर_type_initialized) {
        क्रम (पूर्णांक i = 0; i < dev_गिनती; i++) {
            स्वतः & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_बफर_types[i] = {
                /* .iface    = */ ggml_backend_sycl_बफर_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ नया ggml_backend_sycl_बफर_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_बफर_type_initialized = true;
    }
    वापस &ggml_backend_sycl_बफर_types[device];
}

अटल ggml_backend_बफर_type_t ggml_backend_sycl_बफर_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    पूर्णांक device = ctx->device;
    अगर (device>=ggml_sycl_info().device_गिनती or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_गिनती-1);
        GGML_ASSERT(device<ggml_sycl_info().device_गिनती);
    }
    अटल काष्ठा ggml_backend_बफर_type ggml_backend_sycl_बफर_types[GGML_SYCL_MAX_DEVICES];

    अटल bool ggml_backend_sycl_बफर_type_initialized = false;

    अगर (!ggml_backend_sycl_बफर_type_initialized) {
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; i++) {
            ggml_backend_sycl_बफर_types[i] = {
                /* .iface    = */ ggml_backend_sycl_बफर_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ नया ggml_backend_sycl_बफर_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_बफर_type_initialized = true;
    }
    वापस &ggml_backend_sycl_बफर_types[device];
}

// sycl split buffer

अटल int64_t get_row_rounding(ggml_type type, स्थिर std::array<भग्न, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = पूर्णांक_उच्च;
    int64_t max_compute_capability = पूर्णांक_न्यून;
    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        अगर (tensor_split[i] < (i + 1 < ggml_sycl_info().device_गिनती ? tensor_split[i + 1] : 1.0f)) {
            अगर (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            अगर (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    चयन(type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
            वापस max_compute_capability >= VER_GEN9 ? 128 : 64;
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
            वापस 64;
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_F32:
            वापस 1;
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_IQ2_XXS:
        हाल GGML_TYPE_IQ2_XS:
        हाल GGML_TYPE_IQ2_S:
        हाल GGML_TYPE_IQ1_S:
        हाल GGML_TYPE_IQ1_M:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ4_XS:
        हाल GGML_TYPE_IQ4_NL:
            वापस max_compute_capability >= VER_GEN9 ? 128 : 64;
        हाल GGML_TYPE_IQ3_S:
            वापस max_compute_capability >= VER_GEN9 ? 128 : 64;
        हाल GGML_TYPE_Q6_K:
            वापस 64;
        शेष:
            GGML_ABORT("fatal error");
    }
}

अटल व्योम get_row_split(int64_t * row_निचला, int64_t * row_ऊपरी, स्थिर ggml_tensor * tensor, स्थिर std::array<भग्न, GGML_SYCL_MAX_DEVICES> & tensor_split, पूर्णांक id) {
    स्थिर int64_t nrows = ggml_nrows(tensor);
    स्थिर int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_निचला = id == 0 ? 0 : nrows*tensor_split[id];
    *row_निचला -= *row_निचला % rounding;
    अगर (id == ggml_sycl_info().device_गिनती - 1) {
        *row_ऊपरी = nrows;
    } अन्यथा {
        *row_ऊपरी = nrows*tensor_split[id + 1];
        *row_ऊपरी -= *row_ऊपरी % rounding;
    }
}

अटल माप_प्रकार ggml_nbytes_split(स्थिर काष्ठा ggml_tensor * tensor, पूर्णांक nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    वापस nrows_split*ggml_row_आकार(tensor->type, tensor->ne[0]);
}

काष्ठा ggml_backend_sycl_split_बफर_type_context {
    std::array<भग्न, GGML_SYCL_MAX_DEVICES> tensor_split;
};

काष्ठा ggml_backend_sycl_split_बफर_context {
    ~ggml_backend_sycl_split_बफर_context() प्रयास {
        क्रम (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    पकड़ो (sycl::exception स्थिर &exc) {
      std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
                << ", line:" << __LINE__ << std::प_पूर्ण;
      std::निकास(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

अटल व्योम ggml_backend_sycl_split_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_sycl_split_बफर_context * ctx = (ggml_backend_sycl_split_बफर_context *)बफर->context;
    मिटाओ ctx;
}

अटल व्योम * ggml_backend_sycl_split_बफर_get_base(ggml_backend_बफर_t बफर) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    वापस (व्योम *)0x1000;

    GGML_UNUSED(बफर);
}

अटल क्रमागत ggml_status
ggml_backend_sycl_split_बफर_init_tensor(ggml_backend_बफर_t बफर,
                                           ggml_tensor *tensor) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_बफर_context * ctx = (ggml_backend_sycl_split_बफर_context *)बफर->context;
    ggml_backend_sycl_split_बफर_type_context * buft_ctx = (ggml_backend_sycl_split_बफर_type_context *)बफर->buft->context;

    स्थिर int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = नया ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    अगर (ctx->streams.empty()) {
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        int64_t row_निचला, row_ऊपरी;
        get_row_split(&row_निचला, &row_ऊपरी, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ऊपरी - row_निचला;
        अगर (nrows_split == 0) {
            जारी;
        }

        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिर माप_प्रकार original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        स्थिर queue_ptr stream = ctx->streams[i];
        अक्षर * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (अक्षर *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_बफर)));
        अगर (!buf) {
            अक्षर err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            फेंको std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        अगर (size > original_आकार) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .स_रखो(buf + original_आकार, 0, size - original_आकार)
                    .रुको()));
        }

        extra->तथ्य_device[i] = buf;

        क्रम (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = नया sycl::event()));
        }
    }
    tensor->extra = extra;
    वापस GGML_STATUS_SUCCESS;
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम
ggml_backend_sycl_split_बफर_set_tensor(ggml_backend_बफर_t बफर,
                                          ggml_tensor *tensor, स्थिर व्योम *तथ्य,
                                          माप_प्रकार offset, माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_बफर_context * ctx = (ggml_backend_sycl_split_बफर_context *)बफर->context;
    ggml_backend_sycl_split_बफर_type_context * buft_ctx = (ggml_backend_sycl_split_बफर_type_context *)बफर->buft->context;

    स्थिर int64_t ne0 = tensor->ne[0];
    स्थिर माप_प्रकार nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        int64_t row_निचला, row_ऊपरी;
        get_row_split(&row_निचला, &row_ऊपरी, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ऊपरी - row_निचला;
        अगर (nrows_split == 0) {
            जारी;
        }

        स्थिर माप_प्रकार offset_split = row_निचला*nb1;
        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिर माप_प्रकार original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        स्थिर अक्षर * buf_host = (स्थिर अक्षर *)तथ्य + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        स्थिर queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_नकल(extra->तथ्य_device[i], buf_host, original_आकार)
                .रुको()));
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम
ggml_backend_sycl_split_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                          स्थिर ggml_tensor *tensor, व्योम *तथ्य,
                                          माप_प्रकार offset, माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_बफर_context * ctx = (ggml_backend_sycl_split_बफर_context *)बफर->context;
    ggml_backend_sycl_split_बफर_type_context * buft_ctx = (ggml_backend_sycl_split_बफर_type_context *)बफर->buft->context;

    स्थिर int64_t ne0 = tensor->ne[0];
    स्थिर माप_प्रकार nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        int64_t row_निचला, row_ऊपरी;
        get_row_split(&row_निचला, &row_ऊपरी, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ऊपरी - row_निचला;
        अगर (nrows_split == 0) {
            जारी;
        }

        स्थिर माप_प्रकार offset_split = row_निचला*nb1;
        माप_प्रकार size = ggml_nbytes_split(tensor, nrows_split);
        स्थिर माप_प्रकार original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        अक्षर * buf_host = (अक्षर *)तथ्य + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        स्थिर queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स_नकल(buf_host, extra->तथ्य_device[i], original_आकार)
                .रुको()));
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_split_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_UNUSED(बफर);
    GGML_UNUSED(value);
}

अटल काष्ठा ggml_backend_बफर_i ggml_backend_sycl_split_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_sycl_split_बफर_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_बफर_init_tensor,
    /* .memset_tensor   = */ शून्य,
    /* .set_tensor      = */ ggml_backend_sycl_split_बफर_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_बफर_get_tensor,
    /* .set_tensor_2d   = */ शून्य,
    /* .get_tensor_2d   = */ शून्य,
    /* .cpy_tensor      = */ शून्य,
    /* .clear           = */ ggml_backend_sycl_split_बफर_clear,
    /* .reset           = */ शून्य,
};

// sycl split buffer type

अटल स्थिर अक्षर * ggml_backend_sycl_split_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    वापस GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

अटल bool ggml_backend_बफर_is_sycl_split(ggml_backend_बफर_t बफर) {
   वापस बफर->buft->iface.get_name == ggml_backend_sycl_split_बफर_type_get_name;
}

अटल ggml_backend_बफर_t ggml_backend_sycl_split_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_बफर_context * ctx = नया ggml_backend_sycl_split_बफर_context();

    वापस ggml_backend_बफर_init(buft, ggml_backend_sycl_split_बफर_interface, ctx, size);
}

अटल माप_प्रकार ggml_backend_sycl_split_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    वापस SYCL_बफर_ALIGNMENT;
    GGML_UNUSED(buft);
}

अटल माप_प्रकार ggml_backend_sycl_split_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, स्थिर ggml_tensor * tensor) {
    ggml_backend_sycl_split_बफर_type_context * ctx = (ggml_backend_sycl_split_बफर_type_context *)buft->context;

    माप_प्रकार योग_आकार = 0;

    स्थिर int64_t ne0 = tensor->ne[0];

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        int64_t row_निचला, row_ऊपरी;
        get_row_split(&row_निचला, &row_ऊपरी, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_ऊपरी - row_निचला;
        अगर (nrows_split == 0) {
            जारी;
        }

        योग_आकार += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        अगर (ne0 % MATRIX_ROW_PADDING != 0) {
            योग_आकार += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    वापस योग_आकार;
}

अटल bool ggml_backend_sycl_split_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    वापस false;

    GGML_UNUSED(buft);
}

अटल ggml_backend_बफर_type_i ggml_backend_sycl_split_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ शून्य,
    /* .get_alignment       = */ ggml_backend_sycl_split_बफर_type_get_alignment,
    /* .get_max_size        = */ शून्य, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ शून्य,
    /* .is_host             = */ ggml_backend_sycl_split_बफर_type_is_host,
};

ggml_backend_बफर_type_t ggml_backend_sycl_split_बफर_type(पूर्णांक main_device, स्थिर भग्न * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    अटल std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    अटल std::map<std::array<भग्न, GGML_SYCL_MAX_DEVICES>, काष्ठा ggml_backend_बफर_type> buft_map;

    std::array<भग्न, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    bool all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](भग्न x) { वापस x == 0.0f; });
    अगर (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } अन्यथा {
        भग्न split_योग = 0.0f;
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
            tensor_split_arr[i] = split_योग;
            split_योग += tensor_split[i];
        }
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
            tensor_split_arr[i] /= split_योग;
        }
    }

    स्वतः it = buft_map.find(tensor_split_arr);
    अगर (it != buft_map.end()) {
        वापस &it->second;
    }

    काष्ठा ggml_backend_बफर_type buft {
        /* .iface   = */ ggml_backend_sycl_split_बफर_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ नया ggml_backend_sycl_split_बफर_type_context{tensor_split_arr},
    };

    स्वतः परिणाम = buft_map.emplace(tensor_split_arr, buft);
    वापस &परिणाम.first->second;
}

// host buffer type

काष्ठा ggml_backend_sycl_device_context {
    पूर्णांक device;
    std::string name;
    std::string description;
    पूर्णांक op_offload_min_batch_आकार;
};

अटल स्थिर अक्षर * ggml_backend_sycl_host_बफर_type_name(ggml_backend_बफर_type_t buft) {
    वापस GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

अटल पूर्णांक ggml_backend_sycl_host_बफर_type_device(ggml_backend_बफर_type_t buft) {
    वापस static_cast<स्थिर ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
अटल व्योम * ggml_backend_sycl_host_malloc(पूर्णांक device, माप_प्रकार size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    व्योम * ptr = nullptr;
    प्रयास {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        स्वतः & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } पकड़ो (...) {
        ptr = nullptr;
    }
    अगर (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    वापस ptr;
}

अटल व्योम ggml_backend_sycl_host_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    अगर (बफर->context == nullptr) {
        वापस;
    }
    अगर (g_ggml_sycl_enable_host_pinned_mem) {
        स्थिर पूर्णांक device = ggml_backend_sycl_host_बफर_type_device(बफर->buft);
        स्वतः & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(बफर->context, q)));
    } अन्यथा {
        free_aligned_mem_host((व्योम *) बफर->context);
    }
}

अटल ggml_backend_बफर_t ggml_backend_sycl_host_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, माप_प्रकार size) {
    व्योम * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_बफर_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    अगर (ptr == nullptr) {
        // fallback to cpu buffer
        वापस ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft = buft;
    बफर->iface.free_बफर = ggml_backend_sycl_host_बफर_free_बफर;

    वापस बफर;
}

अटल माप_प्रकार ggml_backend_sycl_host_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {

    अगर (g_ggml_sycl_enable_host_pinned_mem) {
        स्थिर पूर्णांक device = ggml_backend_sycl_host_बफर_type_device(buft);
        माप_प्रकार max_alloc_आकार = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_आकार();
        अगर (g_ggml_sycl_host_pinned_mem_2g) {
            वापस std::min(max_alloc_आकार, (माप_प्रकार) 2LL*1024*1024*1024);
        } अन्यथा {
            वापस max_alloc_आकार;
        }
    } अन्यथा {
        वापस SIZE_MAX;
    }
}

अटल ggml_backend_बफर_type_t ggml_backend_sycl_host_बफर_type_for_device(पूर्णांक device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    अटल std::vector<ggml_backend_बफर_type> बफर_types_host = [] {
        std::vector<ggml_backend_बफर_type> bufts(ggml_backend_sycl_get_device_गिनती());
        क्रम (माप_प्रकार i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_बफर_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_बफर_type_alloc_बफर,
                    /* .alloc_buffer_n      = */ शून्य,
                    /* .get_alignment       = */ ggml_backend_cpu_बफर_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_बफर_type_get_max_आकार,
                    /* .get_alloc_size      = */ ggml_backend_cpu_बफर_type()->iface.get_alloc_आकार,
                    /* .get_alloc_size_n    = */ शून्य,
                    /* .is_host             = */ ggml_backend_cpu_बफर_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        वापस bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (पूर्णांक) बफर_types_host.size());

    वापस &बफर_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_बफर_type_t ggml_backend_sycl_host_बफर_type() {
    वापस ggml_backend_sycl_host_बफर_type_for_device(0);
}

// buffer pool for sycl (legacy)
काष्ठा ggml_sycl_pool_leg : खुला ggml_sycl_pool {
    अटल स्थिर पूर्णांक MAX_SYCL_BUFFERS = 256;

    पूर्णांक device;
    queue_ptr qptr;
    काष्ठा ggml_sycl_बफर {
        व्योम * ptr = nullptr;
        माप_प्रकार size = 0;
    };

    ggml_sycl_बफर बफर_pool[MAX_SYCL_BUFFERS] = {};
    माप_प्रकार pool_आकार = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, पूर्णांक device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#अगर_घोषित DEBUG_SYCL_POOL
        पूर्णांक    n_cached    = 0;
        माप_प्रकार bytes_cached = 0;
        क्रम (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            अगर (बफर_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += बफर_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        स्थिर स्वतः slots = format_slots_in_alloc_order();
        अगर (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#पूर्ण_अगर

        क्रम (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            अगर (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_आकार -= b.size;
            }
        }
        GGML_ASSERT(pool_आकार == 0);
    }

#अगर_घोषित DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() स्थिर {
        std::string पंक्ति;
        अक्षर buf[32];
        bool first = true;
        क्रम (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            अगर (बफर_pool[i].ptr == nullptr) {
                जारी;
            }
            अगर (!first) {
                पंक्ति += '/';
            }
            first = false;
            snprintf(buf, माप(buf), "%.2f", बफर_pool[i].size / 1024.0 / 1024.0);
            पंक्ति += buf;
        }
        वापस पंक्ति;
    }
#पूर्ण_अगर

    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
#अगर_घोषित DEBUG_sycl_MALLOC
        पूर्णांक nnz = 0;
        माप_प्रकार max_size = 0;
#पूर्ण_अगर
        माप_प्रकार best_diff = 1ull << 36;
        पूर्णांक ibest = -1;
        क्रम (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर& b = बफर_pool[i];
            अगर (b.ptr != nullptr) {
#अगर_घोषित DEBUG_sycl_MALLOC
                ++nnz;
                अगर (b.size > max_size) max_size = b.size;
#पूर्ण_अगर
                अगर (b.size >= size) {
                    माप_प्रकार diff = b.size - size;
                    अगर (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        अगर (!best_diff) {
                            व्योम * ptr = b.ptr;
                            *actual_आकार = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            वापस ptr;
                        }
                    }
                }
            }
        }
        अगर (ibest >= 0) {
            ggml_sycl_बफर& b = बफर_pool[ibest];
            व्योम * ptr = b.ptr;
            *actual_आकार = b.size;
            b.ptr = nullptr;
            b.size = 0;
            वापस ptr;
        }
        व्योम * ptr;
        माप_प्रकार look_ahead_आकार = (माप_प्रकार) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (व्योम *)ggml_sycl_malloc_device(look_ahead_आकार, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        अगर (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_आकार);
            वापस nullptr;
        }

        *actual_आकार = look_ahead_आकार;
        pool_आकार += look_ahead_आकार;

#अगर_घोषित DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_आकार[id]/1024/1024), (uint32_t)(size/1024/1024));
#पूर्ण_अगर

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        वापस ptr;
    }

    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
        क्रम (पूर्णांक i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर& b = बफर_pool[i];
            अगर (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                वापस;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_आकार -= size;
    }
};

// pool with virtual memory management
#अगर defined(GGML_SYCL_SUPPORT_VMM)
काष्ठा ggml_sycl_pool_vmm : खुला ggml_sycl_pool {
    अटल स्थिर माप_प्रकार SYCL_POOL_VMM_MAX_आकार = 1ull << 35; // 32 GB

    पूर्णांक           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    माप_प्रकार    pool_used = 0;
    माप_प्रकार    pool_आकार = 0;
    माप_प्रकार    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    काष्ठा mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        व्योम * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, पूर्णांक device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        अगर (pool_addr == 0) {
            वापस;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        क्रम (स्वतः & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_आकार, ctx)));
    }

    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_बफर_ALIGNMENT);

        माप_प्रकार avail = pool_आकार - pool_used;

        अगर (size > avail) {
            // round up to the next multiple of the granularity
            माप_प्रकार reserve_आकार = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_आकार + reserve_आकार <= SYCL_POOL_VMM_MAX_आकार);

            अगर (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_आकार / (1024 * 1024), pool_आकार / (1024 * 1024),
                              (pool_आकार + reserve_आकार) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_आकार)));

            // reserve virtual address space (if not already reserved)
            अगर (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_आकार, ctx)));
            }

            // map at the end of the pool
            व्योम * map_ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_आकार, reserve_आकार,
                                    sycl::ext::oneapi::experimental::address_access_mode::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_आकार += reserve_आकार;
            ggml_sycl_memtrace_add(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_आकार);

#अगर_घोषित DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (अचिन्हित दीर्घ दीर्घ) (pool_आकार/1024/1024),
                          (अचिन्हित दीर्घ दीर्घ) (reserve_आकार/1024/1024));
#पूर्ण_अगर
        }

        GGML_ASSERT(pool_addr != 0);

        व्योम * ptr = reinterpret_cast<व्योम *>(pool_addr + pool_used);
        *actual_आकार = size;
        pool_used += size;

#अगर_घोषित DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (अचिन्हित दीर्घ दीर्घ) size, ptr);
#पूर्ण_अगर

        वापस ptr;
    }

    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
#अगर_घोषित DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (अचिन्हित दीर्घ दीर्घ) size, ptr);
#पूर्ण_अगर

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<व्योम *>(pool_addr + pool_used));
    }
};
#पूर्ण_अगर // defined(GGML_SYCL_SUPPORT_VMM)

काष्ठा ggml_sycl_pool_host : खुला ggml_sycl_pool {
    queue_ptr qptr;
    पूर्णांक       device;

    काष्ठा ggml_sycl_बफर {
        व्योम * ptr  = nullptr;
        माप_प्रकार size = 0;
    };

    // Set arbitrarly to 64
    अटल constexpr पूर्णांक          MAX_POOL_आकार{ 64 };
    std::vector<ggml_sycl_बफर> बफर_pool = std::vector<ggml_sycl_बफर>(MAX_POOL_आकार);
    माप_प्रकार                        pool_आकार   = 0;
    पूर्णांक                           गणक     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, पूर्णांक device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        क्रम (पूर्णांक i = 0; i < MAX_POOL_आकार; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            अगर (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_आकार -= b.size;
                b.size = 0;
            }
        }
    }

    व्योम * alloc(माप_प्रकार size, माप_प्रकार * actual_आकार) override {
        अगर (गणक == MAX_POOL_आकार) {
            // the pool wrapped around, no slot is reserved for this allocation
            व्योम * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (व्योम *) sycl::malloc_host(size, *qptr)));
            अगर (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                वापस nullptr;
            }
            pool_आकार    += size;
            *actual_आकार = size;
            गणक      = 0;
            वापस ptr;
        }
        ggml_sycl_बफर & b = बफर_pool[गणक];
        ++गणक;

        अगर (b.ptr != nullptr) {
            अगर (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                व्योम * ptr = b.ptr;
                *actual_आकार = b.size;
                b.ptr = nullptr;
                b.size = 0;
                वापस ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
            pool_आकार -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        व्योम * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (व्योम *) sycl::malloc_host(size, *qptr)));
        अगर (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            वापस nullptr;
        }
        pool_आकार    += size;
        *actual_आकार = size;
        वापस ptr;
    }

    व्योम मुक्त(व्योम * ptr, माप_प्रकार size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        क्रम (पूर्णांक i = 0; i < MAX_POOL_आकार; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            अगर (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                वापस;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, *qptr)));
        pool_आकार -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, पूर्णांक device) {
    // return pool for the host to speed up memory management
    वापस std::unique_ptr<ggml_sycl_pool>(नया ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, पूर्णांक device) {
#अगर defined(GGML_SYCL_SUPPORT_VMM)
    अगर (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        वापस std::unique_ptr<ggml_sycl_pool>(नया ggml_sycl_pool_vmm(qptr, device));
    }
#पूर्ण_अगर // defined(GGML_SYCL_SUPPORT_VMM)
    वापस std::unique_ptr<ggml_sycl_pool>(नया ggml_sycl_pool_leg(qptr, device));
}

/// kernels
प्रकार व्योम (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    स्थिर ggml_tensor *src0, स्थिर ggml_tensor *src1, ggml_tensor *dst,
    स्थिर अक्षर *src0_dd_i, स्थिर भग्न *src1_ddf_i, स्थिर अक्षर *src1_ddq_i,
    भग्न *dst_dd_i, स्थिर int64_t row_निचला, स्थिर int64_t row_ऊपरी,
    स्थिर int64_t src1_ncols, स्थिर int64_t src1_padded_row_आकार,
    स्थिर queue_ptr &stream);



अटल व्योम mul_mat_p021_f16_f32(
    स्थिर व्योम * __restrict__ vx, स्थिर भग्न * __restrict__ y, भग्न * __restrict__ dst,
    स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक nchannels_x, स्थिर पूर्णांक nchannels_y,
    स्थिर sycl::nd_item<3> &item_ct1) {

    स्थिर sycl::half *x = (स्थिर sycl::half *)vx;

    स्थिर पूर्णांक row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    स्थिर पूर्णांक channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    स्थिर पूर्णांक channel_x = channel / (nchannels_y / nchannels_x);

    स्थिर पूर्णांक nrows_y = ncols_x;
    स्थिर पूर्णांक nrows_dst = nrows_x;
    स्थिर पूर्णांक row_dst = row_x;

    भग्न tmp = 0.0f;

    क्रम (पूर्णांक col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        स्थिर पूर्णांक col_x = col_x0 + item_ct1.get_local_id(2);

        अगर (col_x >= ncols_x) {
            अवरोध;
        }

        // x is transposed and permuted
        स्थिर पूर्णांक ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        स्थिर भग्न xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<भग्न, sycl::rounding_mode::automatic>()[0];

        स्थिर पूर्णांक row_y = col_x;


        // y is not transposed but permuted
        स्थिर पूर्णांक iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    स्थिर पूर्णांक idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#आशय unroll
    क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    अगर (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

अटल व्योम mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    स्थिर व्योम * __restrict__ vx, स्थिर भग्न * __restrict__ y, भग्न * __restrict__ dst, स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x,
    स्थिर पूर्णांक row_stride_x, स्थिर पूर्णांक channel_stride_x,स्थिर पूर्णांक channel_stride_y, स्थिर पूर्णांक channel_x_भाजक,
    स्थिर sycl::nd_item<3> &item_ct1) {

    स्थिर sycl::half *x = (स्थिर sycl::half *)vx;

    स्थिर पूर्णांक row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    स्थिर पूर्णांक channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    स्थिर पूर्णांक channel_x = channel / channel_x_भाजक;

    स्थिर पूर्णांक nrows_dst = nrows_x;
    स्थिर पूर्णांक row_dst   = row_x;

    स्थिर पूर्णांक idst = channel*nrows_dst + row_dst;

    भग्न tmp = 0.0f;

    क्रम (पूर्णांक col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        स्थिर पूर्णांक col_x = col_x0 + item_ct1.get_local_id(2);

        अगर (col_x >= ncols_x) {
            अवरोध;
        }

        स्थिर पूर्णांक row_y = col_x;

        स्थिर पूर्णांक ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        स्थिर पूर्णांक iy = channel * channel_stride_y + row_y;

        स्थिर भग्न xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<भग्न, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#आशय unroll
    क्रम (पूर्णांक mask = WARP_आकार / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    अगर (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

अटल व्योम k_योग_rows_f32(स्थिर भग्न * x, भग्न * dst, स्थिर पूर्णांक ncols,
                           स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर पूर्णांक row = item_ct1.get_group(1);
    स्थिर पूर्णांक col = item_ct1.get_local_id(2);

    भग्न योग_2 = 0.0f;
    क्रम (पूर्णांक i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        योग_2 += x[row * ncols + i];
    }

    योग_2 = warp_reduce_योग(योग_2, item_ct1);

    अगर (col == 0) {
        dst[row] = योग_2;
    }
}


ढाँचा<typename T>
अटल अंतरभूत व्योम ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

ढाँचा <ggml_sort_order order>
__dpct_inline__ अटल व्योम
k_argsort_f32_i32(स्थिर भग्न *x, पूर्णांक *dst, स्थिर पूर्णांक ncols, पूर्णांक ncols_pad,
                  स्थिर पूर्णांक tasks_per_thread, स्थिर sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    पूर्णांक col_सूचकांक =  item_ct1.get_local_id(2);
    पूर्णांक row = item_ct1.get_group(1);

    क्रम (पूर्णांक i = 0; i < tasks_per_thread; i++) {
        पूर्णांक col = col_सूचकांक * tasks_per_thread + i;
        अगर (col >= ncols_pad) {
            वापस;
        }
    }

    स्थिर भग्न * x_row = x + row * ncols;
    स्वतः dst_row = (पूर्णांक *)dpct_local;

    // initialize indices
    क्रम (पूर्णांक i=0;i<tasks_per_thread;i++){
        पूर्णांक col = col_सूचकांक*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    क्रम (पूर्णांक k = 2; k <= ncols_pad; k *= 2) {
        क्रम (पूर्णांक j = k / 2; j > 0; j /= 2) {
            क्रम (पूर्णांक i = 0; i < tasks_per_thread; i++) {
                पूर्णांक col = col_सूचकांक * tasks_per_thread + i;
                पूर्णांक ixj = col ^ j;
                अगर (ixj > col) {
                    अगर ((col & k) == 0) {
                        अगर (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } अन्यथा {
                        अगर (dst_row[ixj] >= ncols ||
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
    क्रम (पूर्णांक i = 0; i < tasks_per_thread; i++) {
        पूर्णांक col = col_सूचकांक * tasks_per_thread + i;
        अगर (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

अटल व्योम diag_mask_inf_f32(स्थिर भग्न * x, भग्न * dst, स्थिर पूर्णांक ncols, स्थिर पूर्णांक rows_per_channel, स्थिर पूर्णांक n_past,
                              स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर पूर्णांक col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    स्थिर पूर्णांक row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    अगर (col >= ncols) {
        वापस;
    }

    स्थिर पूर्णांक i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * भग्न_उच्च;
}

अटल व्योम scale_f32(स्थिर भग्न * x, भग्न * dst, स्थिर भग्न scale, स्थिर भग्न bias, स्थिर पूर्णांक k,
                      स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर पूर्णांक i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    अगर (i >= k) {
        वापस;
    }

    dst[i] = scale * x[i] + bias;
}


अटल व्योम ggml_mul_mat_p021_f16_f32_sycl(स्थिर व्योम *vx, स्थिर भग्न *y,
                                           भग्न *dst, स्थिर पूर्णांक ncols_x,
                                           स्थिर पूर्णांक nrows_x,
                                           स्थिर पूर्णांक nchannels_x,
                                           स्थिर पूर्णांक nchannels_y,
                                           queue_ptr stream) {

    स्थिर sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    स्थिर sycl::range<3> block_dims(1, 1, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

अटल व्योम ggml_mul_mat_vec_nc_f16_f32_sycl(
    स्थिर व्योम *vx, स्थिर भग्न *y, भग्न *dst, स्थिर पूर्णांक ncols_x,
    स्थिर पूर्णांक nrows_x, स्थिर पूर्णांक row_stride_x, स्थिर पूर्णांक nchannels_x,
    स्थिर पूर्णांक nchannels_y, स्थिर पूर्णांक channel_stride_x, स्थिर पूर्णांक channel_stride_y, queue_ptr stream) {

    स्थिर sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    स्थिर sycl::range<3> block_dims(1, 1, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



अटल व्योम scale_f32_sycl(स्थिर भग्न *x, भग्न *dst, स्थिर भग्न scale, स्थिर भग्न bias,
                           स्थिर पूर्णांक k, queue_ptr stream) {
    स्थिर पूर्णांक अंक_blocks = (k + SYCL_SCALE_BLOCK_आकार - 1) / SYCL_SCALE_BLOCK_आकार;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अंक_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_आकार),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


अटल व्योम योग_rows_f32_sycl(स्थिर भग्न *x, भग्न *dst, स्थिर पूर्णांक ncols,
                              स्थिर पूर्णांक nrows, queue_ptr stream) {
    स्थिर sycl::range<3> block_dims(1, 1, WARP_आकार);
    स्थिर sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
                                 k_योग_rows_f32(x, dst, ncols, item_ct1);
                             });
}

अटल पूर्णांक अग्र_power_of_2(पूर्णांक x) {
    पूर्णांक n = 1;
    जबतक (n < x) {
        n *= 2;
    }
    वापस n;
}

अटल व्योम init_argsort_indices_padded(
        पूर्णांक * idx,
        स्थिर पूर्णांक nrows,
        स्थिर पूर्णांक ncols_pad,
        स्थिर sycl::nd_item<1> & item_ct1) {
    स्थिर माप_प्रकार gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिर माप_प्रकार योग = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;

    अगर (gid >= योग) {
        वापस;
    }

    idx[gid] = (पूर्णांक) (gid % (माप_प्रकार) ncols_pad);
}

ढाँचा <ggml_sort_order order>
अटल व्योम argsort_f32_i32_global_pass(स्थिर भग्न *            x,
                                        पूर्णांक *                    idx,
                                        स्थिर पूर्णांक                ncols,
                                        स्थिर पूर्णांक                nrows,
                                        स्थिर पूर्णांक                ncols_pad,
                                        स्थिर पूर्णांक                j,
                                        स्थिर पूर्णांक                k,
                                        स्थिर sycl::nd_item<1> & item_ct1) {
    स्थिर माप_प्रकार gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिर माप_प्रकार योग = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;

    अगर (gid >= योग) {
        वापस;
    }

    स्थिर पूर्णांक row = (पूर्णांक) (gid / (माप_प्रकार) ncols_pad);
    स्थिर पूर्णांक col = (पूर्णांक) (gid % (माप_प्रकार) ncols_pad);
    स्थिर पूर्णांक ixj = col ^ j;

    अगर (ixj <= col || ixj >= ncols_pad) {
        वापस;
    }

    स्थिर माप_प्रकार base  = (माप_प्रकार) row * (माप_प्रकार) ncols_pad;
    स्थिर माप_प्रकार pos_a = base + (माप_प्रकार) col;
    स्थिर माप_प्रकार pos_b = base + (माप_प्रकार) ixj;

    स्थिर पूर्णांक a = idx[pos_a];
    स्थिर पूर्णांक b = idx[pos_b];

    bool do_swap = false;

    अगर ((col & k) == 0) {
        अगर (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] > x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b] :
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] < x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b]))) {
            do_swap = true;
        }
    } अन्यथा {
        अगर (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] < x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b] :
                  x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) a] > x[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) b]))) {
            do_swap = true;
        }
    }

    अगर (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

अटल व्योम copy_argsort_indices_unpadded(स्थिर पूर्णांक *              idx_padded,
                                          पूर्णांक *                    dst,
                                          स्थिर पूर्णांक                nrows,
                                          स्थिर पूर्णांक                ncols,
                                          स्थिर पूर्णांक                ncols_pad,
                                          स्थिर sycl::nd_item<1> & item_ct1) {
    स्थिर माप_प्रकार gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    स्थिर माप_प्रकार योग = (माप_प्रकार) nrows * (माप_प्रकार) ncols;

    अगर (gid >= योग) {
        वापस;
    }

    स्थिर पूर्णांक row = (पूर्णांक) (gid / (माप_प्रकार) ncols);
    स्थिर पूर्णांक col = (पूर्णांक) (gid % (माप_प्रकार) ncols);

    dst[(माप_प्रकार) row * (माप_प्रकार) ncols + (माप_प्रकार) col] = idx_padded[(माप_प्रकार) row * (माप_प्रकार) ncols_pad + (माप_प्रकार) col];
}

अटल व्योम argsort_f32_i32_sycl(स्थिर भग्न *x, पूर्णांक *dst, स्थिर पूर्णांक ncols,
                                 स्थिर पूर्णांक nrows, ggml_sort_order order,
                                 queue_ptr stream, पूर्णांक device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    स्थिर पूर्णांक ncols_pad = अग्र_power_of_2(ncols);
    स्थिर माप_प्रकार shared_mem = (माप_प्रकार) ncols_pad * माप(पूर्णांक);
    स्थिर माप_प्रकार smpbo = ggml_sycl_info().devices[device].smpbo;

    अगर (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<पूर्णांक> idx_padded_alloc(pool, (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad);
        पूर्णांक *                     idx_padded = idx_padded_alloc.लो();

        constexpr माप_प्रकार block_आकार_2     = 256;
        स्थिर माप_प्रकार     योग_padded   = (माप_प्रकार) nrows * (माप_प्रकार) ncols_pad;
        स्थिर माप_प्रकार     nblocks_padded = (योग_padded + block_आकार_2 - 1) / block_आकार_2;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_आकार_2), sycl::range<1>(block_आकार_2)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        क्रम (पूर्णांक k = 2; k <= ncols_pad; k *= 2) {
            क्रम (पूर्णांक j = k / 2; j > 0; j /= 2) {
                अगर (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_आकार_2), sycl::range<1>(block_आकार_2)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } अन्यथा अगर (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_आकार_2), sycl::range<1>(block_आकार_2)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } अन्यथा {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        स्थिर माप_प्रकार योग   = (माप_प्रकार) nrows * (माप_प्रकार) ncols;
        स्थिर माप_प्रकार nblocks = (योग + block_आकार_2 - 1) / block_आकार_2;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_आकार_2), sycl::range<1>(block_आकार_2)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        वापस;
    }

    पूर्णांक nth = 1;
    पूर्णांक max_block_आकार = ggml_sycl_info().max_work_group_sizes[device];
    जबतक (nth < ncols_pad && nth < max_block_आकार)
        nth *= 2;
    अगर (nth > max_block_आकार)
        nth = max_block_आकार;

    स्थिर पूर्णांक tasks_per_thread = ncols_pad / nth;

    स्थिर sycl::range<3> block_dims(1, 1, nth);
    स्थिर sycl::range<3> block_nums(1, nrows, 1);

    अगर (order == GGML_SORT_ORDER_ASC) {
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
                            .लो());
                });
        });
    } अन्यथा अगर (order == GGML_SORT_ORDER_DESC) {
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
                            .लो());
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
अटल व्योम शीर्ष_k_scan_merge_f32(
    स्थिर भग्न *   src_vals,
    स्थिर int32_t * src_map,
    स्थिर पूर्णांक       begin,
    स्थिर पूर्णांक       end,
    स्थिर पूर्णांक       k,
    स्थिर पूर्णांक       block_आकार_2,
    भग्न *         shared_vals,
    पूर्णांक *           shared_idx,
    भग्न *         out_vals,
    int32_t *       out_idx,
    स्थिर bool      swap01,
    स्थिर sycl::nd_item<1> & item_ct1
) {
    स्थिर पूर्णांक tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    भग्न * lv = shared_vals + tid;
    पूर्णांक * li = shared_idx + tid;

    क्रम (पूर्णांक i = 0; i < k; i++) {
        lv[i * block_आकार_2] = -भग्न_उच्च;
        li[i * block_आकार_2] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    भग्न kth = -भग्न_उच्च;

    क्रम (पूर्णांक col = begin + tid; col < end; col += block_आकार_2) {
        भग्न val = src_vals[col];

        अगर (val > kth) {
            पूर्णांक pos = k - 1;
            जबतक (pos > 0 && val > lv[(pos - 1) * block_आकार_2]) {
                pos--;
            }

            क्रम (पूर्णांक i = k - 1; i > pos; i--) {
                lv[i * block_आकार_2] = lv[(i - 1) * block_आकार_2];
                li[i * block_आकार_2] = li[(i - 1) * block_आकार_2];
            }
            lv[pos * block_आकार_2] = val;
            li[pos * block_आकार_2] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_आकार_2];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    अगर (tid != 0) {
        वापस;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    भग्न * fv = shared_vals + (माप_प्रकार) k * block_आकार_2;
    पूर्णांक * fi = shared_idx + (माप_प्रकार) k * block_आकार_2;

    क्रम (पूर्णांक i = 0; i < k; i++) {
        fv[i] = -भग्न_उच्च;
        fi[i] = -1;
    }

    भग्न fkth = -भग्न_उच्च;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    क्रम (पूर्णांक t = 0; t < block_आकार_2; t++) {
        क्रम (पूर्णांक i = 0; i < k; i++) {
            भग्न val = shared_vals[i * block_आकार_2 + t];

            अगर (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                अवरोध;
            }

            पूर्णांक idx = shared_idx[i * block_आकार_2 + t];

            पूर्णांक pos = k - 1;
            जबतक (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            क्रम (पूर्णांक j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    अगर (out_vals) {
        क्रम (पूर्णांक i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    क्रम (पूर्णांक i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    अगर (swap01 && k > 1) {
        int32_t अस्थायी = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = अस्थायी;
    }
}

अटल व्योम शीर्ष_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    स्थिर भग्न * src,
    int32_t * dst_indices,
    स्थिर int64_t ncols,
    स्थिर int64_t nrows,
    स्थिर पूर्णांक k,
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
    अगर (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        स्थिर int64_t want = ncols / split_block;
        nsplit = (पूर्णांक) (want > max_splits ? max_splits : want);
    }

    अगर (nsplit > 1) {
        स्थिर पूर्णांक nchunk = (पूर्णांक) ((ncols + nsplit - 1) / nsplit);
        स्थिर माप_प्रकार ncand = (माप_प्रकार) nrows * nsplit * k;

        ggml_sycl_pool_alloc<भग्न>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        भग्न *   pv = part_vals.लो();
        int32_t * pi = part_idx.लो();

        स्थिर sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<भग्न, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    स्थिर पूर्णांक grp = item_ct1.get_group(0);
                    स्थिर पूर्णांक row = grp / nsplit;
                    स्थिर पूर्णांक part = grp % nsplit;

                    स्थिर पूर्णांक begin = part * nchunk;
                    पूर्णांक end = begin + nchunk;
                    अगर (end > (पूर्णांक) ncols) {
                        end = (पूर्णांक) ncols;
                    }

                    शीर्ष_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().लो(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().लो(),
                        pv + (माप_प्रकार) grp * k, pi + (माप_प्रकार) grp * k, false, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<भग्न, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    स्थिर पूर्णांक row = item_ct1.get_group(0);
                    स्थिर माप_प्रकार off = (माप_प्रकार) row * nsplit * k;

                    शीर्ष_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().लो(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().लो(),
                        nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
                });
        });

        वापस;
    }

    स्थिर पूर्णांक block_आकार_2 = 128;

    स्थिर sycl::range<1> block_dims(block_आकार_2);
    स्थिर sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<भग्न, 1> shared_vals(sycl::range<1>((block_आकार_2 + 1) * k), cgh);
        sycl::local_accessor<पूर्णांक, 1> shared_idx(sycl::range<1>((block_आकार_2 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                स्थिर पूर्णांक row = item_ct1.get_group(0);

                अगर (row >= nrows) वापस;

                शीर्ष_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (पूर्णांक) ncols, k, block_आकार_2,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().लो(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().लो(),
                    nullptr, dst_indices + (int64_t) row * k, true, item_ct1);
            });
    });
}

अटल व्योम argmax_f32_i32_sycl(स्थिर भग्न *x, पूर्णांक *dst, स्थिर पूर्णांक ncols,
                               स्थिर पूर्णांक nrows, queue_ptr stream) {
    स्थिर sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_आकार);
    स्थिर sycl::range<3> block_nums(1, nrows, 1);
    स्थिर माप_प्रकार shared_mem = 256 * माप(भग्न);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<भग्न, 1> shared_तथ्य(
            sycl::range<1>(shared_mem/माप(भग्न)), cgh);
        sycl::local_accessor<पूर्णांक, 1> shared_indices(
            sycl::range<1>(shared_mem/माप(भग्न)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                स्थिर पूर्णांक tid = item_ct1.get_local_id(2);
                स्थिर पूर्णांक row = item_ct1.get_global_id(1);

                भग्न max_val = -अनन्त;
                पूर्णांक max_idx = -1;

                क्रम (पूर्णांक col = tid; col < ncols; col += 256) {
                    भग्न val = x[row * ncols + col];
                    अगर (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_तथ्य[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                क्रम (पूर्णांक stride = 256/2; stride > 0; stride >>= 1) {
                    अगर (tid < stride) {
                        भग्न val1 = shared_तथ्य[tid];
                        भग्न val2 = shared_तथ्य[tid + stride];
                        अगर (val2 > val1) {
                            shared_तथ्य[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                अगर (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
अटल व्योम diag_mask_inf_f32_sycl(स्थिर भग्न *x, भग्न *dst,
                                   स्थिर पूर्णांक ncols_x, स्थिर पूर्णांक nrows_x,
                                   स्थिर पूर्णांक rows_per_channel, स्थिर पूर्णांक n_past,
                                   queue_ptr stream) {
    स्थिर sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_आकार, 1);
    स्थिर पूर्णांक block_अंक_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_आकार - 1) / SYCL_DIAG_MASK_INF_BLOCK_आकार;
    स्थिर sycl::range<3> block_nums(1, block_अंक_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

अटल dpct::err0 ggml_sycl_cpy_tensor_2d(व्योम *dst,
                                          स्थिर काष्ठा ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_निचला, int64_t i1_ऊपरी,
                                          queue_ptr stream) प्रयास {

    dpct::memcpy_direction kind;
    अक्षर * src_ptr;
    अगर (ggml_backend_बफर_is_host(src->बफर)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (अक्षर *) src->तथ्य;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } अन्यथा अगर (ggml_backend_बफर_is_sycl(src->बफर)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (अक्षर *) src->तथ्य;
    } अन्यथा अगर (ggml_backend_बफर_is_sycl_split(src->बफर)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_निचला == 0 && i1_ऊपरी == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        पूर्णांक id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (अक्षर *) extra->तथ्य_device[id];
    } अन्यथा {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    अक्षर * dst_ptr = (अक्षर *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    स्थिर क्रमागत ggml_type type = src->type;
    स्थिर int64_t ts = ggml_type_आकार(type);
    स्थिर int64_t bs = ggml_blck_आकार(type);
    int64_t i1_diff = i1_ऊपरी - i1_निचला;

    स्थिर अक्षर * x = src_ptr + i1_निचला*nb1 + i2*nb2 + i3*nb3;
    अगर (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        वापस CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } अन्यथा अगर (nb0 == ts) {
        वापस CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } अन्यथा {
        क्रम (int64_t i1 = 0; i1 < i1_diff; i1++) {
            स्थिर व्योम * rx = (स्थिर व्योम *) ((स्थिर अक्षर *) x + i1*nb1);
            व्योम * rd = (व्योम *) (dst_ptr + i1*ts*ne0/bs);
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
            अगर (r != 0) वापस r;
        }
        वापस 0;
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अंतरभूत व्योम ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    स्थिर ggml_tensor *src0, स्थिर ggml_tensor *src1, ggml_tensor *dst,
    स्थिर अक्षर *src0_dd_i, स्थिर भग्न *src1_ddf_i, स्थिर अक्षर *src1_ddq_i,
    भग्न *dst_dd_i, स्थिर int64_t row_निचला, स्थिर int64_t row_ऊपरी,
    स्थिर int64_t src1_ncols, स्थिर int64_t src1_padded_row_आकार,
    स्थिर queue_ptr &stream) प्रयास {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    स्थिर int64_t row_diff = row_ऊपरी - row_निचला;

    पूर्णांक id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    स्थिर int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    पूर्णांक ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#अगर_घोषित GGML_SYCL_F16
    bool use_fp16 = true;  // TODO(Yu) SYCL capability check
#अन्यथा
    bool use_fp16 = false;
#पूर्ण_अगर

#अगर GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    अगर (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        अगर (src1->type != GGML_TYPE_BF16) {
            स्थिर to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.लो(), src1_ncols*ne10, stream);
        } अन्यथा {
            stream->स_नकल(src1_as_bf16.लो(), src1_ddf_i, src1_ncols*ne10*माप(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.लो(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<भग्न>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_आकार);
        वापस;
    }
#पूर्ण_अगर

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    bool src0_reordered = false;
    अगर (स्थिर स्वतः * src0_extra = static_cast<स्थिर ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    अगर (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        वापस;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    अगर ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        अगर (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            स्थिर to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            माप_प्रकार ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.लो(), ne, stream);
        }
        स्थिर sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (स्थिर sycl::half *)src1_ddf_i
                : src1_as_f16.लो();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        अगर (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            स्थिर to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            माप_प्रकार ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.लो(), ne, stream);
        }
        स्थिर sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (स्थिर sycl::half *)src0_dd_i
                                         : src0_as_f16.लो();

#अगर GGML_SYCL_DNNL
        अगर (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<भग्न>(), stream);
        }
        अन्यथा
#पूर्ण_अगर
        {
            स्थिर भग्न alpha = 1.0f;
            स्थिर भग्न beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_तथ्य_t::real_half, ne00,
                src1_ptr, dpct::library_तथ्य_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_तथ्य_t::real_float, ldc,
                dpct::library_तथ्य_t::real_float)));
        }
    } अन्यथा {
        ggml_sycl_pool_alloc<भग्न> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<भग्न> src1_ddq_as_f32(ctx.pool());
        अगर (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            स्थिर to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.लो(), row_diff*ne00, stream);
        }
        अगर (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            स्थिर to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.लो(), src1_ncols*ne10, stream);
        }
        स्थिर भग्न * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (स्थिर भग्न *) src0_dd_i : src0_ddq_as_f32.लो();
        स्थिर भग्न * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (स्थिर भग्न *) src1_ddf_i : src1_ddq_as_f32.लो();

        {
#अगर GGML_SYCL_DNNL
            स्थिर int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            स्थिर bool use_mkl_direct = gemm_flops < 256 * 256 * 256;
            अगर (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<भग्न>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<भग्न>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<भग्न>(), stream);
            }
            अन्यथा
#पूर्ण_अगर
            {
                स्थिर भग्न alpha = 1.0f;
                स्थिर भग्न beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_मूल्य(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_मूल्य(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_आकार);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अंतरभूत व्योम ggml_sycl_op_योग(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    योग_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

अंतरभूत व्योम ggml_sycl_op_योग_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int64_t ncols = dst->src[0]->ne[0];
    स्थिर int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    योग_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

अंतरभूत व्योम ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int64_t ncols = dst->src[0]->ne[0];
    स्थिर int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    योग_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);

    main_stream->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


अंतरभूत व्योम ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->तथ्य);


    स्थिर int64_t ncols = dst->src[0]->ne[0];
    स्थिर int64_t nrows = ggml_nrows(dst->src[0]);

    क्रमागत ggml_sort_order order = (क्रमागत ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (पूर्णांक *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

अटल व्योम ggml_sycl_op_शीर्ष_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(src0->तथ्य);
    int32_t * dst_dd = static_cast<int32_t *>(dst->तथ्य);

    स्थिर पूर्णांक k = dst->ne[0];
    स्थिर int64_t ncols = src0->ne[0];
    स्थिर int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    अगर (k <= SYCL_शीर्ष_K_SCAN_MERGE_MAX_K) {
        शीर्ष_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } अन्यथा {
        ggml_sycl_शीर्ष_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

अंतरभूत व्योम ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->तथ्य);

    स्थिर int64_t ncols = dst->src[0]->ne[0];
    स्थिर int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

अंतरभूत व्योम ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर int64_t ne00 = dst->src[0]->ne[0];
    स्थिर int64_t ne01 = dst->src[0]->ne[1];
    स्थिर पूर्णांक nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    स्थिर पूर्णांक n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

अटल व्योम tri_f32_sycl(
    स्थिर भग्न * src,
    भग्न * dst,
    स्थिर int64_t ne0,
    स्थिर int64_t ne1,
    स्थिर int64_t ne2,
    स्थिर int64_t ne3,
    स्थिर ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    स्थिर माप_प्रकार योग = (माप_प्रकार) ne0 * (माप_प्रकार) ne1 * (माप_प्रकार) ne2 * (माप_प्रकार) ne3;

    main_stream->parallel_for(sycl::range<1>(योग), [=](sycl::id<1> tid) {
        स्थिर int64_t idx = (int64_t) tid[0];

        स्थिर int64_t i0 = idx % ne0;
        स्थिर int64_t t1 = idx / ne0;
        स्थिर int64_t i1 = t1 % ne1;

        bool keep = false;
        चयन (ttype) {
            हाल GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); अवरोध;
            हाल GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); अवरोध;
            हाल GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); अवरोध;
            हाल GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); अवरोध;
            शेष: keep = false; अवरोध;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

अटल व्योम ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिर ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(src0->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    स्थिर int64_t ne0 = src0->ne[0];
    स्थिर int64_t ne1 = src0->ne[1];
    स्थिर int64_t ne2 = src0->ne[2];
    स्थिर int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


अंतरभूत व्योम ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    स्थिर भग्न * src0_dd = static_cast<स्थिर भग्न *>(dst->src[0]->तथ्य);
    भग्न *       dst_dd  = static_cast<भग्न *>(dst->तथ्य);

    भग्न scale;
    भग्न bias;
    स_नकल(&scale, (भग्न *) dst->op_params + 0, माप(भग्न));
    स_नकल(&bias,  (भग्न *) dst->op_params + 1, माप(भग्न));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

अटल व्योम ggml_sycl_set_peer_access(स्थिर पूर्णांक n_tokens, [[maybe_unused]] पूर्णांक main_device) {
    अटल bool peer_access_enabled = false;

    स्थिर bool enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_आकार;

    अगर (peer_access_enabled == enable_peer_access) {
        वापस;
    }

#अगर_घोषित न_संशोधन
    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        क्रम (पूर्णांक id_other = 0; id_other < ggml_sycl_info().device_गिनती; ++id_other) {
            अगर (i == id_other) {
                जारी;
            }
            अगर (i != main_device && id_other != main_device) {
                जारी;
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
#पूर्ण_अगर // NDEBUG

    peer_access_enabled = enable_peer_access;
}

ढाँचा <ढाँचा <पूर्णांक> typename quantize_f>
अटल व्योम ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor *src0,
                                 स्थिर ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) प्रयास {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    स्थिर int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    स्थिर int64_t ne0 = dst->ne[0];
    स्थिर int64_t ne1 = dst->ne[1];

    स्थिर पूर्णांक nb2 = dst->nb[2];
    स्थिर पूर्णांक nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(dst->बफर));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(src1->बफर));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    स्थिर int64_t i02_भाजक = ne12 / ne02;

    स्थिर माप_प्रकार src0_ts = ggml_type_आकार(src0->type);
    स्थिर माप_प्रकार src0_bs = ggml_blck_आकार(src0->type);
    स्थिर माप_प्रकार q8_1_ts = माप(block_q8_1);
    स्थिर माप_प्रकार q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    स्थिर bool src0_is_contiguous = ggml_is_contiguous(src0);
    स्थिर bool src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_आकार = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    स्थिर bool split = ggml_backend_बफर_is_sycl_split(src0->बफर);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<भग्न, GGML_SYCL_MAX_DEVICES> tensor_split;
    अगर (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_बफर_type_context * buft_ctx = (ggml_backend_sycl_split_बफर_type_context *) src0->बफर->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    काष्ठा dev_तथ्य {
        ggml_sycl_pool_alloc<अक्षर> src0_dd_alloc;
        ggml_sycl_pool_alloc<भग्न> src1_ddf_alloc;
        ggml_sycl_pool_alloc<अक्षर> src1_ddq_alloc;
        ggml_sycl_pool_alloc<भग्न> dst_dd_alloc;

        अक्षर *src0_dd = nullptr;
        भग्न *src1_ddf = nullptr; // float
        अक्षर *src1_ddq = nullptr;  // q8_1
        भग्न *dst_dd = nullptr;

        int64_t row_निचला;
        int64_t row_ऊपरी;
    };

    dev_तथ्य dev[GGML_SYCL_MAX_DEVICES];

    पूर्णांक used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        // by default, use all rows
        dev[i].row_निचला  = 0;
        dev[i].row_ऊपरी = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        अगर (split) {
            स्थिर int64_t rounding = get_row_rounding(src0->type, tensor_split);

            अगर (i != 0) {
                dev[i].row_निचला  = ne01*tensor_split[i];
                अगर (dev[i].row_निचला < ne01) {
                    dev[i].row_निचला -= dev[i].row_निचला % rounding;
                }
            }

            अगर (i != ggml_sycl_info().device_गिनती - 1) {
                dev[i].row_ऊपरी  = ne01*tensor_split[i + 1];
                अगर (dev[i].row_ऊपरी < ne01) {
                    dev[i].row_ऊपरी -= dev[i].row_ऊपरी % rounding;
                }
            }
        }
    }

    constexpr bool quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_आकार>,
                                                      no_quantize_q8_1<QK8_1 / WARP_आकार>>;
    क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
        अगर ((!split && i != ctx.device) || dev[i].row_निचला == dev[i].row_ऊपरी) {
            जारी;
        }

        used_devices++;

        स्थिर bool src1_on_device = i == ctx.device;
        स्थिर bool  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        अगर (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (अक्षर *) src0_extra->तथ्य_device[i] : (अक्षर *) src0->तथ्य;
        } अन्यथा {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        अगर (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (भग्न *) src1->तथ्य;
        } अन्यथा {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        अगर constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_आकार*q8_1_ts/q8_1_bs);

            अगर (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                प्रयास {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_आकार, stream);
                } पकड़ो (sycl::exception स्थिर &exc) {
                    std::अ_त्रुटि << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __खाता__
                              << ", line:" << __LINE__ << std::प_पूर्ण;
                    std::निकास(1);
                }
            }
        }

        अगर (dst_on_device) {
            dev[i].dst_dd = (भग्न *) dst->तथ्य;
        } अन्यथा {
            स्थिर माप_प्रकार आकार_dst_ddf = split ? (dev[i].row_ऊपरी - dev[i].row_निचला)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), आकार_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    अगर (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    स्थिर int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    क्रम (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        स्थिर int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        स्थिर int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
            अगर ((!split && i != ctx.device) || dev[i].row_निचला == dev[i].row_ऊपरी) {
                जारी;
            }

            स्थिर bool src1_on_device = i == ctx.device;
            स्थिर bool  dst_on_device = i == ctx.device;
            स्थिर int64_t row_diff = dev[i].row_ऊपरी - dev[i].row_निचला;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            अगर (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            क्रम (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                स्थिर int64_t i03 = i0 / ne12;
                स्थिर int64_t i02 = i0 % ne12;

                स्थिर माप_प्रकार src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_आकार*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                अक्षर  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_भाजक) * (ne01*ne00*src0_ts)/src0_bs;
                भग्न * src1_ddf_i = (भग्न *) ((अक्षर *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_आकार(src1->type));
                अक्षर  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                भग्न *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                अगर (i == ctx.device) {
                    dst_dd_i += dev[i].row_निचला; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                अगर (src1_is_contiguous) {
                    अगर (i != ctx.device) {
                        अगर constexpr (quantize_enabled) {
                            अक्षर * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->स_नकल(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_आकार * q8_1_ts / q8_1_bs)
                                                    .रुको()));
                        } अन्यथा {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            स्थिर अक्षर * src1_ddf_i_source = (स्थिर अक्षर *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_आकार(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_आकार(src1->type))));
                        }
                    }
                } अन्यथा {
                    अगर (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } अन्यथा {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    अगर constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        प्रयास {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_आकार, stream);
                        } पकड़ो (स्थिर sycl::exception & exc) {
                            std::अ_त्रुटि << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __खाता__ << ", line:" << __LINE__ << std::प_पूर्ण;
                            std::निकास(1);
                        }
                    }
                }

                अगर (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_भाजक == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_भाजक, dev[i].row_निचला, dev[i].row_ऊपरी, stream));
                }
                अगर (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_आकार = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_निचला, dev[i].row_ऊपरी, src1_ncols, src1_padded_col_आकार, stream)));

                // copy dst to host or other device if necessary
                अगर (!dst_on_device) {
                    व्योम * dst_off_device = dst->तथ्य;
                    अगर (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        भग्न * dhf_dst_i = (भग्न *) ((अक्षर *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*माप(भग्न));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_निचला;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * माप(भग्न), dst_dd_i,
                            row_diff * माप(भग्न), row_diff * माप(भग्न),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } अन्यथा {
                        भग्न * dhf_dst_i = (भग्न *) ((अक्षर *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*माप(भग्न));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->स_नकल(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * माप(भग्न)).रुको()));
                    }
                }

                // add event for the main device to wait on until other device is done
                अगर (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    अगर (split && ggml_sycl_info().device_गिनती > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; ++i) {
            अगर (dev[i].row_निचला == dev[i].row_ऊपरी) {
                जारी;
            }
            क्रम (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

अटल व्योम ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

अटल व्योम ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

अटल व्योम ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

अटल व्योम ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

अटल व्योम ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

अटल व्योम ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

अटल व्योम ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

अटल व्योम ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor *src0,
                                       स्थिर ggml_tensor *src1,
                                       ggml_tensor *dst) प्रयास {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(src0->बफर));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];

    स्थिर int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    व्योम  * src0_ddq = src0->तथ्य;
    भग्न * src1_ddf = (भग्न *) src1->तथ्य;
    भग्न * dst_ddf  = (भग्न *) dst->तथ्य;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor *src0,
                                     स्थिर ggml_tensor *src1,
                                     ggml_tensor *dst) प्रयास {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(src0->बफर));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    स्थिर int64_t ne00 = src0->ne[0];
    स्थिर int64_t ne01 = src0->ne[1];
    स्थिर int64_t ne02 = src0->ne[2];

    स्थिर int64_t nb01 = src0->nb[1];
    स्थिर int64_t nb02 = src0->nb[2];

    स्थिर int64_t ne12 = src1->ne[2];
    स्थिर int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    व्योम  * src0_ddq = src0->तथ्य;
    भग्न * src1_ddf = (भग्न *) src1->तथ्य;
    भग्न * dst_ddf  = (भग्न *) dst->तथ्य;

    स्थिर int64_t row_stride_x = nb01 / माप(sycl::half);
    स्थिर int64_t channel_stride_x = nb02 / माप(sycl::half);
    स्थिर int64_t channel_stride_y = nb11 / माप(भग्न);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम k_compute_batched_ptrs(स्थिर sycl::half * src0_as_f16, स्थिर sycl::half * src1_as_f16, व्योम * dst,
                                   स्थिर व्योम ** ptrs_src, व्योम ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   माप_प्रकार nb02, माप_प्रकार nb03, माप_प्रकार nb12, माप_प्रकार nb13, माप_प्रकार nbd2, माप_प्रकार nbd3,
                                   int64_t r2, int64_t r3, स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    स्थिर int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    अगर (i13 >= ne13 || i12 >= ne12) {
        वापस;
    }

    स्थिर int64_t i03 = i13 / r3;
    स्थिर int64_t i02 = i12 / r2;

    स्थिर uint8_t * src0_bytes = reinterpret_cast<स्थिर uint8_t *>(src0_as_f16);
    स्थिर uint8_t * src1_bytes = reinterpret_cast<स्थिर uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

अटल व्योम ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0,
                                           स्थिर ggml_tensor * src1, ggml_tensor * dst) प्रयास {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(src0->बफर));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विआधारी_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    स्थिर sycl::half * src0_f16 = static_cast<स्थिर sycl::half *>(src0->तथ्य);
    भग्न *            dst_ddf  = static_cast<भग्न *>(dst->तथ्य);

    स्थिर sycl::half * src1_f16       = static_cast<स्थिर sycl::half *>(src1->तथ्य);
#अगर GGML_SYCL_DNNL
    स्थिर माप_प्रकार       type_आकार_src0 = ggml_type_आकार(src0->type);
#पूर्ण_अगर
    स्थिर माप_प्रकार       type_आकार_src1 = ggml_type_आकार(src1->type);

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_आकार_src1;
    int64_t                          s12 = nb12 / type_आकार_src1;
    int64_t                          s13 = nb13 / type_आकार_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    अगर (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#अगर GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        पूर्णांक last_dim=0;
        पूर्णांक last_str=0;
        माप_प्रकार largest_str=0;
        क्रम(पूर्णांक i = 0; i< 4; i++){
            // last stride is always the largest
            अगर(src1->nb[i] == largest_str){
                अगर(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            अगर(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        स्थिर int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_आकार_src1;
        src1_f16_alloc.alloc(ne_src1);
        स्थिर to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.लो(), ne_src1, queue);
# अन्यथा
        स्थिर int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        स्थिर to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.लो(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#पूर्ण_अगर

        src1_f16 = src1_f16_alloc.लो();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = true;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_तथ्य_t mkl_compute_type = dpct::library_तथ्य_t::real_float;
    dpct::library_तथ्य_t mkl_तथ्य_type    = dpct::library_तथ्य_t::real_float;

    // dst strides
    माप_प्रकार nbd2 = dst->nb[2];
    माप_प्रकार nbd3 = dst->nb[3];

    स्थिर भग्न alpha_f32 = 1.0f;
    स्थिर भग्न beta_f32  = 0.0f;

    स्थिर व्योम * alpha = &alpha_f32;
    स्थिर व्योम * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    स्थिर int64_t r2 = ne12 / ne02;
    स्थिर int64_t r3 = ne13 / ne03;

#अगर GGML_SYCL_DNNL
    अगर (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_आकार_src0;
            int64_t str_a1 = nb01 / type_आकार_src0;
            int64_t str_a2 = nb02 / type_आकार_src0;

            int64_t str_b0 = nb10 / type_आकार_src1;
            int64_t str_b1 = nb11 / type_आकार_src1;
            int64_t str_b2 = nb12 / type_आकार_src1;

            स्वतः launch_gemm_for_batches = [&ctx, queue](स्थिर sycl::half *src0,
                                                स्थिर sycl::half *src1, भग्न *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                bool supported_broadcast = batchb == batcha ? true
                        : batchb == 1 || batcha == 1        ? true
                                                            : false;
                अगर (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<भग्न>(), queue, batcha, batchb);
                } अन्यथा {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    अगर (batches0 > batches1) {
                        int64_t अंक_mul_mats = batches1;
                        int64_t sub_batch = batches0 / अंक_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        क्रम (int64_t i0 = 0; i0 < अंक_mul_mats; i0++) {
                            स्थिर sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            स्थिर sycl::half *src1_shifted = src1 + (sb2 * i0);
                            भग्न *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<भग्न>(),
                                    queue, sub_batch, 1);
                        }
                    } अन्यथा {
                        int64_t अंक_mul_mats = batches0;
                        int64_t sub_batch = batches1 / अंक_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        क्रम (int64_t i1 = 0; i1 < अंक_mul_mats; i1++) {
                            स्थिर sycl::half *src0_shifted = src0 + (sa2 * i1);
                            स्थिर sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            भग्न *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<भग्न>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            स्थिर bool cont_batches_dim2_a = nb02 * ne02 == nb03;
            स्थिर bool cont_batches_dim2_b = nb12 * ne12 == nb13;
            स्थिर bool cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            स्थिर bool cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            अगर (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / माप(भग्न));
            } अन्यथा अगर (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_आकार_src0;
                int64_t str_b3 = nb13 / type_आकार_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / माप(भग्न));
            } अन्यथा {
                क्रम (int64_t b_a = 0; b_a < ne03; b_a++) {
                    स्थिर sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_आकार_src0);
                    स्थिर sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_आकार_src1);
                    भग्न *dst_shifted = dst_ddf + (nb3 * b_a / माप(भग्न));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / माप(भग्न));
                }
            }

    }
    अन्यथा
#पूर्ण_अगर
    {
        अगर (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            स्थिर int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            स्थिर int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_तथ्य_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_तथ्य_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_तथ्य_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } अन्यथा {
            स्थिर पूर्णांक ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<स्थिर व्योम *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<व्योम *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<भग्न>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                स्थिर व्योम ** ptrs_src_get = ptrs_src.लो();
                व्योम **       ptrs_dst_get = ptrs_dst.लो();
                माप_प्रकार        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * माप(sycl::half);
                माप_प्रकार        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * माप(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (स्थिर व्योम **) (ptrs_src.लो() + 0 * ne23), dpct::library_तथ्य_t::real_half, nb01 / nb00,
                (स्थिर व्योम **) (ptrs_src.लो() + 1 * ne23), dpct::library_तथ्य_t::real_half, s11, beta,
                (व्योम **) (ptrs_dst.लो() + 0 * ne23), mkl_तथ्य_type, ne0, ne23, mkl_compute_type, matrix_info.लो())));
        }
    }
} पकड़ो (स्थिर sycl::exception & exc) {
    std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__ << ", line:" << __LINE__ << std::प_पूर्ण;
    std::निकास(1);
}

क्रमागत श्रेणी mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

अंतरभूत bool ggml_sycl_supports_mmq(क्रमागत ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    वापस false;
}

अंतरभूत bool ggml_sycl_supports_reorder_mul_mat_sycl(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q8_0:
            वापस true;
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
            वापस !g_ggml_sycl_prioritize_dmmv;
        शेष:
            वापस false;
    }
}

अंतरभूत bool ggml_sycl_supports_reorder_dmmv(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
            वापस true;
        शेष:
            वापस false;
    }
}

अंतरभूत bool ggml_sycl_supports_reorder_mmvq(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_IQ3_XXS:
        हाल GGML_TYPE_IQ3_S:
            वापस true;
        शेष:
            वापस false;
    }
}

अटल bool ggml_sycl_supports_reorder_esimd(क्रमागत ggml_type type) {
#अगर_घोषित GGML_SYCL_DMMV_HAS_ESIMD
    चयन (type) {
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_Q8_0:
            वापस true;
        शेष:
            वापस false;
    }
#अन्यथा
    GGML_UNUSED(type);
    वापस false;
#पूर्ण_अगर
}

अटल bool ggml_sycl_supports_dmmv(क्रमागत ggml_type type) {
    चयन (type) {
        हाल GGML_TYPE_Q1_0:
        हाल GGML_TYPE_Q4_0:
        हाल GGML_TYPE_Q4_1:
        हाल GGML_TYPE_Q5_0:
        हाल GGML_TYPE_Q5_1:
        हाल GGML_TYPE_Q8_0:
        हाल GGML_TYPE_Q2_K:
        हाल GGML_TYPE_Q3_K:
        हाल GGML_TYPE_Q4_K:
        हाल GGML_TYPE_Q5_K:
        हाल GGML_TYPE_Q6_K:
        हाल GGML_TYPE_F16:
        हाल GGML_TYPE_BF16:
            वापस true;
        शेष:
            वापस false;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
अटल अंतरभूत व्योम * sycl_ext_malloc_device(dpct::queue_ptr stream, माप_प्रकार size) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#अगर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    अगर (use_async) {
        व्योम * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        वापस ptr;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#पूर्ण_अगर
    वापस ggml_sycl_malloc_device(size, *stream);
}

अटल अंतरभूत व्योम sycl_ext_free(dpct::queue_ptr stream, व्योम * ptr) {
    bool use_async = g_ggml_sycl_use_async_mem_op;
#अगर defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    अगर (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        वापस;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#पूर्ण_अगर
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
काष्ठा sycl_reorder_अस्थायी_बफर {
    व्योम *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_अस्थायी_बफर(dpct::queue_ptr stream, माप_प्रकार size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#अगर_घोषित GGML_SYCL_HOST_MEM_FALLBACK
        अगर (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            अगर (ptr) {
                host_fallback = true;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#पूर्ण_अगर
    }

    ~sycl_reorder_अस्थायी_बफर() {
        अगर (!ptr) {
            वापस;
        }
        अगर (host_fallback) {
            sycl::मुक्त(ptr, *stream);
        } अन्यथा {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit चालक bool() स्थिर { वापस ptr != nullptr; }

    sycl_reorder_अस्थायी_बफर(स्थिर sycl_reorder_अस्थायी_बफर &)            = मिटाओ;
    sycl_reorder_अस्थायी_बफर & चालक=(स्थिर sycl_reorder_अस्थायी_बफर &) = मिटाओ;

निजी:
    bool host_fallback = false;
};

अटल bool reorder_qw_q4_0(uint8_t * तथ्य_device, स्थिर पूर्णांक ncols, स्थिर पूर्णांक nrows, माप_प्रकार size, माप_प्रकार offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    GGML_ASSERT((size % माप(block_q4_0) == 0));
    GGML_ASSERT((offset % माप(block_q4_0) == 0));
    पूर्णांक offset_blks = offset / माप(block_q4_0);
    स्वतः qs_ptr      = तथ्य_device + offset_blks * QK4_0 / 2;
    स्वतः d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    स्वतः reorder_event = stream->parallel_for(
        size / माप(block_q4_0),
            [=](स्वतः i) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            स्थिर block_q4_0* x = (स्थिर block_q4_0*)tmp_buf;
            स्थिर पूर्णांक ib = i;

            क्रम (पूर्णांक j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_q8_0(uint8_t * तथ्य_device, स्थिर पूर्णांक ncols, स्थिर पूर्णांक nrows, माप_प्रकार size, माप_प्रकार offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    GGML_ASSERT((size % माप(block_q8_0) == 0));
    GGML_ASSERT((offset % माप(block_q8_0) == 0));
    पूर्णांक offset_blks = offset / माप(block_q8_0);
    स्वतः qs_ptr = तथ्य_device + offset_blks * QK8_0;
    स्वतः d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    स्वतः reorder_event = stream->parallel_for(
        size / माप(block_q8_0),
            [=](स्वतः i) [[sycl::reqd_sub_group_आकार(WARP_आकार)]] {
            स्थिर block_q8_0* x = (स्थिर block_q8_0*)tmp_buf;
            स्थिर पूर्णांक ib = i;

            क्रम (पूर्णांक j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
अटल bool reorder_qw_q8_0_moe(uint8_t * तथ्य_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % माप(block_q8_0) == 0);
    स्थिर पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / माप(block_q8_0));
    स्थिर माप_प्रकार योग_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_bytes);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_bytes);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, योग_bytes)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिर पूर्णांक योग_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(योग_blocks, [=](स्वतः gb_) {
        स्थिर पूर्णांक          gb   = gb_;
        स्थिर पूर्णांक          e    = gb / blocks_per_expert;
        स्थिर पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिर block_q8_0 * x    = (स्थिर block_q8_0 *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = तथ्य_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr = base;
        स्वतः * d_ptr  = (sycl::half *) (qs_ptr + (माप_प्रकार) blocks_per_expert * QK8_0);

        क्रम (पूर्णांक j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_q4_k(uint8_t * तथ्य_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_q4_K) == 0);
    GGML_ASSERT(offset % माप(block_q4_K) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_q4_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr     = तथ्य_device;
    स्वतः * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_q4_K * x  = (स्थिर block_q4_K *) tmp_buf;
        स्थिर पूर्णांक          ib = i;

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        क्रम (पूर्णांक j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

// Reorder each expert slice into a self-contained SoA layout.
अटल bool reorder_qw_q4_k_moe(uint8_t * तथ्य_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % माप(block_q4_K) == 0);
    स्थिर पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / माप(block_q4_K));
    स्थिर माप_प्रकार योग_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_bytes);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_bytes);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, योग_bytes)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिर पूर्णांक योग_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(योग_blocks, [=](स्वतः gb_) {
        स्थिर पूर्णांक          gb   = gb_;
        स्थिर पूर्णांक          e    = gb / blocks_per_expert;
        स्थिर पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिर block_q4_K * x    = (स्थिर block_q4_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = तथ्य_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr     = base;
        स्वतः * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * blocks_per_expert);

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        क्रम (पूर्णांक j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
अटल bool reorder_qw_q5_k_moe(uint8_t * तथ्य_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % माप(block_q5_K) == 0);
    स्थिर पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / माप(block_q5_K));
    स्थिर माप_प्रकार योग_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_bytes);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_bytes);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, योग_bytes)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिर पूर्णांक योग_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(योग_blocks, [=](स्वतः gb_) {
        स्थिर पूर्णांक          gb   = gb_;
        स्थिर पूर्णांक          e    = gb / blocks_per_expert;
        स्थिर पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिर block_q5_K * x    = (स्थिर block_q5_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = तथ्य_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * qs_ptr     = base;
        स्वतः * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        स्वतः * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * blocks_per_expert);

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        क्रम (पूर्णांक j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
अटल bool reorder_qw_q6_k_moe(uint8_t * तथ्य_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % माप(block_q6_K) == 0);
    स्थिर पूर्णांक    blocks_per_expert = (पूर्णांक) (expert_bytes / माप(block_q6_K));
    स्थिर माप_प्रकार योग_bytes       = expert_bytes * (माप_प्रकार) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_bytes);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_bytes);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, योग_bytes)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्थिर पूर्णांक योग_blocks = blocks_per_expert * (पूर्णांक) n_expert;
    स्वतः reorder_event = stream->parallel_for(योग_blocks, [=](स्वतः gb_) {
        स्थिर पूर्णांक          gb   = gb_;
        स्थिर पूर्णांक          e    = gb / blocks_per_expert;
        स्थिर पूर्णांक          ib   = gb % blocks_per_expert;
        स्थिर block_q6_K * x    = (स्थिर block_q6_K *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
        uint8_t *          base = तथ्य_device + (माप_प्रकार) e * expert_bytes;

        स्वतः * ql_ptr     = base;
        स्वतः * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        स्वतः * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        स्वतः * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
अटल bool reorder_qw_mxfp4_moe(uint8_t * तथ्य_device, माप_प्रकार expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % माप(block_mxfp4) == 0);
    स्थिर पूर्णांक     blocks_per_expert = (पूर्णांक) (expert_bytes / माप(block_mxfp4));
    स्थिर माप_प्रकार  max_chunk_bytes   = 32u << 20;
    स्थिर int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_अस्थायी_बफर tmp(stream, (माप_प्रकार) chunk_experts * expert_bytes);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (माप_प्रकार) chunk_experts * expert_bytes);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    क्रम (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        स्थिर int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = तथ्य_device + (माप_प्रकार) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, chunk, (माप_प्रकार) n_chunk * expert_bytes)));
        अगर (!g_ggml_sycl_use_async_mem_op) {
            copy_event.रुको();
        }

        स्थिर पूर्णांक योग_blocks = blocks_per_expert * (पूर्णांक) n_chunk;
        स्वतः reorder_event = stream->parallel_for(योग_blocks, [=](स्वतः gb_) {
            स्थिर पूर्णांक           gb   = gb_;
            स्थिर पूर्णांक           e    = gb / blocks_per_expert;
            स्थिर पूर्णांक           ib   = gb % blocks_per_expert;
            स्थिर block_mxfp4 * x    = (स्थिर block_mxfp4 *) (tmp_buf + (माप_प्रकार) e * expert_bytes);
            uint8_t *           base = chunk + (माप_प्रकार) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (माप_प्रकार) blocks_per_expert;

            क्रम (पूर्णांक j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(माप_प्रकार) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        अगर (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    वापस true;
}

अटल bool reorder_qw_q2_k(uint8_t * तथ्य_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_q2_K) == 0);
    GGML_ASSERT(offset % माप(block_q2_K) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_q2_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *        qs_ptr     = तथ्य_device;
    स्वतः *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_q2_K * x  = (स्थिर block_q2_K *) tmp_buf;
        स्थिर पूर्णांक          ib = i;

        क्रम (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_q3_k(uint8_t * तथ्य_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_q3_K) == 0);
    GGML_ASSERT(offset % माप(block_q3_K) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_q3_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *       qs_ptr     = तथ्य_device;
    स्वतः *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    स्वतः *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_q3_K * x  = (स्थिर block_q3_K *) tmp_buf;
        स्थिर पूर्णांक          ib = i;

        क्रम (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        क्रम (पूर्णांक j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_q5_k(uint8_t * तथ्य_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_q5_K) == 0);
    GGML_ASSERT(offset % माप(block_q5_K) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_q5_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr     = तथ्य_device;
    स्वतः * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    स्वतः * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    स्वतः * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_q5_K * x  = (स्थिर block_q5_K *) tmp_buf;
        स्थिर पूर्णांक          ib = i;

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        क्रम (पूर्णांक j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_q6_k(uint8_t * तथ्य_device, माप_प्रकार size, माप_प्रकार offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_q6_K) == 0);
    GGML_ASSERT(offset % माप(block_q6_K) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_q6_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः *       ql_ptr     = तथ्य_device;
    स्वतः *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    स्वतः *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_q6_K * x  = (स्थिर block_q6_K *) tmp_buf;
        स्थिर पूर्णांक          ib = i;

        स्थिर uint8_t * ql              = x[ib].ql;
        स्थिर uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        क्रम (पूर्णांक j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        क्रम (पूर्णांक j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_iq3_s(uint8_t * तथ्य_device, माप_प्रकार size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_iq3_s) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_iq3_s);
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr       = तथ्य_device;
    स्वतः * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    स्वतः * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    स्वतः * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_iq3_s * x = (स्थिर block_iq3_s *) tmp_buf;
        स्थिर पूर्णांक ib = i;

        क्रम (पूर्णांक j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        क्रम (पूर्णांक j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (माप(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        क्रम (पूर्णांक j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[माप(ggml_half) + j] = x[ib].scales[j];
        }
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw_iq3_xxs(uint8_t * तथ्य_device, माप_प्रकार size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % माप(block_iq3_xxs) == 0);

    स्थिर पूर्णांक nblocks = size / माप(block_iq3_xxs);
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    अगर (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        वापस false;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स_नकल(tmp_buf, तथ्य_device, size)));
    अगर (!g_ggml_sycl_use_async_mem_op) {
        copy_event.रुको();
    }

    स्वतः * qs_ptr = तथ्य_device;
    स्वतः * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    स्वतः reorder_event = stream->parallel_for(nblocks, [=](स्वतः i) {
        स्थिर block_iq3_xxs * x = (स्थिर block_iq3_xxs *) tmp_buf;
        स्थिर पूर्णांक ib = i;

        क्रम (पूर्णांक j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    अगर (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    वापस true;
}

अटल bool reorder_qw(स्थिर ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * तथ्य_device = (uint8_t *) src0->तथ्य;
    माप_प्रकार ncols = src0->ne[0];
    माप_प्रकार nrows = src0->ne[1];
    माप_प्रकार size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    अगर (src0->ne[2] > 1) {
        GGML_ASSERT((माप_प्रकार) size == (माप_प्रकार) src0->ne[2] * src0->nb[2]);
        चयन (src0->type) {
            हाल GGML_TYPE_Q8_0:
                वापस reorder_qw_q8_0_moe(तथ्य_device, src0->nb[2], src0->ne[2], stream);
            हाल GGML_TYPE_Q4_K:
                वापस reorder_qw_q4_k_moe(तथ्य_device, src0->nb[2], src0->ne[2], stream);
            हाल GGML_TYPE_Q5_K:
                वापस reorder_qw_q5_k_moe(तथ्य_device, src0->nb[2], src0->ne[2], stream);
            हाल GGML_TYPE_Q6_K:
                वापस reorder_qw_q6_k_moe(तथ्य_device, src0->nb[2], src0->ne[2], stream);
            हाल GGML_TYPE_MXFP4:
                वापस reorder_qw_mxfp4_moe(तथ्य_device, src0->nb[2], src0->ne[2], stream);
            शेष:
                वापस false;
        }
    }

    चयन (src0->type) {
        हाल GGML_TYPE_Q4_0:
            वापस reorder_qw_q4_0(तथ्य_device, ncols, nrows, size, 0, stream);
        हाल GGML_TYPE_Q8_0:
            वापस reorder_qw_q8_0(तथ्य_device, ncols, nrows, size, 0, stream);
        हाल GGML_TYPE_Q2_K:
            वापस reorder_qw_q2_k(तथ्य_device, size, 0, stream);
        हाल GGML_TYPE_Q3_K:
            वापस reorder_qw_q3_k(तथ्य_device, size, 0, stream);
        हाल GGML_TYPE_Q4_K:
            वापस reorder_qw_q4_k(तथ्य_device, size, 0, stream);
        हाल GGML_TYPE_Q5_K:
            वापस reorder_qw_q5_k(तथ्य_device, size, 0, stream);
        हाल GGML_TYPE_Q6_K:
            वापस reorder_qw_q6_k(तथ्य_device, size, 0, stream);
        हाल GGML_TYPE_IQ3_XXS:
            वापस reorder_qw_iq3_xxs(तथ्य_device, size, stream);
        हाल GGML_TYPE_IQ3_S:
            वापस reorder_qw_iq3_s(तथ्य_device, size, stream);
        शेष:
            वापस false;
    }
}

अटल bool should_reorder_tensor(ggml_backend_sycl_context& ctx, स्थिर ggml_tensor * dst) {
    वापस g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

अटल व्योम opt_for_reorder(ggml_backend_sycl_context * ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    अगर (!should_reorder_tensor(*ctx, dst)) {
        वापस;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    अगर (!extra || extra->optimized_feature.reorder) {
        वापस;  // Skip permutations and already reordered tensors
    }

    चयन (mm_algorithm) {
        हाल mul_mat_algo::DMMV:
            अगर (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                वापस;
            }
            अवरोध;
        हाल mul_mat_algo::MMVQ:
            अगर (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                वापस;
            }
            अवरोध;
        हाल mul_mat_algo::MUL_MAT_SYCL:
            अगर (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                वापस;
            }
            अवरोध;
    }

    अगर (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
अटल व्योम opt_for_reorder_id(ggml_backend_sycl_context * ctx, स्थिर ggml_tensor * src0) {
    अगर (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        वापस;
    }
    अगर (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        वापस;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    अगर (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->तथ्य % 16 != 0)) {
        वापस;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    अगर (!extra || extra->optimized_feature.reorder) {
        वापस;
    }
    अगर (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = true;
    }
}


अटल bool can_use_dequantize_mul_mat_vec(स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    स्थिर int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    वापस ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

अटल bool can_use_mul_mat_vec_q(स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    वापस ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_आकार;
}

अटल व्योम ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    अगर (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        वापस;
    }

    स्थिर bool split = ggml_backend_बफर_is_sycl_split(src0->बफर);
    int64_t min_compute_capability = पूर्णांक_उच्च;

    अगर (split) {
        ggml_backend_sycl_split_बफर_type_context * buft_ctx =
            (ggml_backend_sycl_split_बफर_type_context *) src0->बफर->buft->context;
        स्वतः & tensor_split = buft_ctx->tensor_split;
        क्रम (पूर्णांक id = 0; id < ggml_sycl_info().device_गिनती; ++id) {
            // skip devices that are not going to do any work:
            अगर (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_गिनती ? tensor_split[id + 1] : 1.0f)) {
                जारी;
            }

            अगर (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } अन्यथा {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    bool use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#अगर_घोषित GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#पूर्ण_अगर

    bool use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    bool use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#अगर_घोषित SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_आकार);
#पूर्ण_अगर // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    अगर (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        bool use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        अगर (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    अगर (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        अगर (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } अन्यथा {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } अन्यथा अगर (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } अन्यथा अगर (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } अन्यथा अगर (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } अन्यथा अगर (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        अगर (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } अन्यथा {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } अन्यथा अगर (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } अन्यथा {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
अटल bool ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, स्थिर ggml_tensor * wu,
                                             स्थिर ggml_tensor * wg, स्थिर ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    स्थिर स्वतः * extra_u = static_cast<स्थिर ggml_tensor_extra_gpu *>(wu->extra);
    स्थिर स्वतः * extra_g = static_cast<स्थिर ggml_tensor_extra_gpu *>(wg->extra);
    अगर ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        वापस false;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    स्थिर स्वतः plain_type = [](ggml_type t) { वापस t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    अगर (!plain_type(wu->type) || !plain_type(wg->type)) {
        वापस false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    स्थिर int64_t ne00 = wu->ne[0];
    स्थिर int64_t ne11 = act->ne[1];

    स्थिर queue_ptr stream = ctx.stream();
    स्थिर पूर्णांक src1_padded_cols = GGML_PAD((पूर्णांक) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
        (माप_प्रकार) ne11 * src1_padded_cols * माप(block_q8_1) / QK8_1);
    अक्षर * src1_ddq = src1_q8_alloc.लो();

    quantize_row_q8_1_sycl<quantize_q8_1>((स्थिर भग्न *) act->तथ्य, src1_ddq, (पूर्णांक) ne00, (पूर्णांक) ne11,
                                          src1_padded_cols, stream);

    वापस ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->तथ्य, wu->तथ्य,
                                             src1_ddq, (भग्न *) glu->तथ्य, (पूर्णांक) ne00, (पूर्णांक) wu->ne[1],
                                             (पूर्णांक) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(पूर्णांक) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

#अगर_घोषित GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
अटल bool ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, स्थिर ggml_tensor * wu,
                                             स्थिर ggml_tensor * wg, स्थिर ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    स्थिर अक्षर * act_begin = (स्थिर अक्षर *) act->तथ्य;
    स्थिर अक्षर * glu_begin = (स्थिर अक्षर *) glu->तथ्य;
    अगर (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        वापस false;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    स्थिर स्वतः * extra_u = static_cast<स्थिर ggml_tensor_extra_gpu *>(wu->extra);
    स्थिर स्वतः * extra_g = static_cast<स्थिर ggml_tensor_extra_gpu *>(wg->extra);
    अगर (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        वापस false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    वापस ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->तथ्य, wg->तथ्य,
                                                              (स्थिर भग्न *) act->तथ्य, (भग्न *) glu->तथ्य,
                                                              (पूर्णांक) wu->ne[0], (पूर्णांक) wu->ne[1], ctx.stream());
}
#पूर्ण_अगर

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
अटल bool ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    अगर (!ggml_sycl_can_fuse(cgraph, गाँठ_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        वापस false;
    }

    ggml_tensor *       glu  = cgraph->nodes[गाँठ_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    स्थिर ggml_tensor * wu   = up->src[0];
    स्थिर ggml_tensor * wg   = gate->src[0];
    स्थिर ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    अगर (ggml_backend_बफर_is_sycl_split(wu->बफर) || ggml_backend_बफर_is_sycl_split(wg->बफर)) {
        वापस false;
    }

#अगर_घोषित GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    अगर (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        वापस true;
    }
#पूर्ण_अगर

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    अगर (g_ggml_sycl_prioritize_dmmv) {
        वापस false;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    स्थिर bool reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    अगर (!reorder_pair) {
        वापस ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    अगर (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        वापस false;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    स्थिर स्वतः * extra_u = static_cast<स्थिर ggml_tensor_extra_gpu *>(wu->extra);
    स्थिर स्वतः * extra_g = static_cast<स्थिर ggml_tensor_extra_gpu *>(wg->extra);
    अगर (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        वापस false;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    स्थिर int64_t ne00 = wu->ne[0];
    स्थिर int64_t ne11 = act->ne[1];

    स्थिर queue_ptr stream           = ctx.stream();
    स्थिर पूर्णांक       src1_padded_cols = GGML_PAD((पूर्णांक) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
                                             (माप_प्रकार) ne11 * src1_padded_cols * माप(block_q8_1) / QK8_1);
    अक्षर *                     src1_ddq = src1_q8_alloc.लो();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((स्थिर भग्न *) act->तथ्य, src1_ddq, (पूर्णांक) ne00, (पूर्णांक) ne11,
                                                          src1_padded_cols, stream);

    वापस ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->तथ्य, wg->तथ्य, src1_ddq,
                                               (भग्न *) glu->तथ्य, (पूर्णांक) ne00, (पूर्णांक) wu->ne[1], (पूर्णांक) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (पूर्णांक) माप(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(पूर्णांक) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
अटल पूर्णांक ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णांक गाँठ_idx) {
    स्थिर ggml_tensor * गाँठ = cgraph->nodes[गाँठ_idx];
    अगर (ggml_sycl_info().device_गिनती != 1 || गाँठ->type != GGML_TYPE_F32 ||
        गाँठ->src[0]->type != GGML_TYPE_F32 || गाँठ->src[0]->ne[0] >= 1024) {
        वापस 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    पूर्णांक           count = 0;
    पूर्णांक           last  = गाँठ_idx;
    भग्न         eps0;
    स_नकल(&eps0, गाँठ->op_params, माप(भग्न));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    स्वतः overlaps = [](स्थिर ggml_tensor * a, स्थिर ggml_tensor * b) {
        स्थिर अक्षर * ab = (स्थिर अक्षर *) a->तथ्य;
        स्थिर अक्षर * bb = (स्थिर अक्षर *) b->तथ्य;
        वापस ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    क्रम (पूर्णांक j = गाँठ_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        अगर (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;  // not a launch; cannot break a run of adjacent norms
        }
        अगर (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, गाँठ) ||
            !ggml_are_same_shape(nj->src[0], गाँठ->src[0])) {
            अवरोध;  // any other launch ends the run
        }
        bool same_nb = true;
        क्रम (पूर्णांक d = 0; d < GGML_MAX_DIMS; ++d) {
            अगर (nj->nb[d] != गाँठ->nb[d] || nj->src[0]->nb[d] != गाँठ->src[0]->nb[d]) {
                same_nb = false;
                अवरोध;
            }
        }
        अगर (!same_nb) {
            अवरोध;  // one nb[] stride set is shared by the whole batch
        }
        भग्न epsj;
        स_नकल(&epsj, nj->op_params, माप(भग्न));
        अगर (epsj != eps0) {
            अवरोध;  // eps mismatch ends the run
        }
        bool indep = true;
        क्रम (पूर्णांक k = 0; k < count; ++k) {
            अगर (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = false;
                अवरोध;
            }
        }
        अगर (!indep) {
            अवरोध;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    अगर (count < 2) {
        वापस 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    वापस last - गाँठ_idx;
}


__dpct_inline__ अटल व्योम k_copy_src1_to_contiguous(
    स्थिर अक्षर *__restrict__ src1_original, अक्षर *__restrict__ src1_contiguous,
    स्थिर mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, माप_प्रकार nb11, माप_प्रकार nb12,
    स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर int32_t src1_row = item_ct1.get_group(2);

    स्थिर int32_t iid1 = row_mapping[src1_row].i2;
    स्थिर int32_t id   = row_mapping[src1_row].i1;

    स्थिर int64_t i11 = id % ne11;
    स्थिर int64_t i12 = iid1;

    स्थिर भग्न * src1_row_original = (स्थिर भग्न *)(src1_original + i11*nb11 + i12*nb12);
    भग्न * src1_row_contiguous = (भग्न *)(src1_contiguous + src1_row*nb11);

#आशय unroll
    क्रम (पूर्णांक i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ अटल व्योम k_copy_dst_from_contiguous(
    अक्षर *__restrict__ dst_original, स्थिर अक्षर *__restrict__ dst_contiguous,
    स्थिर mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, माप_प्रकार nb1,
    माप_प्रकार nb2, स्थिर sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    स्थिर int32_t i1 = row_mapping[i].i1;
    स्थिर int32_t i2 = row_mapping[i].i2;

    स्थिर भग्न * dst_row_contiguous = (स्थिर भग्न *)(dst_contiguous + i*nb1);
    भग्न * dst_row_original = (भग्न *)(dst_original + i1*nb1 + i2*nb2);

#आशय unroll
    क्रम (पूर्णांक j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
अटल bool ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0,
    स्थिर ggml_tensor * src1, स्थिर ggml_tensor * ids, ggml_tensor * dst)
{
    स्थिर int64_t ne10 = src1->ne[0];
    स्थिर int64_t ne11 = src1->ne[1];
    स्थिर int64_t ne12 = src1->ne[2];
    अगर (ne12 != 1) वापस false;
    अगर (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) वापस false;
    अगर (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) वापस false;
    अगर (!ggml_is_contiguous(src1)) वापस false;

    स्थिर int64_t n_ids_per_group = ids->ne[0];
    अगर (ids->ne[1] != 1) वापस false;
    अगर (ne11 != 1 && ne11 != n_ids_per_group) वापस false;

    स्थिर queue_ptr stream           = ctx.stream();
    स्थिर पूर्णांक       src1_padded_cols = GGML_PAD((पूर्णांक) ne10, MATRIX_ROW_PADDING);
    स्थिर पूर्णांक       n_experts_used   = (पूर्णांक) n_ids_per_group;
    स्थिर पूर्णांक       nrows            = (पूर्णांक) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    स्थिर ggml_tensor_extra_gpu * src0_extra =
        static_cast<स्थिर ggml_tensor_extra_gpu *>(src0->extra);
    स्थिर bool use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<अक्षर> src1_q8_alloc(ctx.pool(),
        (माप_प्रकार) ne11 * src1_padded_cols * माप(block_q8_1) / QK8_1);
    अक्षर * src1_ddq = src1_q8_alloc.लो();
    अगर (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (स्थिर भग्न *) src1->तथ्य, src1_ddq, (पूर्णांक) ne10, (पूर्णांक) ne11,
            src1_padded_cols, stream);
    } अन्यथा {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (स्थिर भग्न *) src1->तथ्य, src1_ddq, (पूर्णांक) ne10, (पूर्णांक) ne11,
            src1_padded_cols, stream);
    }

    स्थिर माप_प्रकार bytes_per_qrow = (माप_प्रकार) src1_padded_cols * माप(block_q8_1) / QK8_1;
    स्थिर माप_प्रकार src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    अगर (use_reorder) {
        वापस ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->तथ्य, src1_ddq, (स्थिर int32_t *) ids->तथ्य,
            (भग्न *) dst->तथ्य, (पूर्णांक) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    वापस ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->तथ्य, src1_ddq, (स्थिर int32_t *) ids->तथ्य,
        (भग्न *) dst->तथ्य, (पूर्णांक) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
अटल व्योम mmid_counting_sort_rows(
        स्थिर ggml_tensor * ids, स्थिर अक्षर * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    क्रम (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        क्रम (int64_t id = 0; id < n_ids; id++) {
            स्थिर int32_t row_id_i = *(स्थिर int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    क्रम (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_अग्र = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    क्रम (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        क्रम (int64_t id = 0; id < n_ids; id++) {
            स्थिर int32_t row_id_i = *(स्थिर int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            स्थिर int64_t routed_row = expert_row_अग्र[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

अटल व्योम ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) प्रयास {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    स्थिर ggml_tensor *src0 = dst->src[0];
    स्थिर ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_split(src0->बफर) && "mul_mat_id does not support split buffers");

    स्थिर ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_द्विआधारी_OP_LOCALS

    स्थिर queue_ptr stream = ctx.stream();

    स्थिर int64_t n_as = ne02;
    स्थिर int64_t n_ids = ids->ne[0];

    अगर (ne12 == 1) {
        अगर (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            वापस;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    अगर (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<अक्षर> ids_host(ggml_nbytes(ids));
    स्थिर अक्षर * ids_dev = (स्थिर अक्षर *) ids->तथ्य;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->स_नकल(ids_host.तथ्य(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->रुको()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    अक्षर *src0_original = (अक्षर *)src0->तथ्य;
    अक्षर *src1_original = (अक्षर *)src1->तथ्य;
    अक्षर *dst_original = (अक्षर *)dst->तथ्य;

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
    अगर (ne12 == 1) {
        क्रम (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            क्रम (int64_t id = 0; id < n_ids; id++) {
                स्थिर int32_t i02 = *(स्थिर int32_t *) (ids_host.तथ्य() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                स्थिर int64_t i11 = id % ne11;
                स्थिर int64_t i12 = iid1;

                स्थिर int64_t i1 = id;
                स्थिर int64_t i2 = i12;

            src0_row.तथ्य = src0_original + i02*nb02;
            src1_row.तथ्य = src1_original + i11*nb11 + i12*nb12;
            dst_row.तथ्य = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } अन्यथा {
        स्थिर int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<अक्षर> src1_contiguous(ctx.pool(), माप(भग्न)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<अक्षर>  dst_contiguous(ctx.pool(), माप(भग्न)*n_routed_rows*ne0);

        src1_row.तथ्य = src1_contiguous.लो();
        dst_row.तथ्य  =  dst_contiguous.लो();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

        mmid_counting_sort_rows(ids, ids_host.तथ्य(), n_ids, n_as, n_routed_rows,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_rows);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->स_नकल(dev_row_mapping.लो(), routed_row_src.तथ्य(), n_routed_rows*माप(mmid_row_mapping))));

        स्थिर अचिन्हित पूर्णांक max_work_group_आकार = ggml_sycl_info().max_work_group_sizes[ctx.device];
        निश्चित(max_work_group_आकार % (WARP_आकार * WARP_आकार) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((अचिन्हित पूर्णांक)ne10, max_work_group_आकार));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                अक्षर *__restrict src1_contiguous_get =
                    src1_contiguous.लो();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.लो();

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

        bool grouped = false;
        अगर (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == माप(भग्न)*ne10 && nb1 == माप(भग्न)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            अगर (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            स्थिर bool src0_reordered =
                src0->extra && ((स्थिर ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (स्थिर भग्न *) src1_contiguous.लो(), ggml_sycl_src1_prec(dst),
                                                     (भग्न *) dst_contiguous.लो(),
                                                     expert_row_offsets.तथ्य(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        क्रम (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            स्थिर int64_t अंक_src1_rows = expert_row_counts[i02];

            अगर (अंक_src1_rows == 0) {
                जारी;
            }

            स्थिर int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.तथ्य = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == माप(भग्न)*ne10);
            GGML_ASSERT(nb1 == माप(भग्न)*ne0);
            src1_row.तथ्य = src1_contiguous.लो() + expert_row_offset*nb11;
            src1_row.ne[1] = अंक_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = अंक_src1_rows*nb11;
            src1_row.nb[3] = अंक_src1_rows*nb11;

            dst_row.तथ्य = dst_contiguous.लो() + expert_row_offset*nb1;
            dst_row.ne[1] = अंक_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = अंक_src1_rows*nb1;
            dst_row.nb[3] = अंक_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((अचिन्हित पूर्णांक)ne0, max_work_group_आकार));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                स्थिर अक्षर *__restrict dst_contiguous_get =
                    dst_contiguous.लो();
                स्थिर mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.लो();

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
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

अटल व्योम ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

अटल व्योम ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

अटल व्योम ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

अटल व्योम ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

अटल व्योम ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

अटल व्योम ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

अटल व्योम ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

अटल व्योम ggml_sycl_योग(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_योग(ctx, dst);
}

अटल व्योम ggml_sycl_योग_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_योग_rows(ctx, dst);
}

अटल व्योम ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

अटल व्योम ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

अटल व्योम ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


अटल व्योम ggml_sycl_set_main_device(स्थिर पूर्णांक main_device) प्रयास {
    अगर (dpct::get_current_device_id() == static_cast<अचिन्हित पूर्णांक> (main_device)) {
        वापस;
    }
    check_allow_gpu_सूचकांक(main_device);
    dpct::select_device(main_device);

    अगर (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल bool ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, काष्ठा ggml_tensor * dst) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    अगर (!g_sycl_loaded) वापस false;

    अगर (dst->src[0] != nullptr && ggml_backend_बफर_is_sycl_split(dst->src[0]->बफर)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    चयन (dst->op) {
        हाल GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            अवरोध;
        हाल GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            अवरोध;
        हाल GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            अवरोध;
        हाल GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            अवरोध;
        हाल GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            अवरोध;
        हाल GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            अवरोध;
        हाल GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            अवरोध;
        हाल GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            अवरोध;
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            अवरोध;
        हाल GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            अवरोध;
        हाल GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            अवरोध;
        हाल GGML_OP_गिनती_EQUAL:
            ggml_sycl_गिनती_equal(ctx, dst);
            अवरोध;
        हाल GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            अवरोध;
        हाल GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            अवरोध;
        हाल GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            अवरोध;
        हाल GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            अवरोध;
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(dst)) {
                हाल GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    अवरोध;
                हाल GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    अवरोध;
                शेष:
                    वापस false;
            }
            अवरोध;
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(dst)) {
                हाल GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    अवरोध;
                हाल GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    अवरोध;
                शेष:
                    वापस false;
            }
            अवरोध;
        हाल GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            अवरोध;
        हाल GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            अवरोध;
        हाल GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            अवरोध;
        हाल GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            अवरोध;
        हाल GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            अवरोध;
        हाल GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            अवरोध;
        हाल GGML_OP_MUL_MAT:
            अगर (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                वापस false;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            अवरोध;
        हाल GGML_OP_MUL_MAT_ID:
            अगर (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                वापस false;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            अवरोध;
        हाल GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            अवरोध;
        हाल GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            अवरोध;
        हाल GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            अवरोध;
        हाल GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            अवरोध;
        हाल GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            अवरोध;
        हाल GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            अवरोध;
        हाल GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            अवरोध;
        हाल GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            अवरोध;
        हाल GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            अवरोध;
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            अवरोध;
        हाल GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            अवरोध;
        हाल GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            अवरोध;
        हाल GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            अवरोध;
        हाल GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            अवरोध;
        हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            अवरोध;
        हाल GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            अवरोध;
        हाल GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            अवरोध;
        हाल GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            अवरोध;
        हाल GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            अवरोध;
        हाल GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            अवरोध;
        हाल GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            अवरोध;
        हाल GGML_OP_योग:
            ggml_sycl_योग(ctx, dst);
            अवरोध;
        हाल GGML_OP_योग_ROWS:
            ggml_sycl_योग_rows(ctx, dst);
            अवरोध;
        हाल GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            अवरोध;
        हाल GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            अवरोध;
        हाल GGML_OP_शीर्ष_K:
            ggml_sycl_op_शीर्ष_k(ctx, dst);
            अवरोध;
        हाल GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            अवरोध;
        हाल GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            अवरोध;
        हाल GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            अवरोध;
        हाल GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            अवरोध;
        हाल GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            अवरोध;
        हाल GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            अवरोध;
        हाल GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            अवरोध;
        हाल GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            अवरोध;
        हाल GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            अवरोध;
        हाल GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            अवरोध;
        हाल GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            अवरोध;
        हाल GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            अवरोध;
        हाल GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            अवरोध;
        हाल GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            अवरोध;
        हाल GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            अवरोध;
        हाल GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            अवरोध;
        शेष:
            वापस false;
    }

    वापस true;
} पकड़ो (sycl::exception & e) {
    std::अ_त्रुटि << e.what() << "Exception caught at file:" << __खाता__ << ", line:" << __LINE__ << std::प_पूर्ण;
    std::अ_त्रुटि << "Error OP "<<ggml_op_name(dst->op)<< std::प_पूर्ण;
    std::निकास(1);
}

GGML_API व्योम ggml_backend_sycl_get_device_description(पूर्णांक device, अक्षर *description,
                                      माप_प्रकार description_आकार) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_आकार, "%s", prop.get_name());
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल bool sycl_get_mem_info(पूर्णांक device, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_परिणाम_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#अन्यथा
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#पूर्ण_अगर
    bool res = get_memory_आकार(dpct::dev_mgr::instance().get_device(device),
        *मुक्त, *योग, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *योग, *मुक्त);
    वापस res;
}

व्योम ggml_backend_sycl_get_device_memory(पूर्णांक device, माप_प्रकार * मुक्त, माप_प्रकार * योग) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    अगर (!sycl_get_mem_info(device, मुक्त, योग)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *मुक्त, *योग);
} पकड़ो (स्थिर sycl::exception & exc) {
    std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__ << ", line:" << __LINE__ << std::प_पूर्ण;
    std::निकास(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

अटल स्थिर अक्षर * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    वापस sycl_ctx->name.c_str();
}

अटल व्योम ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    मिटाओ sycl_ctx;
    मिटाओ backend;
}

अटल व्योम ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               स्थिर व्योम *तथ्य, माप_प्रकार offset,
                                               माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) && "unsupported buffer type");
    स्थिर queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->स_नकल((अक्षर *)tensor->तथ्य + offset, तथ्य, size)));
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               स्थिर ggml_tensor *tensor,
                                               व्योम *तथ्य, माप_प्रकार offset,
                                               माप_प्रकार size) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) && "unsupported buffer type");
    स्थिर queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->स_नकल(
        तथ्य, (स्थिर अक्षर *)tensor->तथ्य + offset, size)));
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल bool ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               स्थिर ggml_tensor *src,
                                               ggml_tensor *dst) प्रयास {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    bool is_cpy_supported                = dst->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) &&
                            ggml_backend_बफर_is_sycl(src->बफर);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    अगर (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        स्थिर queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->स_नकल(
            dst->तथ्य, src->तथ्य, ggml_nbytes(dst))));
        वापस true;
    }

    वापस false;
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल व्योम ggml_backend_sycl_synchronize(ggml_backend_t backend) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    स्थिर queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->रुको()));

    GGML_UNUSED(backend);
}
पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल bool ggml_sycl_is_view_or_noop(स्थिर ggml_tensor * t) {
    वापस ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
अटल पूर्णांक ggml_sycl_try_gdn_cache_fusion(स्थिर ggml_cgraph * cgraph, पूर्णांक गाँठ_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    अगर (!g_ggml_sycl_enable_fusion) {
        वापस 0;
    }

    स्थिर ggml_tensor * gdn = cgraph->nodes[गाँठ_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    अगर (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        वापस 0;
    }

    स्थिर ggml_tensor * src_v     = gdn->src[2];
    स्थिर int64_t       S_v       = src_v->ne[0];
    स्थिर int64_t       H         = src_v->ne[1];
    स्थिर int64_t       n_tokens  = src_v->ne[2];
    स्थिर int64_t       n_seqs    = src_v->ne[3];
    स्थिर int64_t       D         = S_v * S_v * H;
    स्थिर int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    स्थिर int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    स्थिर माप_प्रकार पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    स्थिर ggml_tensor * cpy  = nullptr;
    पूर्णांक                 skip = 0;
    क्रम (पूर्णांक j = गाँठ_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        स्थिर ggml_tensor * n = cgraph->nodes[j];
        अगर (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }
        अगर (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            वापस 0;
        }
        cpy  = n;
        skip = j - गाँठ_idx;
    }
    अगर (cpy == nullptr) {
        वापस 0;
    }

    स्थिर ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    स्थिर ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    अगर (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != पुच्छ_off ||
        !ggml_is_contiguous(src)) {
        वापस 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    स्थिर std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    अगर (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->तथ्य == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) ||
        dst->nb[1] != (माप_प्रकार) ggml_row_आकार(GGML_TYPE_F32, D)) {
        वापस 0;
    }

    fused_state_cpy.तथ्य        = (भग्न *) dst->तथ्य; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / माप(भग्न)) : 0;
    वापस skip;
}

अटल व्योम ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * गाँठ = cgraph->nodes[i];
        अगर (ggml_sycl_is_view_or_noop(गाँठ)) {
            जारी;
        }
        अगर ((गाँठ->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            जारी;
        }

        स्थिर पूर्णांक nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        अगर (nodes_to_skip != 0) {
            i += nodes_to_skip;
            जारी;
        }
#अगर_अघोषित न_संशोधन
        निश्चित(गाँठ->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device));
        क्रम (पूर्णांक j = 0; j < GGML_MAX_SRC; j++) {
            अगर (गाँठ->src[j] != nullptr) {
                निश्चित(गाँठ->src[j]->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device));
            }
        }
#पूर्ण_अगर
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        अगर (गाँठ->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            स्थिर पूर्णांक gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            अगर (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, गाँठ, fused_state_cpy);
                i += gdn_nodes_to_skip;
                जारी;
            }
        }
        अगर (गाँठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, गाँठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            जारी;
        }
        अगर (गाँठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;
            जारी;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        अगर (गाँठ->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;
            जारी;
        }
        अगर (गाँठ->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;
            जारी;
        }
        अगर (गाँठ->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(गाँठ) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1]);
            i++;
            जारी;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        अगर (गाँठ->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            जारी;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        अगर (गाँठ->op == GGML_OP_L2_NORM) {
            स्थिर पूर्णांक l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            अगर (l2_batch_skip > 0) {
                i += l2_batch_skip;
                जारी;
            }
        }

        अगर (गाँठ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, गाँठ, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            जारी;
        }

        अगर (गाँठ->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, गाँठ, nullptr, cgraph->nodes[i + 1]);
            i++;
            जारी;
        }

        अगर (गाँठ->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            जारी;
        }

        bool ok = ggml_sycl_compute_forward(*sycl_ctx, गाँठ);
        अगर (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, गाँठ->name, ggml_op_name(गाँठ->op));
        }
        GGML_ASSERT(ok);
    }
}

#अगर_घोषित GGML_SYCL_GRAPH
अटल bool check_graph_compatibility(ggml_cgraph * cgraph) {
    अगर (ggml_sycl_info().device_गिनती > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        वापस false;
    }

    क्रम (पूर्णांक i = 0; i < cgraph->n_nodes; i++) {
        स्थिर ggml_op गाँठ_op = cgraph->nodes[i]->op;
        चयन (गाँठ_op) {
            शेष:
                अवरोध;
            हाल GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            हाल GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(गाँठ_op));
                वापस false;
            हाल GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                अगर (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(गाँठ_op));
                    वापस false;
                }
        }
    }
    वापस true;
}
#पूर्ण_अगर

अटल ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    स्वतः * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#अगर_घोषित GGML_SYCL_GRAPH
    bool use_sycl_graph = false;
    अगर (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    अगर (use_sycl_graph) {
        स्थिर bool graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        अगर (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            वापस GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_बफर_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        स्थिर bool graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        अगर (!sycl_ctx->exec_graph || !graph_update_support) {
            स्वतः exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } अन्यथा {
            प्रयास {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } पकड़ो (sycl::exception स्थिर & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                स्वतः exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } अन्यथा
#पूर्ण_अगर
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    वापस GGML_STATUS_SUCCESS;
}

अटल व्योम ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
प्रयास
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    स्थिर queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
पकड़ो (sycl::exception स्थिर &exc)
{
    std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
              << ", line:" << __LINE__ << std::प_पूर्ण;
    std::निकास(1);
}

अटल व्योम ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) प्रयास {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    अगर (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->रुको()));
    } अन्यथा
        GGML_ABORT("fatal error");
} पकड़ो (sycl::exception स्थिर& exc) {
    std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
              << ", line:" << __LINE__ << std::प_पूर्ण;
    std::निकास(1);
}

अटल ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ शून्य,
    /* .get_tensor_2d_async     = */ शून्य,
    /* .cpy_tensor_async        = */ शून्य, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ शून्य,
    /* .graph_plan_free         = */ शून्य,
    /* .graph_plan_update       = */ शून्य,
    /* .graph_plan_compute      = */ शून्य,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ शून्य,
};

अटल ggml_guid_t ggml_backend_sycl_guid() {
    अटल ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    वापस &guid;
}

bool ggml_backend_is_sycl(ggml_backend_t backend) {
    वापस backend != शून्य && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

पूर्णांक ggml_backend_sycl_get_device_गिनती(व्योम) {
    वापस ggml_sycl_info().device_गिनती;
}


// backend device

अटल स्थिर अक्षर * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस ctx->name.c_str();
}

अटल स्थिर अक्षर * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस ctx->description.c_str();
}

अटल व्योम ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, माप_प्रकार * मुक्त, माप_प्रकार * योग) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    अगर (!sycl_get_mem_info(ctx->device, मुक्त, योग)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *योग, *मुक्त);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *मुक्त, *योग);
}

अटल क्रमागत ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    स्थिर sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    अगर (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        वापस GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    वापस GGML_BACKEND_DEVICE_TYPE_GPU;
}

अटल व्योम ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_योग);

    bool host_बफर = दो_पर्या("GGML_SYCL_NO_PINNED") == nullptr;
#अगर_घोषित GGML_SYCL_NO_PEER_COPY
    bool events = false;
#अन्यथा
    bool events = true;
#पूर्ण_अगर

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_बफर,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ true,
    };
}

अटल ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, स्थिर अक्षर * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस ggml_backend_sycl_init(ctx->device);
}

अटल ggml_backend_बफर_type_t ggml_backend_sycl_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस ggml_backend_sycl_बफर_type(ctx->device);
}

अटल ggml_backend_बफर_type_t ggml_backend_sycl_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    वापस ggml_backend_sycl_host_बफर_type_for_device(ctx->device);
}

अटल ggml_backend_बफर_t ggml_backend_sycl_device_बफर_from_host_ptr(ggml_backend_dev_t dev, व्योम * ptr, माप_प्रकार size, माप_प्रकार max_tensor_आकार) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_आकार);
    वापस nullptr;
}

अटल bool do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    पूर्णांक device = sycl_ctx->device;
    चयन (op->op) {
        हाल GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                अगर (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    वापस true;
                }
                वापस false;
            }
        हाल GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            वापस (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_आकार(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_आकार(op->src[0]->type));
        हाल GGML_OP_CONV_2D_DW:
        हाल GGML_OP_CONV_TRANSPOSE_2D:
            वापस true;
        हाल GGML_OP_UNARY:
            चयन (ggml_get_unary_op(op)) {
                हाल GGML_UNARY_OP_SGN:
                हाल GGML_UNARY_OP_ABS:
                हाल GGML_UNARY_OP_NEG:
                हाल GGML_UNARY_OP_STEP:
                हाल GGML_UNARY_OP_RELU:
                हाल GGML_UNARY_OP_HARDSIGMOID:
                हाल GGML_UNARY_OP_TANH:
                हाल GGML_UNARY_OP_GELU:
                हाल GGML_UNARY_OP_SILU:
                हाल GGML_UNARY_OP_SIGMOID:
                हाल GGML_UNARY_OP_HARDSWISH:
                हाल GGML_UNARY_OP_GELU_QUICK:
                हाल GGML_UNARY_OP_GELU_ERF:
                हाल GGML_UNARY_OP_EXP:
                हाल GGML_UNARY_OP_EXPM1:
                हाल GGML_UNARY_OP_SOFTPLUS:
                हाल GGML_UNARY_OP_ELU:
                हाल GGML_UNARY_OP_XIELU:
                हाल GGML_UNARY_OP_CEIL:
                    वापस true;
                हाल GGML_UNARY_OP_FLOOR:
                हाल GGML_UNARY_OP_ROUND:
                हाल GGML_UNARY_OP_TRUNC:
                    वापस true;
                शेष:
                    वापस false;
            }
        हाल GGML_OP_GLU:
            चयन (ggml_get_glu_op(op)) {
                हाल GGML_GLU_OP_REGLU:
                हाल GGML_GLU_OP_GEGLU:
                हाल GGML_GLU_OP_SWIGLU:
                हाल GGML_GLU_OP_SWIGLU_OAI:
                हाल GGML_GLU_OP_GEGLU_ERF:
                हाल GGML_GLU_OP_GEGLU_QUICK:
                हाल GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    वापस ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                शेष:
                    वापस false;
            }
        हाल GGML_OP_MUL_MAT:
        हाल GGML_OP_MUL_MAT_ID:
            {
                काष्ठा ggml_tensor * a = op->src[0];
                काष्ठा ggml_tensor * b = op->src[1];

                अगर (a->ne[3] != b->ne[3]) {
                    वापस false;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                अगर (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  वापस false;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                अगर (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    वापस false;
                }

                अगर (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    वापस false;
                }

                वापस true;
            }
        हाल GGML_OP_OUT_PROD:
            वापस op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        हाल GGML_OP_GET_ROWS:
            {
                चयन (op->src[0]->type) {
                    हाल GGML_TYPE_I32:
                    हाल GGML_TYPE_F16:
                    हाल GGML_TYPE_BF16:
                    हाल GGML_TYPE_F32:
                    हाल GGML_TYPE_Q1_0:
                    हाल GGML_TYPE_MXFP4:
                    हाल GGML_TYPE_NVFP4:
                    हाल GGML_TYPE_IQ2_XXS:
                    हाल GGML_TYPE_IQ2_XS:
                    हाल GGML_TYPE_IQ2_S:
                    हाल GGML_TYPE_IQ3_XXS:
                    हाल GGML_TYPE_IQ1_S:
                    हाल GGML_TYPE_IQ1_M:
                    हाल GGML_TYPE_IQ3_S:
                    हाल GGML_TYPE_IQ4_NL:
                    हाल GGML_TYPE_IQ4_XS:
                    हाल GGML_TYPE_Q2_K:
                    हाल GGML_TYPE_Q3_K:
                    हाल GGML_TYPE_Q4_0:
                    हाल GGML_TYPE_Q4_1:
                    हाल GGML_TYPE_Q4_K:
                    हाल GGML_TYPE_Q5_0:
                    हाल GGML_TYPE_Q5_1:
                    हाल GGML_TYPE_Q5_K:
                    हाल GGML_TYPE_Q6_K:
                    हाल GGML_TYPE_Q8_0:
                        वापस true;
                    शेष:
                        वापस false;
                }
            }
        हाल GGML_OP_GET_ROWS_BACK:
            // return true;
             वापस op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         हाल GGML_OP_SET:
               वापस (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        हाल GGML_OP_SET_ROWS:
            {
                अगर (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    वापस false;
                }
                स्वतः res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                वापस res;
            }
        हाल GGML_OP_DSV4_HC_PRE:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        हाल GGML_OP_DSV4_HC_COMB:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        हाल GGML_OP_DSV4_HC_POST:
            वापस op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        हाल GGML_OP_LIGHTNING_INDEXER:
            वापस op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_आकार * 8;
        हाल GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                अगर (src0_type == GGML_TYPE_F16) {
                    अगर (src1_type == GGML_TYPE_Q2_K ||
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
                        वापस false;
                    }
                }

                अगर (src0_type == GGML_TYPE_BF16) {
                    अगर (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        वापस false;
                    }
                }

                अगर (src0_type == GGML_TYPE_F32) {
                    अगर (src1_type == GGML_TYPE_Q2_K ||
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
                        वापस false;
                    }
                }

                अगर (src1_type == GGML_TYPE_F32) {
                    अगर (src0_type == GGML_TYPE_Q1_0 ||
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
                        वापस false;
                    }
                }

                अगर (src0_type == src1_type) {
                    अगर (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        वापस false;
                    }
                }

                अगर (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    वापस false;
                }

                वापस true;
            }
        हाल GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                वापस src0_type == GGML_TYPE_F32;
            }
        हाल GGML_OP_CONCAT:
        हाल GGML_OP_DUP:
        हाल GGML_OP_ARGMAX:
        हाल GGML_OP_NONE:
        हाल GGML_OP_RESHAPE:
        हाल GGML_OP_VIEW:
        हाल GGML_OP_PERMUTE:
        हाल GGML_OP_TRANSPOSE:
        हाल GGML_OP_ADD:
        हाल GGML_OP_ADD1:
        हाल GGML_OP_ADD_ID:
        हाल GGML_OP_SUB:
        हाल GGML_OP_गिनती_EQUAL:
        हाल GGML_OP_MUL:
        हाल GGML_OP_DIV:
        हाल GGML_OP_REPEAT:
            वापस true;
        हाल GGML_OP_PAD_REFLECT_1D:
            वापस ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_SQR:
        हाल GGML_OP_SQRT:
        हाल GGML_OP_SIN:
        हाल GGML_OP_COS:
        हाल GGML_OP_CLAMP:
        हाल GGML_OP_LOG:
        हाल GGML_OP_NORM:
        हाल GGML_OP_L2_NORM:
        हाल GGML_OP_GROUP_NORM:
        हाल GGML_OP_RMS_NORM:
            वापस true;
        हाल GGML_OP_RMS_NORM_BACK:
            वापस ggml_is_contiguous(op->src[0]);
        हाल GGML_OP_SCALE:
            // the scale kernel is f32 only
            वापस op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        हाल GGML_OP_CONT:
            वापस true;
        हाल GGML_OP_TRI:
            {
                स्थिर ggml_tensor * src0 = op->src[0];
                वापस src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        हाल GGML_OP_DIAG_MASK_INF:
            वापस true;
        हाल GGML_OP_SOFT_MAX:
            वापस true;
        हाल GGML_OP_SOFT_MAX_BACK: {
            भग्न max_bias = 0.0f;
            स_नकल(&max_bias, (स्थिर भग्न *) op->op_params + 1, माप(भग्न));
            वापस max_bias == 0.0f;
        }
        हाल GGML_OP_ROPE:
        हाल GGML_OP_ROPE_BACK:
        हाल GGML_OP_IM2COL:
        हाल GGML_OP_IM2COL_3D:
        हाल GGML_OP_UPSCALE:
            वापस true;
        हाल GGML_OP_COL2IM_1D:
            वापस ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#अगर_घोषित GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#पूर्ण_अगर
                   ) &&
                   op->src[0]->type == op->type;
        हाल GGML_OP_CONV_3D: {
            स्थिर int32_t * opts = (स्थिर int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            स्थिर int64_t knl_n_योग = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            स्थिर int64_t patch_योग = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            वापस op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_योग > 0 && patch_योग > 0;
        }
        हाल GGML_OP_योग:
        हाल GGML_OP_योग_ROWS:
        हाल GGML_OP_MEAN:
            वापस ggml_is_contiguous(op->src[0]);
        हाल GGML_OP_ARGSORT:
            वापस true;
        हाल GGML_OP_शीर्ष_K: {
            स्थिर ggml_tensor * src0 = op->src[0];
            स्थिर पूर्णांक k = op->ne[0];
            वापस src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        हाल GGML_OP_POOL_2D:
        हाल GGML_OP_POOL_1D:
        हाल GGML_OP_ACC:
            वापस true;
        हाल GGML_OP_PAD:
            अगर (ggml_get_op_params_i32(op, 8) != 0) {
                वापस false;
            }
            वापस true;
        हाल GGML_OP_LEAKY_RELU:
        हाल GGML_OP_TIMESTEP_EMBEDDING:
        हाल GGML_OP_RWKV_WKV6:
        हाल GGML_OP_RWKV_WKV7:
        हाल GGML_OP_GATED_LINEAR_ATTN:
        हाल GGML_OP_GATED_DELTA_NET:
        हाल GGML_OP_OPT_STEP_ADAMW:
        हाल GGML_OP_OPT_STEP_SGD:
            वापस true;
        हाल GGML_OP_SSM_CONV:
            वापस op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        हाल GGML_OP_ROLL:
            वापस op->type == GGML_TYPE_F32;
        हाल GGML_OP_ARANGE:
            वापस op->type == GGML_TYPE_F32;
        हाल GGML_OP_SSM_SCAN:
            अगर (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                वापस (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_आकार == 0;
            } अन्यथा {
                // TODO Mamba-1 not yet ported to SYCL
                वापस false;
            }
        हाल GGML_OP_FILL:
        हाल GGML_OP_CUMSUM:
        हाल GGML_OP_DIAG:
        हाल GGML_OP_CROSS_ENTROPY_LOSS:
        हाल GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            वापस true;
        हाल GGML_OP_SOLVE_TRI:
            वापस op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        हाल GGML_OP_FLASH_ATTN_EXT:
            वापस ggml_sycl_flash_attn_ext_supported(device, op);
        शेष:
            वापस false;
    }

    GGML_UNUSED(dev);
}

अटल bool ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    bool res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    वापस res;
}

अटल bool ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    अगर (buft->iface.get_name != ggml_backend_sycl_बफर_type_get_name) {
        वापस false;
    }
    ggml_backend_sycl_बफर_type_context * buft_ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस buft_ctx->device == sycl_ctx->device;
}

अटल int64_t get_op_batch_आकार(स्थिर ggml_tensor * op) {
    चयन (op->op) {
        हाल GGML_OP_GET_ROWS:
            वापस 0;
        हाल GGML_OP_MUL_MAT:
            वापस op->ne[1];
        हाल GGML_OP_MUL_MAT_ID:
        हाल GGML_OP_ROPE:
            वापस op->ne[2];
        शेष:
            वापस ggml_nrows(op);
    }
}

अटल bool ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, स्थिर ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    वापस get_op_batch_आकार(op) >= sycl_ctx->op_offload_min_batch_आकार;
}

अटल ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#अगर_घोषित GGML_SYCL_NO_PEER_COPY
    वापस nullptr;
#अन्यथा
  sycl::event *event_ptr = नया sycl::event();

  वापस नया ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#पूर्ण_अगर
}

अटल व्योम ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) प्रयास {
  GGML_UNUSED(dev);
  अगर (event == nullptr) {
    वापस;
  }

  अगर (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    मिटाओ sycl_event;
    event->context = nullptr;
  }

  मिटाओ event;
} पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}


अटल व्योम ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) प्रयास {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->रुको()));
} पकड़ो (sycl::exception स्थिर &exc) {
  std::अ_त्रुटि << exc.what() << "Exception caught at file:" << __खाता__
            << ", line:" << __LINE__ << std::प_पूर्ण;
  std::निकास(1);
}

अटल स्थिर ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_memory,
    /* .get_type                = */ ggml_backend_sycl_device_get_type,
    /* .get_props               = */ ggml_backend_sycl_device_get_props,
    /* .init_backend            = */ ggml_backend_sycl_device_init,
    /* .get_buffer_type         = */ ggml_backend_sycl_device_get_बफर_type,
    /* .get_host_buffer_type    = */ ggml_backend_sycl_device_get_host_बफर_type,
    /* .buffer_from_host_ptr    = */ ggml_backend_sycl_device_बफर_from_host_ptr,
    /* .supports_op             = */ ggml_backend_sycl_device_supports_op,
    /* .supports_buft           = */ ggml_backend_sycl_device_supports_buft,
    /* .offload_op              = */ ggml_backend_sycl_device_offload_op,
    /* .event_new               = */ ggml_backend_sycl_device_event_new,
    /* .event_free              = */ ggml_backend_sycl_device_event_free,
    /* .event_synchronize       = */ ggml_backend_sycl_device_event_synchronize,
};

// backend reg

काष्ठा ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

अटल स्थिर अक्षर * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    वापस GGML_SYCL_NAME;
}

अटल माप_प्रकार ggml_backend_sycl_reg_get_device_गिनती(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    वापस ctx->devices.size();
}

अटल ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, माप_प्रकार सूचकांक) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(सूचकांक < ctx->devices.size());
    वापस ctx->devices[सूचकांक];
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

काष्ठा ggml_backend_sycl_comm_context {
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

    व्योम free_host() {
        क्रम (पूर्णांक i = 0; i < 2; ++i) {
            स्थिर sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            अगर (host_out[i]) { sycl::मुक्त(host_out[i], sctx); host_out[i] = nullptr; }
            अगर (host_in[i])  { sycl::मुक्त(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
अटल व्योम ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    व्योम * dst0, स्थिर व्योम * src1, व्योम * dst1, स्थिर व्योम * src0,
                                    माप_प्रकार nbytes) {
    अगर (comm->host_bytes < nbytes) {
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

व्योम * ggml_backend_sycl_comm_init(ggml_backend_t * backends, माप_प्रकार n_backends) प्रयास {
    क्रम (माप_प्रकार i = 0; i < n_backends; ++i) {
        अगर (!ggml_backend_is_sycl(backends[i])) {
            वापस nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    अगर (n_backends != 2) {
        वापस nullptr;
    }

    स्वतः * ctx = नया ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    स्वतः * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    स्वतः * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    वापस ctx;
}
पकड़ो (स्थिर sycl::exception &) { वापस nullptr; }
पकड़ो (...)                     { वापस nullptr; }

व्योम ggml_backend_sycl_comm_free(व्योम * comm_ctx_v) {
    स्वतः * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    अगर (comm_ctx == nullptr) {
        वापस;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    अगर (comm_ctx->backends.size() == 2) {
        स्वतः * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        स्वतः * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        प्रयास {
            sctx0->stream()->रुको();
            sctx1->stream()->रुको();
            comm_ctx->free_host();
        } पकड़ो (...) { /* best effort during shutdown */ }
    }

    मिटाओ comm_ctx;
}

bool ggml_backend_sycl_comm_allreduce_tensor(व्योम * comm_ctx_v, काष्ठा ggml_tensor ** tensors) प्रयास {
    अगर (comm_ctx_v == nullptr) {
        वापस false;
    }

    स्वतः * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    स्थिर माप_प्रकार n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    अगर (n_backends != 2) {
        वापस false;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    अगर (tensors[0]->type != tensors[1]->type) {
        वापस false;
    }
    अगर (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        वापस false;
    }
    अगर (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        वापस false;
    }
    अगर (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        वापस false;
    }

    स्थिर int64_t nelem  = ggml_nelements(tensors[0]);
    स्थिर माप_प्रकार  nbytes = ggml_nbytes(tensors[0]);
    अगर (nelem == 0) {
        वापस true;
    }

    स्वतः * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    स्वतः * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    अगर (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->पुनः_स्मृति(nelem * 4);
        comm_ctx->buf1->पुनः_स्मृति(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->लो();
    uint8_t * buf1 = comm_ctx->buf1->लो();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    अगर (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->तथ्य;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->तथ्य;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->तथ्य, f16_tmp1, tensors[0]->तथ्य, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((भग्न) f16_out0[i] + (भग्न) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((भग्न) f16_out1[i] + (भग्न) f16_tmp1[i]);
            });
        });
        वापस true;
    }

    भग्न * out0 = (भग्न *) tensors[0]->तथ्य;
    भग्न * out1 = (भग्न *) tensors[1]->तथ्य;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    अटल constexpr int64_t BF16_THRESHOLD = 32768;

    अगर (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        भग्न * tmp0 = (भग्न *) buf0;
        भग्न * tmp1 = (भग्न *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->तथ्य, tmp1, tensors[0]->तथ्य, nbytes);

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
        वापस true;
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
    स्थिर माप_प्रकार bf16_bytes = nelem * माप(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<भग्न>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<भग्न>(((uint32_t) inbox1[i]) << 16);
        });
    });

    वापस true;
}
पकड़ो (स्थिर sycl::exception &) { वापस false; }
पकड़ो (...)                     { वापस false; }

अटल व्योम *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, स्थिर अक्षर *name) {
    GGML_UNUSED(reg);

    अगर (म_भेद(name, "ggml_backend_split_buffer_type") == 0) {
        वापस (व्योम *)ggml_backend_sycl_split_बफर_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    अगर (म_भेद(name, "ggml_backend_comm_init") == 0) {
        वापस (व्योम *)ggml_backend_sycl_comm_init;
    }
    अगर (म_भेद(name, "ggml_backend_comm_free") == 0) {
        वापस (व्योम *)ggml_backend_sycl_comm_free;
    }
    अगर (म_भेद(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        वापस (व्योम *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    वापस nullptr;
}

अटल स्थिर ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_गिनती,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    अटल ggml_backend_reg reg;
    अटल bool initialized = false;

    {
        अटल std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        अगर (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = नया ggml_backend_sycl_reg_context;
            स्थिर पूर्णांक min_batch_आकार = दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH") ? म_से_प(दो_पर्या("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            क्रम (पूर्णांक i = 0; i < ggml_sycl_info().device_गिनती; i++) {
                ggml_backend_sycl_device_context * dev_ctx = नया ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_आकार = min_batch_आकार;

                ggml_backend_dev_t dev = नया ggml_backend_device {
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

        initialized = true;
    }

    वापस &reg;
}

ggml_backend_t ggml_backend_sycl_init(पूर्णांक device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_सूचकांक(device);

    ggml_backend_sycl_context * ctx = नया ggml_backend_sycl_context(device);
    अगर (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        वापस nullptr;
    };

    ggml_backend_t sycl_backend = नया ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    वापस sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
