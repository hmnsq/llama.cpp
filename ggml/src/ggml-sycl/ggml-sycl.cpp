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
#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#अन्तर्भाव <level_zero/ze_api.h>
#अन्तर्भाव <level_zero/zes_api.h>
#यदि_अन्तः
#यदि defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_स्मृतिः_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#यदि_अन्तः
#यदि SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#यदि_अन्तः
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

#परिभाषय MEM_आकार_2_M	0x00200000
#परिभाषय MEM_आकार_1_G	0x40000000

स्थिर तार्किक g_sycl_loaded = असत्य;
पूर्णाङ्क g_ggml_sycl_debug = 0;
पूर्णाङ्क g_ggml_sycl_dev_debug = 0;
पूर्णाङ्क g_ggml_sycl_enable_optimize = 1;
पूर्णाङ्क g_ggml_sycl_enable_graph = 0;
पूर्णाङ्क g_ggml_sycl_enable_dnn = 1;
पूर्णाङ्क g_ggml_sycl_fa_onednn = 1;
पूर्णाङ्क g_ggml_sycl_fa_onednn_max_kv = 0;
पूर्णाङ्क g_ggml_sycl_enable_mkl_fa = 1;
पूर्णाङ्क g_ggml_sycl_memtrace = 0;
पूर्णाङ्क g_ggml_sycl_memtrace_चरणः = 64;
पूर्णाङ्क g_ggml_sycl_enable_vmm = 1;
पूर्णाङ्क g_ggml_sycl_enable_fusion = 1;
पूर्णाङ्क g_ggml_sycl_enable_esimd = 1;
पूर्णाङ्क g_ggml_sycl_mmvq_wide = 1;
पूर्णाङ्क g_ggml_sycl_prioritize_dmmv = 0;
पूर्णाङ्क g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
पूर्णाङ्क g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
पूर्णाङ्क g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
पूर्णाङ्क g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
स्थिर ध्रुव वर्ण * ggml_sycl_dynamic_precision_नामानि[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
स्थिर पूर्णाङ्क ggml_sycl_get_env_precision(ध्रुव वर्ण * name, पूर्णाङ्क def) {
    ध्रुव वर्ण * env = getenv(name);
    यदि (!env) {
        प्रत्यागम def;
    }
    std::string mode(env);
    प्रति (वर्ण & c : mode) {
        c = (वर्ण) std::महाक्षरम्((अचिह्नित वर्ण) c);
    }
    प्रति (पूर्णाङ्क i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        यदि (mode == ggml_sycl_dynamic_precision_नामानि[i]) {
            प्रत्यागम i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_नामानि[def]);
    प्रत्यागम def;
}
पूर्णाङ्क g_ggml_sycl_use_async_mem_op = 0;
पूर्णाङ्क g_ggml_sycl_use_async_mem_op_requested = 1;
पूर्णाङ्क g_ggml_sycl_use_level_zero_api = 0;
पूर्णाङ्क g_ggml_sycl_enable_flash_attention = 1;
पूर्णाङ्क g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
पूर्णाङ्क g_ggml_sycl_usm_system = 0;
पूर्णाङ्क g_ggml_sycl_enable_host_pinned_mem = 1;
पूर्णाङ्क g_ggml_sycl_host_pinned_mem_2g = 0;
पूर्णाङ्क g_ggml_sycl_upload_staging_slots = 4;
पूर्णाङ्क g_ggml_sycl_get_mem_api = स्मृतिः_API_TYPE_LEVEL_ZERO;
पूर्णाङ्क g_ggml_sycl_enable_sparse_fa = 0;
पूर्णाङ्क g_ggml_sycl_debug_sparse_fa = 0;
पूर्णाङ्क g_ggml_sycl_sparse_fa_margin = 256;

#यदि GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
स्थिर तार्किक ggml_sycl_dnnl_detect_optimized_gemm(पूर्णाङ्क device) {
    उपयुज्य dt = dnnl::स्मृतिः::दत्त_type;

    यत्न {
        ध्रुव dnnl::स्मृतिः::dims dims    = { 1, 64, 64 };
        ध्रुव dnnl::स्मृतिः::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.समुच्चयः_scratchpad_बहुलकम्(dnnl::scratchpad_बहुलकम्::user);

        ध्रुव स्वचालित & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        ध्रुव स्वचालित   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ध्रुव स्वचालित   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::स्मृतिः::desc(dims, dt::f16, strides),
                                                        dnnl::स्मृतिः::desc(dims, dt::f16, strides),
                                                        dnnl::स्मृतिः::desc(dims, dt::f32, strides), attr);

        ध्रुव std::string impl = pd.impl_info_str();
        यदि (impl.find("ref") == std::string::npos) {
            प्रत्यागम सत्य;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } गृहाण (ध्रुव std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    प्रत्यागम असत्य;
}
#यदि_अन्तः

स्थिर ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    यत्न {
        info.device_गणनम् = dpct::dev_mgr::instance().device_गणनम्();
    } गृहाण (sycl::exception ध्रुव &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_गणनम् = 0;
    }
    यदि (info.device_गणनम् == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        प्रत्यागम info;
    }

    GGML_ASSERT(info.device_गणनम् <= GGML_SYCL_MAX_DEVICES);

    int64_t योग_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    प्रति (पूर्णाङ्क i = 0; i < info.device_गणनम्; ++i) {
        dpct::device_info prop;
        स्वचालित & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#यदि !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#अन्यथा
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        यदि (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ध्रुव size_t physical_page = 2ull << 20; // 2 MiB
            info.devices[i].vmm_granularity = std::max<size_t>(
                sycl::ext::oneapi::experimental::get_mem_granularity(
                    device, sycl::context(device)),
                physical_page);
        }
#यदि_अन्तः

        info.default_tensor_विभज[i] = योग_vram;
        योग_vram += prop.get_global_mem_आकार();

        info.devices[i].cc =
            100 * prop.get_major_संस्करणम्() + 10 * prop.get_minor_संस्करणम्();
        info.devices[i].nsm = prop.get_max_compute_units() / 16; //16: Number of Xe Cores
        info.devices[i].opt_feature.reorder = device.ext_oneapi_architecture_is(syclex::arch_category::intel_gpu);
#यदि GGML_SYCL_DNNL
        info.devices[i].opt_feature.onednn_optimized_gemm = ggml_sycl_dnnl_detect_optimized_gemm(i);
#यदि_अन्तः
        info.devices[i].smpbo = prop.get_local_mem_आकार();
        info.devices[i].warp_आकार = WARP_आकार;
        info.devices[i].usm_system_support = device.has(sycl::aspect::usm_system_allocations);

        info.max_work_गणः_sizes[i] = prop.get_max_work_गणः_आकार();
        info.devices[i].max_wg_प्रत्येकम्_cu = info.max_work_गणः_sizes[i] / prop.get_max_compute_units();
        info.devices[i].hw_info = get_device_hw_info(&device);

        // Only check GPU devices; CPU devices use OpenCL and would otherwise
        // disable Level Zero for the GPUs on systems without ONEAPI_DEVICE_SELECTOR set.
        यदि (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = असत्य;
        }

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        यदि (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_परिणाम_t r = zeDeviceGetProperties(ze_dev, &props);
            यदि (r == ZE_परिणाम_SUCCESS) {
                info.devices[i].l0_device_type_valid = सत्य;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#यदि_अन्तः
    }

    प्रति (पूर्णाङ्क id = 0; id < info.device_गणनम्; ++id) {
        info.default_tensor_विभज[id] /= योग_vram;
    }

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#यदि_अन्तः

    प्रत्यागम info;
}

ध्रुव ggml_sycl_device_info & ggml_sycl_info() {
    स्थिर ggml_sycl_device_info info = ggml_sycl_init();
    प्रत्यागम info;
}

स्थिर शून्य print_device_detail(पूर्णाङ्क id, sycl::device &device, std::string device_type) {

    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(
        dpct::get_device_info(prop, device)));

    std::string संस्करणम्_2;
    संस्करणम्_2 += std::to_string(prop.get_major_संस्करणम्());
    संस्करणम्_2 += ".";
    संस्करणम्_2 += std::to_string(prop.get_minor_संस्करणम्());

    device_type = std::regex_replace(device_type, std::regex("ext_oneapi_"), "");
    std::string name = std::string(prop.get_name());
    name = std::regex_replace(name, std::regex("\\(R\\)"), "");
    name = std::regex_replace(name, std::regex("\\(TM\\)"), "");

    स्वचालित global_mem_आकार = prop.get_global_mem_आकार()/1000000;
    GGML_LOG_INFO("|%2d|%19s|%39s|%7s|%7d|%8d|%5d|%6luM|%21s|\n", id, device_type.c_str(),
            name.c_str(), संस्करणम्_2.c_str(), prop.get_max_compute_units(),
            prop.get_max_work_गणः_आकार(), prop.get_max_sub_गणः_आकार(),
            global_mem_आकार, device.get_info<sycl::info::device::driver_संस्करणम्>().c_str());
}

स्थिर शून्य print_device_opt_feature(पूर्णाङ्क device_गणनम्) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    प्रति (पूर्णाङ्क id = 0; id < device_गणनम्; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      पूर्णाङ्क type_id = DeviceNums[backend_type]++;
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
    पूर्णाङ्क device_गणनम् = dpct::dev_mgr::instance().device_गणनम्();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_गणनम्);

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

    प्रति (पूर्णाङ्क id = 0; id < device_गणनम्; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      पूर्णाङ्क type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_गणनम्);
}

स्थिर ध्रुव वर्ण* dev2dev_int2str(पूर्णाङ्क dev2dev) {
    यदि (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        प्रत्यागम "SYCL API";
    } अन्यथा यदि (dev2dev == DEV2DEV_MEMCPY_L0) {
        प्रत्यागम "Level Zero API";
    } अन्यथा यदि (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        प्रत्यागम "Host Forward";
    } अन्यथा {
        प्रत्यागम "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर ze_परिणाम_t init_zes() {
    ze_परिणाम_t res = zesInit(0);
    यदि (res != ZE_परिणाम_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (पूर्णाङ्क) res);
    }
    प्रत्यागम res;
}

स्थिर ze_परिणाम_t get_zes_init_res() {
    स्थिर ze_परिणाम_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (पूर्णाङ्क) zes_init_res);
    प्रत्यागम zes_init_res;
}
#यदि_अन्तः

स्थिर शून्य initialize_sycl_begining() {
#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#यदि_अन्तः
}

स्थिर शून्य ggml_check_sycl() यत्न {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    स्थिर तार्किक initialized = असत्य;

    यदि (!initialized) {
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
        g_ggml_sycl_memtrace_चरणः = ggml_sycl_get_env("GGML_SYCL_MEMTRACE_STEP", 64);
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

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        g_ggml_sycl_use_level_zero_api = ggml_sycl_get_env("GGML_SYCL_USE_LEVEL_ZERO_API", 1);
#अन्यथा
        g_ggml_sycl_use_level_zero_api = 0;
#यदि_अन्तः
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", स्मृतिः_API_TYPE_LEVEL_ZERO);
        यदि (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = स्मृतिः_API_TYPE_SYCL;
        }

#यदि_परिभाषितम् SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#अन्यथा
        g_ggml_sycl_enable_flash_attention = 0;
#यदि_अन्तः

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
#यदि defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#यदि_अन्तः

#यदि defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#यदि_अन्तः

#यदि defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#यदि_अन्तः

#यदि defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#यदि_अन्तः

#यदि defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#यदि_अन्तः
#यदि defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#यदि_अन्तः

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#यदि_अन्तः

#यदि defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#यदि_अन्तः
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_चरणः);
#यदि_परिभाषितम् SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#यदि_अन्तः

#यदि_परिभाषितम् GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#यदि_अन्तः

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_नामानि[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_नामानि[g_ggml_sycl_dynamic_required_precision]);

#यदि defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#यदि_अन्तः

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#यदि defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#यदि_अन्तः
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#अन्यथा
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: Disable Level Zero API usage by compile flag\n");
#यदि_अन्तः

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
#यदि defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_स्मृतिः_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        यदि (g_ggml_sycl_use_async_mem_op) {
            प्रति (अचिह्नित पूर्णाङ्क i = 0; i < dpct::dev_mgr::instance().device_गणनम्(); ++i) {
                यदि (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_स्मृतिः_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    भङ्ग;
                }
            }
        }
#यदि_अन्तः
        यदि (CHECK_TRY_ERROR(g_all_sycl_device_गणनम् =
                            dpct::dev_mgr::instance().device_गणनम्()) != 0) {
            initialized = सत्य;
            g_sycl_loaded = असत्य;
            प्रत्यागम;
        }
        GGML_ASSERT(g_all_sycl_device_गणनम् <= GGML_SYCL_MAX_DEVICES);

        initialized = सत्य;
        g_sycl_loaded = सत्य;
        ggml_backend_sycl_print_sycl_devices();
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
अन्तःस्थ शून्य check_allow_gpu_सूचकांक(ध्रुव पूर्णाङ्क device_सूचकांक) {
  यदि (device_सूचकांक >= ggml_sycl_info().device_गणनम्) {
    वर्ण error_buf[256];
    snprintf(
        error_buf,
        परिमाण(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_सूचकांक,
        ggml_sycl_info().device_गणनम् - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    प्रतिज्ञा(असत्य);
  }
}

GGML_API शून्य ggml_backend_sycl_get_gpu_list(पूर्णाङ्क *id_list, पूर्णाङ्क max_len) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    प्रति(पूर्णाङ्क i=0;i<max_len;i++) id_list[i] = -1;

    प्रति (पूर्णाङ्क i=0;i< ggml_sycl_info().device_गणनम्;i++){
        यदि (i>=max_len) भङ्ग;
        id_list[i] = i;
    }
    प्रत्यागम;
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

अन्तःस्थ शून्य free_aligned_mem_host(शून्य * memblock) {
#यदि_परिभाषितम् _WIN32
    _aligned_free(memblock);
#अन्यथा
    मुक्त(memblock);
#यदि_अन्तः
}

// sycl buffer

संरचना ggml_backend_sycl_बफर_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    स्थिर constexpr size_t staging_slot_आकार = 8*1024*1024;

    संरचना host_staging {
        शून्य * दत्त = शून्यसूचकः;
        std::vector<sycl::event> events;
        std::vector<तार्किक> submitted;
        पूर्णाङ्क अग्र = 0;
    };

    पूर्णाङ्क device;
    शून्य * dev_ptr = शून्यसूचकः;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    तार्किक is_usm_system;

    ggml_backend_sycl_बफर_context(पूर्णाङ्क device, शून्य * dev_ptr, queue_ptr stream, तार्किक is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_सूचकांक(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    शून्य drop_host_staging() {
        प्रति (size_t i = 0; i < staging.submitted.size(); ++i) {
            यदि (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = असत्य;
            }
        }
        यदि (staging.दत्त != शून्यसूचकः) {
            sycl::मुक्त(staging.दत्त, *stream);
            staging.दत्त = शून्यसूचकः;
        }
    }

    ~ggml_backend_sycl_बफर_context() {
        drop_host_staging();
        यदि (dev_ptr != शून्यसूचकः) {
            ggml_sycl_समुच्चयः_device(device);
            यदि (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            अन्यथा
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        प्रति (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

स्थिर ध्रुव वर्ण * ggml_backend_sycl_बफर_type_get_name(ggml_backend_बफर_type_t buft);

स्थिर तार्किक ggml_backend_बफर_is_sycl(ggml_backend_बफर_t बफर) {
    प्रत्यागम बफर->buft->iface.get_name == ggml_backend_sycl_बफर_type_get_name;
}

स्थिर शून्य
ggml_backend_sycl_बफर_free_बफर(ggml_backend_बफर_t बफर) यत्न {
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    ggml_sycl_समुच्चयः_device(ctx->device);

    विलोपय ctx;
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य * ggml_backend_sycl_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    प्रत्यागम ctx->dev_ptr;
}

स्थिर गणना ggml_स्थितिः
ggml_backend_sycl_बफर_init_tensor(ggml_backend_बफर_t बफर,
                                     ggml_tensor *tensor) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *)बफर->context;

    यदि (tensor->view_src != रिक्त) {
        प्रतिज्ञा(tensor->view_src->बफर->buft == बफर->buft);
        प्रत्यागम GGML_स्थितिः_SUCCESS;
    }

    यदि (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        विकल्प (tensor->type) {
            पक्ष GGML_TYPE_Q4_0:
            पक्ष GGML_TYPE_Q8_0:
            पक्ष GGML_TYPE_Q2_K:
            पक्ष GGML_TYPE_Q3_K:
            पक्ष GGML_TYPE_Q4_K:
            पक्ष GGML_TYPE_Q5_K:
            पक्ष GGML_TYPE_Q6_K:
            पक्ष GGML_TYPE_IQ3_XXS:
            पक्ष GGML_TYPE_IQ3_S:
            पक्ष GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = नूतन ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                भङ्ग;
            }
            स्वभाव:
                भङ्ग;
        }
    }

    यदि (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_आकार = ggml_nbytes(tensor);
        size_t padded_आकार = ggml_backend_buft_get_alloc_आकार(बफर->buft, tensor);

        यदि (padded_आकार > original_आकार && tensor->view_src == शून्यसूचकः) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->स्मृतिं_पूरय(
                (वर्ण *)tensor->दत्त + original_आकार, 0,
                padded_आकार - original_आकार).प्रतीक्षस्व()));
        }
    }
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर,
                                                ggml_tensor *tensor,
                                                ध्रुव शून्य *दत्त, size_t विस्थापनम्,
                                                size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;
    ggml_sycl_समुच्चयः_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ध्रुव पूर्णाङ्क n_slots = g_ggml_sycl_upload_staging_slots;
    यदि (n_slots > 0 && ctx->staging.दत्त == शून्यसूचकः) {
        ctx->staging.दत्त = sycl::malloc_host(n_slots * ctx->staging_slot_आकार, *ctx->stream);
        यदि (ctx->staging.दत्त != शून्यसूचकः) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.नियोजय(n_slots, असत्य);
        }
    }
    यदि (ctx->staging.दत्त != शून्यसूचकः) {
        queue_ptr    stream    = ctx->stream;
        वर्ण *       dst       = (वर्ण *) tensor->दत्त + विस्थापनम्;
        ध्रुव वर्ण * src       = (ध्रुव वर्ण *) दत्त;
        size_t       remaining = size;
        यावत् (remaining > 0) {
            ध्रुव size_t chunk = std::min(remaining, ctx->staging_slot_आकार);
            ध्रुव पूर्णाङ्क    slot  = ctx->staging.अग्र;
            ctx->staging.अग्र = (ctx->staging.अग्र + 1) % (पूर्णाङ्क) ctx->staging.submitted.size();
            यदि (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            शून्य * stage = (वर्ण *) ctx->staging.दत्त + slot * ctx->staging_slot_आकार;
            स्मृतिं_प्रतिलिख(stage, src, chunk);
            ctx->staging.events[slot] = stream->स्मृतिं_प्रतिलिख(dst, stage, chunk);
            ctx->staging.submitted[slot] = सत्य;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        प्रत्यागम;
    }

    स्वचालित stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#यदि_अपरिभाषितम् _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    वर्ण * host_buf = (वर्ण *) स्मृतिं_देहि(size);
    स्मृतिं_प्रतिलिख(host_buf, दत्त, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स्मृतिं_प्रतिलिख((वर्ण *) tensor->दत्त + विस्थापनम्, host_buf, size).प्रतीक्षस्व()));
    मुक्त(host_buf);
#अन्यथा
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स्मृतिं_प्रतिलिख((वर्ण *) tensor->दत्त + विस्थापनम्, दत्त, size).प्रतीक्षस्व()));
#यदि_अन्तः
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                                ध्रुव ggml_tensor *tensor,
                                                शून्य *दत्त, size_t विस्थापनम्,
                                                size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    ggml_backend_sycl_बफर_context * ctx = ( ggml_backend_sycl_बफर_context *)बफर->context;

    ggml_sycl_समुच्चयः_device(ctx->device);
    स्वचालित stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.स्मृतिं_प्रतिलिख(दत्त, (ध्रुव वर्ण *)tensor->दत्त + विस्थापनम्, size)
            .प्रतीक्षस्व()));
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
स्थिर तार्किक ggml_sycl_is_l0_discrete_gpu(पूर्णाङ्क device) {
    प्रत्यागम ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#यदि_अन्तः

स्थिर शून्य memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, शून्य *ptr_dst,
                         ध्रुव शून्य *ptr_src, size_t size) {
    वर्ण *host_buf = (वर्ण *)स्मृतिं_देहि(size);
    यदि (host_buf == शून्यसूचकः) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.स्मृतिं_प्रतिलिख(host_buf, (ध्रुव वर्ण *)ptr_src, size).प्रतीक्षस्व();
    q_dst.स्मृतिं_प्रतिलिख((वर्ण *)ptr_dst, host_buf, size).प्रतीक्षस्व();
    मुक्त(host_buf);
}

स्थिर शून्य dev2dev_memcpy(पूर्णाङ्क device_dst, sycl::queue &q_dst, पूर्णाङ्क device_src, sycl::queue &q_src, शून्य *ptr_dst,
                    ध्रुव शून्य *ptr_src, size_t size) {

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    यदि (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ध्रुव तार्किक l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        यदि (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            स्वचालित ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            स्वचालित ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, शून्यसूचकः, 0, 0,
                                            0, ZE_COMMAND_QUEUE_बहुलकम्_SYNCHRONOUS, ZE_COMMAND_QUEUE_प्राथमिकता_प्रसामान्यम्};
            ze_command_list_handle_t cl;
            ze_परिणाम_t r = ze_Command_List_Create_तात्कालिकम्(ze_ctx, ze_dev, &cq_desc, &cl);
            यदि (r == ZE_परिणाम_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = ze_Command_List_Append_स्मृतिः_Copy(cl, ptr_dst, ptr_src, size, शून्यसूचकः, 0, शून्यसूचकः);
                zeCommandListDestroy(cl);
                यदि (r == ZE_परिणाम_SUCCESS) {
                    प्रत्यागम;
                }
            }
        }
    }
#यदि_अन्तः

    यदि (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        यदि (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.स्मृतिं_प्रतिलिख(ptr_dst, ptr_src, size).प्रतीक्षस्व()));
            प्रत्यागम;
        }
    }

    // Host-staged copy
    यदि(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } अन्यथा {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

स्थिर तार्किक
ggml_backend_sycl_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                    ध्रुव ggml_tensor *src,
                                    ggml_tensor *dst) यत्न {
    तार्किक is_cpy_supported = ggml_backend_बफर_is_sycl(src->बफर);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    यदि (is_cpy_supported) {
        ggml_backend_sycl_बफर_context * src_ctx = (ggml_backend_sycl_बफर_context *)src->बफर->context;
        ggml_backend_sycl_बफर_context * dst_ctx = (ggml_backend_sycl_बफर_context *)dst->बफर->context;

        ggml_sycl_समुच्चयः_device(src_ctx->device);
        /*
        DPCT1009:198: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(src_ctx->device).queues_wait_and_throw()));
        ggml_sycl_समुच्चयः_device(dst_ctx->device);
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

        queue_ptr प्रवाहः_dst = dst_ctx->stream;
        queue_ptr प्रवाहः_src = src_ctx->stream;
        size_t size = ggml_nbytes(src);

        //todo. it's dirty solutino to walkaroud known issue:device2device cross GPUs.
        dev2dev_memcpy(dst_ctx->device, *प्रवाहः_dst, src_ctx->device, *प्रवाहः_src, dst->दत्त, src->दत्त, size);

//todo, it's known issue：error in device2device cross GPUs. reused when the issue is fixed. DON"T remove
#यदि 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).स्मृतिं_प्रतिलिख(
            (वर्ण *)dst->दत्त, (ध्रुव वर्ण *)src->दत्त, size).प्रतीक्षस्व()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#यदि_अन्तः
        प्रत्यागम सत्य;
    }
    प्रत्यागम असत्य;
    GGML_अप्रयुक्तम्(बफर);
} गृहाण (ध्रुव sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_बफर_clear(ggml_backend_बफर_t बफर,
                                           uint8_t value) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, बफर->size);
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;

    ggml_sycl_समुच्चयः_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    प्रति (size_t off = 0; off < बफर->size; off += MAX_CHUNK) {
        size_t chunk = std::min(बफर->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स्मृतिं_पूरय(static_cast<वर्ण*>(ctx->dev_ptr) + off, value, chunk)
                .प्रतीक्षस्व()
        ));
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_बफर_memset_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor, uint8_t value,
                                                   size_t विस्थापनम्, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, विस्थापनम्, value);
    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx->device));
    स्वचालित stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    यदि (size == 0) {
        प्रत्यागम;  // Nothing to do
    }
    यदि (tensor->दत्त == शून्यसूचकः) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    शून्य * target_ptr = static_cast<वर्ण *>(tensor->दत्त) + विस्थापनम्;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).स्मृतिं_पूरय(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).प्रतीक्षस्व()));
}

स्थिर शून्य ggml_backend_sycl_बफर_reset(ggml_backend_बफर_t बफर) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    यदि (बफर == शून्यसूचकः) {
        प्रत्यागम;
    }

    ggml_backend_sycl_बफर_context * ctx = (ggml_backend_sycl_बफर_context *) बफर->context;

    यदि (ctx != शून्यसूचकः) {
        प्रति (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

स्थिर ध्रुव ggml_backend_बफर_i ggml_backend_sycl_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_sycl_बफर_get_आधारः,
    /* .init_tensor     = */ ggml_backend_sycl_बफर_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_बफर_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_बफर_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_sycl_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_बफर_clear,
    /* .reset           = */ ggml_backend_sycl_बफर_reset,
};

// sycl buffer type
संरचना ggml_backend_sycl_बफर_type_context {
    पूर्णाङ्क device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = शून्यसूचकः;
};

स्थिर ध्रुव वर्ण * ggml_backend_sycl_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    ggml_backend_sycl_बफर_type_context * ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;

    प्रत्यागम ctx->name.c_str();
}

स्थिर तार्किक check_usm_system(पूर्णाङ्क device, size_t size) {
    तार्किक use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_आकार_1_G);

    यदि (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = असत्य;
    }

    प्रत्यागम use_usm_system;
}

अन्तःस्थ शून्य * aligned_malloc_host(size_t alignment, size_t size) {
#यदि_परिभाषितम् _WIN32
    प्रत्यागम _aligned_malloc(size, alignment);
#अन्यथा
    प्रत्यागम aligned_alloc(alignment, size);
#यदि_अन्तः
}

स्थिर ggml_backend_बफर_t
ggml_backend_sycl_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft,
                                           size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_बफर_type_context * buft_ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    ggml_sycl_समुच्चयः_device(buft_ctx->device);
    ध्रुव queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_आकार_2_M;
    size_t aligned_आकार = ((size + alignment - 1) / alignment) * alignment;
    तार्किक use_usm_system = check_usm_system(buft_ctx->device, aligned_आकार);

    शून्य * dev_ptr;
    यदि (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (शून्य *)aligned_malloc_host(alignment, aligned_आकार);
        यदि (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            प्रत्यागम शून्यसूचकः;
        }
    } अन्यथा {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (शून्य *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_बफर)));
        यदि (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          प्रत्यागम शून्यसूचकः;
        }
    }
    ggml_backend_sycl_बफर_context * ctx = नूतन  ggml_backend_sycl_बफर_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_sycl_बफर_interface, ctx, size);
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर size_t ggml_backend_sycl_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    प्रत्यागम SYCL_बफर_ALIGNMENT;
    GGML_अप्रयुक्तम्(buft);
}

स्थिर तार्किक is_bmg_g31_arch(पूर्णाङ्क device) {
    प्रत्यागम ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

स्थिर size_t ggml_backend_sycl_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    ggml_backend_sycl_बफर_type_context * ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    पूर्णाङ्क device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_आकार = dpct::get_device(device).get_max_mem_alloc_आकार();
    यदि (g_ggml_sycl_host_pinned_mem_2g) {
        प्रत्यागम std::min(max_alloc_आकार, (size_t) 2LL*1024*1024*1024);
    } अन्यथा {
        यदि(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_आकार = max_alloc_आकार*0.6;
        }
        प्रत्यागम max_alloc_आकार;

    }
    GGML_अप्रयुक्तम्(buft);
}

स्थिर size_t ggml_backend_sycl_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_आकार(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    यदि (ggml_is_quantized(tensor->type)) {
        यदि (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    प्रत्यागम size;

    GGML_अप्रयुक्तम्(buft);
}

स्थिर ध्रुव ggml_backend_बफर_type_i ggml_backend_sycl_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_sycl_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_sycl_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ रिक्त,
};

ggml_backend_बफर_type_t ggml_backend_sycl_बफर_type(पूर्णाङ्क device) {
    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> तालकम्(mutex);


    स्वचालित dev_गणनम् = ggml_backend_sycl_get_device_गणनम्();

    यदि (device>=dev_गणनम् or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_गणनम्-1);
        GGML_ASSERT(device<dev_गणनम्);
    }
    स्थिर संरचना ggml_backend_बफर_type ggml_backend_sycl_बफर_types[GGML_SYCL_MAX_DEVICES];

    स्थिर तार्किक ggml_backend_sycl_बफर_type_initialized = असत्य;

    यदि (!ggml_backend_sycl_बफर_type_initialized) {
        प्रति (पूर्णाङ्क i = 0; i < dev_गणनम्; i++) {
            स्वचालित & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_बफर_types[i] = {
                /* .iface    = */ ggml_backend_sycl_बफर_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ नूतन ggml_backend_sycl_बफर_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_बफर_type_initialized = सत्य;
    }
    प्रत्यागम &ggml_backend_sycl_बफर_types[device];
}

स्थिर ggml_backend_बफर_type_t ggml_backend_sycl_बफर_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    पूर्णाङ्क device = ctx->device;
    यदि (device>=ggml_sycl_info().device_गणनम् or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_गणनम्-1);
        GGML_ASSERT(device<ggml_sycl_info().device_गणनम्);
    }
    स्थिर संरचना ggml_backend_बफर_type ggml_backend_sycl_बफर_types[GGML_SYCL_MAX_DEVICES];

    स्थिर तार्किक ggml_backend_sycl_बफर_type_initialized = असत्य;

    यदि (!ggml_backend_sycl_बफर_type_initialized) {
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; i++) {
            ggml_backend_sycl_बफर_types[i] = {
                /* .iface    = */ ggml_backend_sycl_बफर_type_interface,
                /* .device   = */ शून्यसूचकः,
                /* .context  = */ नूतन ggml_backend_sycl_बफर_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_बफर_type_initialized = सत्य;
    }
    प्रत्यागम &ggml_backend_sycl_बफर_types[device];
}

// sycl split buffer

स्थिर int64_t get_row_rounding(ggml_type type, ध्रुव std::array<भिन्न, GGML_SYCL_MAX_DEVICES> & tensor_विभज) {
    int64_t min_compute_capability = अधिकतम_पूर्णाङ्कः;
    int64_t max_compute_capability = INT_MIN;
    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        यदि (tensor_विभज[i] < (i + 1 < ggml_sycl_info().device_गणनम् ? tensor_विभज[i + 1] : 1.0f)) {
            यदि (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            यदि (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    विकल्प(type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
            प्रत्यागम max_compute_capability >= VER_GEN9 ? 128 : 64;
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम 64;
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_F32:
            प्रत्यागम 1;
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_IQ2_XXS:
        पक्ष GGML_TYPE_IQ2_XS:
        पक्ष GGML_TYPE_IQ2_S:
        पक्ष GGML_TYPE_IQ1_S:
        पक्ष GGML_TYPE_IQ1_M:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ4_XS:
        पक्ष GGML_TYPE_IQ4_NL:
            प्रत्यागम max_compute_capability >= VER_GEN9 ? 128 : 64;
        पक्ष GGML_TYPE_IQ3_S:
            प्रत्यागम max_compute_capability >= VER_GEN9 ? 128 : 64;
        पक्ष GGML_TYPE_Q6_K:
            प्रत्यागम 64;
        स्वभाव:
            GGML_ABORT("fatal error");
    }
}

स्थिर शून्य get_row_विभज(int64_t * row_नीचः, int64_t * row_उच्चः, ध्रुव ggml_tensor * tensor, ध्रुव std::array<भिन्न, GGML_SYCL_MAX_DEVICES> & tensor_विभज, पूर्णाङ्क id) {
    ध्रुव int64_t nrows = ggml_nrows(tensor);
    ध्रुव int64_t rounding = get_row_rounding(tensor->type, tensor_विभज);

    *row_नीचः = id == 0 ? 0 : nrows*tensor_विभज[id];
    *row_नीचः -= *row_नीचः % rounding;
    यदि (id == ggml_sycl_info().device_गणनम् - 1) {
        *row_उच्चः = nrows;
    } अन्यथा {
        *row_उच्चः = nrows*tensor_विभज[id + 1];
        *row_उच्चः -= *row_उच्चः % rounding;
    }
}

स्थिर size_t ggml_nbytes_विभज(ध्रुव संरचना ggml_tensor * tensor, पूर्णाङ्क nrows_विभज) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    प्रत्यागम nrows_विभज*ggml_row_आकार(tensor->type, tensor->ne[0]);
}

संरचना ggml_backend_sycl_विभज_बफर_type_context {
    std::array<भिन्न, GGML_SYCL_MAX_DEVICES> tensor_विभज;
};

संरचना ggml_backend_sycl_विभज_बफर_context {
    ~ggml_backend_sycl_विभज_बफर_context() यत्न {
        प्रति (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    गृहाण (sycl::exception ध्रुव &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
      std::निर्गच्छ(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

स्थिर शून्य ggml_backend_sycl_विभज_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_sycl_विभज_बफर_context * ctx = (ggml_backend_sycl_विभज_बफर_context *)बफर->context;
    विलोपय ctx;
}

स्थिर शून्य * ggml_backend_sycl_विभज_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    प्रत्यागम (शून्य *)0x1000;

    GGML_अप्रयुक्तम्(बफर);
}

स्थिर गणना ggml_स्थितिः
ggml_backend_sycl_विभज_बफर_init_tensor(ggml_backend_बफर_t बफर,
                                           ggml_tensor *tensor) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == शून्यसूचकः); // views of split tensors are not supported

    ggml_backend_sycl_विभज_बफर_context * ctx = (ggml_backend_sycl_विभज_बफर_context *)बफर->context;
    ggml_backend_sycl_विभज_बफर_type_context * buft_ctx = (ggml_backend_sycl_विभज_बफर_type_context *)बफर->buft->context;

    ध्रुव int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = नूतन ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    यदि (ctx->streams.रिक्तानि()) {
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        int64_t row_नीचः, row_उच्चः;
        get_row_विभज(&row_नीचः, &row_उच्चः, tensor, buft_ctx->tensor_विभज, i);

        int64_t nrows_विभज = row_उच्चः - row_नीचः;
        यदि (nrows_विभज == 0) {
            अनुवर्त;
        }

        size_t size = ggml_nbytes_विभज(tensor, nrows_विभज);
        ध्रुव size_t original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        यदि (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_समुच्चयः_device(i);
        ध्रुव queue_ptr stream = ctx->streams[i];
        वर्ण * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (वर्ण *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_बफर)));
        यदि (!buf) {
            वर्ण err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            क्षिप std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        यदि (size > original_आकार) {
            /*
            DPCT1009:209: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(CHECK_TRY_ERROR(
                (*stream)
                    .स्मृतिं_पूरय(buf + original_आकार, 0, size - original_आकार)
                    .प्रतीक्षस्व()));
        }

        extra->दत्त_device[i] = buf;

        प्रति (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = नूतन sycl::event()));
        }
    }
    tensor->extra = extra;
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य
ggml_backend_sycl_विभज_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर,
                                          ggml_tensor *tensor, ध्रुव शून्य *दत्त,
                                          size_t विस्थापनम्, size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(विस्थापनम् == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_विभज_बफर_context * ctx = (ggml_backend_sycl_विभज_बफर_context *)बफर->context;
    ggml_backend_sycl_विभज_बफर_type_context * buft_ctx = (ggml_backend_sycl_विभज_बफर_type_context *)बफर->buft->context;

    ध्रुव int64_t ne0 = tensor->ne[0];
    ध्रुव size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        int64_t row_नीचः, row_उच्चः;
        get_row_विभज(&row_नीचः, &row_उच्चः, tensor, buft_ctx->tensor_विभज, i);

        int64_t nrows_विभज = row_उच्चः - row_नीचः;
        यदि (nrows_विभज == 0) {
            अनुवर्त;
        }

        ध्रुव size_t विस्थापनम्_विभज = row_नीचः*nb1;
        size_t size = ggml_nbytes_विभज(tensor, nrows_विभज);
        ध्रुव size_t original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        यदि (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ध्रुव वर्ण * buf_host = (ध्रुव वर्ण *)दत्त + विस्थापनम्_विभज;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_समुच्चयः_device(i);
        ध्रुव queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स्मृतिं_प्रतिलिख(extra->दत्त_device[i], buf_host, original_आकार)
                .प्रतीक्षस्व()));
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य
ggml_backend_sycl_विभज_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                          ध्रुव ggml_tensor *tensor, शून्य *दत्त,
                                          size_t विस्थापनम्, size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(विस्थापनम् == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_विभज_बफर_context * ctx = (ggml_backend_sycl_विभज_बफर_context *)बफर->context;
    ggml_backend_sycl_विभज_बफर_type_context * buft_ctx = (ggml_backend_sycl_विभज_बफर_type_context *)बफर->buft->context;

    ध्रुव int64_t ne0 = tensor->ne[0];
    ध्रुव size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        int64_t row_नीचः, row_उच्चः;
        get_row_विभज(&row_नीचः, &row_उच्चः, tensor, buft_ctx->tensor_विभज, i);

        int64_t nrows_विभज = row_उच्चः - row_नीचः;
        यदि (nrows_विभज == 0) {
            अनुवर्त;
        }

        ध्रुव size_t विस्थापनम्_विभज = row_नीचः*nb1;
        size_t size = ggml_nbytes_विभज(tensor, nrows_विभज);
        ध्रुव size_t original_आकार = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        यदि (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        वर्ण * buf_host = (वर्ण *)दत्त + विस्थापनम्_विभज;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_समुच्चयः_device(i);
        ध्रुव queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .स्मृतिं_प्रतिलिख(buf_host, extra->दत्त_device[i], original_आकार)
                .प्रतीक्षस्व()));
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_विभज_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    GGML_अप्रयुक्तम्(बफर);
    GGML_अप्रयुक्तम्(value);
}

स्थिर संरचना ggml_backend_बफर_i ggml_backend_sycl_विभज_बफर_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_विभज_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_sycl_विभज_बफर_get_आधारः,
    /* .init_tensor     = */ ggml_backend_sycl_विभज_बफर_init_tensor,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_sycl_विभज_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_विभज_बफर_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ रिक्त,
    /* .clear           = */ ggml_backend_sycl_विभज_बफर_clear,
    /* .reset           = */ रिक्त,
};

// sycl split buffer type

स्थिर ध्रुव वर्ण * ggml_backend_sycl_विभज_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    प्रत्यागम GGML_SYCL_NAME "_Split";

    GGML_अप्रयुक्तम्(buft);
}

स्थिर तार्किक ggml_backend_बफर_is_sycl_विभज(ggml_backend_बफर_t बफर) {
   प्रत्यागम बफर->buft->iface.get_name == ggml_backend_sycl_विभज_बफर_type_get_name;
}

स्थिर ggml_backend_बफर_t ggml_backend_sycl_विभज_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_विभज_बफर_context * ctx = नूतन ggml_backend_sycl_विभज_बफर_context();

    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_sycl_विभज_बफर_interface, ctx, size);
}

स्थिर size_t ggml_backend_sycl_विभज_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    प्रत्यागम SYCL_बफर_ALIGNMENT;
    GGML_अप्रयुक्तम्(buft);
}

स्थिर size_t ggml_backend_sycl_विभज_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor) {
    ggml_backend_sycl_विभज_बफर_type_context * ctx = (ggml_backend_sycl_विभज_बफर_type_context *)buft->context;

    size_t योग_आकार = 0;

    ध्रुव int64_t ne0 = tensor->ne[0];

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        int64_t row_नीचः, row_उच्चः;
        get_row_विभज(&row_नीचः, &row_उच्चः, tensor, ctx->tensor_विभज, i);

        int64_t nrows_विभज = row_उच्चः - row_नीचः;
        यदि (nrows_विभज == 0) {
            अनुवर्त;
        }

        योग_आकार += ggml_nbytes_विभज(tensor, nrows_विभज);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        यदि (ne0 % MATRIX_ROW_PADDING != 0) {
            योग_आकार += ggml_row_आकार(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    प्रत्यागम योग_आकार;
}

स्थिर तार्किक ggml_backend_sycl_विभज_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    प्रत्यागम असत्य;

    GGML_अप्रयुक्तम्(buft);
}

स्थिर ggml_backend_बफर_type_i ggml_backend_sycl_विभज_बफर_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_विभज_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_विभज_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_sycl_विभज_बफर_type_get_alignment,
    /* .get_max_size        = */ रिक्त, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_विभज_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ ggml_backend_sycl_विभज_बफर_type_is_host,
};

ggml_backend_बफर_type_t ggml_backend_sycl_विभज_बफर_type(पूर्णाङ्क main_device, ध्रुव भिन्न * tensor_विभज) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_अप्रयुक्तम्(main_device);

    स्थिर std::mutex mutex;
    std::lock_guard<std::mutex> तालकम्(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    स्थिर std::map<std::array<भिन्न, GGML_SYCL_MAX_DEVICES>, संरचना ggml_backend_बफर_type> buft_map;

    std::array<भिन्न, GGML_SYCL_MAX_DEVICES> tensor_विभज_arr = {};

    तार्किक all_zero = tensor_विभज == शून्यसूचकः || std::all_of(tensor_विभज, tensor_विभज + GGML_SYCL_MAX_DEVICES, [](भिन्न x) { प्रत्यागम x == 0.0f; });
    यदि (all_zero) {
        tensor_विभज_arr = ggml_sycl_info().default_tensor_विभज;
    } अन्यथा {
        भिन्न विभज_योग = 0.0f;
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
            tensor_विभज_arr[i] = विभज_योग;
            विभज_योग += tensor_विभज[i];
        }
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
            tensor_विभज_arr[i] /= विभज_योग;
        }
    }

    स्वचालित it = buft_map.find(tensor_विभज_arr);
    यदि (it != buft_map.end()) {
        प्रत्यागम &it->second;
    }

    संरचना ggml_backend_बफर_type buft {
        /* .iface   = */ ggml_backend_sycl_विभज_बफर_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ नूतन ggml_backend_sycl_विभज_बफर_type_context{tensor_विभज_arr},
    };

    स्वचालित परिणाम = buft_map.emplace(tensor_विभज_arr, buft);
    प्रत्यागम &परिणाम.first->second;
}

// host buffer type

संरचना ggml_backend_sycl_device_context {
    पूर्णाङ्क device;
    std::string name;
    std::string description;
    पूर्णाङ्क op_offload_min_batch_आकार;
};

स्थिर ध्रुव वर्ण * ggml_backend_sycl_host_बफर_type_name(ggml_backend_बफर_type_t buft) {
    प्रत्यागम GGML_SYCL_NAME "_Host";

    GGML_अप्रयुक्तम्(buft);
}

स्थिर पूर्णाङ्क ggml_backend_sycl_host_बफर_type_device(ggml_backend_बफर_type_t buft) {
    प्रत्यागम static_cast<ध्रुव ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
स्थिर शून्य * ggml_backend_sycl_host_malloc(पूर्णाङ्क device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    शून्य * ptr = शून्यसूचकः;
    यत्न {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        स्वचालित & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } गृहाण (...) {
        ptr = शून्यसूचकः;
    }
    यदि (ptr == शून्यसूचकः) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    प्रत्यागम ptr;
}

स्थिर शून्य ggml_backend_sycl_host_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    यदि (बफर->context == शून्यसूचकः) {
        प्रत्यागम;
    }
    यदि (g_ggml_sycl_enable_host_pinned_mem) {
        ध्रुव पूर्णाङ्क device = ggml_backend_sycl_host_बफर_type_device(बफर->buft);
        स्वचालित & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(बफर->context, q)));
    } अन्यथा {
        free_aligned_mem_host((शून्य *) बफर->context);
    }
}

स्थिर ggml_backend_बफर_t ggml_backend_sycl_host_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    शून्य * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_बफर_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    यदि (ptr == शून्यसूचकः) {
        // fallback to cpu buffer
        प्रत्यागम ggml_backend_buft_alloc_बफर(ggml_backend_cpu_बफर_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_बफर_t बफर = ggml_backend_cpu_बफर_from_ptr(ptr, size);
    बफर->buft = buft;
    बफर->iface.free_बफर = ggml_backend_sycl_host_बफर_free_बफर;

    प्रत्यागम बफर;
}

स्थिर size_t ggml_backend_sycl_host_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {

    यदि (g_ggml_sycl_enable_host_pinned_mem) {
        ध्रुव पूर्णाङ्क device = ggml_backend_sycl_host_बफर_type_device(buft);
        size_t max_alloc_आकार = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_आकार();
        यदि (g_ggml_sycl_host_pinned_mem_2g) {
            प्रत्यागम std::min(max_alloc_आकार, (size_t) 2LL*1024*1024*1024);
        } अन्यथा {
            प्रत्यागम max_alloc_आकार;
        }
    } अन्यथा {
        प्रत्यागम SIZE_MAX;
    }
}

स्थिर ggml_backend_बफर_type_t ggml_backend_sycl_host_बफर_type_for_device(पूर्णाङ्क device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    स्थिर std::vector<ggml_backend_बफर_type> बफर_types_host = [] {
        std::vector<ggml_backend_बफर_type> bufts(ggml_backend_sycl_get_device_गणनम्());
        प्रति (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_बफर_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_बफर_type_alloc_बफर,
                    /* .alloc_buffer_n      = */ रिक्त,
                    /* .get_alignment       = */ ggml_backend_cpu_बफर_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_बफर_type_get_max_आकार,
                    /* .get_alloc_size      = */ ggml_backend_cpu_बफर_type()->iface.get_alloc_आकार,
                    /* .get_alloc_size_n    = */ रिक्त,
                    /* .is_host             = */ ggml_backend_cpu_बफर_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ शून्यसूचकः,
            };
        }
        प्रत्यागम bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (पूर्णाङ्क) बफर_types_host.size());

    प्रत्यागम &बफर_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_बफर_type_t ggml_backend_sycl_host_बफर_type() {
    प्रत्यागम ggml_backend_sycl_host_बफर_type_for_device(0);
}

// buffer pool for sycl (legacy)
संरचना ggml_sycl_pool_leg : सार्वजनिक ggml_sycl_pool {
    स्थिर ध्रुव पूर्णाङ्क MAX_SYCL_BUFFERS = 256;

    पूर्णाङ्क device;
    queue_ptr qptr;
    संरचना ggml_sycl_बफर {
        शून्य * ptr = शून्यसूचकः;
        size_t size = 0;
    };

    ggml_sycl_बफर बफर_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_आकार = 0;

    स्पष्ट ggml_sycl_pool_leg(queue_ptr qptr_, पूर्णाङ्क device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#यदि_परिभाषितम् DEBUG_SYCL_POOL
        पूर्णाङ्क    n_cached    = 0;
        size_t अष्टकानि_cached = 0;
        प्रति (पूर्णाङ्क i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            यदि (बफर_pool[i].ptr != शून्यसूचकः) {
                ++n_cached;
                अष्टकानि_cached += बफर_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, अष्टकानि_cached / 1024.0 / 1024.0);
        ध्रुव स्वचालित slots = format_slots_अन्तर्_alloc_क्रमः();
        यदि (!slots.रिक्तानि()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#यदि_अन्तः

        प्रति (पूर्णाङ्क i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            यदि (b.ptr != शून्यसूचकः) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_आकार -= b.size;
            }
        }
        GGML_ASSERT(pool_आकार == 0);
    }

#यदि_परिभाषितम् DEBUG_SYCL_POOL
    std::string format_slots_अन्तर्_alloc_क्रमः() ध्रुव {
        std::string पङ्क्तिः;
        वर्ण buf[32];
        तार्किक first = सत्य;
        प्रति (पूर्णाङ्क i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            यदि (बफर_pool[i].ptr == शून्यसूचकः) {
                अनुवर्त;
            }
            यदि (!first) {
                पङ्क्तिः += '/';
            }
            first = असत्य;
            snprintf(buf, परिमाण(buf), "%.2f", बफर_pool[i].size / 1024.0 / 1024.0);
            पङ्क्तिः += buf;
        }
        प्रत्यागम पङ्क्तिः;
    }
#यदि_अन्तः

    शून्य * alloc(size_t size, size_t * actual_आकार) override {
#यदि_परिभाषितम् DEBUG_sycl_MALLOC
        पूर्णाङ्क nnz = 0;
        size_t max_size = 0;
#यदि_अन्तः
        size_t best_diff = 1ull << 36;
        पूर्णाङ्क ibest = -1;
        प्रति (पूर्णाङ्क i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर& b = बफर_pool[i];
            यदि (b.ptr != शून्यसूचकः) {
#यदि_परिभाषितम् DEBUG_sycl_MALLOC
                ++nnz;
                यदि (b.size > max_size) max_size = b.size;
#यदि_अन्तः
                यदि (b.size >= size) {
                    size_t diff = b.size - size;
                    यदि (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        यदि (!best_diff) {
                            शून्य * ptr = b.ptr;
                            *actual_आकार = b.size;
                            b.ptr = शून्यसूचकः;
                            b.size = 0;
                            प्रत्यागम ptr;
                        }
                    }
                }
            }
        }
        यदि (ibest >= 0) {
            ggml_sycl_बफर& b = बफर_pool[ibest];
            शून्य * ptr = b.ptr;
            *actual_आकार = b.size;
            b.ptr = शून्यसूचकः;
            b.size = 0;
            प्रत्यागम ptr;
        }
        शून्य * ptr;
        size_t look_ahead_आकार = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *)ggml_sycl_malloc_device(look_ahead_आकार, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        यदि (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_आकार);
            प्रत्यागम शून्यसूचकः;
        }

        *actual_आकार = look_ahead_आकार;
        pool_आकार += look_ahead_आकार;

#यदि_परिभाषितम् DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_आकार[id]/1024/1024), (uint32_t)(size/1024/1024));
#यदि_अन्तः

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        प्रत्यागम ptr;
    }

    शून्य मुक्त(शून्य * ptr, size_t size) override {
        प्रति (पूर्णाङ्क i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_बफर& b = बफर_pool[i];
            यदि (b.ptr == शून्यसूचकः) {
                b.ptr = ptr;
                b.size = size;
                प्रत्यागम;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_आकार -= size;
    }
};

// pool with virtual memory management
#यदि defined(GGML_SYCL_SUPPORT_VMM)
संरचना ggml_sycl_pool_vmm : सार्वजनिक ggml_sycl_pool {
    स्थिर ध्रुव size_t SYCL_POOL_VMM_MAX_आकार = 1ull << 35; // 32 GB

    पूर्णाङ्क           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_प्रयुक्तम् = 0;
    size_t    pool_आकार = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    संरचना mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        शून्य * map_ptr;
    };
    std::vector<mapping> mappings;

    स्पष्ट ggml_sycl_pool_vmm(queue_ptr qptr_, पूर्णाङ्क device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        यदि (pool_addr == 0) {
            प्रत्यागम;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        प्रति (स्वचालित & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_आकार, ctx)));
    }

    शून्य * alloc(size_t size, size_t * actual_आकार) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_बफर_ALIGNMENT);

        size_t avail = pool_आकार - pool_प्रयुक्तम्;

        यदि (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_आकार = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_आकार + reserve_आकार <= SYCL_POOL_VMM_MAX_आकार);

            यदि (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_चिह्नकम् " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_आकार / (1024 * 1024), pool_आकार / (1024 * 1024),
                              (pool_आकार + reserve_आकार) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_आकार)));

            // reserve virtual address space (if not already reserved)
            यदि (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_आकार, ctx)));
            }

            // map at the end of the pool
            शून्य * map_ptr = शून्यसूचकः;
            SYCL_CHECK(CHECK_TRY_ERROR(
                map_ptr = phys->map(pool_addr + pool_आकार, reserve_आकार,
                                    sycl::ext::oneapi::experimental::स्थानाङ्कः_access_बहुलकम्::read_write)));

            // stash these so we could unmap this exact range in dtor
            mappings.push_back({
                std::move(*phys),
                map_ptr,
            });

            // add to the pool
            pool_आकार += reserve_आकार;
            ggml_sycl_memtrace_सङ्कलय(GGML_SYCL_MEM_POOL_VMM, map_ptr, reserve_आकार);

#यदि_परिभाषितम् DEBUG_SYCL_MALLOC
            GGML_LOG_INFO("sycl pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
                          device, (अचिह्नित दीर्घ दीर्घ) (pool_आकार/1024/1024),
                          (अचिह्नित दीर्घ दीर्घ) (reserve_आकार/1024/1024));
#यदि_अन्तः
        }

        GGML_ASSERT(pool_addr != 0);

        शून्य * ptr = reinterpret_cast<शून्य *>(pool_addr + pool_प्रयुक्तम्);
        *actual_आकार = size;
        pool_प्रयुक्तम् += size;

#यदि_परिभाषितम् DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (अचिह्नित दीर्घ दीर्घ) size, ptr);
#यदि_अन्तः

        प्रत्यागम ptr;
    }

    शून्य मुक्त(शून्य * ptr, size_t size) override {
#यदि_परिभाषितम् DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (अचिह्नित दीर्घ दीर्घ) size, ptr);
#यदि_अन्तः

        pool_प्रयुक्तम् -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<शून्य *>(pool_addr + pool_प्रयुक्तम्));
    }
};
#यदि_अन्तः // defined(GGML_SYCL_SUPPORT_VMM)

संरचना ggml_sycl_pool_host : सार्वजनिक ggml_sycl_pool {
    queue_ptr qptr;
    पूर्णाङ्क       device;

    संरचना ggml_sycl_बफर {
        शून्य * ptr  = शून्यसूचकः;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    स्थिर constexpr पूर्णाङ्क          MAX_POOL_आकार{ 64 };
    std::vector<ggml_sycl_बफर> बफर_pool = std::vector<ggml_sycl_बफर>(MAX_POOL_आकार);
    size_t                        pool_आकार   = 0;
    पूर्णाङ्क                           गणक     = 0;

    स्पष्ट ggml_sycl_pool_host(queue_ptr qptr_, पूर्णाङ्क device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        प्रति (पूर्णाङ्क i = 0; i < MAX_POOL_आकार; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            यदि (b.ptr != शून्यसूचकः) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
                b.ptr = शून्यसूचकः;
                pool_आकार -= b.size;
                b.size = 0;
            }
        }
    }

    शून्य * alloc(size_t size, size_t * actual_आकार) override {
        यदि (गणक == MAX_POOL_आकार) {
            // the pool wrapped around, no slot is reserved for this allocation
            शून्य * ptr = शून्यसूचकः;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *) sycl::malloc_host(size, *qptr)));
            यदि (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                प्रत्यागम शून्यसूचकः;
            }
            pool_आकार    += size;
            *actual_आकार = size;
            गणक      = 0;
            प्रत्यागम ptr;
        }
        ggml_sycl_बफर & b = बफर_pool[गणक];
        ++गणक;

        यदि (b.ptr != शून्यसूचकः) {
            यदि (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                शून्य * ptr = b.ptr;
                *actual_आकार = b.size;
                b.ptr = शून्यसूचकः;
                b.size = 0;
                प्रत्यागम ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(b.ptr, *qptr)));
            pool_आकार -= b.size;
            b.ptr = शून्यसूचकः;
            b.size = 0;
        }

        शून्य * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (शून्य *) sycl::malloc_host(size, *qptr)));
        यदि (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            प्रत्यागम शून्यसूचकः;
        }
        pool_आकार    += size;
        *actual_आकार = size;
        प्रत्यागम ptr;
    }

    शून्य मुक्त(शून्य * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        प्रति (पूर्णाङ्क i = 0; i < MAX_POOL_आकार; ++i) {
            ggml_sycl_बफर & b = बफर_pool[i];
            यदि (b.ptr == शून्यसूचकः) {
                b.ptr  = ptr;
                b.size = size;
                प्रत्यागम;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::मुक्त(ptr, *qptr)));
        pool_आकार -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, पूर्णाङ्क device) {
    // return pool for the host to speed up memory management
    प्रत्यागम std::unique_ptr<ggml_sycl_pool>(नूतन ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, पूर्णाङ्क device) {
#यदि defined(GGML_SYCL_SUPPORT_VMM)
    यदि (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        प्रत्यागम std::unique_ptr<ggml_sycl_pool>(नूतन ggml_sycl_pool_vmm(qptr, device));
    }
#यदि_अन्तः // defined(GGML_SYCL_SUPPORT_VMM)
    प्रत्यागम std::unique_ptr<ggml_sycl_pool>(नूतन ggml_sycl_pool_leg(qptr, device));
}

/// kernels
प्रकारनिर्देश शून्य (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ध्रुव ggml_tensor *src0, ध्रुव ggml_tensor *src1, ggml_tensor *dst,
    ध्रुव वर्ण *src0_dd_i, ध्रुव भिन्न *src1_ddf_i, ध्रुव वर्ण *src1_ddq_i,
    भिन्न *dst_dd_i, ध्रुव int64_t row_नीचः, ध्रुव int64_t row_उच्चः,
    ध्रुव int64_t src1_ncols, ध्रुव int64_t src1_padded_row_आकार,
    ध्रुव queue_ptr &stream);



स्थिर शून्य mul_mat_p021_f16_f32(
    ध्रुव शून्य * __restrict__ vx, ध्रुव भिन्न * __restrict__ y, भिन्न * __restrict__ dst,
    ध्रुव पूर्णाङ्क ncols_x, ध्रुव पूर्णाङ्क nrows_x, ध्रुव पूर्णाङ्क nchannels_x, ध्रुव पूर्णाङ्क nchannels_y,
    ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव sycl::half *x = (ध्रुव sycl::half *)vx;

    ध्रुव पूर्णाङ्क row_x = item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                      item_ct1.get_local_id(1);
    ध्रुव पूर्णाङ्क channel = item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                        item_ct1.get_local_id(0);
    ध्रुव पूर्णाङ्क channel_x = channel / (nchannels_y / nchannels_x);

    ध्रुव पूर्णाङ्क nrows_y = ncols_x;
    ध्रुव पूर्णाङ्क nrows_dst = nrows_x;
    ध्रुव पूर्णाङ्क row_dst = row_x;

    भिन्न tmp = 0.0f;

    प्रति (पूर्णाङ्क col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        ध्रुव पूर्णाङ्क col_x = col_x0 + item_ct1.get_local_id(2);

        यदि (col_x >= ncols_x) {
            भङ्ग;
        }

        // x is transposed and permuted
        ध्रुव पूर्णाङ्क ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        ध्रुव भिन्न xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];

        ध्रुव पूर्णाङ्क row_y = col_x;


        // y is not transposed but permuted
        ध्रुव पूर्णाङ्क iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ध्रुव पूर्णाङ्क idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

स्थिर शून्य mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ध्रुव शून्य * __restrict__ vx, ध्रुव भिन्न * __restrict__ y, भिन्न * __restrict__ dst, ध्रुव पूर्णाङ्क ncols_x, ध्रुव पूर्णाङ्क nrows_x,
    ध्रुव पूर्णाङ्क row_stride_x, ध्रुव पूर्णाङ्क channel_stride_x,ध्रुव पूर्णाङ्क channel_stride_y, ध्रुव पूर्णाङ्क channel_x_भाजकः,
    ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव sycl::half *x = (ध्रुव sycl::half *)vx;

    ध्रुव पूर्णाङ्क row_x = item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                      item_ct1.get_local_id(1);
    ध्रुव पूर्णाङ्क channel = item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                        item_ct1.get_local_id(0);
    ध्रुव पूर्णाङ्क channel_x = channel / channel_x_भाजकः;

    ध्रुव पूर्णाङ्क nrows_dst = nrows_x;
    ध्रुव पूर्णाङ्क row_dst   = row_x;

    ध्रुव पूर्णाङ्क idst = channel*nrows_dst + row_dst;

    भिन्न tmp = 0.0f;

    प्रति (पूर्णाङ्क col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        ध्रुव पूर्णाङ्क col_x = col_x0 + item_ct1.get_local_id(2);

        यदि (col_x >= ncols_x) {
            भङ्ग;
        }

        ध्रुव पूर्णाङ्क row_y = col_x;

        ध्रुव पूर्णाङ्क ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        ध्रुव पूर्णाङ्क iy = channel * channel_stride_y + row_y;

        ध्रुव भिन्न xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#निर्देशः unroll
    प्रति (पूर्णाङ्क आवरणम् = WARP_आकार / 2; आवरणम् > 0; आवरणम् >>= 1) {
        tmp +=
            dpct::permute_sub_गणः_by_xor(item_ct1.get_sub_गणः(), tmp, आवरणम्);
    }

    यदि (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

स्थिर शून्य k_योग_पङ्क्तिकाः_f32(ध्रुव भिन्न * x, भिन्न * dst, ध्रुव पूर्णाङ्क ncols,
                           ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(1);
    ध्रुव पूर्णाङ्क col = item_ct1.get_local_id(2);

    भिन्न योग_2 = 0.0f;
    प्रति (पूर्णाङ्क i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        योग_2 += x[row * ncols + i];
    }

    योग_2 = warp_reduce_योग(योग_2, item_ct1);

    यदि (col == 0) {
        dst[row] = योग_2;
    }
}


प्रतिरूपम्<प्रकारनाम T>
स्थिर अन्तःस्थ शून्य ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

प्रतिरूपम् <ggml_क्रमय_क्रमः क्रमः>
__dpct_inline__ स्थिर शून्य
k_argsort_f32_i32(ध्रुव भिन्न *x, पूर्णाङ्क *dst, ध्रुव पूर्णाङ्क ncols, पूर्णाङ्क ncols_pad,
                  ध्रुव पूर्णाङ्क tasks_प्रत्येकम्_thread, ध्रुव sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    पूर्णाङ्क col_सूचकांक =  item_ct1.get_local_id(2);
    पूर्णाङ्क row = item_ct1.get_गणः(1);

    प्रति (पूर्णाङ्क i = 0; i < tasks_प्रत्येकम्_thread; i++) {
        पूर्णाङ्क col = col_सूचकांक * tasks_प्रत्येकम्_thread + i;
        यदि (col >= ncols_pad) {
            प्रत्यागम;
        }
    }

    ध्रुव भिन्न * x_row = x + row * ncols;
    स्वचालित dst_row = (पूर्णाङ्क *)dpct_local;

    // initialize indices
    प्रति (पूर्णाङ्क i=0;i<tasks_प्रत्येकम्_thread;i++){
        पूर्णाङ्क col = col_सूचकांक*tasks_प्रत्येकम्_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    प्रति (पूर्णाङ्क k = 2; k <= ncols_pad; k *= 2) {
        प्रति (पूर्णाङ्क j = k / 2; j > 0; j /= 2) {
            प्रति (पूर्णाङ्क i = 0; i < tasks_प्रत्येकम्_thread; i++) {
                पूर्णाङ्क col = col_सूचकांक * tasks_प्रत्येकम्_thread + i;
                पूर्णाङ्क ixj = col ^ j;
                यदि (ixj > col) {
                    यदि ((col & k) == 0) {
                        यदि (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (क्रमः == GGML_क्रमय_क्रमः_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } अन्यथा {
                        यदि (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols &&
                             (क्रमः == GGML_क्रमय_क्रमः_ASC
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
    प्रति (पूर्णाङ्क i = 0; i < tasks_प्रत्येकम्_thread; i++) {
        पूर्णाङ्क col = col_सूचकांक * tasks_प्रत्येकम्_thread + i;
        यदि (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

स्थिर शून्य diag_आवरणम्_inf_f32(ध्रुव भिन्न * x, भिन्न * dst, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_channel, ध्रुव पूर्णाङ्क n_past,
                              ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क col = item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                    item_ct1.get_local_id(1);
    ध्रुव पूर्णाङ्क row = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                    item_ct1.get_local_id(2);

    यदि (col >= ncols) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % पङ्क्तिकाः_प्रत्येकम्_channel) * FLT_MAX;
}

स्थिर शून्य scale_f32(ध्रुव भिन्न * x, भिन्न * dst, ध्रुव भिन्न scale, ध्रुव भिन्न bias, ध्रुव पूर्णाङ्क k,
                      ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क i = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                  item_ct1.get_local_id(2);

    यदि (i >= k) {
        प्रत्यागम;
    }

    dst[i] = scale * x[i] + bias;
}


स्थिर शून्य ggml_mul_mat_p021_f16_f32_sycl(ध्रुव शून्य *vx, ध्रुव भिन्न *y,
                                           भिन्न *dst, ध्रुव पूर्णाङ्क ncols_x,
                                           ध्रुव पूर्णाङ्क nrows_x,
                                           ध्रुव पूर्णाङ्क nchannels_x,
                                           ध्रुव पूर्णाङ्क nchannels_y,
                                           queue_ptr stream) {

    ध्रुव sycl::range<3> खण्डम्_nums(nchannels_y, nrows_x, 1);
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_p021_f16_f32(vx, y, dst, ncols_x, nrows_x, nchannels_x,
                                     nchannels_y, item_ct1);
            });
    }
}

स्थिर शून्य ggml_mul_mat_vec_nc_f16_f32_sycl(
    ध्रुव शून्य *vx, ध्रुव भिन्न *y, भिन्न *dst, ध्रुव पूर्णाङ्क ncols_x,
    ध्रुव पूर्णाङ्क nrows_x, ध्रुव पूर्णाङ्क row_stride_x, ध्रुव पूर्णाङ्क nchannels_x,
    ध्रुव पूर्णाङ्क nchannels_y, ध्रुव पूर्णाङ्क channel_stride_x, ध्रुव पूर्णाङ्क channel_stride_y, queue_ptr stream) {

    ध्रुव sycl::range<3> खण्डम्_nums(nchannels_y, nrows_x, 1);
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, WARP_आकार);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                mul_mat_vec_nc_f16_f32(vx, y, dst, ncols_x, nrows_x,
                                       row_stride_x, channel_stride_x, channel_stride_y,
                                       nchannels_y / nchannels_x, item_ct1);
            });
    }
}



स्थिर शून्य scale_f32_sycl(ध्रुव भिन्न *x, भिन्न *dst, ध्रुव भिन्न scale, ध्रुव भिन्न bias,
                           ध्रुव पूर्णाङ्क k, queue_ptr stream) {
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि = (k + SYCL_SCALE_खण्डम्_आकार - 1) / SYCL_SCALE_खण्डम्_आकार;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, अङ्कः_खण्डानि) *
                              sycl::range<3>(1, 1, SYCL_SCALE_खण्डम्_आकार),
                          sycl::range<3>(1, 1, SYCL_SCALE_खण्डम्_आकार)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


स्थिर शून्य योग_पङ्क्तिकाः_f32_sycl(ध्रुव भिन्न *x, भिन्न *dst, ध्रुव पूर्णाङ्क ncols,
                              ध्रुव पूर्णाङ्क nrows, queue_ptr stream) {
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, WARP_आकार);
    ध्रुव sycl::range<3> खण्डम्_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                                 k_योग_पङ्क्तिकाः_f32(x, dst, ncols, item_ct1);
                             });
}

स्थिर पूर्णाङ्क अग्र_power_सम्बन्धि_2(पूर्णाङ्क x) {
    पूर्णाङ्क n = 1;
    यावत् (n < x) {
        n *= 2;
    }
    प्रत्यागम n;
}

स्थिर शून्य init_argsort_indices_padded(
        पूर्णाङ्क * idx,
        ध्रुव पूर्णाङ्क nrows,
        ध्रुव पूर्णाङ्क ncols_pad,
        ध्रुव sycl::nd_item<1> & item_ct1) {
    ध्रुव size_t gid = item_ct1.get_local_range(0) * item_ct1.get_गणः(0) + item_ct1.get_local_id(0);
    ध्रुव size_t योग = (size_t) nrows * (size_t) ncols_pad;

    यदि (gid >= योग) {
        प्रत्यागम;
    }

    idx[gid] = (पूर्णाङ्क) (gid % (size_t) ncols_pad);
}

प्रतिरूपम् <ggml_क्रमय_क्रमः क्रमः>
स्थिर शून्य argsort_f32_i32_global_pass(ध्रुव भिन्न *            x,
                                        पूर्णाङ्क *                    idx,
                                        ध्रुव पूर्णाङ्क                ncols,
                                        ध्रुव पूर्णाङ्क                nrows,
                                        ध्रुव पूर्णाङ्क                ncols_pad,
                                        ध्रुव पूर्णाङ्क                j,
                                        ध्रुव पूर्णाङ्क                k,
                                        ध्रुव sycl::nd_item<1> & item_ct1) {
    ध्रुव size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_गणः(0) + item_ct1.get_local_id(0);
    ध्रुव size_t योग = (size_t) nrows * (size_t) ncols_pad;

    यदि (gid >= योग) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row = (पूर्णाङ्क) (gid / (size_t) ncols_pad);
    ध्रुव पूर्णाङ्क col = (पूर्णाङ्क) (gid % (size_t) ncols_pad);
    ध्रुव पूर्णाङ्क ixj = col ^ j;

    यदि (ixj <= col || ixj >= ncols_pad) {
        प्रत्यागम;
    }

    ध्रुव size_t आधारः  = (size_t) row * (size_t) ncols_pad;
    ध्रुव size_t pos_a = आधारः + (size_t) col;
    ध्रुव size_t pos_b = आधारः + (size_t) ixj;

    ध्रुव पूर्णाङ्क a = idx[pos_a];
    ध्रुव पूर्णाङ्क b = idx[pos_b];

    तार्किक do_swap = असत्य;

    यदि ((col & k) == 0) {
        यदि (a >= ncols ||
            (b < ncols &&
             (क्रमः == GGML_क्रमय_क्रमः_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = सत्य;
        }
    } अन्यथा {
        यदि (b >= ncols ||
            (a < ncols &&
             (क्रमः == GGML_क्रमय_क्रमः_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = सत्य;
        }
    }

    यदि (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

स्थिर शून्य copy_argsort_indices_unpadded(ध्रुव पूर्णाङ्क *              idx_padded,
                                          पूर्णाङ्क *                    dst,
                                          ध्रुव पूर्णाङ्क                nrows,
                                          ध्रुव पूर्णाङ्क                ncols,
                                          ध्रुव पूर्णाङ्क                ncols_pad,
                                          ध्रुव sycl::nd_item<1> & item_ct1) {
    ध्रुव size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_गणः(0) + item_ct1.get_local_id(0);
    ध्रुव size_t योग = (size_t) nrows * (size_t) ncols;

    यदि (gid >= योग) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क row = (पूर्णाङ्क) (gid / (size_t) ncols);
    ध्रुव पूर्णाङ्क col = (पूर्णाङ्क) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

स्थिर शून्य argsort_f32_i32_sycl(ध्रुव भिन्न *x, पूर्णाङ्क *dst, ध्रुव पूर्णाङ्क ncols,
                                 ध्रुव पूर्णाङ्क nrows, ggml_क्रमय_क्रमः क्रमः,
                                 queue_ptr stream, पूर्णाङ्क device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ध्रुव पूर्णाङ्क ncols_pad = अग्र_power_सम्बन्धि_2(ncols);
    ध्रुव size_t shared_mem = (size_t) ncols_pad * परिमाण(पूर्णाङ्क);
    ध्रुव size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    यदि (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<पूर्णाङ्क> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        पूर्णाङ्क *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t खण्डम्_आकार_3     = 256;
        ध्रुव size_t     योग_padded   = (size_t) nrows * (size_t) ncols_pad;
        ध्रुव size_t     nblocks_padded = (योग_padded + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * खण्डम्_आकार_3), sycl::range<1>(खण्डम्_आकार_3)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        प्रति (पूर्णाङ्क k = 2; k <= ncols_pad; k *= 2) {
            प्रति (पूर्णाङ्क j = k / 2; j > 0; j /= 2) {
                यदि (क्रमः == GGML_क्रमय_क्रमः_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * खण्डम्_आकार_3), sycl::range<1>(खण्डम्_आकार_3)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_क्रमय_क्रमः_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } अन्यथा यदि (क्रमः == GGML_क्रमय_क्रमः_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * खण्डम्_आकार_3), sycl::range<1>(खण्डम्_आकार_3)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_क्रमय_क्रमः_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } अन्यथा {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ध्रुव size_t योग   = (size_t) nrows * (size_t) ncols;
        ध्रुव size_t nblocks = (योग + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * खण्डम्_आकार_3), sycl::range<1>(खण्डम्_आकार_3)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        प्रत्यागम;
    }

    पूर्णाङ्क nth = 1;
    पूर्णाङ्क max_खण्डम्_आकार = ggml_sycl_info().max_work_गणः_sizes[device];
    यावत् (nth < ncols_pad && nth < max_खण्डम्_आकार)
        nth *= 2;
    यदि (nth > max_खण्डम्_आकार)
        nth = max_खण्डम्_आकार;

    ध्रुव पूर्णाङ्क tasks_प्रत्येकम्_thread = ncols_pad / nth;

    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, nth);
    ध्रुव sycl::range<3> खण्डम्_nums(1, nrows, 1);

    यदि (क्रमः == GGML_क्रमय_क्रमः_ASC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_क्रमय_क्रमः_ASC>(
                        x, dst, ncols, ncols_pad, tasks_प्रत्येकम्_thread, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                });
        });
    } अन्यथा यदि (क्रमः == GGML_क्रमय_क्रमः_DESC) {
        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(shared_mem), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1) {
                    k_argsort_f32_i32<GGML_क्रमय_क्रमः_DESC>(
                        x, dst, ncols, ncols_pad, tasks_प्रत्येकम्_thread, item_ct1,
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
स्थिर शून्य शिखरम्_k_अवलोकनम्_merge_f32(
    ध्रुव भिन्न *   src_vals,
    ध्रुव int32_t * src_map,
    ध्रुव पूर्णाङ्क       begin,
    ध्रुव पूर्णाङ्क       end,
    ध्रुव पूर्णाङ्क       k,
    ध्रुव पूर्णाङ्क       खण्डम्_आकार_3,
    भिन्न *         shared_vals,
    पूर्णाङ्क *           shared_idx,
    भिन्न *         बहिर्गमनम्_vals,
    int32_t *       बहिर्गमनम्_idx,
    ध्रुव तार्किक      swap01,
    ध्रुव sycl::nd_item<1> & item_ct1
) {
    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    भिन्न * lv = shared_vals + tid;
    पूर्णाङ्क * li = shared_idx + tid;

    प्रति (पूर्णाङ्क i = 0; i < k; i++) {
        lv[i * खण्डम्_आकार_3] = -FLT_MAX;
        li[i * खण्डम्_आकार_3] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    भिन्न kth = -FLT_MAX;

    प्रति (पूर्णाङ्क col = begin + tid; col < end; col += खण्डम्_आकार_3) {
        भिन्न val = src_vals[col];

        यदि (val > kth) {
            पूर्णाङ्क pos = k - 1;
            यावत् (pos > 0 && val > lv[(pos - 1) * खण्डम्_आकार_3]) {
                pos--;
            }

            प्रति (पूर्णाङ्क i = k - 1; i > pos; i--) {
                lv[i * खण्डम्_आकार_3] = lv[(i - 1) * खण्डम्_आकार_3];
                li[i * खण्डम्_आकार_3] = li[(i - 1) * खण्डम्_आकार_3];
            }
            lv[pos * खण्डम्_आकार_3] = val;
            li[pos * खण्डम्_आकार_3] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * खण्डम्_आकार_3];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    यदि (tid != 0) {
        प्रत्यागम;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    भिन्न * fv = shared_vals + (size_t) k * खण्डम्_आकार_3;
    पूर्णाङ्क * fi = shared_idx + (size_t) k * खण्डम्_आकार_3;

    प्रति (पूर्णाङ्क i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    भिन्न fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    प्रति (पूर्णाङ्क t = 0; t < खण्डम्_आकार_3; t++) {
        प्रति (पूर्णाङ्क i = 0; i < k; i++) {
            भिन्न val = shared_vals[i * खण्डम्_आकार_3 + t];

            यदि (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                भङ्ग;
            }

            पूर्णाङ्क idx = shared_idx[i * खण्डम्_आकार_3 + t];

            पूर्णाङ्क pos = k - 1;
            यावत् (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            प्रति (पूर्णाङ्क j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    यदि (बहिर्गमनम्_vals) {
        प्रति (पूर्णाङ्क i = 0; i < k; i++) {
            बहिर्गमनम्_vals[i] = fv[i];
        }
    }

    प्रति (पूर्णाङ्क i = 0; i < k; i++) {
        बहिर्गमनम्_idx[i] = fi[i];
    }

    यदि (swap01 && k > 1) {
        int32_t अस्थायी = बहिर्गमनम्_idx[0];
        बहिर्गमनम्_idx[0] = बहिर्गमनम्_idx[1];
        बहिर्गमनम्_idx[1] = अस्थायी;
    }
}

स्थिर शून्य शिखरम्_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ध्रुव भिन्न * src,
    int32_t * dst_indices,
    ध्रुव int64_t ncols,
    ध्रुव int64_t nrows,
    ध्रुव पूर्णाङ्क k,
    dpct::queue_ptr main_प्रवाहः
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
    constexpr पूर्णाङ्क विभज_खण्डम् = 128;
    constexpr पूर्णाङ्क max_splits  = 128;
    constexpr पूर्णाङ्क min_स्तम्भसङ्ख्या    = 8192;

    पूर्णाङ्क nsplit = 1;
    यदि (ncols >= min_स्तम्भसङ्ख्या) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ध्रुव int64_t want = ncols / विभज_खण्डम्;
        nsplit = (पूर्णाङ्क) (want > max_splits ? max_splits : want);
    }

    यदि (nsplit > 1) {
        ध्रुव पूर्णाङ्क nchunk = (पूर्णाङ्क) ((ncols + nsplit - 1) / nsplit);
        ध्रुव size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<भिन्न>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        भिन्न *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ध्रुव sycl::range<1> खण्डम्_dims(विभज_खण्डम्);

        main_प्रवाहः->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<भिन्न, 1> shared_vals(sycl::range<1>((विभज_खण्डम् + 1) * k), cgh);
            sycl::local_accessor<पूर्णाङ्क, 1> shared_idx(sycl::range<1>((विभज_खण्डम् + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    ध्रुव पूर्णाङ्क grp = item_ct1.get_गणः(0);
                    ध्रुव पूर्णाङ्क row = grp / nsplit;
                    ध्रुव पूर्णाङ्क part = grp % nsplit;

                    ध्रुव पूर्णाङ्क begin = part * nchunk;
                    पूर्णाङ्क end = begin + nchunk;
                    यदि (end > (पूर्णाङ्क) ncols) {
                        end = (पूर्णाङ्क) ncols;
                    }

                    शिखरम्_k_अवलोकनम्_merge_f32(
                        src + (int64_t) row * ncols, शून्यसूचकः, begin, end, k, विभज_खण्डम्,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, असत्य, item_ct1);
                });
        });

        main_प्रवाहः->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<भिन्न, 1> shared_vals(sycl::range<1>((विभज_खण्डम् + 1) * k), cgh);
            sycl::local_accessor<पूर्णाङ्क, 1> shared_idx(sycl::range<1>((विभज_खण्डम् + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(0);
                    ध्रुव size_t off = (size_t) row * nsplit * k;

                    शिखरम्_k_अवलोकनम्_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, विभज_खण्डम्,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        शून्यसूचकः, dst_indices + (int64_t) row * k, सत्य, item_ct1);
                });
        });

        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क खण्डम्_आकार_3 = 128;

    ध्रुव sycl::range<1> खण्डम्_dims(खण्डम्_आकार_3);
    ध्रुव sycl::range<1> grid_dims(nrows);

    main_प्रवाहः->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<भिन्न, 1> shared_vals(sycl::range<1>((खण्डम्_आकार_3 + 1) * k), cgh);
        sycl::local_accessor<पूर्णाङ्क, 1> shared_idx(sycl::range<1>((खण्डम्_आकार_3 + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<1> item_ct1) {
                ध्रुव पूर्णाङ्क row = item_ct1.get_गणः(0);

                यदि (row >= nrows) प्रत्यागम;

                शिखरम्_k_अवलोकनम्_merge_f32(
                    src + (int64_t) row * ncols, शून्यसूचकः, 0, (पूर्णाङ्क) ncols, k, खण्डम्_आकार_3,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    शून्यसूचकः, dst_indices + (int64_t) row * k, सत्य, item_ct1);
            });
    });
}

स्थिर शून्य argmax_f32_i32_sycl(ध्रुव भिन्न *x, पूर्णाङ्क *dst, ध्रुव पूर्णाङ्क ncols,
                               ध्रुव पूर्णाङ्क nrows, queue_ptr stream) {
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, SYCL_ARGMAX_खण्डम्_आकार);
    ध्रुव sycl::range<3> खण्डम्_nums(1, nrows, 1);
    ध्रुव size_t shared_mem = 256 * परिमाण(भिन्न);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<भिन्न, 1> shared_दत्त(
            sycl::range<1>(shared_mem/परिमाण(भिन्न)), cgh);
        sycl::local_accessor<पूर्णाङ्क, 1> shared_indices(
            sycl::range<1>(shared_mem/परिमाण(भिन्न)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);
                ध्रुव पूर्णाङ्क row = item_ct1.get_global_id(1);

                भिन्न max_val = -INFINITY;
                पूर्णाङ्क max_idx = -1;

                प्रति (पूर्णाङ्क col = tid; col < ncols; col += 256) {
                    भिन्न val = x[row * ncols + col];
                    यदि (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_दत्त[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                प्रति (पूर्णाङ्क stride = 256/2; stride > 0; stride >>= 1) {
                    यदि (tid < stride) {
                        भिन्न val1 = shared_दत्त[tid];
                        भिन्न val2 = shared_दत्त[tid + stride];
                        यदि (val2 > val1) {
                            shared_दत्त[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                यदि (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
स्थिर शून्य diag_आवरणम्_inf_f32_sycl(ध्रुव भिन्न *x, भिन्न *dst,
                                   ध्रुव पूर्णाङ्क ncols_x, ध्रुव पूर्णाङ्क nrows_x,
                                   ध्रुव पूर्णाङ्क पङ्क्तिकाः_प्रत्येकम्_channel, ध्रुव पूर्णाङ्क n_past,
                                   queue_ptr stream) {
    ध्रुव sycl::range<3> खण्डम्_dims(1, SYCL_DIAG_आवरणम्_INF_खण्डम्_आकार, 1);
    ध्रुव पूर्णाङ्क खण्डम्_अङ्कः_x = (ncols_x + SYCL_DIAG_आवरणम्_INF_खण्डम्_आकार - 1) / SYCL_DIAG_आवरणम्_INF_खण्डम्_आकार;
    ध्रुव sycl::range<3> खण्डम्_nums(1, खण्डम्_अङ्कः_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_आवरणम्_inf_f32(x, dst, ncols_x,
                                               पङ्क्तिकाः_प्रत्येकम्_channel, n_past,
                                               item_ct1);
                         });
}

स्थिर dpct::err0 ggml_sycl_cpy_tensor_2d(शून्य *dst,
                                          ध्रुव संरचना ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_नीचः, int64_t i1_उच्चः,
                                          queue_ptr stream) यत्न {

    dpct::memcpy_direction विधा;
    वर्ण * src_ptr;
    यदि (ggml_backend_बफर_is_host(src->बफर)) {
        विधा = dpct::host_गन्तव्यम्_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (वर्ण *) src->दत्त;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } अन्यथा यदि (ggml_backend_बफर_is_sycl(src->बफर)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        विधा    = dpct::device_गन्तव्यम्_device;
        src_ptr = (वर्ण *) src->दत्त;
    } अन्यथा यदि (ggml_backend_बफर_is_sycl_विभज(src->बफर)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_नीचः == 0 && i1_उच्चः == src->ne[1]);
        विधा = dpct::device_गन्तव्यम्_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        पूर्णाङ्क id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (वर्ण *) extra->दत्त_device[id];
    } अन्यथा {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    वर्ण * dst_ptr = (वर्ण *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ध्रुव गणना ggml_type type = src->type;
    ध्रुव int64_t ts = ggml_type_आकार(type);
    ध्रुव int64_t bs = ggml_blck_आकार(type);
    int64_t i1_diff = i1_उच्चः - i1_नीचः;

    ध्रुव वर्ण * x = src_ptr + i1_नीचः*nb1 + i2*nb2 + i3*nb3;
    यदि (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        प्रत्यागम CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    विधा, *stream));

    } अन्यथा यदि (nb0 == ts) {
        प्रत्यागम CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, विधा, *stream));
    } अन्यथा {
        प्रति (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ध्रुव शून्य * rx = (ध्रुव शून्य *) ((ध्रुव वर्ण *) x + i1*nb1);
            शून्य * rd = (शून्य *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            dpct::err0 r = CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                rd, ts / bs, rx, nb0, ts / bs, ne0, विधा, *stream));
            /*
            DPCT1001:85: The statement could not be removed.
            */
            /*
            DPCT1000:86: Error handling if-stmt was detected but could not be
            rewritten.
            */
            यदि (r != 0) प्रत्यागम r;
        }
        प्रत्यागम 0;
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

अन्तःस्थ शून्य ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ध्रुव ggml_tensor *src0, ध्रुव ggml_tensor *src1, ggml_tensor *dst,
    ध्रुव वर्ण *src0_dd_i, ध्रुव भिन्न *src1_ddf_i, ध्रुव वर्ण *src1_ddq_i,
    भिन्न *dst_dd_i, ध्रुव int64_t row_नीचः, ध्रुव int64_t row_उच्चः,
    ध्रुव int64_t src1_ncols, ध्रुव int64_t src1_padded_row_आकार,
    ध्रुव queue_ptr &stream) यत्न {

    GGML_ASSERT(src0_dd_i  != शून्यसूचकः);
    GGML_ASSERT(src1_ddf_i != शून्यसूचकः);
    GGML_ASSERT(dst_dd_i   != शून्यसूचकः);

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ध्रुव int64_t row_diff = row_उच्चः - row_नीचः;

    पूर्णाङ्क id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ध्रुव int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    पूर्णाङ्क ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#यदि_परिभाषितम् GGML_SYCL_F16
    तार्किक use_fp16 = सत्य;  // TODO(Yu) SYCL capability check
#अन्यथा
    तार्किक use_fp16 = असत्य;
#यदि_अन्तः

#यदि GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    यदि (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        उपयुज्य bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        यदि (src1->type != GGML_TYPE_BF16) {
            ध्रुव गन्तव्यम्_bf16_sycl_t गन्तव्यम्_bf16_sycl = ggml_get_गन्तव्यम्_bf16_sycl(src1->type, dst);
            GGML_ASSERT(गन्तव्यम्_bf16_sycl != शून्यसूचकः);
            गन्तव्यम्_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } अन्यथा {
            stream->स्मृतिं_प्रतिलिख(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*परिमाण(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::गन्तव्यम्_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::गन्तव्यम्_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(), stream);
        GGML_अप्रयुक्तम्(dst);
        GGML_अप्रयुक्तम्(src1_ddq_i);
        GGML_अप्रयुक्तम्(src1_padded_row_आकार);
        प्रत्यागम;
    }
#यदि_अन्तः

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    तार्किक src0_reordered = असत्य;
    यदि (ध्रुव स्वचालित * src0_extra = static_cast<ध्रुव ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    यदि (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        प्रत्यागम;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    यदि ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        यदि (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ध्रुव गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16_sycl = ggml_get_गन्तव्यम्_fp16_sycl(src1->type, dst);
            GGML_ASSERT(गन्तव्यम्_fp16_sycl != शून्यसूचकः);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            गन्तव्यम्_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ध्रुव sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ध्रुव sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        यदि (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ध्रुव गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16_sycl = ggml_get_गन्तव्यम्_fp16_sycl(src0->type, dst);
            GGML_ASSERT(गन्तव्यम्_fp16_sycl != शून्यसूचकः);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            गन्तव्यम्_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ध्रुव sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ध्रुव sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#यदि GGML_SYCL_DNNL
        यदि (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(), stream);
        }
        अन्यथा
#यदि_अन्तः
        {
            ध्रुव भिन्न alpha = 1.0f;
            ध्रुव भिन्न beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_दत्त_t::real_half, ne00,
                src1_ptr, dpct::library_दत्त_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_दत्त_t::real_float, ldc,
                dpct::library_दत्त_t::real_float)));
        }
    } अन्यथा {
        ggml_sycl_pool_alloc<भिन्न> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<भिन्न> src1_ddq_as_f32(ctx.pool());
        यदि (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ध्रुव गन्तव्यम्_fp32_sycl_t गन्तव्यम्_fp32_sycl = ggml_get_गन्तव्यम्_fp32_sycl(src0->type, dst);
            GGML_ASSERT(गन्तव्यम्_fp32_sycl != शून्यसूचकः);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            गन्तव्यम्_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        यदि (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ध्रुव गन्तव्यम्_fp32_sycl_t गन्तव्यम्_fp32_sycl = ggml_get_गन्तव्यम्_fp32_sycl(src1->type, dst);
            GGML_ASSERT(गन्तव्यम्_fp32_sycl != शून्यसूचकः);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            गन्तव्यम्_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ध्रुव भिन्न * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ध्रुव भिन्न *) src0_dd_i : src0_ddq_as_f32.get();
        ध्रुव भिन्न * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ध्रुव भिन्न *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#यदि GGML_SYCL_DNNL
            ध्रुव int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            ध्रुव तार्किक use_mkl_प्रत्यक्षम् = gemm_flops < 256 * 256 * 256;
            यदि (g_ggml_sycl_enable_dnn && !use_mkl_प्रत्यक्षम् && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(), src1_ddf1_i, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(),
                                          dst_dd_i, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(), stream);
            }
            अन्यथा
#यदि_अन्तः
            {
                ध्रुव भिन्न alpha = 1.0f;
                ध्रुव भिन्न beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_मानम्(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_मानम्(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_अप्रयुक्तम्(dst);
    GGML_अप्रयुक्तम्(src1_ddq_i);
    GGML_अप्रयुक्तम्(src1_padded_row_आकार);
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

अन्तःस्थ शून्य ggml_sycl_op_योग(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    योग_पङ्क्तिकाः_f32_sycl(src0_dd, dst_dd, ne, 1, main_प्रवाहः);
}

अन्तःस्थ शून्य ggml_sycl_op_योग_पङ्क्तिकाः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव int64_t ncols = dst->src[0]->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    योग_पङ्क्तिकाः_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_प्रवाहः);
}

अन्तःस्थ शून्य ggml_sycl_op_माध्यम्(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव int64_t ncols = dst->src[0]->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    योग_पङ्क्तिकाः_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_प्रवाहः);

    main_प्रवाहः->parallel_for(
        sycl::range<1>(nrows),
        [=](sycl::id<1> row) {
            dst_dd[row] /= ncols;
        }
    );
}


अन्तःस्थ शून्य ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->दत्त);


    ध्रुव int64_t ncols = dst->src[0]->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(dst->src[0]);

    गणना ggml_क्रमय_क्रमः क्रमः = (गणना ggml_क्रमय_क्रमः) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (पूर्णाङ्क *)dst_dd, ncols, nrows, क्रमः,
                         main_प्रवाहः, ctx.device, ctx.pool());
}

स्थिर शून्य ggml_sycl_op_शिखरम्_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(src0->दत्त);
    int32_t * dst_dd = static_cast<int32_t *>(dst->दत्त);

    ध्रुव पूर्णाङ्क k = dst->ne[0];
    ध्रुव int64_t ncols = src0->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    यदि (k <= SYCL_शिखरम्_K_अवलोकनम्_MERGE_MAX_K) {
        शिखरम्_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_प्रवाहः);
    } अन्यथा {
        ggml_sycl_शिखरम्_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_प्रवाहः);
    }
}

अन्तःस्थ शून्य ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->दत्त);

    ध्रुव int64_t ncols = dst->src[0]->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_प्रवाहः);
}

अन्तःस्थ शून्य ggml_sycl_op_diag_आवरणम्_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव int64_t ne00 = dst->src[0]->ne[0];
    ध्रुव int64_t ne01 = dst->src[0]->ne[1];
    ध्रुव पूर्णाङ्क nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ध्रुव पूर्णाङ्क n_past = ((int32_t *) dst->op_params)[0];

    diag_आवरणम्_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_प्रवाहः);
}

स्थिर शून्य tri_f32_sycl(
    ध्रुव भिन्न * src,
    भिन्न * dst,
    ध्रुव int64_t ne0,
    ध्रुव int64_t ne1,
    ध्रुव int64_t ne2,
    ध्रुव int64_t ne3,
    ध्रुव ggml_tri_type ttype,
    dpct::queue_ptr main_प्रवाहः
) {
    ध्रुव size_t योग = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_प्रवाहः->parallel_for(sycl::range<1>(योग), [=](sycl::id<1> tid) {
        ध्रुव int64_t idx = (int64_t) tid[0];

        ध्रुव int64_t i0 = idx % ne0;
        ध्रुव int64_t t1 = idx / ne0;
        ध्रुव int64_t i1 = t1 % ne1;

        तार्किक keep = असत्य;
        विकल्प (ttype) {
            पक्ष GGML_TRI_TYPE_अधस्तनम्:      keep = (i0 <  i1); भङ्ग;
            पक्ष GGML_TRI_TYPE_अधस्तनम्_DIAG: keep = (i0 <= i1); भङ्ग;
            पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्:      keep = (i0 >  i1); भङ्ग;
            पक्ष GGML_TRI_TYPE_ऊर्ध्वस्थम्_DIAG: keep = (i0 >= i1); भङ्ग;
            स्वभाव: keep = असत्य; भङ्ग;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

स्थिर शून्य ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(src0->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ध्रुव int64_t ne0 = src0->ne[0];
    ध्रुव int64_t ne1 = src0->ne[1];
    ध्रुव int64_t ne2 = src0->ne[2];
    ध्रुव int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_प्रवाहः);
}


अन्तःस्थ शून्य ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    ध्रुव भिन्न * src0_dd = static_cast<ध्रुव भिन्न *>(dst->src[0]->दत्त);
    भिन्न *       dst_dd  = static_cast<भिन्न *>(dst->दत्त);

    भिन्न scale;
    भिन्न bias;
    स्मृतिं_प्रतिलिख(&scale, (भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&bias,  (भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_प्रवाहः);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

स्थिर शून्य ggml_sycl_समुच्चयः_peer_access(ध्रुव पूर्णाङ्क n_tokens, [[maybe_unused]] पूर्णाङ्क main_device) {
    स्थिर तार्किक peer_access_enabled = असत्य;

    ध्रुव तार्किक enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_आकार;

    यदि (peer_access_enabled == enable_peer_access) {
        प्रत्यागम;
    }

#यदि_परिभाषितम् NDEBUG
    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        SYCL_CHECK(ggml_sycl_समुच्चयः_device(i));
    }

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        SYCL_CHECK(ggml_sycl_समुच्चयः_device(i));

        प्रति (पूर्णाङ्क id_other = 0; id_other < ggml_sycl_info().device_गणनम्; ++id_other) {
            यदि (i == id_other) {
                अनुवर्त;
            }
            यदि (i != main_device && id_other != main_device) {
                अनुवर्त;
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
#यदि_अन्तः // NDEBUG

    peer_access_enabled = enable_peer_access;
}

प्रतिरूपम् <प्रतिरूपम् <पूर्णाङ्क> प्रकारनाम quantize_f>
स्थिर शून्य ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor *src0,
                                 ध्रुव ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) यत्न {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ध्रुव int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];

    ध्रुव पूर्णाङ्क nb2 = dst->nb[2];
    ध्रुव पूर्णाङ्क nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(dst->बफर));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(src1->बफर));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ध्रुव int64_t i02_भाजकः = ne12 / ne02;

    ध्रुव size_t src0_ts = ggml_type_आकार(src0->type);
    ध्रुव size_t src0_bs = ggml_blck_आकार(src0->type);
    ध्रुव size_t q8_1_ts = परिमाण(खण्डम्_q8_1);
    ध्रुव size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ध्रुव तार्किक src0_is_contiguous = ggml_is_contiguous(src0);
    ध्रुव तार्किक src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_आकार = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    ध्रुव तार्किक विभज = ggml_backend_बफर_is_sycl_विभज(src0->बफर);
    GGML_ASSERT(!(विभज && ne02 > 1));
    GGML_ASSERT(!(विभज && ne03 > 1));
    GGML_ASSERT(!(विभज && ne02 < ne12));

    std::array<भिन्न, GGML_SYCL_MAX_DEVICES> tensor_विभज;
    यदि (विभज) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_विभज_बफर_type_context * buft_ctx = (ggml_backend_sycl_विभज_बफर_type_context *) src0->बफर->buft->context;
        tensor_विभज = buft_ctx->tensor_विभज;
    }

    संरचना dev_दत्त {
        ggml_sycl_pool_alloc<वर्ण> src0_dd_alloc;
        ggml_sycl_pool_alloc<भिन्न> src1_ddf_alloc;
        ggml_sycl_pool_alloc<वर्ण> src1_ddq_alloc;
        ggml_sycl_pool_alloc<भिन्न> dst_dd_alloc;

        वर्ण *src0_dd = शून्यसूचकः;
        भिन्न *src1_ddf = शून्यसूचकः; // float
        वर्ण *src1_ddq = शून्यसूचकः;  // q8_1
        भिन्न *dst_dd = शून्यसूचकः;

        int64_t row_नीचः;
        int64_t row_उच्चः;
    };

    dev_दत्त dev[GGML_SYCL_MAX_DEVICES];

    पूर्णाङ्क प्रयुक्तम्_devices = 0;
    queue_ptr main_प्रवाहः = ctx.stream();

    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        // by default, use all rows
        dev[i].row_नीचः  = 0;
        dev[i].row_उच्चः = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        यदि (विभज) {
            ध्रुव int64_t rounding = get_row_rounding(src0->type, tensor_विभज);

            यदि (i != 0) {
                dev[i].row_नीचः  = ne01*tensor_विभज[i];
                यदि (dev[i].row_नीचः < ne01) {
                    dev[i].row_नीचः -= dev[i].row_नीचः % rounding;
                }
            }

            यदि (i != ggml_sycl_info().device_गणनम् - 1) {
                dev[i].row_उच्चः  = ne01*tensor_विभज[i + 1];
                यदि (dev[i].row_उच्चः < ne01) {
                    dev[i].row_उच्चः -= dev[i].row_उच्चः % rounding;
                }
            }
        }
    }

    constexpr तार्किक quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_आकार>,
                                                      no_quantize_q8_1<QK8_1 / WARP_आकार>>;
    प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
        यदि ((!विभज && i != ctx.device) || dev[i].row_नीचः == dev[i].row_उच्चः) {
            अनुवर्त;
        }

        प्रयुक्तम्_devices++;

        ध्रुव तार्किक src1_on_device = i == ctx.device;
        ध्रुव तार्किक  dst_on_device = i == ctx.device;

        ggml_sycl_समुच्चयः_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        यदि (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = विभज ? (वर्ण *) src0_extra->दत्त_device[i] : (वर्ण *) src0->दत्त;
        } अन्यथा {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        यदि (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (भिन्न *) src1->दत्त;
        } अन्यथा {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        यदि constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_आकार*q8_1_ts/q8_1_bs);

            यदि (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                यत्न {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_आकार, stream);
                } गृहाण (sycl::exception ध्रुव &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
                    std::निर्गच्छ(1);
                }
            }
        }

        यदि (dst_on_device) {
            dev[i].dst_dd = (भिन्न *) dst->दत्त;
        } अन्यथा {
            ध्रुव size_t आकार_dst_ddf = विभज ? (dev[i].row_उच्चः - dev[i].row_नीचः)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), आकार_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    यदि (विभज && प्रयुक्तम्_devices > 1) {
        ggml_sycl_समुच्चयः_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ध्रुव int64_t src1_col_stride = विभज && प्रयुक्तम्_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    प्रति (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        ध्रुव int64_t is = विभज ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ध्रुव int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
            यदि ((!विभज && i != ctx.device) || dev[i].row_नीचः == dev[i].row_उच्चः) {
                अनुवर्त;
            }

            ध्रुव तार्किक src1_on_device = i == ctx.device;
            ध्रुव तार्किक  dst_on_device = i == ctx.device;
            ध्रुव int64_t row_diff = dev[i].row_उच्चः - dev[i].row_नीचः;

            ggml_sycl_समुच्चयः_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            यदि (विभज && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            प्रति (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ध्रुव int64_t i03 = i0 / ne12;
                ध्रुव int64_t i02 = i0 % ne12;

                ध्रुव size_t src1_ddq_i_विस्थापनम् = (i0*ne11 + src1_col_0) * src1_padded_col_आकार*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                वर्ण  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_भाजकः) * (ne01*ne00*src0_ts)/src0_bs;
                भिन्न * src1_ddf_i = (भिन्न *) ((वर्ण *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_आकार(src1->type));
                वर्ण  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_विस्थापनम्;
                भिन्न *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                यदि (i == ctx.device) {
                    dst_dd_i += dev[i].row_नीचः; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                यदि (src1_is_contiguous) {
                    यदि (i != ctx.device) {
                        यदि constexpr (quantize_enabled) {
                            वर्ण * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_विस्थापनम्;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->स्मृतिं_प्रतिलिख(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_आकार * q8_1_ts / q8_1_bs)
                                                    .प्रतीक्षस्व()));
                        } अन्यथा {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ध्रुव वर्ण * src1_ddf_i_source = (ध्रुव वर्ण *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_आकार(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_प्रवाहः, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_आकार(src1->type))));
                        }
                    }
                } अन्यथा {
                    यदि (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } अन्यथा {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    यदि constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        यत्न {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_आकार, stream);
                        } गृहाण (ध्रुव sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
                            std::निर्गच्छ(1);
                        }
                    }
                }

                यदि (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_भाजकः == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_भाजकः, dev[i].row_नीचः, dev[i].row_उच्चः, stream));
                }
                यदि (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_आकार = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_नीचः, dev[i].row_उच्चः, src1_ncols, src1_padded_col_आकार, stream)));

                // copy dst to host or other device if necessary
                यदि (!dst_on_device) {
                    शून्य * dst_off_device = dst->दत्त;
                    यदि (विभज) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        भिन्न * dhf_dst_i = (भिन्न *) ((वर्ण *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*परिमाण(भिन्न));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_नीचः;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * परिमाण(भिन्न), dst_dd_i,
                            row_diff * परिमाण(भिन्न), row_diff * परिमाण(भिन्न),
                            src1_ncols, dpct::device_गन्तव्यम्_device, *stream)));
                    } अन्यथा {
                        भिन्न * dhf_dst_i = (भिन्न *) ((वर्ण *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*परिमाण(भिन्न));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->स्मृतिं_प्रतिलिख(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * परिमाण(भिन्न)).प्रतीक्षस्व()));
                    }
                }

                // add event for the main device to wait on until other device is done
                यदि (विभज && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    यदि (विभज && ggml_sycl_info().device_गणनम् > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_समुच्चयः_device(ctx.device);
        प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; ++i) {
            यदि (dev[i].row_नीचः == dev[i].row_उच्चः) {
                अनुवर्त;
            }
            प्रति (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_sycl_repeat_पृष्ठभागः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_पृष्ठभागः(ctx, dst);
}

स्थिर शून्य ggml_sycl_get_पङ्क्तिकाः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_पङ्क्तिकाः(ctx, dst);
}

स्थिर शून्य ggml_sycl_get_पङ्क्तिकाः_पृष्ठभागः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_पङ्क्तिकाः_पृष्ठभागः(ctx, dst);
}

स्थिर शून्य ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_rms_norm_पृष्ठभागः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_पृष्ठभागः(ctx, dst);
}

स्थिर शून्य ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_गणः_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_गणः_norm(ctx, dst);
}

स्थिर शून्य ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor *src0,
                                       ध्रुव ggml_tensor *src1,
                                       ggml_tensor *dst) यत्न {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(src0->बफर));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];

    ध्रुव int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    queue_ptr main_प्रवाहः = ctx.stream();

    शून्य  * src0_ddq = src0->दत्त;
    भिन्न * src1_ddf = (भिन्न *) src1->दत्त;
    भिन्न * dst_ddf  = (भिन्न *) dst->दत्त;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_प्रवाहः);
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor *src0,
                                     ध्रुव ggml_tensor *src1,
                                     ggml_tensor *dst) यत्न {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(src0->बफर));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    ध्रुव int64_t ne00 = src0->ne[0];
    ध्रुव int64_t ne01 = src0->ne[1];
    ध्रुव int64_t ne02 = src0->ne[2];

    ध्रुव int64_t nb01 = src0->nb[1];
    ध्रुव int64_t nb02 = src0->nb[2];

    ध्रुव int64_t ne12 = src1->ne[2];
    ध्रुव int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    queue_ptr main_प्रवाहः = ctx.stream();

    शून्य  * src0_ddq = src0->दत्त;
    भिन्न * src1_ddf = (भिन्न *) src1->दत्त;
    भिन्न * dst_ddf  = (भिन्न *) dst->दत्त;

    ध्रुव int64_t row_stride_x = nb01 / परिमाण(sycl::half);
    ध्रुव int64_t channel_stride_x = nb02 / परिमाण(sycl::half);
    ध्रुव int64_t channel_stride_y = nb11 / परिमाण(भिन्न);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_प्रवाहः);
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य k_compute_batched_ptrs(ध्रुव sycl::half * src0_as_f16, ध्रुव sycl::half * src1_as_f16, शून्य * dst,
                                   ध्रुव शून्य ** ptrs_src, शून्य ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव int64_t i13 = item_ct1.get_गणः(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    ध्रुव int64_t i12 = item_ct1.get_गणः(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    यदि (i13 >= ne13 || i12 >= ne12) {
        प्रत्यागम;
    }

    ध्रुव int64_t i03 = i13 / r3;
    ध्रुव int64_t i02 = i12 / r2;

    ध्रुव uint8_t * src0_अष्टकानि = reinterpret_cast<ध्रुव uint8_t *>(src0_as_f16);
    ध्रुव uint8_t * src1_अष्टकानि = reinterpret_cast<ध्रुव uint8_t *>(src1_as_f16);
    uint8_t *       dst_अष्टकानि  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_अष्टकानि + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_अष्टकानि + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_अष्टकानि + i12 * nbd2 + i13 * nbd3;
}

स्थिर शून्य ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0,
                                           ध्रुव ggml_tensor * src1, ggml_tensor * dst) यत्न {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(src0->बफर));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_द्विमान_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ध्रुव sycl::half * src0_f16 = static_cast<ध्रुव sycl::half *>(src0->दत्त);
    भिन्न *            dst_ddf  = static_cast<भिन्न *>(dst->दत्त);

    ध्रुव sycl::half * src1_f16       = static_cast<ध्रुव sycl::half *>(src1->दत्त);
#यदि GGML_SYCL_DNNL
    ध्रुव size_t       type_आकार_src0 = ggml_type_आकार(src0->type);
#यदि_अन्तः
    ध्रुव size_t       type_आकार_src1 = ggml_type_आकार(src1->type);

    तार्किक is_src0_cont_2 = ggml_is_contiguous_2(src0);
    तार्किक is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_आकार_src1;
    int64_t                          s12 = nb12 / type_आकार_src1;
    int64_t                          s13 = nb13 / type_आकार_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    यदि (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#यदि GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        पूर्णाङ्क चरमः_dim=0;
        पूर्णाङ्क चरमः_str=0;
        size_t largest_str=0;
        प्रति(पूर्णाङ्क i = 0; i< 4; i++){
            // last stride is always the largest
            यदि(src1->nb[i] == largest_str){
                यदि(src1->ne[चरमः_dim] == 1){
                    चरमः_str = i;
                    चरमः_dim = i;
                }
            }
            यदि(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                चरमः_str = i;
                चरमः_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ध्रुव int64_t ne_src1 = src1->nb[चरमः_str] * src1->ne[चरमः_dim] / type_आकार_src1;
        src1_f16_alloc.alloc(ne_src1);
        ध्रुव गन्तव्यम्_fp16_sycl_t गन्तव्यम्_fp16_sycl = ggml_get_गन्तव्यम्_fp16_sycl(src1->type, dst);
        GGML_ASSERT(गन्तव्यम्_fp16_sycl != शून्यसूचकः);
        गन्तव्यम्_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# अन्यथा
        ध्रुव int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ध्रुव गन्तव्यम्_fp16_nc_sycl_t गन्तव्यम्_fp16_nc_sycl = ggml_get_गन्तव्यम्_fp16_nc_sycl(src1->type);
        GGML_ASSERT(गन्तव्यम्_fp16_nc_sycl != शून्यसूचकः);
        गन्तव्यम्_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#यदि_अन्तः

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = सत्य;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_दत्त_t mkl_compute_type = dpct::library_दत्त_t::real_float;
    dpct::library_दत्त_t mkl_दत्त_type    = dpct::library_दत्त_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ध्रुव भिन्न alpha_f32 = 1.0f;
    ध्रुव भिन्न beta_f32  = 0.0f;

    ध्रुव शून्य * alpha = &alpha_f32;
    ध्रुव शून्य * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

#यदि GGML_SYCL_DNNL
    यदि (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_आकार_src0;
            int64_t str_a1 = nb01 / type_आकार_src0;
            int64_t str_a2 = nb02 / type_आकार_src0;

            int64_t str_b0 = nb10 / type_आकार_src1;
            int64_t str_b1 = nb11 / type_आकार_src1;
            int64_t str_b2 = nb12 / type_आकार_src1;

            स्वचालित launch_gemm_for_batches = [&ctx, queue](ध्रुव sycl::half *src0,
                                                ध्रुव sycl::half *src1, भिन्न *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                तार्किक supported_प्रसारणम् = batchb == batcha ? सत्य
                        : batchb == 1 || batcha == 1        ? सत्य
                                                            : असत्य;
                यदि (supported_प्रसारणम्) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(), queue, batcha, batchb);
                } अन्यथा {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    यदि (batches0 > batches1) {
                        int64_t अङ्कः_mul_mats = batches1;
                        int64_t sub_batch = batches0 / अङ्कः_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        प्रति (int64_t i0 = 0; i0 < अङ्कः_mul_mats; i0++) {
                            ध्रुव sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ध्रुव sycl::half *src1_shifted = src1 + (sb2 * i0);
                            भिन्न *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(),
                                    queue, sub_batch, 1);
                        }
                    } अन्यथा {
                        int64_t अङ्कः_mul_mats = batches0;
                        int64_t sub_batch = batches1 / अङ्कः_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        प्रति (int64_t i1 = 0; i1 < अङ्कः_mul_mats; i1++) {
                            ध्रुव sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ध्रुव sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            भिन्न *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::गन्तव्यम्_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::गन्तव्यम्_dt<भिन्न>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ध्रुव तार्किक cont_batches_dim2_a = nb02 * ne02 == nb03;
            ध्रुव तार्किक cont_batches_dim2_b = nb12 * ne12 == nb13;
            ध्रुव तार्किक cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ध्रुव तार्किक cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            यदि (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / परिमाण(भिन्न));
            } अन्यथा यदि (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_आकार_src0;
                int64_t str_b3 = nb13 / type_आकार_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / परिमाण(भिन्न));
            } अन्यथा {
                प्रति (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ध्रुव sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_आकार_src0);
                    ध्रुव sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_आकार_src1);
                    भिन्न *dst_shifted = dst_ddf + (nb3 * b_a / परिमाण(भिन्न));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / परिमाण(भिन्न));
                }
            }

    }
    अन्यथा
#यदि_अन्तः
    {
        यदि (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ध्रुव int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ध्रुव int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_दत्त_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_दत्त_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_दत्त_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } अन्यथा {
            ध्रुव पूर्णाङ्क ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ध्रुव शून्य *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<शून्य *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<भिन्न>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> खण्डम्_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ध्रुव शून्य ** ptrs_src_get = ptrs_src.get();
                शून्य **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * परिमाण(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * परिमाण(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(खण्डम्_dims, खण्डम्_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ध्रुव शून्य **) (ptrs_src.get() + 0 * ne23), dpct::library_दत्त_t::real_half, nb01 / nb00,
                (ध्रुव शून्य **) (ptrs_src.get() + 1 * ne23), dpct::library_दत्त_t::real_half, s11, beta,
                (शून्य **) (ptrs_dst.get() + 0 * ne23), mkl_दत्त_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} गृहाण (ध्रुव sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

गणना वर्ग mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

अन्तःस्थ तार्किक ggml_sycl_supports_mmq(गणना ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_अप्रयुक्तम्(type);
    प्रत्यागम असत्य;
}

अन्तःस्थ तार्किक ggml_sycl_supports_reorder_mul_mat_sycl(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम सत्य;
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
            प्रत्यागम !g_ggml_sycl_prioritize_dmmv;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

अन्तःस्थ तार्किक ggml_sycl_supports_reorder_dmmv(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

अन्तःस्थ तार्किक ggml_sycl_supports_reorder_mmvq(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_IQ3_XXS:
        पक्ष GGML_TYPE_IQ3_S:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

स्थिर तार्किक ggml_sycl_supports_reorder_esimd(गणना ggml_type type) {
#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
    विकल्प (type) {
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
#अन्यथा
    GGML_अप्रयुक्तम्(type);
    प्रत्यागम असत्य;
#यदि_अन्तः
}

स्थिर तार्किक ggml_sycl_supports_dmmv(गणना ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_Q1_0:
        पक्ष GGML_TYPE_Q4_0:
        पक्ष GGML_TYPE_Q4_1:
        पक्ष GGML_TYPE_Q5_0:
        पक्ष GGML_TYPE_Q5_1:
        पक्ष GGML_TYPE_Q8_0:
        पक्ष GGML_TYPE_Q2_K:
        पक्ष GGML_TYPE_Q3_K:
        पक्ष GGML_TYPE_Q4_K:
        पक्ष GGML_TYPE_Q5_K:
        पक्ष GGML_TYPE_Q6_K:
        पक्ष GGML_TYPE_F16:
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम सत्य;
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
स्थिर अन्तःस्थ शून्य * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    तार्किक use_async = g_ggml_sycl_use_async_mem_op;
#यदि defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_स्मृतिः_ALLOC
    यदि (use_async) {
        शून्य * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_सङ्कलय(GGML_SYCL_MEM_ASYNC, ptr, size);
        प्रत्यागम ptr;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#यदि_अन्तः
    प्रत्यागम ggml_sycl_malloc_device(size, *stream);
}

स्थिर अन्तःस्थ शून्य sycl_ext_free(dpct::queue_ptr stream, शून्य * ptr) {
    तार्किक use_async = g_ggml_sycl_use_async_mem_op;
#यदि defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_स्मृतिः_ALLOC
    यदि (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        प्रत्यागम;
    }
#अन्यथा
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#यदि_अन्तः
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
संरचना sycl_reorder_अस्थायी_बफर {
    शून्य *          ptr  = शून्यसूचकः;
    dpct::queue_ptr stream;

    sycl_reorder_अस्थायी_बफर(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#यदि_परिभाषितम् GGML_SYCL_HOST_MEM_FALLBACK
        यदि (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            यदि (ptr) {
                host_fallback = सत्य;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#यदि_अन्तः
    }

    ~sycl_reorder_अस्थायी_बफर() {
        यदि (!ptr) {
            प्रत्यागम;
        }
        यदि (host_fallback) {
            sycl::मुक्त(ptr, *stream);
        } अन्यथा {
            sycl_ext_free(stream, ptr);
        }
    }

    स्पष्ट संक्रियकः तार्किक() ध्रुव { प्रत्यागम ptr != शून्यसूचकः; }

    sycl_reorder_अस्थायी_बफर(ध्रुव sycl_reorder_अस्थायी_बफर &)            = विलोपय;
    sycl_reorder_अस्थायी_बफर & संक्रियकः=(ध्रुव sycl_reorder_अस्थायी_बफर &) = विलोपय;

गुप्त:
    तार्किक host_fallback = असत्य;
};

स्थिर तार्किक reorder_qw_q4_0(uint8_t * दत्त_device, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, size_t size, size_t विस्थापनम्,
                            dpct::queue_ptr stream) {
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    GGML_ASSERT((size % परिमाण(खण्डम्_q4_0) == 0));
    GGML_ASSERT((विस्थापनम् % परिमाण(खण्डम्_q4_0) == 0));
    पूर्णाङ्क विस्थापनम्_blks = विस्थापनम् / परिमाण(खण्डम्_q4_0);
    स्वचालित qs_ptr      = दत्त_device + विस्थापनम्_blks * QK4_0 / 2;
    स्वचालित d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + विस्थापनम्_blks;

    स्वचालित reorder_event = stream->parallel_for(
        size / परिमाण(खण्डम्_q4_0),
            [=](स्वचालित i) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            ध्रुव खण्डम्_q4_0* x = (ध्रुव खण्डम्_q4_0*)tmp_buf;
            ध्रुव पूर्णाङ्क ib = i;

            प्रति (पूर्णाङ्क j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q8_0(uint8_t * दत्त_device, ध्रुव पूर्णाङ्क ncols, ध्रुव पूर्णाङ्क nrows, size_t size, size_t विस्थापनम्,
                            dpct::queue_ptr stream) {
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    GGML_ASSERT((size % परिमाण(खण्डम्_q8_0) == 0));
    GGML_ASSERT((विस्थापनम् % परिमाण(खण्डम्_q8_0) == 0));
    पूर्णाङ्क विस्थापनम्_blks = विस्थापनम् / परिमाण(खण्डम्_q8_0);
    स्वचालित qs_ptr = दत्त_device + विस्थापनम्_blks * QK8_0;
    स्वचालित d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + विस्थापनम्_blks;

    स्वचालित reorder_event = stream->parallel_for(
        size / परिमाण(खण्डम्_q8_0),
            [=](स्वचालित i) [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
            ध्रुव खण्डम्_q8_0* x = (ध्रुव खण्डम्_q8_0*)tmp_buf;
            ध्रुव पूर्णाङ्क ib = i;

            प्रति (पूर्णाङ्क j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
स्थिर तार्किक reorder_qw_q8_0_moe(uint8_t * दत्त_device, size_t expert_अष्टकानि, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_अष्टकानि % परिमाण(खण्डम्_q8_0) == 0);
    ध्रुव पूर्णाङ्क    खण्डानि_प्रत्येकम्_expert = (पूर्णाङ्क) (expert_अष्टकानि / परिमाण(खण्डम्_q8_0));
    ध्रुव size_t योग_अष्टकानि       = expert_अष्टकानि * (size_t) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_अष्टकानि);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_अष्टकानि);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, योग_अष्टकानि)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    ध्रुव पूर्णाङ्क योग_खण्डानि = खण्डानि_प्रत्येकम्_expert * (पूर्णाङ्क) n_expert;
    स्वचालित reorder_event = stream->parallel_for(योग_खण्डानि, [=](स्वचालित gb_) {
        ध्रुव पूर्णाङ्क          gb   = gb_;
        ध्रुव पूर्णाङ्क          e    = gb / खण्डानि_प्रत्येकम्_expert;
        ध्रुव पूर्णाङ्क          ib   = gb % खण्डानि_प्रत्येकम्_expert;
        ध्रुव खण्डम्_q8_0 * x    = (ध्रुव खण्डम्_q8_0 *) (tmp_buf + (size_t) e * expert_अष्टकानि);
        uint8_t *          आधारः = दत्त_device + (size_t) e * expert_अष्टकानि;

        स्वचालित * qs_ptr = आधारः;
        स्वचालित * d_ptr  = (sycl::half *) (qs_ptr + (size_t) खण्डानि_प्रत्येकम्_expert * QK8_0);

        प्रति (पूर्णाङ्क j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q4_k(uint8_t * दत्त_device, size_t size, size_t विस्थापनम्, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_q4_K) == 0);
    GGML_ASSERT(विस्थापनम् % परिमाण(खण्डम्_q4_K) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_q4_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित * qs_ptr     = दत्त_device;
    स्वचालित * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    स्वचालित * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_q4_K * x  = (ध्रुव खण्डम्_q4_K *) tmp_buf;
        ध्रुव पूर्णाङ्क          ib = i;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

// Reorder each expert slice into a self-contained SoA layout.
स्थिर तार्किक reorder_qw_q4_k_moe(uint8_t * दत्त_device, size_t expert_अष्टकानि, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_अष्टकानि % परिमाण(खण्डम्_q4_K) == 0);
    ध्रुव पूर्णाङ्क    खण्डानि_प्रत्येकम्_expert = (पूर्णाङ्क) (expert_अष्टकानि / परिमाण(खण्डम्_q4_K));
    ध्रुव size_t योग_अष्टकानि       = expert_अष्टकानि * (size_t) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_अष्टकानि);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_अष्टकानि);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, योग_अष्टकानि)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    ध्रुव पूर्णाङ्क योग_खण्डानि = खण्डानि_प्रत्येकम्_expert * (पूर्णाङ्क) n_expert;
    स्वचालित reorder_event = stream->parallel_for(योग_खण्डानि, [=](स्वचालित gb_) {
        ध्रुव पूर्णाङ्क          gb   = gb_;
        ध्रुव पूर्णाङ्क          e    = gb / खण्डानि_प्रत्येकम्_expert;
        ध्रुव पूर्णाङ्क          ib   = gb % खण्डानि_प्रत्येकम्_expert;
        ध्रुव खण्डम्_q4_K * x    = (ध्रुव खण्डम्_q4_K *) (tmp_buf + (size_t) e * expert_अष्टकानि);
        uint8_t *          आधारः = दत्त_device + (size_t) e * expert_अष्टकानि;

        स्वचालित * qs_ptr     = आधारः;
        स्वचालित * scales_ptr = qs_ptr + QK_K / 2 * खण्डानि_प्रत्येकम्_expert;
        स्वचालित * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * खण्डानि_प्रत्येकम्_expert);

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
स्थिर तार्किक reorder_qw_q5_k_moe(uint8_t * दत्त_device, size_t expert_अष्टकानि, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_अष्टकानि % परिमाण(खण्डम्_q5_K) == 0);
    ध्रुव पूर्णाङ्क    खण्डानि_प्रत्येकम्_expert = (पूर्णाङ्क) (expert_अष्टकानि / परिमाण(खण्डम्_q5_K));
    ध्रुव size_t योग_अष्टकानि       = expert_अष्टकानि * (size_t) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_अष्टकानि);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_अष्टकानि);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, योग_अष्टकानि)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    ध्रुव पूर्णाङ्क योग_खण्डानि = खण्डानि_प्रत्येकम्_expert * (पूर्णाङ्क) n_expert;
    स्वचालित reorder_event = stream->parallel_for(योग_खण्डानि, [=](स्वचालित gb_) {
        ध्रुव पूर्णाङ्क          gb   = gb_;
        ध्रुव पूर्णाङ्क          e    = gb / खण्डानि_प्रत्येकम्_expert;
        ध्रुव पूर्णाङ्क          ib   = gb % खण्डानि_प्रत्येकम्_expert;
        ध्रुव खण्डम्_q5_K * x    = (ध्रुव खण्डम्_q5_K *) (tmp_buf + (size_t) e * expert_अष्टकानि);
        uint8_t *          आधारः = दत्त_device + (size_t) e * expert_अष्टकानि;

        स्वचालित * qs_ptr     = आधारः;
        स्वचालित * qh_ptr     = qs_ptr + (QK_K / 2) * खण्डानि_प्रत्येकम्_expert;
        स्वचालित * scales_ptr = qh_ptr + (QK_K / 8) * खण्डानि_प्रत्येकम्_expert;
        स्वचालित * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * खण्डानि_प्रत्येकम्_expert);

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
स्थिर तार्किक reorder_qw_q6_k_moe(uint8_t * दत्त_device, size_t expert_अष्टकानि, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_अष्टकानि % परिमाण(खण्डम्_q6_K) == 0);
    ध्रुव पूर्णाङ्क    खण्डानि_प्रत्येकम्_expert = (पूर्णाङ्क) (expert_अष्टकानि / परिमाण(खण्डम्_q6_K));
    ध्रुव size_t योग_अष्टकानि       = expert_अष्टकानि * (size_t) n_expert;

    sycl_reorder_अस्थायी_बफर tmp(stream, योग_अष्टकानि);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, योग_अष्टकानि);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, योग_अष्टकानि)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    ध्रुव पूर्णाङ्क योग_खण्डानि = खण्डानि_प्रत्येकम्_expert * (पूर्णाङ्क) n_expert;
    स्वचालित reorder_event = stream->parallel_for(योग_खण्डानि, [=](स्वचालित gb_) {
        ध्रुव पूर्णाङ्क          gb   = gb_;
        ध्रुव पूर्णाङ्क          e    = gb / खण्डानि_प्रत्येकम्_expert;
        ध्रुव पूर्णाङ्क          ib   = gb % खण्डानि_प्रत्येकम्_expert;
        ध्रुव खण्डम्_q6_K * x    = (ध्रुव खण्डम्_q6_K *) (tmp_buf + (size_t) e * expert_अष्टकानि);
        uint8_t *          आधारः = दत्त_device + (size_t) e * expert_अष्टकानि;

        स्वचालित * ql_ptr     = आधारः;
        स्वचालित * qh_ptr     = ql_ptr + (QK_K / 2) * खण्डानि_प्रत्येकम्_expert;
        स्वचालित * scales_ptr = qh_ptr + (QK_K / 4) * खण्डानि_प्रत्येकम्_expert;
        स्वचालित * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * खण्डानि_प्रत्येकम्_expert);

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
स्थिर तार्किक reorder_qw_mxfp4_moe(uint8_t * दत्त_device, size_t expert_अष्टकानि, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_अष्टकानि % परिमाण(खण्डम्_mxfp4) == 0);
    ध्रुव पूर्णाङ्क     खण्डानि_प्रत्येकम्_expert = (पूर्णाङ्क) (expert_अष्टकानि / परिमाण(खण्डम्_mxfp4));
    ध्रुव size_t  max_chunk_अष्टकानि   = 32u << 20;
    ध्रुव int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_अष्टकानि / expert_अष्टकानि)));

    sycl_reorder_अस्थायी_बफर tmp(stream, (size_t) chunk_experts * expert_अष्टकानि);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_अष्टकानि);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    प्रति (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ध्रुव int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = दत्त_device + (size_t) e0 * expert_अष्टकानि;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, chunk, (size_t) n_chunk * expert_अष्टकानि)));
        यदि (!g_ggml_sycl_use_async_mem_op) {
            copy_event.प्रतीक्षस्व();
        }

        ध्रुव पूर्णाङ्क योग_खण्डानि = खण्डानि_प्रत्येकम्_expert * (पूर्णाङ्क) n_chunk;
        स्वचालित reorder_event = stream->parallel_for(योग_खण्डानि, [=](स्वचालित gb_) {
            ध्रुव पूर्णाङ्क           gb   = gb_;
            ध्रुव पूर्णाङ्क           e    = gb / खण्डानि_प्रत्येकम्_expert;
            ध्रुव पूर्णाङ्क           ib   = gb % खण्डानि_प्रत्येकम्_expert;
            ध्रुव खण्डम्_mxfp4 * x    = (ध्रुव खण्डम्_mxfp4 *) (tmp_buf + (size_t) e * expert_अष्टकानि);
            uint8_t *           आधारः = chunk + (size_t) e * expert_अष्टकानि;

            uint8_t * qs_ptr = आधारः;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) खण्डानि_प्रत्येकम्_expert;

            प्रति (पूर्णाङ्क j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        यदि (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q2_k(uint8_t * दत्त_device, size_t size, size_t विस्थापनम्, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_q2_K) == 0);
    GGML_ASSERT(विस्थापनम् % परिमाण(खण्डम्_q2_K) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_q2_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित *        qs_ptr     = दत्त_device;
    स्वचालित *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_q2_K * x  = (ध्रुव खण्डम्_q2_K *) tmp_buf;
        ध्रुव पूर्णाङ्क          ib = i;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q3_k(uint8_t * दत्त_device, size_t size, size_t विस्थापनम्, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_q3_K) == 0);
    GGML_ASSERT(विस्थापनम् % परिमाण(खण्डम्_q3_K) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_q3_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित *       qs_ptr     = दत्त_device;
    स्वचालित *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    स्वचालित *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_q3_K * x  = (ध्रुव खण्डम्_q3_K *) tmp_buf;
        ध्रुव पूर्णाङ्क          ib = i;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q5_k(uint8_t * दत्त_device, size_t size, size_t विस्थापनम्, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_q5_K) == 0);
    GGML_ASSERT(विस्थापनम् % परिमाण(खण्डम्_q5_K) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_q5_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित * qs_ptr     = दत्त_device;
    स्वचालित * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    स्वचालित * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    स्वचालित * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_आकार * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_q5_K * x  = (ध्रुव खण्डम्_q5_K *) tmp_buf;
        ध्रुव पूर्णाङ्क          ib = i;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < K_SCALE_आकार; ++j) {
            scales_ptr[ib * K_SCALE_आकार + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_q6_k(uint8_t * दत्त_device, size_t size, size_t विस्थापनम्, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_q6_K) == 0);
    GGML_ASSERT(विस्थापनम् % परिमाण(खण्डम्_q6_K) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_q6_K);

    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित *       ql_ptr     = दत्त_device;
    स्वचालित *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    स्वचालित *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_q6_K * x  = (ध्रुव खण्डम्_q6_K *) tmp_buf;
        ध्रुव पूर्णाङ्क          ib = i;

        ध्रुव uint8_t * ql              = x[ib].ql;
        ध्रुव uint8_t * qh              = x[ib].qh;
        uint8_t *       आधारः_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       आधारः_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       आधारः_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 2; ++j) {
            आधारः_ql_ptr[j] = ql[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 4; ++j) {
            आधारः_qh_ptr[j] = qh[j];
        }

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 16; ++j) {
            आधारः_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_iq3_s(uint8_t * दत्त_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_iq3_s) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_iq3_s);
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित * qs_ptr       = दत्त_device;
    स्वचालित * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    स्वचालित * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    स्वचालित * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_iq3_s * x = (ध्रुव खण्डम्_iq3_s *) tmp_buf;
        ध्रुव पूर्णाङ्क ib = i;

        प्रति (पूर्णाङ्क j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        प्रति (पूर्णाङ्क j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (परिमाण(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        प्रति (पूर्णाङ्क j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[परिमाण(ggml_half) + j] = x[ib].scales[j];
        }
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw_iq3_xxs(uint8_t * दत्त_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % परिमाण(खण्डम्_iq3_xxs) == 0);

    ध्रुव पूर्णाङ्क nblocks = size / परिमाण(खण्डम्_iq3_xxs);
    sycl_reorder_अस्थायी_बफर tmp(stream, size);
    यदि (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        प्रत्यागम असत्य;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->स्मृतिं_प्रतिलिख(tmp_buf, दत्त_device, size)));
    यदि (!g_ggml_sycl_use_async_mem_op) {
        copy_event.प्रतीक्षस्व();
    }

    स्वचालित * qs_ptr = दत्त_device;
    स्वचालित * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    स्वचालित reorder_event = stream->parallel_for(nblocks, [=](स्वचालित i) {
        ध्रुव खण्डम्_iq3_xxs * x = (ध्रुव खण्डम्_iq3_xxs *) tmp_buf;
        ध्रुव पूर्णाङ्क ib = i;

        प्रति (पूर्णाङ्क j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    यदि (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    प्रत्यागम सत्य;
}

स्थिर तार्किक reorder_qw(ध्रुव ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * दत्त_device = (uint8_t *) src0->दत्त;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    यदि (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        विकल्प (src0->type) {
            पक्ष GGML_TYPE_Q8_0:
                प्रत्यागम reorder_qw_q8_0_moe(दत्त_device, src0->nb[2], src0->ne[2], stream);
            पक्ष GGML_TYPE_Q4_K:
                प्रत्यागम reorder_qw_q4_k_moe(दत्त_device, src0->nb[2], src0->ne[2], stream);
            पक्ष GGML_TYPE_Q5_K:
                प्रत्यागम reorder_qw_q5_k_moe(दत्त_device, src0->nb[2], src0->ne[2], stream);
            पक्ष GGML_TYPE_Q6_K:
                प्रत्यागम reorder_qw_q6_k_moe(दत्त_device, src0->nb[2], src0->ne[2], stream);
            पक्ष GGML_TYPE_MXFP4:
                प्रत्यागम reorder_qw_mxfp4_moe(दत्त_device, src0->nb[2], src0->ne[2], stream);
            स्वभाव:
                प्रत्यागम असत्य;
        }
    }

    विकल्प (src0->type) {
        पक्ष GGML_TYPE_Q4_0:
            प्रत्यागम reorder_qw_q4_0(दत्त_device, ncols, nrows, size, 0, stream);
        पक्ष GGML_TYPE_Q8_0:
            प्रत्यागम reorder_qw_q8_0(दत्त_device, ncols, nrows, size, 0, stream);
        पक्ष GGML_TYPE_Q2_K:
            प्रत्यागम reorder_qw_q2_k(दत्त_device, size, 0, stream);
        पक्ष GGML_TYPE_Q3_K:
            प्रत्यागम reorder_qw_q3_k(दत्त_device, size, 0, stream);
        पक्ष GGML_TYPE_Q4_K:
            प्रत्यागम reorder_qw_q4_k(दत्त_device, size, 0, stream);
        पक्ष GGML_TYPE_Q5_K:
            प्रत्यागम reorder_qw_q5_k(दत्त_device, size, 0, stream);
        पक्ष GGML_TYPE_Q6_K:
            प्रत्यागम reorder_qw_q6_k(दत्त_device, size, 0, stream);
        पक्ष GGML_TYPE_IQ3_XXS:
            प्रत्यागम reorder_qw_iq3_xxs(दत्त_device, size, stream);
        पक्ष GGML_TYPE_IQ3_S:
            प्रत्यागम reorder_qw_iq3_s(दत्त_device, size, stream);
        स्वभाव:
            प्रत्यागम असत्य;
    }
}

स्थिर तार्किक should_reorder_tensor(ggml_backend_sycl_context& ctx, ध्रुव ggml_tensor * dst) {
    प्रत्यागम g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

स्थिर शून्य opt_for_reorder(ggml_backend_sycl_context * ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    यदि (!should_reorder_tensor(*ctx, dst)) {
        प्रत्यागम;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    यदि (!extra || extra->optimized_feature.reorder) {
        प्रत्यागम;  // Skip permutations and already reordered tensors
    }

    विकल्प (mm_algorithm) {
        पक्ष mul_mat_algo::DMMV:
            यदि (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                प्रत्यागम;
            }
            भङ्ग;
        पक्ष mul_mat_algo::MMVQ:
            यदि (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                प्रत्यागम;
            }
            भङ्ग;
        पक्ष mul_mat_algo::MUL_MAT_SYCL:
            यदि (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                प्रत्यागम;
            }
            भङ्ग;
    }

    यदि (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = सत्य;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
स्थिर शून्य opt_for_reorder_id(ggml_backend_sycl_context * ctx, ध्रुव ggml_tensor * src0) {
    यदि (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        प्रत्यागम;
    }
    यदि (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        प्रत्यागम;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    यदि (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->दत्त % 16 != 0)) {
        प्रत्यागम;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    यदि (!extra || extra->optimized_feature.reorder) {
        प्रत्यागम;
    }
    यदि (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = सत्य;
    }
}


स्थिर तार्किक can_use_dequantize_mul_mat_vec(ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ध्रुव int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    प्रत्यागम ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

स्थिर तार्किक can_use_mul_mat_vec_q(ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    प्रत्यागम ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_आकार;
}

स्थिर शून्य ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    यदि (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        प्रत्यागम;
    }

    ध्रुव तार्किक विभज = ggml_backend_बफर_is_sycl_विभज(src0->बफर);
    int64_t min_compute_capability = अधिकतम_पूर्णाङ्कः;

    यदि (विभज) {
        ggml_backend_sycl_विभज_बफर_type_context * buft_ctx =
            (ggml_backend_sycl_विभज_बफर_type_context *) src0->बफर->buft->context;
        स्वचालित & tensor_विभज = buft_ctx->tensor_विभज;
        प्रति (पूर्णाङ्क id = 0; id < ggml_sycl_info().device_गणनम्; ++id) {
            // skip devices that are not going to do any work:
            यदि (tensor_विभज[id] >= (id + 1 < ggml_sycl_info().device_गणनम् ? tensor_विभज[id + 1] : 1.0f)) {
                अनुवर्त;
            }

            यदि (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } अन्यथा {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    तार्किक use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#यदि_परिभाषितम् GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#यदि_अन्तः

    तार्किक use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    तार्किक use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;


    // mmvq and mmq need the __dp4a instruction which is available for gen12+
    // Workaround in https://github.com/ggml-org/llama.cpp/commit/95f84d5ce8b449a9b16009434aca800df504a02e
    use_mul_mat_q = use_mul_mat_q && (src0->type != GGML_TYPE_IQ2_XXS);
#यदि_परिभाषितम् SYCL_USE_XMX
    use_mul_mat_q = use_mul_mat_q && (src1->ne[1] <= MMQ_MAX_BATCH_आकार);
#यदि_अन्तः // SYCL_USE_XMX

    // When reorder is enabled, both ESIMD, MMVQ and DMMV kernels may be used. For
    // best performance use ESIMD when supported, followed by MMVQ, and finally DMMV.
    // But the reordered ESIMD path cannot be used without reordered MMVQ. A later
    // multi-token call (ne[1] in 2..8) will take the MMVQ path and it would read the
    // reordered bytes as if they were still the unreordered layout.

    यदि (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        तार्किक use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        यदि (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    यदि (!विभज && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        यदि (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } अन्यथा {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } अन्यथा यदि (!विभज && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } अन्यथा यदि (!विभज && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } अन्यथा यदि (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } अन्यथा यदि (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        यदि (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } अन्यथा {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } अन्यथा यदि (use_mul_mat_q) {
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
                                             ggml_tensor * gate, ggml_tensor * ऊर्ध्वम्, ध्रुव ggml_tensor * wu,
                                             ध्रुव ggml_tensor * wg, ध्रुव ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ध्रुव स्वचालित * extra_u = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wu->extra);
    ध्रुव स्वचालित * extra_g = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wg->extra);
    यदि ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        प्रत्यागम असत्य;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ध्रुव स्वचालित plain_type = [](ggml_type t) { प्रत्यागम t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    यदि (!plain_type(wu->type) || !plain_type(wg->type)) {
        प्रत्यागम असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ऊर्ध्वम्, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ध्रुव int64_t ne00 = wu->ne[0];
    ध्रुव int64_t ne11 = act->ne[1];

    ध्रुव queue_ptr stream = ctx.stream();
    ध्रुव पूर्णाङ्क src1_padded_स्तम्भसङ्ख्या = GGML_PAD((पूर्णाङ्क) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<वर्ण> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_स्तम्भसङ्ख्या * परिमाण(खण्डम्_q8_1) / QK8_1);
    वर्ण * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((ध्रुव भिन्न *) act->दत्त, src1_ddq, (पूर्णाङ्क) ne00, (पूर्णाङ्क) ne11,
                                          src1_padded_स्तम्भसङ्ख्या, stream);

    प्रत्यागम ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->दत्त, wu->दत्त,
                                             src1_ddq, (भिन्न *) glu->दत्त, (पूर्णाङ्क) ne00, (पूर्णाङ्क) wu->ne[1],
                                             (पूर्णाङ्क) ne11,
                                             /*stride_col_y=*/src1_padded_स्तम्भसङ्ख्या / QK8_1,
                                             /*stride_col_dst=*/(पूर्णाङ्क) glu->ne[0], stream);
    GGML_अप्रयुक्तम्(gate);
}

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
स्थिर तार्किक ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * ऊर्ध्वम्, ध्रुव ggml_tensor * wu,
                                             ध्रुव ggml_tensor * wg, ध्रुव ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ध्रुव वर्ण * act_begin = (ध्रुव वर्ण *) act->दत्त;
    ध्रुव वर्ण * glu_begin = (ध्रुव वर्ण *) glu->दत्त;
    यदि (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        प्रत्यागम असत्य;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, ऊर्ध्वम्, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ध्रुव स्वचालित * extra_u = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wu->extra);
    ध्रुव स्वचालित * extra_g = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wg->extra);
    यदि (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        प्रत्यागम असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ऊर्ध्वम्, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    प्रत्यागम ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->दत्त, wg->दत्त,
                                                              (ध्रुव भिन्न *) act->दत्त, (भिन्न *) glu->दत्त,
                                                              (पूर्णाङ्क) wu->ne[0], (पूर्णाङ्क) wu->ne[1], ctx.stream());
}
#यदि_अन्तः

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
स्थिर तार्किक ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    यदि (!ggml_sycl_can_fuse(cgraph, ग्रन्थिः_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        प्रत्यागम असत्य;
    }

    ggml_tensor *       glu  = cgraph->nodes[ग्रन्थिः_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       ऊर्ध्वम्   = glu->src[1];
    ध्रुव ggml_tensor * wu   = ऊर्ध्वम्->src[0];
    ध्रुव ggml_tensor * wg   = gate->src[0];
    ध्रुव ggml_tensor * act  = ऊर्ध्वम्->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    यदि (ggml_backend_बफर_is_sycl_विभज(wu->बफर) || ggml_backend_बफर_is_sycl_विभज(wg->बफर)) {
        प्रत्यागम असत्य;
    }

#यदि_परिभाषितम् GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    यदि (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, ऊर्ध्वम्, wu, wg, act)) {
        प्रत्यागम सत्य;
    }
#यदि_अन्तः

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    यदि (g_ggml_sycl_prioritize_dmmv) {
        प्रत्यागम असत्य;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ध्रुव तार्किक reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    यदि (!reorder_pair) {
        प्रत्यागम ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, ऊर्ध्वम्, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    यदि (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        प्रत्यागम असत्य;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, ऊर्ध्वम्, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ध्रुव स्वचालित * extra_u = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wu->extra);
    ध्रुव स्वचालित * extra_g = static_cast<ध्रुव ggml_tensor_extra_gpu *>(wg->extra);
    यदि (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        प्रत्यागम असत्य;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, ऊर्ध्वम्, /*num_src=*/2, " : fused with gate + GLU");

    ध्रुव int64_t ne00 = wu->ne[0];
    ध्रुव int64_t ne11 = act->ne[1];

    ध्रुव queue_ptr stream           = ctx.stream();
    ध्रुव पूर्णाङ्क       src1_padded_स्तम्भसङ्ख्या = GGML_PAD((पूर्णाङ्क) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<वर्ण> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_स्तम्भसङ्ख्या * परिमाण(खण्डम्_q8_1) / QK8_1);
    वर्ण *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ध्रुव भिन्न *) act->दत्त, src1_ddq, (पूर्णाङ्क) ne00, (पूर्णाङ्क) ne11,
                                                          src1_padded_स्तम्भसङ्ख्या, stream);

    प्रत्यागम ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->दत्त, wg->दत्त, src1_ddq,
                                               (भिन्न *) glu->दत्त, (पूर्णाङ्क) ne00, (पूर्णाङ्क) wu->ne[1], (पूर्णाङ्क) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_स्तम्भसङ्ख्या * (पूर्णाङ्क) परिमाण(खण्डम्_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(पूर्णाङ्क) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
स्थिर पूर्णाङ्क ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx) {
    ध्रुव ggml_tensor * ग्रन्थिः = cgraph->nodes[ग्रन्थिः_idx];
    यदि (ggml_sycl_info().device_गणनम् != 1 || ग्रन्थिः->type != GGML_TYPE_F32 ||
        ग्रन्थिः->src[0]->type != GGML_TYPE_F32 || ग्रन्थिः->src[0]->ne[0] >= 1024) {
        प्रत्यागम 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    पूर्णाङ्क           count = 0;
    पूर्णाङ्क           चरमः  = ग्रन्थिः_idx;
    भिन्न         eps0;
    स्मृतिं_प्रतिलिख(&eps0, ग्रन्थिः->op_params, परिमाण(भिन्न));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    स्वचालित overlaps = [](ध्रुव ggml_tensor * a, ध्रुव ggml_tensor * b) {
        ध्रुव वर्ण * ab = (ध्रुव वर्ण *) a->दत्त;
        ध्रुव वर्ण * bb = (ध्रुव वर्ण *) b->दत्त;
        प्रत्यागम ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    प्रति (पूर्णाङ्क j = ग्रन्थिः_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        यदि (ggml_is_रिक्तानि(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;  // not a launch; cannot break a run of adjacent norms
        }
        यदि (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ग्रन्थिः) ||
            !ggml_are_same_shape(nj->src[0], ग्रन्थिः->src[0])) {
            भङ्ग;  // any other launch ends the run
        }
        तार्किक same_nb = सत्य;
        प्रति (पूर्णाङ्क d = 0; d < GGML_MAX_DIMS; ++d) {
            यदि (nj->nb[d] != ग्रन्थिः->nb[d] || nj->src[0]->nb[d] != ग्रन्थिः->src[0]->nb[d]) {
                same_nb = असत्य;
                भङ्ग;
            }
        }
        यदि (!same_nb) {
            भङ्ग;  // one nb[] stride set is shared by the whole batch
        }
        भिन्न epsj;
        स्मृतिं_प्रतिलिख(&epsj, nj->op_params, परिमाण(भिन्न));
        यदि (epsj != eps0) {
            भङ्ग;  // eps mismatch ends the run
        }
        तार्किक indep = सत्य;
        प्रति (पूर्णाङ्क k = 0; k < count; ++k) {
            यदि (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = असत्य;
                भङ्ग;
            }
        }
        यदि (!indep) {
            भङ्ग;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        चरमः           = j;
    }
    यदि (count < 2) {
        प्रत्यागम 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    प्रत्यागम चरमः - ग्रन्थिः_idx;
}


__dpct_inline__ स्थिर शून्य k_copy_src1_गन्तव्यम्_contiguous(
    ध्रुव वर्ण *__restrict__ src1_original, वर्ण *__restrict__ src1_contiguous,
    ध्रुव mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव int32_t src1_row = item_ct1.get_गणः(2);

    ध्रुव int32_t iid1 = row_mapping[src1_row].i2;
    ध्रुव int32_t id   = row_mapping[src1_row].i1;

    ध्रुव int64_t i11 = id % ne11;
    ध्रुव int64_t i12 = iid1;

    ध्रुव भिन्न * src1_row_original = (ध्रुव भिन्न *)(src1_original + i11*nb11 + i12*nb12);
    भिन्न * src1_row_contiguous = (भिन्न *)(src1_contiguous + src1_row*nb11);

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ स्थिर शून्य k_copy_dst_from_contiguous(
    वर्ण *__restrict__ dst_original, ध्रुव वर्ण *__restrict__ dst_contiguous,
    ध्रुव mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ध्रुव sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_गणः(2);

    ध्रुव int32_t i1 = row_mapping[i].i1;
    ध्रुव int32_t i2 = row_mapping[i].i2;

    ध्रुव भिन्न * dst_row_contiguous = (ध्रुव भिन्न *)(dst_contiguous + i*nb1);
    भिन्न * dst_row_original = (भिन्न *)(dst_original + i1*nb1 + i2*nb2);

#निर्देशः unroll
    प्रति (पूर्णाङ्क j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
स्थिर तार्किक ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0,
    ध्रुव ggml_tensor * src1, ध्रुव ggml_tensor * ids, ggml_tensor * dst)
{
    ध्रुव int64_t ne10 = src1->ne[0];
    ध्रुव int64_t ne11 = src1->ne[1];
    ध्रुव int64_t ne12 = src1->ne[2];
    यदि (ne12 != 1) प्रत्यागम असत्य;
    यदि (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) प्रत्यागम असत्य;
    यदि (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) प्रत्यागम असत्य;
    यदि (!ggml_is_contiguous(src1)) प्रत्यागम असत्य;

    ध्रुव int64_t n_ids_प्रत्येकम्_गणः = ids->ne[0];
    यदि (ids->ne[1] != 1) प्रत्यागम असत्य;
    यदि (ne11 != 1 && ne11 != n_ids_प्रत्येकम्_गणः) प्रत्यागम असत्य;

    ध्रुव queue_ptr stream           = ctx.stream();
    ध्रुव पूर्णाङ्क       src1_padded_स्तम्भसङ्ख्या = GGML_PAD((पूर्णाङ्क) ne10, MATRIX_ROW_PADDING);
    ध्रुव पूर्णाङ्क       n_experts_प्रयुक्तम्   = (पूर्णाङ्क) n_ids_प्रत्येकम्_गणः;
    ध्रुव पूर्णाङ्क       nrows            = (पूर्णाङ्क) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ध्रुव ggml_tensor_extra_gpu * src0_extra =
        static_cast<ध्रुव ggml_tensor_extra_gpu *>(src0->extra);
    ध्रुव तार्किक use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<वर्ण> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_स्तम्भसङ्ख्या * परिमाण(खण्डम्_q8_1) / QK8_1);
    वर्ण * src1_ddq = src1_q8_alloc.get();
    यदि (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ध्रुव भिन्न *) src1->दत्त, src1_ddq, (पूर्णाङ्क) ne10, (पूर्णाङ्क) ne11,
            src1_padded_स्तम्भसङ्ख्या, stream);
    } अन्यथा {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (ध्रुव भिन्न *) src1->दत्त, src1_ddq, (पूर्णाङ्क) ne10, (पूर्णाङ्क) ne11,
            src1_padded_स्तम्भसङ्ख्या, stream);
    }

    ध्रुव size_t अष्टकानि_प्रत्येकम्_qrow = (size_t) src1_padded_स्तम्भसङ्ख्या * परिमाण(खण्डम्_q8_1) / QK8_1;
    ध्रुव size_t src1_row_stride = (ne11 == 1) ? 0 : अष्टकानि_प्रत्येकम्_qrow;

    यदि (use_reorder) {
        प्रत्यागम ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->दत्त, src1_ddq, (ध्रुव int32_t *) ids->दत्त,
            (भिन्न *) dst->दत्त, (पूर्णाङ्क) ne10, nrows, n_experts_प्रयुक्तम्,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    प्रत्यागम ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->दत्त, src1_ddq, (ध्रुव int32_t *) ids->दत्त,
        (भिन्न *) dst->दत्त, (पूर्णाङ्क) ne10, nrows, n_experts_प्रयुक्तम्,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
स्थिर शून्य mmid_counting_क्रमय_पङ्क्तिकाः(
        ध्रुव ggml_tensor * ids, ध्रुव वर्ण * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_पङ्क्तिकाः,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.नियोजय(n_as, 0);
    प्रति (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        प्रति (int64_t id = 0; id < n_ids; id++) {
            ध्रुव int32_t row_id_i = *(ध्रुव int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.नियोजय(n_as + 1, 0);
    प्रति (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_अग्र = expert_row_offsets;
    routed_row_src.resize(n_routed_पङ्क्तिकाः);
    प्रति (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        प्रति (int64_t id = 0; id < n_ids; id++) {
            ध्रुव int32_t row_id_i = *(ध्रुव int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ध्रुव int64_t routed_row = expert_row_अग्र[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

स्थिर शून्य ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) यत्न {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ध्रुव ggml_tensor *src0 = dst->src[0];
    ध्रुव ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_बफर_is_sycl_विभज(src0->बफर) && "mul_mat_id does not support split buffers");

    ध्रुव ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_द्विमान_OP_LOCALS

    ध्रुव queue_ptr stream = ctx.stream();

    ध्रुव int64_t n_as = ne02;
    ध्रुव int64_t n_ids = ids->ne[0];

    यदि (ne12 == 1) {
        यदि (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            प्रत्यागम;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    यदि (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<वर्ण> ids_host(ggml_nbytes(ids));
    ध्रुव वर्ण * ids_dev = (ध्रुव वर्ण *) ids->दत्त;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->स्मृतिं_प्रतिलिख(ids_host.दत्त(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->प्रतीक्षस्व()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    वर्ण *src0_original = (वर्ण *)src0->दत्त;
    वर्ण *src1_original = (वर्ण *)src1->दत्त;
    वर्ण *dst_original = (वर्ण *)dst->दत्त;

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
    यदि (ne12 == 1) {
        प्रति (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            प्रति (int64_t id = 0; id < n_ids; id++) {
                ध्रुव int32_t i02 = *(ध्रुव int32_t *) (ids_host.दत्त() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ध्रुव int64_t i11 = id % ne11;
                ध्रुव int64_t i12 = iid1;

                ध्रुव int64_t i1 = id;
                ध्रुव int64_t i2 = i12;

            src0_row.दत्त = src0_original + i02*nb02;
            src1_row.दत्त = src1_original + i11*nb11 + i12*nb12;
            dst_row.दत्त = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } अन्यथा {
        ध्रुव int64_t n_routed_पङ्क्तिकाः = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<वर्ण> src1_contiguous(ctx.pool(), परिमाण(भिन्न)*n_routed_पङ्क्तिकाः*ne10);
        ggml_sycl_pool_alloc<वर्ण>  dst_contiguous(ctx.pool(), परिमाण(भिन्न)*n_routed_पङ्क्तिकाः*ne0);

        src1_row.दत्त = src1_contiguous.get();
        dst_row.दत्त  =  dst_contiguous.get();

        // how many "owned" routed rows to pass to each expert
        std::vector<int64_t> expert_row_counts;
        // where each expert's slice starts and the previous ends (row indices, right-exclusive)
        std::vector<int64_t> expert_row_offsets;
        // the sources (slot/token pairs) of contiguous rows to guide k_copy_src1_to_contiguous
        std::vector<mmid_row_mapping> & routed_row_src = ctx.mmid_row_mapping_host;

        mmid_counting_क्रमय_पङ्क्तिकाः(ids, ids_host.दत्त(), n_ids, n_as, n_routed_पङ्क्तिकाः,
                                expert_row_counts, expert_row_offsets, routed_row_src);

        ggml_sycl_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool(), n_routed_पङ्क्तिकाः);
        SYCL_CHECK(CHECK_TRY_ERROR(
                stream->स्मृतिं_प्रतिलिख(dev_row_mapping.get(), routed_row_src.दत्त(), n_routed_पङ्क्तिकाः*परिमाण(mmid_row_mapping))));

        ध्रुव अचिह्नित पूर्णाङ्क max_work_गणः_आकार = ggml_sycl_info().max_work_गणः_sizes[ctx.device];
        प्रतिज्ञा(max_work_गणः_आकार % (WARP_आकार * WARP_आकार) == 0);

        {
            sycl::range<3> खण्डम्_dims(1, 1, std::min((अचिह्नित पूर्णाङ्क)ne10, max_work_गणः_आकार));
            sycl::range<3> grid_dims(1, 1, n_routed_पङ्क्तिकाः);
            stream->submit([&](sycl::handler &cgh) {
                वर्ण *__restrict src1_contiguous_get =
                    src1_contiguous.get();
                mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_copy_src1_गन्तव्यम्_contiguous(
                            src1_original, src1_contiguous_get,
                            dev_row_mapping_get,
                            ne11, ne10, nb11, nb12,
                            item_ct1);
                    });
            });
        }

        तार्किक grouped = असत्य;
        यदि (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == परिमाण(भिन्न)*ne10 && nb1 == परिमाण(भिन्न)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            यदि (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ध्रुव तार्किक src0_reordered =
                src0->extra && ((ध्रुव ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ध्रुव भिन्न *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (भिन्न *) dst_contiguous.get(),
                                                     expert_row_offsets.दत्त(), n_as, ne01, ne10, n_routed_पङ्क्तिकाः,
                                                     ctx.mmid_tile_कालसारणी_host, ctx.pool(), stream);
        }

        प्रति (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ध्रुव int64_t अङ्कः_src1_पङ्क्तिकाः = expert_row_counts[i02];

            यदि (अङ्कः_src1_पङ्क्तिकाः == 0) {
                अनुवर्त;
            }

            ध्रुव int64_t expert_row_विस्थापनम् = expert_row_offsets[i02];

            src0_row.दत्त = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == परिमाण(भिन्न)*ne10);
            GGML_ASSERT(nb1 == परिमाण(भिन्न)*ne0);
            src1_row.दत्त = src1_contiguous.get() + expert_row_विस्थापनम्*nb11;
            src1_row.ne[1] = अङ्कः_src1_पङ्क्तिकाः;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = अङ्कः_src1_पङ्क्तिकाः*nb11;
            src1_row.nb[3] = अङ्कः_src1_पङ्क्तिकाः*nb11;

            dst_row.दत्त = dst_contiguous.get() + expert_row_विस्थापनम्*nb1;
            dst_row.ne[1] = अङ्कः_src1_पङ्क्तिकाः;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = अङ्कः_src1_पङ्क्तिकाः*nb1;
            dst_row.nb[3] = अङ्कः_src1_पङ्क्तिकाः*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> खण्डम्_dims(1, 1, std::min((अचिह्नित पूर्णाङ्क)ne0, max_work_गणः_आकार));
            sycl::range<3> grid_dims(1, 1, n_routed_पङ्क्तिकाः);
            stream->submit([&](sycl::handler &cgh) {
                ध्रुव वर्ण *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ध्रुव mmid_row_mapping *__restrict dev_row_mapping_get =
                    dev_row_mapping.get();

                cgh.parallel_for(
                    sycl::nd_range<3>(grid_dims * खण्डम्_dims, खण्डम्_dims),
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
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

स्थिर शून्य ggml_sycl_diag_आवरणम्_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_आवरणम्_inf(ctx, dst);
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

स्थिर शून्य ggml_sycl_योग(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_योग(ctx, dst);
}

स्थिर शून्य ggml_sycl_योग_पङ्क्तिकाः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_योग_पङ्क्तिकाः(ctx, dst);
}

स्थिर शून्य ggml_sycl_माध्यम्(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_माध्यम्(ctx, dst);
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


स्थिर शून्य ggml_sycl_समुच्चयः_main_device(ध्रुव पूर्णाङ्क main_device) यत्न {
    यदि (dpct::get_current_device_id() == static_cast<अचिह्नित पूर्णाङ्क> (main_device)) {
        प्रत्यागम;
    }
    check_allow_gpu_सूचकांक(main_device);
    dpct::select_device(main_device);

    यदि (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर तार्किक ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, संरचना ggml_tensor * dst) यत्न {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    यदि (!g_sycl_loaded) प्रत्यागम असत्य;

    यदि (dst->src[0] != शून्यसूचकः && ggml_backend_बफर_is_sycl_विभज(dst->src[0]->बफर)) {
        ggml_sycl_समुच्चयः_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    विकल्प (dst->op) {
        पक्ष GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_REPEAT_पृष्ठभागः:
            ggml_sycl_repeat_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            ggml_sycl_get_पङ्क्तिकाः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
            ggml_sycl_get_पङ्क्तिकाः_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः:
            ggml_sycl_op_समुच्चयः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            ggml_sycl_op_समुच्चयः_पङ्क्तिकाः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_सङ्कलय(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_सङ्कलय_ID:
            ggml_sycl_सङ्कलय_id(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_गणनम्_EQUAL:
            ggml_sycl_गणनम्_equal(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(dst)) {
                पक्ष GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_चरणः:
                    ggml_sycl_चरणः(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                    ggml_sycl_gelu_शीघ्र(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_आवर्तनम्:
                    ggml_sycl_आवर्तनम्(ctx, dst);
                    भङ्ग;
                पक्ष GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    भङ्ग;
                स्वभाव:
                    प्रत्यागम असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(dst)) {
                पक्ष GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    भङ्ग;
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                    ggml_sycl_geglu_शीघ्र(ctx, dst);
                    भङ्ग;
                स्वभाव:
                    प्रत्यागम असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_गणः_NORM:
            ggml_sycl_गणः_norm(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            भङ्ग;
        पक्ष GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
            ggml_sycl_rms_norm_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            यदि (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                प्रत्यागम असत्य;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            यदि (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                प्रत्यागम असत्य;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            ggml_sycl_op_बहिर्गमनम्_prod(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            भङ्ग;
        पक्ष GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            भङ्ग;
        पक्ष GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
            ggml_sycl_diag_आवरणम्_inf(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः:
            ggml_sycl_op_soft_max_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
            ggml_sycl_cross_entropy_loss_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
            ggml_sycl_rope_पृष्ठभागः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_योग:
            ggml_sycl_योग(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            ggml_sycl_योग_पङ्क्तिकाः(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_माध्यम्:
            ggml_sycl_माध्यम्(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_शिखरम्_K:
            ggml_sycl_op_शिखरम्_k(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
            ggml_sycl_op_gated_रैखिकम्_attn(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_OPT_चरणः_ADAMW:
            ggml_sycl_opt_चरणः_adamw(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_OPT_चरणः_SGD:
            ggml_sycl_opt_चरणः_sgd(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SSM_अवलोकनम्:
            ggml_sycl_ssm_अवलोकनम्(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            भङ्ग;
        स्वभाव:
            प्रत्यागम असत्य;
    }

    प्रत्यागम सत्य;
} गृहाण (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

GGML_API शून्य ggml_backend_sycl_get_device_description(पूर्णाङ्क device, वर्ण *description,
                                      size_t description_आकार) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_आकार, "%s", prop.get_name());
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर तार्किक sycl_get_mem_info(पूर्णाङ्क device, size_t * मुक्त, size_t * योग) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    स्मृतिः_APIType mem_api_type = स्मृतिः_APIType::स्मृतिः_API_TYPE_SYCL;

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_परिणाम_SUCCESS ?
        (स्मृतिः_APIType) g_ggml_sycl_get_mem_api : स्मृतिः_APIType::स्मृतिः_API_TYPE_SYCL;
#अन्यथा
    mem_api_type = स्मृतिः_APIType::स्मृतिः_API_TYPE_SYCL;
#यदि_अन्तः
    तार्किक res = get_स्मृतिः_आकार(dpct::dev_mgr::instance().get_device(device),
        *मुक्त, *योग, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *योग, *मुक्त);
    प्रत्यागम res;
}

शून्य ggml_backend_sycl_get_device_स्मृतिः(पूर्णाङ्क device, size_t * मुक्त, size_t * योग) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    यदि (!sycl_get_mem_info(device, मुक्त, योग)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *मुक्त, *योग);
} गृहाण (ध्रुव sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

स्थिर ध्रुव वर्ण * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    प्रत्यागम sycl_ctx->name.c_str();
}

स्थिर शून्य ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    विलोपय sycl_ctx;
    विलोपय backend;
}

स्थिर शून्य ggml_backend_sycl_समुच्चयः_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ध्रुव शून्य *दत्त, size_t विस्थापनम्,
                                               size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) && "unsupported buffer type");
    ध्रुव queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->स्मृतिं_प्रतिलिख((वर्ण *)tensor->दत्त + विस्थापनम्, दत्त, size)));
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ध्रुव ggml_tensor *tensor,
                                               शून्य *दत्त, size_t विस्थापनम्,
                                               size_t size) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, विस्थापनम्);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_बफर_t buf = tensor->view_src ? tensor->view_src->बफर : tensor->बफर;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) && "unsupported buffer type");
    ध्रुव queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->स्मृतिं_प्रतिलिख(
        दत्त, (ध्रुव वर्ण *)tensor->दत्त + विस्थापनम्, size)));
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर तार्किक ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ध्रुव ggml_tensor *src,
                                               ggml_tensor *dst) यत्न {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    तार्किक is_cpy_supported                = dst->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device) &&
                            ggml_backend_बफर_is_sycl(src->बफर);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    यदि (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ध्रुव queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->स्मृतिं_प्रतिलिख(
            dst->दत्त, src->दत्त, ggml_nbytes(dst))));
        प्रत्यागम सत्य;
    }

    प्रत्यागम असत्य;
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_synchronize(ggml_backend_t backend) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ध्रुव queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->प्रतीक्षस्व()));

    GGML_अप्रयुक्तम्(backend);
}
गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर तार्किक ggml_sycl_is_view_or_noop(ध्रुव ggml_tensor * t) {
    प्रत्यागम ggml_is_रिक्तानि(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
स्थिर पूर्णाङ्क ggml_sycl_try_gdn_cache_fusion(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    यदि (!g_ggml_sycl_enable_fusion) {
        प्रत्यागम 0;
    }

    ध्रुव ggml_tensor * gdn = cgraph->nodes[ग्रन्थिः_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    यदि (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        प्रत्यागम 0;
    }

    ध्रुव ggml_tensor * src_v     = gdn->src[2];
    ध्रुव int64_t       S_v       = src_v->ne[0];
    ध्रुव int64_t       H         = src_v->ne[1];
    ध्रुव int64_t       n_tokens  = src_v->ne[2];
    ध्रुव int64_t       n_seqs    = src_v->ne[3];
    ध्रुव int64_t       D         = S_v * S_v * H;
    ध्रुव int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ध्रुव int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ध्रुव size_t पुच्छ_off = ggml_row_आकार(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ध्रुव ggml_tensor * cpy  = शून्यसूचकः;
    पूर्णाङ्क                 skip = 0;
    प्रति (पूर्णाङ्क j = ग्रन्थिः_idx + 1; j < cgraph->n_nodes && cpy == शून्यसूचकः; ++j) {
        ध्रुव ggml_tensor * n = cgraph->nodes[j];
        यदि (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }
        यदि (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            प्रत्यागम 0;
        }
        cpy  = n;
        skip = j - ग्रन्थिः_idx;
    }
    यदि (cpy == शून्यसूचकः) {
        प्रत्यागम 0;
    }

    ध्रुव ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ध्रुव ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    यदि (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != पुच्छ_off ||
        !ggml_is_contiguous(src)) {
        प्रत्यागम 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ध्रुव std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    यदि (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->दत्त == शून्यसूचकः ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_आकार(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_आकार(GGML_TYPE_F32, D)) {
        प्रत्यागम 0;
    }

    fused_state_cpy.दत्त        = (भिन्न *) dst->दत्त; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / परिमाण(भिन्न)) : 0;
    प्रत्यागम skip;
}

स्थिर शून्य ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_समुच्चयः_main_device(sycl_ctx->device);

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = cgraph->nodes[i];
        यदि (ggml_sycl_is_view_or_noop(ग्रन्थिः)) {
            अनुवर्त;
        }
        यदि ((ग्रन्थिः->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            अनुवर्त;
        }

        ध्रुव पूर्णाङ्क nodes_गन्तव्यम्_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        यदि (nodes_गन्तव्यम्_skip != 0) {
            i += nodes_गन्तव्यम्_skip;
            अनुवर्त;
        }
#यदि_अपरिभाषितम् NDEBUG
        प्रतिज्ञा(ग्रन्थिः->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device));
        प्रति (पूर्णाङ्क j = 0; j < GGML_MAX_SRC; j++) {
            यदि (ग्रन्थिः->src[j] != शून्यसूचकः) {
                प्रतिज्ञा(ग्रन्थिः->src[j]->बफर->buft == ggml_backend_sycl_बफर_type(sycl_ctx->device));
            }
        }
#यदि_अन्तः
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        यदि (ग्रन्थिः->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ध्रुव पूर्णाङ्क gdn_nodes_गन्तव्यम्_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            यदि (gdn_nodes_गन्तव्यम्_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ग्रन्थिः, fused_state_cpy);
                i += gdn_nodes_गन्तव्यम्_skip;
                अनुवर्त;
            }
        }
        यदि (ग्रन्थिः->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_सङ्कलय }, {})) {
            ggml_sycl_op_rms_norm_fused_सङ्कलय(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            अनुवर्त;
        }
        यदि (ग्रन्थिः->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;
            अनुवर्त;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        यदि (ग्रन्थिः->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;
            अनुवर्त;
        }
        यदि (ग्रन्थिः->op == GGML_OP_सङ्कलय &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_सङ्कलय, GGML_OP_सङ्कलय }, {})) {
            ggml_sycl_op_सङ्कलय_सङ्कलय_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;
            अनुवर्त;
        }
        यदि (ग्रन्थिः->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ग्रन्थिः) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;
            अनुवर्त;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        यदि (ग्रन्थिः->op == GGML_OP_सङ्कलय && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_सङ्कलय, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_सङ्कलय_unary_mul_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            अनुवर्त;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        यदि (ग्रन्थिः->op == GGML_OP_L2_NORM) {
            ध्रुव पूर्णाङ्क l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            यदि (l2_batch_skip > 0) {
                i += l2_batch_skip;
                अनुवर्त;
            }
        }

        यदि (ग्रन्थिः->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_सङ्कलय, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ग्रन्थिः, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            अनुवर्त;
        }

        यदि (ग्रन्थिः->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ग्रन्थिः, शून्यसूचकः, cgraph->nodes[i + 1]);
            i++;
            अनुवर्त;
        }

        यदि (ग्रन्थिः->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            अनुवर्त;
        }

        तार्किक ok = ggml_sycl_compute_forward(*sycl_ctx, ग्रन्थिः);
        यदि (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ग्रन्थिः->name, ggml_op_name(ग्रन्थिः->op));
        }
        GGML_ASSERT(ok);
    }
}

#यदि_परिभाषितम् GGML_SYCL_GRAPH
स्थिर तार्किक check_graph_compatibility(ggml_cgraph * cgraph) {
    यदि (ggml_sycl_info().device_गणनम् > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        प्रत्यागम असत्य;
    }

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ध्रुव ggml_op ग्रन्थिः_op = cgraph->nodes[i]->op;
        विकल्प (ग्रन्थिः_op) {
            स्वभाव:
                भङ्ग;
            पक्ष GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            पक्ष GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ग्रन्थिः_op));
                प्रत्यागम असत्य;
            पक्ष GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                यदि (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ग्रन्थिः_op));
                    प्रत्यागम असत्य;
                }
        }
    }
    प्रत्यागम सत्य;
}
#यदि_अन्तः

स्थिर ggml_स्थितिः ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    स्वचालित * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#यदि_परिभाषितम् GGML_SYCL_GRAPH
    तार्किक use_sycl_graph = असत्य;
    यदि (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    यदि (use_sycl_graph) {
        ध्रुव तार्किक graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        यदि (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            प्रत्यागम GGML_स्थितिः_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_बफर_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.अन्तः_recording();

        ध्रुव तार्किक graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        यदि (!sycl_ctx->exec_graph || !graph_update_support) {
            स्वचालित exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } अन्यथा {
            यत्न {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } गृहाण (sycl::exception ध्रुव & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                स्वचालित exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } अन्यथा
#यदि_अन्तः
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

स्थिर शून्य ggml_backend_sycl_event_लेखः(ggml_backend_t backend, ggml_backend_event_t event)
यत्न
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ध्रुव queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
गृहाण (sycl::exception ध्रुव &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

स्थिर शून्य ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) यत्न {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    यदि (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->प्रतीक्षस्व()));
    } अन्यथा
        GGML_ABORT("fatal error");
} गृहाण (sycl::exception ध्रुव& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
    std::निर्गच्छ(1);
}

स्थिर ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_समुच्चयः_tensor_async,
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
    /* .event_record            = */ ggml_backend_sycl_event_लेखः,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ggml_guid_t ggml_backend_sycl_guid() {
    स्थिर ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    प्रत्यागम &guid;
}

तार्किक ggml_backend_is_sycl(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

पूर्णाङ्क ggml_backend_sycl_get_device_गणनम्(शून्य) {
    प्रत्यागम ggml_sycl_info().device_गणनम्;
}


// backend device

स्थिर ध्रुव वर्ण * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम ctx->name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम ctx->description.c_str();
}

स्थिर शून्य ggml_backend_sycl_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    यदि (!sycl_get_mem_info(ctx->device, मुक्त, योग)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *योग, *मुक्त);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *मुक्त, *योग);
}

स्थिर गणना ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ध्रुव sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    यदि (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        प्रत्यागम GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);

    तार्किक host_बफर = getenv("GGML_SYCL_NO_PINNED") == शून्यसूचकः;
#यदि_परिभाषितम् GGML_SYCL_NO_PEER_COPY
    तार्किक events = असत्य;
#अन्यथा
    तार्किक events = सत्य;
#यदि_अन्तः

    props->caps = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ host_बफर,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ events,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    GGML_अप्रयुक्तम्(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम ggml_backend_sycl_init(ctx->device);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_sycl_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम ggml_backend_sycl_बफर_type(ctx->device);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_sycl_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    प्रत्यागम ggml_backend_sycl_host_बफर_type_for_device(ctx->device);
}

स्थिर ggml_backend_बफर_t ggml_backend_sycl_device_बफर_from_host_ptr(ggml_backend_dev_t dev, शून्य * ptr, size_t size, size_t max_tensor_आकार) {
    GGML_अप्रयुक्तम्(dev);
    GGML_अप्रयुक्तम्(ptr);
    GGML_अप्रयुक्तम्(size);
    GGML_अप्रयुक्तम्(max_tensor_आकार);
    प्रत्यागम शून्यसूचकः;
}

स्थिर तार्किक do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    पूर्णाङ्क device = sycl_ctx->device;
    विकल्प (op->op) {
        पक्ष GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                यदि (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    प्रत्यागम सत्य;
                }
                प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            प्रत्यागम (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_आकार(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_आकार(op->src[0]->type));
        पक्ष GGML_OP_CONV_2D_DW:
        पक्ष GGML_OP_CONV_TRANSPOSE_2D:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_UNARY:
            विकल्प (ggml_get_unary_op(op)) {
                पक्ष GGML_UNARY_OP_SGN:
                पक्ष GGML_UNARY_OP_ABS:
                पक्ष GGML_UNARY_OP_NEG:
                पक्ष GGML_UNARY_OP_चरणः:
                पक्ष GGML_UNARY_OP_RELU:
                पक्ष GGML_UNARY_OP_HARDSIGMOID:
                पक्ष GGML_UNARY_OP_TANH:
                पक्ष GGML_UNARY_OP_GELU:
                पक्ष GGML_UNARY_OP_SILU:
                पक्ष GGML_UNARY_OP_SIGMOID:
                पक्ष GGML_UNARY_OP_HARDSWISH:
                पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                पक्ष GGML_UNARY_OP_GELU_ERF:
                पक्ष GGML_UNARY_OP_EXP:
                पक्ष GGML_UNARY_OP_EXPM1:
                पक्ष GGML_UNARY_OP_SOFTPLUS:
                पक्ष GGML_UNARY_OP_ELU:
                पक्ष GGML_UNARY_OP_XIELU:
                पक्ष GGML_UNARY_OP_CEIL:
                    प्रत्यागम सत्य;
                पक्ष GGML_UNARY_OP_FLOOR:
                पक्ष GGML_UNARY_OP_आवर्तनम्:
                पक्ष GGML_UNARY_OP_TRUNC:
                    प्रत्यागम सत्य;
                स्वभाव:
                    प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_GLU:
            विकल्प (ggml_get_glu_op(op)) {
                पक्ष GGML_GLU_OP_REGLU:
                पक्ष GGML_GLU_OP_GEGLU:
                पक्ष GGML_GLU_OP_SWIGLU:
                पक्ष GGML_GLU_OP_SWIGLU_OAI:
                पक्ष GGML_GLU_OP_GEGLU_ERF:
                पक्ष GGML_GLU_OP_GEGLU_शीघ्र:
                पक्ष GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    प्रत्यागम ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                स्वभाव:
                    प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_MUL_MAT:
        पक्ष GGML_OP_MUL_MAT_ID:
            {
                संरचना ggml_tensor * a = op->src[0];
                संरचना ggml_tensor * b = op->src[1];

                यदि (a->ne[3] != b->ne[3]) {
                    प्रत्यागम असत्य;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                यदि (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  प्रत्यागम असत्य;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                यदि (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    प्रत्यागम असत्य;
                }

                यदि (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    प्रत्यागम असत्य;
                }

                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_बहिर्गमनम्_PROD:
            प्रत्यागम op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            {
                विकल्प (op->src[0]->type) {
                    पक्ष GGML_TYPE_I32:
                    पक्ष GGML_TYPE_F16:
                    पक्ष GGML_TYPE_BF16:
                    पक्ष GGML_TYPE_F32:
                    पक्ष GGML_TYPE_Q1_0:
                    पक्ष GGML_TYPE_MXFP4:
                    पक्ष GGML_TYPE_NVFP4:
                    पक्ष GGML_TYPE_IQ2_XXS:
                    पक्ष GGML_TYPE_IQ2_XS:
                    पक्ष GGML_TYPE_IQ2_S:
                    पक्ष GGML_TYPE_IQ3_XXS:
                    पक्ष GGML_TYPE_IQ1_S:
                    पक्ष GGML_TYPE_IQ1_M:
                    पक्ष GGML_TYPE_IQ3_S:
                    पक्ष GGML_TYPE_IQ4_NL:
                    पक्ष GGML_TYPE_IQ4_XS:
                    पक्ष GGML_TYPE_Q2_K:
                    पक्ष GGML_TYPE_Q3_K:
                    पक्ष GGML_TYPE_Q4_0:
                    पक्ष GGML_TYPE_Q4_1:
                    पक्ष GGML_TYPE_Q4_K:
                    पक्ष GGML_TYPE_Q5_0:
                    पक्ष GGML_TYPE_Q5_1:
                    पक्ष GGML_TYPE_Q5_K:
                    पक्ष GGML_TYPE_Q6_K:
                    पक्ष GGML_TYPE_Q8_0:
                        प्रत्यागम सत्य;
                    स्वभाव:
                        प्रत्यागम असत्य;
                }
            }
        पक्ष GGML_OP_GET_पङ्क्तिकाः_पृष्ठभागः:
            // return true;
             प्रत्यागम op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         पक्ष GGML_OP_समुच्चयः:
               प्रत्यागम (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            {
                यदि (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    प्रत्यागम असत्य;
                }
                स्वचालित res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                प्रत्यागम res;
            }
        पक्ष GGML_OP_DSV4_HC_PRE:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_DSV4_HC_COMB:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_DSV4_HC_POST:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == शून्यसूचकः || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_LIGHTNING_INDEXER:
            प्रत्यागम op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_आकार * 8;
        पक्ष GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                यदि (src0_type == GGML_TYPE_F16) {
                    यदि (src1_type == GGML_TYPE_Q2_K ||
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
                        प्रत्यागम असत्य;
                    }
                }

                यदि (src0_type == GGML_TYPE_BF16) {
                    यदि (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        प्रत्यागम असत्य;
                    }
                }

                यदि (src0_type == GGML_TYPE_F32) {
                    यदि (src1_type == GGML_TYPE_Q2_K ||
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
                        प्रत्यागम असत्य;
                    }
                }

                यदि (src1_type == GGML_TYPE_F32) {
                    यदि (src0_type == GGML_TYPE_Q1_0 ||
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
                        प्रत्यागम असत्य;
                    }
                }

                यदि (src0_type == src1_type) {
                    यदि (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        प्रत्यागम असत्य;
                    }
                }

                यदि (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    प्रत्यागम असत्य;
                }

                प्रत्यागम सत्य;
            }
        पक्ष GGML_OP_REPEAT_पृष्ठभागः:
            {
                ggml_type src0_type = op->src[0]->type;
                प्रत्यागम src0_type == GGML_TYPE_F32;
            }
        पक्ष GGML_OP_CONCAT:
        पक्ष GGML_OP_DUP:
        पक्ष GGML_OP_ARGMAX:
        पक्ष GGML_OP_NONE:
        पक्ष GGML_OP_RESHAPE:
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_ADD1:
        पक्ष GGML_OP_सङ्कलय_ID:
        पक्ष GGML_OP_SUB:
        पक्ष GGML_OP_गणनम्_EQUAL:
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_DIV:
        पक्ष GGML_OP_REPEAT:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_PAD_REFLECT_1D:
            प्रत्यागम ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_SQR:
        पक्ष GGML_OP_SQRT:
        पक्ष GGML_OP_SIN:
        पक्ष GGML_OP_COS:
        पक्ष GGML_OP_CLAMP:
        पक्ष GGML_OP_LOG:
        पक्ष GGML_OP_NORM:
        पक्ष GGML_OP_L2_NORM:
        पक्ष GGML_OP_गणः_NORM:
        पक्ष GGML_OP_RMS_NORM:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_RMS_NORM_पृष्ठभागः:
            प्रत्यागम ggml_is_contiguous(op->src[0]);
        पक्ष GGML_OP_SCALE:
            // the scale kernel is f32 only
            प्रत्यागम op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_CONT:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_TRI:
            {
                ध्रुव ggml_tensor * src0 = op->src[0];
                प्रत्यागम src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        पक्ष GGML_OP_DIAG_आवरणम्_INF:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_SOFT_MAX:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_SOFT_MAX_पृष्ठभागः: {
            भिन्न max_bias = 0.0f;
            स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) op->op_params + 1, परिमाण(भिन्न));
            प्रत्यागम max_bias == 0.0f;
        }
        पक्ष GGML_OP_ROPE:
        पक्ष GGML_OP_ROPE_पृष्ठभागः:
        पक्ष GGML_OP_IM2COL:
        पक्ष GGML_OP_IM2COL_3D:
        पक्ष GGML_OP_UPSCALE:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_COL2IM_1D:
            प्रत्यागम ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#यदि_अन्तः
                   ) &&
                   op->src[0]->type == op->type;
        पक्ष GGML_OP_CONV_3D: {
            ध्रुव int32_t * opts = (ध्रुव int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ध्रुव int64_t knl_n_योग = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ध्रुव int64_t patch_योग = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            प्रत्यागम op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_योग > 0 && patch_योग > 0;
        }
        पक्ष GGML_OP_योग:
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
        पक्ष GGML_OP_माध्यम्:
            प्रत्यागम ggml_is_contiguous(op->src[0]);
        पक्ष GGML_OP_ARGSORT:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_शिखरम्_K: {
            ध्रुव ggml_tensor * src0 = op->src[0];
            ध्रुव पूर्णाङ्क k = op->ne[0];
            प्रत्यागम src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        पक्ष GGML_OP_POOL_2D:
        पक्ष GGML_OP_POOL_1D:
        पक्ष GGML_OP_ACC:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_PAD:
            यदि (ggml_get_op_params_i32(op, 8) != 0) {
                प्रत्यागम असत्य;
            }
            प्रत्यागम सत्य;
        पक्ष GGML_OP_LEAKY_RELU:
        पक्ष GGML_OP_TIMESTEP_EMBEDDING:
        पक्ष GGML_OP_RWKV_WKV6:
        पक्ष GGML_OP_RWKV_WKV7:
        पक्ष GGML_OP_GATED_रैखिकम्_ATTN:
        पक्ष GGML_OP_GATED_DELTA_NET:
        पक्ष GGML_OP_OPT_चरणः_ADAMW:
        पक्ष GGML_OP_OPT_चरणः_SGD:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_SSM_CONV:
            प्रत्यागम op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        पक्ष GGML_OP_ROLL:
            प्रत्यागम op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_ARANGE:
            प्रत्यागम op->type == GGML_TYPE_F32;
        पक्ष GGML_OP_SSM_अवलोकनम्:
            यदि (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                प्रत्यागम (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_आकार == 0;
            } अन्यथा {
                // TODO Mamba-1 not yet ported to SYCL
                प्रत्यागम असत्य;
            }
        पक्ष GGML_OP_FILL:
        पक्ष GGML_OP_CUMSUM:
        पक्ष GGML_OP_DIAG:
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS:
        पक्ष GGML_OP_CROSS_ENTROPY_LOSS_पृष्ठभागः:
            प्रत्यागम सत्य;
        पक्ष GGML_OP_SOLVE_TRI:
            प्रत्यागम op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            प्रत्यागम ggml_sycl_flash_attn_ext_supported(device, op);
        स्वभाव:
            प्रत्यागम असत्य;
    }

    GGML_अप्रयुक्तम्(dev);
}

स्थिर तार्किक ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    तार्किक res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    प्रत्यागम res;
}

स्थिर तार्किक ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    यदि (buft->iface.get_name != ggml_backend_sycl_बफर_type_get_name) {
        प्रत्यागम असत्य;
    }
    ggml_backend_sycl_बफर_type_context * buft_ctx = (ggml_backend_sycl_बफर_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम buft_ctx->device == sycl_ctx->device;
}

स्थिर int64_t get_op_batch_आकार(ध्रुव ggml_tensor * op) {
    विकल्प (op->op) {
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            प्रत्यागम 0;
        पक्ष GGML_OP_MUL_MAT:
            प्रत्यागम op->ne[1];
        पक्ष GGML_OP_MUL_MAT_ID:
        पक्ष GGML_OP_ROPE:
            प्रत्यागम op->ne[2];
        स्वभाव:
            प्रत्यागम ggml_nrows(op);
    }
}

स्थिर तार्किक ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    प्रत्यागम get_op_batch_आकार(op) >= sycl_ctx->op_offload_min_batch_आकार;
}

स्थिर ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#यदि_परिभाषितम् GGML_SYCL_NO_PEER_COPY
    प्रत्यागम शून्यसूचकः;
#अन्यथा
  sycl::event *event_ptr = नूतन sycl::event();

  प्रत्यागम नूतन ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#यदि_अन्तः
}

स्थिर शून्य ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) यत्न {
  GGML_अप्रयुक्तम्(dev);
  यदि (event == शून्यसूचकः) {
    प्रत्यागम;
  }

  यदि (event->context != शून्यसूचकः) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    विलोपय sycl_event;
    event->context = शून्यसूचकः;
  }

  विलोपय event;
} गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}


स्थिर शून्य ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) यत्न {
  GGML_अप्रयुक्तम्(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->प्रतीक्षस्व()));
} गृहाण (sycl::exception ध्रुव &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
  std::निर्गच्छ(1);
}

स्थिर ध्रुव ggml_backend_device_i ggml_backend_sycl_device_interface = {
    /* .get_name                = */ ggml_backend_sycl_device_get_name,
    /* .get_description         = */ ggml_backend_sycl_device_get_description,
    /* .get_memory              = */ ggml_backend_sycl_device_get_स्मृतिः,
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

संरचना ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

स्थिर ध्रुव वर्ण * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_अप्रयुक्तम्(reg);
    प्रत्यागम GGML_SYCL_NAME;
}

स्थिर size_t ggml_backend_sycl_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    प्रत्यागम ctx->devices.size();
}

स्थिर ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t सूचकांक) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(सूचकांक < ctx->devices.size());
    प्रत्यागम ctx->devices[सूचकांक];
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
    uint8_t * host_बहिर्गमनम्[2] = { शून्यसूचकः, शून्यसूचकः };
    uint8_t * host_अन्तर्[2]  = { शून्यसूचकः, शून्यसूचकः };
    size_t    host_अष्टकानि  = 0;

    शून्य free_host() {
        प्रति (पूर्णाङ्क i = 0; i < 2; ++i) {
            ध्रुव sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            यदि (host_बहिर्गमनम्[i]) { sycl::मुक्त(host_बहिर्गमनम्[i], sctx); host_बहिर्गमनम्[i] = शून्यसूचकः; }
            यदि (host_अन्तर्[i])  { sycl::मुक्त(host_अन्तर्[i],  sctx); host_अन्तर्[i]  = शून्यसूचकः; }
        }
        host_अष्टकानि = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
स्थिर शून्य ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    शून्य * dst0, ध्रुव शून्य * src1, शून्य * dst1, ध्रुव शून्य * src0,
                                    size_t nbytes) {
    यदि (comm->host_अष्टकानि < nbytes) {
        q0->प्रतीक्षस्व();
        q1->प्रतीक्षस्व();
        comm->free_host();
        comm->host_बहिर्गमनम्[0] = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_अन्तर्[0]  = sycl::malloc_host<uint8_t>(nbytes, q0->get_context());
        comm->host_बहिर्गमनम्[1] = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        comm->host_अन्तर्[1]  = sycl::malloc_host<uint8_t>(nbytes, q1->get_context());
        GGML_ASSERT(comm->host_बहिर्गमनम्[0] && comm->host_अन्तर्[0] && comm->host_बहिर्गमनम्[1] && comm->host_अन्तर्[1]);
        comm->host_अष्टकानि = nbytes;
    }

    sycl::event e0 = q0->स्मृतिं_प्रतिलिख(comm->host_बहिर्गमनम्[0], src0, nbytes);
    sycl::event e1 = q1->स्मृतिं_प्रतिलिख(comm->host_बहिर्गमनम्[1], src1, nbytes);
    e0.प्रतीक्षस्व();
    e1.प्रतीक्षस्व();
    std::स्मृतिं_प्रतिलिख(comm->host_अन्तर्[0], comm->host_बहिर्गमनम्[1], nbytes);
    std::स्मृतिं_प्रतिलिख(comm->host_अन्तर्[1], comm->host_बहिर्गमनम्[0], nbytes);
    q0->स्मृतिं_प्रतिलिख(dst0, comm->host_अन्तर्[0], nbytes);
    q1->स्मृतिं_प्रतिलिख(dst1, comm->host_अन्तर्[1], nbytes);
}

शून्य * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) यत्न {
    प्रति (size_t i = 0; i < n_backends; ++i) {
        यदि (!ggml_backend_is_sycl(backends[i])) {
            प्रत्यागम शून्यसूचकः;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    यदि (n_backends != 2) {
        प्रत्यागम शून्यसूचकः;
    }

    स्वचालित * ctx = नूतन ggml_backend_sycl_comm_context;
    ctx->backends.नियोजय(backends, backends + n_backends);
    स्वचालित * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    स्वचालित * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    प्रत्यागम ctx;
}
गृहाण (ध्रुव sycl::exception &) { प्रत्यागम शून्यसूचकः; }
गृहाण (...)                     { प्रत्यागम शून्यसूचकः; }

शून्य ggml_backend_sycl_comm_free(शून्य * comm_ctx_v) {
    स्वचालित * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    यदि (comm_ctx == शून्यसूचकः) {
        प्रत्यागम;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    यदि (comm_ctx->backends.size() == 2) {
        स्वचालित * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        स्वचालित * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        यत्न {
            sctx0->stream()->प्रतीक्षस्व();
            sctx1->stream()->प्रतीक्षस्व();
            comm_ctx->free_host();
        } गृहाण (...) { /* best effort during shutdown */ }
    }

    विलोपय comm_ctx;
}

तार्किक ggml_backend_sycl_comm_allreduce_tensor(शून्य * comm_ctx_v, संरचना ggml_tensor ** tensors) यत्न {
    यदि (comm_ctx_v == शून्यसूचकः) {
        प्रत्यागम असत्य;
    }

    स्वचालित * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ध्रुव size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    यदि (n_backends != 2) {
        प्रत्यागम असत्य;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    यदि (tensors[0]->type != tensors[1]->type) {
        प्रत्यागम असत्य;
    }
    यदि (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        प्रत्यागम असत्य;
    }
    यदि (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        प्रत्यागम असत्य;
    }
    यदि (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        प्रत्यागम असत्य;
    }

    ध्रुव int64_t nelem  = ggml_nelements(tensors[0]);
    ध्रुव size_t  nbytes = ggml_nbytes(tensors[0]);
    यदि (nelem == 0) {
        प्रत्यागम सत्य;
    }

    स्वचालित * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    स्वचालित * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    यदि (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    यदि (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->दत्त;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->दत्त;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->दत्त, f16_tmp1, tensors[0]->दत्त, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((भिन्न) f16_out0[i] + (भिन्न) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((भिन्न) f16_out1[i] + (भिन्न) f16_tmp1[i]);
            });
        });
        प्रत्यागम सत्य;
    }

    भिन्न * out0 = (भिन्न *) tensors[0]->दत्त;
    भिन्न * out1 = (भिन्न *) tensors[1]->दत्त;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    स्थिर constexpr int64_t BF16_देहली = 32768;

    यदि (nelem < BF16_देहली) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        भिन्न * tmp0 = (भिन्न *) buf0;
        भिन्न * tmp1 = (भिन्न *) buf1;

        // separate SYCL contexts: a raw peer-USM q->memcpy is a silent no-op, so stage through host memory
        ggml_sycl_comm_exchange(comm_ctx, q0, q1, tmp0, tensors[1]->दत्त, tmp1, tensors[0]->दत्त, nbytes);

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
        प्रत्यागम सत्य;
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
    ध्रुव size_t bf16_अष्टकानि = nelem * परिमाण(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_अष्टकानि);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<भिन्न>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<भिन्न>(((uint32_t) inbox1[i]) << 16);
        });
    });

    प्रत्यागम सत्य;
}
गृहाण (ध्रुव sycl::exception &) { प्रत्यागम असत्य; }
गृहाण (...)                     { प्रत्यागम असत्य; }

स्थिर शून्य *ggml_backend_sycl_reg_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण *name) {
    GGML_अप्रयुक्तम्(reg);

    यदि (वाक्यं_तुलय(name, "ggml_backend_split_buffer_type") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_sycl_विभज_बफर_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    यदि (वाक्यं_तुलय(name, "ggml_backend_comm_init") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_sycl_comm_init;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_comm_free") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_sycl_comm_free;
    }
    यदि (वाक्यं_तुलय(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        प्रत्यागम (शून्य *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_अप्रयुक्तम्(name);
    प्रत्यागम शून्यसूचकः;
}

स्थिर ध्रुव ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_गणनम्,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_स्थानाङ्कः,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    स्थिर ggml_backend_reg reg;
    स्थिर तार्किक initialized = असत्य;

    {
        स्थिर std::mutex mutex;
        std::lock_guard<std::mutex> तालकम्(mutex);
        यदि (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = नूतन ggml_backend_sycl_reg_context;
            ध्रुव पूर्णाङ्क min_batch_आकार = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            प्रति (पूर्णाङ्क i = 0; i < ggml_sycl_info().device_गणनम्; i++) {
                ggml_backend_sycl_device_context * dev_ctx = नूतन ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_समुच्चयः_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_आकार = min_batch_आकार;

                ggml_backend_dev_t dev = नूतन ggml_backend_device {
                    /* .iface       = */ ggml_backend_sycl_device_interface,
                    /* .reg         = */ &reg,
                    /* .context     = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
                /* .iface       = */ ggml_backend_sycl_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = सत्य;
    }

    प्रत्यागम &reg;
}

ggml_backend_t ggml_backend_sycl_init(पूर्णाङ्क device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_सूचकांक(device);

    ggml_backend_sycl_context * ctx = नूतन ggml_backend_sycl_context(device);
    यदि (ctx == शून्यसूचकः) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        प्रत्यागम शून्यसूचकः;
    };

    ggml_backend_t sycl_backend = नूतन ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    प्रत्यागम sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
