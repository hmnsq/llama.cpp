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

#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <array>
#ଅନ୍ତର୍ଭାଵ <assert.h>
#ଅନ୍ତର୍ଭାଵ <atomic>
#ଅନ୍ତର୍ଭାଵ <cctype>
#ଅନ୍ତର୍ଭାଵ <cinttypes>
#ଅନ୍ତର୍ଭାଵ <cstddef>
#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <float.h>
#ଅନ୍ତର୍ଭାଵ <limits>
#ଅନ୍ତର୍ଭାଵ <optional>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <cmath>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <fstream>
#ଅନ୍ତର୍ଭାଵ <stdio.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <regex>

#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/backend.hpp>
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ଅନ୍ତର୍ଭାଵ <level_zero/ze_api.h>
#ଅନ୍ତର୍ଭାଵ <level_zero/zes_api.h>
#endif
#ଯଦି defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
#    include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#endif
#ଯଦି SYCL_EXT_ONEAPI_VIRTUAL_MEM
#    include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#    include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#    define GGML_SYCL_SUPPORT_VMM
#endif
#ଅନ୍ତର୍ଭାଵ <sycl/half_type.hpp>

#ଅନ୍ତର୍ଭାଵ "ggml.h"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"

#ଅନ୍ତର୍ଭାଵ "ggml-sycl/add-id.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/backend.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/common.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/element_wise.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/fwht.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/fused-gemm.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/gemm.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/getrows.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/mem.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/norm.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/presets.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/quantize.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/repeat_back.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/set_rows.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/set.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/dsv4-hc.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/lightning-indexer.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/conv2d.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/conv2d-dw.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/conv2d-transpose.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/ssm_conv.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/sycl_hw.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/ssm_scan.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/fill.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/cumsum.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/diag.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/opt-step.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/solve_tri.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/gated_delta_net.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/pool.hpp"
#ଅନ୍ତର୍ଭାଵ "ggml-sycl/cross_entropy_loss.hpp"

#define MEM_SIZE_2M	0x00200000
#define MEM_SIZE_1G	0x40000000

ସ୍ଥିର ତାର୍କିକ g_sycl_loaded = ମିଥ୍ୟା;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_debug = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dev_debug = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_optimize = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_graph = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_dnn = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_fa_onednn = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_fa_onednn_max_kv = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_mkl_fa = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_memtrace = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_memtrace_step = 64;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_vmm = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_fusion = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_esimd = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_mmvq_wide = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_prioritize_dmmv = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_xmx_gather_types = GGML_SYCL_XMX_GATHER_TYPES_DEFAULT;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_xmx_gather_shapes = GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dynamic_precision = GGML_SYCL_DYNAMIC_PRECISION_DEFAULT;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dynamic_required_precision = GGML_SYCL_DYNAMIC_PRECISION_F32;
ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_sycl_dynamic_precision_names[] = { "F16", "BF16", "TF32", "F32" };

// value of a GGML_SYCL_DYNAMIC_PRECISION-style variable; def if unset or invalid
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_get_env_precision(ଅଚଳ ଅକ୍ଷର * name, ପୂର୍ଣ୍ଣସଂଖ୍ୟା def) {
    ଅଚଳ ଅକ୍ଷର * env = getenv(name);
    ଯଦି (!env) {
        ଫେରନ୍ତୁ def;
    }
    std::string mode(env);
    ପାଇଁ (ଅକ୍ଷର & c : mode) {
        c = (ଅକ୍ଷର) std::toupper((unsigned ଅକ୍ଷର) c);
    }
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = GGML_SYCL_DYNAMIC_PRECISION_F16; i <= GGML_SYCL_DYNAMIC_PRECISION_F32; i++) {
        ଯଦି (mode == ggml_sycl_dynamic_precision_names[i]) {
            ଫେରନ୍ତୁ i;
        }
    }
    GGML_LOG_WARN("%s: unknown %s=%s, using %s\n", __func__, name, env, ggml_sycl_dynamic_precision_names[def]);
    ଫେରନ୍ତୁ def;
}
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_use_async_mem_op = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_use_async_mem_op_requested = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_use_level_zero_api = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_flash_attention = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_usm_system = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_host_pinned_mem = 1;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_host_pinned_mem_2g = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_upload_staging_slots = 4;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_LEVEL_ZERO;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_enable_sparse_fa = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_debug_sparse_fa = 0;
ପୂର୍ଣ୍ଣସଂଖ୍ୟା g_ggml_sycl_sparse_fa_margin = 256;

#ଯଦି GGML_SYCL_DNNL
// ask oneDNN which matmul it picks for a small f16 problem
ସ୍ଥିର ତାର୍କିକ ggml_sycl_dnnl_detect_optimized_gemm(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    using dt = dnnl::memory::data_type;

    ଚେଷ୍ଟା {
        ଅଚଳ dnnl::memory::dims dims    = { 1, 64, 64 };
        ଅଚଳ dnnl::memory::dims strides = { 64 * 64, 64, 1 };

        dnnl::primitive_attr attr;
        attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

        ଅଚଳ auto & q   = dpct::dev_mgr::instance().get_device(device).default_queue();
        ଅଚଳ auto   eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
        ଅଚଳ auto   pd  = dnnl::matmul::primitive_desc(eng,
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f16, strides),
                                                        dnnl::memory::desc(dims, dt::f32, strides), attr);

        ଅଚଳ std::string impl = pd.impl_info_str();
        ଯଦି (impl.find("ref") == std::string::npos) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }

        GGML_LOG_WARN("%s: oneDNN has no optimized matmul for device %d (picks %s), using SYCL kernels\n",
                      __func__, device, impl.c_str());
    } ଧର (ଅଚଳ std::exception & e) {
        GGML_LOG_WARN("%s: oneDNN matmul probe failed on device %d (%s), using SYCL kernels\n",
                      __func__, device, e.what());
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}
#endif

ସ୍ଥିର ggml_sycl_device_info ggml_sycl_init() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_sycl_init\n");
    ggml_sycl_device_info info = {};

    // Do not hard crash when there exists no SYCL devices.
    // We want to allow the user to use non-SYCL tools when SYCL is compiled (such as llama-quantize)
    ଚେଷ୍ଟା {
        info.device_ଗଣନା = dpct::dev_mgr::instance().device_ଗଣନା();
    } ଧର (sycl::exception ଅଚଳ &exc) {
        GGML_LOG_INFO("%s: no SYCL device available: %s\n", __func__, exc.what());
        info.device_ଗଣନା = 0;
    }
    ଯଦି (info.device_ଗଣନା == 0) {
        GGML_LOG_ERROR("%s: failed to initialize: %s\n", GGML_SYCL_NAME, __func__);
        ଫେରନ୍ତୁ info;
    }

    GGML_ASSERT(info.device_ଗଣନା <= GGML_SYCL_MAX_DEVICES);

    int64_t total_vram = 0;
/* This is a bit misleading;  reserved for later */
// #if defined(SYCL_USE_XMX)
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: yes\n", __func__);
// #else
//     GGML_LOG_INFO("%s: SYCL_USE_XMX: no\n", __func__);
// #endif
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < info.device_ଗଣନା; ++i) {
        dpct::device_info prop;
        auto & device = dpct::dev_mgr::instance().get_device(i);

        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, device)));

#ଯଦି !defined(GGML_SYCL_SUPPORT_VMM)
        info.devices[i].vmm = 0;
#ଅନ୍ୟଥା
        info.devices[i].vmm = device.has(sycl::aspect::ext_oneapi_virtual_mem);
        ଯଦି (info.devices[i].vmm) {
            // NB: SYCL's get_mem_granularity always returns the _minimum_ granularity,
            // but the L0 API requires a larger page size for allocs above 2 MiB and
            // rejects non-multiples with UR_RESULT_ERROR_INVALID_VALUE [sic].
            // Here we clamp it to 2 MiB for simplicity, but other devices may require
            // calling zeVirtualMemQueryPageSize or yet unexposed public API.
            ଅଚଳ size_t physical_page = 2ull << 20; // 2 MiB
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
#ଯଦି GGML_SYCL_DNNL
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
        ଯଦି (device.is_gpu() && device.default_queue().get_backend() != sycl::backend::ext_oneapi_level_zero) {
            GGML_LOG_WARN("SYCL GPU device %d does not use Level Zero backend, disabling Level Zero memory API\n", i);
            info.ext_oneapi_level_zero = ମିଥ୍ୟା;
        }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        ଯଦି (info.ext_oneapi_level_zero && device.is_gpu() && device.default_queue().get_backend() == sycl::backend::ext_oneapi_level_zero) {
            ze_device_handle_t ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(device.default_queue().get_device());
            ze_device_properties_t props = {};
            props.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
            ze_result_t r = zeDeviceGetProperties(ze_dev, &props);
            ଯଦି (r == ZE_RESULT_SUCCESS) {
                info.devices[i].l0_device_type_valid = ସତ୍ୟ;
                info.devices[i].l0_discrete_gpu = !(props.flags & ZE_DEVICE_PROPERTY_FLAG_INTEGRATED);
            }
        }
#endif
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < info.device_ଗଣନା; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //update g_ggml_sycl_use_level_zero_api according to the device support
    g_ggml_sycl_use_level_zero_api =
        info.ext_oneapi_level_zero && g_ggml_sycl_use_level_zero_api;
#endif

    ଫେରନ୍ତୁ info;
}

ଅଚଳ ggml_sycl_device_info & ggml_sycl_info() {
    ସ୍ଥିର ggml_sycl_device_info info = ggml_sycl_init();
    ଫେରନ୍ତୁ info;
}

ସ୍ଥିର ଶୂନ୍ୟ print_device_detail(ପୂର୍ଣ୍ଣସଂଖ୍ୟା id, sycl::device &device, std::string device_type) {

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

ସ୍ଥିର ଶୂନ୍ୟ print_device_opt_feature(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_ଗଣନା) {
    GGML_LOG_INFO("SYCL Optimization Feature:\n");
    GGML_LOG_INFO(
        "|ID|        Device Type|Reorder|\n");
    GGML_LOG_INFO(
        "|--|-------------------|-------|\n");
    std::map<std::string, size_t> DeviceNums;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < device_ଗଣନା; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      std::string device_type_s = device_type.str();
      device_type_s = std::regex_replace(device_type_s, std::regex("ext_oneapi_"), "");
      GGML_LOG_INFO("|%2d|%19s|%7s|\n", id, device_type_s.c_str(),
        ggml_sycl_info().devices[id].opt_feature.reorder ? "Y": "N");
    }

}
ଶୂନ୍ୟ ggml_backend_sycl_print_sycl_devices() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_print_sycl_devices\n");
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_ଗଣନା = dpct::dev_mgr::instance().device_ଗଣନା();
    std::map<std::string, size_t> DeviceNums;
    GGML_LOG_INFO("Found %d SYCL devices:\n", device_ଗଣନା);

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

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < device_ଗଣନା; ++id) {
      sycl::device device = dpct::dev_mgr::instance().get_device(id);
      std::string backend_type = get_device_backend_and_type(device);
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା type_id = DeviceNums[backend_type]++;
      std::stringstream device_type;
      device_type << "[" << backend_type << ":" << std::to_string(type_id)
                  << "]";
      print_device_detail(id, device, device_type.str());
    }

    print_device_opt_feature(device_ଗଣନା);
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର* dev2dev_int2str(ପୂର୍ଣ୍ଣସଂଖ୍ୟା dev2dev) {
    ଯଦି (dev2dev == DEV2DEV_MEMCPY_SYCL) {
        ଫେରନ୍ତୁ "SYCL API";
    } ଅନ୍ୟଥା ଯଦି (dev2dev == DEV2DEV_MEMCPY_L0) {
        ଫେରନ୍ତୁ "Level Zero API";
    } ଅନ୍ୟଥା ଯଦି (dev2dev == DEV2DEV_MEMCPY_FORWARD) {
        ଫେରନ୍ତୁ "Host Forward";
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ "Unknown";
    }
}

/*
* There are several entry APIs to be called as first function in SYCL backend in different cases.
* It's the first internal function to be called by them in SYCL backend.
* This function is used to do initialize work for the SYCL backend and set the global variables.
*/
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ସ୍ଥିର ze_result_t init_zes() {
    ze_result_t res = zesInit(0);
    ଯଦି (res != ZE_RESULT_SUCCESS) {
        GGML_SYCL_DEBUG("Warning: [%s] zesInit failed with code %d. Sysman free-memory query be unavailable.\n",
                            __func__, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) res);
    }
    ଫେରନ୍ତୁ res;
}

ସ୍ଥିର ze_result_t get_zes_init_res() {
    ସ୍ଥିର ze_result_t zes_init_res = init_zes();
    GGML_SYCL_DEBUG("[SYCL] call %s: zesInit result: %d\n", __func__, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) zes_init_res);
    ଫେରନ୍ତୁ zes_init_res;
}
#endif

ସ୍ଥିର ଶୂନ୍ୟ initialize_sycl_begining() {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    //must be called in initialization stage, before any other Level Zero API calls
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    get_zes_init_res();
#endif
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_check_sycl() ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] ggml_check_sycl()\n");
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;

    ଯଦି (!initialized) {
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
#ଅନ୍ୟଥା
        g_ggml_sycl_use_level_zero_api = 0;
#endif
        g_ggml_sycl_dev2dev_memcpy = ggml_sycl_get_env("GGML_SYCL_DEV2DEV_MEMCPY", DEV2DEV_MEMCPY_SYCL);
        g_ggml_sycl_get_mem_api = ggml_sycl_get_env("GGML_SYCL_GET_MEM_API", MEMORY_API_TYPE_LEVEL_ZERO);
        ଯଦି (g_ggml_sycl_use_level_zero_api == 0) {
            g_ggml_sycl_dev2dev_memcpy = DEV2DEV_MEMCPY_SYCL;
            g_ggml_sycl_get_mem_api = MEMORY_API_TYPE_SYCL;
        }

#ifdef SYCL_FLASH_ATTN
        g_ggml_sycl_enable_flash_attention = ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#ଅନ୍ୟଥା
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
#ଯଦି defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_DNNL: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_DNNL: no\n");
#endif

#ଯଦି defined(GGML_SYCL_F16)
        GGML_LOG_INFO("  GGML_SYCL_F16: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_F16: no\n");
#endif

#ଯଦି defined(GGML_SYCL_FORCE_MMQ)
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_FORCE_MMQ: no\n");
#endif

#ଯଦି defined(GGML_SYCL_GRAPH)
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_GRAPH: no\n");
#endif

#ଯଦି defined(GGML_SYCL_SUPPORT_LEVEL_ZERO_API)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_LEVEL_ZERO_API: no\n");
#endif
#ଯଦି defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: yes\n");
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_SUPPORT_VMM: no\n");
#endif

        //Print the running environment variables for SYCL backend
        GGML_LOG_INFO("Running with Environment Variables:\n");
        GGML_LOG_INFO("  GGML_SYCL_DEBUG: %d\n", g_ggml_sycl_debug);
        GGML_LOG_INFO("  GGML_SYCL_DEV_DEBUG: %d\n", g_ggml_sycl_dev_debug);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s)\n", g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s)\n", g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_DEV2DEV_MEMCPY: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_dev2dev_memcpy, dev2dev_int2str(g_ggml_sycl_dev2dev_memcpy));
        GGML_LOG_INFO("  GGML_SYCL_GET_MEM_API: %d (%s), enable to SYCL API since missing GGML_SYCL_SUPPORT_LEVEL_ZERO_API\n",
                      g_ggml_sycl_get_mem_api, mem_api_int2str(g_ggml_sycl_get_mem_api));
#endif

#ଯଦି defined(GGML_SYCL_DNNL)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: %d\n", g_ggml_sycl_enable_dnn);
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_DNN: DNN disabled by compile flag\n");
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN: %d\n", g_ggml_sycl_fa_onednn);
#endif
        GGML_LOG_INFO("  GGML_SYCL_FA_ONEDNN_MAX_KV: %d\n", g_ggml_sycl_fa_onednn_max_kv);
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_MKL_FA: %d\n", g_ggml_sycl_enable_mkl_fa);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE: %d\n", g_ggml_sycl_memtrace);
        GGML_LOG_INFO("  GGML_SYCL_MEMTRACE_STEP: %d\n", g_ggml_sycl_memtrace_step);
#ifdef SYCL_FLASH_ATTN
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d\n", g_ggml_sycl_enable_flash_attention);
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FLASH_ATTN: %d disabled by compile flag\n",
            g_ggml_sycl_enable_flash_attention);
#endif

#ifdef GGML_SYCL_GRAPH
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: %d\n", g_ggml_sycl_enable_graph);
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_GRAPH: graph disabled by compile flag\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_OPT: %d\n", g_ggml_sycl_enable_optimize);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_TYPES: %d\n", g_ggml_sycl_xmx_gather_types);
        GGML_LOG_INFO("  GGML_SYCL_XMX_GATHER_SHAPES: %d\n", g_ggml_sycl_xmx_gather_shapes);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_precision]);
        GGML_LOG_INFO("  GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: %s\n",
                      ggml_sycl_dynamic_precision_names[g_ggml_sycl_dynamic_required_precision]);

#ଯଦି defined(GGML_SYCL_SUPPORT_VMM)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: %d\n", g_ggml_sycl_enable_vmm);
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_VMM: virtual memory extension is not available\n");
#endif

        GGML_LOG_INFO("  GGML_SYCL_ENABLE_FUSION: %d\n", g_ggml_sycl_enable_fusion);
        GGML_LOG_INFO("  GGML_SYCL_UPLOAD_STAGING_SLOTS: %d\n", g_ggml_sycl_upload_staging_slots);

#ଯଦି defined(__INTEL_LLVM_COMPILER)
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d\n", g_ggml_sycl_enable_esimd);
#ଅନ୍ୟଥା
        GGML_LOG_INFO("  GGML_SYCL_ENABLE_ESIMD: %d disabled by compile flag\n", g_ggml_sycl_enable_esimd);
#endif
        GGML_LOG_INFO("  GGML_SYCL_MMVQ_WIDE: %d\n", g_ggml_sycl_mmvq_wide);
        GGML_LOG_INFO("  GGML_SYCL_PRIORITIZE_DMMV: %d\n", g_ggml_sycl_prioritize_dmmv);

        g_ggml_sycl_use_async_mem_op_requested = ggml_sycl_get_env("GGML_SYCL_USE_ASYNC_MEM_OP", 1);
        GGML_LOG_INFO("  GGML_SYCL_USE_ASYNC_MEM_OP: %d\n", g_ggml_sycl_use_async_mem_op_requested);

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_LOG_INFO("  GGML_SYCL_USE_LEVEL_ZERO_API: %d\n", g_ggml_sycl_use_level_zero_api);
#ଅନ୍ୟଥା
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
#ଯଦି defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
        g_ggml_sycl_use_async_mem_op = g_ggml_sycl_use_async_mem_op_requested || g_ggml_sycl_enable_graph;
        ଯଦି (g_ggml_sycl_use_async_mem_op) {
            ପାଇଁ (unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < dpct::dev_mgr::instance().device_ଗଣନା(); ++i) {
                ଯଦି (!dpct::dev_mgr::instance().get_device(i).has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
                    g_ggml_sycl_use_async_mem_op = 0;
                    ଭାଙ୍ଗ;
                }
            }
        }
#endif
        ଯଦି (CHECK_TRY_ERROR(g_all_sycl_device_ଗଣନା =
                            dpct::dev_mgr::instance().device_ଗଣନା()) != 0) {
            initialized = ସତ୍ୟ;
            g_sycl_loaded = ମିଥ୍ୟା;
            ଫେରନ୍ତୁ;
        }
        GGML_ASSERT(g_all_sycl_device_ଗଣନା <= GGML_SYCL_MAX_DEVICES);

        initialized = ସତ୍ୟ;
        g_sycl_loaded = ସତ୍ୟ;
        ggml_backend_sycl_print_sycl_devices();
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

/*
device_index: device index from 0 to n (continue numbers).
    It is used for device select/set in SYCL backend internal data structure.
*/
ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ check_allow_gpu_index(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_index) {
  ଯଦି (device_index >= ggml_sycl_info().device_ଗଣନା) {
    ଅକ୍ଷର error_buf[256];
    snprintf(
        error_buf,
        ଆକାର(error_buf),
        "%s error: device_index:%d is out of range: [0-%d]",
        __func__,
        device_index,
        ggml_sycl_info().device_ଗଣନା - 1);
    GGML_LOG_ERROR("%s\n", error_buf);
    ପ୍ରତିଜ୍ଞା(ମିଥ୍ୟା);
  }
}

GGML_API ଶୂନ୍ୟ ggml_backend_sycl_get_gpu_list(ପୂର୍ଣ୍ଣସଂଖ୍ୟା *id_list, ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_len) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_gpu_list\n");
    ପାଇଁ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0;i<max_len;i++) id_list[i] = -1;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0;i< ggml_sycl_info().device_ଗଣନା;i++){
        ଯଦି (i>=max_len) ଭାଙ୍ଗ;
        id_list[i] = i;
    }
    ଫେରନ୍ତୁ;
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ free_aligned_mem_host(ଶୂନ୍ୟ * memblock) {
#ifdef _WIN32
    _aligned_free(memblock);
#ଅନ୍ୟଥା
    ମୁକ୍ତ(memblock);
#endif
}

// sycl buffer

ସଂରଚନା ggml_backend_sycl_buffer_context {
    // pinned staging for uploads; the host fills one slot while the previous one transfers
    ସ୍ଥିର constexpr size_t staging_slot_size = 8*1024*1024;

    ସଂରଚନା host_staging {
        ଶୂନ୍ୟ * data = nullptr;
        std::vector<sycl::event> events;
        std::vector<ତାର୍କିକ> submitted;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା next = 0;
    };

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    ଶୂନ୍ୟ * dev_ptr = nullptr;
    queue_ptr stream;
    std::string name;
    optimize_feature opt_feature;
    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    host_staging staging;
    ତାର୍କିକ is_usm_system;

    ggml_backend_sycl_buffer_context(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ଶୂନ୍ୟ * dev_ptr, queue_ptr stream, ତାର୍କିକ is_usm_system) :
        device(device), dev_ptr(dev_ptr), stream(stream), is_usm_system(is_usm_system) {
            check_allow_gpu_index(device);
            name = (GGML_SYCL_NAME + std::to_string(device));
            opt_feature = ggml_sycl_info().devices[device].opt_feature;
        }

    // waits for every queued upload, then releases the pinned block
    ଶୂନ୍ୟ drop_host_staging() {
        ପାଇଁ (size_t i = 0; i < staging.submitted.size(); ++i) {
            ଯଦି (staging.submitted[i]) {
                staging.events[i].wait_and_throw();
                staging.submitted[i] = ମିଥ୍ୟା;
            }
        }
        ଯଦି (staging.data != nullptr) {
            sycl::ମୁକ୍ତ(staging.data, *stream);
            staging.data = nullptr;
        }
    }

    ~ggml_backend_sycl_buffer_context() {
        drop_host_staging();
        ଯଦି (dev_ptr != nullptr) {
            ggml_sycl_set_device(device);
            ଯଦି (is_usm_system)
                free_aligned_mem_host(dev_ptr);
            ଅନ୍ୟଥା
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(dev_ptr, *stream)));
        }

        //release extra used by tensors
        ପାଇଁ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra);
        }

    }
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft);

ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_sycl(ggml_backend_buffer_t buffer) {
    ଫେରନ୍ତୁ buffer->buft->iface.get_name == ggml_backend_sycl_buffer_type_get_name;
}

ସ୍ଥିର ଶୂନ୍ୟ
ggml_backend_sycl_buffer_free_buffer(ggml_backend_buffer_t buffer) ଚେଷ୍ଟା {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    delete ctx;
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_sycl_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ଫେରନ୍ତୁ ctx->dev_ptr;
}

ସ୍ଥିର ଗଣନା ggml_status
ggml_backend_sycl_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                     ggml_tensor *tensor) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *)buffer->context;

    ଯଦି (tensor->view_src != ଖାଲି) {
        ପ୍ରତିଜ୍ଞା(tensor->view_src->buffer->buft == buffer->buft);
        ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
    }

    ଯଦି (g_ggml_sycl_enable_optimize) {
        // set reorder extra buffer based on supported type
        ବାଛ (tensor->type) {
            ସ୍ଥିତି GGML_TYPE_Q4_0:
            ସ୍ଥିତି GGML_TYPE_Q8_0:
            ସ୍ଥିତି GGML_TYPE_Q2_K:
            ସ୍ଥିତି GGML_TYPE_Q3_K:
            ସ୍ଥିତି GGML_TYPE_Q4_K:
            ସ୍ଥିତି GGML_TYPE_Q5_K:
            ସ୍ଥିତି GGML_TYPE_Q6_K:
            ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
            ସ୍ଥିତି GGML_TYPE_IQ3_S:
            ସ୍ଥିତି GGML_TYPE_MXFP4:{
                ggml_tensor_extra_gpu * extra = ନୂତନ ggml_tensor_extra_gpu{};
                tensor->extra                 = extra;
                ctx->tensor_extras.push_back(extra);
                ଭାଙ୍ଗ;
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
        }
    }

    ଯଦି (ggml_is_quantized(tensor->type)) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        ଯଦି (padded_size > original_size && tensor->view_src == nullptr) {
            SYCL_CHECK(CHECK_TRY_ERROR(ctx->stream->memset(
                (ଅକ୍ଷର *)tensor->data + original_size, 0,
                padded_size - original_size).wait()));
        }
    }
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                ggml_tensor *tensor,
                                                ଅଚଳ ଶୂନ୍ୟ *data, size_t offset,
                                                size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;
    ggml_sycl_set_device(ctx->device);

    // copy through pinned memory so the device never reads mmap()ed pages directly
    // chunks pipeline on the in-order compute queue, so no drain per tensor is needed
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_slots = g_ggml_sycl_upload_staging_slots;
    ଯଦି (n_slots > 0 && ctx->staging.data == nullptr) {
        ctx->staging.data = sycl::malloc_host(n_slots * ctx->staging_slot_size, *ctx->stream);
        ଯଦି (ctx->staging.data != nullptr) {
            ctx->staging.events.resize(n_slots);
            ctx->staging.submitted.assign(n_slots, ମିଥ୍ୟା);
        }
    }
    ଯଦି (ctx->staging.data != nullptr) {
        queue_ptr    stream    = ctx->stream;
        ଅକ୍ଷର *       dst       = (ଅକ୍ଷର *) tensor->data + offset;
        ଅଚଳ ଅକ୍ଷର * src       = (ଅଚଳ ଅକ୍ଷର *) data;
        size_t       remaining = size;
        ଯେତେବେଳେ (remaining > 0) {
            ଅଚଳ size_t chunk = std::min(remaining, ctx->staging_slot_size);
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    slot  = ctx->staging.next;
            ctx->staging.next = (ctx->staging.next + 1) % (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ctx->staging.submitted.size();
            ଯଦି (ctx->staging.submitted[slot]) {
                ctx->staging.events[slot].wait_and_throw();
            }
            ଶୂନ୍ୟ * stage = (ଅକ୍ଷର *) ctx->staging.data + slot * ctx->staging_slot_size;
            memcpy(stage, src, chunk);
            ctx->staging.events[slot] = stream->memcpy(dst, stage, chunk);
            ctx->staging.submitted[slot] = ସତ୍ୟ;
            src       += chunk;
            dst       += chunk;
            remaining -= chunk;
        }
        ଫେରନ୍ତୁ;
    }

    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::dev_mgr::instance().get_device(ctx->device).queues_wait_and_throw()));
#ifndef _WIN32
    // Note: Use host buffer to save the data from mmap(), then copy to device. It's workaround for mmap() issue on PVC GPU.
    // This function will be called during load model from disk. Use memory buffer replace dynamic won't save more time and brings potential memory leak risk here.
    ଅକ୍ଷର * host_buf = (ଅକ୍ଷର *) malloc(size);
    memcpy(host_buf, data, size);
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((ଅକ୍ଷର *) tensor->data + offset, host_buf, size).wait()));
    ମୁକ୍ତ(host_buf);
#ଅନ୍ୟଥା
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy((ଅକ୍ଷର *) tensor->data + offset, data, size).wait()));
#endif
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                ଅଚଳ ggml_tensor *tensor,
                                                ଶୂନ୍ୟ *data, size_t offset,
                                                size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_buffer_context * ctx = ( ggml_backend_sycl_buffer_context *)buffer->context;

    ggml_sycl_set_device(ctx->device);
    auto stream = dpct::dev_mgr::instance().get_device(ctx->device).default_queue();

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream.memcpy(data, (ଅଚଳ ଅକ୍ଷର *)tensor->data + offset, size)
            .wait()));
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
ସ୍ଥିର ତାର୍କିକ ggml_sycl_is_l0_discrete_gpu(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଫେରନ୍ତୁ ggml_sycl_info().devices[device].l0_discrete_gpu;
}
#endif

ସ୍ଥିର ଶୂନ୍ୟ memcpy_host_forward(sycl::queue &q_dst, sycl::queue &q_src, ଶୂନ୍ୟ *ptr_dst,
                         ଅଚଳ ଶୂନ୍ୟ *ptr_src, size_t size) {
    ଅକ୍ଷର *host_buf = (ଅକ୍ଷର *)malloc(size);
    ଯଦି (host_buf == nullptr) {
        GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory for host staging\n", __func__, size);
        GGML_ABORT("fatal error");
    }
    q_src.memcpy(host_buf, (ଅଚଳ ଅକ୍ଷର *)ptr_src, size).wait();
    q_dst.memcpy((ଅକ୍ଷର *)ptr_dst, host_buf, size).wait();
    ମୁକ୍ତ(host_buf);
}

ସ୍ଥିର ଶୂନ୍ୟ dev2dev_memcpy(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_dst, sycl::queue &q_dst, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_src, sycl::queue &q_src, ଶୂନ୍ୟ *ptr_dst,
                    ଅଚଳ ଶୂନ୍ୟ *ptr_src, size_t size) {

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    ଯଦି (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_L0) {
        // Use Level Zero direct copy for dGPU-to-dGPU transfers.
        ଅଚଳ ତାର୍କିକ l0_copy_supported =
            ggml_sycl_is_l0_discrete_gpu(device_dst) && ggml_sycl_is_l0_discrete_gpu(device_src);
        ଯଦି (g_ggml_sycl_use_level_zero_api && l0_copy_supported) {
            auto ze_ctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_context());
            auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q_dst.get_device());
            ze_command_queue_desc_t cq_desc = {ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC, nullptr, 0, 0,
                                            0, ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS, ZE_COMMAND_QUEUE_PRIORITY_NORMAL};
            ze_command_list_handle_t cl;
            ze_result_t r = zeCommandListCreateImmediate(ze_ctx, ze_dev, &cq_desc, &cl);
            ଯଦି (r == ZE_RESULT_SUCCESS) {
                GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by L0\n");
                r = zeCommandListAppendMemoryCopy(cl, ptr_dst, ptr_src, size, nullptr, 0, nullptr);
                zeCommandListDestroy(cl);
                ଯଦି (r == ZE_RESULT_SUCCESS) {
                    ଫେରନ୍ତୁ;
                }
            }
        }
    }
#endif

    ଯଦି (g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_SYCL) {
        ଯଦି (q_dst.get_device().ext_oneapi_can_access_peer(q_src.get_device(),
                                                          sycl::ext::oneapi::peer_access::access_supported)) {
            GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by SYCL\n");
            q_dst.get_device().ext_oneapi_enable_peer_access(q_src.get_device());
            SYCL_CHECK(CHECK_TRY_ERROR(q_dst.memcpy(ptr_dst, ptr_src, size).wait()));
            ଫେରନ୍ତୁ;
        }
    }

    // Host-staged copy
    ଯଦି(g_ggml_sycl_dev2dev_memcpy == DEV2DEV_MEMCPY_FORWARD) {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for setting GGML_SYCL_DEV2DEV_MEMCPY=2\n");
    } ଅନ୍ୟଥା {
        GGML_SYCL_DEBUG("[SYCL] dev2dev memcpy by host forward for SYCL/L0 fallback\n");
    }
    memcpy_host_forward(q_dst, q_src, ptr_dst, ptr_src, size);
}

ସ୍ଥିର ତାର୍କିକ
ggml_backend_sycl_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                    ଅଚଳ ggml_tensor *src,
                                    ggml_tensor *dst) ଚେଷ୍ଟା {
    ତାର୍କିକ is_cpy_supported = ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ଯଦି (is_cpy_supported) {
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
#ଯଦି 0
        SYCL_CHECK(CHECK_TRY_ERROR((*stream).memcpy(
            (ଅକ୍ଷର *)dst->data, (ଅଚଳ ଅକ୍ଷର *)src->data, size).wait()));

        /*
        DPCT1009:201: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        SYCL_CHECK(CHECK_TRY_ERROR(
            dpct::dev_mgr::instance().get_device(dst_ctx->device).queues_wait_and_throw()));
#endif
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
    GGML_UNUSED(buffer);
} ଧର (ଅଚଳ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_buffer_clear(ggml_backend_buffer_t buffer,
                                           uint8_t value) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, buffer->size);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ggml_sycl_set_device(ctx->device);
    queue_ptr stream = ctx->stream;
    SYCL_CHECK(
        CHECK_TRY_ERROR(dpct::get_current_device().queues_wait_and_throw()));

    constexpr size_t MAX_CHUNK = 2ULL << 30;  // 2 GiB
    ପାଇଁ (size_t off = 0; off < buffer->size; off += MAX_CHUNK) {
        size_t chunk = std::min(buffer->size - off, MAX_CHUNK);
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memset(static_cast<ଅକ୍ଷର*>(ctx->dev_ptr) + off, value, chunk)
                .wait()
        ));
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value,
                                                   size_t offset, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu value=%u\n", size, offset, value);
    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;
    SYCL_CHECK(ggml_sycl_set_device(ctx->device));
    auto stream = &(dpct::dev_mgr::instance().get_device(ctx->device).default_queue());
    ଯଦି (size == 0) {
        ଫେରନ୍ତୁ;  // Nothing to do
    }
    ଯଦି (tensor->data == nullptr) {
        GGML_ABORT("Error: Tensor data pointer is null.\n");
    }
    ଶୂନ୍ୟ * target_ptr = static_cast<ଅକ୍ଷର *>(tensor->data) + offset;
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).memset(target_ptr, value, size)));
    SYCL_CHECK(CHECK_TRY_ERROR((*stream).wait()));
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ଯଦି (buffer == nullptr) {
        ଫେରନ୍ତୁ;
    }

    ggml_backend_sycl_buffer_context * ctx = (ggml_backend_sycl_buffer_context *) buffer->context;

    ଯଦି (ctx != nullptr) {
        ପାଇଁ (ggml_tensor_extra_gpu * extra : ctx->tensor_extras) {
            release_extra_gpu(extra);
        }
        ctx->tensor_extras.clear();  // reset the tensor_extras vector
    }
}

ସ୍ଥିର ଅଚଳ ggml_backend_buffer_i ggml_backend_sycl_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_sycl_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_sycl_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_buffer_get_tensor,
    /* .set_tensor_2d   = */ ଖାଲି,
    /* .get_tensor_2d   = */ ଖାଲି,
    /* .cpy_tensor      = */ ggml_backend_sycl_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_sycl_buffer_clear,
    /* .reset           = */ ggml_backend_sycl_buffer_reset,
};

// sycl buffer type
ସଂରଚନା ggml_backend_sycl_buffer_type_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    std::string name;

    // each buffer type has its own stream
    queue_ptr stream = nullptr;
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;

    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ତାର୍କିକ check_usm_system(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t size) {
    ତାର୍କିକ use_usm_system = g_ggml_sycl_usm_system && size >= ((size_t)4 * MEM_SIZE_1G);

    ଯଦି (use_usm_system && !ggml_sycl_info().devices[device].usm_system_support) {
        GGML_LOG_INFO("Device does not support USM system allocations\n");
        use_usm_system = ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ use_usm_system;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * aligned_malloc_host(size_t alignment, size_t size) {
#ifdef _WIN32
    ଫେରନ୍ତୁ _aligned_malloc(size, alignment);
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ aligned_alloc(alignment, size);
#endif
}

ସ୍ଥିର ggml_backend_buffer_t
ggml_backend_sycl_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                           size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s: size=%zu\n", __func__, size);
    ggml_check_sycl();

    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_sycl_set_device(buft_ctx->device);
    ଅଚଳ queue_ptr stream = buft_ctx->stream;
    size = std::max(size, (size_t)1); // syclMalloc returns null for size 0
    /*
    Alignment below ensures best performance. While in theory it could lead to
    wasting memory, this is acceptable because in practice only few buffers are
    allocated and even less exceed the minimum size accepted here for USM system
    allocations.
     */
    size_t alignment = MEM_SIZE_2M;
    size_t aligned_size = ((size + alignment - 1) / alignment) * alignment;
    ତାର୍କିକ use_usm_system = check_usm_system(buft_ctx->device, aligned_size);

    ଶୂନ୍ୟ * dev_ptr;
    ଯଦି (use_usm_system) {
        GGML_SYCL_DEBUG("[SYCL] allocating %zu Bytes with USM system\n", size);
        dev_ptr = (ଶୂନ୍ୟ *)aligned_malloc_host(alignment, aligned_size);
        ଯଦି (!dev_ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ଫେରନ୍ତୁ nullptr;
        }
    } ଅନ୍ୟଥା {
        SYCL_CHECK(CHECK_TRY_ERROR(dev_ptr = (ଶୂନ୍ୟ *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        ଯଦି (!dev_ptr) {
          GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
          ଫେରନ୍ତୁ nullptr;
        }
    }
    ggml_backend_sycl_buffer_context * ctx = ନୂତନ  ggml_backend_sycl_buffer_context(buft_ctx->device, dev_ptr, buft_ctx->stream, use_usm_system);
    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_sycl_buffer_interface, ctx, size);
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର size_t ggml_backend_sycl_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

ସ୍ଥିର ତାର୍କିକ is_bmg_g31_arch(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ଫେରନ୍ତୁ ggml_sycl_info().devices[device].hw_info.arch == gpu_arch::intel_gpu_bmg_g31;
}

ସ୍ଥିର size_t ggml_backend_sycl_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_sycl_buffer_type_context * ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = ctx->device;
    // use the buft device, the current device may differ on multi-GPU systems
    size_t max_alloc_size = dpct::get_device(device).get_max_mem_alloc_size();
    ଯଦି (g_ggml_sycl_host_pinned_mem_2g) {
        ଫେରନ୍ତୁ std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
    } ଅନ୍ୟଥା {
        ଯଦି(is_bmg_g31_arch(device)) {
            //Todo, it's workaround for BMG-G31, which has a known issue with large allocations.
            //The max alloc size is reduced to 60% of the reported max alloc size.
            //remove it after https://github.com/intel/compute-runtime/issues/998 is fixed.
            max_alloc_size = max_alloc_size*0.6;
        }
        ଫେରନ୍ତୁ max_alloc_size;

    }
    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_sycl_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    // Reserve the additional scratch so it's visible to the graph allocator
    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_sycl_flash_attn_ext_get_alloc_size(tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    ଯଦି (ggml_is_quantized(tensor->type)) {
        ଯଦି (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    ଫେରନ୍ତୁ size;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ଅଚଳ ggml_backend_buffer_type_i ggml_backend_sycl_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ଖାଲି,
    /* .get_alignment       = */ ggml_backend_sycl_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_sycl_buffer_type_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_sycl_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ଖାଲି,
};

ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);


    auto dev_ଗଣନା = ggml_backend_sycl_get_device_ଗଣନା();

    ଯଦି (device>=dev_ଗଣନା or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, dev_ଗଣନା-1);
        GGML_ASSERT(device<dev_ଗଣନା);
    }
    ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_buffer_type_initialized = ମିଥ୍ୟା;

    ଯଦି (!ggml_backend_sycl_buffer_type_initialized) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < dev_ଗଣନା; i++) {
            auto & device_i = dpct::dev_mgr::instance().get_device(i);
            queue_ptr stream = &(device_i.default_queue());
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ ନୂତନ ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), stream},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ &ggml_backend_sycl_buffer_types[device];
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_sycl_buffer_type(ggml_backend_sycl_context * ctx) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_buffer_type\n");

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = ctx->device;
    ଯଦି (device>=ggml_sycl_info().device_ଗଣନା or device<0) {
        GGML_LOG_ERROR("ggml_backend_sycl_buffer_type error: device_index:%d is out of range [0, %d], miss to call ggml_backend_sycl_set_single_device()\n",
            device, ggml_sycl_info().device_ଗଣନା-1);
        GGML_ASSERT(device<ggml_sycl_info().device_ଗଣନା);
    }
    ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_type ggml_backend_sycl_buffer_types[GGML_SYCL_MAX_DEVICES];

    ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_buffer_type_initialized = ମିଥ୍ୟା;

    ଯଦି (!ggml_backend_sycl_buffer_type_initialized) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; i++) {
            ggml_backend_sycl_buffer_types[i] = {
                /* .iface    = */ ggml_backend_sycl_buffer_type_interface,
                /* .device   = */ nullptr,
                /* .context  = */ ନୂତନ ggml_backend_sycl_buffer_type_context{i, GGML_SYCL_NAME + std::to_string(i), ctx->stream(i, 0)},
            };
        }
        ggml_backend_sycl_buffer_type_initialized = ସତ୍ୟ;
    }
    ଫେରନ୍ତୁ &ggml_backend_sycl_buffer_types[device];
}

// sycl split buffer

ସ୍ଥିର int64_t get_row_rounding(ggml_type type, ଅଚଳ std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> & tensor_split) {
    int64_t min_compute_capability = INT_MAX;
    int64_t max_compute_capability = INT_MIN;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        ଯଦି (tensor_split[i] < (i + 1 < ggml_sycl_info().device_ଗଣନା ? tensor_split[i + 1] : 1.0f)) {
            ଯଦି (min_compute_capability > ggml_sycl_info().devices[i].cc) {
                min_compute_capability = ggml_sycl_info().devices[i].cc;
            }
            ଯଦି (max_compute_capability < ggml_sycl_info().devices[i].cc) {
                max_compute_capability = ggml_sycl_info().devices[i].cc;
            }
        }
    }

    ବାଛ(type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
            ଫେରନ୍ତୁ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ 64;
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_F32:
            ଫେରନ୍ତୁ 1;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ2_XS:
        ସ୍ଥିତି GGML_TYPE_IQ2_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_S:
        ସ୍ଥିତି GGML_TYPE_IQ1_M:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ4_XS:
        ସ୍ଥିତି GGML_TYPE_IQ4_NL:
            ଫେରନ୍ତୁ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            ଫେରନ୍ତୁ max_compute_capability >= VER_GEN9 ? 128 : 64;
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            ଫେରନ୍ତୁ 64;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("fatal error");
    }
}

ସ୍ଥିର ଶୂନ୍ୟ get_row_split(int64_t * row_ତଳ, int64_t * row_ଉପର, ଅଚଳ ggml_tensor * tensor, ଅଚଳ std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> & tensor_split, ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) {
    ଅଚଳ int64_t nrows = ggml_nrows(tensor);
    ଅଚଳ int64_t rounding = get_row_rounding(tensor->type, tensor_split);

    *row_ତଳ = id == 0 ? 0 : nrows*tensor_split[id];
    *row_ତଳ -= *row_ତଳ % rounding;
    ଯଦି (id == ggml_sycl_info().device_ଗଣନା - 1) {
        *row_ଉପର = nrows;
    } ଅନ୍ୟଥା {
        *row_ଉପର = nrows*tensor_split[id + 1];
        *row_ଉପର -= *row_ଉପର % rounding;
    }
}

ସ୍ଥିର size_t ggml_nbytes_split(ଅଚଳ ସଂରଚନା ggml_tensor * tensor, ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_split) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    ଫେରନ୍ତୁ nrows_split*ggml_row_size(tensor->type, tensor->ne[0]);
}

ସଂରଚନା ggml_backend_sycl_split_buffer_type_context {
    std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> tensor_split;
};

ସଂରଚନା ggml_backend_sycl_split_buffer_context {
    ~ggml_backend_sycl_split_buffer_context() ଚେଷ୍ଟା {
        ପାଇଁ (ggml_tensor_extra_gpu * extra : tensor_extras) {
            release_extra_gpu(extra, streams);
        }
    }
    ଧର (sycl::exception ଅଚଳ &exc) {
      std::cerr << exc.what() << "Exception caught at file:" << __FILE__
                << ", line:" << __LINE__ << std::endl;
      std::exit(1);
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
    std::vector<queue_ptr> streams;
};

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    delete ctx;
}

ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_sycl_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)0x1000;

    GGML_UNUSED(buffer);
}

ସ୍ଥିର ଗଣନା ggml_status
ggml_backend_sycl_split_buffer_init_tensor(ggml_backend_buffer_t buffer,
                                           ggml_tensor *tensor) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor, "\n").c_str());
    GGML_ASSERT(tensor->view_src == nullptr); // views of split tensors are not supported

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    ଅଚଳ int64_t ne0 = tensor->ne[0];

    ggml_tensor_extra_gpu * extra = ନୂତନ ggml_tensor_extra_gpu{};

    ctx->tensor_extras.push_back(extra);
    // consumers index the queues by device id, so size the vector once with one queue per device
    ଯଦି (ctx->streams.empty()) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
            ctx->streams.push_back(&(dpct::get_device(i).default_queue()));
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        int64_t row_ତଳ, row_ଉପର;
        get_row_split(&row_ତଳ, &row_ଉପର, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ଉପର - row_ତଳ;
        ଯଦି (nrows_split == 0) {
            ଜାରି;
        }

        size_t size = ggml_nbytes_split(tensor, nrows_split);
        ଅଚଳ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ଯଦି (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ggml_sycl_set_device(i);
        ଅଚଳ queue_ptr stream = ctx->streams[i];
        ଅକ୍ଷର * buf;
        SYCL_CHECK(CHECK_TRY_ERROR(buf = (ଅକ୍ଷର *)ggml_sycl_malloc_device(size, *stream, GGML_SYCL_MEM_BUFFER)));
        ଯଦି (!buf) {
            ଅକ୍ଷର err_buf[1024];
            snprintf(err_buf, 1023, "%s: can't allocate %zu Bytes of memory on device\n", __func__, size);
            ଫିଙ୍ଗ std::runtime_error(err_buf);
        }
        // set padding to 0 to avoid possible NaN values
        ଯଦି (size > original_size) {
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

        ପାଇଁ (int64_t is = 0; is < GGML_SYCL_MAX_STREAMS; ++is) {
            /*
            DPCT1009:210: SYCL uses exceptions to report errors and does not use
            the error codes. The original code was commented out and a warning
            string was inserted. You need to rewrite this code.
            */
            SYCL_CHECK(
                CHECK_TRY_ERROR(extra->events[i][is] = ନୂତନ sycl::event()));
        }
    }
    tensor->extra = extra;
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ
ggml_backend_sycl_split_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                          ggml_tensor *tensor, ଅଚଳ ଶୂନ୍ୟ *data,
                                          size_t offset, size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    ଅଚଳ int64_t ne0 = tensor->ne[0];
    ଅଚଳ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        int64_t row_ତଳ, row_ଉପର;
        get_row_split(&row_ତଳ, &row_ଉପର, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ଉପର - row_ତଳ;
        ଯଦି (nrows_split == 0) {
            ଜାରି;
        }

        ଅଚଳ size_t offset_split = row_ତଳ*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        ଅଚଳ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ଯଦି (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ଅଚଳ ଅକ୍ଷର * buf_host = (ଅଚଳ ଅକ୍ଷର *)data + offset_split;
        /*
        DPCT1009:211: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        ଅଚଳ queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(extra->data_device[i], buf_host, original_size)
                .wait()));
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ
ggml_backend_sycl_split_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                          ଅଚଳ ggml_tensor *tensor, ଶୂନ୍ୟ *data,
                                          size_t offset, size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    ggml_backend_sycl_split_buffer_context * ctx = (ggml_backend_sycl_split_buffer_context *)buffer->context;
    ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *)buffer->buft->context;

    ଅଚଳ int64_t ne0 = tensor->ne[0];
    ଅଚଳ size_t nb1 = tensor->nb[1];
    ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *)tensor->extra;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        int64_t row_ତଳ, row_ଉପର;
        get_row_split(&row_ତଳ, &row_ଉପର, tensor, buft_ctx->tensor_split, i);

        int64_t nrows_split = row_ଉପର - row_ତଳ;
        ଯଦି (nrows_split == 0) {
            ଜାରି;
        }

        ଅଚଳ size_t offset_split = row_ତଳ*nb1;
        size_t size = ggml_nbytes_split(tensor, nrows_split);
        ଅଚଳ size_t original_size = size;

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ଯଦି (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }

        ଅକ୍ଷର * buf_host = (ଅକ୍ଷର *)data + offset_split;
        /*
        DPCT1009:212: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ggml_sycl_set_device(i);
        ଅଚଳ queue_ptr stream = ctx->streams[i];
        SYCL_CHECK(CHECK_TRY_ERROR(
            (*stream)
                .memcpy(buf_host, extra->data_device[i], original_size)
                .wait()));
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

ସ୍ଥିର ସଂରଚନା ggml_backend_buffer_i ggml_backend_sycl_split_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_sycl_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_sycl_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_sycl_split_buffer_init_tensor,
    /* .memset_tensor   = */ ଖାଲି,
    /* .set_tensor      = */ ggml_backend_sycl_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_sycl_split_buffer_get_tensor,
    /* .set_tensor_2d   = */ ଖାଲି,
    /* .get_tensor_2d   = */ ଖାଲି,
    /* .cpy_tensor      = */ ଖାଲି,
    /* .clear           = */ ggml_backend_sycl_split_buffer_clear,
    /* .reset           = */ ଖାଲି,
};

// sycl split buffer type

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_split_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ GGML_SYCL_NAME "_Split";

    GGML_UNUSED(buft);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_buffer_is_sycl_split(ggml_backend_buffer_t buffer) {
   ଫେରନ୍ତୁ buffer->buft->iface.get_name == ggml_backend_sycl_split_buffer_type_get_name;
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_sycl_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_sycl_split_buffer_context * ctx = ନୂତନ ggml_backend_sycl_split_buffer_context();

    ଫେରନ୍ତୁ ggml_backend_buffer_init(buft, ggml_backend_sycl_split_buffer_interface, ctx, size);
}

ସ୍ଥିର size_t ggml_backend_sycl_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ SYCL_BUFFER_ALIGNMENT;
    GGML_UNUSED(buft);
}

ସ୍ଥିର size_t ggml_backend_sycl_split_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, ଅଚଳ ggml_tensor * tensor) {
    ggml_backend_sycl_split_buffer_type_context * ctx = (ggml_backend_sycl_split_buffer_type_context *)buft->context;

    size_t total_size = 0;

    ଅଚଳ int64_t ne0 = tensor->ne[0];

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        int64_t row_ତଳ, row_ଉପର;
        get_row_split(&row_ତଳ, &row_ଉପର, tensor, ctx->tensor_split, i);

        int64_t nrows_split = row_ଉପର - row_ତଳ;
        ଯଦି (nrows_split == 0) {
            ଜାରି;
        }

        total_size += ggml_nbytes_split(tensor, nrows_split);

        // pad last row to a multiple of 512 elements to avoid out-of-bounds memory accesses
        ଯଦି (ne0 % MATRIX_ROW_PADDING != 0) {
            total_size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    ଫେରନ୍ତୁ total_size;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ ମିଥ୍ୟା;

    GGML_UNUSED(buft);
}

ସ୍ଥିର ggml_backend_buffer_type_i ggml_backend_sycl_split_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_sycl_split_buffer_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_sycl_split_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ ଖାଲି,
    /* .get_alignment       = */ ggml_backend_sycl_split_buffer_type_get_alignment,
    /* .get_max_size        = */ ଖାଲି, // defaults to SIZE_MAX
    /* .get_alloc_size      = */ ggml_backend_sycl_split_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ ଖାଲି,
    /* .is_host             = */ ggml_backend_sycl_split_buffer_type_is_host,
};

ggml_backend_buffer_type_t ggml_backend_sycl_split_buffer_type(ପୂର୍ଣ୍ଣସଂଖ୍ୟା main_device, ଅଚଳ ଦଶମିକ * tensor_split) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_split_buffer_type\n");

    GGML_UNUSED(main_device);

    ସ୍ଥିର std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    ggml_check_sycl();
    // FIXME: this is not thread safe
    ସ୍ଥିର std::map<std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES>, ସଂରଚନା ggml_backend_buffer_type> buft_map;

    std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> tensor_split_arr = {};

    ତାର୍କିକ all_zero = tensor_split == nullptr || std::all_of(tensor_split, tensor_split + GGML_SYCL_MAX_DEVICES, [](ଦଶମିକ x) { ଫେରନ୍ତୁ x == 0.0f; });
    ଯଦି (all_zero) {
        tensor_split_arr = ggml_sycl_info().default_tensor_split;
    } ଅନ୍ୟଥା {
        ଦଶମିକ split_sum = 0.0f;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
            tensor_split_arr[i] = split_sum;
            split_sum += tensor_split[i];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
            tensor_split_arr[i] /= split_sum;
        }
    }

    auto it = buft_map.find(tensor_split_arr);
    ଯଦି (it != buft_map.end()) {
        ଫେରନ୍ତୁ &it->second;
    }

    ସଂରଚନା ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_sycl_split_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), 0),
        /* .context = */ ନୂତନ ggml_backend_sycl_split_buffer_type_context{tensor_split_arr},
    };

    auto result = buft_map.emplace(tensor_split_arr, buft);
    ଫେରନ୍ତୁ &result.first->second;
}

// host buffer type

ସଂରଚନା ggml_backend_sycl_device_context {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    std::string name;
    std::string description;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_offload_min_batch_size;
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ GGML_SYCL_NAME "_Host";

    GGML_UNUSED(buft);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_sycl_host_buffer_type_device(ggml_backend_buffer_type_t buft) {
    ଫେରନ୍ତୁ static_cast<ଅଚଳ ggml_backend_sycl_device_context *>(buft->device->context)->device;
}

//host pinned memory
ସ୍ଥିର ଶୂନ୍ୟ * ggml_backend_sycl_host_malloc(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t size) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_malloc of size %.2f MiB on device %d\n", size / 1024.0 / 1024.0, device);
    ଶୂନ୍ୟ * ptr = nullptr;
    ଚେଷ୍ଟା {
        ggml_check_sycl();
        // USM host memory is page-locked and device-accessible by construction
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        ptr = sycl::malloc_host(size, q, sycl::property_list{});
    } ଧର (...) {
        ptr = nullptr;
    }
    ଯଦି (ptr == nullptr) {
        GGML_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory\n", __func__,
                size / 1024.0 / 1024.0);
    }

    ଫେରନ୍ତୁ ptr;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ଯଦି (buffer->context == nullptr) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (g_ggml_sycl_enable_host_pinned_mem) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = ggml_backend_sycl_host_buffer_type_device(buffer->buft);
        auto & q = dpct::dev_mgr::instance().get_device(device).default_queue();
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(buffer->context, q)));
    } ଅନ୍ୟଥା {
        free_aligned_mem_host((ଶୂନ୍ୟ *) buffer->context);
    }
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_sycl_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ଶୂନ୍ୟ * ptr = g_ggml_sycl_enable_host_pinned_mem ?
                     ggml_backend_sycl_host_malloc(ggml_backend_sycl_host_buffer_type_device(buft), size) :
                     aligned_malloc_host(TENSOR_ALIGNMENT, size);
    ଯଦି (ptr == nullptr) {
        // fallback to cpu buffer
        ଫେରନ୍ତୁ ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    // FIXME: this is a hack to avoid having to implement a new buffer type
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_sycl_host_buffer_free_buffer;

    ଫେରନ୍ତୁ buffer;
}

ସ୍ଥିର size_t ggml_backend_sycl_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {

    ଯଦି (g_ggml_sycl_enable_host_pinned_mem) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = ggml_backend_sycl_host_buffer_type_device(buft);
        size_t max_alloc_size = dpct::dev_mgr::instance().get_device(device).get_max_mem_alloc_size();
        ଯଦି (g_ggml_sycl_host_pinned_mem_2g) {
            ଫେରନ୍ତୁ std::min(max_alloc_size, (size_t) 2LL*1024*1024*1024);
        } ଅନ୍ୟଥା {
            ଫେରନ୍ତୁ max_alloc_size;
        }
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ SIZE_MAX;
    }
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type_for_device(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_host_buffer_type_for_device on device %d\n", device);

    // the vector is never resized after this, so the returned pointers stay valid
    ସ୍ଥିର std::vector<ggml_backend_buffer_type> buffer_types_host = [] {
        std::vector<ggml_backend_buffer_type> bufts(ggml_backend_sycl_get_device_ଗଣନା());
        ପାଇଁ (size_t i = 0; i < bufts.size(); i++) {
            bufts[i] = {
                /* .iface    = */ {
                    /* .get_name            = */ ggml_backend_sycl_host_buffer_type_name,
                    /* .alloc_buffer        = */ ggml_backend_sycl_host_buffer_type_alloc_buffer,
                    /* .alloc_buffer_n      = */ ଖାଲି,
                    /* .get_alignment       = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
                    /* .get_max_size        = */ ggml_backend_sycl_host_buffer_type_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
                    /* .get_alloc_size_n    = */ ଖାଲି,
                    /* .is_host             = */ ggml_backend_cpu_buffer_type()->iface.is_host,
                },
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), i),
                /* .context  = */ nullptr,
            };
        }
        ଫେରନ୍ତୁ bufts;
    }();

    GGML_ASSERT(device >= 0 && device < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) buffer_types_host.size());

    ଫେରନ୍ତୁ &buffer_types_host[device];
}

// TODO: this function is unused and is a temporary hack to avoid breaking changes
ggml_backend_buffer_type_t ggml_backend_sycl_host_buffer_type() {
    ଫେରନ୍ତୁ ggml_backend_sycl_host_buffer_type_for_device(0);
}

// buffer pool for sycl (legacy)
ସଂରଚନା ggml_sycl_pool_leg : ସର୍ବସାଧାରଣ ggml_sycl_pool {
    ସ୍ଥିର ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା MAX_SYCL_BUFFERS = 256;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device;
    queue_ptr qptr;
    ସଂରଚନା ggml_sycl_buffer {
        ଶୂନ୍ୟ * ptr = nullptr;
        size_t size = 0;
    };

    ggml_sycl_buffer buffer_pool[MAX_SYCL_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_sycl_pool_leg(queue_ptr qptr_, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_) : device(device_), qptr(qptr_) {}

    ~ggml_sycl_pool_leg() {
#ifdef DEBUG_SYCL_POOL
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା    n_cached    = 0;
        size_t bytes_cached = 0;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ଯଦି (buffer_pool[i].ptr != nullptr) {
                ++n_cached;
                bytes_cached += buffer_pool[i].size;
            }
        }
        GGML_LOG_INFO("%s: %d buffers, cached = %.2f MiB\n", __func__,
                      n_cached, bytes_cached / 1024.0 / 1024.0);
        ଅଚଳ auto slots = format_slots_in_alloc_order();
        ଯଦି (!slots.empty()) {
            GGML_LOG_INFO("%s: slots MiB: %s\n", __func__, slots.c_str());
        }
#endif

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            ଯଦି (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(b.ptr, *qptr)));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

#ifdef DEBUG_SYCL_POOL
    std::string format_slots_in_alloc_order() ଅଚଳ {
        std::string ଧାଡ଼ି;
        ଅକ୍ଷର buf[32];
        ତାର୍କିକ first = ସତ୍ୟ;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ଯଦି (buffer_pool[i].ptr == nullptr) {
                ଜାରି;
            }
            ଯଦି (!first) {
                ଧାଡ଼ି += '/';
            }
            first = ମିଥ୍ୟା;
            snprintf(buf, ଆକାର(buf), "%.2f", buffer_pool[i].size / 1024.0 / 1024.0);
            ଧାଡ଼ି += buf;
        }
        ଫେରନ୍ତୁ ଧାଡ଼ି;
    }
#endif

    ଶୂନ୍ୟ * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_sycl_MALLOC
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା ibest = -1;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            ଯଦି (b.ptr != nullptr) {
#ifdef DEBUG_sycl_MALLOC
                ++nnz;
                ଯଦି (b.size > max_size) max_size = b.size;
#endif
                ଯଦି (b.size >= size) {
                    size_t diff = b.size - size;
                    ଯଦି (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        ଯଦି (!best_diff) {
                            ଶୂନ୍ୟ * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            ଫେରନ୍ତୁ ptr;
                        }
                    }
                }
            }
        }
        ଯଦି (ibest >= 0) {
            ggml_sycl_buffer& b = buffer_pool[ibest];
            ଶୂନ୍ୟ * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            ଫେରନ୍ତୁ ptr;
        }
        ଶୂନ୍ୟ * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);

        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ଶୂନ୍ୟ *)ggml_sycl_malloc_device(look_ahead_size, *qptr, GGML_SYCL_MEM_POOL_LEG)));
        ଯଦି (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on device/GPU\n", __func__, look_ahead_size);
            ଫେରନ୍ତୁ nullptr;
        }

        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_DEBUG("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, id, nnz,
                (uint32_t)(max_size/1024/1024), (uint32_t)(g_sycl_pool_size[id]/1024/1024), (uint32_t)(size/1024/1024));
#endif

        // GGML_SYCL_DEBUG("ggml_sycl_pool_malloc_leg look_ahead_size=%zu, return %p\n", look_ahead_size, ptr);
        ଫେରନ୍ତୁ ptr;
    }

    ଶୂନ୍ୟ ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size) override {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_SYCL_BUFFERS; ++i) {
            ggml_sycl_buffer& b = buffer_pool[i];
            ଯଦି (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                ଫେରନ୍ତୁ;
            }
        }
        GGML_LOG_WARN("WARNING: sycl buffer pool full, increase MAX_sycl_BUFFERS\n");
        SYCL_CHECK(CHECK_TRY_ERROR(ggml_sycl_free_device(ptr, *qptr)));
        pool_size -= size;
    }
};

// pool with virtual memory management
#ଯଦି defined(GGML_SYCL_SUPPORT_VMM)
ସଂରଚନା ggml_sycl_pool_vmm : ସର୍ବସାଧାରଣ ggml_sycl_pool {
    ସ୍ଥିର ଅଚଳ size_t SYCL_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           device;
    sycl::context ctx;
    sycl::device  dev;

    uintptr_t pool_addr = 0;
    size_t    pool_used = 0;
    size_t    pool_size = 0;
    size_t    granularity;

    // physical_mem owns the commits (unlike cuMemMap)
    ସଂରଚନା mapping {
        sycl::ext::oneapi::experimental::physical_mem phys;
        ଶୂନ୍ୟ * map_ptr;
    };
    std::vector<mapping> mappings;

    explicit ggml_sycl_pool_vmm(queue_ptr qptr_, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_) :
        device(device_),
        ctx(qptr_->get_context()),
        dev(qptr_->get_device()),
        granularity(ggml_sycl_info().devices[device_].vmm_granularity) {
    }

    ~ggml_sycl_pool_vmm() {
        ଯଦି (pool_addr == 0) {
            ଫେରନ୍ତୁ;
        }

        // Per spec, unmap must (a) match the exact (ptr, size) of an earlier
        // physical_mem::map() call and (b) precede destruction of the
        // physical_mem objects (their dtors won't unmap).
        ପାଇଁ (auto & m : mappings) {
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::unmap(
                m.map_ptr, m.phys.size(), ctx)));
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ext::oneapi::experimental::free_virtual_mem(
            pool_addr, SYCL_POOL_VMM_MAX_SIZE, ctx)));
    }

    ଶୂନ୍ୟ * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        size = GGML_PAD(size, SYCL_BUFFER_ALIGNMENT);

        size_t avail = pool_size - pool_used;

        ଯଦି (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = GGML_PAD(size - avail, granularity);

            GGML_ASSERT(pool_size + reserve_size <= SYCL_POOL_VMM_MAX_SIZE);

            ଯଦି (ggml_sycl_memtrace_enabled()) {
                GGML_LOG_INFO(GGML_SYCL_MEMTRACE_TAG " pool_vmm[%d] committing %5zu MiB (pool %5zu -> %5zu MiB)\n",
                              device, reserve_size / (1024 * 1024), pool_size / (1024 * 1024),
                              (pool_size + reserve_size) / (1024 * 1024));
                ggml_sycl_memtrace_report("before pool_vmm commit");
            }

            // allocate more physical memory
            std::optional<sycl::ext::oneapi::experimental::physical_mem> phys;
            SYCL_CHECK(CHECK_TRY_ERROR(phys.emplace(dev, ctx, reserve_size)));

            // reserve virtual address space (if not already reserved)
            ଯଦି (pool_addr == 0) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    pool_addr = sycl::ext::oneapi::experimental::reserve_virtual_mem(
                        SYCL_POOL_VMM_MAX_SIZE, ctx)));
            }

            // map at the end of the pool
            ଶୂନ୍ୟ * map_ptr = nullptr;
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

        ଶୂନ୍ୟ * ptr = reinterpret_cast<ଶୂନ୍ୟ *>(pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: allocated %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        ଫେରନ୍ତୁ ptr;
    }

    ଶୂନ୍ୟ ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size) override {
#ifdef DEBUG_SYCL_MALLOC
        GGML_LOG_INFO("sycl pool[%d]: freed %llu bytes at %p\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == reinterpret_cast<ଶୂନ୍ୟ *>(pool_addr + pool_used));
    }
};
#endif // defined(GGML_SYCL_SUPPORT_VMM)

ସଂରଚନା ggml_sycl_pool_host : ସର୍ବସାଧାରଣ ggml_sycl_pool {
    queue_ptr qptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା       device;

    ସଂରଚନା ggml_sycl_buffer {
        ଶୂନ୍ୟ * ptr  = nullptr;
        size_t size = 0;
    };

    // Set arbitrarly to 64
    ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା          MAX_POOL_SIZE{ 64 };
    std::vector<ggml_sycl_buffer> buffer_pool = std::vector<ggml_sycl_buffer>(MAX_POOL_SIZE);
    size_t                        pool_size   = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                           counter     = 0;

    explicit ggml_sycl_pool_host(queue_ptr qptr_, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_) : qptr(qptr_), device(device_) {}

    ~ggml_sycl_pool_host() {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            ଯଦି (b.ptr != nullptr) {
                SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(b.ptr, *qptr)));
                b.ptr = nullptr;
                pool_size -= b.size;
                b.size = 0;
            }
        }
    }

    ଶୂନ୍ୟ * alloc(size_t size, size_t * actual_size) override {
        ଯଦି (counter == MAX_POOL_SIZE) {
            // the pool wrapped around, no slot is reserved for this allocation
            ଶୂନ୍ୟ * ptr = nullptr;
            SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ଶୂନ୍ୟ *) sycl::malloc_host(size, *qptr)));
            ଯଦି (!ptr) {
                GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
                ଫେରନ୍ତୁ nullptr;
            }
            pool_size    += size;
            *actual_size = size;
            counter      = 0;
            ଫେରନ୍ତୁ ptr;
        }
        ggml_sycl_buffer & b = buffer_pool[counter];
        ++counter;

        ଯଦି (b.ptr != nullptr) {
            ଯଦି (b.size >= size) {
                // hand out the cached buffer, free() puts it back into the pool
                ଶୂନ୍ୟ * ptr = b.ptr;
                *actual_size = b.size;
                b.ptr = nullptr;
                b.size = 0;
                ଫେରନ୍ତୁ ptr;
            }
            // the cached buffer is too small, replace it
            SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(b.ptr, *qptr)));
            pool_size -= b.size;
            b.ptr = nullptr;
            b.size = 0;
        }

        ଶୂନ୍ୟ * ptr;
        SYCL_CHECK(CHECK_TRY_ERROR(ptr = (ଶୂନ୍ୟ *) sycl::malloc_host(size, *qptr)));
        ଯଦି (!ptr) {
            GGML_LOG_ERROR("%s: can't allocate %zu Bytes of memory on host\n", __func__, size);
            ଫେରନ୍ତୁ nullptr;
        }
        pool_size    += size;
        *actual_size = size;
        ଫେରନ୍ତୁ ptr;
    }

    ଶୂନ୍ୟ ମୁକ୍ତ(ଶୂନ୍ୟ * ptr, size_t size) override {
        // if the pool is not completed add the pointer to it in place of the first nullptr found.
        // Otherwise free the buffer, all slots are cached.
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < MAX_POOL_SIZE; ++i) {
            ggml_sycl_buffer & b = buffer_pool[i];
            ଯଦି (b.ptr == nullptr) {
                b.ptr  = ptr;
                b.size = size;
                ଫେରନ୍ତୁ;
            }
        }
        SYCL_CHECK(CHECK_TRY_ERROR(sycl::ମୁକ୍ତ(ptr, *qptr)));
        pool_size -= size;
    }
};

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_host(queue_ptr qptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    // return pool for the host to speed up memory management
    ଫେରନ୍ତୁ std::unique_ptr<ggml_sycl_pool>(ନୂତନ ggml_sycl_pool_host(qptr, device));
}

std::unique_ptr<ggml_sycl_pool> ggml_backend_sycl_context::new_pool_for_device(queue_ptr qptr, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
#ଯଦି defined(GGML_SYCL_SUPPORT_VMM)
    ଯଦି (g_ggml_sycl_enable_vmm && ggml_sycl_info().devices[device].vmm) {
        ଫେରନ୍ତୁ std::unique_ptr<ggml_sycl_pool>(ନୂତନ ggml_sycl_pool_vmm(qptr, device));
    }
#endif // defined(GGML_SYCL_SUPPORT_VMM)
    ଫେରନ୍ତୁ std::unique_ptr<ggml_sycl_pool>(ନୂତନ ggml_sycl_pool_leg(qptr, device));
}

/// kernels
ପ୍ରକାରବ୍ୟାଖ୍ୟା ଶୂନ୍ୟ (*ggml_sycl_op_mul_mat_t)(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ggml_tensor *src0, ଅଚଳ ggml_tensor *src1, ggml_tensor *dst,
    ଅଚଳ ଅକ୍ଷର *src0_dd_i, ଅଚଳ ଦଶମିକ *src1_ddf_i, ଅଚଳ ଅକ୍ଷର *src1_ddq_i,
    ଦଶମିକ *dst_dd_i, ଅଚଳ int64_t row_ତଳ, ଅଚଳ int64_t row_ଉପର,
    ଅଚଳ int64_t src1_ncols, ଅଚଳ int64_t src1_padded_row_size,
    ଅଚଳ queue_ptr &stream);



ସ୍ଥିର ଶୂନ୍ୟ mul_mat_p021_f16_f32(
    ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଅଚଳ ଦଶମିକ * __restrict__ y, ଦଶମିକ * __restrict__ dst,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_y,
    ଅଚଳ sycl::nd_item<3> &item_ct1) {

    ଅଚଳ sycl::half *x = (ଅଚଳ sycl::half *)vx;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_x = channel / (nchannels_y / nchannels_x);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_y = ncols_x;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_dst = nrows_x;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst = row_x;

    ଦଶମିକ tmp = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_x = col_x0 + item_ct1.get_local_id(2);

        ଯଦି (col_x >= ncols_x) {
            ଭାଙ୍ଗ;
        }

        // x is transposed and permuted
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        ଅଚଳ ଦଶମିକ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0];

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_y = col_x;


        // y is not transposed but permuted
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    ଅଚଳ ଶୂନ୍ୟ * __restrict__ vx, ଅଚଳ ଦଶମିକ * __restrict__ y, ଦଶମିକ * __restrict__ dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_x,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_stride_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_stride_x,ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_stride_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_x_ଭାଜକ,
    ଅଚଳ sycl::nd_item<3> &item_ct1) {

    ଅଚଳ sycl::half *x = (ଅଚଳ sycl::half *)vx;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_x = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                      item_ct1.get_local_id(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel = item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                        item_ct1.get_local_id(0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_x = channel / channel_x_ଭାଜକ;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_dst = nrows_x;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_dst   = row_x;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା idst = channel*nrows_dst + row_dst;

    ଦଶମିକ tmp = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_x0 = 0; col_x0 < ncols_x;
         col_x0 += item_ct1.get_local_range(2)) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_x = col_x0 + item_ct1.get_local_id(2);

        ଯଦି (col_x >= ncols_x) {
            ଭାଙ୍ଗ;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_y = col_x;

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା iy = channel * channel_stride_y + row_y;

        ଅଚଳ ଦଶମିକ xi =
            sycl::vec<sycl::half, 1>(x[ix])
                .convert<ଦଶମିକ, sycl::rounding_mode::automatic>()[0];

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp +=
            dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), tmp, mask);
    }

    ଯଦି (item_ct1.get_local_id(2) == 0) {
        dst[idst] = tmp;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ k_sum_rows_f32(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                           ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = item_ct1.get_local_id(2);

    ଦଶମିକ sum = 0.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = col; i < ncols; i += item_ct1.get_local_range(2)) {
        sum += x[row * ncols + i];
    }

    sum = warp_reduce_sum(sum, item_ct1);

    ଯଦି (col == 0) {
        dst[row] = sum;
    }
}


template<typename T>
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_swap(T & a, T & b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <ggml_sort_order order>
__dpct_inline__ ସ୍ଥିର ଶୂନ୍ୟ
k_argsort_f32_i32(ଅଚଳ ଦଶମିକ *x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_pad,
                  ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tasks_per_thread, ଅଚଳ sycl::nd_item<3> &item_ct1,
                  uint8_t *dpct_local) {
    // bitonic sort
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା col_index =  item_ct1.get_local_id(2);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(1);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < tasks_per_thread; i++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col_index * tasks_per_thread + i;
        ଯଦି (col >= ncols_pad) {
            ଫେରନ୍ତୁ;
        }
    }

    ଅଚଳ ଦଶମିକ * x_row = x + row * ncols;
    auto dst_row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)dpct_local;

    // initialize indices
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i=0;i<tasks_per_thread;i++){
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col_index*tasks_per_thread+i;
        dst_row[col] = col;
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 2; k <= ncols_pad; k *= 2) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = k / 2; j > 0; j /= 2) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < tasks_per_thread; i++) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col_index * tasks_per_thread + i;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା ixj = col ^ j;
                ଯଦି (ixj > col) {
                    ଯଦି ((col & k) == 0) {
                        ଯଦି (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols &&
                             (order == GGML_SORT_ORDER_ASC
                                  ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                  : x_row[dst_row[col]] <
                                        x_row[dst_row[ixj]]))) {
                            ggml_sycl_swap(dst_row[col], dst_row[ixj]);
                        }
                    } ଅନ୍ୟଥା {
                        ଯଦି (dst_row[ixj] >= ncols ||
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
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < tasks_per_thread; i++) {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = col_index * tasks_per_thread + i;
        ଯଦି (col < ncols) {
            dst[row * ncols + col] = dst_row[col];
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ diag_mask_inf_f32(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_channel, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_past,
                              ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);

    ଯଦି (col >= ncols) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = row*ncols + col;
    //dst[i] = col > (n_past + row % rows_per_channel) ? -INFINITY : x[i];
    //dst[i] = x[i] - (col > n_past + row % rows_per_channel) * INT_MAX; // equivalent within rounding error but slightly faster on GPU
    dst[i] = x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX;
}

ସ୍ଥିର ଶୂନ୍ୟ scale_f32(ଅଚଳ ଦଶମିକ * x, ଦଶମିକ * dst, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ bias, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
                      ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    dst[i] = scale * x[i] + bias;
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_mul_mat_p021_f16_f32_sycl(ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଦଶମିକ *y,
                                           ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_x,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_x,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_x,
                                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_y,
                                           queue_ptr stream) {

    ଅଚଳ sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    ଅଚଳ sycl::range<3> block_dims(1, 1, WARP_SIZE);
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

ସ୍ଥିର ଶୂନ୍ୟ ggml_mul_mat_vec_nc_f16_f32_sycl(
    ଅଚଳ ଶୂନ୍ୟ *vx, ଅଚଳ ଦଶମିକ *y, ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_x,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row_stride_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_x,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchannels_y, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_stride_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channel_stride_y, queue_ptr stream) {

    ଅଚଳ sycl::range<3> block_nums(nchannels_y, nrows_x, 1);
    ଅଚଳ sycl::range<3> block_dims(1, 1, WARP_SIZE);
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



ସ୍ଥିର ଶୂନ୍ୟ scale_f32_sycl(ଅଚଳ ଦଶମିକ *x, ଦଶମିକ *dst, ଅଚଳ ଦଶମିକ scale, ଅଚଳ ଦଶମିକ bias,
                           ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, queue_ptr stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (k + SYCL_SCALE_BLOCK_SIZE - 1) / SYCL_SCALE_BLOCK_SIZE;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ଅଙ୍କ_blocks) *
                              sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_SCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            scale_f32(x, dst, scale, bias, k, item_ct1);
        });
}


ସ୍ଥିର ଶୂନ୍ୟ sum_rows_f32_sycl(ଅଚଳ ଦଶମିକ *x, ଦଶମିକ *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                              ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, queue_ptr stream) {
    ଅଚଳ sycl::range<3> block_dims(1, 1, WARP_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, nrows, 1);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                 k_sum_rows_f32(x, dst, ncols, item_ct1);
                             });
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା next_power_of_2(ପୂର୍ଣ୍ଣସଂଖ୍ୟା x) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 1;
    ଯେତେବେଳେ (n < x) {
        n *= 2;
    }
    ଫେରନ୍ତୁ n;
}

ସ୍ଥିର ଶୂନ୍ୟ init_argsort_indices_padded(
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା * idx,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows,
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_pad,
        ଅଚଳ sycl::nd_item<1> & item_ct1) {
    ଅଚଳ size_t gid = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    ଅଚଳ size_t total = (size_t) nrows * (size_t) ncols_pad;

    ଯଦି (gid >= total) {
        ଫେରନ୍ତୁ;
    }

    idx[gid] = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (gid % (size_t) ncols_pad);
}

template <ggml_sort_order order>
ସ୍ଥିର ଶୂନ୍ୟ argsort_f32_i32_global_pass(ଅଚଳ ଦଶମିକ *            x,
                                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                    idx,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols_pad,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                j,
                                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                k,
                                        ଅଚଳ sycl::nd_item<1> & item_ct1) {
    ଅଚଳ size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    ଅଚଳ size_t total = (size_t) nrows * (size_t) ncols_pad;

    ଯଦି (gid >= total) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (gid / (size_t) ncols_pad);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (gid % (size_t) ncols_pad);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ixj = col ^ j;

    ଯଦି (ixj <= col || ixj >= ncols_pad) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ size_t base  = (size_t) row * (size_t) ncols_pad;
    ଅଚଳ size_t pos_a = base + (size_t) col;
    ଅଚଳ size_t pos_b = base + (size_t) ixj;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା a = idx[pos_a];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b = idx[pos_b];

    ତାର୍କିକ do_swap = ମିଥ୍ୟା;

    ଯଦି ((col & k) == 0) {
        ଯଦି (a >= ncols ||
            (b < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = ସତ୍ୟ;
        }
    } ଅନ୍ୟଥା {
        ଯଦି (b >= ncols ||
            (a < ncols &&
             (order == GGML_SORT_ORDER_ASC ?
                  x[(size_t) row * (size_t) ncols + (size_t) a] < x[(size_t) row * (size_t) ncols + (size_t) b] :
                  x[(size_t) row * (size_t) ncols + (size_t) a] > x[(size_t) row * (size_t) ncols + (size_t) b]))) {
            do_swap = ସତ୍ୟ;
        }
    }

    ଯଦି (do_swap) {
        idx[pos_a] = b;
        idx[pos_b] = a;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ copy_argsort_indices_unpadded(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *              idx_padded,
                                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                    dst,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                nrows,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols,
                                          ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା                ncols_pad,
                                          ଅଚଳ sycl::nd_item<1> & item_ct1) {
    ଅଚଳ size_t gid   = item_ct1.get_local_range(0) * item_ct1.get_group(0) + item_ct1.get_local_id(0);
    ଅଚଳ size_t total = (size_t) nrows * (size_t) ncols;

    ଯଦି (gid >= total) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (gid / (size_t) ncols);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (gid % (size_t) ncols);

    dst[(size_t) row * (size_t) ncols + (size_t) col] = idx_padded[(size_t) row * (size_t) ncols_pad + (size_t) col];
}

ସ୍ଥିର ଶୂନ୍ୟ argsort_f32_i32_sycl(ଅଚଳ ଦଶମିକ *x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                                 ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, ggml_sort_order order,
                                 queue_ptr stream, ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ggml_sycl_pool & pool) {
    // bitonic sort requires ncols to be power of 2
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_pad = next_power_of_2(ncols);
    ଅଚଳ size_t shared_mem = (size_t) ncols_pad * ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା);
    ଅଚଳ size_t smpbo = ggml_sycl_info().devices[device].smpbo;

    ଯଦି (shared_mem > smpbo) {
        ggml_sycl_pool_alloc<ପୂର୍ଣ୍ଣସଂଖ୍ୟା> idx_padded_alloc(pool, (size_t) nrows * (size_t) ncols_pad);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା *                     idx_padded = idx_padded_alloc.get();

        constexpr size_t block_size     = 256;
        ଅଚଳ size_t     total_padded   = (size_t) nrows * (size_t) ncols_pad;
        ଅଚଳ size_t     nblocks_padded = (total_padded + block_size - 1) / block_size;

        stream->parallel_for(
            sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
            [=](sycl::nd_item<1> item_ct1) { init_argsort_indices_padded(idx_padded, nrows, ncols_pad, item_ct1); });

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 2; k <= ncols_pad; k *= 2) {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = k / 2; j > 0; j /= 2) {
                ଯଦି (order == GGML_SORT_ORDER_ASC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_ASC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                             k, item_ct1);
                        });
                } ଅନ୍ୟଥା ଯଦି (order == GGML_SORT_ORDER_DESC) {
                    stream->parallel_for(
                        sycl::nd_range<1>(sycl::range<1>(nblocks_padded * block_size), sycl::range<1>(block_size)),
                        [=](sycl::nd_item<1> item_ct1) {
                            argsort_f32_i32_global_pass<GGML_SORT_ORDER_DESC>(x, idx_padded, ncols, nrows, ncols_pad, j,
                                                                              k, item_ct1);
                        });
                } ଅନ୍ୟଥା {
                    GGML_ABORT("invalid sort order");
                }
            }
        }

        ଅଚଳ size_t total   = (size_t) nrows * (size_t) ncols;
        ଅଚଳ size_t nblocks = (total + block_size - 1) / block_size;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(nblocks * block_size), sycl::range<1>(block_size)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 copy_argsort_indices_unpadded(idx_padded, dst, nrows, ncols, ncols_pad, item_ct1);
                             });

        ଫେରନ୍ତୁ;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth = 1;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_block_size = ggml_sycl_info().max_work_group_sizes[device];
    ଯେତେବେଳେ (nth < ncols_pad && nth < max_block_size)
        nth *= 2;
    ଯଦି (nth > max_block_size)
        nth = max_block_size;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tasks_per_thread = ncols_pad / nth;

    ଅଚଳ sycl::range<3> block_dims(1, 1, nth);
    ଅଚଳ sycl::range<3> block_nums(1, nrows, 1);

    ଯଦି (order == GGML_SORT_ORDER_ASC) {
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
    } ଅନ୍ୟଥା ଯଦି (order == GGML_SORT_ORDER_DESC) {
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
    } ଅନ୍ୟଥା {
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
ସ୍ଥିର ଶୂନ୍ୟ ଶୀର୍ଷ_k_scan_merge_f32(
    ଅଚଳ ଦଶମିକ *   src_vals,
    ଅଚଳ int32_t * src_map,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       begin,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       end,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       k,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       block_size,
    ଦଶମିକ *         shared_vals,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା *           shared_idx,
    ଦଶମିକ *         out_vals,
    int32_t *       out_idx,
    ଅଚଳ ତାର୍କିକ      swap01,
    ଅଚଳ sycl::nd_item<1> & item_ct1
) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(0);

    // The running top-k lives in SLM (shared local memory) rather than a private array:
    // an array indexed by a runtime position cannot be register-allocated, so a private
    // one lands in scratch, i.e. device memory, and insertion is this kernel's dominant
    // cost.
    //
    // Lane-strided (lv[i * block_size]) rather than lane-blocked (lv[i]) so a given i is
    // contiguous across lanes; a k-strided layout would put every lane of a shift step in
    // the same SLM bank.
    ଦଶମିକ * lv = shared_vals + tid;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା * li = shared_idx + tid;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < k; i++) {
        lv[i * block_size] = -FLT_MAX;
        li[i * block_size] = -1;
    }

    // The k-th best, cached in a register. The reject test is taken for the large
    // majority of elements scanned, and in that case touches no memory.
    ଦଶମିକ kth = -FLT_MAX;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = begin + tid; col < end; col += block_size) {
        ଦଶମିକ val = src_vals[col];

        ଯଦି (val > kth) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = k - 1;
            ଯେତେବେଳେ (pos > 0 && val > lv[(pos - 1) * block_size]) {
                pos--;
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = k - 1; i > pos; i--) {
                lv[i * block_size] = lv[(i - 1) * block_size];
                li[i * block_size] = li[(i - 1) * block_size];
            }
            lv[pos * block_size] = val;
            li[pos * block_size] = src_map ? src_map[col] : col;

            kth = lv[(k - 1) * block_size];
        }
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    ଯଦି (tid != 0) {
        ଫେରନ୍ତୁ;
    }

    // Same treatment for the merge accumulator, past the per-lane region.
    ଦଶମିକ * fv = shared_vals + (size_t) k * block_size;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା * fi = shared_idx + (size_t) k * block_size;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < k; i++) {
        fv[i] = -FLT_MAX;
        fi[i] = -1;
    }

    ଦଶମିକ fkth = -FLT_MAX;

    // Candidates are visited in the same (t, i) order as before, so tie-breaking is
    // unchanged.
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା t = 0; t < block_size; t++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < k; i++) {
            ଦଶମିକ val = shared_vals[i * block_size + t];

            ଯଦି (val <= fkth) {
                // Lane t's list is sorted descending, so once one of its entries loses
                // to the k-th best, every later entry loses too. fkth only rises, so
                // that stays true for the rest of the merge. This turns the merge from
                // block_size*k steps into roughly block_size plus the candidates
                // accepted.
                ଭାଙ୍ଗ;
            }

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା idx = shared_idx[i * block_size + t];

            ପୂର୍ଣ୍ଣସଂଖ୍ୟା pos = k - 1;
            ଯେତେବେଳେ (pos > 0 && val > fv[pos - 1]) {
                pos--;
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = k - 1; j > pos; j--) {
                fv[j] = fv[j - 1];
                fi[j] = fi[j - 1];
            }
            fv[pos] = val;
            fi[pos] = idx;

            fkth = fv[k - 1];
        }
    }

    ଯଦି (out_vals) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < k; i++) {
            out_vals[i] = fv[i];
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < k; i++) {
        out_idx[i] = fi[i];
    }

    ଯଦି (swap01 && k > 1) {
        int32_t temp = out_idx[0];
        out_idx[0] = out_idx[1];
        out_idx[1] = temp;
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ଶୀର୍ଷ_k_f32_sycl(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ଦଶମିକ * src,
    int32_t * dst_indices,
    ଅଚଳ int64_t ncols,
    ଅଚଳ int64_t nrows,
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k,
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
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା split_block = 128;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_splits  = 128;
    constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା min_cols    = 8192;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsplit = 1;
    ଯଦି (ncols >= min_cols) {
        // A partition is then always >= split_block = 128 columns, hence always more than
        // the k <= 32 ceiling, so no pass is ever padded with -FLT_MAX sentinels.
        ଅଚଳ int64_t want = ncols / split_block;
        nsplit = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (want > max_splits ? max_splits : want);
    }

    ଯଦି (nsplit > 1) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nchunk = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((ncols + nsplit - 1) / nsplit);
        ଅଚଳ size_t ncand = (size_t) nrows * nsplit * k;

        ggml_sycl_pool_alloc<ଦଶମିକ>   part_vals(ctx.pool(), ncand);
        ggml_sycl_pool_alloc<int32_t> part_idx(ctx.pool(), ncand);

        ଦଶମିକ *   pv = part_vals.get();
        int32_t * pi = part_idx.get();

        ଅଚଳ sycl::range<1> block_dims(split_block);

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows * nsplit) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା grp = item_ct1.get_group(0);
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = grp / nsplit;
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା part = grp % nsplit;

                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା begin = part * nchunk;
                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା end = begin + nchunk;
                    ଯଦି (end > (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ncols) {
                        end = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ncols;
                    }

                    ଶୀର୍ଷ_k_scan_merge_f32(
                        src + (int64_t) row * ncols, nullptr, begin, end, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        pv + (size_t) grp * k, pi + (size_t) grp * k, ମିଥ୍ୟା, item_ct1);
                });
        });

        main_stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<ଦଶମିକ, 1> shared_vals(sycl::range<1>((split_block + 1) * k), cgh);
            sycl::local_accessor<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 1> shared_idx(sycl::range<1>((split_block + 1) * k), cgh);

            cgh.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(nrows) * block_dims, block_dims),
                [=](sycl::nd_item<1> item_ct1) {
                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(0);
                    ଅଚଳ size_t off = (size_t) row * nsplit * k;

                    ଶୀର୍ଷ_k_scan_merge_f32(
                        pv + off, pi + off, 0, nsplit * k, k, split_block,
                        shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                        shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        nullptr, dst_indices + (int64_t) row * k, ସତ୍ୟ, item_ct1);
                });
        });

        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_size = 128;

    ଅଚଳ sycl::range<1> block_dims(block_size);
    ଅଚଳ sycl::range<1> grid_dims(nrows);

    main_stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ଦଶମିକ, 1> shared_vals(sycl::range<1>((block_size + 1) * k), cgh);
        sycl::local_accessor<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 1> shared_idx(sycl::range<1>((block_size + 1) * k), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(grid_dims * block_dims, block_dims),
            [=](sycl::nd_item<1> item_ct1) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_group(0);

                ଯଦି (row >= nrows) ଫେରନ୍ତୁ;

                ଶୀର୍ଷ_k_scan_merge_f32(
                    src + (int64_t) row * ncols, nullptr, 0, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ncols, k, block_size,
                    shared_vals.get_multi_ptr<sycl::access::decorated::no>().get(),
                    shared_idx.get_multi_ptr<sycl::access::decorated::no>().get(),
                    nullptr, dst_indices + (int64_t) row * k, ସତ୍ୟ, item_ct1);
            });
    });
}

ସ୍ଥିର ଶୂନ୍ୟ argmax_f32_i32_sycl(ଅଚଳ ଦଶମିକ *x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *dst, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols,
                               ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, queue_ptr stream) {
    ଅଚଳ sycl::range<3> block_dims(1, 1, SYCL_ARGMAX_BLOCK_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, nrows, 1);
    ଅଚଳ size_t shared_mem = 256 * ଆକାର(ଦଶମିକ);

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<ଦଶମିକ, 1> shared_data(
            sycl::range<1>(shared_mem/ଆକାର(ଦଶମିକ)), cgh);
        sycl::local_accessor<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, 1> shared_indices(
            sycl::range<1>(shared_mem/ଆକାର(ଦଶମିକ)), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା tid = item_ct1.get_local_id(2);
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା row = item_ct1.get_global_id(1);

                ଦଶମିକ max_val = -INFINITY;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_idx = -1;

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା col = tid; col < ncols; col += 256) {
                    ଦଶମିକ val = x[row * ncols + col];
                    ଯଦି (val > max_val) {
                        max_val = val;
                        max_idx = col;
                    }
                }

                shared_data[tid] = max_val;
                shared_indices[tid] = max_idx;
                item_ct1.barrier(sycl::access::fence_space::local_space);

                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride = 256/2; stride > 0; stride >>= 1) {
                    ଯଦି (tid < stride) {
                        ଦଶମିକ val1 = shared_data[tid];
                        ଦଶମିକ val2 = shared_data[tid + stride];
                        ଯଦି (val2 > val1) {
                            shared_data[tid] = val2;
                            shared_indices[tid] = shared_indices[tid + stride];
                        }
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);
                }


                ଯଦି (tid == 0) {
                    dst[row] = shared_indices[0];
                }
            });
    });
}
ସ୍ଥିର ଶୂନ୍ୟ diag_mask_inf_f32_sycl(ଅଚଳ ଦଶମିକ *x, ଦଶମିକ *dst,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols_x, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows_x,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା rows_per_channel, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_past,
                                   queue_ptr stream) {
    ଅଚଳ sycl::range<3> block_dims(1, SYCL_DIAG_MASK_INF_BLOCK_SIZE, 1);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା block_ଅଙ୍କ_x = (ncols_x + SYCL_DIAG_MASK_INF_BLOCK_SIZE - 1) / SYCL_DIAG_MASK_INF_BLOCK_SIZE;
    ଅଚଳ sycl::range<3> block_nums(1, block_ଅଙ୍କ_x, nrows_x);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             diag_mask_inf_f32(x, dst, ncols_x,
                                               rows_per_channel, n_past,
                                               item_ct1);
                         });
}

ସ୍ଥିର dpct::err0 ggml_sycl_cpy_tensor_2d(ଶୂନ୍ୟ *dst,
                                          ଅଚଳ ସଂରଚନା ggml_tensor *src,
                                          int64_t i3, int64_t i2,
                                          int64_t i1_ତଳ, int64_t i1_ଉପର,
                                          queue_ptr stream) ଚେଷ୍ଟା {

    dpct::memcpy_direction kind;
    ଅକ୍ଷର * src_ptr;
    ଯଦି (ggml_backend_buffer_is_host(src->buffer)) {
        kind = dpct::host_to_device;
        //GGML_SYCL_DEBUG("%s: Host buffer type src tensor\n", __func__);
        src_ptr = (ଅକ୍ଷର *) src->data;
        // GGML_SYCL_DEBUG("ggml_sycl_cpy_tensor_2d  GGML_BACKEND_TYPE_CPU src_ptr %p\n", src_ptr);
    } ଅନ୍ୟଥା ଯଦି (ggml_backend_buffer_is_sycl(src->buffer)) {
        // If buffer is a SYCL buffer
        //GGML_SYCL_DEBUG("%s: SYCL buffer type src tensor\n", __func__);
        kind    = dpct::device_to_device;
        src_ptr = (ଅକ୍ଷର *) src->data;
    } ଅନ୍ୟଥା ଯଦି (ggml_backend_buffer_is_sycl_split(src->buffer)) {
        /*
        If buffer is a SYCL split buffer
        */
        //GGML_SYCL_DEBUG("%s: Split buffer type src tensor\n", __func__);
        GGML_ASSERT(i1_ତଳ == 0 && i1_ଉପର == src->ne[1]);
        kind = dpct::device_to_device;
        ggml_tensor_extra_gpu * extra = (ggml_tensor_extra_gpu *) src->extra;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା id;
        SYCL_CHECK(CHECK_TRY_ERROR(
            id = get_current_device_id()));
        // GGML_SYCL_DEBUG("current device index %d\n", id);
        src_ptr = (ଅକ୍ଷର *) extra->data_device[id];
    } ଅନ୍ୟଥା {
        // GGML_SYCL_DEBUG("GGML_ABORT("fatal error")\n");
        GGML_ABORT("fatal error");
    }
    ଅକ୍ଷର * dst_ptr = (ଅକ୍ଷର *) dst;

    GGML_TENSOR_LOCALS_1(int64_t, ne, src, ne);
    GGML_TENSOR_LOCALS(int64_t, nb, src, nb);
    ଅଚଳ ଗଣନା ggml_type type = src->type;
    ଅଚଳ int64_t ts = ggml_type_size(type);
    ଅଚଳ int64_t bs = ggml_blck_size(type);
    int64_t i1_diff = i1_ଉପର - i1_ତଳ;

    ଅଚଳ ଅକ୍ଷର * x = src_ptr + i1_ତଳ*nb1 + i2*nb2 + i3*nb3;
    ଯଦି (nb0 == ts && nb1 == ts*ne0/bs) {
        // GGML_SYCL_DEBUG("stream->memcpy: dst_ptr=%p, x=%p, size=%lu\n", dst_ptr, x, i1_diff * nb1);
        // return CHECK_TRY_ERROR(stream->memcpy(dst_ptr, x, i1_diff * nb1));
        ଫେରନ୍ତୁ CHECK_TRY_ERROR(dpct::async_dpct_memcpy(dst_ptr, x, i1_diff * nb1,
                                    kind, *stream));

    } ଅନ୍ୟଥା ଯଦି (nb0 == ts) {
        ଫେରନ୍ତୁ CHECK_TRY_ERROR(
            dpct::async_dpct_memcpy(dst_ptr, ts * ne0 / bs, x, nb1,
                                    ts * ne0 / bs, i1_diff, kind, *stream));
    } ଅନ୍ୟଥା {
        ପାଇଁ (int64_t i1 = 0; i1 < i1_diff; i1++) {
            ଅଚଳ ଶୂନ୍ୟ * rx = (ଅଚଳ ଶୂନ୍ୟ *) ((ଅଚଳ ଅକ୍ଷର *) x + i1*nb1);
            ଶୂନ୍ୟ * rd = (ଶୂନ୍ୟ *) (dst_ptr + i1*ts*ne0/bs);
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
            ଯଦି (r != 0) ଫେରନ୍ତୁ r;
        }
        ଫେରନ୍ତୁ 0;
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_mul_mat_sycl(
    ggml_backend_sycl_context & ctx,
    ଅଚଳ ggml_tensor *src0, ଅଚଳ ggml_tensor *src1, ggml_tensor *dst,
    ଅଚଳ ଅକ୍ଷର *src0_dd_i, ଅଚଳ ଦଶମିକ *src1_ddf_i, ଅଚଳ ଅକ୍ଷର *src1_ddq_i,
    ଦଶମିକ *dst_dd_i, ଅଚଳ int64_t row_ତଳ, ଅଚଳ int64_t row_ଉପର,
    ଅଚଳ int64_t src1_ncols, ଅଚଳ int64_t src1_padded_row_size,
    ଅଚଳ queue_ptr &stream) ଚେଷ୍ଟା {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne00 == ne10);

    ଅଚଳ int64_t row_diff = row_ଉପର - row_ତଳ;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା id;
    SYCL_CHECK(
        CHECK_TRY_ERROR(id = get_current_device_id()));

    ଅଚଳ int64_t ne0 = dst->ne[0]; // used by MKL only
    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc = id == ctx.device ? ne0 : row_diff; // used by MKL only

#ifdef GGML_SYCL_F16
    ତାର୍କିକ use_fp16 = ସତ୍ୟ;  // TODO(Yu) SYCL capability check
#ଅନ୍ୟଥା
    ତାର୍କିକ use_fp16 = ମିଥ୍ୟା;
#endif

#ଯଦି GGML_SYCL_DNNL && defined(GGML_SYCL_HAS_BF16)
    // Fast path for bf16 src0
    ଯଦି (src0->type == GGML_TYPE_BF16 && g_ggml_sycl_enable_dnn &&
        ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device()) && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1]) {
        using bf16_t = sycl::ext::oneapi::bfloat16;
        ggml_sycl_pool_alloc<bf16_t> src1_as_bf16(ctx.pool(), src1_ncols*ne10);
        ଯଦି (src1->type != GGML_TYPE_BF16) {
            ଅଚଳ to_bf16_sycl_t to_bf16_sycl = ggml_get_to_bf16_sycl(src1->type, dst);
            GGML_ASSERT(to_bf16_sycl != nullptr);
            to_bf16_sycl(src1_ddf_i, src1_as_bf16.get(), src1_ncols*ne10, stream);
        } ଅନ୍ୟଥା {
            stream->memcpy(src1_as_bf16.get(), src1_ddf_i, src1_ncols*ne10*ଆକାର(bf16_t));
        }
        DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10,
                                  src0_dd_i, DnnlGemmWrapper::to_dt<bf16_t>(),
                                  src1_as_bf16.get(), DnnlGemmWrapper::to_dt<bf16_t>(),
                                  dst_dd_i, DnnlGemmWrapper::to_dt<ଦଶମିକ>(), stream);
        GGML_UNUSED(dst);
        GGML_UNUSED(src1_ddq_i);
        GGML_UNUSED(src1_padded_row_size);
        ଫେରନ୍ତୁ;
    }
#endif

    // dequantize inside the GEMM instead of writing the f16 weights out and reading them back; src1
    // goes in its own type, so there is no separate conversion pass. The A stage
    // decodes the weight bytes in place, so an installed persistent reorder must be
    // visible to it: pass the flag through; types without an SoA A-stage decline
    // inside and this falls through to the reorder-aware conversion route below.
    ତାର୍କିକ src0_reordered = ମିଥ୍ୟା;
    ଯଦି (ଅଚଳ auto * src0_extra = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(src0->extra)) {
        src0_reordered = src0_extra->optimized_feature.reorder;
    }
    ଯଦି (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] &&
        ggml_sycl_fused_dequant_gemm(src0->type, src0_reordered, src0_dd_i, src1_ddf_i, src1->type, ggml_sycl_src1_prec(dst), dst_dd_i,
                                     row_diff, src1_ncols, ne10, ldc, ctx.pool(), stream)) {
        ଫେରନ୍ତୁ;
    }

    // the f16 route converts src1 to f16 [TAG_GGML_PREC]
    use_fp16 = use_fp16 && ggml_sycl_src1_f16_ok(dst);
    ଯଦି ((src0->type == GGML_TYPE_F16 || ggml_is_quantized(src0->type)) && use_fp16 && ggml_is_contiguous(src0) &&
        row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        ggml_sycl_pool_alloc<sycl::half> src1_as_f16(ctx.pool());
        ଯଦି (src1->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp16");
            ଅଚଳ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_sycl(src1_ddf_i, src1_as_f16.get(), ne, stream);
        }
        ଅଚଳ sycl::half *src1_ptr = src1->type == GGML_TYPE_F16
                ? (ଅଚଳ sycl::half *)src1_ddf_i
                : src1_as_f16.get();

        ggml_sycl_pool_alloc<sycl::half> src0_as_f16(ctx.pool());
        ଯଦି (src0->type != GGML_TYPE_F16) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp16_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp16");
            ଅଚଳ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src0->type, dst);
            GGML_ASSERT(to_fp16_sycl != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_sycl(src0_dd_i, src0_as_f16.get(), ne, stream);
        }
        ଅଚଳ sycl::half *src0_ptr = src0->type == GGML_TYPE_F16
                                         ? (ଅଚଳ sycl::half *)src0_dd_i
                                         : src0_as_f16.get();

#ଯଦି GGML_SYCL_DNNL
        ଯଦି (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx,row_diff, src1_ncols , ne10, src0_ptr,
                                     DnnlGemmWrapper::to_dt<sycl::half>(), src1_ptr, DnnlGemmWrapper::to_dt<sycl::half>(),
                                      dst_dd_i, DnnlGemmWrapper::to_dt<ଦଶମିକ>(), stream);
        }
        ଅନ୍ୟଥା
#endif
        {
            ଅଚଳ ଦଶମିକ alpha = 1.0f;
            ଅଚଳ ଦଶମିକ beta  = 0.0f;
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm(
                *stream, oneapi::mkl::transpose::trans,
                oneapi::mkl::transpose::nontrans, row_diff, src1_ncols, ne10,
                &alpha, src0_ptr, dpct::library_data_t::real_half, ne00,
                src1_ptr, dpct::library_data_t::real_half, ne10, &beta,
                dst_dd_i, dpct::library_data_t::real_float, ldc,
                dpct::library_data_t::real_float)));
        }
    } ଅନ୍ୟଥା {
        ggml_sycl_pool_alloc<ଦଶମିକ> src0_ddq_as_f32(ctx.pool());
        ggml_sycl_pool_alloc<ଦଶମିକ> src1_ddq_as_f32(ctx.pool());
        ଯଦି (src0->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src0 to fp32");
            ଅଚଳ to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src0->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_sycl(src0_dd_i, src0_ddq_as_f32.get(), row_diff*ne00, stream);
        }
        ଯଦି (src1->type != GGML_TYPE_F32) {
            scope_op_debug_print scope_dbg_print(__func__, "/to_fp32_sycl", dst, /*num_src=*/2,
                                                 " : converting src1 to fp32");
            ଅଚଳ to_fp32_sycl_t to_fp32_sycl = ggml_get_to_fp32_sycl(src1->type, dst);
            GGML_ASSERT(to_fp32_sycl != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_sycl(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols*ne10, stream);
        }
        ଅଚଳ ଦଶମିକ * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (ଅଚଳ ଦଶମିକ *) src0_dd_i : src0_ddq_as_f32.get();
        ଅଚଳ ଦଶମିକ * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (ଅଚଳ ଦଶମିକ *) src1_ddf_i : src1_ddq_as_f32.get();

        {
#ଯଦି GGML_SYCL_DNNL
            ଅଚଳ int64_t gemm_flops = (int64_t)row_diff * src1_ncols * ne10;
            ଅଚଳ ତାର୍କିକ use_mkl_direct = gemm_flops < 256 * 256 * 256;
            ଯଦି (g_ggml_sycl_enable_dnn && !use_mkl_direct && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
                DnnlGemmWrapper::row_gemm(ctx, row_diff, src1_ncols, ne10, src0_ddf_i,
                                          DnnlGemmWrapper::to_dt<ଦଶମିକ>(), src1_ddf1_i, DnnlGemmWrapper::to_dt<ଦଶମିକ>(),
                                          dst_dd_i, DnnlGemmWrapper::to_dt<ଦଶମିକ>(), stream);
            }
            ଅନ୍ୟଥା
#endif
            {
                ଅଚଳ ଦଶମିକ alpha = 1.0f;
                ଅଚଳ ଦଶମିକ beta  = 0.0f;
                SYCL_CHECK(CHECK_TRY_ERROR(oneapi::mkl::blas::column_major::gemm(
                    *stream, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, row_diff,
                    src1_ncols, ne10, dpct::get_ମୂଲ୍ୟ(&alpha, *stream), src0_ddf_i, ne00, src1_ddf1_i, ne10,
                    dpct::get_ମୂଲ୍ୟ(&beta, *stream), dst_dd_i, ldc)));
            }
        }
    }
    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sum(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int64_t ne = ggml_nelements(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ne, 1, main_stream);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int64_t ncols = dst->src[0]->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    sum_rows_f32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int64_t ncols = dst->src[0]->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(dst->src[0]);
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


ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);


    ଅଚଳ int64_t ncols = dst->src[0]->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(dst->src[0]);

    ଗଣନା ggml_sort_order order = (ଗଣନା ggml_sort_order) dst->op_params[0];

    argsort_f32_i32_sycl(src0_dd, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)dst_dd, ncols, nrows, order,
                         main_stream, ctx.device, ctx.pool());
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_ଶୀର୍ଷ_k(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(src0->data);
    int32_t * dst_dd = static_cast<int32_t *>(dst->data);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = dst->ne[0];
    ଅଚଳ int64_t ncols = src0->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(src0);

    GGML_ASSERT(k > 0);
    GGML_ASSERT(k <= ncols);

    ଯଦି (k <= SYCL_ଶୀର୍ଷ_K_SCAN_MERGE_MAX_K) {
        ଶୀର୍ଷ_k_f32_sycl(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    } ଅନ୍ୟଥା {
        ggml_sycl_ଶୀର୍ଷ_k_radix(ctx, src0_dd, dst_dd, ncols, nrows, k, main_stream);
    }
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    int32_t *       dst_dd  = static_cast<int32_t *>(dst->data);

    ଅଚଳ int64_t ncols = dst->src[0]->ne[0];
    ଅଚଳ int64_t nrows = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ncols * nrows <= INT32_MAX);

    argmax_f32_i32_sycl(src0_dd, dst_dd, ncols, nrows, main_stream);
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ int64_t ne00 = dst->src[0]->ne[0];
    ଅଚଳ int64_t ne01 = dst->src[0]->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows0 = ggml_nrows(dst->src[0]);
    // the kernel indexes in 32-bit
    GGML_ASSERT(ne00 * nrows0 <= INT32_MAX);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_past = ((int32_t *) dst->op_params)[0];

    diag_mask_inf_f32_sycl(src0_dd, dst_dd, ne00, nrows0, ne01, n_past, main_stream);
}

ସ୍ଥିର ଶୂନ୍ୟ tri_f32_sycl(
    ଅଚଳ ଦଶମିକ * src,
    ଦଶମିକ * dst,
    ଅଚଳ int64_t ne0,
    ଅଚଳ int64_t ne1,
    ଅଚଳ int64_t ne2,
    ଅଚଳ int64_t ne3,
    ଅଚଳ ggml_tri_type ttype,
    dpct::queue_ptr main_stream
) {
    ଅଚଳ size_t total = (size_t) ne0 * (size_t) ne1 * (size_t) ne2 * (size_t) ne3;

    main_stream->parallel_for(sycl::range<1>(total), [=](sycl::id<1> tid) {
        ଅଚଳ int64_t idx = (int64_t) tid[0];

        ଅଚଳ int64_t i0 = idx % ne0;
        ଅଚଳ int64_t t1 = idx / ne0;
        ଅଚଳ int64_t i1 = t1 % ne1;

        ତାର୍କିକ keep = ମିଥ୍ୟା;
        ବାଛ (ttype) {
            ସ୍ଥିତି GGML_TRI_TYPE_LOWER:      keep = (i0 <  i1); ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TRI_TYPE_LOWER_DIAG: keep = (i0 <= i1); ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TRI_TYPE_UPPER:      keep = (i0 >  i1); ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_TRI_TYPE_UPPER_DIAG: keep = (i0 >= i1); ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ: keep = ମିଥ୍ୟା; ଭାଙ୍ଗ;
        }

        dst[idx] = keep ? src[idx] : 0.0f;
    });
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_tri(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0);

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(src0->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ ggml_tri_type ttype = (ggml_tri_type) ggml_get_op_params_i32(dst, 0);

    ଅଚଳ int64_t ne0 = src0->ne[0];
    ଅଚଳ int64_t ne1 = src0->ne[1];
    ଅଚଳ int64_t ne2 = src0->ne[2];
    ଅଚଳ int64_t ne3 = src0->ne[3];

    tri_f32_sycl(src0_dd, dst_dd, ne0, ne1, ne2, ne3, ttype, main_stream);
}


ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    ଅଚଳ ଦଶମିକ * src0_dd = static_cast<ଅଚଳ ଦଶମିକ *>(dst->src[0]->data);
    ଦଶମିକ *       dst_dd  = static_cast<ଦଶମିକ *>(dst->data);

    ଦଶମିକ scale;
    ଦଶମିକ bias;
    memcpy(&scale, (ଦଶମିକ *) dst->op_params + 0, ଆକାର(ଦଶମିକ));
    memcpy(&bias,  (ଦଶମିକ *) dst->op_params + 1, ଆକାର(ଦଶମିକ));

    // the kernel indexes in 32-bit
    GGML_ASSERT(ggml_nelements(dst->src[0]) <= INT32_MAX);

    scale_f32_sycl(src0_dd, dst_dd, scale, bias, ggml_nelements(dst->src[0]), main_stream);
    /*
    DPCT1010:87: SYCL uses exceptions to report errors and does not use the
    error codes. The call was replaced with 0. You need to rewrite this code.
    */
    SYCL_CHECK(0);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_set_peer_access(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_tokens, [[maybe_unused]] ପୂର୍ଣ୍ଣସଂଖ୍ୟା main_device) {
    ସ୍ଥିର ତାର୍କିକ peer_access_enabled = ମିଥ୍ୟା;

    ଅଚଳ ତାର୍କିକ enable_peer_access = n_tokens <= GGML_SYCL_PEER_MAX_BATCH_SIZE;

    ଯଦି (peer_access_enabled == enable_peer_access) {
        ଫେରନ୍ତୁ;
    }

#ifdef NDEBUG
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        SYCL_CHECK(ggml_sycl_set_device(i));

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id_other = 0; id_other < ggml_sycl_info().device_ଗଣନା; ++id_other) {
            ଯଦି (i == id_other) {
                ଜାରି;
            }
            ଯଦି (i != main_device && id_other != main_device) {
                ଜାରି;
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

template <template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା> typename quantize_f>
ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_op_mul_mat(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor *src0,
                                 ଅଚଳ ggml_tensor *src1, ggml_tensor *dst,
                                 ggml_sycl_op_mul_mat_t op) ଚେଷ୍ଟା {

    GGML_TENSOR_LOCALS(int64_t, ne0, src0, ne);

    GGML_TENSOR_LOCALS(int64_t, ne1, src1, ne);
    ଅଚଳ int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb2 = dst->nb[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nb3 = dst->nb[3];

    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(dst->buffer));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src1->buffer));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    ଅଚଳ int64_t i02_ଭାଜକ = ne12 / ne02;

    ଅଚଳ size_t src0_ts = ggml_type_size(src0->type);
    ଅଚଳ size_t src0_bs = ggml_blck_size(src0->type);
    ଅଚଳ size_t q8_1_ts = ଆକାର(block_q8_1);
    ଅଚଳ size_t q8_1_bs = QK8_1;

    ggml_tensor_extra_gpu * src0_extra = (ggml_tensor_extra_gpu *) src0->extra;

    ଅଚଳ ତାର୍କିକ src0_is_contiguous = ggml_is_contiguous(src0);
    ଅଚଳ ତାର୍କିକ src1_is_contiguous = ggml_is_contiguous(src1);

    int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    ଅଚଳ ତାର୍କିକ split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    GGML_ASSERT(!(split && ne02 > 1));
    GGML_ASSERT(!(split && ne03 > 1));
    GGML_ASSERT(!(split && ne02 < ne12));

    std::array<ଦଶମିକ, GGML_SYCL_MAX_DEVICES> tensor_split;
    ଯଦି (split) {
        // TODO: check that src0->buffer->buft is a split buffer type, replace GGML_BACKEND_TYPE_GPU_SPLIT check
        // GGML_ASSERT(src0->buffer != nullptr && src0->buffer->buft == ...);
        ggml_backend_sycl_split_buffer_type_context * buft_ctx = (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        tensor_split = buft_ctx->tensor_split;
    }

    ସଂରଚନା dev_data {
        ggml_sycl_pool_alloc<ଅକ୍ଷର> src0_dd_alloc;
        ggml_sycl_pool_alloc<ଦଶମିକ> src1_ddf_alloc;
        ggml_sycl_pool_alloc<ଅକ୍ଷର> src1_ddq_alloc;
        ggml_sycl_pool_alloc<ଦଶମିକ> dst_dd_alloc;

        ଅକ୍ଷର *src0_dd = nullptr;
        ଦଶମିକ *src1_ddf = nullptr; // float
        ଅକ୍ଷର *src1_ddq = nullptr;  // q8_1
        ଦଶମିକ *dst_dd = nullptr;

        int64_t row_ତଳ;
        int64_t row_ଉପର;
    };

    dev_data dev[GGML_SYCL_MAX_DEVICES];

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା used_devices = 0;
    queue_ptr main_stream = ctx.stream();

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        // by default, use all rows
        dev[i].row_ତଳ  = 0;
        dev[i].row_ଉପର = ne01;

        // for multi GPU, get the row boundaries from tensor split
        // and round to mul_mat_q tile sizes
        ଯଦି (split) {
            ଅଚଳ int64_t rounding = get_row_rounding(src0->type, tensor_split);

            ଯଦି (i != 0) {
                dev[i].row_ତଳ  = ne01*tensor_split[i];
                ଯଦି (dev[i].row_ତଳ < ne01) {
                    dev[i].row_ତଳ -= dev[i].row_ତଳ % rounding;
                }
            }

            ଯଦି (i != ggml_sycl_info().device_ଗଣନା - 1) {
                dev[i].row_ଉପର  = ne01*tensor_split[i + 1];
                ଯଦି (dev[i].row_ଉପର < ne01) {
                    dev[i].row_ଉପର -= dev[i].row_ଉପର % rounding;
                }
            }
        }
    }

    constexpr ତାର୍କିକ quantize_enabled = !std::is_same_v<quantize_f<QK8_1 / WARP_SIZE>,
                                                      no_quantize_q8_1<QK8_1 / WARP_SIZE>>;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
        ଯଦି ((!split && i != ctx.device) || dev[i].row_ତଳ == dev[i].row_ଉପର) {
            ଜାରି;
        }

        used_devices++;

        ଅଚଳ ତାର୍କିକ src1_on_device = i == ctx.device;
        ଅଚଳ ତାର୍କିକ  dst_on_device = i == ctx.device;

        ggml_sycl_set_device(i);
        queue_ptr stream = ctx.stream(i, 0);

        ଯଦି (src0_is_contiguous) {
            // a split tensor keeps its rows in the per-device slices of the tensor extra
            dev[i].src0_dd = split ? (ଅକ୍ଷର *) src0_extra->data_device[i] : (ଅକ୍ଷର *) src0->data;
        } ଅନ୍ୟଥା {
            dev[i].src0_dd = dev[i].src0_dd_alloc.alloc(ctx.pool(i), ggml_nbytes(src0));
        }

        ଯଦି (src1_on_device && src1_is_contiguous) {
            dev[i].src1_ddf = (ଦଶମିକ *) src1->data;
        } ଅନ୍ୟଥା {
            dev[i].src1_ddf = dev[i].src1_ddf_alloc.alloc(ctx.pool(i), ggml_nelements(src1));
        }

        ଯଦି constexpr(quantize_enabled) {
            dev[i].src1_ddq = dev[i].src1_ddq_alloc.alloc(ctx.pool(i), nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs);

            ଯଦି (src1_on_device && src1_is_contiguous) {
                scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                     /*num_src=*/2, " : converting src1 to Q8_1");
                ଚେଷ୍ଟା {
                    quantize_row_q8_1_sycl<quantize_f>(dev[i].src1_ddf, dev[i].src1_ddq, ne10, nrows1, src1_padded_col_size, stream);
                } ଧର (sycl::exception ଅଚଳ &exc) {
                    std::cerr << "Quantize_row_q8_1_sycl error" << exc.what() << "Exception caught at file:" << __FILE__
                              << ", line:" << __LINE__ << std::endl;
                    std::exit(1);
                }
            }
        }

        ଯଦି (dst_on_device) {
            dev[i].dst_dd = (ଦଶମିକ *) dst->data;
        } ଅନ୍ୟଥା {
            ଅଚଳ size_t size_dst_ddf = split ? (dev[i].row_ଉପର - dev[i].row_ତଳ)*ne1 : ggml_nelements(dst);
            dev[i].dst_dd = dev[i].dst_dd_alloc.alloc(ctx.pool(i), size_dst_ddf);
        }
    }

    // if multiple devices are used they need to wait for the main device
    // here an event is recorded that signals that the main device has finished calculating the input data
    ଯଦି (split && used_devices > 1) {
        ggml_sycl_set_device(ctx.device);
        SYCL_CHECK(CHECK_TRY_ERROR(
            *src0_extra->events[ctx.device][0] =
                ctx.stream()->ext_oneapi_submit_barrier()));
    }

    ଅଚଳ int64_t src1_col_stride = split && used_devices > 1 ? MUL_MAT_SRC1_COL_STRIDE : ne11;
    ପାଇଁ (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        ଅଚଳ int64_t is = split ? (src1_col_0/src1_col_stride) % GGML_SYCL_MAX_STREAMS : 0;
        ଅଚଳ int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
            ଯଦି ((!split && i != ctx.device) || dev[i].row_ତଳ == dev[i].row_ଉପର) {
                ଜାରି;
            }

            ଅଚଳ ତାର୍କିକ src1_on_device = i == ctx.device;
            ଅଚଳ ତାର୍କିକ  dst_on_device = i == ctx.device;
            ଅଚଳ int64_t row_diff = dev[i].row_ଉପର - dev[i].row_ତଳ;

            ggml_sycl_set_device(i);
            queue_ptr stream = ctx.stream(i, is);

            // wait for main GPU data if necessary
            ଯଦି (split && (i != ctx.device || is != 0)) {
                SYCL_CHECK(CHECK_TRY_ERROR(stream->ext_oneapi_submit_barrier(
                    {*src0_extra->events[ctx.device][0]})));
            }

            ପାଇଁ (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                ଅଚଳ int64_t i03 = i0 / ne12;
                ଅଚଳ int64_t i02 = i0 % ne12;

                ଅଚଳ size_t src1_ddq_i_offset = (i0*ne11 + src1_col_0) * src1_padded_col_size*q8_1_ts/q8_1_bs;

                // for split tensors the data begins at i0 == i0_offset_low
                ଅକ୍ଷର  *  src0_dd_i =  dev[i].src0_dd + (i0/i02_ଭାଜକ) * (ne01*ne00*src0_ts)/src0_bs;
                ଦଶମିକ * src1_ddf_i = (ଦଶମିକ *) ((ଅକ୍ଷର *) dev[i].src1_ddf +
                    (i0*ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type));
                ଅକ୍ଷର  * src1_ddq_i = dev[i].src1_ddq +  src1_ddq_i_offset;
                ଦଶମିକ *   dst_dd_i =   dev[i].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                ଯଦି (i == ctx.device) {
                    dst_dd_i += dev[i].row_ତଳ; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                ଯଦି (src1_is_contiguous) {
                    ଯଦି (i != ctx.device) {
                        ଯଦି constexpr (quantize_enabled) {
                            ଅକ୍ଷର * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            SYCL_CHECK(
                                CHECK_TRY_ERROR(stream
                                                    ->memcpy(src1_ddq_i, src1_ddq_i_source,
                                                             src1_ncols * src1_padded_col_size * q8_1_ts / q8_1_bs)
                                                    .wait()));
                        } ଅନ୍ୟଥା {
                            // src1 is not split, on the main device it lives at dev[ctx.device].src1_ddf
                            ଅଚଳ ଅକ୍ଷର * src1_ddf_i_source = (ଅଚଳ ଅକ୍ଷର *) dev[ctx.device].src1_ddf +
                                (i0 * ne11 + src1_col_0) * ne10 * ggml_type_size(src1->type);

                            SYCL_CHECK(
                                CHECK_TRY_ERROR(dev2dev_memcpy(i, *stream, ctx.device, *main_stream, src1_ddf_i, src1_ddf_i_source,
                                                               src1_ncols * ne10 * ggml_type_size(src1->type))));
                        }
                    }
                } ଅନ୍ୟଥା {
                    ଯଦି (src1_on_device) {
                        SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src1_ddf_i, src1, i03, i02, src1_col_0,
                                                           src1_col_0 + src1_ncols, stream));
                    } ଅନ୍ୟଥା {
                        GGML_ABORT("src1 is non-contiguous and not on device");
                    }

                    ଯଦି constexpr (quantize_enabled) {
                        scope_op_debug_print scope_dbg_print(__func__, "/quantize_row_q8_1_sycl", dst,
                                                             /*num_src=*/2, " : converting src1 to Q8_1");
                        ଚେଷ୍ଟା {
                            quantize_row_q8_1_sycl<quantize_q8_1>(src1_ddf_i, src1_ddq_i, ne10, src1_ncols,
                                                                  src1_padded_col_size, stream);
                        } ଧର (ଅଚଳ sycl::exception & exc) {
                            std::cerr << "Quantize_row_q8_1_sycl error" << exc.what()
                                      << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
                            std::exit(1);
                        }
                    }
                }

                ଯଦି (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_ଭାଜକ == 0) {
                    SYCL_CHECK(ggml_sycl_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_ଭାଜକ, dev[i].row_ତଳ, dev[i].row_ଉପର, stream));
                }
                ଯଦି (src1->type == GGML_TYPE_F16) {
                    src1_padded_col_size = (i0 * ne11 + src1_col_0) * ne10;
                }
                // do the computation
                SYCL_CHECK(CHECK_TRY_ERROR(op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[i].row_ତଳ, dev[i].row_ଉପର, src1_ncols, src1_padded_col_size, stream)));

                // copy dst to host or other device if necessary
                ଯଦି (!dst_on_device) {
                    ଶୂନ୍ୟ * dst_off_device = dst->data;
                    ଯଦି (split) {
                        // src0 = weight matrix is saved as a transposed matrix for better memory layout.
                        // dst is NOT transposed.
                        // The outputs of matrix matrix multiplications can therefore NOT simply be concatenated for >1 GPU.
                        // Instead they need to be copied to the correct slice in ne0 = dst row index.
                        // If dst is a vector with ne0 == 1 then you don't have to do this but it still produces correct results.
                        ଦଶମିକ * dhf_dst_i = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ଆକାର(ଦଶମିକ));
                        dhf_dst_i += src1_col_0*ne0 + dev[i].row_ତଳ;

                        SYCL_CHECK(CHECK_TRY_ERROR(dpct::async_dpct_memcpy(
                            dhf_dst_i, ne0 * ଆକାର(ଦଶମିକ), dst_dd_i,
                            row_diff * ଆକାର(ଦଶମିକ), row_diff * ଆକାର(ଦଶମିକ),
                            src1_ncols, dpct::device_to_device, *stream)));
                    } ଅନ୍ୟଥା {
                        ଦଶମିକ * dhf_dst_i = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_off_device + i02*nb2 + i03*nb3);
                        GGML_ASSERT(dst->nb[1] == ne0*ଆକାର(ଦଶମିକ));
                        dhf_dst_i += src1_col_0*ne0;
                        SYCL_CHECK(CHECK_TRY_ERROR(
                            stream->memcpy(dhf_dst_i, dst_dd_i,
                                           src1_ncols * ne0 * ଆକାର(ଦଶମିକ)).wait()));
                    }
                }

                // add event for the main device to wait on until other device is done
                ଯଦି (split && (i != ctx.device || is != 0)) {
                    SYCL_CHECK(CHECK_TRY_ERROR(
                        *src0_extra->events[i][is] =
                            stream->ext_oneapi_submit_barrier()));
                }
            }
        }
    }

    // main device waits for all other devices to be finished
    ଯଦି (split && ggml_sycl_info().device_ଗଣନା > 1) {
        int64_t is_max = (ne11 + MUL_MAT_SRC1_COL_STRIDE - 1) / MUL_MAT_SRC1_COL_STRIDE;
        is_max = is_max <= GGML_SYCL_MAX_STREAMS ? is_max : GGML_SYCL_MAX_STREAMS;

        ggml_sycl_set_device(ctx.device);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; ++i) {
            ଯଦି (dev[i].row_ତଳ == dev[i].row_ଉପର) {
                ଜାରି;
            }
            ପାଇଁ (int64_t is = 0; is < is_max; ++is) {
                SYCL_CHECK(CHECK_TRY_ERROR(
                    ctx.stream()->ext_oneapi_submit_barrier(
                        {*src0_extra->events[i][is]})));
            }
        }
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_repeat_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat_back(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_get_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_get_rows_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_get_rows_back(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_norm(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_rms_norm(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_rms_norm_back(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_l2_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_l2_norm(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_group_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_group_norm(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mul_mat_vec_p021(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor *src0,
                                       ଅଚଳ ggml_tensor *src1,
                                       ggml_tensor *dst) ଚେଷ୍ଟା {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne01 = src0->ne[1];
    ଅଚଳ int64_t ne02 = src0->ne[2];

    ଅଚଳ int64_t ne12 = src1->ne[2];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ଶୂନ୍ୟ  * src0_ddq = src0->data;
    ଦଶମିକ * src1_ddf = (ଦଶମିକ *) src1->data;
    ଦଶମିକ * dst_ddf  = (ଦଶମିକ *) dst->data;

    ggml_mul_mat_p021_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mul_mat_vec_nc(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor *src0,
                                     ଅଚଳ ggml_tensor *src1,
                                     ggml_tensor *dst) ଚେଷ୍ଟା {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] == 1);
    GGML_ASSERT(src1->ne[3] == 1);

    ଅଚଳ int64_t ne00 = src0->ne[0];
    ଅଚଳ int64_t ne01 = src0->ne[1];
    ଅଚଳ int64_t ne02 = src0->ne[2];

    ଅଚଳ int64_t nb01 = src0->nb[1];
    ଅଚଳ int64_t nb02 = src0->nb[2];

    ଅଚଳ int64_t ne12 = src1->ne[2];
    ଅଚଳ int64_t nb11 = src1->nb[1];

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr main_stream = ctx.stream();

    ଶୂନ୍ୟ  * src0_ddq = src0->data;
    ଦଶମିକ * src1_ddf = (ଦଶମିକ *) src1->data;
    ଦଶମିକ * dst_ddf  = (ଦଶମିକ *) dst->data;

    ଅଚଳ int64_t row_stride_x = nb01 / ଆକାର(sycl::half);
    ଅଚଳ int64_t channel_stride_x = nb02 / ଆକାର(sycl::half);
    ଅଚଳ int64_t channel_stride_y = nb11 / ଆକାର(ଦଶମିକ);

    ggml_mul_mat_vec_nc_f16_f32_sycl(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x,channel_stride_y, main_stream);
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ k_compute_batched_ptrs(ଅଚଳ sycl::half * src0_as_f16, ଅଚଳ sycl::half * src1_as_f16, ଶୂନ୍ୟ * dst,
                                   ଅଚଳ ଶୂନ୍ୟ ** ptrs_src, ଶୂନ୍ୟ ** ptrs_dst, int64_t ne12, int64_t ne13, int64_t ne23,
                                   size_t nb02, size_t nb03, size_t nb12, size_t nb13, size_t nbd2, size_t nbd3,
                                   int64_t r2, int64_t r3, ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ int64_t i13 = item_ct1.get_group(2) * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);
    ଅଚଳ int64_t i12 = item_ct1.get_group(1) * item_ct1.get_local_range(1) + item_ct1.get_local_id(1);

    ଯଦି (i13 >= ne13 || i12 >= ne12) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t i03 = i13 / r3;
    ଅଚଳ int64_t i02 = i12 / r2;

    ଅଚଳ uint8_t * src0_bytes = reinterpret_cast<ଅଚଳ uint8_t *>(src0_as_f16);
    ଅଚଳ uint8_t * src1_bytes = reinterpret_cast<ଅଚଳ uint8_t *>(src1_as_f16);
    uint8_t *       dst_bytes  = static_cast<uint8_t *>(dst);

    ptrs_src[0 * ne23 + i12 + i13 * ne12] = src0_bytes + i02 * nb02 + i03 * nb03;
    ptrs_src[1 * ne23 + i12 + i13 * ne12] = src1_bytes + i12 * nb12 + i13 * nb13;
    ptrs_dst[0 * ne23 + i12 + i13 * ne12] = dst_bytes + i12 * nbd2 + i13 * nbd3;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mul_mat_batched_sycl(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0,
                                           ଅଚଳ ggml_tensor * src1, ggml_tensor * dst) ଚେଷ୍ଟା {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    // TODO: see https://github.com/ggml-org/llama.cpp/pull/13155
    // Batched mul_mat requires a rewrite to support both oneDNN and non-contiguous dst
    GGML_ASSERT(ggml_is_contiguous(dst));

    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    queue_ptr queue = ctx.stream();

    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    ଅଚଳ sycl::half * src0_f16 = static_cast<ଅଚଳ sycl::half *>(src0->data);
    ଦଶମିକ *            dst_ddf  = static_cast<ଦଶମିକ *>(dst->data);

    ଅଚଳ sycl::half * src1_f16       = static_cast<ଅଚଳ sycl::half *>(src1->data);
#ଯଦି GGML_SYCL_DNNL
    ଅଚଳ size_t       type_size_src0 = ggml_type_size(src0->type);
#endif
    ଅଚଳ size_t       type_size_src1 = ggml_type_size(src1->type);

    ତାର୍କିକ is_src0_cont_2 = ggml_is_contiguous_2(src0);
    ତାର୍କିକ is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // SRC1 strides
    int64_t                          s11 = nb11 / type_size_src1;
    int64_t                          s12 = nb12 / type_size_src1;
    int64_t                          s13 = nb13 / type_size_src1;
    ggml_sycl_pool_alloc<sycl::half> src1_f16_alloc(ctx.pool());

    // convert src1 to fp16
    ଯଦି (src1->type != GGML_TYPE_F16) {
        scope_op_debug_print    scope_dbg_print(__func__, "/to_fp16_nc_sycl", dst, /*num_src=*/2,
                                                " : converting src1 to fp16");

#ଯଦି GGML_SYCL_DNNL
        // iterate tensor dims and find the slowest moving dim and stride
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା last_dim=0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା last_str=0;
        size_t largest_str=0;
        ପାଇଁ(ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i< 4; i++){
            // last stride is always the largest
            ଯଦି(src1->nb[i] == largest_str){
                ଯଦି(src1->ne[last_dim] == 1){
                    last_str = i;
                    last_dim = i;
                }
            }
            ଯଦି(src1->nb[i] > largest_str){
                largest_str = src1->nb[i];
                last_str = i;
                last_dim = i;
            }

        }
        // oneDNN handles strided data and does not need overhead of ggml_get_to_fp16_nc_sycl
        ଅଚଳ int64_t ne_src1 = src1->nb[last_str] * src1->ne[last_dim] / type_size_src1;
        src1_f16_alloc.alloc(ne_src1);
        ଅଚଳ to_fp16_sycl_t to_fp16_sycl = ggml_get_to_fp16_sycl(src1->type, dst);
        GGML_ASSERT(to_fp16_sycl != nullptr);
        to_fp16_sycl(src1_f16, src1_f16_alloc.get(), ne_src1, queue);
# ଅନ୍ୟଥା
        ଅଚଳ int64_t ne_src1 = ggml_nelements(src1);
        src1_f16_alloc.alloc(ne_src1);
        ଅଚଳ to_fp16_nc_sycl_t to_fp16_nc_sycl = ggml_get_to_fp16_nc_sycl(src1->type);
        GGML_ASSERT(to_fp16_nc_sycl != nullptr);
        to_fp16_nc_sycl(src1_f16, src1_f16_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, queue);
#endif

        src1_f16 = src1_f16_alloc.get();
        s11      = ne10;
        s12      = ne11 * s11;
        s13      = ne12 * s12;

        is_src1_cont_2 = ସତ୍ୟ;
    }

    ggml_sycl_pool_alloc<sycl::half> dst_f16(ctx.pool());

    dpct::library_data_t mkl_compute_type = dpct::library_data_t::real_float;
    dpct::library_data_t mkl_data_type    = dpct::library_data_t::real_float;

    // dst strides
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    ଅଚଳ ଦଶମିକ alpha_f32 = 1.0f;
    ଅଚଳ ଦଶମିକ beta_f32  = 0.0f;

    ଅଚଳ ଶୂନ୍ୟ * alpha = &alpha_f32;
    ଅଚଳ ଶୂନ୍ୟ * beta  = &beta_f32;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);
    GGML_ASSERT(ne01 == static_cast<int64_t>(nb1/nb0));
    GGML_ASSERT(ne10 == ne00);

    // broadcast factors
    ଅଚଳ int64_t r2 = ne12 / ne02;
    ଅଚଳ int64_t r3 = ne13 / ne03;

#ଯଦି GGML_SYCL_DNNL
    ଯଦି (g_ggml_sycl_enable_dnn && ggml_sycl_dnnl_has_optimized_gemm(ggml_sycl_get_device())) {
            int64_t str_a0 = nb00 / type_size_src0;
            int64_t str_a1 = nb01 / type_size_src0;
            int64_t str_a2 = nb02 / type_size_src0;

            int64_t str_b0 = nb10 / type_size_src1;
            int64_t str_b1 = nb11 / type_size_src1;
            int64_t str_b2 = nb12 / type_size_src1;

            auto launch_gemm_for_batches = [&ctx, queue](ଅଚଳ sycl::half *src0,
                                                ଅଚଳ sycl::half *src1, ଦଶମିକ *dst,
                                                int64_t a0, int64_t a1, int64_t batcha,
                                                int64_t /*b0*/, int64_t b1, int64_t batchb,
                                                int64_t sa0, int64_t sa1, int64_t sa2,
                                                int64_t sb0, int64_t sb1, int64_t sb2,
                                                int64_t sd2) {
                ତାର୍କିକ supported_broadcast = batchb == batcha ? ସତ୍ୟ
                        : batchb == 1 || batcha == 1        ? ସତ୍ୟ
                                                            : ମିଥ୍ୟା;
                ଯଦି (supported_broadcast) {
                    DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2, src1,
                            DnnlGemmWrapper::to_dt<sycl::half>(), sb0, sb1, sb2, dst,
                            DnnlGemmWrapper::to_dt<ଦଶମିକ>(), queue, batcha, batchb);
                } ଅନ୍ୟଥା {
                    // iterate over batches from smaller set of matrices (matrix 0)
                    int64_t batches0 = batcha;
                    int64_t batches1 = batchb;

                    ଯଦି (batches0 > batches1) {
                        int64_t ଅଙ୍କ_mul_mats = batches1;
                        int64_t sub_batch = batches0 / ଅଙ୍କ_mul_mats;
                        // src0 is batched and bigger, shift and multiply with src1
                        ପାଇଁ (int64_t i0 = 0; i0 < ଅଙ୍କ_mul_mats; i0++) {
                            ଅଚଳ sycl::half *src0_shifted = src0 + (sa2 * i0 * sub_batch);
                            ଅଚଳ sycl::half *src1_shifted = src1 + (sb2 * i0);
                            ଦଶମିକ *dst_shifted = dst + (sd2 * i0 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<ଦଶମିକ>(),
                                    queue, sub_batch, 1);
                        }
                    } ଅନ୍ୟଥା {
                        int64_t ଅଙ୍କ_mul_mats = batches0;
                        int64_t sub_batch = batches1 / ଅଙ୍କ_mul_mats;
                        // src1 is batched and bigger, shift and multiply with src0
                        ପାଇଁ (int64_t i1 = 0; i1 < ଅଙ୍କ_mul_mats; i1++) {
                            ଅଚଳ sycl::half *src0_shifted = src0 + (sa2 * i1);
                            ଅଚଳ sycl::half *src1_shifted = src1 + (sb2 * i1 * sub_batch);
                            ଦଶମିକ *dst_shifted = dst + (sd2 * i1 * sub_batch);
                            DnnlGemmWrapper::gemm(ctx, a1, b1, a0, src0_shifted,
                                    DnnlGemmWrapper::to_dt<sycl::half>(), sa0, sa1, sa2,
                                    src1_shifted, DnnlGemmWrapper::to_dt<sycl::half>(), sb0,
                                    sb1, sb2, dst_shifted, DnnlGemmWrapper::to_dt<ଦଶମିକ>(),
                                    queue, 1, sub_batch);
                        }
                    }
                }
            };

            ଅଚଳ ତାର୍କିକ cont_batches_dim2_a = nb02 * ne02 == nb03;
            ଅଚଳ ତାର୍କିକ cont_batches_dim2_b = nb12 * ne12 == nb13;
            ଅଚଳ ତାର୍କିକ cont_batches_dim3_a = ne02 == 1 && nb02 * ne01 == nb03;
            ଅଚଳ ତାର୍କିକ cont_batches_dim3_b = ne12 == 1 && nb12 * ne11 == nb13;
            ଯଦି (cont_batches_dim2_a && cont_batches_dim2_b) {
                // A batch is considered contiguous if the dimension 2 is not strided
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a2, str_b0, str_b1,
                        str_b2, nb2 / ଆକାର(ଦଶମିକ));
            } ଅନ୍ୟଥା ଯଦି (cont_batches_dim3_a && cont_batches_dim3_b) {
                // This case is similar to the one above with the difference that only the batch in dimension 3 is used and the dimension 2 is of size 1.
                int64_t batches0 = ne02 * ne03;
                int64_t batches1 = ne12 * ne13;
                int64_t str_a3 = nb03 / type_size_src0;
                int64_t str_b3 = nb13 / type_size_src1;
                launch_gemm_for_batches(src0_f16, src1_f16, dst_ddf, ne00, ne01, batches0,
                        ne10, ne11, batches1, str_a0, str_a1, str_a3, str_b0, str_b1,
                        str_b3, nb2 / ଆକାର(ଦଶମିକ));
            } ଅନ୍ୟଥା {
                ପାଇଁ (int64_t b_a = 0; b_a < ne03; b_a++) {
                    ଅଚଳ sycl::half *src0_f16_shifted
                            = src0_f16 + (nb03 * b_a / type_size_src0);
                    ଅଚଳ sycl::half *src1_f16_shifted
                            = src1_f16 + (nb13 * b_a / type_size_src1);
                    ଦଶମିକ *dst_shifted = dst_ddf + (nb3 * b_a / ଆକାର(ଦଶମିକ));
                    int64_t batches0 = ne02;
                    int64_t batches1 = ne12;
                    launch_gemm_for_batches(src0_f16_shifted, src1_f16_shifted, dst_shifted,
                            ne00, ne01, batches0, ne10, ne11, batches1, str_a0, str_a1,
                            str_a2, str_b0, str_b1, str_b2, nb2 / ଆକାର(ଦଶମିକ));
                }
            }

    }
    ଅନ୍ୟଥା
#endif
    {
        ଯଦି (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
            // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
            ଅଚଳ int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
            ଅଚଳ int64_t smb = ne12 == 1 ? s13       : s12;

            // there is no broadcast and src0, src1 are contiguous across dims 2, 3
            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(*queue, oneapi::mkl::transpose::trans,
                                                        oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                                                        src0_f16, dpct::library_data_t::real_half, nb01 / nb00, sma,
                                                        src1_f16, dpct::library_data_t::real_half, s11, smb, beta, dst_ddf,
                                                        mkl_data_type, ne0, ne1 * ne0, ne12 * ne13, mkl_compute_type)));
        } ଅନ୍ୟଥା {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ne23 = ne12 * ne13;

            ggml_sycl_pool_alloc<ଅଚଳ ଶୂନ୍ୟ *>         ptrs_src(ctx.pool(), 2 * ne23);
            ggml_sycl_pool_alloc<ଶୂନ୍ୟ *>               ptrs_dst(ctx.pool(), 1 * ne23);
            ggml_sycl_pool_alloc<matrix_info_t<ଦଶମିକ>> matrix_info(ctx.host_pool(), 1);

            sycl::range<3> block_dims(1, ne12, ne13);
            queue->submit([&](sycl::handler & cgh) {
                ଅଚଳ ଶୂନ୍ୟ ** ptrs_src_get = ptrs_src.get();
                ଶୂନ୍ୟ **       ptrs_dst_get = ptrs_dst.get();
                size_t        nb12_scaled  = src1->type == GGML_TYPE_F16 ? nb12 : s12 * ଆକାର(sycl::half);
                size_t        nb13_scaled  = src1->type == GGML_TYPE_F16 ? nb13 : s13 * ଆକାର(sycl::half);
                cgh.parallel_for(sycl::nd_range<3>(block_dims, block_dims), [=](sycl::nd_item<3> item_ct1) {
                    k_compute_batched_ptrs(src0_f16, src1_f16, dst_ddf, ptrs_src_get, ptrs_dst_get, ne12, ne13, ne23, nb02,
                                           nb03, nb12_scaled, nb13_scaled, nbd2, nbd3, r2, r3, item_ct1);
                });
            });

            SYCL_CHECK(CHECK_TRY_ERROR(dpct::gemm_batch(
                *queue, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, ne01, ne11, ne10, alpha,
                (ଅଚଳ ଶୂନ୍ୟ **) (ptrs_src.get() + 0 * ne23), dpct::library_data_t::real_half, nb01 / nb00,
                (ଅଚଳ ଶୂନ୍ୟ **) (ptrs_src.get() + 1 * ne23), dpct::library_data_t::real_half, s11, beta,
                (ଶୂନ୍ୟ **) (ptrs_dst.get() + 0 * ne23), mkl_data_type, ne0, ne23, mkl_compute_type, matrix_info.get())));
        }
    }
} ଧର (ଅଚଳ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ଗଣନା ଶ୍ରେଣୀ mul_mat_algo {
    DMMV         = 0,
    MMVQ         = 1,
    MUL_MAT_SYCL = 2,
};

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_supports_mmq(ଗଣନା ggml_type type) {
    // TODO: accuracy issues in MMQ
    GGML_UNUSED(type);
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_supports_reorder_mul_mat_sycl(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            ଫେରନ୍ତୁ !g_ggml_sycl_prioritize_dmmv;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_supports_reorder_dmmv(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ଧାଡ଼ିଭିତରେ ତାର୍କିକ ggml_sycl_supports_reorder_mmvq(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_supports_reorder_esimd(ଗଣନା ggml_type type) {
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
#ଅନ୍ୟଥା
    GGML_UNUSED(type);
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
#endif
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_supports_dmmv(ଗଣନା ggml_type type) {
    ବାଛ (type) {
        ସ୍ଥିତି GGML_TYPE_Q1_0:
        ସ୍ଥିତି GGML_TYPE_Q4_0:
        ସ୍ଥିତି GGML_TYPE_Q4_1:
        ସ୍ଥିତି GGML_TYPE_Q5_0:
        ସ୍ଥିତି GGML_TYPE_Q5_1:
        ସ୍ଥିତି GGML_TYPE_Q8_0:
        ସ୍ଥିତି GGML_TYPE_Q2_K:
        ସ୍ଥିତି GGML_TYPE_Q3_K:
        ସ୍ଥିତି GGML_TYPE_Q4_K:
        ସ୍ଥିତି GGML_TYPE_Q5_K:
        ସ୍ଥିତି GGML_TYPE_Q6_K:
        ସ୍ଥିତି GGML_TYPE_F16:
        ସ୍ଥିତି GGML_TYPE_BF16:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

// Helper functions to unify device memory allocation for both async and sync paths
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ * sycl_ext_malloc_device(dpct::queue_ptr stream, size_t size) {
    ତାର୍କିକ use_async = g_ggml_sycl_use_async_mem_op;
#ଯଦି defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    ଯଦି (use_async) {
        ଶୂନ୍ୟ * ptr = syclex::async_malloc(*stream, sycl::usm::alloc::device, size);
        ggml_sycl_memtrace_add(GGML_SYCL_MEM_ASYNC, ptr, size);
        ଫେରନ୍ତୁ ptr;
    }
#ଅନ୍ୟଥା
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ଫେରନ୍ତୁ ggml_sycl_malloc_device(size, *stream);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sycl_ext_free(dpct::queue_ptr stream, ଶୂନ୍ୟ * ptr) {
    ତାର୍କିକ use_async = g_ggml_sycl_use_async_mem_op;
#ଯଦି defined(GGML_SYCL_GRAPH) && SYCL_EXT_ONEAPI_ASYNC_MEMORY_ALLOC
    ଯଦି (use_async) {
        ggml_sycl_memtrace_del(ptr);
        syclex::async_free(*stream, ptr);
        ଫେରନ୍ତୁ;
    }
#ଅନ୍ୟଥା
    // If async allocation extension is not available, use_async should always be false.
    GGML_ASSERT(!use_async);
#endif
    ggml_sycl_free_device(ptr, *stream);
}

// RAII wrapper for temporary reorder buffers with optional host memory fallback.
// When device allocation fails and GGML_SYCL_HOST_MEM_FALLBACK is enabled,
// falls back to host memory so the reorder kernel can still run (over PCIe).
// Device access to host memory requires Linux kernel 6.8+ (Ubuntu 26.04+).
ସଂରଚନା sycl_reorder_temp_buffer {
    ଶୂନ୍ୟ *          ptr  = nullptr;
    dpct::queue_ptr stream;

    sycl_reorder_temp_buffer(dpct::queue_ptr stream, size_t size) : stream(stream) {
        ptr = sycl_ext_malloc_device(stream, size);
#ifdef GGML_SYCL_HOST_MEM_FALLBACK
        ଯଦି (!ptr) {
            ptr = sycl::malloc_host(size, *stream);
            ଯଦି (ptr) {
                host_fallback = ସତ୍ୟ;
                GGML_LOG_WARN("%s: device alloc of %zu bytes failed, using host memory fallback\n", __func__, size);
            }
        }
#endif
    }

    ~sycl_reorder_temp_buffer() {
        ଯଦି (!ptr) {
            ଫେରନ୍ତୁ;
        }
        ଯଦି (host_fallback) {
            sycl::ମୁକ୍ତ(ptr, *stream);
        } ଅନ୍ୟଥା {
            sycl_ext_free(stream, ptr);
        }
    }

    explicit operator ତାର୍କିକ() ଅଚଳ { ଫେରନ୍ତୁ ptr != nullptr; }

    sycl_reorder_temp_buffer(ଅଚଳ sycl_reorder_temp_buffer &)            = delete;
    sycl_reorder_temp_buffer & operator=(ଅଚଳ sycl_reorder_temp_buffer &) = delete;

ବ୍ୟକ୍ତିଗତ:
    ତାର୍କିକ host_fallback = ମିଥ୍ୟା;
};

ସ୍ଥିର ତାର୍କିକ reorder_qw_q4_0(uint8_t * data_device, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ଆକାର(block_q4_0) == 0));
    GGML_ASSERT((offset % ଆକାର(block_q4_0) == 0));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset_blks = offset / ଆକାର(block_q4_0);
    auto qs_ptr      = data_device + offset_blks * QK4_0 / 2;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows / 2) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / ଆକାର(block_q4_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            ଅଚଳ block_q4_0* x = (ଅଚଳ block_q4_0*)tmp_buf;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib = i;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK4_0/2; j ++)
            {
                *(qs_ptr + ib * QK4_0 / 2 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q8_0(uint8_t * data_device, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ncols, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nrows, size_t size, size_t offset,
                            dpct::queue_ptr stream) {
    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    GGML_ASSERT((size % ଆକାର(block_q8_0) == 0));
    GGML_ASSERT((offset % ଆକାର(block_q8_0) == 0));
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା offset_blks = offset / ଆକାର(block_q8_0);
    auto qs_ptr = data_device + offset_blks * QK8_0;
    auto d_ptr = (sycl::half*)(qs_ptr + ncols * nrows) + offset_blks;

    auto reorder_event = stream->parallel_for(
        size / ଆକାର(block_q8_0),
            [=](auto i) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            ଅଚଳ block_q8_0* x = (ଅଚଳ block_q8_0*)tmp_buf;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib = i;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; j++)
            {
                *((int8_t*)qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
            }
            *(d_ptr + ib) = x[ib].d;
        });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Reorder each Q8_0 expert slice into a self-contained [qs][d] SoA layout.
ସ୍ଥିର ତାର୍କିକ reorder_qw_q8_0_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % ଆକାର(block_q8_0) == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    blocks_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (expert_bytes / ଆକାର(block_q8_0));
    ଅଚଳ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_blocks = blocks_per_expert * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          gb   = gb_;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          e    = gb / blocks_per_expert;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib   = gb % blocks_per_expert;
        ଅଚଳ block_q8_0 * x    = (ଅଚଳ block_q8_0 *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr = base;
        auto * d_ptr  = (sycl::half *) (qs_ptr + (size_t) blocks_per_expert * QK8_0);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK8_0; ++j) {
            *((int8_t *) qs_ptr + ib * QK8_0 + j) = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q4_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_q4_K) == 0);
    GGML_ASSERT(offset % ଆକାର(block_q4_K) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_q4_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * scales_ptr = qs_ptr + QK_K / 2 * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_q4_K * x  = (ଅଚଳ block_q4_K *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Reorder each expert slice into a self-contained SoA layout.
ସ୍ଥିର ତାର୍କିକ reorder_qw_q4_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % ଆକାର(block_q4_K) == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    blocks_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (expert_bytes / ଆକାର(block_q4_K));
    ଅଚଳ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_blocks = blocks_per_expert * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          gb   = gb_;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          e    = gb / blocks_per_expert;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib   = gb % blocks_per_expert;
        ଅଚଳ block_q4_K * x    = (ଅଚଳ block_q4_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * scales_ptr = qs_ptr + QK_K / 2 * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Reorder each Q5_K expert slice into [qs][qh][scales][dm].
ସ୍ଥିର ତାର୍କିକ reorder_qw_q5_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % ଆକାର(block_q5_K) == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    blocks_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (expert_bytes / ଆକାର(block_q5_K));
    ଅଚଳ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_blocks = blocks_per_expert * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          gb   = gb_;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          e    = gb / blocks_per_expert;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib   = gb % blocks_per_expert;
        ଅଚଳ block_q5_K * x    = (ଅଚଳ block_q5_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * qs_ptr     = base;
        auto * qh_ptr     = qs_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 8) * blocks_per_expert;
        auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * blocks_per_expert);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }
        dm_ptr[ib] = x[ib].dm;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Reorder each Q6_K expert slice into [ql][qh][scales][d].
ସ୍ଥିର ତାର୍କିକ reorder_qw_q6_k_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % ଆକାର(block_q6_K) == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା    blocks_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (expert_bytes / ଆକାର(block_q6_K));
    ଅଚଳ size_t total_bytes       = expert_bytes * (size_t) n_expert;

    sycl_reorder_temp_buffer tmp(stream, total_bytes);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, total_bytes);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, total_bytes)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_blocks = blocks_per_expert * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_expert;
    auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          gb   = gb_;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          e    = gb / blocks_per_expert;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib   = gb % blocks_per_expert;
        ଅଚଳ block_q6_K * x    = (ଅଚଳ block_q6_K *) (tmp_buf + (size_t) e * expert_bytes);
        uint8_t *          base = data_device + (size_t) e * expert_bytes;

        auto * ql_ptr     = base;
        auto * qh_ptr     = ql_ptr + (QK_K / 2) * blocks_per_expert;
        auto * scales_ptr = qh_ptr + (QK_K / 4) * blocks_per_expert;
        auto * d_ptr      = (sycl::half *) (scales_ptr + (QK_K / 16) * blocks_per_expert);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            ql_ptr[ib * (QK_K / 2) + j] = x[ib].ql[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 4; ++j) {
            qh_ptr[ib * (QK_K / 4) + j] = x[ib].qh[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

// Reorder each MXFP4 expert slice into [qs][e]: 16-byte nibble blocks, then one E8M0 byte per block.
// Experts are self-contained, so the tensor is reordered a few experts at a time through a small
// temporary: a whole-tensor temporary (hundreds of MB) can exceed the VRAM left on a nearly full card,
// and on Windows the driver then pages device memory out to host RAM instead of failing.
ସ୍ଥିର ତାର୍କିକ reorder_qw_mxfp4_moe(uint8_t * data_device, size_t expert_bytes, int64_t n_expert, dpct::queue_ptr stream) {
    GGML_ASSERT(expert_bytes % ଆକାର(block_mxfp4) == 0);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     blocks_per_expert = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (expert_bytes / ଆକାର(block_mxfp4));
    ଅଚଳ size_t  max_chunk_bytes   = 32u << 20;
    ଅଚଳ int64_t chunk_experts     = std::max<int64_t>(1, std::min<int64_t>(n_expert, (int64_t) (max_chunk_bytes / expert_bytes)));

    sycl_reorder_temp_buffer tmp(stream, (size_t) chunk_experts * expert_bytes);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__,
                      (size_t) chunk_experts * expert_bytes);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    // the queue is in-order: each chunk's copy into tmp_buf waits for the previous chunk's kernel
    ପାଇଁ (int64_t e0 = 0; e0 < n_expert; e0 += chunk_experts) {
        ଅଚଳ int64_t n_chunk = std::min(chunk_experts, n_expert - e0);
        uint8_t *     chunk   = data_device + (size_t) e0 * expert_bytes;

        sycl::event copy_event;
        SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, chunk, (size_t) n_chunk * expert_bytes)));
        ଯଦି (!g_ggml_sycl_use_async_mem_op) {
            copy_event.wait();
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_blocks = blocks_per_expert * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_chunk;
        auto reorder_event = stream->parallel_for(total_blocks, [=](auto gb_) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           gb   = gb_;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           e    = gb / blocks_per_expert;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା           ib   = gb % blocks_per_expert;
            ଅଚଳ block_mxfp4 * x    = (ଅଚଳ block_mxfp4 *) (tmp_buf + (size_t) e * expert_bytes);
            uint8_t *           base = chunk + (size_t) e * expert_bytes;

            uint8_t * qs_ptr = base;
            uint8_t * e_ptr  = qs_ptr + (QK_MXFP4 / 2) * (size_t) blocks_per_expert;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_MXFP4 / 2; ++j) {
                qs_ptr[(size_t) ib * (QK_MXFP4 / 2) + j] = x[ib].qs[j];
            }
            e_ptr[ib] = x[ib].e;
        });
        ଯଦି (!g_ggml_sycl_use_async_mem_op) {
            reorder_event.wait_and_throw();
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q2_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_q2_K) == 0);
    GGML_ASSERT(offset % ଆକାର(block_q2_K) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_q2_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *        qs_ptr     = data_device;
    auto *        scales_ptr = qs_ptr + (QK_K / 4) * nblocks;
    sycl::half2 * dm_ptr     = (sycl::half2 *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_q2_K * x  = (ଅଚଳ block_q2_K *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 16; ++j) {
            scales_ptr[ib * (QK_K / 16) + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q3_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_q3_K) == 0);
    GGML_ASSERT(offset % ଆକାର(block_q3_K) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_q3_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       qs_ptr     = data_device;
    auto *       hmask_ptr  = qs_ptr + (QK_K / 4) * nblocks;
    auto *       scales_ptr = hmask_ptr + (QK_K / 8) * nblocks;
    sycl::half * d_ptr      = (sycl::half *) (scales_ptr + 12 * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_q3_K * x  = (ଅଚଳ block_q3_K *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 8; ++j) {
            hmask_ptr[ib * (QK_K / 8) + j] = x[ib].hmask[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 12; ++j) {
            scales_ptr[ib * 12 + j] = x[ib].scales[j];
        }

        d_ptr[ib] = x[ib].d;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q5_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_q5_K) == 0);
    GGML_ASSERT(offset % ଆକାର(block_q5_K) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_q5_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr     = data_device;
    auto * qh_ptr     = qs_ptr + (QK_K / 2) * nblocks;
    auto * scales_ptr = qh_ptr + (QK_K / 8) * nblocks;
    auto * dm_ptr     = (sycl::half2 *) (scales_ptr + K_SCALE_SIZE * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_q5_K * x  = (ଅଚଳ block_q5_K *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            qs_ptr[ib * (QK_K / 2) + j] = x[ib].qs[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 8; ++j) {
            qh_ptr[ib * (QK_K / 8) + j] = x[ib].qh[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < K_SCALE_SIZE; ++j) {
            scales_ptr[ib * K_SCALE_SIZE + j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].dm;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_q6_k(uint8_t * data_device, size_t size, size_t offset, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_q6_K) == 0);
    GGML_ASSERT(offset % ଆକାର(block_q6_K) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_q6_K);

    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto *       ql_ptr     = data_device;
    auto *       qh_ptr     = ql_ptr + (QK_K / 2) * nblocks;
    auto *       scales_ptr = qh_ptr + (QK_K / 4) * nblocks;
    sycl::half * dm_ptr     = (sycl::half *) (scales_ptr + (QK_K / 16) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_q6_K * x  = (ଅଚଳ block_q6_K *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା          ib = i;

        ଅଚଳ uint8_t * ql              = x[ib].ql;
        ଅଚଳ uint8_t * qh              = x[ib].qh;
        uint8_t *       base_ql_ptr     = ql_ptr + (QK_K / 2) * ib;
        uint8_t *       base_qh_ptr     = qh_ptr + (QK_K / 4) * ib;
        uint8_t *       base_scales_ptr = scales_ptr + (QK_K / 16) * ib;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 2; ++j) {
            base_ql_ptr[j] = ql[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 4; ++j) {
            base_qh_ptr[j] = qh[j];
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 16; ++j) {
            base_scales_ptr[j] = x[ib].scales[j];
        }

        dm_ptr[ib] = x[ib].d;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_iq3_s(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_iq3_s) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_iq3_s);
    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr       = data_device;
    auto * qh_ptr       = qs_ptr + (QK_K / 4) * nblocks;
    auto * signs_ptr    = qh_ptr + (QK_K / 32) * nblocks;
    auto * metadata_ptr = signs_ptr + (QK_K / 8) * nblocks;

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_iq3_s * x = (ଅଚଳ block_iq3_s *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 4; ++j) {
            qs_ptr[ib * (QK_K / 4) + j] = x[ib].qs[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 32; ++j) {
            qh_ptr[ib * (QK_K / 32) + j] = x[ib].qh[j];
        }
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < QK_K / 8; ++j) {
            signs_ptr[ib * (QK_K / 8) + j] = x[ib].signs[j];
        }

        uint8_t * metadata = metadata_ptr + ib * (ଆକାର(ggml_half) + IQ3S_N_SCALE);
        *reinterpret_cast<ggml_half *>(metadata) = x[ib].d;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < IQ3S_N_SCALE; ++j) {
            metadata[ଆକାର(ggml_half) + j] = x[ib].scales[j];
        }
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw_iq3_xxs(uint8_t * data_device, size_t size, dpct::queue_ptr stream) {
    GGML_ASSERT(size % ଆକାର(block_iq3_xxs) == 0);

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nblocks = size / ଆକାର(block_iq3_xxs);
    sycl_reorder_temp_buffer tmp(stream, size);
    ଯଦି (!tmp) {
        GGML_LOG_WARN("%s: failed to allocate %zu bytes for reorder temp buffer, skipping reorder\n", __func__, size);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    uint8_t * tmp_buf = static_cast<uint8_t *>(tmp.ptr);

    sycl::event copy_event;
    SYCL_CHECK(CHECK_TRY_ERROR(copy_event = stream->memcpy(tmp_buf, data_device, size)));
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        copy_event.wait();
    }

    auto * qs_ptr = data_device;
    auto * d_ptr  = reinterpret_cast<ggml_half *>(qs_ptr + (3 * QK_K / 8) * nblocks);

    auto reorder_event = stream->parallel_for(nblocks, [=](auto i) {
        ଅଚଳ block_iq3_xxs * x = (ଅଚଳ block_iq3_xxs *) tmp_buf;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ib = i;

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 3 * QK_K / 8; ++j) {
            qs_ptr[ib * (3 * QK_K / 8) + j] = x[ib].qs[j];
        }
        d_ptr[ib] = x[ib].d;
    });
    ଯଦି (!g_ggml_sycl_use_async_mem_op) {
        reorder_event.wait_and_throw();
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}

ସ୍ଥିର ତାର୍କିକ reorder_qw(ଅଚଳ ggml_tensor * src0, dpct::queue_ptr stream) {
    uint8_t * data_device = (uint8_t *) src0->data;
    size_t ncols = src0->ne[0];
    size_t nrows = src0->ne[1];
    size_t size = ggml_nbytes(src0);

    // MoE expert weights are addressed per expert via nb[2], so each slice must
    // remain self-contained after reorder.
    ଯଦି (src0->ne[2] > 1) {
        GGML_ASSERT((size_t) size == (size_t) src0->ne[2] * src0->nb[2]);
        ବାଛ (src0->type) {
            ସ୍ଥିତି GGML_TYPE_Q8_0:
                ଫେରନ୍ତୁ reorder_qw_q8_0_moe(data_device, src0->nb[2], src0->ne[2], stream);
            ସ୍ଥିତି GGML_TYPE_Q4_K:
                ଫେରନ୍ତୁ reorder_qw_q4_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            ସ୍ଥିତି GGML_TYPE_Q5_K:
                ଫେରନ୍ତୁ reorder_qw_q5_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            ସ୍ଥିତି GGML_TYPE_Q6_K:
                ଫେରନ୍ତୁ reorder_qw_q6_k_moe(data_device, src0->nb[2], src0->ne[2], stream);
            ସ୍ଥିତି GGML_TYPE_MXFP4:
                ଫେରନ୍ତୁ reorder_qw_mxfp4_moe(data_device, src0->nb[2], src0->ne[2], stream);
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }

    ବାଛ (src0->type) {
        ସ୍ଥିତି GGML_TYPE_Q4_0:
            ଫେରନ୍ତୁ reorder_qw_q4_0(data_device, ncols, nrows, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q8_0:
            ଫେରନ୍ତୁ reorder_qw_q8_0(data_device, ncols, nrows, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q2_K:
            ଫେରନ୍ତୁ reorder_qw_q2_k(data_device, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q3_K:
            ଫେରନ୍ତୁ reorder_qw_q3_k(data_device, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q4_K:
            ଫେରନ୍ତୁ reorder_qw_q4_k(data_device, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q5_K:
            ଫେରନ୍ତୁ reorder_qw_q5_k(data_device, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_Q6_K:
            ଫେରନ୍ତୁ reorder_qw_q6_k(data_device, size, 0, stream);
        ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
            ଫେରନ୍ତୁ reorder_qw_iq3_xxs(data_device, size, stream);
        ସ୍ଥିତି GGML_TYPE_IQ3_S:
            ଫେରନ୍ତୁ reorder_qw_iq3_s(data_device, size, stream);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}

ସ୍ଥିର ତାର୍କିକ should_reorder_tensor(ggml_backend_sycl_context& ctx, ଅଚଳ ggml_tensor * dst) {
    ଫେରନ୍ତୁ g_ggml_sycl_enable_optimize && //allow optimize, controlled by $GGML_SYCL_ENABLE_OPT
           ctx.opt_feature.reorder &&      //allow this device due to good perf, skip the devices with bad perf.
           dst->op == GGML_OP_MUL_MAT &&   //limit to some supported cases of Q4_0, to do for more cases.
           // ne[1] <= 8 so multi-column decode (spec / MTP verify) also bootstraps the reorder;
           // all reorderable types have a _switch_ncols kernel.
           dst->src[1]->ne[1] <= 8 && dst->src[1]->ne[2]==1 && dst->src[1]->ne[3]==1;
}

ସ୍ଥିର ଶୂନ୍ୟ opt_for_reorder(ggml_backend_sycl_context * ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * /* src1 */,
                            ggml_tensor * dst, mul_mat_algo mm_algorithm) {
    ଯଦି (!should_reorder_tensor(*ctx, dst)) {
        ଫେରନ୍ତୁ;
    }

    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ଯଦି (!extra || extra->optimized_feature.reorder) {
        ଫେରନ୍ତୁ;  // Skip permutations and already reordered tensors
    }

    ବାଛ (mm_algorithm) {
        ସ୍ଥିତି mul_mat_algo::DMMV:
            ଯଦି (!ggml_sycl_supports_reorder_dmmv(src0->type)) {
                ଫେରନ୍ତୁ;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି mul_mat_algo::MMVQ:
            ଯଦି (!ggml_sycl_supports_reorder_mmvq(src0->type)) {
                ଫେରନ୍ତୁ;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି mul_mat_algo::MUL_MAT_SYCL:
            ଯଦି (!ggml_sycl_supports_reorder_mul_mat_sycl(src0->type)) {
                ଫେରନ୍ତୁ;
            }
            ଭାଙ୍ଗ;
    }

    ଯଦି (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ସତ୍ୟ;  // Used to decode/dequan in next steps and avoid re-reordering
    }
}

// Lazily reorder supported MoE expert weights once their fused path is used.
ସ୍ଥିର ଶୂନ୍ୟ opt_for_reorder_id(ggml_backend_sycl_context * ctx, ଅଚଳ ggml_tensor * src0) {
    ଯଦି (!g_ggml_sycl_enable_optimize || !ctx->opt_feature.reorder) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (src0->type != GGML_TYPE_Q8_0 && src0->type != GGML_TYPE_Q4_K &&
        src0->type != GGML_TYPE_Q5_K && src0->type != GGML_TYPE_Q6_K && src0->type != GGML_TYPE_MXFP4) {
        ଫେରନ୍ତୁ;
    }
    // The MXFP4 reorder kernels use 8-byte vector loads, so every expert slice must stay aligned.
    ଯଦି (src0->type == GGML_TYPE_MXFP4 && (src0->nb[2] % 16 != 0 || (uintptr_t) src0->data % 16 != 0)) {
        ଫେରନ୍ତୁ;
    }
    ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
    ଯଦି (!extra || extra->optimized_feature.reorder) {
        ଫେରନ୍ତୁ;
    }
    ଯଦି (reorder_qw(src0, ctx->stream())) {
        extra->optimized_feature.reorder = ସତ୍ୟ;
    }
}


ସ୍ଥିର ତାର୍କିକ can_use_dequantize_mul_mat_vec(ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1, ggml_tensor * dst) {
    // The F16/BF16 qk=1 kernel iterates with stride 2*DMMV_X, requiring ne[0] to be
    // a multiple of 2*DMMV_X. Quantized types use block-structured kernels that only
    // need ne[0] % DMMV_X == 0.
    ଅଚଳ int64_t dmmv_x_required = (src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F16) ?
                                    2*GGML_SYCL_DMMV_X : GGML_SYCL_DMMV_X;
    ଫେରନ୍ତୁ ggml_sycl_supports_dmmv(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src0->ne[0] % dmmv_x_required == 0 && src1->ne[1] == 1;
}

ସ୍ଥିର ତାର୍କିକ can_use_mul_mat_vec_q(ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1, ggml_tensor * dst) {
    ଫେରନ୍ତୁ ggml_is_quantized(src0->type) && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
           src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mul_mat(ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0, ଅଚଳ ggml_tensor * src1, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    // Handle HADAMARAD hint given from further up the pipeline and pass it to the correct
    // kernel.
    //
    // The op check is not redundant: this backend also routes MUL_MAT_ID through here with a
    // stack copy of dst, which carries MUL_MAT_ID's own op_params. ggml_mul_mat_set_hint()
    // asserts GGML_OP_MUL_MAT for the same reason.
    ଯଦି (dst->op == GGML_OP_MUL_MAT && ggml_get_op_params_i32(dst, 1) == GGML_HINT_SRC0_IS_HADAMARD &&
        ggml_sycl_op_fwht(ctx, src1, dst)) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ତାର୍କିକ split = ggml_backend_buffer_is_sycl_split(src0->buffer);
    int64_t min_compute_capability = INT_MAX;

    ଯଦି (split) {
        ggml_backend_sycl_split_buffer_type_context * buft_ctx =
            (ggml_backend_sycl_split_buffer_type_context *) src0->buffer->buft->context;
        auto & tensor_split = buft_ctx->tensor_split;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0; id < ggml_sycl_info().device_ଗଣନା; ++id) {
            // skip devices that are not going to do any work:
            ଯଦି (tensor_split[id] >= (id + 1 < ggml_sycl_info().device_ଗଣନା ? tensor_split[id + 1] : 1.0f)) {
                ଜାରି;
            }

            ଯଦି (min_compute_capability > ggml_sycl_info().devices[id].cc) {
                min_compute_capability = ggml_sycl_info().devices[id].cc;
            }
        }
    } ଅନ୍ୟଥା {
        min_compute_capability = ggml_sycl_info().devices[ctx.device].cc;
    }

    // check data types and tensor shapes for custom matrix multiplication kernels:
    ତାର୍କିକ use_dequantize_mul_mat_vec = can_use_dequantize_mul_mat_vec(src0, src1, dst);
#ifdef GGML_SYCL_F16
    // dmmv may convert src1 to f16 in this build [TAG_GGML_PREC]
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && ggml_sycl_src1_f16_ok(dst);
#endif

    ତାର୍କିକ use_mul_mat_vec_q = can_use_mul_mat_vec_q(src0, src1, dst);

    ତାର୍କିକ use_mul_mat_q =  ggml_sycl_supports_mmq(src0->type)
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

    ଯଦି (!g_ggml_sycl_prioritize_dmmv && ((should_reorder_tensor(ctx, dst) &&
                                          ggml_sycl_supports_reorder_mmvq(src0->type)))) {
        ତାର୍କିକ use = g_ggml_sycl_enable_esimd && ggml_sycl_supports_reorder_esimd(src0->type);
        // Arc770 get benefit with Q4_0 by skipping MMVQ path
        ଯଦି (!(ggml_sycl_info().devices[ctx.device].hw_info.arch ==
                    gpu_arch::intel_gpu_acm_g10 &&
                src0->type == GGML_TYPE_Q4_0)) {
            use = use || !use_mul_mat_vec_q;
        }
        use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && use;
    }

    ଯଦି (!split && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        // TODO: Refactor and cleanup of mul mat dispatching.
        ଯଦି (src0->ne[3] == 1 && src1->ne[3] == 1) {
            // KQ single-batch
            // mmv p021 was specific for these dimensions
            ggml_sycl_mul_mat_vec_p021(ctx, src0, src1, dst);
        } ଅନ୍ୟଥା {
            // The kernel from the if path is faster for that specific case, but does not support all mul mats.
            ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
        }
    } ଅନ୍ୟଥା ଯଦି (!split && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        // KQV single-batch
        ggml_sycl_mul_mat_vec_nc(ctx, src0, src1, dst);
    } ଅନ୍ୟଥା ଯଦି (!split && src0->type == GGML_TYPE_F16 && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2] * src1->ne[3] > 1) {
        // KQ + KQV multi-batch
        ggml_sycl_mul_mat_batched_sycl(ctx, src0, src1, dst);
    } ଅନ୍ୟଥା ଯଦି (use_dequantize_mul_mat_vec) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::DMMV);
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_dequantize_mul_mat_vec);
    } ଅନ୍ୟଥା ଯଦି (use_mul_mat_vec_q) {
        opt_for_reorder(&ctx, src0, src1, dst, mul_mat_algo::MMVQ);
        ggml_tensor_extra_gpu * extra = static_cast<ggml_tensor_extra_gpu *>(src0->extra);
        ଯଦି (extra && extra->optimized_feature.reorder) {
            ggml_sycl_op_mul_mat<quantize_and_reorder_q8_1_soa>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        } ଅନ୍ୟଥା {
            ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_vec_q);
        }
    } ଅନ୍ୟଥା ଯଦି (use_mul_mat_q) {
        ggml_sycl_op_mul_mat<quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_q);
    } ଅନ୍ୟଥା {
        ggml_sycl_op_mul_mat<no_quantize_q8_1>(ctx, src0, src1, dst, ggml_sycl_op_mul_mat_sycl);
    }
}

// {mul_mat(gate), mul_mat(up), GLU} over the standard (non-reorder) weight layout,
// for quant pairs the reorder kernel does not cover (mixed gate/up types, e.g. UD-Q4_K_XL's
// iq4_xs gate + q5_K up). Two launches replace five: one shared q8_1 quantization and one
// dual-GEMV+GLU.
ସ୍ଥିର ତାର୍କିକ ggml_sycl_mul_mat_glu_mmvq_plain(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, ଅଚଳ ggml_tensor * wu,
                                             ଅଚଳ ggml_tensor * wg, ଅଚଳ ggml_tensor * act) {
    // weights already migrated to the reorder layout would be misread by the plain kernel
    ଅଚଳ auto * extra_u = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wu->extra);
    ଅଚଳ auto * extra_g = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wg->extra);
    ଯଦି ((extra_u && extra_u->optimized_feature.reorder) || (extra_g && extra_g->optimized_feature.reorder)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // the plain kernel only covers q5_K / iq4_xs pairs: bail before quantizing the activation
    ଅଚଳ auto plain_type = [](ggml_type t) { ଫେରନ୍ତୁ t == GGML_TYPE_Q5_K || t == GGML_TYPE_IQ4_XS; };
    ଯଦି (!plain_type(wu->type) || !plain_type(wg->type)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (plain layout)");

    ଅଚଳ int64_t ne00 = wu->ne[0];
    ଅଚଳ int64_t ne11 = act->ne[1];

    ଅଚଳ queue_ptr stream = ctx.stream();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା src1_padded_cols = GGML_PAD((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, MATRIX_ROW_PADDING);

    ggml_sycl_pool_alloc<ଅକ୍ଷର> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * ଆକାର(block_q8_1) / QK8_1);
    ଅକ୍ଷର * src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_q8_1>((ଅଚଳ ଦଶମିକ *) act->data, src1_ddq, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
                                          src1_padded_cols, stream);

    ଫେରନ୍ତୁ ggml_sycl_mul_mat_vec_q_glu_plain(wg->type, wu->type, ggml_get_glu_op(glu), wg->data, wu->data,
                                             src1_ddq, (ଦଶମିକ *) glu->data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wu->ne[1],
                                             (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
                                             /*stride_col_y=*/src1_padded_cols / QK8_1,
                                             /*stride_col_dst=*/(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) glu->ne[0], stream);
    GGML_UNUSED(gate);
}

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// {mul_mat(gate), mul_mat(up), GLU} on the reorder layout via the ESIMD DMMV kernel, for
// same-type K-quant or Q8_0 pairs and a single activation column. It reads the F32 activation
// directly, so one launch replaces three and no q8_1 quantization is needed.
ସ୍ଥିର ତାର୍କିକ ggml_sycl_mul_mat_glu_dmmv_esimd(ggml_backend_sycl_context & ctx, ggml_tensor * glu,
                                             ggml_tensor * gate, ggml_tensor * up, ଅଚଳ ggml_tensor * wu,
                                             ଅଚଳ ggml_tensor * wg, ଅଚଳ ggml_tensor * act) {
    // the kernel reads the activation while writing the output; glu may reuse the dead activation's memory
    ଅଚଳ ଅକ୍ଷର * act_begin = (ଅଚଳ ଅକ୍ଷର *) act->data;
    ଅଚଳ ଅକ୍ଷର * glu_begin = (ଅଚଳ ଅକ୍ଷର *) glu->data;
    ଯଦି (act_begin < glu_begin + ggml_nbytes(glu) && glu_begin < act_begin + ggml_nbytes(act)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // install the reorder (SoA) layout the kernel needs, as the unfused DMMV path would
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::DMMV);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::DMMV);

    ଅଚଳ auto * extra_u = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wu->extra);
    ଅଚଳ auto * extra_g = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wg->extra);
    ଯଦି (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU (ESIMD)");

    ଫେରନ୍ତୁ ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(wu->type, ggml_get_glu_op(glu), wu->data, wg->data,
                                                              (ଅଚଳ ଦଶମିକ *) act->data, (ଦଶମିକ *) glu->data,
                                                              (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wu->ne[0], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wu->ne[1], ctx.stream());
}
#endif

// Fused dense-FFN mat-vec for the {mul_mat(gate), mul_mat(up), GLU} subgraph at node_idx.
// Returns false if it declined, in which case the caller runs the three nodes normally.
ସ୍ଥିର ତାର୍କିକ ggml_sycl_mul_mat_glu_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) {
    ଯଦି (!ggml_sycl_can_fuse(cgraph, ଗଣ୍ଠି_idx, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {})) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ggml_tensor *       glu  = cgraph->nodes[ଗଣ୍ଠି_idx + 2];
    ggml_tensor *       gate = glu->src[0];
    ggml_tensor *       up   = glu->src[1];
    ଅଚଳ ggml_tensor * wu   = up->src[0];
    ଅଚଳ ggml_tensor * wg   = gate->src[0];
    ଅଚଳ ggml_tensor * act  = up->src[1];

    // this writes glu->data directly rather than the per-device row slices that
    // ggml_sycl_op_mul_mat() stitches back together, so it cannot serve split weights
    ଯଦି (ggml_backend_buffer_is_sycl_split(wu->buffer) || ggml_backend_buffer_is_sycl_split(wg->buffer)) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    // single-column same-type K-quant or Q8_0 pairs take the ESIMD kernel, as the unfused mul-mats
    // would via DMMV; if the reorder layout could not be installed, fall through to mmvq
    ଯଦି (g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
        ggml_sycl_supports_reorder_esimd(wu->type) &&
        ggml_sycl_mul_mat_glu_dmmv_esimd(ctx, glu, gate, up, wu, wg, act)) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }
#endif

    // with DMMV prioritised the unfused path would not have gone through mmvq at all
    ଯଦି (g_ggml_sycl_prioritize_dmmv) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // quant pairs the reorder kernel does not serve (mixed gate/up types, q5_K off BMG) take the
    // standard-layout fused path instead; same-type q4_K / q5_K keep the reorder path below
    ଅଚଳ ତାର୍କିକ reorder_pair = wg->type == wu->type &&
        (wu->type == GGML_TYPE_Q4_K || (wu->type == GGML_TYPE_Q5_K && ggml_sycl_q5_k_mmvq_reuse(ctx.device)));
    ଯଦି (!reorder_pair) {
        ଫେରନ୍ତୁ ggml_sycl_mul_mat_glu_mmvq_plain(ctx, glu, gate, up, wu, wg, act);
    }

    // past 5 columns the two unfused q5_K GEMVs are faster than the fused kernel
    ଯଦି (wu->type == GGML_TYPE_Q5_K && act->ne[1] > 5) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // install the reorder (SoA) layout the fused kernel needs, as the unfused mmvq path would;
    // a no-op once done. after the bail checks so a declined op does not pay for it.
    opt_for_reorder(&ctx, wu, act, up, mul_mat_algo::MMVQ);
    opt_for_reorder(&ctx, wg, act, gate, mul_mat_algo::MMVQ);

    ଅଚଳ auto * extra_u = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wu->extra);
    ଅଚଳ auto * extra_g = static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(wg->extra);
    ଯଦି (!extra_u || !extra_g || !extra_u->optimized_feature.reorder || !extra_g->optimized_feature.reorder) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    // log the up mat-mul: glu's own srcs are the two intermediates the fusion never materialises
    scope_op_debug_print scope_dbg_print(__func__, up, /*num_src=*/2, " : fused with gate + GLU");

    ଅଚଳ int64_t ne00 = wu->ne[0];
    ଅଚଳ int64_t ne11 = act->ne[1];

    ଅଚଳ queue_ptr stream           = ctx.stream();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       src1_padded_cols = GGML_PAD((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, MATRIX_ROW_PADDING);

    // one activation, quantized once and fully consumed into src1_ddq before the GEMV on this
    // in-order queue, so glu->data aliasing the dead activation needs no memory-range check
    ggml_sycl_pool_alloc<ଅକ୍ଷର> src1_q8_alloc(ctx.pool(),
                                             (size_t) ne11 * src1_padded_cols * ଆକାର(block_q8_1) / QK8_1);
    ଅକ୍ଷର *                     src1_ddq = src1_q8_alloc.get();

    quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>((ଅଚଳ ଦଶମିକ *) act->data, src1_ddq, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
                                                          src1_padded_cols, stream);

    ଫେରନ୍ତୁ ggml_sycl_mul_mat_vec_q_glu_reorder(wu->type, ggml_get_glu_op(glu), wu->data, wg->data, src1_ddq,
                                               (ଦଶମିକ *) glu->data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne00, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) wu->ne[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
                                               /*stride_col_y_bytes=*/src1_padded_cols * (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ଆକାର(block_q8_1) /
                                                   QK8_1,
                                               /*stride_col_dst=*/(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) glu->ne[0], stream);
}

// Batch the run of consecutive L2_NORM siblings starting at node_idx into one launch.
// Returns the number of extra graph nodes consumed, or 0 if the run is shorter than two
// (the caller then runs the norm through the per-tensor kernel).
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_l2_norm_batch_fused(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx) {
    ଅଚଳ ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[ଗଣ୍ଠି_idx];
    ଯଦି (ggml_sycl_info().device_ଗଣନା != 1 || ଗଣ୍ଠି->type != GGML_TYPE_F32 ||
        ଗଣ୍ଠି->src[0]->type != GGML_TYPE_F32 || ଗଣ୍ଠି->src[0]->ne[0] >= 1024) {
        ଫେରନ୍ତୁ 0;
    }

    ggml_tensor * batch[GGML_SYCL_L2_BATCH_MAX];
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           count = 0;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା           last  = ଗଣ୍ଠି_idx;
    ଦଶମିକ         eps0;
    memcpy(&eps0, ଗଣ୍ଠି->op_params, ଆକାର(ଦଶମିକ));

    // Conservative aliasing test: the batched norms run concurrently in one kernel,
    // so none may read what another writes, and none may write where another writes.
    auto overlaps = [](ଅଚଳ ggml_tensor * a, ଅଚଳ ggml_tensor * b) {
        ଅଚଳ ଅକ୍ଷର * ab = (ଅଚଳ ଅକ୍ଷର *) a->data;
        ଅଚଳ ଅକ୍ଷର * bb = (ଅଚଳ ଅକ୍ଷର *) b->data;
        ଫେରନ୍ତୁ ab < bb + ggml_nbytes(b) && bb < ab + ggml_nbytes(a);
    };

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ଗଣ୍ଠି_idx; j < cgraph->n_nodes && count < GGML_SYCL_L2_BATCH_MAX; ++j) {
        ggml_tensor * nj = cgraph->nodes[j];
        ଯଦି (ggml_is_empty(nj) || nj->op == GGML_OP_RESHAPE || nj->op == GGML_OP_TRANSPOSE ||
            nj->op == GGML_OP_VIEW || nj->op == GGML_OP_PERMUTE || nj->op == GGML_OP_NONE ||
            (nj->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;  // not a launch; cannot break a run of adjacent norms
        }
        ଯଦି (nj->op != GGML_OP_L2_NORM || nj->type != GGML_TYPE_F32 ||
            nj->src[0]->type != GGML_TYPE_F32 || !ggml_are_same_shape(nj, ଗଣ୍ଠି) ||
            !ggml_are_same_shape(nj->src[0], ଗଣ୍ଠି->src[0])) {
            ଭାଙ୍ଗ;  // any other launch ends the run
        }
        ତାର୍କିକ same_nb = ସତ୍ୟ;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 0; d < GGML_MAX_DIMS; ++d) {
            ଯଦି (nj->nb[d] != ଗଣ୍ଠି->nb[d] || nj->src[0]->nb[d] != ଗଣ୍ଠି->src[0]->nb[d]) {
                same_nb = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (!same_nb) {
            ଭାଙ୍ଗ;  // one nb[] stride set is shared by the whole batch
        }
        ଦଶମିକ epsj;
        memcpy(&epsj, nj->op_params, ଆକାର(ଦଶମିକ));
        ଯଦି (epsj != eps0) {
            ଭାଙ୍ଗ;  // eps mismatch ends the run
        }
        ତାର୍କିକ indep = ସତ୍ୟ;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < count; ++k) {
            ଯଦି (overlaps(nj->src[0], batch[k]) || overlaps(nj, batch[k])) {
                indep = ମିଥ୍ୟା;
                ଭାଙ୍ଗ;
            }
        }
        ଯଦି (!indep) {
            ଭାଙ୍ଗ;  // an overlapping tensor would race inside one launch
        }
        batch[count++] = nj;
        last           = j;
    }
    ଯଦି (count < 2) {
        ଫେରନ୍ତୁ 0;  // a lone norm falls through to the per-tensor kernel
    }
    ggml_sycl_l2_norm_batch(ctx, batch, count);
    ଫେରନ୍ତୁ last - ଗଣ୍ଠି_idx;
}


__dpct_inline__ ସ୍ଥିର ଶୂନ୍ୟ k_copy_src1_to_contiguous(
    ଅଚଳ ଅକ୍ଷର *__restrict__ src1_original, ଅକ୍ଷର *__restrict__ src1_contiguous,
    ଅଚଳ mmid_row_mapping *__restrict__ row_mapping,
    int64_t ne11, int64_t ne10, size_t nb11, size_t nb12,
    ଅଚଳ sycl::nd_item<3> &item_ct1) {
    ଅଚଳ int32_t src1_row = item_ct1.get_group(2);

    ଅଚଳ int32_t iid1 = row_mapping[src1_row].i2;
    ଅଚଳ int32_t id   = row_mapping[src1_row].i1;

    ଅଚଳ int64_t i11 = id % ne11;
    ଅଚଳ int64_t i12 = iid1;

    ଅଚଳ ଦଶମିକ * src1_row_original = (ଅଚଳ ଦଶମିକ *)(src1_original + i11*nb11 + i12*nb12);
    ଦଶମିକ * src1_row_contiguous = (ଦଶମିକ *)(src1_contiguous + src1_row*nb11);

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = item_ct1.get_local_id(2); i < ne10;
         i += item_ct1.get_local_range(2)) {
        src1_row_contiguous[i] = src1_row_original[i];
    }
}

__dpct_inline__ ସ୍ଥିର ଶୂନ୍ୟ k_copy_dst_from_contiguous(
    ଅକ୍ଷର *__restrict__ dst_original, ଅଚଳ ଅକ୍ଷର *__restrict__ dst_contiguous,
    ଅଚଳ mmid_row_mapping *__restrict__ row_mapping, int64_t ne0, size_t nb1,
    size_t nb2, ଅଚଳ sycl::nd_item<3> &item_ct1) {
    int32_t i = item_ct1.get_group(2);

    ଅଚଳ int32_t i1 = row_mapping[i].i1;
    ଅଚଳ int32_t i2 = row_mapping[i].i2;

    ଅଚଳ ଦଶମିକ * dst_row_contiguous = (ଅଚଳ ଦଶମିକ *)(dst_contiguous + i*nb1);
    ଦଶମିକ * dst_row_original = (ଦଶମିକ *)(dst_original + i1*nb1 + i2*nb2);

#pragma unroll
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = item_ct1.get_local_id(2); j < ne0;
         j += item_ct1.get_local_range(2)) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

// Fused MoE TG fast path. Returns false to fall back to the per-expert loop below.
ସ୍ଥିର ତାର୍କିକ ggml_sycl_mul_mat_id_mmvq_fused(
    ggml_backend_sycl_context & ctx, ଅଚଳ ggml_tensor * src0,
    ଅଚଳ ggml_tensor * src1, ଅଚଳ ggml_tensor * ids, ggml_tensor * dst)
{
    ଅଚଳ int64_t ne10 = src1->ne[0];
    ଅଚଳ int64_t ne11 = src1->ne[1];
    ଅଚଳ int64_t ne12 = src1->ne[2];
    ଯଦି (ne12 != 1) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (ne10 != src0->ne[0] || ne10 % QK8_1 != 0) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (!ggml_is_contiguous(src1)) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଅଚଳ int64_t n_ids_per_group = ids->ne[0];
    ଯଦି (ids->ne[1] != 1) ଫେରନ୍ତୁ ମିଥ୍ୟା;
    ଯଦି (ne11 != 1 && ne11 != n_ids_per_group) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଅଚଳ queue_ptr stream           = ctx.stream();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       src1_padded_cols = GGML_PAD((ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne10, MATRIX_ROW_PADDING);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       n_experts_used   = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) n_ids_per_group;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       nrows            = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) src0->ne[1];

    // Lazily reorder supported expert weights into a per-expert SoA layout, then run the reorder
    // GEMV. Placed after the bail checks so a non-dispatchable op does not pay the reorder cost.
    opt_for_reorder_id(&ctx, src0);
    ଅଚଳ ggml_tensor_extra_gpu * src0_extra =
        static_cast<ଅଚଳ ggml_tensor_extra_gpu *>(src0->extra);
    ଅଚଳ ତାର୍କିକ use_reorder = src0_extra && src0_extra->optimized_feature.reorder;

    ggml_sycl_pool_alloc<ଅକ୍ଷର> src1_q8_alloc(ctx.pool(),
        (size_t) ne11 * src1_padded_cols * ଆକାର(block_q8_1) / QK8_1);
    ଅକ୍ଷର * src1_ddq = src1_q8_alloc.get();
    ଯଦି (use_reorder) {
        quantize_row_q8_1_sycl<quantize_and_reorder_q8_1_soa>(
            (ଅଚଳ ଦଶମିକ *) src1->data, src1_ddq, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne10, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
            src1_padded_cols, stream);
    } ଅନ୍ୟଥା {
        quantize_row_q8_1_sycl<quantize_q8_1>(
            (ଅଚଳ ଦଶମିକ *) src1->data, src1_ddq, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne10, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne11,
            src1_padded_cols, stream);
    }

    ଅଚଳ size_t bytes_per_qrow = (size_t) src1_padded_cols * ଆକାର(block_q8_1) / QK8_1;
    ଅଚଳ size_t src1_row_stride = (ne11 == 1) ? 0 : bytes_per_qrow;

    ଯଦି (use_reorder) {
        ଫେରନ୍ତୁ ggml_sycl_mul_mat_vec_q_id_reorder(
            src0->type, src0->data, src1_ddq, (ଅଚଳ int32_t *) ids->data,
            (ଦଶମିକ *) dst->data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne10, nrows, n_experts_used,
            /*expert_weight_stride=*/ src0->nb[2],
            /*dst_row_stride=*/ dst->nb[1],
            src1_row_stride, stream);
    }
    ଫେରନ୍ତୁ ggml_sycl_mul_mat_vec_q_id(
        src0->type, src0->data, src1_ddq, (ଅଚଳ int32_t *) ids->data,
        (ଦଶମିକ *) dst->data, (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ne10, nrows, n_experts_used,
        /*expert_weight_stride=*/ src0->nb[2],
        /*dst_row_stride=*/ dst->nb[1],
        src1_row_stride, stream);
}

// counting sort of the routed rows by expert id (row_id_i, as chosen by the router):
// builds a projection of a memory layout where each expert's slice is contiguous
ସ୍ଥିର ଶୂନ୍ୟ mmid_counting_sort_rows(
        ଅଚଳ ggml_tensor * ids, ଅଚଳ ଅକ୍ଷର * ids_host,
        int64_t n_ids, int64_t n_as, int64_t n_routed_rows,
        std::vector<int64_t> & expert_counts,
        std::vector<int64_t> & expert_row_offsets,
        std::vector<mmid_row_mapping> & routed_row_src) {

    // frequencies: how many routed rows each expert "owns"
    expert_counts.assign(n_as, 0);
    ପାଇଁ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ପାଇଁ (int64_t id = 0; id < n_ids; id++) {
            ଅଚଳ int32_t row_id_i = *(ଅଚଳ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);
            expert_counts[row_id_i]++;
        }
    }

    // where each expert's slice starts (row indices) and the previous ends
    expert_row_offsets.assign(n_as + 1, 0);
    ପାଇଁ (int64_t i02 = 0; i02 < n_as; i02++) {
        expert_row_offsets[i02 + 1] = expert_row_offsets[i02] + expert_counts[i02];
    }

    std::vector<int64_t> expert_row_next = expert_row_offsets;
    routed_row_src.resize(n_routed_rows);
    ପାଇଁ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        ପାଇଁ (int64_t id = 0; id < n_ids; id++) {
            ଅଚଳ int32_t row_id_i = *(ଅଚଳ int32_t *) (ids_host + iid1*ids->nb[1] + id*ids->nb[0]);
            GGML_ASSERT(row_id_i >= 0 && row_id_i < n_as);

            // find and validate the next free row for a given expert (row_id_i)
            ଅଚଳ int64_t routed_row = expert_row_next[row_id_i]++;
            GGML_ASSERT(routed_row >= expert_row_offsets[row_id_i]);
            GGML_ASSERT(routed_row < expert_row_offsets[row_id_i + 1]);
            routed_row_src[routed_row] = {(int32_t) id, (int32_t) iid1};
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mul_mat_id(ggml_backend_sycl_context & ctx,
                                 ggml_tensor *dst) ଚେଷ୍ଟା {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);
    ଅଚଳ ggml_tensor *src0 = dst->src[0];
    ଅଚଳ ggml_tensor *src1 = dst->src[1];
    GGML_ASSERT(!ggml_backend_buffer_is_sycl_split(src0->buffer) && "mul_mat_id does not support split buffers");

    ଅଚଳ ggml_tensor *ids = dst->src[2];
    GGML_TENSOR_ଦ୍ୱିମିକ_OP_LOCALS

    ଅଚଳ queue_ptr stream = ctx.stream();

    ଅଚଳ int64_t n_as = ne02;
    ଅଚଳ int64_t n_ids = ids->ne[0];

    ଯଦି (ne12 == 1) {
        ଯଦି (ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, src1, ids, dst)) {
            ଫେରନ୍ତୁ;
        }
    }

    // The per-expert loop below reads the experts in whatever layout they have: reorder MXFP4 here as well, so prompt processing does not depend on a single-token decode having run first.
    ଯଦି (src0->type == GGML_TYPE_MXFP4) {
        opt_for_reorder_id(&ctx, src0);
    }

    std::vector<ଅକ୍ଷର> ids_host(ggml_nbytes(ids));
    ଅଚଳ ଅକ୍ଷର * ids_dev = (ଅଚଳ ଅକ୍ଷର *) ids->data;

    SYCL_CHECK(CHECK_TRY_ERROR(
        stream->memcpy(ids_host.data(), ids_dev, ggml_nbytes(ids))));

    // also ensures ctx.mmid_row_mapping_host and ctx.mmid_tile_schedule_host are drained before we
    // refill them: the grouped GEMM enqueues an async copy out of the tile schedule, so removing
    // this wait would let the next node overwrite a buffer the device is still reading
    SYCL_CHECK(CHECK_TRY_ERROR(stream->wait()));

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row = *dst;

    ଅକ୍ଷର *src0_original = (ଅକ୍ଷର *)src0->data;
    ଅକ୍ଷର *src1_original = (ଅକ୍ଷର *)src1->data;
    ଅକ୍ଷର *dst_original = (ଅକ୍ଷର *)dst->data;

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
    ଯଦି (ne12 == 1) {
        ପାଇଁ (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
            ପାଇଁ (int64_t id = 0; id < n_ids; id++) {
                ଅଚଳ int32_t i02 = *(ଅଚଳ int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
                GGML_ASSERT(i02 >= 0 && i02 < n_as);

                ଅଚଳ int64_t i11 = id % ne11;
                ଅଚଳ int64_t i12 = iid1;

                ଅଚଳ int64_t i1 = id;
                ଅଚଳ int64_t i2 = i12;

            src0_row.data = src0_original + i02*nb02;
            src1_row.data = src1_original + i11*nb11 + i12*nb12;
            dst_row.data = dst_original + i1*nb1 + i2*nb2;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
            }
        }
    } ଅନ୍ୟଥା {
        ଅଚଳ int64_t n_routed_rows = ids->ne[1] * n_ids;
        ggml_sycl_pool_alloc<ଅକ୍ଷର> src1_contiguous(ctx.pool(), ଆକାର(ଦଶମିକ)*n_routed_rows*ne10);
        ggml_sycl_pool_alloc<ଅକ୍ଷର>  dst_contiguous(ctx.pool(), ଆକାର(ଦଶମିକ)*n_routed_rows*ne0);

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
                stream->memcpy(dev_row_mapping.get(), routed_row_src.data(), n_routed_rows*ଆକାର(mmid_row_mapping))));

        ଅଚଳ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_group_size = ggml_sycl_info().max_work_group_sizes[ctx.device];
        ପ୍ରତିଜ୍ଞା(max_work_group_size % (WARP_SIZE * WARP_SIZE) == 0);

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne10, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                ଅକ୍ଷର *__restrict src1_contiguous_get =
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

        ତାର୍କିକ grouped = ମିଥ୍ୟା;
        ଯଦି (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
            dst->type == GGML_TYPE_F32 && nb11 == ଆକାର(ଦଶମିକ)*ne10 && nb1 == ଆକାର(ଦଶମିକ)*ne0) {
            // the grouped GEMM reads the reorder (SoA) layout faster, and the first decode step installs it
            // anyway: install it here already, so prefill does not depend on whether a decode ran before
            ଯଦି (g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
                ggml_sycl_xmx_gather_type_enabled(src0->type)) {
                opt_for_reorder_id(&ctx, src0);
            }
            ଅଚଳ ତାର୍କିକ src0_reordered =
                src0->extra && ((ଅଚଳ ggml_tensor_extra_gpu *) src0->extra)->optimized_feature.reorder;
            grouped = ggml_sycl_grouped_dequant_gemm(src0->type, src0_reordered, src0_original, nb02,
                                                     (ଅଚଳ ଦଶମିକ *) src1_contiguous.get(), ggml_sycl_src1_prec(dst),
                                                     (ଦଶମିକ *) dst_contiguous.get(),
                                                     expert_row_offsets.data(), n_as, ne01, ne10, n_routed_rows,
                                                     ctx.mmid_tile_schedule_host, ctx.pool(), stream);
        }

        ପାଇଁ (int64_t i02 = 0; i02 < n_as && !grouped; i02++) {
            ଅଚଳ int64_t ଅଙ୍କ_src1_rows = expert_row_counts[i02];

            ଯଦି (ଅଙ୍କ_src1_rows == 0) {
                ଜାରି;
            }

            ଅଚଳ int64_t expert_row_offset = expert_row_offsets[i02];

            src0_row.data = src0_original + i02*nb02;

            GGML_ASSERT(nb11 == ଆକାର(ଦଶମିକ)*ne10);
            GGML_ASSERT(nb1 == ଆକାର(ଦଶମିକ)*ne0);
            src1_row.data = src1_contiguous.get() + expert_row_offset*nb11;
            src1_row.ne[1] = ଅଙ୍କ_src1_rows;

            src1_row.nb[1] = nb11;
            src1_row.nb[2] = ଅଙ୍କ_src1_rows*nb11;
            src1_row.nb[3] = ଅଙ୍କ_src1_rows*nb11;

            dst_row.data = dst_contiguous.get() + expert_row_offset*nb1;
            dst_row.ne[1] = ଅଙ୍କ_src1_rows;
            dst_row.nb[1] = nb1;
            dst_row.nb[2] = ଅଙ୍କ_src1_rows*nb1;
            dst_row.nb[3] = ଅଙ୍କ_src1_rows*nb1;

            ggml_sycl_mul_mat(ctx, &src0_row, &src1_row, &dst_row);
        }

        {
            sycl::range<3> block_dims(1, 1, std::min((unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା)ne0, max_work_group_size));
            sycl::range<3> grid_dims(1, 1, n_routed_rows);
            stream->submit([&](sycl::handler &cgh) {
                ଅଚଳ ଅକ୍ଷର *__restrict dst_contiguous_get =
                    dst_contiguous.get();
                ଅଚଳ mmid_row_mapping *__restrict dev_row_mapping_get =
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
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_scale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_scale(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_diag_mask_inf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag_mask_inf(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_pool2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool2d(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_pool1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pool1d(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_im2col_3d(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_col2im_1d(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_conv_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_conv_3d(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_sum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_sum_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_sum_rows(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_mean(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_mean(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_argsort(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argsort(ctx, dst);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_argmax(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    GGML_ASSERT(ggml_is_contiguous(dst->src[0]));
    ggml_sycl_op_argmax(ctx, dst);
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_set_main_device(ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା main_device) ଚେଷ୍ଟା {
    ଯଦି (dpct::get_current_device_id() == static_cast<unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା> (main_device)) {
        ଫେରନ୍ତୁ;
    }
    check_allow_gpu_index(main_device);
    dpct::select_device(main_device);

    ଯଦି (g_ggml_sycl_debug) {
        dpct::device_info prop;
        SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
            prop, dpct::dev_mgr::instance().get_device(main_device))));
        GGML_LOG_INFO("Using device %d (%s) as main device\n",
                main_device, prop.get_name());
    }
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_compute_forward(ggml_backend_sycl_context & ctx, ସଂରଚନା ggml_tensor * dst) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] ggml_sycl_compute_forward: dst=%s, op=%s\n", dst->name, ggml_op_name(dst->op));
    ଯଦି (!g_sycl_loaded) ଫେରନ୍ତୁ ମିଥ୍ୟା;

    ଯଦି (dst->src[0] != nullptr && ggml_backend_buffer_is_sycl_split(dst->src[0]->buffer)) {
        ggml_sycl_set_peer_access(dst->src[1]->ne[1], ctx.device);
    }

    ବାଛ (dst->op) {
        ସ୍ଥିତି GGML_OP_ARGMAX:
            ggml_sycl_argmax(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_2D:
            ggml_sycl_op_conv2d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_2D_DW:
            ggml_sycl_op_conv2d_dw(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_3D:
            ggml_sycl_conv_3d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
            ggml_sycl_op_conv_transpose_1d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
            ggml_sycl_op_conv2d_transpose(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT:
            ggml_sycl_repeat(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_REPEAT_BACK:
            ggml_sycl_repeat_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            ggml_sycl_get_rows(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GET_ROWS_BACK:
            ggml_sycl_get_rows_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET:
            ggml_sycl_op_set(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SET_ROWS:
            ggml_sycl_op_set_rows(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_PRE:
            ggml_sycl_op_dsv4_hc_pre(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_COMB:
            ggml_sycl_op_dsv4_hc_comb(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DSV4_HC_POST:
            ggml_sycl_op_dsv4_hc_post(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER:
            ggml_sycl_op_lightning_indexer(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DUP:
            ggml_sycl_dup(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_sycl_add(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ADD_ID:
            ggml_sycl_add_id(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUB:
            ggml_sycl_sub(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL:
            ggml_sycl_ଗଣନା_equal(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ACC:
            ggml_sycl_acc(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL:
            ggml_sycl_mul(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LOG:
            ggml_sycl_log(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIV:
            ggml_sycl_div(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UNARY:
            ବାଛ (ggml_get_unary_op(dst)) {
                ସ୍ଥିତି GGML_UNARY_OP_NEG:
                    ggml_sycl_neg(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_STEP:
                    ggml_sycl_step(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_GELU:
                    ggml_sycl_gelu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SILU:
                    ggml_sycl_silu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK:
                    ggml_sycl_gelu_quick(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:
                    ggml_sycl_gelu_erf(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_TANH:
                    ggml_sycl_tanh(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_RELU:
                    ggml_sycl_relu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:
                    ggml_sycl_sigmoid(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_HARDSIGMOID:
                    ggml_sycl_hardsigmoid(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_HARDSWISH:
                    ggml_sycl_hardswish(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_EXP:
                    ggml_sycl_exp(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_EXPM1:
                    ggml_sycl_expm1(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:
                    ggml_sycl_softplus(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_SGN:
                    ggml_sycl_sgn(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_ABS:
                    ggml_sycl_abs(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_ELU:
                    ggml_sycl_elu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_XIELU:
                    ggml_sycl_xielu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_FLOOR:
                    ggml_sycl_floor(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_CEIL:
                    ggml_sycl_ceil(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_ROUND:
                    ggml_sycl_round(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_UNARY_OP_TRUNC:
                    ggml_sycl_trunc(ctx, dst);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GLU:
            ବାଛ (ggml_get_glu_op(dst)) {
                ସ୍ଥିତି GGML_GLU_OP_REGLU:
                    ggml_sycl_reglu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                    ggml_sycl_geglu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                    ggml_sycl_swiglu(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                    ggml_sycl_swiglu_oai(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_sycl_swiglu_clamp(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                    ggml_sycl_geglu_erf(ctx, dst);
                    ଭାଙ୍ଗ;
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                    ggml_sycl_geglu_quick(ctx, dst);
                    ଭାଙ୍ଗ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NORM:
            ggml_sycl_norm(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GROUP_NORM:
            ggml_sycl_group_norm(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONCAT:
            ggml_sycl_op_concat(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PAD_REFLECT_1D:
            ggml_sycl_op_pad_reflect_1d(ctx,dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_UPSCALE:
            ggml_sycl_upscale(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_PAD:
            ggml_sycl_pad(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:
            ggml_sycl_leaky_relu(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
            ggml_sycl_rms_norm_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RMS_NORM:
            ggml_sycl_rms_norm(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_L2_NORM:
            ggml_sycl_l2_norm(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ଯଦି (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            /* ggml_sycl_mul_mat_id is dependent on ggml_sycl_mul_mat */
            ggml_sycl_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            ଯଦି (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ggml_sycl_mul_mat_id(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OUT_PROD:
            ggml_sycl_op_out_prod(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SCALE:
            ggml_sycl_scale(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQR:
            ggml_sycl_sqr(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SQRT:
            ggml_sycl_sqrt(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SIN:
            ggml_sycl_sin(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_COS:
            ggml_sycl_cos(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CLAMP:
            ggml_sycl_clamp(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CPY:
            ggml_sycl_cpy(ctx, dst->src[0], dst->src[1]);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CONT:
            ggml_sycl_dup(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
            GGML_SYCL_DEBUG("%s: Tensor NO-OP\n", __func__);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TRI:
            ggml_sycl_op_tri(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
            ggml_sycl_diag_mask_inf(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            ggml_sycl_op_soft_max(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK:
            ggml_sycl_op_soft_max_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_sycl_cross_entropy_loss(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_sycl_cross_entropy_loss_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE:
            ggml_sycl_rope(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROPE_BACK:
            ggml_sycl_rope_back(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL:
            ggml_sycl_im2col(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_IM2COL_3D:
            ggml_sycl_im2col_3d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_COL2IM_1D:
            ggml_sycl_col2im_1d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_2D:
            ggml_sycl_pool2d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_POOL_1D:
            ggml_sycl_pool1d(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM:
            ggml_sycl_sum(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
            ggml_sycl_sum_rows(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_MEAN:
            ggml_sycl_mean(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARGSORT:
            ggml_sycl_argsort(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K:
            ggml_sycl_op_ଶୀର୍ଷ_k(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_TIMESTEP_EMBEDDING:
            ggml_sycl_op_timestep_embedding(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RWKV_WKV6:
            ggml_sycl_op_rwkv_wkv6(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_RWKV_WKV7:
            ggml_sycl_op_rwkv_wkv7(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GATED_LINEAR_ATTN:
            ggml_sycl_op_gated_linear_attn(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
            ggml_sycl_gated_delta_net(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OPT_STEP_ADAMW:
            ggml_sycl_opt_step_adamw(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_OPT_STEP_SGD:
            ggml_sycl_opt_step_sgd(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SSM_CONV:
            ggml_sycl_ssm_conv(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SSM_SCAN:
            ggml_sycl_ssm_scan(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FILL:
            ggml_sycl_fill(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_CUMSUM:
            ggml_sycl_cumsum(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_DIAG:
            ggml_sycl_diag(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
            ggml_sycl_solve_tri(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ROLL:
            ggml_sycl_roll(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_ARANGE:
            ggml_sycl_arange(ctx, dst);
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            ggml_sycl_flash_attn_ext(ctx, dst);
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଫେରନ୍ତୁ ସତ୍ୟ;
} ଧର (sycl::exception & e) {
    std::cerr << e.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::cerr << "Error OP "<<ggml_op_name(dst->op)<< std::endl;
    std::exit(1);
}

GGML_API ଶୂନ୍ୟ ggml_backend_sycl_get_device_description(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, ଅକ୍ଷର *description,
                                      size_t description_size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_description\n");
    dpct::device_info prop;
    SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
        prop, dpct::dev_mgr::instance().get_device(device))));
    snprintf(description, description_size, "%s", prop.get_name());
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ତାର୍କିକ sycl_get_mem_info(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t * ମୁକ୍ତ, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] [%s] g_ggml_sycl_get_mem_api=%d\n",
        __func__, g_ggml_sycl_get_mem_api);

    MemoryAPIType mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
    mem_api_type = get_zes_init_res() == ZE_RESULT_SUCCESS ?
        (MemoryAPIType) g_ggml_sycl_get_mem_api : MemoryAPIType::MEMORY_API_TYPE_SYCL;
#ଅନ୍ୟଥା
    mem_api_type = MemoryAPIType::MEMORY_API_TYPE_SYCL;
#endif
    ତାର୍କିକ res = get_memory_size(dpct::dev_mgr::instance().get_device(device),
        *ମୁକ୍ତ, *total, mem_api_type);
    GGML_SYCL_DEBUG("[SYCL] [%s] total = %zu free = %zu\n", __func__, *total, *ମୁକ୍ତ);
    ଫେରନ୍ତୁ res;
}

ଶୂନ୍ୟ ggml_backend_sycl_get_device_memory(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device, size_t * ମୁକ୍ତ, size_t * total) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_get_device_memory\n");
    ଯଦି (!sycl_get_mem_info(device, ମୁକ୍ତ, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    ggml_sycl_memtrace_report_device("device memory query", device, *ମୁକ୍ତ, *total);
} ଧର (ଅଚଳ sycl::exception & exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__ << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

////////////////////////////////////////////////////////////////////////////////

// backend

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_get_name(ggml_backend_t backend) {

    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    ଫେରନ୍ତୁ sycl_ctx->name.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_free(ggml_backend_t backend) {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;

    delete sycl_ctx;
    delete backend;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_set_tensor_async(ggml_backend_t backend,
                                               ggml_tensor *tensor,
                                               ଅଚଳ ଶୂନ୍ୟ *data, size_t offset,
                                               size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    ଅଚଳ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR(
        (stream)->memcpy((ଅକ୍ଷର *)tensor->data + offset, data, size)));
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_get_tensor_async(ggml_backend_t backend,
                                               ଅଚଳ ggml_tensor *tensor,
                                               ଶୂନ୍ୟ *data, size_t offset,
                                               size_t size) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": tensor", tensor).c_str());
    GGML_SYCL_DEBUG(" size=%zu offset=%zu\n", size, offset);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) && "unsupported buffer type");
    ଅଚଳ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
        data, (ଅଚଳ ଅକ୍ଷର *)tensor->data + offset, size)));
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_cpy_tensor_async(ggml_backend_t backend,
                                               ଅଚଳ ggml_tensor *src,
                                               ggml_tensor *dst) ଚେଷ୍ଟା {
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ତାର୍କିକ is_cpy_supported                = dst->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device) &&
                            ggml_backend_buffer_is_sycl(src->buffer);
    GGML_SYCL_DEBUG("[SYCL] call %s", __func__);
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(": dst", dst).c_str());
    GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" src", src).c_str());
    GGML_SYCL_DEBUG(" is_cpy_supported=%d\n", is_cpy_supported);
    ଯଦି (is_cpy_supported) {
        /*
        DPCT1009:215: SYCL uses exceptions to report errors and does not use the
        error codes. The original code was commented out and a warning string
        was inserted. You need to rewrite this code.
        */
        ଅଚଳ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
        SYCL_CHECK(CHECK_TRY_ERROR((stream)->memcpy(
            dst->data, src->data, ggml_nbytes(dst))));
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_synchronize(ggml_backend_t backend) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_context * sycl_ctx = (ggml_backend_sycl_context *)backend->context;
    ଅଚଳ queue_ptr stream = sycl_ctx->stream(sycl_ctx->device, 0);
    SYCL_CHECK(CHECK_TRY_ERROR((stream)->wait()));

    GGML_UNUSED(backend);
}
ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ତାର୍କିକ ggml_sycl_is_view_or_noop(ଅଚଳ ggml_tensor * t) {
    ଫେରନ୍ତୁ ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
// returns the number of following nodes to skip (0 = no fusion)
// ported from ggml_cuda_try_gdn_cache_fusion - pure graph inspection, backend-agnostic
ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_sycl_try_gdn_cache_fusion(ଅଚଳ ggml_cgraph * cgraph, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଗଣ୍ଠି_idx,
                                          ggml_sycl_gated_delta_net_fused_cache & fused_state_cpy) {
    ଯଦି (!g_ggml_sycl_enable_fusion) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ggml_tensor * gdn = cgraph->nodes[ଗଣ୍ଠି_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output, and the cpy
    // found below is taken to be its only reader, as it is in every graph that builds this op
    ଯଦି (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ggml_tensor * src_v     = gdn->src[2];
    ଅଚଳ int64_t       S_v       = src_v->ne[0];
    ଅଚଳ int64_t       H         = src_v->ne[1];
    ଅଚଳ int64_t       n_tokens  = src_v->ne[2];
    ଅଚଳ int64_t       n_seqs    = src_v->ne[3];
    ଅଚଳ int64_t       D         = S_v * S_v * H;
    ଅଚଳ int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    ଅଚଳ int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    ଅଚଳ size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy must be the first node the compute loop below runs, so nothing can read the cache first.
    // skip exactly what that loop skips: views, no-ops, and nodes the graph does not compute.
    ଅଚଳ ggml_tensor * cpy  = nullptr;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା                 skip = 0;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ଗଣ୍ଠି_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        ଅଚଳ ggml_tensor * n = cgraph->nodes[j];
        ଯଦି (ggml_sycl_is_view_or_noop(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }
        ଯଦି (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            ଫେରନ୍ତୁ 0;
        }
        cpy  = n;
        skip = j - ଗଣ୍ଠି_idx;
    }
    ଯଦି (cpy == nullptr) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    ଅଚଳ ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    ଯଦି (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        ଫେରନ୍ତୁ 0;
    }

    // dst is the [D, n_seqs, n_written] cache view, with the per-seq stride D that the kernel assumes.
    // ggml_cpy pins src to the same element count, so src needs no shape check of its own.
    ଅଚଳ std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    ଯଦି (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(GGML_TYPE_F32) ||
        dst->nb[1] != (size_t) ggml_row_size(GGML_TYPE_F32, D)) {
        ଫେରନ୍ତୁ 0;
    }

    fused_state_cpy.data        = (ଦଶମିକ *) dst->data; // rollback group 0 (newest)
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / ଆକାର(ଦଶମିକ)) : 0;
    ଫେରନ୍ତୁ skip;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_graph_compute_impl(ggml_backend_sycl_context * sycl_ctx, ggml_cgraph * cgraph) {
    ggml_sycl_set_main_device(sycl_ctx->device);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ଗଣ୍ଠି = cgraph->nodes[i];
        ଯଦି (ggml_sycl_is_view_or_noop(ଗଣ୍ଠି)) {
            ଜାରି;
        }
        ଯଦି ((ଗଣ୍ଠି->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            ଜାରି;
        }

        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nodes_to_skip = ggml_sycl_fuse(*sycl_ctx, cgraph, i);
        ଯଦି (nodes_to_skip != 0) {
            i += nodes_to_skip;
            ଜାରି;
        }
#ifndef NDEBUG
        ପ୍ରତିଜ୍ଞା(ଗଣ୍ଠି->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < GGML_MAX_SRC; j++) {
            ଯଦି (ଗଣ୍ଠି->src[j] != nullptr) {
                ପ୍ରତିଜ୍ଞା(ଗଣ୍ଠି->src[j]->buffer->buft == ggml_backend_sycl_buffer_type(sycl_ctx->device));
            }
        }
#endif
        // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_GATED_DELTA_NET) {
            ggml_sycl_gated_delta_net_fused_cache fused_state_cpy;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା gdn_nodes_to_skip = ggml_sycl_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
            ଯଦି (gdn_nodes_to_skip > 0) {
                ggml_sycl_op_gated_delta_net_fused_cache(*sycl_ctx, ଗଣ୍ଠି, fused_state_cpy);
                i += gdn_nodes_to_skip;
                ଜାରି;
            }
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
            ggml_sycl_op_rms_norm_fused_add(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ଜାରି;
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
            ggml_sycl_op_rms_norm_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1]);
            i++;
            ଜାରି;
        }
        // qwen35 GDN l2 norms are emitted as rms_norm + scalar scale (models.h
        // build_gdn_l2_norm), which the rms_norm+mul fusion above cannot match
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_RMS_NORM &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE }, {})) {
            ggml_sycl_op_rms_norm_scale_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1]);
            i++;
            ଜାରି;
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_ADD &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_ADD }, {})) {
            ggml_sycl_op_add_add_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1]);
            i++;
            ଜାରି;
        }
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { ggml_get_unary_op(ଗଣ୍ଠି) })) {
            ggml_sycl_op_unary_mul_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1]);
            i++;
            ଜାରି;
        }
        // ADD(bias) + UNARY + MUL(scale) with both broadcast over dim 0, the form the branch
        // above cannot take; ggml_get_unary_op() asserts, so check the op first.
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_ADD && i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_UNARY &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_MUL },
                               { ggml_get_unary_op(cgraph->nodes[i + 1]) })) {
            ggml_sycl_op_add_unary_mul_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ଜାରି;
        }

        // Batch consecutive independent same-shape F32 L2_NORM siblings (the GDN q/k
        // norms) into one launch; sources are strided views of the fused qkv buffer, so
        // the scan skips the interleaved view nodes instead of breaking on them.
        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_L2_NORM) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା l2_batch_skip = ggml_sycl_l2_norm_batch_fused(*sycl_ctx, cgraph, i);
            ଯଦି (l2_batch_skip > 0) {
                i += l2_batch_skip;
                ଜାରି;
            }
        }

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ଗଣ୍ଠି, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
            i += 2;
            ଜାରି;
        }

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_SSM_CONV &&
            ggml_sycl_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
            ggml_sycl_ssm_conv_fused(*sycl_ctx, ଗଣ୍ଠି, nullptr, cgraph->nodes[i + 1]);
            i++;
            ଜାରି;
        }

        ଯଦି (ଗଣ୍ଠି->op == GGML_OP_MUL_MAT && ggml_sycl_mul_mat_glu_fused(*sycl_ctx, cgraph, i)) {
            i += 2;
            ଜାରି;
        }

        ତାର୍କିକ ok = ggml_sycl_compute_forward(*sycl_ctx, ଗଣ୍ଠି);
        ଯଦି (!ok) {
            GGML_LOG_ERROR("%s: error: op not supported %s (%s)\n", __func__, ଗଣ୍ଠି->name, ggml_op_name(ଗଣ୍ଠି->op));
        }
        GGML_ASSERT(ok);
    }
}

#ifdef GGML_SYCL_GRAPH
ସ୍ଥିର ତାର୍କିକ check_graph_compatibility(ggml_cgraph * cgraph) {
    ଯଦି (ggml_sycl_info().device_ଗଣନା > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < cgraph->n_nodes; i++) {
        ଅଚଳ ggml_op ଗଣ୍ଠି_op = cgraph->nodes[i]->op;
        ବାଛ (ଗଣ୍ଠି_op) {
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
            ସ୍ଥିତି GGML_OP_CONCAT:
                // ggml_sycl_op_concat() does a blocking host wait after memcpy operations,
                // but wait() can't be called on the events returned by a queue recording
                // to a graph.
                [[fallthrough]];
            ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait on the sycl queue after
                // submitting a memcpy operation, but wait() can't be called on a queue that
                // is recording to a graph.
                GGML_LOG_INFO("%s: disabling SYCL graphs due to unsupported node type %s\n", __func__,
                              ggml_op_name(ଗଣ୍ଠି_op));
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            ସ୍ଥିତି GGML_OP_MUL_MAT:
                // We cannot use graphs with ggml_sycl_mul_mat() when SYCL async memory allocation extensions are not available,
                // as SYCL malloc / free and host wait calls are not supported when recording to a graph which are all present
                // in reordering.
                ଯଦି (!g_ggml_sycl_use_async_mem_op) {
                    GGML_LOG_INFO(
                        "%s: disabling SYCL graphs due to unsupported node type when using a compiler without the "
                        "oneAPI async memory allocation extension "
                        "%s\n",
                        __func__, ggml_op_name(ଗଣ୍ଠି_op));
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
        }
    }
    ଫେରନ୍ତୁ ସତ୍ୟ;
}
#endif

ସ୍ଥିର ggml_status ggml_backend_sycl_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * sycl_ctx = static_cast<ggml_backend_sycl_context *>(backend->context);

#ifdef GGML_SYCL_GRAPH
    ତାର୍କିକ use_sycl_graph = ମିଥ୍ୟା;
    ଯଦି (g_ggml_sycl_enable_graph) {
        use_sycl_graph = check_graph_compatibility(cgraph);
    }
    ଯଦି (use_sycl_graph) {
        ଅଚଳ ତାର୍କିକ graph_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_limited_graph);
        ଯଦି (!graph_support) {
            GGML_SYCL_DEBUG("[SYCL-GRAPH] can not use graphs on device:%d\n", sycl_ctx->device);
            ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
            ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
        }

        sycl_ex::command_graph model_sycl_graph(*(sycl_ctx->stream()), {sycl_ex::property::graph::assume_buffer_outlives_graph{}});

        model_sycl_graph.begin_recording(*(sycl_ctx->stream()));
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
        model_sycl_graph.end_recording();

        ଅଚଳ ତାର୍କିକ graph_update_support = dpct::get_device(sycl_ctx->device).has(sycl::aspect::ext_oneapi_graph);
        ଯଦି (!sycl_ctx->exec_graph || !graph_update_support) {
            auto exec_graph = graph_update_support ? model_sycl_graph.finalize(sycl_ex::property::graph::updatable{}) :
                                                     model_sycl_graph.finalize();
            sycl_ctx->exec_graph = std::make_unique<
                sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
        } ଅନ୍ୟଥା {
            ଚେଷ୍ଟା {
                sycl_ctx->exec_graph->update(model_sycl_graph);
                GGML_SYCL_DEBUG("[SYCL-GRAPH] update success\n");
            } ଧର (sycl::exception ଅଚଳ & e) {
                GGML_SYCL_DEBUG("[SYCL-GRAPH] Exception when updating graph, %s\n", e.what());
                auto exec_graph = model_sycl_graph.finalize({sycl_ex::property::graph::updatable{}});
                sycl_ctx->exec_graph = std::make_unique<
                    sycl_ex::command_graph<sycl_ex::graph_state::executable>>(exec_graph);
            }
        }

        sycl_ctx->stream()->ext_oneapi_graph(*(sycl_ctx->exec_graph));
    } ଅନ୍ୟଥା
#endif
    {
        ggml_backend_sycl_graph_compute_impl(sycl_ctx, cgraph);
    }
    ଫେରନ୍ତୁ GGML_STATUS_SUCCESS;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_event_record(ggml_backend_t backend, ggml_backend_event_t event)
ଚେଷ୍ଟା
{
    ggml_backend_sycl_context *sycl_ctx =
        (ggml_backend_sycl_context *)backend->context;

    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);

    ଅଚଳ queue_ptr &stream = sycl_ctx->stream(sycl_ctx->device, 0);
    // Record the current state of the queue
    SYCL_CHECK(CHECK_TRY_ERROR(*sycl_event = stream->ext_oneapi_submit_barrier()));
}
ଧର (sycl::exception ଅଚଳ &exc)
{
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_event_wait(ggml_backend_t backend, ggml_backend_event_t event) ଚେଷ୍ଟା {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    sycl::event* sycl_event = static_cast<sycl::event*>(event->context);

    ଯଦି (ggml_backend_is_sycl(backend)) {
        SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
    } ଅନ୍ୟଥା
        GGML_ABORT("fatal error");
} ଧର (sycl::exception ଅଚଳ& exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
}

ସ୍ଥିର ggml_backend_i ggml_backend_sycl_interface = {
    /* .get_name                = */ ggml_backend_sycl_get_name,
    /* .free                    = */ ggml_backend_sycl_free,
    /* .set_tensor_async        = */ ggml_backend_sycl_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_sycl_get_tensor_async,
    /* .set_tensor_2d_async     = */ ଖାଲି,
    /* .get_tensor_2d_async     = */ ଖାଲି,
    /* .cpy_tensor_async        = */ ଖାଲି, // ggml_backend_sycl_cpy_tensor_async,
                                           // // TODO: update for the new
                                           // interface
    /* .synchronize             = */ ggml_backend_sycl_synchronize,
    /* .graph_plan_create       = */ ଖାଲି,
    /* .graph_plan_free         = */ ଖାଲି,
    /* .graph_plan_update       = */ ଖାଲି,
    /* .graph_plan_compute      = */ ଖାଲି,
    /* .graph_compute           = */ ggml_backend_sycl_graph_compute,
    /* .event_record            = */ ggml_backend_sycl_event_record,
    /* .event_wait              = */ ggml_backend_sycl_event_wait,
    /* .graph_optimize          = */ ଖାଲି,
};

ସ୍ଥିର ggml_guid_t ggml_backend_sycl_guid() {
    ସ୍ଥିର ggml_guid guid = { 0x58, 0x05, 0x13, 0x8f, 0xcd, 0x3a, 0x61, 0x9d, 0xe7, 0xcd, 0x98, 0xa9, 0x03, 0xfd, 0x7c, 0x53 };
    ଫେରନ୍ତୁ &guid;
}

ତାର୍କିକ ggml_backend_is_sycl(ggml_backend_t backend) {
    ଫେରନ୍ତୁ backend != ଖାଲି && ggml_guid_matches(backend->guid, ggml_backend_sycl_guid());
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା ggml_backend_sycl_get_device_ଗଣନା(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ ggml_sycl_info().device_ଗଣନା;
}


// backend device

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ ctx->name.c_str();
}

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ ctx->description.c_str();
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_device_get_memory(ggml_backend_dev_t dev, size_t * ମୁକ୍ତ, size_t * total) {
    GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ଯଦି (!sycl_get_mem_info(ctx->device, ମୁକ୍ତ, total)) {
        GGML_ABORT("[%s] failed to get device memory size", __func__);
    }
    GGML_SYCL_DEBUG("[SYCL] call %s total %zu free %zu\n", __func__, *total, *ମୁକ୍ତ);
    ggml_sycl_memtrace_report_device("device memory query (dev)", ctx->device, *ମୁକ୍ତ, *total);
}

ସ୍ଥିର ଗଣନା ggml_backend_dev_type ggml_backend_sycl_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଅଚଳ sycl_device_info & info = ggml_sycl_info().devices[ctx->device];
    ଯଦି (info.l0_device_type_valid && !info.l0_discrete_gpu) {
        ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_IGPU;
    }
    ଫେରନ୍ତୁ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_sycl_device_get_name(dev);
    props->description = ggml_backend_sycl_device_get_description(dev);
    props->type        = ggml_backend_sycl_device_get_type(dev);
    ggml_backend_sycl_device_get_memory(dev, &props->memory_free, &props->memory_total);

    ତାର୍କିକ host_buffer = getenv("GGML_SYCL_NO_PINNED") == nullptr;
#ifdef GGML_SYCL_NO_PEER_COPY
    ତାର୍କିକ events = ମିଥ୍ୟା;
#ଅନ୍ୟଥା
    ତାର୍କିକ events = ସତ୍ୟ;
#endif

    props->caps = {
        /* .async                 = */ ସତ୍ୟ,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ ମିଥ୍ୟା,
        /* .events                = */ events,
        /* .mmap_support          = */ ସତ୍ୟ,
    };
}

ସ୍ଥିର ggml_backend_t ggml_backend_sycl_device_init(ggml_backend_dev_t dev, ଅଚଳ ଅକ୍ଷର * params) {
    GGML_UNUSED(params);
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ ggml_backend_sycl_init(ctx->device);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_sycl_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ ggml_backend_sycl_buffer_type(ctx->device);
}

ସ୍ଥିର ggml_backend_buffer_type_t ggml_backend_sycl_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_sycl_device_context * ctx = (ggml_backend_sycl_device_context *) dev->context;
    ଫେରନ୍ତୁ ggml_backend_sycl_host_buffer_type_for_device(ctx->device);
}

ସ୍ଥିର ggml_backend_buffer_t ggml_backend_sycl_device_buffer_from_host_ptr(ggml_backend_dev_t dev, ଶୂନ୍ୟ * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(ptr);
    GGML_UNUSED(size);
    GGML_UNUSED(max_tensor_size);
    ଫେରନ୍ତୁ nullptr;
}

ସ୍ଥିର ତାର୍କିକ do_ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ggml_backend_sycl_device_context *sycl_ctx =
        (ggml_backend_sycl_device_context *)dev->context;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା device = sycl_ctx->device;
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                ଯଦି (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                }
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        ସ୍ଥିତି GGML_OP_CONV_2D:
            // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
            // CWHN: channel-fastest input (C fastest) and kernel with OC fastest
            ଫେରନ୍ତୁ (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) ||
                   (op->src[1]->nb[2] == ggml_type_size(op->src[1]->type) &&
                    op->src[0]->nb[3] == ggml_type_size(op->src[0]->type));
        ସ୍ଥିତି GGML_OP_CONV_2D_DW:
        ସ୍ଥିତି GGML_OP_CONV_TRANSPOSE_2D:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_UNARY:
            ବାଛ (ggml_get_unary_op(op)) {
                ସ୍ଥିତି GGML_UNARY_OP_SGN:
                ସ୍ଥିତି GGML_UNARY_OP_ABS:
                ସ୍ଥିତି GGML_UNARY_OP_NEG:
                ସ୍ଥିତି GGML_UNARY_OP_STEP:
                ସ୍ଥିତି GGML_UNARY_OP_RELU:
                ସ୍ଥିତି GGML_UNARY_OP_HARDSIGMOID:
                ସ୍ଥିତି GGML_UNARY_OP_TANH:
                ସ୍ଥିତି GGML_UNARY_OP_GELU:
                ସ୍ଥିତି GGML_UNARY_OP_SILU:
                ସ୍ଥିତି GGML_UNARY_OP_SIGMOID:
                ସ୍ଥିତି GGML_UNARY_OP_HARDSWISH:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_QUICK:
                ସ୍ଥିତି GGML_UNARY_OP_GELU_ERF:
                ସ୍ଥିତି GGML_UNARY_OP_EXP:
                ସ୍ଥିତି GGML_UNARY_OP_EXPM1:
                ସ୍ଥିତି GGML_UNARY_OP_SOFTPLUS:
                ସ୍ଥିତି GGML_UNARY_OP_ELU:
                ସ୍ଥିତି GGML_UNARY_OP_XIELU:
                ସ୍ଥିତି GGML_UNARY_OP_CEIL:
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                ସ୍ଥିତି GGML_UNARY_OP_FLOOR:
                ସ୍ଥିତି GGML_UNARY_OP_ROUND:
                ସ୍ଥିତି GGML_UNARY_OP_TRUNC:
                    ଫେରନ୍ତୁ ସତ୍ୟ;
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        ସ୍ଥିତି GGML_OP_GLU:
            ବାଛ (ggml_get_glu_op(op)) {
                ସ୍ଥିତି GGML_GLU_OP_REGLU:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_OAI:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_ERF:
                ସ୍ଥିତି GGML_GLU_OP_GEGLU_QUICK:
                ସ୍ଥିତି GGML_GLU_OP_SWIGLU_CLAMP:
                    // the fused GLU kernels only have f32/f16 instantiations
                    ଫେରନ୍ତୁ ggml_is_contiguous_1(op->src[0]) &&
                           (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
                ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        ସ୍ଥିତି GGML_OP_MUL_MAT:
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
            {
                ସଂରଚନା ggml_tensor * a = op->src[0];
                ସଂରଚନା ggml_tensor * b = op->src[1];

                ଯଦି (a->ne[3] != b->ne[3]) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ggml_type src0_type = op->src[0]->type;

                // TODO: The configuration below needs more work to be supported with oneDNN
                ଯଦି (ggml_is_permuted(a) && !ggml_is_contiguous(a) &&
                    a->ne[2] > 1 && a->ne[3] > 1 && src0_type == GGML_TYPE_F16) {
                  ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                // TODO: This specific configuration can fail with oneDNN and needs more debugging
                ଯଦି (!ggml_is_permuted(a) && ggml_is_permuted(b) && b->ne[2] > 1 && b->ne[3] > 1 &&
                    a->ne[0] > 128 && a->ne[2] == 1 && src0_type == GGML_TYPE_F16) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଯଦି (src0_type == GGML_TYPE_TQ2_0 || src0_type == GGML_TYPE_TQ1_0) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        ସ୍ଥିତି GGML_OP_OUT_PROD:
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 ||
                    (op->src[0]->type == GGML_TYPE_Q1_0 && op->src[0]->ne[2] == op->src[1]->ne[2] &&
                     op->src[0]->ne[3] == op->src[1]->ne[3])) &&
                   op->src[1]->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            {
                ବାଛ (op->src[0]->type) {
                    ସ୍ଥିତି GGML_TYPE_I32:
                    ସ୍ଥିତି GGML_TYPE_F16:
                    ସ୍ଥିତି GGML_TYPE_BF16:
                    ସ୍ଥିତି GGML_TYPE_F32:
                    ସ୍ଥିତି GGML_TYPE_Q1_0:
                    ସ୍ଥିତି GGML_TYPE_MXFP4:
                    ସ୍ଥିତି GGML_TYPE_NVFP4:
                    ସ୍ଥିତି GGML_TYPE_IQ2_XXS:
                    ସ୍ଥିତି GGML_TYPE_IQ2_XS:
                    ସ୍ଥିତି GGML_TYPE_IQ2_S:
                    ସ୍ଥିତି GGML_TYPE_IQ3_XXS:
                    ସ୍ଥିତି GGML_TYPE_IQ1_S:
                    ସ୍ଥିତି GGML_TYPE_IQ1_M:
                    ସ୍ଥିତି GGML_TYPE_IQ3_S:
                    ସ୍ଥିତି GGML_TYPE_IQ4_NL:
                    ସ୍ଥିତି GGML_TYPE_IQ4_XS:
                    ସ୍ଥିତି GGML_TYPE_Q2_K:
                    ସ୍ଥିତି GGML_TYPE_Q3_K:
                    ସ୍ଥିତି GGML_TYPE_Q4_0:
                    ସ୍ଥିତି GGML_TYPE_Q4_1:
                    ସ୍ଥିତି GGML_TYPE_Q4_K:
                    ସ୍ଥିତି GGML_TYPE_Q5_0:
                    ସ୍ଥିତି GGML_TYPE_Q5_1:
                    ସ୍ଥିତି GGML_TYPE_Q5_K:
                    ସ୍ଥିତି GGML_TYPE_Q6_K:
                    ସ୍ଥିତି GGML_TYPE_Q8_0:
                        ଫେରନ୍ତୁ ସତ୍ୟ;
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
            }
        ସ୍ଥିତି GGML_OP_GET_ROWS_BACK:
            // return true;
             ଫେରନ୍ତୁ op->type == GGML_TYPE_F32 &&
                 (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                 op->src[1]->type == GGML_TYPE_I32 &&
                 op->ne[2] == 1 && op->ne[3] == 1;
         ସ୍ଥିତି GGML_OP_SET:
               ଫେରନ୍ତୁ (op->type == GGML_TYPE_F32) &&
                      (op->src[0] && op->src[1]) &&
                      (op->src[0]->type == GGML_TYPE_F32) &&
                      (op->src[1]->type == GGML_TYPE_F32);

        ସ୍ଥିତି GGML_OP_SET_ROWS:
            {
                ଯଦି (op->type == GGML_TYPE_TQ2_0 || op->type == GGML_TYPE_TQ1_0) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }
                auto res = (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                            op->src[0]->type == GGML_TYPE_BF16) &&
                           (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
                ଫେରନ୍ତୁ res;
            }
        ସ୍ଥିତି GGML_OP_DSV4_HC_PRE:
            ଫେରନ୍ତୁ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_DSV4_HC_COMB:
            ଫେରନ୍ତୁ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_DSV4_HC_POST:
            ଫେରନ୍ତୁ op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_LIGHTNING_INDEXER:
            ଫେରନ୍ତୁ op->src[0]->type == GGML_TYPE_F32 &&
                (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32 ||
                 op->src[1]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_Q8_0 ||
                 op->src[1]->type == GGML_TYPE_Q5_1 || op->src[1]->type == GGML_TYPE_Q5_0 ||
                 op->src[1]->type == GGML_TYPE_Q4_1 || op->src[1]->type == GGML_TYPE_Q4_0 ||
                 op->src[1]->type == GGML_TYPE_IQ4_NL) &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->src[3]->type == GGML_TYPE_F16 &&
                op->type == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == WARP_SIZE * 8;
        ସ୍ଥିତି GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;

                ଯଦି (src0_type == GGML_TYPE_F16) {
                    ଯଦି (src1_type == GGML_TYPE_Q2_K ||
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
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }

                ଯଦି (src0_type == GGML_TYPE_BF16) {
                    ଯଦି (src1_type == GGML_TYPE_Q4_0 || //big error in ut
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
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }

                ଯଦି (src0_type == GGML_TYPE_F32) {
                    ଯଦି (src1_type == GGML_TYPE_Q2_K ||
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
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }

                ଯଦି (src1_type == GGML_TYPE_F32) {
                    ଯଦି (src0_type == GGML_TYPE_Q1_0 ||
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
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }

                ଯଦି (src0_type == src1_type) {
                    ଯଦି (src1_type == GGML_TYPE_IQ2_XXS ||
                        src1_type == GGML_TYPE_IQ2_XS ||
                        src1_type == GGML_TYPE_IQ2_S ||
                        src1_type == GGML_TYPE_IQ3_XXS ||
                        src1_type == GGML_TYPE_IQ3_S ||
                        src1_type == GGML_TYPE_IQ1_S ||
                        src1_type == GGML_TYPE_IQ1_M ||
                        src1_type == GGML_TYPE_TQ2_0 ||
                        src1_type == GGML_TYPE_TQ1_0) {
                        ଫେରନ୍ତୁ ମିଥ୍ୟା;
                    }
                }

                ଯଦି (src0_type == GGML_TYPE_TQ2_0 || src1_type == GGML_TYPE_TQ2_0 ||
                    src0_type == GGML_TYPE_TQ1_0 || src1_type == GGML_TYPE_TQ1_0) {
                    ଫେରନ୍ତୁ ମିଥ୍ୟା;
                }

                ଫେରନ୍ତୁ ସତ୍ୟ;
            }
        ସ୍ଥିତି GGML_OP_REPEAT_BACK:
            {
                ggml_type src0_type = op->src[0]->type;
                ଫେରନ୍ତୁ src0_type == GGML_TYPE_F32;
            }
        ସ୍ଥିତି GGML_OP_CONCAT:
        ସ୍ଥିତି GGML_OP_DUP:
        ସ୍ଥିତି GGML_OP_ARGMAX:
        ସ୍ଥିତି GGML_OP_NONE:
        ସ୍ଥିତି GGML_OP_RESHAPE:
        ସ୍ଥିତି GGML_OP_VIEW:
        ସ୍ଥିତି GGML_OP_PERMUTE:
        ସ୍ଥିତି GGML_OP_TRANSPOSE:
        ସ୍ଥିତି GGML_OP_ADD:
        ସ୍ଥିତି GGML_OP_ADD1:
        ସ୍ଥିତି GGML_OP_ADD_ID:
        ସ୍ଥିତି GGML_OP_SUB:
        ସ୍ଥିତି GGML_OP_ଗଣନା_EQUAL:
        ସ୍ଥିତି GGML_OP_MUL:
        ସ୍ଥିତି GGML_OP_DIV:
        ସ୍ଥିତି GGML_OP_REPEAT:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_PAD_REFLECT_1D:
            ଫେରନ୍ତୁ ggml_is_contiguous(op->src[0]) && op-> type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_SQR:
        ସ୍ଥିତି GGML_OP_SQRT:
        ସ୍ଥିତି GGML_OP_SIN:
        ସ୍ଥିତି GGML_OP_COS:
        ସ୍ଥିତି GGML_OP_CLAMP:
        ସ୍ଥିତି GGML_OP_LOG:
        ସ୍ଥିତି GGML_OP_NORM:
        ସ୍ଥିତି GGML_OP_L2_NORM:
        ସ୍ଥିତି GGML_OP_GROUP_NORM:
        ସ୍ଥିତି GGML_OP_RMS_NORM:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_RMS_NORM_BACK:
            ଫେରନ୍ତୁ ggml_is_contiguous(op->src[0]);
        ସ୍ଥିତି GGML_OP_SCALE:
            // the scale kernel is f32 only
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_CONT:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_TRI:
            {
                ଅଚଳ ggml_tensor * src0 = op->src[0];
                ଫେରନ୍ତୁ src0 &&
                       op->type == GGML_TYPE_F32 &&
                       ggml_is_contiguous(src0);
            }
        ସ୍ଥିତି GGML_OP_DIAG_MASK_INF:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_SOFT_MAX_BACK: {
            ଦଶମିକ max_bias = 0.0f;
            memcpy(&max_bias, (ଅଚଳ ଦଶମିକ *) op->op_params + 1, ଆକାର(ଦଶମିକ));
            ଫେରନ୍ତୁ max_bias == 0.0f;
        }
        ସ୍ଥିତି GGML_OP_ROPE:
        ସ୍ଥିତି GGML_OP_ROPE_BACK:
        ସ୍ଥିତି GGML_OP_IM2COL:
        ସ୍ଥିତି GGML_OP_IM2COL_3D:
        ସ୍ଥିତି GGML_OP_UPSCALE:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_COL2IM_1D:
            ଫେରନ୍ତୁ ggml_is_contiguous(op->src[0]) &&
                   (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16
#ifdef GGML_SYCL_HAS_BF16
                    || op->type == GGML_TYPE_BF16
#endif
                   ) &&
                   op->src[0]->type == op->type;
        ସ୍ଥିତି GGML_OP_CONV_3D: {
            ଅଚଳ int32_t * opts = (ଅଚଳ int32_t *) op->op_params;
            // zero kernel volume, channels or output makes the im2col GEMM degenerate (k == 0 or m == 0)
            // and MKL BLAS rejects it; let those fall back to the CPU backend
            ଅଚଳ int64_t knl_n_total = op->src[0]->ne[0] * op->src[0]->ne[1] * op->src[0]->ne[2] * opts[9];
            ଅଚଳ int64_t patch_total = op->ne[0] * op->ne[1] * op->ne[2] * opts[10];
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   knl_n_total > 0 && patch_total > 0;
        }
        ସ୍ଥିତି GGML_OP_SUM:
        ସ୍ଥିତି GGML_OP_SUM_ROWS:
        ସ୍ଥିତି GGML_OP_MEAN:
            ଫେରନ୍ତୁ ggml_is_contiguous(op->src[0]);
        ସ୍ଥିତି GGML_OP_ARGSORT:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_ଶୀର୍ଷ_K: {
            ଅଚଳ ggml_tensor * src0 = op->src[0];
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = op->ne[0];
            ଫେରନ୍ତୁ src0 &&
                op->type == GGML_TYPE_I32 &&
                src0->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(src0) &&
                k > 0 && k <= src0->ne[0];
        }
        ସ୍ଥିତି GGML_OP_POOL_2D:
        ସ୍ଥିତି GGML_OP_POOL_1D:
        ସ୍ଥିତି GGML_OP_ACC:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_PAD:
            ଯଦି (ggml_get_op_params_i32(op, 8) != 0) {
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_LEAKY_RELU:
        ସ୍ଥିତି GGML_OP_TIMESTEP_EMBEDDING:
        ସ୍ଥିତି GGML_OP_RWKV_WKV6:
        ସ୍ଥିତି GGML_OP_RWKV_WKV7:
        ସ୍ଥିତି GGML_OP_GATED_LINEAR_ATTN:
        ସ୍ଥିତି GGML_OP_GATED_DELTA_NET:
        ସ୍ଥିତି GGML_OP_OPT_STEP_ADAMW:
        ସ୍ଥିତି GGML_OP_OPT_STEP_SGD:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_SSM_CONV:
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32 &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_ROLL:
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_ARANGE:
            ଫେରନ୍ତୁ op->type == GGML_TYPE_F32;
        ସ୍ଥିତି GGML_OP_SSM_SCAN:
            ଯଦି (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % WARP_SIZE == 0)
                ଫେରନ୍ତୁ (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % WARP_SIZE == 0;
            } ଅନ୍ୟଥା {
                // TODO Mamba-1 not yet ported to SYCL
                ଫେରନ୍ତୁ ମିଥ୍ୟା;
            }
        ସ୍ଥିତି GGML_OP_FILL:
        ସ୍ଥିତି GGML_OP_CUMSUM:
        ସ୍ଥିତି GGML_OP_DIAG:
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS:
        ସ୍ଥିତି GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ଫେରନ୍ତୁ ସତ୍ୟ;
        ସ୍ଥିତି GGML_OP_SOLVE_TRI:
            ଫେରନ୍ତୁ op->src[0]->ne[0] <= SYCL_SOLVE_TRI_MAX_N && op->src[1]->ne[0] <= SYCL_SOLVE_TRI_MAX_K;
        ସ୍ଥିତି GGML_OP_FLASH_ATTN_EXT:
            ଫେରନ୍ତୁ ggml_sycl_flash_attn_ext_supported(device, op);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    GGML_UNUSED(dev);
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_device_supports_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ତାର୍କିକ res = do_ggml_backend_sycl_device_supports_op(dev, op);
    GGML_SYCL_DEBUG("[SYCL] call %s op->op=%s op->type=%s -> %s\n", __func__, ggml_op_name(op->op),
                    ggml_type_name(op->type), res ? "true" : "false");
    ଫେରନ୍ତୁ res;
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ଯଦି (buft->iface.get_name != ggml_backend_sycl_buffer_type_get_name) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ggml_backend_sycl_buffer_type_context * buft_ctx = (ggml_backend_sycl_buffer_type_context *)buft->context;
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ buft_ctx->device == sycl_ctx->device;
}

ସ୍ଥିର int64_t get_op_batch_size(ଅଚଳ ggml_tensor * op) {
    ବାଛ (op->op) {
        ସ୍ଥିତି GGML_OP_GET_ROWS:
            ଫେରନ୍ତୁ 0;
        ସ୍ଥିତି GGML_OP_MUL_MAT:
            ଫେରନ୍ତୁ op->ne[1];
        ସ୍ଥିତି GGML_OP_MUL_MAT_ID:
        ସ୍ଥିତି GGML_OP_ROPE:
            ଫେରନ୍ତୁ op->ne[2];
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ ggml_nrows(op);
    }
}

ସ୍ଥିର ତାର୍କିକ ggml_backend_sycl_device_offload_op(ggml_backend_dev_t dev, ଅଚଳ ggml_tensor * op) {
    ggml_backend_sycl_device_context * sycl_ctx = (ggml_backend_sycl_device_context *)dev->context;
    ଫେରନ୍ତୁ get_op_batch_size(op) >= sycl_ctx->op_offload_min_batch_size;
}

ସ୍ଥିର ggml_backend_event_t
ggml_backend_sycl_device_event_new(ggml_backend_dev_t dev) {

#ifdef GGML_SYCL_NO_PEER_COPY
    ଫେରନ୍ତୁ nullptr;
#ଅନ୍ୟଥା
  sycl::event *event_ptr = ନୂତନ sycl::event();

  ଫେରନ୍ତୁ ନୂତନ ggml_backend_event{
      /* .device = */ dev,
      /* .context = */ event_ptr,
  };
#endif
}

ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) ଚେଷ୍ଟା {
  GGML_UNUSED(dev);
  ଯଦି (event == nullptr) {
    ଫେରନ୍ତୁ;
  }

  ଯଦି (event->context != nullptr) {
    sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
    delete sycl_event;
    event->context = nullptr;
  }

  delete event;
} ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}


ସ୍ଥିର ଶୂନ୍ୟ ggml_backend_sycl_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) ଚେଷ୍ଟା {
  GGML_UNUSED(dev);
  GGML_SYCL_DEBUG("[SYCL] call %s\n", __func__);

  sycl::event *sycl_event = static_cast<sycl::event *>(event->context);
  SYCL_CHECK(CHECK_TRY_ERROR(sycl_event->wait()));
} ଧର (sycl::exception ଅଚଳ &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

ସ୍ଥିର ଅଚଳ ggml_backend_device_i ggml_backend_sycl_device_interface = {
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

ସଂରଚନା ggml_backend_sycl_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

ସ୍ଥିର ଅଚଳ ଅକ୍ଷର * ggml_backend_sycl_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    ଫେରନ୍ତୁ GGML_SYCL_NAME;
}

ସ୍ଥିର size_t ggml_backend_sycl_reg_get_device_ଗଣନା(ggml_backend_reg_t reg) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    ଫେରନ୍ତୁ ctx->devices.size();
}

ସ୍ଥିର ggml_backend_dev_t ggml_backend_sycl_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_sycl_reg_context * ctx = (ggml_backend_sycl_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    ଫେରନ୍ତୁ ctx->devices[index];
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

ସଂରଚନା ggml_backend_sycl_comm_context {
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

    ଶୂନ୍ୟ free_host() {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 2; ++i) {
            ଅଚଳ sycl::context sctx = ((ggml_backend_sycl_context *) backends[i]->context)->stream()->get_context();
            ଯଦି (host_out[i]) { sycl::ମୁକ୍ତ(host_out[i], sctx); host_out[i] = nullptr; }
            ଯଦି (host_in[i])  { sycl::ମୁକ୍ତ(host_in[i],  sctx); host_in[i]  = nullptr; }
        }
        host_bytes = 0;
    }
};

// dst0 <- src1 and dst1 <- src0 through pinned host memory, with a single host wait
ସ୍ଥିର ଶୂନ୍ୟ ggml_sycl_comm_exchange(ggml_backend_sycl_comm_context * comm, queue_ptr q0, queue_ptr q1,
                                    ଶୂନ୍ୟ * dst0, ଅଚଳ ଶୂନ୍ୟ * src1, ଶୂନ୍ୟ * dst1, ଅଚଳ ଶୂନ୍ୟ * src0,
                                    size_t nbytes) {
    ଯଦି (comm->host_bytes < nbytes) {
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

ଶୂନ୍ୟ * ggml_backend_sycl_comm_init(ggml_backend_t * backends, size_t n_backends) ଚେଷ୍ଟା {
    ପାଇଁ (size_t i = 0; i < n_backends; ++i) {
        ଯଦି (!ggml_backend_is_sycl(backends[i])) {
            ଫେରନ୍ତୁ nullptr;
        }
    }

    // Initial version: N=2 only. For N!=2, returning null makes the
    // meta-backend skip this backend-specific allreduce entirely.
    ଯଦି (n_backends != 2) {
        ଫେରନ୍ତୁ nullptr;
    }

    auto * ctx = ନୂତନ ggml_backend_sycl_comm_context;
    ctx->backends.assign(backends, backends + n_backends);
    auto * sctx0 = (ggml_backend_sycl_context *) backends[0]->context;
    auto * sctx1 = (ggml_backend_sycl_context *) backends[1]->context;
    ctx->buf0 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx0->pool());
    ctx->buf1 = std::make_unique<ggml_sycl_pool_alloc<uint8_t>>(sctx1->pool());
    ଫେରନ୍ତୁ ctx;
}
ଧର (ଅଚଳ sycl::exception &) { ଫେରନ୍ତୁ nullptr; }
ଧର (...)                     { ଫେରନ୍ତୁ nullptr; }

ଶୂନ୍ୟ ggml_backend_sycl_comm_free(ଶୂନ୍ୟ * comm_ctx_v) {
    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ଯଦି (comm_ctx == nullptr) {
        ଫେରନ୍ତୁ;
    }

    // Sync both per-device queues so the pool_alloc destructors don't
    // return memory still in use by the last kernel.
    ଯଦି (comm_ctx->backends.size() == 2) {
        auto * sctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
        auto * sctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
        ଚେଷ୍ଟା {
            sctx0->stream()->wait();
            sctx1->stream()->wait();
            comm_ctx->free_host();
        } ଧର (...) { /* best effort during shutdown */ }
    }

    delete comm_ctx;
}

ତାର୍କିକ ggml_backend_sycl_comm_allreduce_tensor(ଶୂନ୍ୟ * comm_ctx_v, ସଂରଚନା ggml_tensor ** tensors) ଚେଷ୍ଟା {
    ଯଦି (comm_ctx_v == nullptr) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    auto * comm_ctx = static_cast<ggml_backend_sycl_comm_context *>(comm_ctx_v);
    ଅଚଳ size_t n_backends = comm_ctx->backends.size();

    // Fast path: N=2, F32/F16, contiguous, matching shapes.
    ଯଦି (n_backends != 2) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    // Accept F32 or F16 inputs natively (types must match). F16 takes the
    // direct 2-byte memcpy + add path below; other types return false so the
    // meta-backend uses its generic all-reduce.
    ଯଦି (tensors[0]->type != tensors[1]->type) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (tensors[0]->type != GGML_TYPE_F32 && tensors[0]->type != GGML_TYPE_F16) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (!ggml_is_contiguous(tensors[0]) || !ggml_is_contiguous(tensors[1])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
    ଯଦି (ggml_nelements(tensors[0]) != ggml_nelements(tensors[1])) {
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }

    ଅଚଳ int64_t nelem  = ggml_nelements(tensors[0]);
    ଅଚଳ size_t  nbytes = ggml_nbytes(tensors[0]);
    ଯଦି (nelem == 0) {
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    auto * ctx0 = (ggml_backend_sycl_context *) comm_ctx->backends[0]->context;
    auto * ctx1 = (ggml_backend_sycl_context *) comm_ctx->backends[1]->context;
    queue_ptr q0 = ctx0->stream();
    queue_ptr q1 = ctx1->stream();

    // Grow per-device byte buffers if needed (4 * nelem bytes each).
    ଯଦି (comm_ctx->buf_nelem < nelem) {
        comm_ctx->buf0->realloc(nelem * 4);
        comm_ctx->buf1->realloc(nelem * 4);
        comm_ctx->buf_nelem = nelem;
    }
    uint8_t * buf0 = comm_ctx->buf0->get();
    uint8_t * buf1 = comm_ctx->buf1->get();

    // F16 native path: direct 2-byte cross-device copy + add, skipping the
    // F32 round-trip the meta-backend fallback would force. The devices are in separate SYCL
    // contexts (a raw peer-USM q->memcpy would be a silent no-op), so copies go through host memory.
    ଯଦି (tensors[0]->type == GGML_TYPE_F16) {
        sycl::half * f16_out0 = (sycl::half *) tensors[0]->data;
        sycl::half * f16_out1 = (sycl::half *) tensors[1]->data;
        sycl::half * f16_tmp0 = (sycl::half *) buf0;
        sycl::half * f16_tmp1 = (sycl::half *) buf1;

        ggml_sycl_comm_exchange(comm_ctx, q0, q1, f16_tmp0, tensors[1]->data, f16_tmp1, tensors[0]->data, nbytes);

        q0->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out0[i] = (sycl::half) ((ଦଶମିକ) f16_out0[i] + (ଦଶମିକ) f16_tmp0[i]);
            });
        });
        q1->submit([&](sycl::handler & h) {
            h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
                f16_out1[i] = (sycl::half) ((ଦଶମିକ) f16_out1[i] + (ଦଶମିକ) f16_tmp1[i]);
            });
        });
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    ଦଶମିକ * out0 = (ଦଶମିକ *) tensors[0]->data;
    ଦଶମିକ * out1 = (ଦଶମିକ *) tensors[1]->data;

    // BF16 threshold: above this, the PCIe savings from halving the
    // cross-device bytes outweigh the 2 extra compress kernels.
    // Below: stay on the FP32 fast path.  Threshold mirrors the CUDA
    // NCCL allreduce pattern for n_backends=2.
    ସ୍ଥିର constexpr int64_t BF16_THRESHOLD = 32768;

    ଯଦି (nelem < BF16_THRESHOLD) {
        // FP32 small path: 4 SYCL submissions per allreduce.
        ଦଶମିକ * tmp0 = (ଦଶମିକ *) buf0;
        ଦଶମିକ * tmp1 = (ଦଶମିକ *) buf1;

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
        ଫେରନ୍ତୁ ସତ୍ୟ;
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
    ଅଚଳ size_t bf16_bytes = nelem * ଆକାର(uint16_t);
    ggml_sycl_comm_exchange(comm_ctx, q0, q1, inbox0, outbox1, inbox1, outbox0, bf16_bytes);

    // Phase C: decompress + add into local FP32 partial.
    q0->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out0[i] += sycl::bit_cast<ଦଶମିକ>(((uint32_t) inbox0[i]) << 16);
        });
    });

    q1->submit([&](sycl::handler & h) {
        h.parallel_for(sycl::range<1>(nelem), [=](sycl::id<1> i) {
            out1[i] += sycl::bit_cast<ଦଶମିକ>(((uint32_t) inbox1[i]) << 16);
        });
    });

    ଫେରନ୍ତୁ ସତ୍ୟ;
}
ଧର (ଅଚଳ sycl::exception &) { ଫେରନ୍ତୁ ମିଥ୍ୟା; }
ଧର (...)                     { ଫେରନ୍ତୁ ମିଥ୍ୟା; }

ସ୍ଥିର ଶୂନ୍ୟ *ggml_backend_sycl_reg_get_proc_address(ggml_backend_reg_t reg, ଅଚଳ ଅକ୍ଷର *name) {
    GGML_UNUSED(reg);

    ଯଦି (strcmp(name, "ggml_backend_split_buffer_type") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_sycl_split_buffer_type;
    }

    // Tensor parallelism (--split-mode tensor) entry points.
    ଯଦି (strcmp(name, "ggml_backend_comm_init") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_sycl_comm_init;
    }
    ଯଦି (strcmp(name, "ggml_backend_comm_free") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_sycl_comm_free;
    }
    ଯଦି (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        ଫେରନ୍ତୁ (ଶୂନ୍ୟ *)ggml_backend_sycl_comm_allreduce_tensor;
    }

    // SYCL doesn't support registering host memory, left here for reference
    // "ggml_backend_register_host_buffer"
    // "ggml_backend_unregister_host_buffer"
    GGML_UNUSED(name);
    ଫେରନ୍ତୁ nullptr;
}

ସ୍ଥିର ଅଚଳ ggml_backend_reg_i ggml_backend_sycl_reg_interface = {
    /* .get_name          = */ ggml_backend_sycl_reg_get_name,
    /* .get_device_count  = */ ggml_backend_sycl_reg_get_device_ଗଣନା,
    /* .get_device        = */ ggml_backend_sycl_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_sycl_reg_get_proc_address,
};


// backend registry

ggml_backend_reg_t ggml_backend_sycl_reg() {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_reg\n");
    ସ୍ଥିର ggml_backend_reg reg;
    ସ୍ଥିର ତାର୍କିକ initialized = ମିଥ୍ୟା;

    {
        ସ୍ଥିର std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        ଯଦି (!initialized) {
            ggml_check_sycl();
            ggml_backend_sycl_reg_context * ctx = ନୂତନ ggml_backend_sycl_reg_context;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ggml_sycl_info().device_ଗଣନା; i++) {
                ggml_backend_sycl_device_context * dev_ctx = ନୂତନ ggml_backend_sycl_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_SYCL_NAME + std::to_string(i);

                ggml_sycl_set_device(i);

                dpct::device_info prop;
                SYCL_CHECK(CHECK_TRY_ERROR(dpct::get_device_info(
                    prop, dpct::dev_mgr::instance().get_device(i))));

                dev_ctx->description = prop.get_name();
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = ନୂତନ ggml_backend_device {
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

        initialized = ସତ୍ୟ;
    }

    ଫେରନ୍ତୁ &reg;
}

ggml_backend_t ggml_backend_sycl_init(ପୂର୍ଣ୍ଣସଂଖ୍ୟା device) {
    GGML_SYCL_DEBUG("[SYCL] call ggml_backend_sycl_init\n");
    ggml_check_sycl();

    check_allow_gpu_index(device);

    ggml_backend_sycl_context * ctx = ନୂତନ ggml_backend_sycl_context(device);
    ଯଦି (ctx == nullptr) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        ଫେରନ୍ତୁ nullptr;
    };

    ggml_backend_t sycl_backend = ନୂତନ ggml_backend {
        /* .guid    = */ ggml_backend_sycl_guid(),
        /* .iface   = */ ggml_backend_sycl_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_sycl_reg(), device),
        /* .context = */ ctx
    };

    ଫେରନ୍ତୁ sycl_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_sycl_reg)
