#ଅନ୍ତର୍ଭାଵ <sycl/ext/oneapi/backend/level_zero.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ଅନ୍ତର୍ଭାଵ <level_zero/ze_api.h>
#ଅନ୍ତର୍ଭାଵ <level_zero/zes_api.h>
#endif

#ଅନ୍ତର୍ଭାଵ <cstdint>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <vector>

#ଅନ୍ତର୍ଭାଵ "base.hpp"
#ଅନ୍ତର୍ଭାଵ "mem.hpp"

ଅଚଳ ଅକ୍ଷର * mem_api_int2str(ପୂର୍ଣ୍ଣସଂଖ୍ୟା mem_api) {
    ଯଦି (mem_api == MEMORY_API_TYPE_SYCL) {
        ଫେରନ୍ତୁ "SYCL API";
    } ଅନ୍ୟଥା ଯଦି (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        ଫେରନ୍ତୁ "Level Zero API";
    } ଅନ୍ୟଥା {
        ଫେରନ୍ତୁ "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ସ୍ଥିର ତାର୍କିକ query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_ଗଣନା = 0;

#ଯଦି defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ଅନ୍ୟଥା
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    ଚେଷ୍ଟା {

        ଯଦି (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ଯଦି (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_ଗଣନା, nullptr);
        ଯଦି (r != ZE_RESULT_SUCCESS || module_ଗଣନା == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        std::vector<zes_mem_handle_t> modules(module_ଗଣନା);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_ଗଣନା, modules.data());
        ଯଦି (r != ZE_RESULT_SUCCESS || module_ଗଣନା == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }

        ପାଇଁ (uint32_t i = 0; i < module_ଗଣନା; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            ଯଦି (r != ZE_RESULT_SUCCESS) {
                ଜାରି;
            }

            free_bytes += state.ମୁକ୍ତ;
            total_bytes += state.size;
        }

        ଯଦି (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
        ଫେରନ୍ତୁ total_bytes >= free_bytes;

    } ଧର (ଅଚଳ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ଫେରନ୍ତୁ ମିଥ୍ୟା;
    }
}
#endif

ସ୍ଥିର ତାର୍କିକ get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#ଯଦି (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ଯଦି (dev.has(sycl::aspect::ext_intel_free_memory)) {
        ଚେଷ୍ଟା {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            ଫେରନ୍ତୁ ସତ୍ୟ;
        } ଧର (ଅଚଳ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    } ଅନ୍ୟଥା {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ଅନ୍ୟଥା
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    ଫେରନ୍ତୁ ମିଥ୍ୟା;
}

ତାର୍କିକ get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ଯଦି (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ଯଦି (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            ଫେରନ୍ତୁ ସତ୍ୟ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    ଯଦି(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        ଫେରନ୍ତୁ ସତ୍ୟ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    ଫେରନ୍ତୁ ସତ୍ୟ;
}
