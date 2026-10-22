#அந்தர்பாவ <sycl/ext/oneapi/backend/level_zero.hpp>
#அந்தர்பாவ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#அந்தர்பாவ <level_zero/ze_api.h>
#அந்தர்பாவ <level_zero/zes_api.h>
#endif

#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <iostream>
#அந்தர்பாவ <vector>

#அந்தர்பாவ "base.hpp"
#அந்தர்பாவ "mem.hpp"

மாறிலி எழுத்து * mem_api_int2str(முழு mem_api) {
    எனில் (mem_api == MEMORY_API_TYPE_SYCL) {
        திரும்பு "SYCL API";
    } இல்லையெனில் எனில் (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        திரும்பு "Level Zero API";
    } இல்லையெனில் {
        திரும்பு "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
நிலையான தருக்க query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_மொத்தம் = 0;

#எனில் defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#இல்லையெனில்
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    முயல் {

        எனில் (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            திரும்பு பொய்;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        எனில் (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            திரும்பு பொய்;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_மொத்தம், nullptr);
        எனில் (r != ZE_RESULT_SUCCESS || module_மொத்தம் == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            திரும்பு பொய்;
        }

        std::vector<zes_mem_handle_t> modules(module_மொத்தம்);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_மொத்தம், modules.data());
        எனில் (r != ZE_RESULT_SUCCESS || module_மொத்தம் == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            திரும்பு பொய்;
        }

        க்கு (uint32_t i = 0; i < module_மொத்தம்; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            எனில் (r != ZE_RESULT_SUCCESS) {
                தொடர்;
            }

            free_bytes += state.விடுவி;
            total_bytes += state.size;
        }

        எனில் (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            திரும்பு பொய்;
        }
        திரும்பு total_bytes >= free_bytes;

    } பிடி (மாறிலி sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        திரும்பு பொய்;
    }
}
#endif

நிலையான தருக்க get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#எனில் (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    எனில் (dev.has(sycl::aspect::ext_intel_free_memory)) {
        முயல் {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            திரும்பு உண்மை;
        } பிடி (மாறிலி sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            திரும்பு பொய்;
        }
    } இல்லையெனில் {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#இல்லையெனில்
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    திரும்பு பொய்;
}

தருக்க get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    எனில் (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        எனில் (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            திரும்பு உண்மை;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    எனில்(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        திரும்பு உண்மை;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    திரும்பு உண்மை;
}
