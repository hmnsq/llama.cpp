#अन्तर्भाव <sycl/ext/oneapi/backend/level_zero.hpp>
#अन्तर्भाव <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#अन्तर्भाव <level_zero/ze_api.h>
#अन्तर्भाव <level_zero/zes_api.h>
#endif

#अन्तर्भाव <cstdint>
#अन्तर्भाव <iostream>
#अन्तर्भाव <vector>

#अन्तर्भाव "base.hpp"
#अन्तर्भाव "mem.hpp"

ध्रुव वर्ण * mem_api_int2str(पूर्णाङ्क mem_api) {
    सचे (mem_api == MEMORY_API_TYPE_SYCL) {
        पटिनिवत्त "SYCL API";
    } अञ्ञथा सचे (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        पटिनिवत्त "Level Zero API";
    } अञ्ञथा {
        पटिनिवत्त "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
स्थिर तार्किक query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_गणनम् = 0;

#सचे defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#अञ्ञथा
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    यत्न {

        सचे (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            पटिनिवत्त मिच्छा;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        सचे (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            पटिनिवत्त मिच्छा;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_गणनम्, nullptr);
        सचे (r != ZE_RESULT_SUCCESS || module_गणनम् == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            पटिनिवत्त मिच्छा;
        }

        std::vector<zes_mem_handle_t> modules(module_गणनम्);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_गणनम्, modules.data());
        सचे (r != ZE_RESULT_SUCCESS || module_गणनम् == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            पटिनिवत्त मिच्छा;
        }

        अत्थाय (uint32_t i = 0; i < module_गणनम्; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            सचे (r != ZE_RESULT_SUCCESS) {
                अनुवर्त;
            }

            free_bytes += state.मुक्त;
            total_bytes += state.size;
        }

        सचे (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            पटिनिवत्त मिच्छा;
        }
        पटिनिवत्त total_bytes >= free_bytes;

    } गृहाण (ध्रुव sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        पटिनिवत्त मिच्छा;
    }
}
#endif

स्थिर तार्किक get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#सचे (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    सचे (dev.has(sycl::aspect::ext_intel_free_memory)) {
        यत्न {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            पटिनिवत्त सच्च;
        } गृहाण (ध्रुव sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            पटिनिवत्त मिच्छा;
        }
    } अञ्ञथा {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#अञ्ञथा
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    पटिनिवत्त मिच्छा;
}

तार्किक get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    सचे (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        सचे (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            पटिनिवत्त सच्च;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    सचे(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        पटिनिवत्त सच्च;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    पटिनिवत्त सच्च;
}
