#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/backend/level_zero.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ಅನ್ತರ್ಭಾವ <level_zero/ze_api.h>
#ಅನ್ತರ್ಭಾವ <level_zero/zes_api.h>
#endif

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <vector>

#ಅನ್ತರ್ಭಾವ "base.hpp"
#ಅನ್ತರ್ಭಾವ "mem.hpp"

ಧ್ರುವ ವರ್ಣ * mem_api_int2str(ಪೂರ್ಣಾಙ್ಕ mem_api) {
    ಯದಿ (mem_api == MEMORY_API_TYPE_SYCL) {
        ಪ್ರತ್ಯಾಗಮ "SYCL API";
    } ಅನ್ಯಥಾ ಯದಿ (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        ಪ್ರತ್ಯಾಗಮ "Level Zero API";
    } ಅನ್ಯಥಾ {
        ಪ್ರತ್ಯಾಗಮ "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ಸ್ಥಿರ ತಾರ್ಕಿಕ query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_ಎಣಿಕೆ = 0;

#ಯದಿ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ಅನ್ಯಥಾ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    ಯತ್ನ {

        ಯದಿ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ಯದಿ (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_ಎಣಿಕೆ, nullptr);
        ಯದಿ (r != ZE_RESULT_SUCCESS || module_ಎಣಿಕೆ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        std::vector<zes_mem_handle_t> modules(module_ಎಣಿಕೆ);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_ಎಣಿಕೆ, modules.data());
        ಯದಿ (r != ZE_RESULT_SUCCESS || module_ಎಣಿಕೆ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }

        ಪ್ರತಿ (uint32_t i = 0; i < module_ಎಣಿಕೆ; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            ಯದಿ (r != ZE_RESULT_SUCCESS) {
                ಅನುವರ್ತ;
            }

            free_bytes += state.ಮುಕ್ತ;
            total_bytes += state.size;
        }

        ಯದಿ (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
        ಪ್ರತ್ಯಾಗಮ total_bytes >= free_bytes;

    } ಗೃಹಾಣ (ಧ್ರುವ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
    }
}
#endif

ಸ್ಥಿರ ತಾರ್ಕಿಕ get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#ಯದಿ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ಯದಿ (dev.has(sycl::aspect::ext_intel_free_memory)) {
        ಯತ್ನ {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        } ಗೃಹಾಣ (ಧ್ರುವ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    } ಅನ್ಯಥಾ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ಅನ್ಯಥಾ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
}

ತಾರ್ಕಿಕ get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ಯದಿ (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ಯದಿ (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    ಯದಿ(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
}
