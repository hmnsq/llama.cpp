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
    यदि (mem_api == MEMORY_API_TYPE_SYCL) {
        प्रत्यागम "SYCL API";
    } अन्यथा यदि (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        प्रत्यागम "Level Zero API";
    } अन्यथा {
        प्रत्यागम "Unknown";
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

    uint32_t module_मोजणी = 0;

#यदि defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#अन्यथा
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    यत्न {

        यदि (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            प्रत्यागम असत्य;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        यदि (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            प्रत्यागम असत्य;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_मोजणी, nullptr);
        यदि (r != ZE_RESULT_SUCCESS || module_मोजणी == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            प्रत्यागम असत्य;
        }

        std::vector<zes_mem_handle_t> modules(module_मोजणी);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_मोजणी, modules.data());
        यदि (r != ZE_RESULT_SUCCESS || module_मोजणी == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            प्रत्यागम असत्य;
        }

        प्रति (uint32_t i = 0; i < module_मोजणी; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            यदि (r != ZE_RESULT_SUCCESS) {
                अनुवर्त;
            }

            free_bytes += state.मुक्त;
            total_bytes += state.size;
        }

        यदि (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            प्रत्यागम असत्य;
        }
        प्रत्यागम total_bytes >= free_bytes;

    } गृहाण (ध्रुव sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        प्रत्यागम असत्य;
    }
}
#endif

स्थिर तार्किक get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#यदि (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    यदि (dev.has(sycl::aspect::ext_intel_free_memory)) {
        यत्न {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            प्रत्यागम सत्य;
        } गृहाण (ध्रुव sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            प्रत्यागम असत्य;
        }
    } अन्यथा {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#अन्यथा
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    प्रत्यागम असत्य;
}

तार्किक get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    यदि (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        यदि (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            प्रत्यागम सत्य;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    यदि(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        प्रत्यागम सत्य;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    प्रत्यागम सत्य;
}
