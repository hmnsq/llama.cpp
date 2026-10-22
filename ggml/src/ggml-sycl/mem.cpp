#अन्तर्भाव <sycl/ext/oneapi/backend/level_zero.hpp>
#अन्तर्भाव <sycl/sycl.hpp>

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#अन्तर्भाव <level_zero/ze_api.h>
#अन्तर्भाव <level_zero/zes_api.h>
#यदि_अन्तः

#अन्तर्भाव <cstdint>
#अन्तर्भाव <iostream>
#अन्तर्भाव <vector>

#अन्तर्भाव "base.hpp"
#अन्तर्भाव "mem.hpp"

ध्रुव वर्ण * mem_api_int2str(पूर्णाङ्क mem_api) {
    यदि (mem_api == स्मृतिः_API_TYPE_SYCL) {
        प्रत्यागम "SYCL API";
    } अन्यथा यदि (mem_api == स्मृतिः_API_TYPE_LEVEL_ZERO) {
        प्रत्यागम "Level Zero API";
    } अन्यथा {
        प्रत्यागम "Unknown";
    }
}

#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
स्थिर तार्किक query_free_स्मृतिः_by_ze(sycl::device dev, size_t & free_अष्टकानि, size_t & योग_अष्टकानि) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_अष्टकानि  = 0;
    योग_अष्टकानि = 0;

    uint32_t module_गणनम् = 0;

#यदि defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#अन्यथा
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#यदि_अन्तः

    यत्न {

        यदि (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            प्रत्यागम असत्य;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        यदि (ze_dev == शून्यसूचकः) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            प्रत्यागम असत्य;
        }

        ze_परिणाम_t r = zes_Device_Enum_स्मृतिः_Modules(ze_dev, &module_गणनम्, शून्यसूचकः);
        यदि (r != ZE_परिणाम_SUCCESS || module_गणनम् == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            प्रत्यागम असत्य;
        }

        std::vector<zes_mem_handle_t> modules(module_गणनम्);
        r = zes_Device_Enum_स्मृतिः_Modules(ze_dev, &module_गणनम्, modules.दत्त());
        यदि (r != ZE_परिणाम_SUCCESS || module_गणनम् == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            प्रत्यागम असत्य;
        }

        प्रति (uint32_t i = 0; i < module_गणनम्; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_अग्र           = शून्यसूचकः;

            r = zes_स्मृतिः_Get_State(modules[i], &state);
            यदि (r != ZE_परिणाम_SUCCESS) {
                अनुवर्त;
            }

            free_अष्टकानि += state.मुक्त;
            योग_अष्टकानि += state.size;
        }

        यदि (योग_अष्टकानि == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            प्रत्यागम असत्य;
        }
        प्रत्यागम योग_अष्टकानि >= free_अष्टकानि;

    } गृहाण (ध्रुव sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        प्रत्यागम असत्य;
    }
}
#यदि_अन्तः

स्थिर तार्किक get_स्मृतिः_आकार_by_sycl_api(sycl::device dev, size_t & free_अष्टकानि, size_t & योग_अष्टकानि) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    योग_अष्टकानि = dev.get_info<sycl::info::device::global_mem_आकार>();

#यदि (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    यदि (dev.has(sycl::aspect::ext_intel_free_स्मृतिः)) {
        यत्न {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_अष्टकानि = dev.get_info<sycl::ext::intel::info::device::free_स्मृतिः>();
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
#यदि_अन्तः
    प्रत्यागम असत्य;
}

तार्किक get_स्मृतिः_आकार(sycl::device dev, size_t & free_अष्टकानि, size_t & योग_अष्टकानि, स्मृतिः_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    यदि (api_type == स्मृतिः_API_TYPE_LEVEL_ZERO) {
#यदि_परिभाषितम् GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        यदि (query_free_स्मृतिः_by_ze(dev, free_अष्टकानि, योग_अष्टकानि)) {
            प्रत्यागम सत्य;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#यदि_अन्तः
    }

    //MEMORY_API_TYPE_SYCL
    यदि(get_स्मृतिः_आकार_by_sycl_api(dev, free_अष्टकानि, योग_अष्टकानि)){
        प्रत्यागम सत्य;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_अष्टकानि = योग_अष्टकानि;

    प्रत्यागम सत्य;
}
