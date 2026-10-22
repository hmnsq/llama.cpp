#समावेश <sycl/ext/oneapi/backend/level_zero.hpp>
#समावेश <sycl/sycl.hpp>

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#समावेश <level_zero/ze_api.h>
#समावेश <level_zero/zes_api.h>
#पूर्ण_अगर

#समावेश <cstdint>
#समावेश <iostream>
#समावेश <vector>

#समावेश "base.hpp"
#समावेश "mem.hpp"

स्थिर अक्षर * mem_api_int2str(पूर्णांक mem_api) {
    अगर (mem_api == MEMORY_API_TYPE_SYCL) {
        वापस "SYCL API";
    } अन्यथा अगर (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        वापस "Level Zero API";
    } अन्यथा {
        वापस "Unknown";
    }
}

#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
अटल bool query_free_memory_by_ze(sycl::device dev, माप_प्रकार & free_bytes, माप_प्रकार & योग_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    योग_bytes = 0;

    uint32_t module_गिनती = 0;

#अगर defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#अन्यथा
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#पूर्ण_अगर

    प्रयास {

        अगर (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            वापस false;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        अगर (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            वापस false;
        }

        ze_परिणाम_t r = zesDeviceEnumMemoryModules(ze_dev, &module_गिनती, nullptr);
        अगर (r != ZE_परिणाम_SUCCESS || module_गिनती == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            वापस false;
        }

        std::vector<zes_mem_handle_t> modules(module_गिनती);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_गिनती, modules.तथ्य());
        अगर (r != ZE_परिणाम_SUCCESS || module_गिनती == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            वापस false;
        }

        क्रम (uint32_t i = 0; i < module_गिनती; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_अग्र           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            अगर (r != ZE_परिणाम_SUCCESS) {
                जारी;
            }

            free_bytes += state.मुक्त;
            योग_bytes += state.size;
        }

        अगर (योग_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            वापस false;
        }
        वापस योग_bytes >= free_bytes;

    } पकड़ो (स्थिर sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        वापस false;
    }
}
#पूर्ण_अगर

अटल bool get_memory_आकार_by_sycl_api(sycl::device dev, माप_प्रकार & free_bytes, माप_प्रकार & योग_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    योग_bytes = dev.get_info<sycl::info::device::global_mem_आकार>();

#अगर (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    अगर (dev.has(sycl::aspect::ext_intel_free_memory)) {
        प्रयास {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            वापस true;
        } पकड़ो (स्थिर sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            वापस false;
        }
    } अन्यथा {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#अन्यथा
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#पूर्ण_अगर
    वापस false;
}

bool get_memory_आकार(sycl::device dev, माप_प्रकार & free_bytes, माप_प्रकार & योग_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    अगर (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#अगर_घोषित GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        अगर (query_free_memory_by_ze(dev, free_bytes, योग_bytes)) {
            वापस true;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#पूर्ण_अगर
    }

    //MEMORY_API_TYPE_SYCL
    अगर(get_memory_आकार_by_sycl_api(dev, free_bytes, योग_bytes)){
        वापस true;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = योग_bytes;

    वापस true;
}
