#അന്തര്ഭാവ <sycl/ext/oneapi/backend/level_zero.hpp>
#അന്തര്ഭാവ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#അന്തര്ഭാവ <level_zero/ze_api.h>
#അന്തര്ഭാവ <level_zero/zes_api.h>
#endif

#അന്തര്ഭാവ <cstdint>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <vector>

#അന്തര്ഭാവ "base.hpp"
#അന്തര്ഭാവ "mem.hpp"

സ്ഥിരം അക്ഷരം * mem_api_int2str(പൂർണ്ണസംഖ്യ mem_api) {
    എങ്കിൽ (mem_api == MEMORY_API_TYPE_SYCL) {
        തിരികെ "SYCL API";
    } അല്ലെങ്കിൽ എങ്കിൽ (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        തിരികെ "Level Zero API";
    } അല്ലെങ്കിൽ {
        തിരികെ "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
സ്ഥിതം തർക്കം query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_എണ്ണം = 0;

#എങ്കിൽ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#അല്ലെങ്കിൽ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    ശ്രമിക്കുക {

        എങ്കിൽ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            തിരികെ അസത്യം;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        എങ്കിൽ (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            തിരികെ അസത്യം;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_എണ്ണം, nullptr);
        എങ്കിൽ (r != ZE_RESULT_SUCCESS || module_എണ്ണം == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            തിരികെ അസത്യം;
        }

        std::vector<zes_mem_handle_t> modules(module_എണ്ണം);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_എണ്ണം, modules.data());
        എങ്കിൽ (r != ZE_RESULT_SUCCESS || module_എണ്ണം == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            തിരികെ അസത്യം;
        }

        വേണ്ടി (uint32_t i = 0; i < module_എണ്ണം; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            എങ്കിൽ (r != ZE_RESULT_SUCCESS) {
                തുടരുക;
            }

            free_bytes += state.സ്വതന്ത്രം;
            total_bytes += state.size;
        }

        എങ്കിൽ (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            തിരികെ അസത്യം;
        }
        തിരികെ total_bytes >= free_bytes;

    } പിടിക്കുക (സ്ഥിരം sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        തിരികെ അസത്യം;
    }
}
#endif

സ്ഥിതം തർക്കം get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#എങ്കിൽ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    എങ്കിൽ (dev.has(sycl::aspect::ext_intel_free_memory)) {
        ശ്രമിക്കുക {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            തിരികെ സത്യം;
        } പിടിക്കുക (സ്ഥിരം sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            തിരികെ അസത്യം;
        }
    } അല്ലെങ്കിൽ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#അല്ലെങ്കിൽ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    തിരികെ അസത്യം;
}

തർക്കം get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    എങ്കിൽ (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        എങ്കിൽ (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            തിരികെ സത്യം;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    എങ്കിൽ(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        തിരികെ സത്യം;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    തിരികെ സത്യം;
}
