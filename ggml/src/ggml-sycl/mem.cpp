#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/ext/oneapi/backend/level_zero.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ਅਨ੍ਤਰ੍ਭਾਵ <level_zero/ze_api.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <level_zero/zes_api.h>
#ਜੇ_ਅੰਤ

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਅਨ੍ਤਰ੍ਭਾਵ "base.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "mem.hpp"

ਅਚਲ ਅੱਖਰ * mem_api_int2str(ਪੂਰਨ mem_api) {
    ਜੇ (mem_api == ਸਮ੍ਰਿਤੀ_API_TYPE_SYCL) {
        ਵਾਪਸ "SYCL API";
    } ਨਹੀਂਤਾਂ ਜੇ (mem_api == ਸਮ੍ਰਿਤੀ_API_TYPE_LEVEL_ZERO) {
        ਵਾਪਸ "Level Zero API";
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ "Unknown";
    }
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ਸਥਿਰ ਤਾਰਕਿਕ query_free_ਸਮ੍ਰਿਤੀ_by_ze(sycl::device dev, size_t & free_ਬਾਈਟਾਂ, size_t & ਕੁੱਲ_ਬਾਈਟਾਂ) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_ਬਾਈਟਾਂ  = 0;
    ਕੁੱਲ_ਬਾਈਟਾਂ = 0;

    uint32_t module_ਗਿਣਤੀ = 0;

#ਜੇ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ਨਹੀਂਤਾਂ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#ਜੇ_ਅੰਤ

    ਕੋਸ਼ਿਸ਼ {

        ਜੇ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ਵਾਪਸ ਝੂਠ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ਜੇ (ze_dev == ਖਾਲੀ_ਸੂਚਕ) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ਵਾਪਸ ਝੂਠ;
        }

        ze_ਨਤੀਜਾ_t r = zes_Device_Enum_ਸਮ੍ਰਿਤੀ_Modules(ze_dev, &module_ਗਿਣਤੀ, ਖਾਲੀ_ਸੂਚਕ);
        ਜੇ (r != ZE_ਨਤੀਜਾ_SUCCESS || module_ਗਿਣਤੀ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ਵਾਪਸ ਝੂਠ;
        }

        std::vector<zes_mem_handle_t> modules(module_ਗਿਣਤੀ);
        r = zes_Device_Enum_ਸਮ੍ਰਿਤੀ_Modules(ze_dev, &module_ਗਿਣਤੀ, modules.ਅੰਕੜੇ());
        ਜੇ (r != ZE_ਨਤੀਜਾ_SUCCESS || module_ਗਿਣਤੀ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ਵਾਪਸ ਝੂਠ;
        }

        ਲਈ (uint32_t i = 0; i < module_ਗਿਣਤੀ; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_ਅਗਲਾ           = ਖਾਲੀ_ਸੂਚਕ;

            r = zes_ਸਮ੍ਰਿਤੀ_Get_State(modules[i], &state);
            ਜੇ (r != ZE_ਨਤੀਜਾ_SUCCESS) {
                ਜਾਰੀ;
            }

            free_ਬਾਈਟਾਂ += state.ਮੁਕਤ;
            ਕੁੱਲ_ਬਾਈਟਾਂ += state.size;
        }

        ਜੇ (ਕੁੱਲ_ਬਾਈਟਾਂ == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ਵਾਪਸ ਝੂਠ;
        }
        ਵਾਪਸ ਕੁੱਲ_ਬਾਈਟਾਂ >= free_ਬਾਈਟਾਂ;

    } ਫੜੋ (ਅਚਲ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ਵਾਪਸ ਝੂਠ;
    }
}
#ਜੇ_ਅੰਤ

ਸਥਿਰ ਤਾਰਕਿਕ get_ਸਮ੍ਰਿਤੀ_ਮਾਪ_by_sycl_api(sycl::device dev, size_t & free_ਬਾਈਟਾਂ, size_t & ਕੁੱਲ_ਬਾਈਟਾਂ) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    ਕੁੱਲ_ਬਾਈਟਾਂ = dev.get_info<sycl::info::device::global_mem_ਮਾਪ>();

#ਜੇ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ਜੇ (dev.has(sycl::aspect::ext_intel_free_ਸਮ੍ਰਿਤੀ)) {
        ਕੋਸ਼ਿਸ਼ {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_ਬਾਈਟਾਂ = dev.get_info<sycl::ext::intel::info::device::free_ਸਮ੍ਰਿਤੀ>();
            ਵਾਪਸ ਸੱਚ;
        } ਫੜੋ (ਅਚਲ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ਵਾਪਸ ਝੂਠ;
        }
    } ਨਹੀਂਤਾਂ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ਨਹੀਂਤਾਂ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#ਜੇ_ਅੰਤ
    ਵਾਪਸ ਝੂਠ;
}

ਤਾਰਕਿਕ get_ਸਮ੍ਰਿਤੀ_ਮਾਪ(sycl::device dev, size_t & free_ਬਾਈਟਾਂ, size_t & ਕੁੱਲ_ਬਾਈਟਾਂ, ਸਮ੍ਰਿਤੀ_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ਜੇ (api_type == ਸਮ੍ਰਿਤੀ_API_TYPE_LEVEL_ZERO) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ਜੇ (query_free_ਸਮ੍ਰਿਤੀ_by_ze(dev, free_ਬਾਈਟਾਂ, ਕੁੱਲ_ਬਾਈਟਾਂ)) {
            ਵਾਪਸ ਸੱਚ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#ਜੇ_ਅੰਤ
    }

    //MEMORY_API_TYPE_SYCL
    ਜੇ(get_ਸਮ੍ਰਿਤੀ_ਮਾਪ_by_sycl_api(dev, free_ਬਾਈਟਾਂ, ਕੁੱਲ_ਬਾਈਟਾਂ)){
        ਵਾਪਸ ਸੱਚ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_ਬਾਈਟਾਂ = ਕੁੱਲ_ਬਾਈਟਾਂ;

    ਵਾਪਸ ਸੱਚ;
}
