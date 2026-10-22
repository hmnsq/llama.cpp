#ضمن <sycl/ext/oneapi/backend/level_zero.hpp>
#ضمن <sycl/sycl.hpp>

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#ضمن <level_zero/zes_api.h>
#جيڪڏهن_پورو

#ضمن <cstdint>
#ضمن <iostream>
#ضمن <vector>

#ضمن "base.hpp"
#ضمن "mem.hpp"

ثابت حرف * mem_api_int2str(صحيح mem_api) {
    إذا (mem_api == يادگيري_API_TYPE_SYCL) {
        أرجع "SYCL API";
    } وإلا إذا (mem_api == يادگيري_API_TYPE_LEVEL_ZERO) {
        أرجع "Level Zero API";
    } وإلا {
        أرجع "Unknown";
    }
}

#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ساكن منطقي query_free_يادگيري_by_ze(sycl::device dev, size_t & free_بائيٽون, size_t & ڪل_بائيٽون) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_بائيٽون  = 0;
    ڪل_بائيٽون = 0;

    uint32_t module_ڳڻپ = 0;

#إذا defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#وإلا
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#جيڪڏهن_پورو

    حاول {

        إذا (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            أرجع خطأ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        إذا (ze_dev == خالي_اشارو) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            أرجع خطأ;
        }

        ze_نتيجو_t r = zes_Device_Enum_يادگيري_Modules(ze_dev, &module_ڳڻپ, خالي_اشارو);
        إذا (r != ZE_نتيجو_SUCCESS || module_ڳڻپ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        std::vector<zes_mem_handle_t> modules(module_ڳڻپ);
        r = zes_Device_Enum_يادگيري_Modules(ze_dev, &module_ڳڻپ, modules.معلومات());
        إذا (r != ZE_نتيجو_SUCCESS || module_ڳڻپ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        لكل (uint32_t i = 0; i < module_ڳڻپ; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_ايندڙ           = خالي_اشارو;

            r = zes_يادگيري_Get_State(modules[i], &state);
            إذا (r != ZE_نتيجو_SUCCESS) {
                استمر;
            }

            free_بائيٽون += state.حرر;
            ڪل_بائيٽون += state.size;
        }

        إذا (ڪل_بائيٽون == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            أرجع خطأ;
        }
        أرجع ڪل_بائيٽون >= free_بائيٽون;

    } امسك (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        أرجع خطأ;
    }
}
#جيڪڏهن_پورو

ساكن منطقي get_يادگيري_سائيز_by_sycl_api(sycl::device dev, size_t & free_بائيٽون, size_t & ڪل_بائيٽون) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    ڪل_بائيٽون = dev.get_info<sycl::info::device::global_mem_سائيز>();

#إذا (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    إذا (dev.has(sycl::aspect::ext_intel_free_يادگيري)) {
        حاول {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_بائيٽون = dev.get_info<sycl::ext::intel::info::device::free_يادگيري>();
            أرجع سچ;
        } امسك (ثابت sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            أرجع خطأ;
        }
    } وإلا {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#وإلا
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#جيڪڏهن_پورو
    أرجع خطأ;
}

منطقي get_يادگيري_سائيز(sycl::device dev, size_t & free_بائيٽون, size_t & ڪل_بائيٽون, يادگيري_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    إذا (api_type == يادگيري_API_TYPE_LEVEL_ZERO) {
#جيڪڏهن_وصف_ٿيل GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        إذا (query_free_يادگيري_by_ze(dev, free_بائيٽون, ڪل_بائيٽون)) {
            أرجع سچ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#جيڪڏهن_پورو
    }

    //MEMORY_API_TYPE_SYCL
    إذا(get_يادگيري_سائيز_by_sycl_api(dev, free_بائيٽون, ڪل_بائيٽون)){
        أرجع سچ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_بائيٽون = ڪل_بائيٽون;

    أرجع سچ;
}
