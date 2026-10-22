#ضمن <sycl/ext/oneapi/backend/level_zero.hpp>
#ضمن <sycl/sycl.hpp>

#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#ضمن <level_zero/zes_api.h>
#د_که_پای

#ضمن <cstdint>
#ضمن <iostream>
#ضمن <vector>

#ضمن "base.hpp"
#ضمن "mem.hpp"

ثابت حرف * mem_api_int2str(صحيح mem_api) {
    إذا (mem_api == حافظه_API_TYPE_SYCL) {
        أرجع "SYCL API";
    } وإلا إذا (mem_api == حافظه_API_TYPE_LEVEL_ZERO) {
        أرجع "Level Zero API";
    } وإلا {
        أرجع "Unknown";
    }
}

#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ساكن منطقي query_free_حافظه_by_ze(sycl::device dev, size_t & free_بایټونه, size_t & ټول_بایټونه) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_بایټونه  = 0;
    ټول_بایټونه = 0;

    uint32_t module_شمېر = 0;

#إذا defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#وإلا
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#د_که_پای

    حاول {

        إذا (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            أرجع خطأ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        إذا (ze_dev == تش_اشاره) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            أرجع خطأ;
        }

        ze_پایله_t r = zes_Device_Enum_حافظه_Modules(ze_dev, &module_شمېر, تش_اشاره);
        إذا (r != ZE_پایله_SUCCESS || module_شمېر == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        std::vector<zes_mem_handle_t> modules(module_شمېر);
        r = zes_Device_Enum_حافظه_Modules(ze_dev, &module_شمېر, modules.معلومات());
        إذا (r != ZE_پایله_SUCCESS || module_شمېر == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        لكل (uint32_t i = 0; i < module_شمېر; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_بل           = تش_اشاره;

            r = zes_حافظه_Get_State(modules[i], &state);
            إذا (r != ZE_پایله_SUCCESS) {
                استمر;
            }

            free_بایټونه += state.حرر;
            ټول_بایټونه += state.size;
        }

        إذا (ټول_بایټونه == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            أرجع خطأ;
        }
        أرجع ټول_بایټونه >= free_بایټونه;

    } امسك (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        أرجع خطأ;
    }
}
#د_که_پای

ساكن منطقي get_حافظه_اندازه_by_sycl_api(sycl::device dev, size_t & free_بایټونه, size_t & ټول_بایټونه) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    ټول_بایټونه = dev.get_info<sycl::info::device::global_mem_اندازه>();

#إذا (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    إذا (dev.has(sycl::aspect::ext_intel_free_حافظه)) {
        حاول {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_بایټونه = dev.get_info<sycl::ext::intel::info::device::free_حافظه>();
            أرجع ریښتیا;
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
#د_که_پای
    أرجع خطأ;
}

منطقي get_حافظه_اندازه(sycl::device dev, size_t & free_بایټونه, size_t & ټول_بایټونه, حافظه_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    إذا (api_type == حافظه_API_TYPE_LEVEL_ZERO) {
#که_تعریف_شوی GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        إذا (query_free_حافظه_by_ze(dev, free_بایټونه, ټول_بایټونه)) {
            أرجع ریښتیا;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#د_که_پای
    }

    //MEMORY_API_TYPE_SYCL
    إذا(get_حافظه_اندازه_by_sycl_api(dev, free_بایټونه, ټول_بایټونه)){
        أرجع ریښتیا;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_بایټونه = ټول_بایټونه;

    أرجع ریښتیا;
}
