#ضمن <sycl/ext/oneapi/backend/level_zero.hpp>
#ضمن <sycl/sycl.hpp>

#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#ضمن <level_zero/zes_api.h>
#جے_انت

#ضمن <cstdint>
#ضمن <iostream>
#ضمن <vector>

#ضمن "base.hpp"
#ضمن "mem.hpp"

ثابت حرف * mem_api_int2str(صحيح mem_api) {
    إذا (mem_api == حافظہ_API_TYPE_SYCL) {
        أرجع "SYCL API";
    } وإلا إذا (mem_api == حافظہ_API_TYPE_LEVEL_ZERO) {
        أرجع "Level Zero API";
    } وإلا {
        أرجع "Unknown";
    }
}

#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ساكن منطقي query_free_حافظہ_by_ze(sycl::device dev, size_t & free_بائٹاں, size_t & کل_بائٹاں) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_بائٹاں  = 0;
    کل_بائٹاں = 0;

    uint32_t module_گنتی = 0;

#إذا defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#وإلا
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#جے_انت

    حاول {

        إذا (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            أرجع خطأ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        إذا (ze_dev == خالی_سوچک) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            أرجع خطأ;
        }

        ze_نتیجہ_t r = zes_Device_Enum_حافظہ_Modules(ze_dev, &module_گنتی, خالی_سوچک);
        إذا (r != ZE_نتیجہ_SUCCESS || module_گنتی == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        std::vector<zes_mem_handle_t> modules(module_گنتی);
        r = zes_Device_Enum_حافظہ_Modules(ze_dev, &module_گنتی, modules.انکڑے());
        إذا (r != ZE_نتیجہ_SUCCESS || module_گنتی == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            أرجع خطأ;
        }

        لكل (uint32_t i = 0; i < module_گنتی; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_اگلا           = خالی_سوچک;

            r = zes_حافظہ_Get_State(modules[i], &state);
            إذا (r != ZE_نتیجہ_SUCCESS) {
                استمر;
            }

            free_بائٹاں += state.حرر;
            کل_بائٹاں += state.size;
        }

        إذا (کل_بائٹاں == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            أرجع خطأ;
        }
        أرجع کل_بائٹاں >= free_بائٹاں;

    } امسك (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        أرجع خطأ;
    }
}
#جے_انت

ساكن منطقي get_حافظہ_ناپ_by_sycl_api(sycl::device dev, size_t & free_بائٹاں, size_t & کل_بائٹاں) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    کل_بائٹاں = dev.get_info<sycl::info::device::global_mem_ناپ>();

#إذا (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    إذا (dev.has(sycl::aspect::ext_intel_free_حافظہ)) {
        حاول {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_بائٹاں = dev.get_info<sycl::ext::intel::info::device::free_حافظہ>();
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
#جے_انت
    أرجع خطأ;
}

منطقي get_حافظہ_ناپ(sycl::device dev, size_t & free_بائٹاں, size_t & کل_بائٹاں, حافظہ_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    إذا (api_type == حافظہ_API_TYPE_LEVEL_ZERO) {
#جے_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        إذا (query_free_حافظہ_by_ze(dev, free_بائٹاں, کل_بائٹاں)) {
            أرجع سچ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#جے_انت
    }

    //MEMORY_API_TYPE_SYCL
    إذا(get_حافظہ_ناپ_by_sycl_api(dev, free_بائٹاں, کل_بائٹاں)){
        أرجع سچ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_بائٹاں = کل_بائٹاں;

    أرجع سچ;
}
