#شامل <sycl/ext/oneapi/backend/level_zero.hpp>
#شامل <sycl/sycl.hpp>

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#شامل <level_zero/ze_api.h>
#شامل <level_zero/zes_api.h>
#اگر_ختم

#شامل <cstdint>
#شامل <iostream>
#شامل <vector>

#شامل "base.hpp"
#شامل "mem.hpp"

ثابت حرف * mem_api_int2str(صحیح mem_api) {
    اگر (mem_api == حافظہ_API_TYPE_SYCL) {
        واپس "SYCL API";
    } ورنہ اگر (mem_api == حافظہ_API_TYPE_LEVEL_ZERO) {
        واپس "Level Zero API";
    } ورنہ {
        واپس "Unknown";
    }
}

#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ساکن منطقی query_free_حافظہ_by_ze(sycl::device dev, size_t & free_بائٹیں, size_t & کل_بائٹیں) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_بائٹیں  = 0;
    کل_بائٹیں = 0;

    uint32_t module_گنتی = 0;

#اگر defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ورنہ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#اگر_ختم

    کوشش {

        اگر (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            واپس جھوٹ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        اگر (ze_dev == خالی_اشاریہ) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            واپس جھوٹ;
        }

        ze_نتیجہ_t r = zes_Device_Enum_حافظہ_Modules(ze_dev, &module_گنتی, خالی_اشاریہ);
        اگر (r != ZE_نتیجہ_SUCCESS || module_گنتی == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            واپس جھوٹ;
        }

        std::vector<zes_mem_handle_t> modules(module_گنتی);
        r = zes_Device_Enum_حافظہ_Modules(ze_dev, &module_گنتی, modules.ڈیٹا());
        اگر (r != ZE_نتیجہ_SUCCESS || module_گنتی == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            واپس جھوٹ;
        }

        برائے (uint32_t i = 0; i < module_گنتی; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_اگلا           = خالی_اشاریہ;

            r = zes_حافظہ_Get_State(modules[i], &state);
            اگر (r != ZE_نتیجہ_SUCCESS) {
                جاری;
            }

            free_بائٹیں += state.آزاد;
            کل_بائٹیں += state.size;
        }

        اگر (کل_بائٹیں == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            واپس جھوٹ;
        }
        واپس کل_بائٹیں >= free_بائٹیں;

    } پکڑو (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        واپس جھوٹ;
    }
}
#اگر_ختم

ساکن منطقی get_حافظہ_سائز_by_sycl_api(sycl::device dev, size_t & free_بائٹیں, size_t & کل_بائٹیں) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    کل_بائٹیں = dev.get_info<sycl::info::device::global_mem_سائز>();

#اگر (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    اگر (dev.has(sycl::aspect::ext_intel_free_حافظہ)) {
        کوشش {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_بائٹیں = dev.get_info<sycl::ext::intel::info::device::free_حافظہ>();
            واپس سچ;
        } پکڑو (ثابت sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            واپس جھوٹ;
        }
    } ورنہ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ورنہ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#اگر_ختم
    واپس جھوٹ;
}

منطقی get_حافظہ_سائز(sycl::device dev, size_t & free_بائٹیں, size_t & کل_بائٹیں, حافظہ_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    اگر (api_type == حافظہ_API_TYPE_LEVEL_ZERO) {
#اگر_تعریف_شدہ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        اگر (query_free_حافظہ_by_ze(dev, free_بائٹیں, کل_بائٹیں)) {
            واپس سچ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#اگر_ختم
    }

    //MEMORY_API_TYPE_SYCL
    اگر(get_حافظہ_سائز_by_sycl_api(dev, free_بائٹیں, کل_بائٹیں)){
        واپس سچ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_بائٹیں = کل_بائٹیں;

    واپس سچ;
}
