#అన్తర్భావ <sycl/ext/oneapi/backend/level_zero.hpp>
#అన్తర్భావ <sycl/sycl.hpp>

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#అన్తర్భావ <level_zero/ze_api.h>
#అన్తర్భావ <level_zero/zes_api.h>
#అయితే_ముగింపు

#అన్తర్భావ <cstdint>
#అన్తర్భావ <iostream>
#అన్తర్భావ <vector>

#అన్తర్భావ "base.hpp"
#అన్తర్భావ "mem.hpp"

స్థిరరాశి అక్షరం * mem_api_int2str(పూర్ణాంకం mem_api) {
    ఒకవేళ (mem_api == జ్ఞాపకం_API_TYPE_SYCL) {
        తిరిగి "SYCL API";
    } లేకపోతే ఒకవేళ (mem_api == జ్ఞాపకం_API_TYPE_LEVEL_ZERO) {
        తిరిగి "Level Zero API";
    } లేకపోతే {
        తిరిగి "Unknown";
    }
}

#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
స్థిర తార్కిక query_free_జ్ఞాపకం_by_ze(sycl::device dev, size_t & free_బైట్లు, size_t & మొత్తం_బైట్లు) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_బైట్లు  = 0;
    మొత్తం_బైట్లు = 0;

    uint32_t module_లెక్క = 0;

#ఒకవేళ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#లేకపోతే
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#అయితే_ముగింపు

    ప్రయత్నించు {

        ఒకవేళ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            తిరిగి అబద్ధం;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ఒకవేళ (ze_dev == శూన్యసూచిక) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            తిరిగి అబద్ధం;
        }

        ze_ఫలితం_t r = zes_Device_Enum_జ్ఞాపకం_Modules(ze_dev, &module_లెక్క, శూన్యసూచిక);
        ఒకవేళ (r != ZE_ఫలితం_SUCCESS || module_లెక్క == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            తిరిగి అబద్ధం;
        }

        std::vector<zes_mem_handle_t> modules(module_లెక్క);
        r = zes_Device_Enum_జ్ఞాపకం_Modules(ze_dev, &module_లెక్క, modules.దత్తాంశం());
        ఒకవేళ (r != ZE_ఫలితం_SUCCESS || module_లెక్క == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            తిరిగి అబద్ధం;
        }

        కొరకు (uint32_t i = 0; i < module_లెక్క; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_తదుపరి           = శూన్యసూచిక;

            r = zes_జ్ఞాపకం_Get_State(modules[i], &state);
            ఒకవేళ (r != ZE_ఫలితం_SUCCESS) {
                కొనసాగు;
            }

            free_బైట్లు += state.విడుదల;
            మొత్తం_బైట్లు += state.size;
        }

        ఒకవేళ (మొత్తం_బైట్లు == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            తిరిగి అబద్ధం;
        }
        తిరిగి మొత్తం_బైట్లు >= free_బైట్లు;

    } పట్టుకో (స్థిరరాశి sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        తిరిగి అబద్ధం;
    }
}
#అయితే_ముగింపు

స్థిర తార్కిక get_జ్ఞాపకం_కొలత_by_sycl_api(sycl::device dev, size_t & free_బైట్లు, size_t & మొత్తం_బైట్లు) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    మొత్తం_బైట్లు = dev.get_info<sycl::info::device::global_mem_కొలత>();

#ఒకవేళ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ఒకవేళ (dev.has(sycl::aspect::ext_intel_free_జ్ఞాపకం)) {
        ప్రయత్నించు {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_బైట్లు = dev.get_info<sycl::ext::intel::info::device::free_జ్ఞాపకం>();
            తిరిగి నిజం;
        } పట్టుకో (స్థిరరాశి sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            తిరిగి అబద్ధం;
        }
    } లేకపోతే {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#లేకపోతే
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#అయితే_ముగింపు
    తిరిగి అబద్ధం;
}

తార్కిక get_జ్ఞాపకం_కొలత(sycl::device dev, size_t & free_బైట్లు, size_t & మొత్తం_బైట్లు, జ్ఞాపకం_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ఒకవేళ (api_type == జ్ఞాపకం_API_TYPE_LEVEL_ZERO) {
#నిర్వచితమైతే GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ఒకవేళ (query_free_జ్ఞాపకం_by_ze(dev, free_బైట్లు, మొత్తం_బైట్లు)) {
            తిరిగి నిజం;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#అయితే_ముగింపు
    }

    //MEMORY_API_TYPE_SYCL
    ఒకవేళ(get_జ్ఞాపకం_కొలత_by_sycl_api(dev, free_బైట్లు, మొత్తం_బైట్లు)){
        తిరిగి నిజం;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_బైట్లు = మొత్తం_బైట్లు;

    తిరిగి నిజం;
}
