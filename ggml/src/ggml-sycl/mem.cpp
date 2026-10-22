#شامل <sycl/ext/oneapi/backend/level_zero.hpp>
#شامل <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#شامل <level_zero/ze_api.h>
#شامل <level_zero/zes_api.h>
#endif

#شامل <cstdint>
#شامل <iostream>
#شامل <vector>

#شامل "base.hpp"
#شامل "mem.hpp"

ثابت نویسه * mem_api_int2str(صحیح mem_api) {
    اگر (mem_api == MEMORY_API_TYPE_SYCL) {
        بازگشت "SYCL API";
    } وگرنه اگر (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        بازگشت "Level Zero API";
    } وگرنه {
        بازگشت "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ایستا منطقی query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_شمار = 0;

#اگر defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#وگرنه
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    تلاش {

        اگر (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            بازگشت نادرست;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        اگر (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            بازگشت نادرست;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_شمار, nullptr);
        اگر (r != ZE_RESULT_SUCCESS || module_شمار == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            بازگشت نادرست;
        }

        std::vector<zes_mem_handle_t> modules(module_شمار);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_شمار, modules.data());
        اگر (r != ZE_RESULT_SUCCESS || module_شمار == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            بازگشت نادرست;
        }

        برای (uint32_t i = 0; i < module_شمار; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            اگر (r != ZE_RESULT_SUCCESS) {
                ادامه;
            }

            free_bytes += state.آزاد;
            total_bytes += state.size;
        }

        اگر (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            بازگشت نادرست;
        }
        بازگشت total_bytes >= free_bytes;

    } بگیر (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        بازگشت نادرست;
    }
}
#endif

ایستا منطقی get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#اگر (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    اگر (dev.has(sycl::aspect::ext_intel_free_memory)) {
        تلاش {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            بازگشت درست;
        } بگیر (ثابت sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            بازگشت نادرست;
        }
    } وگرنه {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#وگرنه
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    بازگشت نادرست;
}

منطقی get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    اگر (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        اگر (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            بازگشت درست;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    اگر(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        بازگشت درست;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    بازگشت درست;
}
