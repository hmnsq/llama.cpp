#ضمن <sycl/ext/oneapi/backend/level_zero.hpp>
#ضمن <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ضمن <level_zero/ze_api.h>
#ضمن <level_zero/zes_api.h>
#endif

#ضمن <cstdint>
#ضمن <iostream>
#ضمن <vector>

#ضمن "base.hpp"
#ضمن "mem.hpp"

ثابت حرف * mem_api_int2str(صحيح mem_api) {
    ܐܢ (mem_api == MEMORY_API_TYPE_SYCL) {
        ܦܢܐ "SYCL API";
    } ܐܚܪܢܐ ܐܢ (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        ܦܢܐ "Level Zero API";
    } ܐܚܪܢܐ {
        ܦܢܐ "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ساكن منطقي query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_count = 0;

#ܐܢ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ܐܚܪܢܐ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    حاول {

        ܐܢ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ܦܢܐ ܕܓܠܐ;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ܐܢ (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ܦܢܐ ܕܓܠܐ;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_count, nullptr);
        ܐܢ (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ܦܢܐ ܕܓܠܐ;
        }

        std::vector<zes_mem_handle_t> modules(module_count);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_count, modules.data());
        ܐܢ (r != ZE_RESULT_SUCCESS || module_count == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ܦܢܐ ܕܓܠܐ;
        }

        لكل (uint32_t i = 0; i < module_count; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            ܐܢ (r != ZE_RESULT_SUCCESS) {
                استمر;
            }

            free_bytes += state.حرر;
            total_bytes += state.size;
        }

        ܐܢ (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ܦܢܐ ܕܓܠܐ;
        }
        ܦܢܐ total_bytes >= free_bytes;

    } امسك (ثابت sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ܦܢܐ ܕܓܠܐ;
    }
}
#endif

ساكن منطقي get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#ܐܢ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ܐܢ (dev.has(sycl::aspect::ext_intel_free_memory)) {
        حاول {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            ܦܢܐ ܫܪܪܐ;
        } امسك (ثابت sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ܦܢܐ ܕܓܠܐ;
        }
    } ܐܚܪܢܐ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ܐܚܪܢܐ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    ܦܢܐ ܕܓܠܐ;
}

منطقي get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ܐܢ (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ܐܢ (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            ܦܢܐ ܫܪܪܐ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    ܐܢ(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        ܦܢܐ ܫܪܪܐ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    ܦܢܐ ܫܪܪܐ;
}
