#અન્તર્ભાવ <sycl/ext/oneapi/backend/level_zero.hpp>
#અન્તર્ભાવ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#અન્તર્ભાવ <level_zero/ze_api.h>
#અન્તર્ભાવ <level_zero/zes_api.h>
#endif

#અન્તર્ભાવ <cstdint>
#અન્તર્ભાવ <iostream>
#અન્તર્ભાવ <vector>

#અન્તર્ભાવ "base.hpp"
#અન્તર્ભાવ "mem.hpp"

અચલ અક્ષર * mem_api_int2str(પૂર્ણાંક mem_api) {
    જો (mem_api == MEMORY_API_TYPE_SYCL) {
        પાછા "SYCL API";
    } નહીંતર જો (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        પાછા "Level Zero API";
    } નહીંતર {
        પાછા "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
સ્થિર તાર્કિક query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_ગણતરી = 0;

#જો defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#નહીંતર
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    પ્રયાસ {

        જો (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            પાછા ખોટું;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        જો (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            પાછા ખોટું;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_ગણતરી, nullptr);
        જો (r != ZE_RESULT_SUCCESS || module_ગણતરી == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            પાછા ખોટું;
        }

        std::vector<zes_mem_handle_t> modules(module_ગણતરી);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_ગણતરી, modules.data());
        જો (r != ZE_RESULT_SUCCESS || module_ગણતરી == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            પાછા ખોટું;
        }

        માટે (uint32_t i = 0; i < module_ગણતરી; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            જો (r != ZE_RESULT_SUCCESS) {
                ચાલુ;
            }

            free_bytes += state.મુક્ત;
            total_bytes += state.size;
        }

        જો (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            પાછા ખોટું;
        }
        પાછા total_bytes >= free_bytes;

    } પકડો (અચલ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        પાછા ખોટું;
    }
}
#endif

સ્થિર તાર્કિક get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#જો (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    જો (dev.has(sycl::aspect::ext_intel_free_memory)) {
        પ્રયાસ {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            પાછા સાચું;
        } પકડો (અચલ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            પાછા ખોટું;
        }
    } નહીંતર {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#નહીંતર
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    પાછા ખોટું;
}

તાર્કિક get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    જો (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        જો (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            પાછા સાચું;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    જો(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        પાછા સાચું;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    પાછા સાચું;
}
