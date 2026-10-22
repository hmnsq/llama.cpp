#অন্তর্ভাৱ <sycl/ext/oneapi/backend/level_zero.hpp>
#অন্তর্ভাৱ <sycl/sycl.hpp>

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#অন্তর্ভাৱ <level_zero/ze_api.h>
#অন্তর্ভাৱ <level_zero/zes_api.h>
#endif

#অন্তর্ভাৱ <cstdint>
#অন্তর্ভাৱ <iostream>
#অন্তর্ভাৱ <vector>

#অন্তর্ভাৱ "base.hpp"
#অন্তর্ভাৱ "mem.hpp"

ধ্রুৱ ৱর্ণ * mem_api_int2str(পূর্ণাঙ্ক mem_api) {
    যদি (mem_api == MEMORY_API_TYPE_SYCL) {
        ঘূৰাই "SYCL API";
    } অন্যথা যদি (mem_api == MEMORY_API_TYPE_LEVEL_ZERO) {
        ঘূৰাই "Level Zero API";
    } অন্যথা {
        ঘূৰাই "Unknown";
    }
}

#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
স্থির তার্কিক query_free_memory_by_ze(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_bytes  = 0;
    total_bytes = 0;

    uint32_t module_গুনতি = 0;

#যদি defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#অন্যথা
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#endif

    যত্ন {

        যদি (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ঘূৰাই মিছা;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        যদি (ze_dev == nullptr) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ঘূৰাই মিছা;
        }

        ze_result_t r = zesDeviceEnumMemoryModules(ze_dev, &module_গুনতি, nullptr);
        যদি (r != ZE_RESULT_SUCCESS || module_গুনতি == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ঘূৰাই মিছা;
        }

        std::vector<zes_mem_handle_t> modules(module_গুনতি);
        r = zesDeviceEnumMemoryModules(ze_dev, &module_গুনতি, modules.data());
        যদি (r != ZE_RESULT_SUCCESS || module_গুনতি == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ঘূৰাই মিছা;
        }

        বাবে (uint32_t i = 0; i < module_গুনতি; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.pNext           = nullptr;

            r = zesMemoryGetState(modules[i], &state);
            যদি (r != ZE_RESULT_SUCCESS) {
                অনুৱর্ত;
            }

            free_bytes += state.মুক্ত;
            total_bytes += state.size;
        }

        যদি (total_bytes == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ঘূৰাই মিছা;
        }
        ঘূৰাই total_bytes >= free_bytes;

    } গৃহাণ (ধ্রুৱ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ঘূৰাই মিছা;
    }
}
#endif

স্থির তার্কিক get_memory_size_by_sycl_api(sycl::device dev, size_t & free_bytes, size_t & total_bytes) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    total_bytes = dev.get_info<sycl::info::device::global_mem_size>();

#যদি (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    যদি (dev.has(sycl::aspect::ext_intel_free_memory)) {
        যত্ন {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
            ঘূৰাই সঁচা;
        } গৃহাণ (ধ্রুৱ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ঘূৰাই মিছা;
        }
    } অন্যথা {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#অন্যথা
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#endif
    ঘূৰাই মিছা;
}

তার্কিক get_memory_size(sycl::device dev, size_t & free_bytes, size_t & total_bytes, MemoryAPIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    যদি (api_type == MEMORY_API_TYPE_LEVEL_ZERO) {
#ifdef GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        যদি (query_free_memory_by_ze(dev, free_bytes, total_bytes)) {
            ঘূৰাই সঁচা;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#endif
    }

    //MEMORY_API_TYPE_SYCL
    যদি(get_memory_size_by_sycl_api(dev, free_bytes, total_bytes)){
        ঘূৰাই সঁচা;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_bytes = total_bytes;

    ঘূৰাই সঁচা;
}
