#ಅನ್ತರ್ಭಾವ <sycl/ext/oneapi/backend/level_zero.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
#ಅನ್ತರ್ಭಾವ <level_zero/ze_api.h>
#ಅನ್ತರ್ಭಾವ <level_zero/zes_api.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <vector>

#ಅನ್ತರ್ಭಾವ "base.hpp"
#ಅನ್ತರ್ಭಾವ "mem.hpp"

ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * mem_api_int2str(ಪೂರ್ಣಾಂಕ mem_api) {
    ಆದರೆ (mem_api == ಸ್ಮರಣೆ_API_TYPE_SYCL) {
        ಹಿಂತಿರುಗಿ "SYCL API";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (mem_api == ಸ್ಮರಣೆ_API_TYPE_LEVEL_ZERO) {
        ಹಿಂತಿರುಗಿ "Level Zero API";
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ "Unknown";
    }
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
/*
* Depend on to call zesInit(0) before any other Level Zero API calls, otherwise the Level Zero API calls may fail.
*/
ಸ್ಥಿರ ತಾರ್ಕಿಕ query_free_ಸ್ಮರಣೆ_by_ze(sycl::device dev, size_t & free_ಬೈಟುಗಳು, size_t & ಒಟ್ಟು_ಬೈಟುಗಳು) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using Level Zero API.\n", __func__);

    free_ಬೈಟುಗಳು  = 0;
    ಒಟ್ಟು_ಬೈಟುಗಳು = 0;

    uint32_t module_ಎಣಿಕೆ = 0;

#ಆದರೆ defined(SYCL_EXT_ONEAPI_BACKEND_LEVEL_ZERO)
    constexpr sycl::backend kL0Backend = sycl::backend::ext_oneapi_level_zero;
#ಇಲ್ಲದಿದ್ದರೆ
    constexpr sycl::backend kL0Backend = sycl::backend::level_zero;
#ಆದರೆ_ಅಂತ್ಯ

    ಪ್ರಯತ್ನಿಸು {

        ಆದರೆ (dev.get_platform().get_backend() != kL0Backend) {
            GGML_SYCL_DEBUG("Device backend is not Level Zero.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ze_device_handle_t ze_dev = sycl::get_native<kL0Backend>(dev);
        ಆದರೆ (ze_dev == ಶೂನ್ಯಸೂಚಕ) {
            GGML_SYCL_DEBUG("Level Zero device handle is null.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ze_ಫಲಿತಾಂಶ_t r = zes_Device_Enum_ಸ್ಮರಣೆ_Modules(ze_dev, &module_ಎಣಿಕೆ, ಶೂನ್ಯಸೂಚಕ);
        ಆದರೆ (r != ZE_ಫಲಿತಾಂಶ_SUCCESS || module_ಎಣಿಕೆ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        std::vector<zes_mem_handle_t> modules(module_ಎಣಿಕೆ);
        r = zes_Device_Enum_ಸ್ಮರಣೆ_Modules(ze_dev, &module_ಎಣಿಕೆ, modules.ದತ್ತಾಂಶ());
        ಆದರೆ (r != ZE_ಫಲಿತಾಂಶ_SUCCESS || module_ಎಣಿಕೆ == 0) {
            GGML_SYCL_DEBUG("Failed to enumerate Level Zero memory modules.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಗಾಗಿ (uint32_t i = 0; i < module_ಎಣಿಕೆ; ++i) {
            zes_mem_state_t state = {};
            state.stype           = ZES_STRUCTURE_TYPE_MEM_STATE;
            state.p_ಮುಂದಿನ           = ಶೂನ್ಯಸೂಚಕ;

            r = zes_ಸ್ಮರಣೆ_Get_State(modules[i], &state);
            ಆದರೆ (r != ZE_ಫಲಿತಾಂಶ_SUCCESS) {
                ಮುಂದುವರಿ;
            }

            free_ಬೈಟುಗಳು += state.ಬಿಡುಗಡೆ;
            ಒಟ್ಟು_ಬೈಟುಗಳು += state.size;
        }

        ಆದರೆ (ಒಟ್ಟು_ಬೈಟುಗಳು == 0) {
            GGML_SYCL_DEBUG("Level Zero memory query returned zero total bytes.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಹಿಂತಿರುಗಿ ಒಟ್ಟು_ಬೈಟುಗಳು >= free_ಬೈಟುಗಳು;

    } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception & e) {
        GGML_SYCL_DEBUG("Level Zero memory query failed: %s\n", e.what());
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
}
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ ತಾರ್ಕಿಕ get_ಸ್ಮರಣೆ_ಗಾತ್ರ_by_sycl_api(sycl::device dev, size_t & free_ಬೈಟುಗಳು, size_t & ಒಟ್ಟು_ಬೈಟುಗಳು) {
    GGML_SYCL_DEBUG("[SYCL] call %s: Querying free memory using SYCL API.\n", __func__);
    ಒಟ್ಟು_ಬೈಟುಗಳು = dev.get_info<sycl::info::device::global_mem_ಗಾತ್ರ>();

#ಆದರೆ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
    ಆದರೆ (dev.has(sycl::aspect::ext_intel_free_ಸ್ಮರಣೆ)) {
        ಪ್ರಯತ್ನಿಸು {
            GGML_SYCL_DEBUG("Querying free memory using SYCL aspect::ext_intel_free_memory.\n");
            free_ಬೈಟುಗಳು = dev.get_info<sycl::ext::intel::info::device::free_ಸ್ಮರಣೆ>();
            ಹಿಂತಿರುಗಿ ನಿಜ;
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ sycl::exception &) {
            GGML_SYCL_DEBUG(
                "Failed to query free memory using SYCL aspect::ext_intel_free_memory.\n");
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_SYCL_DEBUG(
            "Device does not support SYCL aspect::ext_intel_free_memory.\n");
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_SYCL_DEBUG("SYCL Compiler version is older than 20221105.\n");
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ತಾರ್ಕಿಕ get_ಸ್ಮರಣೆ_ಗಾತ್ರ(sycl::device dev, size_t & free_ಬೈಟುಗಳು, size_t & ಒಟ್ಟು_ಬೈಟುಗಳು, ಸ್ಮರಣೆ_APIType api_type) {

    GGML_SYCL_DEBUG("[%s]GPU Name:          %s\n", __func__,
        dev.get_info<sycl::info::device::name>().c_str());
    GGML_SYCL_DEBUG("[%s]GPU Vendor:        %s\n", __func__,
        dev.get_info<sycl::info::device::vendor>().c_str());

    ಆದರೆ (api_type == ಸ್ಮರಣೆ_API_TYPE_LEVEL_ZERO) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_SUPPORT_LEVEL_ZERO_API
        GGML_SYCL_DEBUG("[%s] Querying free memory using Level Zero API.\n", __func__);
        ಆದರೆ (query_free_ಸ್ಮರಣೆ_by_ze(dev, free_ಬೈಟುಗಳು, ಒಟ್ಟು_ಬೈಟುಗಳು)) {
            ಹಿಂತಿರುಗಿ ನಿಜ;
        }
        //fallback to SYCL API if Level Zero API fails
        GGML_SYCL_DEBUG("[%s] Falling back to SYCL API for memory query.\n", __func__);
#ಆದರೆ_ಅಂತ್ಯ
    }

    //MEMORY_API_TYPE_SYCL
    ಆದರೆ(get_ಸ್ಮರಣೆ_ಗಾತ್ರ_by_sycl_api(dev, free_ಬೈಟುಗಳು, ಒಟ್ಟು_ಬೈಟುಗಳು)){
        ಹಿಂತಿರುಗಿ ನಿಜ;
    }

    //Todo, fallback to other methods to get free memory size, such as using OS-specific APIs (e.g., /proc/meminfo on Linux, GlobalMemoryStatusEx on Windows, etc.)
    GGML_SYCL_DEBUG(
        "[%s] Can't get free mem size by Level Zero and SYCL API. Using total memory as free memory.\n", __func__);
    free_ಬೈಟುಗಳು = ಒಟ್ಟು_ಬೈಟುಗಳು;

    ಹಿಂತಿರುಗಿ ನಿಜ;
}
