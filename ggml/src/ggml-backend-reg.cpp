#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend-dl.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <filesystem>
#ಅನ್ತರ್ಭಾವ <memory>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <type_traits>
#ಅನ್ತರ್ಭಾವ <vector>
#ಅನ್ತರ್ಭಾವ <cctype>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
#    define WIN32_LEAN_AND_ಸರಾಸರಿ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ಅಥವಾಆದರೆ defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#ಇಲ್ಲದಿದ್ದರೆ
#    include <dlfcn.h>
#    include <unistd.h>
#ಆದರೆ_ಅಂತ್ಯ

// Backend registry
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CUDA
#ಅನ್ತರ್ಭಾವ "ggml-cuda.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_METAL
#ಅನ್ತರ್ಭಾವ "ggml-metal.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_SYCL
#ಅನ್ತರ್ಭಾವ "ggml-sycl.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_VULKAN
#ಅನ್ತರ್ಭಾವ "ggml-vulkan.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_WEBGPU
#ಅನ್ತರ್ಭಾವ "ggml-webgpu.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ZDNN
#ಅನ್ತರ್ಭಾವ "ggml-zdnn.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENCL
#ಅನ್ತರ್ಭಾವ "ggml-opencl.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_HEXAGON
#ಅನ್ತರ್ಭಾವ "ggml-hexagon.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_BLAS
#ಅನ್ತರ್ಭಾವ "ggml-blas.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_RPC
#ಅನ್ತರ್ಭಾವ "ggml-rpc.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_VIRTGPU_FRONTEND
#ಅನ್ತರ್ಭಾವ "ggml-virtgpu.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CANN
#ಅನ್ತರ್ಭಾವ "ggml-cann.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ZENDNN
#ಅನ್ತರ್ಭಾವ "ggml-zendnn.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENVINO
#ಅನ್ತರ್ಭಾವ "ggml-openvino.h"
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ET
#ಅನ್ತರ್ಭಾವ "ggml-et.h"
#ಆದರೆ_ಅಂತ್ಯ

ನಾಮಾಕಾಶ fs = std::filesystem;

ಸ್ಥಿರ std::string ಮಾರ್ಗ_str(ಸ್ಥಿರಾಂಕ fs::path & path) {
    ಪ್ರಯತ್ನಿಸು {
#ಆದರೆ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ಸ್ಥಿರಾಂಕ std::u8string u8str = path.u8string();
        ಹಿಂತಿರುಗಿ std::string(reinterpret_cast<ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *>(u8str.ದತ್ತಾಂಶ()), u8str.size());
#ಇಲ್ಲದಿದ್ದರೆ
        // C++17: u8string() returns std::string
        ಹಿಂತಿರುಗಿ path.u8string();
#ಆದರೆ_ಅಂತ್ಯ
    } ಹಿಡಿ (...) {
        ಹಿಂತಿರುಗಿ std::string();
    }
}

ರಚನೆ ggml_backend_reg_ನಮೂದು {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ರಚನೆ ggml_backend_registry {
    std::vector<ggml_backend_reg_ನಮೂದು> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CUDA
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_cuda_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_METAL
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_metal_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_SYCL
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_sycl_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_VULKAN
    // Add runtime disable check
    ಆದರೆ (getenv("GGML_DISABLE_VULKAN") == ಶೂನ್ಯಸೂಚಕ) {
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_vk_reg());
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_WEBGPU
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_webgpu_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ZDNN
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_zdnn_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_VIRTGPU_FRONTEND
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_virtgpu_reg());
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENCL
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_opencl_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ZENDNN
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_zendnn_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_HEXAGON
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_hexagon_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CANN
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_cann_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_BLAS
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_blas_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_RPC
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_rpc_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_OPENVINO
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_openvino_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_ET
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_et_reg());
#ಆದರೆ_ಅಂತ್ಯ
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_USE_CPU
        ರಿಜಿಸ್ಟರ್_backend(ggml_backend_cpu_reg());
#ಆದರೆ_ಅಂತ್ಯ
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ನಮೂದು : backends) {
            ಆದರೆ (ನಮೂದು.handle) {
                ನಮೂದು.handle.release(); // NOLINT
            }
        }
    }

    ಶೂನ್ಯ ರಿಜಿಸ್ಟರ್_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = ಶೂನ್ಯಸೂಚಕ) {
        ಆದರೆ (!reg) {
            ಹಿಂತಿರುಗಿ;
        }

        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ನಮೂದು : backends) {
            ಆದರೆ (ನಮೂದು.reg == reg) {
                ಹಿಂತಿರುಗಿ;
            }
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ಎಣಿಕೆ(reg));
#ಆದರೆ_ಅಂತ್ಯ
        backends.push_back({ reg, std::move(handle) });
        ಗಾಗಿ (size_t i = 0; i < ggml_backend_reg_dev_ಎಣಿಕೆ(reg); i++) {
            ರಿಜಿಸ್ಟರ್_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    ಶೂನ್ಯ ರಿಜಿಸ್ಟರ್_device(ggml_backend_dev_t device) {
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & dev : devices) {
            ಆದರೆ (dev == device) {
                ಹಿಂತಿರುಗಿ;
            }
        }

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#ಆದರೆ_ಅಂತ್ಯ
        devices.push_back(device);
    }

    ggml_backend_reg_t ತುಂಬು_backend(ಸ್ಥಿರಾಂಕ fs::path & path, ತಾರ್ಕಿಕ silent) {
        dl_handle_ptr handle { dl_ತುಂಬು_library(path) };
        ಆದರೆ (!handle) {
            ಆದರೆ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, ಮಾರ್ಗ_str(path).c_str(), dl_error());
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ಸ್ವಯಂಚಾಲಿತ score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ಆದರೆ (score_fn && score_fn() == 0) {
            ಆದರೆ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, ಮಾರ್ಗ_str(path).c_str());
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ಸ್ವಯಂಚಾಲಿತ backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ಆದರೆ (!backend_init_fn) {
            ಆದರೆ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, ಮಾರ್ಗ_str(path).c_str());
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ಆದರೆ (!reg || reg->api_ಆವೃತ್ತಿ != GGML_BACKEND_API_ಆವೃತ್ತಿ) {
            ಆದರೆ (!silent) {
                ಆದರೆ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, ಮಾರ್ಗ_str(path).c_str());
                } ಇಲ್ಲದಿದ್ದರೆ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, ಮಾರ್ಗ_str(path).c_str(), reg->api_ಆವೃತ್ತಿ, GGML_BACKEND_API_ಆವೃತ್ತಿ);
                }
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), ಮಾರ್ಗ_str(path).c_str());

        ರಿಜಿಸ್ಟರ್_backend(reg, std::move(handle));

        ಹಿಂತಿರುಗಿ reg;
    }

    ಶೂನ್ಯ unload_backend(ggml_backend_reg_t reg, ತಾರ್ಕಿಕ silent) {
        ಸ್ವಯಂಚಾಲಿತ it = std::find_if(backends.begin(), backends.end(),
                               [reg](ಸ್ಥಿರಾಂಕ ggml_backend_reg_ನಮೂದು & ನಮೂದು) { ಹಿಂತಿರುಗಿ ನಮೂದು.reg == reg; });

        ಆದರೆ (it == backends.end()) {
            ಆದರೆ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ಹಿಂತಿರುಗಿ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ಸ್ಥಿರ ggml_backend_registry & get_reg() {
    ಸ್ಥಿರ ggml_backend_registry reg;
    ಹಿಂತಿರುಗಿ reg;
}

// Internal API
ಶೂನ್ಯ ggml_backend_ರಿಜಿಸ್ಟರ್(ggml_backend_reg_t reg) {
    get_reg().ರಿಜಿಸ್ಟರ್_backend(reg);
}

ಶೂನ್ಯ ggml_backend_device_ರಿಜಿಸ್ಟರ್(ggml_backend_dev_t device) {
    get_reg().ರಿಜಿಸ್ಟರ್_device(device);
}

// Backend (reg) enumeration
ಸ್ಥಿರ ತಾರ್ಕಿಕ striequals(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * a, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * b) {
    ಗಾಗಿ (; *a && *b; a++, b++) {
        ಆದರೆ (std::tolower(*a) != std::tolower(*b)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ *a == *b;
}

size_t ggml_backend_reg_ಎಣಿಕೆ() {
    ಹಿಂತಿರುಗಿ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < ggml_backend_reg_ಎಣಿಕೆ());
    ಹಿಂತಿರುಗಿ get_reg().backends[ಸೂಚ್ಯಂಕ].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಗಾಗಿ (size_t i = 0; i < ggml_backend_reg_ಎಣಿಕೆ(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ಆದರೆ (striequals(ggml_backend_reg_name(reg), name)) {
            ಹಿಂತಿರುಗಿ reg;
        }
    }
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

// Device enumeration
size_t ggml_backend_dev_ಎಣಿಕೆ() {
    ಹಿಂತಿರುಗಿ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t ಸೂಚ್ಯಂಕ) {
    GGML_ASSERT(ಸೂಚ್ಯಂಕ < ggml_backend_dev_ಎಣಿಕೆ());
    ಹಿಂತಿರುಗಿ get_reg().devices[ಸೂಚ್ಯಂಕ];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    ಗಾಗಿ (size_t i = 0; i < ggml_backend_dev_ಎಣಿಕೆ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ಆದರೆ (striequals(ggml_backend_dev_name(dev), name)) {
            ಹಿಂತಿರುಗಿ dev;
        }
    }
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ಎಣಿಕೆ ggml_backend_dev_type type) {
    ಗಾಗಿ (size_t i = 0; i < ggml_backend_dev_ಎಣಿಕೆ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ಆದರೆ (ggml_backend_dev_type(dev) == type) {
            ಹಿಂತಿರುಗಿ dev;
        }
    }
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ಆದರೆ (!dev) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ಎಣಿಕೆ ggml_backend_dev_type type, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ಆದರೆ (!dev) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_ಉತ್ತಮ(ಶೂನ್ಯ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ಆದರೆ (!dev) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ggml_backend_dev_init(dev, ಶೂನ್ಯಸೂಚಕ);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_ತುಂಬು(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * path) {
    ಹಿಂತಿರುಗಿ get_reg().ತುಂಬು_backend(path, ಸುಳ್ಳು);
}

ಶೂನ್ಯ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, ನಿಜ);
}

ಸ್ಥಿರ fs::path get_executable_ಮಾರ್ಗ() {
#ಆದರೆ defined(__APPLE__)
    // get executable path
    std::vector<ಅಕ್ಷರ> path;
    uint32_t size;
    ಆಗ (ನಿಜ) {
        size = path.size();
        ಆದರೆ (NSGet_Executable_ಮಾರ್ಗ(path.ದತ್ತಾಂಶ(), &size) == 0) {
            ನಿಲ್ಲಿಸು;
        }
        path.resize(size);
    }
    std::string ತಳ_ಮಾರ್ಗ(path.ದತ್ತಾಂಶ(), size);
    // remove executable name
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_slash = ತಳ_ಮಾರ್ಗ.find_last_of('/');
    ಆದರೆ (ಕೊನೆಯ_slash != std::string::npos) {
        ತಳ_ಮಾರ್ಗ = ತಳ_ಮಾರ್ಗ.substr(0, ಕೊನೆಯ_slash);
    }
    ಹಿಂತಿರುಗಿ ತಳ_ಮಾರ್ಗ + "/";
#ಅಥವಾಆದರೆ defined(__linux__) || defined(__FreeBSD__)
    std::string ತಳ_ಮಾರ್ಗ = ".";
    std::vector<ಅಕ್ಷರ> path(1024);
    ಆಗ (ನಿಜ) {
        // get executable path
#    ಆದರೆ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.ದತ್ತಾಂಶ(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.ದತ್ತಾಂಶ(), path.size());
#    endif
        ಆದರೆ (len == -1) {
            ನಿಲ್ಲಿಸು;
        }
        ಆದರೆ (len < (ssize_t) path.size()) {
            ತಳ_ಮಾರ್ಗ = std::string(path.ದತ್ತಾಂಶ(), len);
            // remove executable name
            ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_slash = ತಳ_ಮಾರ್ಗ.find_last_of('/');
            ಆದರೆ (ಕೊನೆಯ_slash != std::string::npos) {
                ತಳ_ಮಾರ್ಗ = ತಳ_ಮಾರ್ಗ.substr(0, ಕೊನೆಯ_slash);
            }
            ನಿಲ್ಲಿಸು;
        }
        path.resize(path.size() * 2);
    }

    ಹಿಂತಿರುಗಿ ತಳ_ಮಾರ್ಗ + "/";
#ಅಥವಾಆದರೆ defined(_WIN32)
    std::vector<wchar_t> path(MAX_ಮಾರ್ಗ);
    DWORD len = GetModuleFileNameW(ಖಾಲಿ, path.ದತ್ತಾಂಶ(), path.size());
    ಆದರೆ (len == 0) {
        ಹಿಂತಿರುಗಿ {};
    }
    std::wstring ತಳ_ಮಾರ್ಗ(path.ದತ್ತಾಂಶ(), len);
    // remove executable name
    ಸ್ವಯಂಚಾಲಿತ ಕೊನೆಯ_slash = ತಳ_ಮಾರ್ಗ.find_last_of('\\');
    ಆದರೆ (ಕೊನೆಯ_slash != std::string::npos) {
        ತಳ_ಮಾರ್ಗ = ತಳ_ಮಾರ್ಗ.substr(0, ಕೊನೆಯ_slash);
    }
    ಹಿಂತಿರುಗಿ ತಳ_ಮಾರ್ಗ + L"\\";
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ {};
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ fs::path backend_filename_ಪೂರ್ವಪ್ರತ್ಯಯ() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ fs::u8path("ggml-");
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ fs::u8path("libggml-");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ fs::path backend_filename_extension() {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ _WIN32
    ಹಿಂತಿರುಗಿ fs::u8path(".dll");
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ fs::u8path(".so");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ ggml_backend_reg_t ggml_backend_ತುಂಬು_ಉತ್ತಮ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name, ತಾರ್ಕಿಕ silent, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * user_ಹುಡುಕು_ಮಾರ್ಗ) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ಸ್ಥಿರಾಂಕ fs::path name_ಮಾರ್ಗ = fs::u8path(name);
    ಸ್ಥಿರಾಂಕ fs::path file_ಪೂರ್ವಪ್ರತ್ಯಯ = backend_filename_ಪೂರ್ವಪ್ರತ್ಯಯ().native() + name_ಮಾರ್ಗ.native() + fs::u8path("-").native();
    ಸ್ಥಿರಾಂಕ fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> ಹುಡುಕು_paths;
    ಆದರೆ (user_ಹುಡುಕು_ಮಾರ್ಗ == ಶೂನ್ಯಸೂಚಕ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_BACKEND_DIR
        ಹುಡುಕು_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#ಆದರೆ_ಅಂತ್ಯ
        // default search paths: executable directory, current directory
        ಹುಡುಕು_paths.push_back(get_executable_ಮಾರ್ಗ());
        std::error_code cwd_ec;
        ಸ್ಥಿರಾಂಕ fs::path cwd = fs::current_path(cwd_ec);
        ಆದರೆ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.ಸಂದೇಶ().c_str());
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಹುಡುಕು_paths.push_back(cwd);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹುಡುಕು_paths.push_back(fs::u8path(user_ಹುಡುಕು_ಮಾರ್ಗ));
    }

    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_score = 0;
    fs::path ಉತ್ತಮ_ಮಾರ್ಗ;
    std::error_code ec;

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಹುಡುಕು_ಮಾರ್ಗ : ಹುಡುಕು_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(ಹುಡುಕು_ಮಾರ್ಗ, dir_ec);

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ನಮೂದು = *dir_it;
            ಆದರೆ (ನಮೂದು.is_regular_file(ec)) {
                ಸ್ವಯಂಚಾಲಿತ filename = ನಮೂದು.path().filename();
                ಸ್ವಯಂಚಾಲಿತ ext = ನಮೂದು.path().extension();
                ಆದರೆ (filename.native().find(file_ಪೂರ್ವಪ್ರತ್ಯಯ) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_ತುಂಬು_library(ನಮೂದು) };
                    ಆದರೆ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, ಮಾರ್ಗ_str(ನಮೂದು.path()).c_str(), dl_error());
                    }
                    ಆದರೆ (handle) {
                        ಸ್ವಯಂಚಾಲಿತ score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ಆದರೆ (score_fn) {
                            ಪೂರ್ಣಾಂಕ s = score_fn();
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, ಮಾರ್ಗ_str(ನಮೂದು.path()).c_str(), s);
#ಆದರೆ_ಅಂತ್ಯ
                            ಆದರೆ (s > ಉತ್ತಮ_score) {
                                ಉತ್ತಮ_score = s;
                                ಉತ್ತಮ_ಮಾರ್ಗ = ನಮೂದು.path();
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಆದರೆ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, ಮಾರ್ಗ_str(ನಮೂದು.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ಆದರೆ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, ಮಾರ್ಗ_str(ಹುಡುಕು_ಮಾರ್ಗ).c_str(), dir_ec.ಸಂದೇಶ().c_str());
        }
    }

    ಆದರೆ (ಉತ್ತಮ_score == 0) {
        // try to load the base backend
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & ಹುಡುಕು_ಮಾರ್ಗ : ಹುಡುಕು_paths) {
            fs::path filename = backend_filename_ಪೂರ್ವಪ್ರತ್ಯಯ().native() + name_ಮಾರ್ಗ.native() + backend_filename_extension().native();
            fs::path path = ಹುಡುಕು_ಮಾರ್ಗ / filename;
            ಆದರೆ (std::error_code ec; fs::exists(path, ec)) {
                ಹಿಂತಿರುಗಿ get_reg().ತುಂಬು_backend(path, silent);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, ಮಾರ್ಗ_str(path).c_str(), ec.ಸಂದೇಶ().c_str());
                }
            }
        }
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ get_reg().ತುಂಬು_backend(ಉತ್ತಮ_ಮಾರ್ಗ, silent);
}

ಶೂನ್ಯ ggml_backend_ತುಂಬು_all() {
    ggml_backend_ತುಂಬು_all_from_ಮಾರ್ಗ(ಶೂನ್ಯಸೂಚಕ);
}

ಶೂನ್ಯ ggml_backend_ತುಂಬು_all_from_ಮಾರ್ಗ(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * dir_ಮಾರ್ಗ) {
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ NDEBUG
    ತಾರ್ಕಿಕ silent = ನಿಜ;
#ಇಲ್ಲದಿದ್ದರೆ
    ತಾರ್ಕಿಕ silent = ಸುಳ್ಳು;
#ಆದರೆ_ಅಂತ್ಯ

    ggml_backend_ತುಂಬು_ಉತ್ತಮ("blas", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("zendnn", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("cann", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("cuda", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("hip", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("metal", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("rpc", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("sycl", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("vulkan", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("virtgpu", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("opencl", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("hexagon", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("musa", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("openvino", silent, dir_ಮಾರ್ಗ);
    ggml_backend_ತುಂಬು_ಉತ್ತಮ("cpu", silent, dir_ಮಾರ್ಗ);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * backend_ಮಾರ್ಗ = std::getenv("GGML_BACKEND_PATH");
    ಆದರೆ (backend_ಮಾರ್ಗ) {
        ggml_backend_ತುಂಬು(backend_ಮಾರ್ಗ);
    }
}
