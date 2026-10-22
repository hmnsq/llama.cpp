#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-dl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <filesystem>
#ਅਨ੍ਤਰ੍ਭਾਵ <memory>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>
#ਅਨ੍ਤਰ੍ਭਾਵ <cctype>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
#    define WIN32_LEAN_AND_ਔਸਤ
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ਜਾਂਜੇ defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#ਨਹੀਂਤਾਂ
#    include <dlfcn.h>
#    include <unistd.h>
#ਜੇ_ਅੰਤ

// Backend registry
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CUDA
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cuda.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_METAL
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_SYCL
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-sycl.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_VULKAN
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-vulkan.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_WEBGPU
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-webgpu.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ZDNN
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zdnn.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENCL
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-opencl.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_HEXAGON
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-hexagon.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_BLAS
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-blas.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_RPC
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-rpc.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_VIRTGPU_FRONTEND
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-virtgpu.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CANN
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cann.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ZENDNN
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-zendnn.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENVINO
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-openvino.h"
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ET
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et.h"
#ਜੇ_ਅੰਤ

ਨਾਮਾਕਾਸ਼ fs = std::filesystem;

ਸਥਿਰ std::string ਰਸਤਾ_str(ਅਚਲ fs::path & path) {
    ਕੋਸ਼ਿਸ਼ {
#ਜੇ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ਅਚਲ std::u8string u8str = path.u8string();
        ਵਾਪਸ std::string(reinterpret_cast<ਅਚਲ ਅੱਖਰ *>(u8str.ਅੰਕੜੇ()), u8str.size());
#ਨਹੀਂਤਾਂ
        // C++17: u8string() returns std::string
        ਵਾਪਸ path.u8string();
#ਜੇ_ਅੰਤ
    } ਫੜੋ (...) {
        ਵਾਪਸ std::string();
    }
}

ਬਣਤਰ ggml_backend_reg_ਇੰਦਰਾਜ {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ਬਣਤਰ ggml_backend_registry {
    std::vector<ggml_backend_reg_ਇੰਦਰਾਜ> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CUDA
        ਰਜਿਸਟਰ_backend(ggml_backend_cuda_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_METAL
        ਰਜਿਸਟਰ_backend(ggml_backend_metal_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_SYCL
        ਰਜਿਸਟਰ_backend(ggml_backend_sycl_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_VULKAN
    // Add runtime disable check
    ਜੇ (getenv("GGML_DISABLE_VULKAN") == ਖਾਲੀ_ਸੂਚਕ) {
        ਰਜਿਸਟਰ_backend(ggml_backend_vk_reg());
    } ਨਹੀਂਤਾਂ {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_WEBGPU
        ਰਜਿਸਟਰ_backend(ggml_backend_webgpu_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ZDNN
        ਰਜਿਸਟਰ_backend(ggml_backend_zdnn_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_VIRTGPU_FRONTEND
        ਰਜਿਸਟਰ_backend(ggml_backend_virtgpu_reg());
#ਜੇ_ਅੰਤ

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENCL
        ਰਜਿਸਟਰ_backend(ggml_backend_opencl_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ZENDNN
        ਰਜਿਸਟਰ_backend(ggml_backend_zendnn_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_HEXAGON
        ਰਜਿਸਟਰ_backend(ggml_backend_hexagon_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CANN
        ਰਜਿਸਟਰ_backend(ggml_backend_cann_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_BLAS
        ਰਜਿਸਟਰ_backend(ggml_backend_blas_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_RPC
        ਰਜਿਸਟਰ_backend(ggml_backend_rpc_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_OPENVINO
        ਰਜਿਸਟਰ_backend(ggml_backend_openvino_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_ET
        ਰਜਿਸਟਰ_backend(ggml_backend_et_reg());
#ਜੇ_ਅੰਤ
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_USE_CPU
        ਰਜਿਸਟਰ_backend(ggml_backend_cpu_reg());
#ਜੇ_ਅੰਤ
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        ਲਈ (ਸਵੈਚਾਲਿਤ & ਇੰਦਰਾਜ : backends) {
            ਜੇ (ਇੰਦਰਾਜ.handle) {
                ਇੰਦਰਾਜ.handle.release(); // NOLINT
            }
        }
    }

    ਖਾਲੀ ਰਜਿਸਟਰ_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = ਖਾਲੀ_ਸੂਚਕ) {
        ਜੇ (!reg) {
            ਵਾਪਸ;
        }

        ਲਈ (ਸਵੈਚਾਲਿਤ & ਇੰਦਰਾਜ : backends) {
            ਜੇ (ਇੰਦਰਾਜ.reg == reg) {
                ਵਾਪਸ;
            }
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ਗਿਣਤੀ(reg));
#ਜੇ_ਅੰਤ
        backends.push_back({ reg, std::move(handle) });
        ਲਈ (size_t i = 0; i < ggml_backend_reg_dev_ਗਿਣਤੀ(reg); i++) {
            ਰਜਿਸਟਰ_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    ਖਾਲੀ ਰਜਿਸਟਰ_device(ggml_backend_dev_t device) {
        ਲਈ (ਸਵੈਚਾਲਿਤ & dev : devices) {
            ਜੇ (dev == device) {
                ਵਾਪਸ;
            }
        }

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#ਜੇ_ਅੰਤ
        devices.push_back(device);
    }

    ggml_backend_reg_t ਲੱਦੋ_backend(ਅਚਲ fs::path & path, ਤਾਰਕਿਕ silent) {
        dl_handle_ptr handle { dl_ਲੱਦੋ_library(path) };
        ਜੇ (!handle) {
            ਜੇ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, ਰਸਤਾ_str(path).c_str(), dl_error());
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਸਵੈਚਾਲਿਤ score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ਜੇ (score_fn && score_fn() == 0) {
            ਜੇ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, ਰਸਤਾ_str(path).c_str());
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ਸਵੈਚਾਲਿਤ backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ਜੇ (!backend_init_fn) {
            ਜੇ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, ਰਸਤਾ_str(path).c_str());
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ਜੇ (!reg || reg->api_ਸੰਸਕਰਣ != GGML_BACKEND_API_ਸੰਸਕਰਣ) {
            ਜੇ (!silent) {
                ਜੇ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, ਰਸਤਾ_str(path).c_str());
                } ਨਹੀਂਤਾਂ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, ਰਸਤਾ_str(path).c_str(), reg->api_ਸੰਸਕਰਣ, GGML_BACKEND_API_ਸੰਸਕਰਣ);
                }
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), ਰਸਤਾ_str(path).c_str());

        ਰਜਿਸਟਰ_backend(reg, std::move(handle));

        ਵਾਪਸ reg;
    }

    ਖਾਲੀ unload_backend(ggml_backend_reg_t reg, ਤਾਰਕਿਕ silent) {
        ਸਵੈਚਾਲਿਤ it = std::find_if(backends.begin(), backends.end(),
                               [reg](ਅਚਲ ggml_backend_reg_ਇੰਦਰਾਜ & ਇੰਦਰਾਜ) { ਵਾਪਸ ਇੰਦਰਾਜ.reg == reg; });

        ਜੇ (it == backends.end()) {
            ਜੇ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ਵਾਪਸ;
        }

        ਜੇ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ਵਾਪਸ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ਸਥਿਰ ggml_backend_registry & get_reg() {
    ਸਥਿਰ ggml_backend_registry reg;
    ਵਾਪਸ reg;
}

// Internal API
ਖਾਲੀ ggml_backend_ਰਜਿਸਟਰ(ggml_backend_reg_t reg) {
    get_reg().ਰਜਿਸਟਰ_backend(reg);
}

ਖਾਲੀ ggml_backend_device_ਰਜਿਸਟਰ(ggml_backend_dev_t device) {
    get_reg().ਰਜਿਸਟਰ_device(device);
}

// Backend (reg) enumeration
ਸਥਿਰ ਤਾਰਕਿਕ striequals(ਅਚਲ ਅੱਖਰ * a, ਅਚਲ ਅੱਖਰ * b) {
    ਲਈ (; *a && *b; a++, b++) {
        ਜੇ (std::tolower(*a) != std::tolower(*b)) {
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ *a == *b;
}

size_t ggml_backend_reg_ਗਿਣਤੀ() {
    ਵਾਪਸ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ < ggml_backend_reg_ਗਿਣਤੀ());
    ਵਾਪਸ get_reg().backends[ਸੂਚਕਾਂਕ].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ਅਚਲ ਅੱਖਰ * name) {
    ਲਈ (size_t i = 0; i < ggml_backend_reg_ਗਿਣਤੀ(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ਜੇ (striequals(ggml_backend_reg_name(reg), name)) {
            ਵਾਪਸ reg;
        }
    }
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

// Device enumeration
size_t ggml_backend_dev_ਗਿਣਤੀ() {
    ਵਾਪਸ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t ਸੂਚਕਾਂਕ) {
    GGML_ASSERT(ਸੂਚਕਾਂਕ < ggml_backend_dev_ਗਿਣਤੀ());
    ਵਾਪਸ get_reg().devices[ਸੂਚਕਾਂਕ];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ਅਚਲ ਅੱਖਰ * name) {
    ਲਈ (size_t i = 0; i < ggml_backend_dev_ਗਿਣਤੀ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ਜੇ (striequals(ggml_backend_dev_name(dev), name)) {
            ਵਾਪਸ dev;
        }
    }
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ਗਿਣਤੀ ggml_backend_dev_type type) {
    ਲਈ (size_t i = 0; i < ggml_backend_dev_ਗਿਣਤੀ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ਜੇ (ggml_backend_dev_type(dev) == type) {
            ਵਾਪਸ dev;
        }
    }
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ਅਚਲ ਅੱਖਰ * name, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ਜੇ (!dev) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ਗਿਣਤੀ ggml_backend_dev_type type, ਅਚਲ ਅੱਖਰ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ਜੇ (!dev) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_ਉੱਤਮ(ਖਾਲੀ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ਜੇ (!dev) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ggml_backend_dev_init(dev, ਖਾਲੀ_ਸੂਚਕ);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_ਲੱਦੋ(ਅਚਲ ਅੱਖਰ * path) {
    ਵਾਪਸ get_reg().ਲੱਦੋ_backend(path, ਝੂਠ);
}

ਖਾਲੀ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, ਸੱਚ);
}

ਸਥਿਰ fs::path get_executable_ਰਸਤਾ() {
#ਜੇ defined(__APPLE__)
    // get executable path
    std::vector<ਅੱਖਰ> path;
    uint32_t size;
    ਜਦੋਂ (ਸੱਚ) {
        size = path.size();
        ਜੇ (NSGet_Executable_ਰਸਤਾ(path.ਅੰਕੜੇ(), &size) == 0) {
            ਰੋਕੋ;
        }
        path.resize(size);
    }
    std::string ਅਧਾਰ_ਰਸਤਾ(path.ਅੰਕੜੇ(), size);
    // remove executable name
    ਸਵੈਚਾਲਿਤ ਆਖਰੀ_slash = ਅਧਾਰ_ਰਸਤਾ.find_last_of('/');
    ਜੇ (ਆਖਰੀ_slash != std::string::npos) {
        ਅਧਾਰ_ਰਸਤਾ = ਅਧਾਰ_ਰਸਤਾ.substr(0, ਆਖਰੀ_slash);
    }
    ਵਾਪਸ ਅਧਾਰ_ਰਸਤਾ + "/";
#ਜਾਂਜੇ defined(__linux__) || defined(__FreeBSD__)
    std::string ਅਧਾਰ_ਰਸਤਾ = ".";
    std::vector<ਅੱਖਰ> path(1024);
    ਜਦੋਂ (ਸੱਚ) {
        // get executable path
#    ਜੇ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.ਅੰਕੜੇ(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.ਅੰਕੜੇ(), path.size());
#    endif
        ਜੇ (len == -1) {
            ਰੋਕੋ;
        }
        ਜੇ (len < (ssize_t) path.size()) {
            ਅਧਾਰ_ਰਸਤਾ = std::string(path.ਅੰਕੜੇ(), len);
            // remove executable name
            ਸਵੈਚਾਲਿਤ ਆਖਰੀ_slash = ਅਧਾਰ_ਰਸਤਾ.find_last_of('/');
            ਜੇ (ਆਖਰੀ_slash != std::string::npos) {
                ਅਧਾਰ_ਰਸਤਾ = ਅਧਾਰ_ਰਸਤਾ.substr(0, ਆਖਰੀ_slash);
            }
            ਰੋਕੋ;
        }
        path.resize(path.size() * 2);
    }

    ਵਾਪਸ ਅਧਾਰ_ਰਸਤਾ + "/";
#ਜਾਂਜੇ defined(_WIN32)
    std::vector<wchar_t> path(MAX_ਰਸਤਾ);
    DWORD len = GetModuleFileNameW(ਨਿਰਾ, path.ਅੰਕੜੇ(), path.size());
    ਜੇ (len == 0) {
        ਵਾਪਸ {};
    }
    std::wstring ਅਧਾਰ_ਰਸਤਾ(path.ਅੰਕੜੇ(), len);
    // remove executable name
    ਸਵੈਚਾਲਿਤ ਆਖਰੀ_slash = ਅਧਾਰ_ਰਸਤਾ.find_last_of('\\');
    ਜੇ (ਆਖਰੀ_slash != std::string::npos) {
        ਅਧਾਰ_ਰਸਤਾ = ਅਧਾਰ_ਰਸਤਾ.substr(0, ਆਖਰੀ_slash);
    }
    ਵਾਪਸ ਅਧਾਰ_ਰਸਤਾ + L"\\";
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ {};
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ fs::path backend_filename_ਅਗੇਤਰ() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ fs::u8path("ggml-");
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ fs::u8path("libggml-");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ fs::path backend_filename_extension() {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ _WIN32
    ਵਾਪਸ fs::u8path(".dll");
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ fs::u8path(".so");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ ggml_backend_reg_t ggml_backend_ਲੱਦੋ_ਉੱਤਮ(ਅਚਲ ਅੱਖਰ * name, ਤਾਰਕਿਕ silent, ਅਚਲ ਅੱਖਰ * user_ਖੋਜ_ਰਸਤਾ) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ਅਚਲ fs::path name_ਰਸਤਾ = fs::u8path(name);
    ਅਚਲ fs::path file_ਅਗੇਤਰ = backend_filename_ਅਗੇਤਰ().native() + name_ਰਸਤਾ.native() + fs::u8path("-").native();
    ਅਚਲ fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> ਖੋਜ_paths;
    ਜੇ (user_ਖੋਜ_ਰਸਤਾ == ਖਾਲੀ_ਸੂਚਕ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_BACKEND_DIR
        ਖੋਜ_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#ਜੇ_ਅੰਤ
        // default search paths: executable directory, current directory
        ਖੋਜ_paths.push_back(get_executable_ਰਸਤਾ());
        std::error_code cwd_ec;
        ਅਚਲ fs::path cwd = fs::current_path(cwd_ec);
        ਜੇ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.ਸੁਨੇਹਾ().c_str());
        } ਨਹੀਂਤਾਂ {
            ਖੋਜ_paths.push_back(cwd);
        }
    } ਨਹੀਂਤਾਂ {
        ਖੋਜ_paths.push_back(fs::u8path(user_ਖੋਜ_ਰਸਤਾ));
    }

    ਪੂਰਨ ਉੱਤਮ_score = 0;
    fs::path ਉੱਤਮ_ਰਸਤਾ;
    std::error_code ec;

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਖੋਜ_ਰਸਤਾ : ਖੋਜ_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(ਖੋਜ_ਰਸਤਾ, dir_ec);

        ਲਈ (ਅਚਲ fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ਅਚਲ ਸਵੈਚਾਲਿਤ & ਇੰਦਰਾਜ = *dir_it;
            ਜੇ (ਇੰਦਰਾਜ.is_regular_file(ec)) {
                ਸਵੈਚਾਲਿਤ filename = ਇੰਦਰਾਜ.path().filename();
                ਸਵੈਚਾਲਿਤ ext = ਇੰਦਰਾਜ.path().extension();
                ਜੇ (filename.native().find(file_ਅਗੇਤਰ) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_ਲੱਦੋ_library(ਇੰਦਰਾਜ) };
                    ਜੇ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, ਰਸਤਾ_str(ਇੰਦਰਾਜ.path()).c_str(), dl_error());
                    }
                    ਜੇ (handle) {
                        ਸਵੈਚਾਲਿਤ score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ਜੇ (score_fn) {
                            ਪੂਰਨ s = score_fn();
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, ਰਸਤਾ_str(ਇੰਦਰਾਜ.path()).c_str(), s);
#ਜੇ_ਅੰਤ
                            ਜੇ (s > ਉੱਤਮ_score) {
                                ਉੱਤਮ_score = s;
                                ਉੱਤਮ_ਰਸਤਾ = ਇੰਦਰਾਜ.path();
                            }
                        } ਨਹੀਂਤਾਂ {
                            ਜੇ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, ਰਸਤਾ_str(ਇੰਦਰਾਜ.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ਜੇ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, ਰਸਤਾ_str(ਖੋਜ_ਰਸਤਾ).c_str(), dir_ec.ਸੁਨੇਹਾ().c_str());
        }
    }

    ਜੇ (ਉੱਤਮ_score == 0) {
        // try to load the base backend
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & ਖੋਜ_ਰਸਤਾ : ਖੋਜ_paths) {
            fs::path filename = backend_filename_ਅਗੇਤਰ().native() + name_ਰਸਤਾ.native() + backend_filename_extension().native();
            fs::path path = ਖੋਜ_ਰਸਤਾ / filename;
            ਜੇ (std::error_code ec; fs::exists(path, ec)) {
                ਵਾਪਸ get_reg().ਲੱਦੋ_backend(path, silent);
            } ਨਹੀਂਤਾਂ {
                ਜੇ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, ਰਸਤਾ_str(path).c_str(), ec.ਸੁਨੇਹਾ().c_str());
                }
            }
        }
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ get_reg().ਲੱਦੋ_backend(ਉੱਤਮ_ਰਸਤਾ, silent);
}

ਖਾਲੀ ggml_backend_ਲੱਦੋ_all() {
    ggml_backend_ਲੱਦੋ_all_from_ਰਸਤਾ(ਖਾਲੀ_ਸੂਚਕ);
}

ਖਾਲੀ ggml_backend_ਲੱਦੋ_all_from_ਰਸਤਾ(ਅਚਲ ਅੱਖਰ * dir_ਰਸਤਾ) {
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ NDEBUG
    ਤਾਰਕਿਕ silent = ਸੱਚ;
#ਨਹੀਂਤਾਂ
    ਤਾਰਕਿਕ silent = ਝੂਠ;
#ਜੇ_ਅੰਤ

    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("blas", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("zendnn", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("cann", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("cuda", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("hip", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("metal", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("rpc", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("sycl", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("vulkan", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("virtgpu", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("opencl", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("hexagon", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("musa", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("openvino", silent, dir_ਰਸਤਾ);
    ggml_backend_ਲੱਦੋ_ਉੱਤਮ("cpu", silent, dir_ਰਸਤਾ);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ਅਚਲ ਅੱਖਰ * backend_ਰਸਤਾ = std::getenv("GGML_BACKEND_PATH");
    ਜੇ (backend_ਰਸਤਾ) {
        ggml_backend_ਲੱਦੋ(backend_ਰਸਤਾ);
    }
}
