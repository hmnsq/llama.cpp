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

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ನೋಚೇದ್ಯದಿ defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#ಅನ್ಯಥಾ
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#ಅನ್ತರ್ಭಾವ "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#ಅನ್ತರ್ಭಾವ "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#ಅನ್ತರ್ಭಾವ "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#ಅನ್ತರ್ಭಾವ "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#ಅನ್ತರ್ಭಾವ "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#ಅನ್ತರ್ಭಾವ "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#ಅನ್ತರ್ಭಾವ "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#ಅನ್ತರ್ಭಾವ "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#ಅನ್ತರ್ಭಾವ "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#ಅನ್ತರ್ಭಾವ "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#ಅನ್ತರ್ಭಾವ "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#ಅನ್ತರ್ಭಾವ "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#ಅನ್ತರ್ಭಾವ "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#ಅನ್ತರ್ಭಾವ "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#ಅನ್ತರ್ಭಾವ "ggml-et.h"
#endif

ನಾಮಾಕಾಶ fs = std::filesystem;

ಸ್ಥಿರ std::string path_str(ಧ್ರುವ fs::path & path) {
    ಯತ್ನ {
#ಯದಿ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ಧ್ರುವ std::u8string u8str = path.u8string();
        ಪ್ರತ್ಯಾಗಮ std::string(reinterpret_cast<ಧ್ರುವ ವರ್ಣ *>(u8str.data()), u8str.size());
#ಅನ್ಯಥಾ
        // C++17: u8string() returns std::string
        ಪ್ರತ್ಯಾಗಮ path.u8string();
#endif
    } ಗೃಹಾಣ (...) {
        ಪ್ರತ್ಯಾಗಮ std::string();
    }
}

ಸಂರಚನಾ ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ಸಂರಚನಾ ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#ifdef GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#endif
#ifdef GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#endif
#ifdef GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#endif
#ifdef GGML_USE_VULKAN
    // Add runtime disable check
    ಯದಿ (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } ಅನ್ಯಥಾ {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#endif
#ifdef GGML_USE_WEBGPU
        register_backend(ggml_backend_webgpu_reg());
#endif
#ifdef GGML_USE_ZDNN
        register_backend(ggml_backend_zdnn_reg());
#endif
#ifdef GGML_USE_VIRTGPU_FRONTEND
        register_backend(ggml_backend_virtgpu_reg());
#endif

#ifdef GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#endif
#ifdef GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#endif
#ifdef GGML_USE_HEXAGON
        register_backend(ggml_backend_hexagon_reg());
#endif
#ifdef GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#endif
#ifdef GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#endif
#ifdef GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#endif
#ifdef GGML_USE_OPENVINO
        register_backend(ggml_backend_openvino_reg());
#endif
#ifdef GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#endif
#ifdef GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#endif
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        ಪ್ರತಿ (auto & entry : backends) {
            ಯದಿ (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    ಶೂನ್ಯ register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        ಯದಿ (!reg) {
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಪ್ರತಿ (auto & entry : backends) {
            ಯದಿ (entry.reg == reg) {
                ಪ್ರತ್ಯಾಗಮ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ಎಣಿಕೆ(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        ಪ್ರತಿ (size_t i = 0; i < ggml_backend_reg_dev_ಎಣಿಕೆ(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    ಶೂನ್ಯ register_device(ggml_backend_dev_t device) {
        ಪ್ರತಿ (auto & dev : devices) {
            ಯದಿ (dev == device) {
                ಪ್ರತ್ಯಾಗಮ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ಧ್ರುವ fs::path & path, ತಾರ್ಕಿಕ silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        ಯದಿ (!handle) {
            ಯದಿ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ಯದಿ (score_fn && score_fn() == 0) {
            ಯದಿ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ಯದಿ (!backend_init_fn) {
            ಯದಿ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ಯದಿ (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            ಯದಿ (!silent) {
                ಯದಿ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } ಅನ್ಯಥಾ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        ಪ್ರತ್ಯಾಗಮ reg;
    }

    ಶೂನ್ಯ unload_backend(ggml_backend_reg_t reg, ತಾರ್ಕಿಕ silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ಧ್ರುವ ggml_backend_reg_entry & entry) { ಪ್ರತ್ಯಾಗಮ entry.reg == reg; });

        ಯದಿ (it == backends.end()) {
            ಯದಿ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಯದಿ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ಸ್ಥಿರ ggml_backend_registry & get_reg() {
    ಸ್ಥಿರ ggml_backend_registry reg;
    ಪ್ರತ್ಯಾಗಮ reg;
}

// Internal API
ಶೂನ್ಯ ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

ಶೂನ್ಯ ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
ಸ್ಥಿರ ತಾರ್ಕಿಕ striequals(ಧ್ರುವ ವರ್ಣ * a, ಧ್ರುವ ವರ್ಣ * b) {
    ಪ್ರತಿ (; *a && *b; a++, b++) {
        ಯದಿ (std::tolower(*a) != std::tolower(*b)) {
            ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
        }
    }
    ಪ್ರತ್ಯಾಗಮ *a == *b;
}

size_t ggml_backend_reg_ಎಣಿಕೆ() {
    ಪ್ರತ್ಯಾಗಮ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_ಎಣಿಕೆ());
    ಪ್ರತ್ಯಾಗಮ get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ಧ್ರುವ ವರ್ಣ * name) {
    ಪ್ರತಿ (size_t i = 0; i < ggml_backend_reg_ಎಣಿಕೆ(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ಯದಿ (striequals(ggml_backend_reg_name(reg), name)) {
            ಪ್ರತ್ಯಾಗಮ reg;
        }
    }
    ಪ್ರತ್ಯಾಗಮ nullptr;
}

// Device enumeration
size_t ggml_backend_dev_ಎಣಿಕೆ() {
    ಪ್ರತ್ಯಾಗಮ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_ಎಣಿಕೆ());
    ಪ್ರತ್ಯಾಗಮ get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ಧ್ರುವ ವರ್ಣ * name) {
    ಪ್ರತಿ (size_t i = 0; i < ggml_backend_dev_ಎಣಿಕೆ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ಯದಿ (striequals(ggml_backend_dev_name(dev), name)) {
            ಪ್ರತ್ಯಾಗಮ dev;
        }
    }
    ಪ್ರತ್ಯಾಗಮ nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ಗಣನಾ ggml_backend_dev_type type) {
    ಪ್ರತಿ (size_t i = 0; i < ggml_backend_dev_ಎಣಿಕೆ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ಯದಿ (ggml_backend_dev_type(dev) == type) {
            ಪ್ರತ್ಯಾಗಮ dev;
        }
    }
    ಪ್ರತ್ಯಾಗಮ nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ಧ್ರುವ ವರ್ಣ * name, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ಯದಿ (!dev) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ಗಣನಾ ggml_backend_dev_type type, ಧ್ರುವ ವರ್ಣ * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ಯದಿ (!dev) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(ಶೂನ್ಯ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ಯದಿ (!dev) {
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }
    ಪ್ರತ್ಯಾಗಮ ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ಧ್ರುವ ವರ್ಣ * path) {
    ಪ್ರತ್ಯಾಗಮ get_reg().load_backend(path, ಅಸತ್ಯ);
}

ಶೂನ್ಯ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, ಸತ್ಯ);
}

ಸ್ಥಿರ fs::path get_executable_path() {
#ಯದಿ defined(__APPLE__)
    // get executable path
    std::vector<ವರ್ಣ> path;
    uint32_t size;
    ಯಾವತ್ (ಸತ್ಯ) {
        size = path.size();
        ಯದಿ (_NSGetExecutablePath(path.data(), &size) == 0) {
            ಭಙ್ಗ;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    ಯದಿ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ಪ್ರತ್ಯಾಗಮ base_path + "/";
#ನೋಚೇದ್ಯದಿ defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<ವರ್ಣ> path(1024);
    ಯಾವತ್ (ಸತ್ಯ) {
        // get executable path
#    ಯದಿ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        ಯದಿ (len == -1) {
            ಭಙ್ಗ;
        }
        ಯದಿ (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            ಯದಿ (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            ಭಙ್ಗ;
        }
        path.resize(path.size() * 2);
    }

    ಪ್ರತ್ಯಾಗಮ base_path + "/";
#ನೋಚೇದ್ಯದಿ defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(ರಿಕ್ತ, path.data(), path.size());
    ಯದಿ (len == 0) {
        ಪ್ರತ್ಯಾಗಮ {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    ಯದಿ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ಪ್ರತ್ಯಾಗಮ base_path + L"\\";
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ {};
#endif
}

ಸ್ಥಿರ fs::path backend_filename_prefix() {
#ifdef _WIN32
    ಪ್ರತ್ಯಾಗಮ fs::u8path("ggml-");
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ fs::u8path("libggml-");
#endif
}

ಸ್ಥಿರ fs::path backend_filename_extension() {
#ifdef _WIN32
    ಪ್ರತ್ಯಾಗಮ fs::u8path(".dll");
#ಅನ್ಯಥಾ
    ಪ್ರತ್ಯಾಗಮ fs::u8path(".so");
#endif
}

ಸ್ಥಿರ ggml_backend_reg_t ggml_backend_load_best(ಧ್ರುವ ವರ್ಣ * name, ತಾರ್ಕಿಕ silent, ಧ್ರುವ ವರ್ಣ * user_ಹುಡುಕು_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ಧ್ರುವ fs::path name_path = fs::u8path(name);
    ಧ್ರುವ fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ಧ್ರುವ fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> ಹುಡುಕು_paths;
    ಯದಿ (user_ಹುಡುಕು_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        ಹುಡುಕು_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        ಹುಡುಕು_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ಧ್ರುವ fs::path cwd = fs::current_path(cwd_ec);
        ಯದಿ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } ಅನ್ಯಥಾ {
            ಹುಡುಕು_paths.push_back(cwd);
        }
    } ಅನ್ಯಥಾ {
        ಹುಡುಕು_paths.push_back(fs::u8path(user_ಹುಡುಕು_path));
    }

    ಪೂರ್ಣಾಙ್ಕ best_score = 0;
    fs::path best_path;
    std::error_code ec;

    ಪ್ರತಿ (ಧ್ರುವ auto & ಹುಡುಕು_path : ಹುಡುಕು_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(ಹುಡುಕು_path, dir_ec);

        ಪ್ರತಿ (ಧ್ರುವ fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ಧ್ರುವ auto & entry = *dir_it;
            ಯದಿ (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                ಯದಿ (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    ಯದಿ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    ಯದಿ (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ಯದಿ (score_fn) {
                            ಪೂರ್ಣಾಙ್ಕ s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            ಯದಿ (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } ಅನ್ಯಥಾ {
                            ಯದಿ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ಯದಿ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(ಹುಡುಕು_path).c_str(), dir_ec.message().c_str());
        }
    }

    ಯದಿ (best_score == 0) {
        // try to load the base backend
        ಪ್ರತಿ (ಧ್ರುವ auto & ಹುಡುಕು_path : ಹುಡುಕು_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = ಹುಡುಕು_path / filename;
            ಯದಿ (std::error_code ec; fs::exists(path, ec)) {
                ಪ್ರತ್ಯಾಗಮ get_reg().load_backend(path, silent);
            } ಅನ್ಯಥಾ {
                ಯದಿ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        ಪ್ರತ್ಯಾಗಮ nullptr;
    }

    ಪ್ರತ್ಯಾಗಮ get_reg().load_backend(best_path, silent);
}

ಶೂನ್ಯ ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

ಶೂನ್ಯ ggml_backend_load_all_from_path(ಧ್ರುವ ವರ್ಣ * dir_path) {
#ifdef NDEBUG
    ತಾರ್ಕಿಕ silent = ಸತ್ಯ;
#ಅನ್ಯಥಾ
    ತಾರ್ಕಿಕ silent = ಅಸತ್ಯ;
#endif

    ggml_backend_load_best("blas", silent, dir_path);
    ggml_backend_load_best("zendnn", silent, dir_path);
    ggml_backend_load_best("cann", silent, dir_path);
    ggml_backend_load_best("cuda", silent, dir_path);
    ggml_backend_load_best("hip", silent, dir_path);
    ggml_backend_load_best("metal", silent, dir_path);
    ggml_backend_load_best("rpc", silent, dir_path);
    ggml_backend_load_best("sycl", silent, dir_path);
    ggml_backend_load_best("vulkan", silent, dir_path);
    ggml_backend_load_best("virtgpu", silent, dir_path);
    ggml_backend_load_best("opencl", silent, dir_path);
    ggml_backend_load_best("hexagon", silent, dir_path);
    ggml_backend_load_best("musa", silent, dir_path);
    ggml_backend_load_best("openvino", silent, dir_path);
    ggml_backend_load_best("cpu", silent, dir_path);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ಧ್ರುವ ವರ್ಣ * backend_path = std::getenv("GGML_BACKEND_PATH");
    ಯದಿ (backend_path) {
        ggml_backend_load(backend_path);
    }
}
