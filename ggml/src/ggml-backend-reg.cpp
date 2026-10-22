#அந்தர்பாவ "ggml-backend-impl.h"
#அந்தர்பாவ "ggml-backend.h"
#அந்தர்பாவ "ggml-backend-dl.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstring>
#அந்தர்பாவ <filesystem>
#அந்தர்பாவ <memory>
#அந்தர்பாவ <string>
#அந்தர்பாவ <type_traits>
#அந்தர்பாவ <vector>
#அந்தர்பாவ <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#அல்லதெனில் defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#இல்லையெனில்
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#அந்தர்பாவ "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#அந்தர்பாவ "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#அந்தர்பாவ "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#அந்தர்பாவ "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#அந்தர்பாவ "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#அந்தர்பாவ "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#அந்தர்பாவ "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#அந்தர்பாவ "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#அந்தர்பாவ "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#அந்தர்பாவ "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#அந்தர்பாவ "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#அந்தர்பாவ "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#அந்தர்பாவ "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#அந்தர்பாவ "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#அந்தர்பாவ "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#அந்தர்பாவ "ggml-et.h"
#endif

நாமாகாஶ fs = std::filesystem;

நிலையான std::string path_str(மாறிலி fs::path & path) {
    முயல் {
#எனில் defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        மாறிலி std::u8string u8str = path.u8string();
        திரும்பு std::string(reinterpret_cast<மாறிலி எழுத்து *>(u8str.data()), u8str.size());
#இல்லையெனில்
        // C++17: u8string() returns std::string
        திரும்பு path.u8string();
#endif
    } பிடி (...) {
        திரும்பு std::string();
    }
}

கட்டமைப்பு ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

கட்டமைப்பு ggml_backend_registry {
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
    எனில் (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } இல்லையெனில் {
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
        க்கு (auto & entry : backends) {
            எனில் (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    வெறுமை register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        எனில் (!reg) {
            திரும்பு;
        }

        க்கு (auto & entry : backends) {
            எனில் (entry.reg == reg) {
                திரும்பு;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_மொத்தம்(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        க்கு (size_t i = 0; i < ggml_backend_reg_dev_மொத்தம்(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    வெறுமை register_device(ggml_backend_dev_t device) {
        க்கு (auto & dev : devices) {
            எனில் (dev == device) {
                திரும்பு;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(மாறிலி fs::path & path, தருக்க silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        எனில் (!handle) {
            எனில் (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            திரும்பு nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        எனில் (score_fn && score_fn() == 0) {
            எனில் (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            திரும்பு nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        எனில் (!backend_init_fn) {
            எனில் (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            திரும்பு nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        எனில் (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            எனில் (!silent) {
                எனில் (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } இல்லையெனில் {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            திரும்பு nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        திரும்பு reg;
    }

    வெறுமை unload_backend(ggml_backend_reg_t reg, தருக்க silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](மாறிலி ggml_backend_reg_entry & entry) { திரும்பு entry.reg == reg; });

        எனில் (it == backends.end()) {
            எனில் (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            திரும்பு;
        }

        எனில் (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { திரும்பு ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

நிலையான ggml_backend_registry & get_reg() {
    நிலையான ggml_backend_registry reg;
    திரும்பு reg;
}

// Internal API
வெறுமை ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

வெறுமை ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
நிலையான தருக்க striequals(மாறிலி எழுத்து * a, மாறிலி எழுத்து * b) {
    க்கு (; *a && *b; a++, b++) {
        எனில் (std::tolower(*a) != std::tolower(*b)) {
            திரும்பு பொய்;
        }
    }
    திரும்பு *a == *b;
}

size_t ggml_backend_reg_மொத்தம்() {
    திரும்பு get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_மொத்தம்());
    திரும்பு get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(மாறிலி எழுத்து * name) {
    க்கு (size_t i = 0; i < ggml_backend_reg_மொத்தம்(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        எனில் (striequals(ggml_backend_reg_name(reg), name)) {
            திரும்பு reg;
        }
    }
    திரும்பு nullptr;
}

// Device enumeration
size_t ggml_backend_dev_மொத்தம்() {
    திரும்பு get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_மொத்தம்());
    திரும்பு get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(மாறிலி எழுத்து * name) {
    க்கு (size_t i = 0; i < ggml_backend_dev_மொத்தம்(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        எனில் (striequals(ggml_backend_dev_name(dev), name)) {
            திரும்பு dev;
        }
    }
    திரும்பு nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(எண்ணிக்கை ggml_backend_dev_type type) {
    க்கு (size_t i = 0; i < ggml_backend_dev_மொத்தம்(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        எனில் (ggml_backend_dev_type(dev) == type) {
            திரும்பு dev;
        }
    }
    திரும்பு nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(மாறிலி எழுத்து * name, மாறிலி எழுத்து * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    எனில் (!dev) {
        திரும்பு nullptr;
    }
    திரும்பு ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(எண்ணிக்கை ggml_backend_dev_type type, மாறிலி எழுத்து * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    எனில் (!dev) {
        திரும்பு nullptr;
    }
    திரும்பு ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(வெறுமை) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    எனில் (!dev) {
        திரும்பு nullptr;
    }
    திரும்பு ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(மாறிலி எழுத்து * path) {
    திரும்பு get_reg().load_backend(path, பொய்);
}

வெறுமை ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, உண்மை);
}

நிலையான fs::path get_executable_path() {
#எனில் defined(__APPLE__)
    // get executable path
    std::vector<எழுத்து> path;
    uint32_t size;
    போது (உண்மை) {
        size = path.size();
        எனில் (_NSGetExecutablePath(path.data(), &size) == 0) {
            நிறுத்து;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    எனில் (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    திரும்பு base_path + "/";
#அல்லதெனில் defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<எழுத்து> path(1024);
    போது (உண்மை) {
        // get executable path
#    எனில் defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        எனில் (len == -1) {
            நிறுத்து;
        }
        எனில் (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            எனில் (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            நிறுத்து;
        }
        path.resize(path.size() * 2);
    }

    திரும்பு base_path + "/";
#அல்லதெனில் defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(வெற்று, path.data(), path.size());
    எனில் (len == 0) {
        திரும்பு {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    எனில் (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    திரும்பு base_path + L"\\";
#இல்லையெனில்
    திரும்பு {};
#endif
}

நிலையான fs::path backend_filename_prefix() {
#ifdef _WIN32
    திரும்பு fs::u8path("ggml-");
#இல்லையெனில்
    திரும்பு fs::u8path("libggml-");
#endif
}

நிலையான fs::path backend_filename_extension() {
#ifdef _WIN32
    திரும்பு fs::u8path(".dll");
#இல்லையெனில்
    திரும்பு fs::u8path(".so");
#endif
}

நிலையான ggml_backend_reg_t ggml_backend_load_best(மாறிலி எழுத்து * name, தருக்க silent, மாறிலி எழுத்து * user_தேடல்_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    மாறிலி fs::path name_path = fs::u8path(name);
    மாறிலி fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    மாறிலி fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> தேடல்_paths;
    எனில் (user_தேடல்_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        தேடல்_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        தேடல்_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        மாறிலி fs::path cwd = fs::current_path(cwd_ec);
        எனில் (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } இல்லையெனில் {
            தேடல்_paths.push_back(cwd);
        }
    } இல்லையெனில் {
        தேடல்_paths.push_back(fs::u8path(user_தேடல்_path));
    }

    முழு best_score = 0;
    fs::path best_path;
    std::error_code ec;

    க்கு (மாறிலி auto & தேடல்_path : தேடல்_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(தேடல்_path, dir_ec);

        க்கு (மாறிலி fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            மாறிலி auto & entry = *dir_it;
            எனில் (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                எனில் (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    எனில் (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    எனில் (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        எனில் (score_fn) {
                            முழு s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            எனில் (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } இல்லையெனில் {
                            எனில் (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        எனில் (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(தேடல்_path).c_str(), dir_ec.message().c_str());
        }
    }

    எனில் (best_score == 0) {
        // try to load the base backend
        க்கு (மாறிலி auto & தேடல்_path : தேடல்_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = தேடல்_path / filename;
            எனில் (std::error_code ec; fs::exists(path, ec)) {
                திரும்பு get_reg().load_backend(path, silent);
            } இல்லையெனில் {
                எனில் (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        திரும்பு nullptr;
    }

    திரும்பு get_reg().load_backend(best_path, silent);
}

வெறுமை ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

வெறுமை ggml_backend_load_all_from_path(மாறிலி எழுத்து * dir_path) {
#ifdef NDEBUG
    தருக்க silent = உண்மை;
#இல்லையெனில்
    தருக்க silent = பொய்;
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
    மாறிலி எழுத்து * backend_path = std::getenv("GGML_BACKEND_PATH");
    எனில் (backend_path) {
        ggml_backend_load(backend_path);
    }
}
