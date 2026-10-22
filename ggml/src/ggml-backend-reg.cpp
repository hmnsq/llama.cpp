#അന്തര്ഭാവ "ggml-backend-impl.h"
#അന്തര്ഭാവ "ggml-backend.h"
#അന്തര്ഭാവ "ggml-backend-dl.h"
#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ <algorithm>
#അന്തര്ഭാവ <cstring>
#അന്തര്ഭാവ <filesystem>
#അന്തര്ഭാവ <memory>
#അന്തര്ഭാവ <string>
#അന്തര്ഭാവ <type_traits>
#അന്തര്ഭാവ <vector>
#അന്തര്ഭാവ <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#അല്ലെങ്കിലെങ്കിൽ defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#അല്ലെങ്കിൽ
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#അന്തര്ഭാവ "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#അന്തര്ഭാവ "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#അന്തര്ഭാവ "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#അന്തര്ഭാവ "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#അന്തര്ഭാവ "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#അന്തര്ഭാവ "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#അന്തര്ഭാവ "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#അന്തര്ഭാവ "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#അന്തര്ഭാവ "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#അന്തര്ഭാവ "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#അന്തര്ഭാവ "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#അന്തര്ഭാവ "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#അന്തര്ഭാവ "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#അന്തര്ഭാവ "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#അന്തര്ഭാവ "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#അന്തര്ഭാവ "ggml-et.h"
#endif

നാമാകാശ fs = std::filesystem;

സ്ഥിതം std::string path_str(സ്ഥിരം fs::path & path) {
    ശ്രമിക്കുക {
#എങ്കിൽ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        സ്ഥിരം std::u8string u8str = path.u8string();
        തിരികെ std::string(reinterpret_cast<സ്ഥിരം അക്ഷരം *>(u8str.data()), u8str.size());
#അല്ലെങ്കിൽ
        // C++17: u8string() returns std::string
        തിരികെ path.u8string();
#endif
    } പിടിക്കുക (...) {
        തിരികെ std::string();
    }
}

ഘടന ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ഘടന ggml_backend_registry {
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
    എങ്കിൽ (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } അല്ലെങ്കിൽ {
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
        വേണ്ടി (auto & entry : backends) {
            എങ്കിൽ (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    ശൂന്യം register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        എങ്കിൽ (!reg) {
            തിരികെ;
        }

        വേണ്ടി (auto & entry : backends) {
            എങ്കിൽ (entry.reg == reg) {
                തിരികെ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_എണ്ണം(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        വേണ്ടി (size_t i = 0; i < ggml_backend_reg_dev_എണ്ണം(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    ശൂന്യം register_device(ggml_backend_dev_t device) {
        വേണ്ടി (auto & dev : devices) {
            എങ്കിൽ (dev == device) {
                തിരികെ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(സ്ഥിരം fs::path & path, തർക്കം silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        എങ്കിൽ (!handle) {
            എങ്കിൽ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            തിരികെ nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        എങ്കിൽ (score_fn && score_fn() == 0) {
            എങ്കിൽ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            തിരികെ nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        എങ്കിൽ (!backend_init_fn) {
            എങ്കിൽ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            തിരികെ nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        എങ്കിൽ (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            എങ്കിൽ (!silent) {
                എങ്കിൽ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } അല്ലെങ്കിൽ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            തിരികെ nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        തിരികെ reg;
    }

    ശൂന്യം unload_backend(ggml_backend_reg_t reg, തർക്കം silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](സ്ഥിരം ggml_backend_reg_entry & entry) { തിരികെ entry.reg == reg; });

        എങ്കിൽ (it == backends.end()) {
            എങ്കിൽ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            തിരികെ;
        }

        എങ്കിൽ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { തിരികെ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

സ്ഥിതം ggml_backend_registry & get_reg() {
    സ്ഥിതം ggml_backend_registry reg;
    തിരികെ reg;
}

// Internal API
ശൂന്യം ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

ശൂന്യം ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
സ്ഥിതം തർക്കം striequals(സ്ഥിരം അക്ഷരം * a, സ്ഥിരം അക്ഷരം * b) {
    വേണ്ടി (; *a && *b; a++, b++) {
        എങ്കിൽ (std::tolower(*a) != std::tolower(*b)) {
            തിരികെ അസത്യം;
        }
    }
    തിരികെ *a == *b;
}

size_t ggml_backend_reg_എണ്ണം() {
    തിരികെ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_എണ്ണം());
    തിരികെ get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(സ്ഥിരം അക്ഷരം * name) {
    വേണ്ടി (size_t i = 0; i < ggml_backend_reg_എണ്ണം(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        എങ്കിൽ (striequals(ggml_backend_reg_name(reg), name)) {
            തിരികെ reg;
        }
    }
    തിരികെ nullptr;
}

// Device enumeration
size_t ggml_backend_dev_എണ്ണം() {
    തിരികെ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_എണ്ണം());
    തിരികെ get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(സ്ഥിരം അക്ഷരം * name) {
    വേണ്ടി (size_t i = 0; i < ggml_backend_dev_എണ്ണം(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        എങ്കിൽ (striequals(ggml_backend_dev_name(dev), name)) {
            തിരികെ dev;
        }
    }
    തിരികെ nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(എണ്ണൽ ggml_backend_dev_type type) {
    വേണ്ടി (size_t i = 0; i < ggml_backend_dev_എണ്ണം(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        എങ്കിൽ (ggml_backend_dev_type(dev) == type) {
            തിരികെ dev;
        }
    }
    തിരികെ nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(സ്ഥിരം അക്ഷരം * name, സ്ഥിരം അക്ഷരം * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    എങ്കിൽ (!dev) {
        തിരികെ nullptr;
    }
    തിരികെ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(എണ്ണൽ ggml_backend_dev_type type, സ്ഥിരം അക്ഷരം * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    എങ്കിൽ (!dev) {
        തിരികെ nullptr;
    }
    തിരികെ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(ശൂന്യം) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    എങ്കിൽ (!dev) {
        തിരികെ nullptr;
    }
    തിരികെ ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(സ്ഥിരം അക്ഷരം * path) {
    തിരികെ get_reg().load_backend(path, അസത്യം);
}

ശൂന്യം ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, സത്യം);
}

സ്ഥിതം fs::path get_executable_path() {
#എങ്കിൽ defined(__APPLE__)
    // get executable path
    std::vector<അക്ഷരം> path;
    uint32_t size;
    ആകുമ്പോൾ (സത്യം) {
        size = path.size();
        എങ്കിൽ (_NSGetExecutablePath(path.data(), &size) == 0) {
            നിർത്തുക;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    എങ്കിൽ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    തിരികെ base_path + "/";
#അല്ലെങ്കിലെങ്കിൽ defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<അക്ഷരം> path(1024);
    ആകുമ്പോൾ (സത്യം) {
        // get executable path
#    എങ്കിൽ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        എങ്കിൽ (len == -1) {
            നിർത്തുക;
        }
        എങ്കിൽ (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            എങ്കിൽ (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            നിർത്തുക;
        }
        path.resize(path.size() * 2);
    }

    തിരികെ base_path + "/";
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(ശൂന്യ, path.data(), path.size());
    എങ്കിൽ (len == 0) {
        തിരികെ {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    എങ്കിൽ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    തിരികെ base_path + L"\\";
#അല്ലെങ്കിൽ
    തിരികെ {};
#endif
}

സ്ഥിതം fs::path backend_filename_prefix() {
#ifdef _WIN32
    തിരികെ fs::u8path("ggml-");
#അല്ലെങ്കിൽ
    തിരികെ fs::u8path("libggml-");
#endif
}

സ്ഥിതം fs::path backend_filename_extension() {
#ifdef _WIN32
    തിരികെ fs::u8path(".dll");
#അല്ലെങ്കിൽ
    തിരികെ fs::u8path(".so");
#endif
}

സ്ഥിതം ggml_backend_reg_t ggml_backend_load_best(സ്ഥിരം അക്ഷരം * name, തർക്കം silent, സ്ഥിരം അക്ഷരം * user_തിരയൽ_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    സ്ഥിരം fs::path name_path = fs::u8path(name);
    സ്ഥിരം fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    സ്ഥിരം fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> തിരയൽ_paths;
    എങ്കിൽ (user_തിരയൽ_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        തിരയൽ_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        തിരയൽ_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        സ്ഥിരം fs::path cwd = fs::current_path(cwd_ec);
        എങ്കിൽ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } അല്ലെങ്കിൽ {
            തിരയൽ_paths.push_back(cwd);
        }
    } അല്ലെങ്കിൽ {
        തിരയൽ_paths.push_back(fs::u8path(user_തിരയൽ_path));
    }

    പൂർണ്ണസംഖ്യ best_score = 0;
    fs::path best_path;
    std::error_code ec;

    വേണ്ടി (സ്ഥിരം auto & തിരയൽ_path : തിരയൽ_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(തിരയൽ_path, dir_ec);

        വേണ്ടി (സ്ഥിരം fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            സ്ഥിരം auto & entry = *dir_it;
            എങ്കിൽ (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                എങ്കിൽ (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    എങ്കിൽ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    എങ്കിൽ (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        എങ്കിൽ (score_fn) {
                            പൂർണ്ണസംഖ്യ s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            എങ്കിൽ (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } അല്ലെങ്കിൽ {
                            എങ്കിൽ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        എങ്കിൽ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(തിരയൽ_path).c_str(), dir_ec.message().c_str());
        }
    }

    എങ്കിൽ (best_score == 0) {
        // try to load the base backend
        വേണ്ടി (സ്ഥിരം auto & തിരയൽ_path : തിരയൽ_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = തിരയൽ_path / filename;
            എങ്കിൽ (std::error_code ec; fs::exists(path, ec)) {
                തിരികെ get_reg().load_backend(path, silent);
            } അല്ലെങ്കിൽ {
                എങ്കിൽ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        തിരികെ nullptr;
    }

    തിരികെ get_reg().load_backend(best_path, silent);
}

ശൂന്യം ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

ശൂന്യം ggml_backend_load_all_from_path(സ്ഥിരം അക്ഷരം * dir_path) {
#ifdef NDEBUG
    തർക്കം silent = സത്യം;
#അല്ലെങ്കിൽ
    തർക്കം silent = അസത്യം;
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
    സ്ഥിരം അക്ഷരം * backend_path = std::getenv("GGML_BACKEND_PATH");
    എങ്കിൽ (backend_path) {
        ggml_backend_load(backend_path);
    }
}
