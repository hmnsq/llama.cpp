#ଅନ୍ତର୍ଭାଵ "ggml-backend-impl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend.h"
#ଅନ୍ତର୍ଭାଵ "ggml-backend-dl.h"
#ଅନ୍ତର୍ଭାଵ "ggml-impl.h"
#ଅନ୍ତର୍ଭାଵ <algorithm>
#ଅନ୍ତର୍ଭାଵ <cstring>
#ଅନ୍ତର୍ଭାଵ <filesystem>
#ଅନ୍ତର୍ଭାଵ <memory>
#ଅନ୍ତର୍ଭାଵ <string>
#ଅନ୍ତର୍ଭାଵ <type_traits>
#ଅନ୍ତର୍ଭାଵ <vector>
#ଅନ୍ତର୍ଭାଵ <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#ନଚେତଯଦି defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#ଅନ୍ୟଥା
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#ଅନ୍ତର୍ଭାଵ "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#ଅନ୍ତର୍ଭାଵ "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#ଅନ୍ତର୍ଭାଵ "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#ଅନ୍ତର୍ଭାଵ "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#ଅନ୍ତର୍ଭାଵ "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#ଅନ୍ତର୍ଭାଵ "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#ଅନ୍ତର୍ଭାଵ "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#ଅନ୍ତର୍ଭାଵ "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#ଅନ୍ତର୍ଭାଵ "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#ଅନ୍ତର୍ଭାଵ "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#ଅନ୍ତର୍ଭାଵ "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#ଅନ୍ତର୍ଭାଵ "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#ଅନ୍ତର୍ଭାଵ "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#ଅନ୍ତର୍ଭାଵ "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#ଅନ୍ତର୍ଭାଵ "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#ଅନ୍ତର୍ଭାଵ "ggml-et.h"
#endif

ନାମାକାଶ fs = std::filesystem;

ସ୍ଥିର std::string path_str(ଅଚଳ fs::path & path) {
    ଚେଷ୍ଟା {
#ଯଦି defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ଅଚଳ std::u8string u8str = path.u8string();
        ଫେରନ୍ତୁ std::string(reinterpret_cast<ଅଚଳ ଅକ୍ଷର *>(u8str.data()), u8str.size());
#ଅନ୍ୟଥା
        // C++17: u8string() returns std::string
        ଫେରନ୍ତୁ path.u8string();
#endif
    } ଧର (...) {
        ଫେରନ୍ତୁ std::string();
    }
}

ସଂରଚନା ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ସଂରଚନା ggml_backend_registry {
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
    ଯଦି (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } ଅନ୍ୟଥା {
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
        ପାଇଁ (auto & entry : backends) {
            ଯଦି (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    ଶୂନ୍ୟ register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        ଯଦି (!reg) {
            ଫେରନ୍ତୁ;
        }

        ପାଇଁ (auto & entry : backends) {
            ଯଦି (entry.reg == reg) {
                ଫେରନ୍ତୁ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ଗଣନା(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        ପାଇଁ (size_t i = 0; i < ggml_backend_reg_dev_ଗଣନା(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    ଶୂନ୍ୟ register_device(ggml_backend_dev_t device) {
        ପାଇଁ (auto & dev : devices) {
            ଯଦି (dev == device) {
                ଫେରନ୍ତୁ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ଅଚଳ fs::path & path, ତାର୍କିକ silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        ଯଦି (!handle) {
            ଯଦି (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            ଫେରନ୍ତୁ nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ଯଦି (score_fn && score_fn() == 0) {
            ଯଦି (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            ଫେରନ୍ତୁ nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ଯଦି (!backend_init_fn) {
            ଯଦି (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            ଫେରନ୍ତୁ nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ଯଦି (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            ଯଦି (!silent) {
                ଯଦି (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } ଅନ୍ୟଥା {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            ଫେରନ୍ତୁ nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        ଫେରନ୍ତୁ reg;
    }

    ଶୂନ୍ୟ unload_backend(ggml_backend_reg_t reg, ତାର୍କିକ silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ଅଚଳ ggml_backend_reg_entry & entry) { ଫେରନ୍ତୁ entry.reg == reg; });

        ଯଦି (it == backends.end()) {
            ଯଦି (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ଫେରନ୍ତୁ;
        }

        ଯଦି (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ଫେରନ୍ତୁ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ସ୍ଥିର ggml_backend_registry & get_reg() {
    ସ୍ଥିର ggml_backend_registry reg;
    ଫେରନ୍ତୁ reg;
}

// Internal API
ଶୂନ୍ୟ ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

ଶୂନ୍ୟ ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
ସ୍ଥିର ତାର୍କିକ striequals(ଅଚଳ ଅକ୍ଷର * a, ଅଚଳ ଅକ୍ଷର * b) {
    ପାଇଁ (; *a && *b; a++, b++) {
        ଯଦି (std::tolower(*a) != std::tolower(*b)) {
            ଫେରନ୍ତୁ ମିଥ୍ୟା;
        }
    }
    ଫେରନ୍ତୁ *a == *b;
}

size_t ggml_backend_reg_ଗଣନା() {
    ଫେରନ୍ତୁ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_ଗଣନା());
    ଫେରନ୍ତୁ get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ଅଚଳ ଅକ୍ଷର * name) {
    ପାଇଁ (size_t i = 0; i < ggml_backend_reg_ଗଣନା(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ଯଦି (striequals(ggml_backend_reg_name(reg), name)) {
            ଫେରନ୍ତୁ reg;
        }
    }
    ଫେରନ୍ତୁ nullptr;
}

// Device enumeration
size_t ggml_backend_dev_ଗଣନା() {
    ଫେରନ୍ତୁ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_ଗଣନା());
    ଫେରନ୍ତୁ get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ଅଚଳ ଅକ୍ଷର * name) {
    ପାଇଁ (size_t i = 0; i < ggml_backend_dev_ଗଣନା(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ଯଦି (striequals(ggml_backend_dev_name(dev), name)) {
            ଫେରନ୍ତୁ dev;
        }
    }
    ଫେରନ୍ତୁ nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ଗଣନା ggml_backend_dev_type type) {
    ପାଇଁ (size_t i = 0; i < ggml_backend_dev_ଗଣନା(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ଯଦି (ggml_backend_dev_type(dev) == type) {
            ଫେରନ୍ତୁ dev;
        }
    }
    ଫେରନ୍ତୁ nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ଅଚଳ ଅକ୍ଷର * name, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ଯଦି (!dev) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ଗଣନା ggml_backend_dev_type type, ଅଚଳ ଅକ୍ଷର * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ଯଦି (!dev) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(ଶୂନ୍ୟ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ଯଦି (!dev) {
        ଫେରନ୍ତୁ nullptr;
    }
    ଫେରନ୍ତୁ ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ଅଚଳ ଅକ୍ଷର * path) {
    ଫେରନ୍ତୁ get_reg().load_backend(path, ମିଥ୍ୟା);
}

ଶୂନ୍ୟ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, ସତ୍ୟ);
}

ସ୍ଥିର fs::path get_executable_path() {
#ଯଦି defined(__APPLE__)
    // get executable path
    std::vector<ଅକ୍ଷର> path;
    uint32_t size;
    ଯେତେବେଳେ (ସତ୍ୟ) {
        size = path.size();
        ଯଦି (_NSGetExecutablePath(path.data(), &size) == 0) {
            ଭାଙ୍ଗ;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    ଯଦି (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ଫେରନ୍ତୁ base_path + "/";
#ନଚେତଯଦି defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<ଅକ୍ଷର> path(1024);
    ଯେତେବେଳେ (ସତ୍ୟ) {
        // get executable path
#    ଯଦି defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        ଯଦି (len == -1) {
            ଭାଙ୍ଗ;
        }
        ଯଦି (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            ଯଦି (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            ଭାଙ୍ଗ;
        }
        path.resize(path.size() * 2);
    }

    ଫେରନ୍ତୁ base_path + "/";
#ନଚେତଯଦି defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(ଖାଲି, path.data(), path.size());
    ଯଦି (len == 0) {
        ଫେରନ୍ତୁ {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    ଯଦି (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ଫେରନ୍ତୁ base_path + L"\\";
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ {};
#endif
}

ସ୍ଥିର fs::path backend_filename_prefix() {
#ifdef _WIN32
    ଫେରନ୍ତୁ fs::u8path("ggml-");
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ fs::u8path("libggml-");
#endif
}

ସ୍ଥିର fs::path backend_filename_extension() {
#ifdef _WIN32
    ଫେରନ୍ତୁ fs::u8path(".dll");
#ଅନ୍ୟଥା
    ଫେରନ୍ତୁ fs::u8path(".so");
#endif
}

ସ୍ଥିର ggml_backend_reg_t ggml_backend_load_best(ଅଚଳ ଅକ୍ଷର * name, ତାର୍କିକ silent, ଅଚଳ ଅକ୍ଷର * user_ସନ୍ଧାନ_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ଅଚଳ fs::path name_path = fs::u8path(name);
    ଅଚଳ fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ଅଚଳ fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> ସନ୍ଧାନ_paths;
    ଯଦି (user_ସନ୍ଧାନ_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        ସନ୍ଧାନ_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        ସନ୍ଧାନ_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ଅଚଳ fs::path cwd = fs::current_path(cwd_ec);
        ଯଦି (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } ଅନ୍ୟଥା {
            ସନ୍ଧାନ_paths.push_back(cwd);
        }
    } ଅନ୍ୟଥା {
        ସନ୍ଧାନ_paths.push_back(fs::u8path(user_ସନ୍ଧାନ_path));
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_score = 0;
    fs::path best_path;
    std::error_code ec;

    ପାଇଁ (ଅଚଳ auto & ସନ୍ଧାନ_path : ସନ୍ଧାନ_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(ସନ୍ଧାନ_path, dir_ec);

        ପାଇଁ (ଅଚଳ fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ଅଚଳ auto & entry = *dir_it;
            ଯଦି (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                ଯଦି (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    ଯଦି (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    ଯଦି (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ଯଦି (score_fn) {
                            ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            ଯଦି (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } ଅନ୍ୟଥା {
                            ଯଦି (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ଯଦି (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(ସନ୍ଧାନ_path).c_str(), dir_ec.message().c_str());
        }
    }

    ଯଦି (best_score == 0) {
        // try to load the base backend
        ପାଇଁ (ଅଚଳ auto & ସନ୍ଧାନ_path : ସନ୍ଧାନ_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = ସନ୍ଧାନ_path / filename;
            ଯଦି (std::error_code ec; fs::exists(path, ec)) {
                ଫେରନ୍ତୁ get_reg().load_backend(path, silent);
            } ଅନ୍ୟଥା {
                ଯଦି (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        ଫେରନ୍ତୁ nullptr;
    }

    ଫେରନ୍ତୁ get_reg().load_backend(best_path, silent);
}

ଶୂନ୍ୟ ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

ଶୂନ୍ୟ ggml_backend_load_all_from_path(ଅଚଳ ଅକ୍ଷର * dir_path) {
#ifdef NDEBUG
    ତାର୍କିକ silent = ସତ୍ୟ;
#ଅନ୍ୟଥା
    ତାର୍କିକ silent = ମିଥ୍ୟା;
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
    ଅଚଳ ଅକ୍ଷର * backend_path = std::getenv("GGML_BACKEND_PATH");
    ଯଦି (backend_path) {
        ggml_backend_load(backend_path);
    }
}
