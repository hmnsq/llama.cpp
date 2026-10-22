#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-backend-dl.h"
#ضمن "ggml-impl.h"
#ضمن <algorithm>
#ضمن <cstring>
#ضمن <filesystem>
#ضمن <memory>
#ضمن <string>
#ضمن <type_traits>
#ضمن <vector>
#ضمن <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وإلاإذا defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#ضمن "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#ضمن "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#ضمن "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#ضمن "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#ضمن "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#ضمن "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#ضمن "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#ضمن "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#ضمن "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#ضمن "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#ضمن "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#ضمن "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#ضمن "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#ضمن "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#ضمن "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#ضمن "ggml-et.h"
#endif

نطاق fs = std::filesystem;

ساكن std::string path_str(ثابت fs::path & path) {
    حاول {
#إذا defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        أرجع std::string(reinterpret_cast<ثابت حرف *>(u8str.data()), u8str.size());
#وإلا
        // C++17: u8string() returns std::string
        أرجع path.u8string();
#endif
    } امسك (...) {
        أرجع std::string();
    }
}

بنية ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

بنية ggml_backend_registry {
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
    إذا (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } وإلا {
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
        لكل (auto & entry : backends) {
            إذا (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    فراغ register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        إذا (!reg) {
            أرجع;
        }

        لكل (auto & entry : backends) {
            إذا (entry.reg == reg) {
                أرجع;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_كدي(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        لكل (size_t i = 0; i < ggml_backend_reg_dev_كدي(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    فراغ register_device(ggml_backend_dev_t device) {
        لكل (auto & dev : devices) {
            إذا (dev == device) {
                أرجع;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ثابت fs::path & path, منطقي silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        إذا (!handle) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            أرجع nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        إذا (score_fn && score_fn() == 0) {
            إذا (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            أرجع nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        إذا (!backend_init_fn) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            أرجع nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        إذا (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            إذا (!silent) {
                إذا (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } وإلا {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            أرجع nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        أرجع reg;
    }

    فراغ unload_backend(ggml_backend_reg_t reg, منطقي silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_entry & entry) { أرجع entry.reg == reg; });

        إذا (it == backends.end()) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            أرجع;
        }

        إذا (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { أرجع ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ساكن ggml_backend_registry & get_reg() {
    ساكن ggml_backend_registry reg;
    أرجع reg;
}

// Internal API
فراغ ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

فراغ ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
ساكن منطقي striequals(ثابت حرف * a, ثابت حرف * b) {
    لكل (; *a && *b; a++, b++) {
        إذا (std::tolower(*a) != std::tolower(*b)) {
            أرجع خطأ;
        }
    }
    أرجع *a == *b;
}

size_t ggml_backend_reg_كدي() {
    أرجع get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_كدي());
    أرجع get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_reg_كدي(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        إذا (striequals(ggml_backend_reg_name(reg), name)) {
            أرجع reg;
        }
    }
    أرجع nullptr;
}

// Device enumeration
size_t ggml_backend_dev_كدي() {
    أرجع get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_كدي());
    أرجع get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_dev_كدي(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (striequals(ggml_backend_dev_name(dev), name)) {
            أرجع dev;
        }
    }
    أرجع nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(تعداد ggml_backend_dev_type type) {
    لكل (size_t i = 0; i < ggml_backend_dev_كدي(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (ggml_backend_dev_type(dev) == type) {
            أرجع dev;
        }
    }
    أرجع nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    إذا (!dev) {
        أرجع nullptr;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(تعداد ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    إذا (!dev) {
        أرجع nullptr;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    إذا (!dev) {
        أرجع nullptr;
    }
    أرجع ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ثابت حرف * path) {
    أرجع get_reg().load_backend(path, خطأ);
}

فراغ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

ساكن fs::path get_executable_path() {
#إذا defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    طالما (true) {
        size = path.size();
        إذا (_NSGetExecutablePath(path.data(), &size) == 0) {
            اكسر;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    إذا (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    أرجع base_path + "/";
#وإلاإذا defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<حرف> path(1024);
    طالما (true) {
        // get executable path
#    إذا defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        إذا (len == -1) {
            اكسر;
        }
        إذا (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            إذا (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            اكسر;
        }
        path.resize(path.size() * 2);
    }

    أرجع base_path + "/";
#وإلاإذا defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(عدم, path.data(), path.size());
    إذا (len == 0) {
        أرجع {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    إذا (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    أرجع base_path + L"\\";
#وإلا
    أرجع {};
#endif
}

ساكن fs::path backend_filename_prefix() {
#ifdef _WIN32
    أرجع fs::u8path("ggml-");
#وإلا
    أرجع fs::u8path("libggml-");
#endif
}

ساكن fs::path backend_filename_extension() {
#ifdef _WIN32
    أرجع fs::u8path(".dll");
#وإلا
    أرجع fs::u8path(".so");
#endif
}

ساكن ggml_backend_reg_t ggml_backend_load_best(ثابت حرف * name, منطقي silent, ثابت حرف * user_نيما_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_path = fs::u8path(name);
    ثابت fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> نيما_paths;
    إذا (user_نيما_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        نيما_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        نيما_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        إذا (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } وإلا {
            نيما_paths.push_back(cwd);
        }
    } وإلا {
        نيما_paths.push_back(fs::u8path(user_نيما_path));
    }

    صحيح best_score = 0;
    fs::path best_path;
    std::error_code ec;

    لكل (ثابت auto & نيما_path : نيما_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(نيما_path, dir_ec);

        لكل (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت auto & entry = *dir_it;
            إذا (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                إذا (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    إذا (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    إذا (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        إذا (score_fn) {
                            صحيح s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            إذا (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } وإلا {
                            إذا (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        إذا (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(نيما_path).c_str(), dir_ec.message().c_str());
        }
    }

    إذا (best_score == 0) {
        // try to load the base backend
        لكل (ثابت auto & نيما_path : نيما_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = نيما_path / filename;
            إذا (std::error_code ec; fs::exists(path, ec)) {
                أرجع get_reg().load_backend(path, silent);
            } وإلا {
                إذا (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        أرجع nullptr;
    }

    أرجع get_reg().load_backend(best_path, silent);
}

فراغ ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

فراغ ggml_backend_load_all_from_path(ثابت حرف * dir_path) {
#ifdef NDEBUG
    منطقي silent = true;
#وإلا
    منطقي silent = خطأ;
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
    ثابت حرف * backend_path = std::getenv("GGML_BACKEND_PATH");
    إذا (backend_path) {
        ggml_backend_load(backend_path);
    }
}
