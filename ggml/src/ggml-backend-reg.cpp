#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-backend-dl.h"
#شامل "ggml-impl.h"
#شامل <algorithm>
#شامل <cstring>
#شامل <filesystem>
#شامل <memory>
#شامل <string>
#شامل <type_traits>
#شامل <vector>
#شامل <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وگرنهاگر defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#وگرنه
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#شامل "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#شامل "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#شامل "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#شامل "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#شامل "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#شامل "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#شامل "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#شامل "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#شامل "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#شامل "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#شامل "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#شامل "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#شامل "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#شامل "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#شامل "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#شامل "ggml-et.h"
#endif

فضاینام fs = std::filesystem;

ایستا std::string path_str(ثابت fs::path & path) {
    تلاش {
#اگر defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        بازگشت std::string(reinterpret_cast<ثابت نویسه *>(u8str.data()), u8str.size());
#وگرنه
        // C++17: u8string() returns std::string
        بازگشت path.u8string();
#endif
    } بگیر (...) {
        بازگشت std::string();
    }
}

ساختار ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

ساختار ggml_backend_registry {
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
    اگر (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } وگرنه {
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
        برای (auto & entry : backends) {
            اگر (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    تهی register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        اگر (!reg) {
            بازگشت;
        }

        برای (auto & entry : backends) {
            اگر (entry.reg == reg) {
                بازگشت;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_شمار(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        برای (size_t i = 0; i < ggml_backend_reg_dev_شمار(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    تهی register_device(ggml_backend_dev_t device) {
        برای (auto & dev : devices) {
            اگر (dev == device) {
                بازگشت;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ثابت fs::path & path, منطقی silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        اگر (!handle) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            بازگشت nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        اگر (score_fn && score_fn() == 0) {
            اگر (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            بازگشت nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        اگر (!backend_init_fn) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            بازگشت nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        اگر (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            اگر (!silent) {
                اگر (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } وگرنه {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            بازگشت nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        بازگشت reg;
    }

    تهی unload_backend(ggml_backend_reg_t reg, منطقی silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_entry & entry) { بازگشت entry.reg == reg; });

        اگر (it == backends.end()) {
            اگر (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            بازگشت;
        }

        اگر (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { بازگشت ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ایستا ggml_backend_registry & get_reg() {
    ایستا ggml_backend_registry reg;
    بازگشت reg;
}

// Internal API
تهی ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

تهی ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
ایستا منطقی striequals(ثابت نویسه * a, ثابت نویسه * b) {
    برای (; *a && *b; a++, b++) {
        اگر (std::tolower(*a) != std::tolower(*b)) {
            بازگشت نادرست;
        }
    }
    بازگشت *a == *b;
}

size_t ggml_backend_reg_شمار() {
    بازگشت get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_شمار());
    بازگشت get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت نویسه * name) {
    برای (size_t i = 0; i < ggml_backend_reg_شمار(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        اگر (striequals(ggml_backend_reg_name(reg), name)) {
            بازگشت reg;
        }
    }
    بازگشت nullptr;
}

// Device enumeration
size_t ggml_backend_dev_شمار() {
    بازگشت get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_شمار());
    بازگشت get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت نویسه * name) {
    برای (size_t i = 0; i < ggml_backend_dev_شمار(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        اگر (striequals(ggml_backend_dev_name(dev), name)) {
            بازگشت dev;
        }
    }
    بازگشت nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(شمارش ggml_backend_dev_type type) {
    برای (size_t i = 0; i < ggml_backend_dev_شمار(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        اگر (ggml_backend_dev_type(dev) == type) {
            بازگشت dev;
        }
    }
    بازگشت nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت نویسه * name, ثابت نویسه * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    اگر (!dev) {
        بازگشت nullptr;
    }
    بازگشت ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(شمارش ggml_backend_dev_type type, ثابت نویسه * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    اگر (!dev) {
        بازگشت nullptr;
    }
    بازگشت ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(تهی) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    اگر (!dev) {
        بازگشت nullptr;
    }
    بازگشت ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ثابت نویسه * path) {
    بازگشت get_reg().load_backend(path, نادرست);
}

تهی ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, درست);
}

ایستا fs::path get_executable_path() {
#اگر defined(__APPLE__)
    // get executable path
    std::vector<نویسه> path;
    uint32_t size;
    وقتی (درست) {
        size = path.size();
        اگر (_NSGetExecutablePath(path.data(), &size) == 0) {
            بشکن;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    اگر (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    بازگشت base_path + "/";
#وگرنهاگر defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<نویسه> path(1024);
    وقتی (درست) {
        // get executable path
#    اگر defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        اگر (len == -1) {
            بشکن;
        }
        اگر (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            اگر (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            بشکن;
        }
        path.resize(path.size() * 2);
    }

    بازگشت base_path + "/";
#وگرنهاگر defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(پوچ, path.data(), path.size());
    اگر (len == 0) {
        بازگشت {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    اگر (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    بازگشت base_path + L"\\";
#وگرنه
    بازگشت {};
#endif
}

ایستا fs::path backend_filename_prefix() {
#ifdef _WIN32
    بازگشت fs::u8path("ggml-");
#وگرنه
    بازگشت fs::u8path("libggml-");
#endif
}

ایستا fs::path backend_filename_extension() {
#ifdef _WIN32
    بازگشت fs::u8path(".dll");
#وگرنه
    بازگشت fs::u8path(".so");
#endif
}

ایستا ggml_backend_reg_t ggml_backend_load_best(ثابت نویسه * name, منطقی silent, ثابت نویسه * user_جستجو_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_path = fs::u8path(name);
    ثابت fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> جستجو_paths;
    اگر (user_جستجو_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        جستجو_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        جستجو_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        اگر (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } وگرنه {
            جستجو_paths.push_back(cwd);
        }
    } وگرنه {
        جستجو_paths.push_back(fs::u8path(user_جستجو_path));
    }

    صحیح best_score = 0;
    fs::path best_path;
    std::error_code ec;

    برای (ثابت auto & جستجو_path : جستجو_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(جستجو_path, dir_ec);

        برای (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت auto & entry = *dir_it;
            اگر (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                اگر (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    اگر (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    اگر (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        اگر (score_fn) {
                            صحیح s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            اگر (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } وگرنه {
                            اگر (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        اگر (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(جستجو_path).c_str(), dir_ec.message().c_str());
        }
    }

    اگر (best_score == 0) {
        // try to load the base backend
        برای (ثابت auto & جستجو_path : جستجو_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = جستجو_path / filename;
            اگر (std::error_code ec; fs::exists(path, ec)) {
                بازگشت get_reg().load_backend(path, silent);
            } وگرنه {
                اگر (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        بازگشت nullptr;
    }

    بازگشت get_reg().load_backend(best_path, silent);
}

تهی ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

تهی ggml_backend_load_all_from_path(ثابت نویسه * dir_path) {
#ifdef NDEBUG
    منطقی silent = درست;
#وگرنه
    منطقی silent = نادرست;
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
    ثابت نویسه * backend_path = std::getenv("GGML_BACKEND_PATH");
    اگر (backend_path) {
        ggml_backend_load(backend_path);
    }
}
