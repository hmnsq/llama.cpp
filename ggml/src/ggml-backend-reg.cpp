#অন্তর্ভাব "ggml-backend-impl.h"
#অন্তর্ভাব "ggml-backend.h"
#অন্তর্ভাব "ggml-backend-dl.h"
#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstring>
#অন্তর্ভাব <filesystem>
#অন্তর্ভাব <memory>
#অন্তর্ভাব <string>
#অন্তর্ভাব <type_traits>
#অন্তর্ভাব <vector>
#অন্তর্ভাব <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#নতুবা_যদি defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#অন্যথা
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#অন্তর্ভাব "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#অন্তর্ভাব "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#অন্তর্ভাব "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#অন্তর্ভাব "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#অন্তর্ভাব "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#অন্তর্ভাব "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#অন্তর্ভাব "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#অন্তর্ভাব "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#অন্তর্ভাব "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#অন্তর্ভাব "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#অন্তর্ভাব "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#অন্তর্ভাব "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#অন্তর্ভাব "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#অন্তর্ভাব "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#অন্তর্ভাব "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#অন্তর্ভাব "ggml-et.h"
#endif

নামাকাশ fs = std::filesystem;

স্থির std::string path_str(ধ্রুবক fs::path & path) {
    চেষ্টা {
#যদি defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ধ্রুবক std::u8string u8str = path.u8string();
        ফেরত std::string(reinterpret_cast<ধ্রুবক অক্ষর *>(u8str.তথ্য()), u8str.size());
#অন্যথা
        // C++17: u8string() returns std::string
        ফেরত path.u8string();
#endif
    } ধরো (...) {
        ফেরত std::string();
    }
}

কাঠামো ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

কাঠামো ggml_backend_registry {
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
    যদি (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } অন্যথা {
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
        জন্য (auto & entry : backends) {
            যদি (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    শূন্য register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        যদি (!reg) {
            ফেরত;
        }

        জন্য (auto & entry : backends) {
            যদি (entry.reg == reg) {
                ফেরত;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_গুনতি(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        জন্য (size_t i = 0; i < ggml_backend_reg_dev_গুনতি(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    শূন্য register_device(ggml_backend_dev_t device) {
        জন্য (auto & dev : devices) {
            যদি (dev == device) {
                ফেরত;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ধ্রুবক fs::path & path, যৌক্তিক silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        যদি (!handle) {
            যদি (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            ফেরত nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        যদি (score_fn && score_fn() == 0) {
            যদি (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            ফেরত nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        যদি (!backend_init_fn) {
            যদি (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            ফেরত nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        যদি (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            যদি (!silent) {
                যদি (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } অন্যথা {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            ফেরত nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        ফেরত reg;
    }

    শূন্য unload_backend(ggml_backend_reg_t reg, যৌক্তিক silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ধ্রুবক ggml_backend_reg_entry & entry) { ফেরত entry.reg == reg; });

        যদি (it == backends.end()) {
            যদি (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ফেরত;
        }

        যদি (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ফেরত ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

স্থির ggml_backend_registry & get_reg() {
    স্থির ggml_backend_registry reg;
    ফেরত reg;
}

// Internal API
শূন্য ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

শূন্য ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
স্থির যৌক্তিক striequals(ধ্রুবক অক্ষর * a, ধ্রুবক অক্ষর * b) {
    জন্য (; *a && *b; a++, b++) {
        যদি (std::tolower(*a) != std::tolower(*b)) {
            ফেরত মিথ্যা;
        }
    }
    ফেরত *a == *b;
}

size_t ggml_backend_reg_গুনতি() {
    ফেরত get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t সূচক) {
    GGML_ASSERT(সূচক < ggml_backend_reg_গুনতি());
    ফেরত get_reg().backends[সূচক].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ধ্রুবক অক্ষর * name) {
    জন্য (size_t i = 0; i < ggml_backend_reg_গুনতি(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        যদি (striequals(ggml_backend_reg_name(reg), name)) {
            ফেরত reg;
        }
    }
    ফেরত nullptr;
}

// Device enumeration
size_t ggml_backend_dev_গুনতি() {
    ফেরত get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t সূচক) {
    GGML_ASSERT(সূচক < ggml_backend_dev_গুনতি());
    ফেরত get_reg().devices[সূচক];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ধ্রুবক অক্ষর * name) {
    জন্য (size_t i = 0; i < ggml_backend_dev_গুনতি(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        যদি (striequals(ggml_backend_dev_name(dev), name)) {
            ফেরত dev;
        }
    }
    ফেরত nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(গণনা ggml_backend_dev_type type) {
    জন্য (size_t i = 0; i < ggml_backend_dev_গুনতি(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        যদি (ggml_backend_dev_type(dev) == type) {
            ফেরত dev;
        }
    }
    ফেরত nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ধ্রুবক অক্ষর * name, ধ্রুবক অক্ষর * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    যদি (!dev) {
        ফেরত nullptr;
    }
    ফেরত ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(গণনা ggml_backend_dev_type type, ধ্রুবক অক্ষর * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    যদি (!dev) {
        ফেরত nullptr;
    }
    ফেরত ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(শূন্য) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    যদি (!dev) {
        ফেরত nullptr;
    }
    ফেরত ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ধ্রুবক অক্ষর * path) {
    ফেরত get_reg().load_backend(path, মিথ্যা);
}

শূন্য ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, সত্য);
}

স্থির fs::path get_executable_path() {
#যদি defined(__APPLE__)
    // get executable path
    std::vector<অক্ষর> path;
    uint32_t size;
    যখন (সত্য) {
        size = path.size();
        যদি (_NSGetExecutablePath(path.তথ্য(), &size) == 0) {
            বিরতি;
        }
        path.resize(size);
    }
    std::string base_path(path.তথ্য(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    যদি (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ফেরত base_path + "/";
#নতুবা_যদি defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<অক্ষর> path(1024);
    যখন (সত্য) {
        // get executable path
#    যদি defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.তথ্য(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.তথ্য(), path.size());
#    endif
        যদি (len == -1) {
            বিরতি;
        }
        যদি (len < (ssize_t) path.size()) {
            base_path = std::string(path.তথ্য(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            যদি (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            বিরতি;
        }
        path.resize(path.size() * 2);
    }

    ফেরত base_path + "/";
#নতুবা_যদি defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(ফাঁকা, path.তথ্য(), path.size());
    যদি (len == 0) {
        ফেরত {};
    }
    std::wstring base_path(path.তথ্য(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    যদি (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ফেরত base_path + L"\\";
#অন্যথা
    ফেরত {};
#endif
}

স্থির fs::path backend_filename_prefix() {
#ifdef _WIN32
    ফেরত fs::u8path("ggml-");
#অন্যথা
    ফেরত fs::u8path("libggml-");
#endif
}

স্থির fs::path backend_filename_extension() {
#ifdef _WIN32
    ফেরত fs::u8path(".dll");
#অন্যথা
    ফেরত fs::u8path(".so");
#endif
}

স্থির ggml_backend_reg_t ggml_backend_load_best(ধ্রুবক অক্ষর * name, যৌক্তিক silent, ধ্রুবক অক্ষর * user_অনুসন্ধান_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ধ্রুবক fs::path name_path = fs::u8path(name);
    ধ্রুবক fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ধ্রুবক fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> অনুসন্ধান_paths;
    যদি (user_অনুসন্ধান_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        অনুসন্ধান_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        অনুসন্ধান_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ধ্রুবক fs::path cwd = fs::current_path(cwd_ec);
        যদি (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } অন্যথা {
            অনুসন্ধান_paths.push_back(cwd);
        }
    } অন্যথা {
        অনুসন্ধান_paths.push_back(fs::u8path(user_অনুসন্ধান_path));
    }

    পূর্ণসংখ্যা best_score = 0;
    fs::path best_path;
    std::error_code ec;

    জন্য (ধ্রুবক auto & অনুসন্ধান_path : অনুসন্ধান_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(অনুসন্ধান_path, dir_ec);

        জন্য (ধ্রুবক fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ধ্রুবক auto & entry = *dir_it;
            যদি (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                যদি (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    যদি (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    যদি (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        যদি (score_fn) {
                            পূর্ণসংখ্যা s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            যদি (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } অন্যথা {
                            যদি (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        যদি (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(অনুসন্ধান_path).c_str(), dir_ec.message().c_str());
        }
    }

    যদি (best_score == 0) {
        // try to load the base backend
        জন্য (ধ্রুবক auto & অনুসন্ধান_path : অনুসন্ধান_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = অনুসন্ধান_path / filename;
            যদি (std::error_code ec; fs::exists(path, ec)) {
                ফেরত get_reg().load_backend(path, silent);
            } অন্যথা {
                যদি (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        ফেরত nullptr;
    }

    ফেরত get_reg().load_backend(best_path, silent);
}

শূন্য ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

শূন্য ggml_backend_load_all_from_path(ধ্রুবক অক্ষর * dir_path) {
#ifdef NDEBUG
    যৌক্তিক silent = সত্য;
#অন্যথা
    যৌক্তিক silent = মিথ্যা;
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
    ধ্রুবক অক্ষর * backend_path = std::getenv("GGML_BACKEND_PATH");
    যদি (backend_path) {
        ggml_backend_load(backend_path);
    }
}
