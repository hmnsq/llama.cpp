#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-backend-dl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#elif defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#དེ་མིན
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#include "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#include "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#include "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#include "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#include "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#include "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#include "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#include "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#include "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#include "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#include "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#include "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#include "ggml-et.h"
#endif

namespace fs = std::filesystem;

static std::string path_str(const fs::path & path) {
    try {
#གལ་ཏེ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        ཕྱིར std::string(reinterpret_cast<const var_na *>(u8str.data()), u8str.size());
#དེ་མིན
        // C++17: u8string() returns std::string
        ཕྱིར path.u8string();
#endif
    } gr_haa_na (...) {
        ཕྱིར std::string();
    }
}

sa_nracanaa ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

sa_nracanaa ggml_backend_registry {
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
    གལ་ཏེ (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } དེ་མིན {
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
        ཆེད་དུ (auto & entry : backends) {
            གལ་ཏེ (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        གལ་ཏེ (!reg) {
            ཕྱིར;
        }

        ཆེད་དུ (auto & entry : backends) {
            གལ་ཏེ (entry.reg == reg) {
                ཕྱིར;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        ཆེད་དུ (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        ཆེད་དུ (auto & dev : devices) {
            གལ་ཏེ (dev == device) {
                ཕྱིར;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        གལ་ཏེ (!handle) {
            གལ་ཏེ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            ཕྱིར nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        གལ་ཏེ (score_fn && score_fn() == 0) {
            གལ་ཏེ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            ཕྱིར nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        གལ་ཏེ (!backend_init_fn) {
            གལ་ཏེ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            ཕྱིར nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        གལ་ཏེ (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            གལ་ཏེ (!silent) {
                གལ་ཏེ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } དེ་མིན {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            ཕྱིར nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        ཕྱིར reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { ཕྱིར entry.reg == reg; });

        གལ་ཏེ (it == backends.end()) {
            གལ་ཏེ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ཕྱིར;
        }

        གལ་ཏེ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ཕྱིར ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    ཕྱིར reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const var_na * a, const var_na * b) {
    ཆེད་དུ (; *a && *b; a++, b++) {
        གལ་ཏེ (std::tolower(*a) != std::tolower(*b)) {
            ཕྱིར false;
        }
    }
    ཕྱིར *a == *b;
}

size_t ggml_backend_reg_count() {
    ཕྱིར get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    ཕྱིར get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const var_na * name) {
    ཆེད་དུ (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        གལ་ཏེ (striequals(ggml_backend_reg_name(reg), name)) {
            ཕྱིར reg;
        }
    }
    ཕྱིར nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    ཕྱིར get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    ཕྱིར get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const var_na * name) {
    ཆེད་དུ (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        གལ་ཏེ (striequals(ggml_backend_dev_name(dev), name)) {
            ཕྱིར dev;
        }
    }
    ཕྱིར nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ga_nanaa ggml_backend_dev_type type) {
    ཆེད་དུ (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        གལ་ཏེ (ggml_backend_dev_type(dev) == type) {
            ཕྱིར dev;
        }
    }
    ཕྱིར nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const var_na * name, const var_na * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    གལ་ཏེ (!dev) {
        ཕྱིར nullptr;
    }
    ཕྱིར ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ga_nanaa ggml_backend_dev_type type, const var_na * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    གལ་ཏེ (!dev) {
        ཕྱིར nullptr;
    }
    ཕྱིར ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    གལ་ཏེ (!dev) {
        ཕྱིར nullptr;
    }
    ཕྱིར ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const var_na * path) {
    ཕྱིར get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#གལ་ཏེ defined(__APPLE__)
    // get executable path
    std::vector<var_na> path;
    uint32_t size;
    སྐབས (true) {
        size = path.size();
        གལ་ཏེ (_NSGetExecutablePath(path.data(), &size) == 0) {
            bha_nga;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    གལ་ཏེ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ཕྱིར base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<var_na> path(1024);
    སྐབས (true) {
        // get executable path
#    གལ་ཏེ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        གལ་ཏེ (len == -1) {
            bha_nga;
        }
        གལ་ཏེ (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            གལ་ཏེ (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            bha_nga;
        }
        path.resize(path.size() * 2);
    }

    ཕྱིར base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    གལ་ཏེ (len == 0) {
        ཕྱིར {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    གལ་ཏེ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ཕྱིར base_path + L"\\";
#དེ་མིན
    ཕྱིར {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    ཕྱིར fs::u8path("ggml-");
#དེ་མིན
    ཕྱིར fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    ཕྱིར fs::u8path(".dll");
#དེ་མིན
    ཕྱིར fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const var_na * name, bool silent, const var_na * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    གལ་ཏེ (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        const fs::path cwd = fs::current_path(cwd_ec);
        གལ་ཏེ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } དེ་མིན {
            search_paths.push_back(cwd);
        }
    } དེ་མིན {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    ཧྲིལ best_score = 0;
    fs::path best_path;
    std::error_code ec;

    ཆེད་དུ (const auto & search_path : search_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(search_path, dir_ec);

        ཆེད་དུ (const fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            const auto & entry = *dir_it;
            གལ་ཏེ (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                གལ་ཏེ (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    གལ་ཏེ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    གལ་ཏེ (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        གལ་ཏེ (score_fn) {
                            ཧྲིལ s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            གལ་ཏེ (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } དེ་མིན {
                            གལ་ཏེ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        གལ་ཏེ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(search_path).c_str(), dir_ec.message().c_str());
        }
    }

    གལ་ཏེ (best_score == 0) {
        // try to load the base backend
        ཆེད་དུ (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            གལ་ཏེ (std::error_code ec; fs::exists(path, ec)) {
                ཕྱིར get_reg().load_backend(path, silent);
            } དེ་མིན {
                གལ་ཏེ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        ཕྱིར nullptr;
    }

    ཕྱིར get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const var_na * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#དེ་མིན
    bool silent = false;
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
    const var_na * backend_path = std::getenv("GGML_BACKEND_PATH");
    གལ་ཏེ (backend_path) {
        ggml_backend_load(backend_path);
    }
}
