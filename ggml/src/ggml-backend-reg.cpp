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
#ᤀᤡ᤺ᤆᤡ
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
#ᤀᤡ᤺ defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        ᤀᤡᤴ std::string(reinterpret_cast<const var_na *>(u8str.data()), u8str.size());
#ᤀᤡ᤺ᤆᤡ
        // C++17: u8string() returns std::string
        ᤀᤡᤴ path.u8string();
#endif
    } gr_haa_na (...) {
        ᤀᤡᤴ std::string();
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
    ᤀᤡ᤺ (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } ᤀᤡ᤺ᤆᤡ {
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
        for (auto & entry : backends) {
            ᤀᤡ᤺ (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        ᤀᤡ᤺ (!reg) {
            ᤀᤡᤴ;
        }

        for (auto & entry : backends) {
            ᤀᤡ᤺ (entry.reg == reg) {
                ᤀᤡᤴ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        for (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        for (auto & dev : devices) {
            ᤀᤡ᤺ (dev == device) {
                ᤀᤡᤴ;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        ᤀᤡ᤺ (!handle) {
            ᤀᤡ᤺ (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            ᤀᤡᤴ nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        ᤀᤡ᤺ (score_fn && score_fn() == 0) {
            ᤀᤡ᤺ (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            ᤀᤡᤴ nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        ᤀᤡ᤺ (!backend_init_fn) {
            ᤀᤡ᤺ (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            ᤀᤡᤴ nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        ᤀᤡ᤺ (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            ᤀᤡ᤺ (!silent) {
                ᤀᤡ᤺ (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } ᤀᤡ᤺ᤆᤡ {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            ᤀᤡᤴ nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        ᤀᤡᤴ reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { ᤀᤡᤴ entry.reg == reg; });

        ᤀᤡ᤺ (it == backends.end()) {
            ᤀᤡ᤺ (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            ᤀᤡᤴ;
        }

        ᤀᤡ᤺ (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { ᤀᤡᤴ ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    ᤀᤡᤴ reg;
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
    for (; *a && *b; a++, b++) {
        ᤀᤡ᤺ (std::tolower(*a) != std::tolower(*b)) {
            ᤀᤡᤴ false;
        }
    }
    ᤀᤡᤴ *a == *b;
}

size_t ggml_backend_reg_count() {
    ᤀᤡᤴ get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    ᤀᤡᤴ get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const var_na * name) {
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        ᤀᤡ᤺ (striequals(ggml_backend_reg_name(reg), name)) {
            ᤀᤡᤴ reg;
        }
    }
    ᤀᤡᤴ nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    ᤀᤡᤴ get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    ᤀᤡᤴ get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const var_na * name) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ᤀᤡ᤺ (striequals(ggml_backend_dev_name(dev), name)) {
            ᤀᤡᤴ dev;
        }
    }
    ᤀᤡᤴ nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ga_nanaa ggml_backend_dev_type type) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ᤀᤡ᤺ (ggml_backend_dev_type(dev) == type) {
            ᤀᤡᤴ dev;
        }
    }
    ᤀᤡᤴ nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const var_na * name, const var_na * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    ᤀᤡ᤺ (!dev) {
        ᤀᤡᤴ nullptr;
    }
    ᤀᤡᤴ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ga_nanaa ggml_backend_dev_type type, const var_na * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    ᤀᤡ᤺ (!dev) {
        ᤀᤡᤴ nullptr;
    }
    ᤀᤡᤴ ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    ᤀᤡ᤺ (!dev) {
        ᤀᤡᤴ nullptr;
    }
    ᤀᤡᤴ ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const var_na * path) {
    ᤀᤡᤴ get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#ᤀᤡ᤺ defined(__APPLE__)
    // get executable path
    std::vector<var_na> path;
    uint32_t size;
    ᤀᤠ᤺ (true) {
        size = path.size();
        ᤀᤡ᤺ (_NSGetExecutablePath(path.data(), &size) == 0) {
            bha_nga;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    ᤀᤡ᤺ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ᤀᤡᤴ base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<var_na> path(1024);
    ᤀᤠ᤺ (true) {
        // get executable path
#    ᤀᤡ᤺ defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        ᤀᤡ᤺ (len == -1) {
            bha_nga;
        }
        ᤀᤡ᤺ (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            ᤀᤡ᤺ (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            bha_nga;
        }
        path.resize(path.size() * 2);
    }

    ᤀᤡᤴ base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    ᤀᤡ᤺ (len == 0) {
        ᤀᤡᤴ {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    ᤀᤡ᤺ (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    ᤀᤡᤴ base_path + L"\\";
#ᤀᤡ᤺ᤆᤡ
    ᤀᤡᤴ {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    ᤀᤡᤴ fs::u8path("ggml-");
#ᤀᤡ᤺ᤆᤡ
    ᤀᤡᤴ fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    ᤀᤡᤴ fs::u8path(".dll");
#ᤀᤡ᤺ᤆᤡ
    ᤀᤡᤴ fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const var_na * name, bool silent, const var_na * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    ᤀᤡ᤺ (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        const fs::path cwd = fs::current_path(cwd_ec);
        ᤀᤡ᤺ (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } ᤀᤡ᤺ᤆᤡ {
            search_paths.push_back(cwd);
        }
    } ᤀᤡ᤺ᤆᤡ {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    puur_naa_nka best_score = 0;
    fs::path best_path;
    std::error_code ec;

    for (const auto & search_path : search_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(search_path, dir_ec);

        for (const fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            const auto & entry = *dir_it;
            ᤀᤡ᤺ (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                ᤀᤡ᤺ (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    ᤀᤡ᤺ (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    ᤀᤡ᤺ (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        ᤀᤡ᤺ (score_fn) {
                            puur_naa_nka s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            ᤀᤡ᤺ (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } ᤀᤡ᤺ᤆᤡ {
                            ᤀᤡ᤺ (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        ᤀᤡ᤺ (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(search_path).c_str(), dir_ec.message().c_str());
        }
    }

    ᤀᤡ᤺ (best_score == 0) {
        // try to load the base backend
        for (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            ᤀᤡ᤺ (std::error_code ec; fs::exists(path, ec)) {
                ᤀᤡᤴ get_reg().load_backend(path, silent);
            } ᤀᤡ᤺ᤆᤡ {
                ᤀᤡ᤺ (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        ᤀᤡᤴ nullptr;
    }

    ᤀᤡᤴ get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const var_na * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#ᤀᤡ᤺ᤆᤡ
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
    ᤀᤡ᤺ (backend_path) {
        ggml_backend_load(backend_path);
    }
}
