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
#아니면
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
#만약 defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        const std::u8string u8str = path.u8string();
        반환 std::string(reinterpret_cast<const char *>(u8str.data()), u8str.size());
#아니면
        // C++17: u8string() returns std::string
        반환 path.u8string();
#endif
    } catch (...) {
        반환 std::string();
    }
}

struct ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

struct ggml_backend_registry {
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
    만약 (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } 아니면 {
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
        위해 (auto & entry : backends) {
            만약 (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    void register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        만약 (!reg) {
            반환;
        }

        위해 (auto & entry : backends) {
            만약 (entry.reg == reg) {
                반환;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_count(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        위해 (size_t i = 0; i < ggml_backend_reg_dev_count(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    void register_device(ggml_backend_dev_t device) {
        위해 (auto & dev : devices) {
            만약 (dev == device) {
                반환;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(const fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        만약 (!handle) {
            만약 (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            반환 nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        만약 (score_fn && score_fn() == 0) {
            만약 (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            반환 nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        만약 (!backend_init_fn) {
            만약 (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            반환 nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        만약 (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            만약 (!silent) {
                만약 (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } 아니면 {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            반환 nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        반환 reg;
    }

    void unload_backend(ggml_backend_reg_t reg, bool silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](const ggml_backend_reg_entry & entry) { 반환 entry.reg == reg; });

        만약 (it == backends.end()) {
            만약 (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            반환;
        }

        만약 (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { 반환 ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

static ggml_backend_registry & get_reg() {
    static ggml_backend_registry reg;
    반환 reg;
}

// Internal API
void ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

void ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
static bool striequals(const char * a, const char * b) {
    위해 (; *a && *b; a++, b++) {
        만약 (std::tolower(*a) != std::tolower(*b)) {
            반환 false;
        }
    }
    반환 *a == *b;
}

size_t ggml_backend_reg_count() {
    반환 get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_count());
    반환 get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(const char * name) {
    위해 (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        만약 (striequals(ggml_backend_reg_name(reg), name)) {
            반환 reg;
        }
    }
    반환 nullptr;
}

// Device enumeration
size_t ggml_backend_dev_count() {
    반환 get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_count());
    반환 get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(const char * name) {
    위해 (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        만약 (striequals(ggml_backend_dev_name(dev), name)) {
            반환 dev;
        }
    }
    반환 nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(enum ggml_backend_dev_type type) {
    위해 (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        만약 (ggml_backend_dev_type(dev) == type) {
            반환 dev;
        }
    }
    반환 nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(const char * name, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    만약 (!dev) {
        반환 nullptr;
    }
    반환 ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(enum ggml_backend_dev_type type, const char * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    만약 (!dev) {
        반환 nullptr;
    }
    반환 ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(void) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    만약 (!dev) {
        반환 nullptr;
    }
    반환 ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(const char * path) {
    반환 get_reg().load_backend(path, false);
}

void ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

static fs::path get_executable_path() {
#만약 defined(__APPLE__)
    // get executable path
    std::vector<char> path;
    uint32_t size;
    동안 (true) {
        size = path.size();
        만약 (_NSGetExecutablePath(path.data(), &size) == 0) {
            break;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    만약 (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    반환 base_path + "/";
#elif defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<char> path(1024);
    동안 (true) {
        // get executable path
#    만약 defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        만약 (len == -1) {
            break;
        }
        만약 (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            만약 (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            break;
        }
        path.resize(path.size() * 2);
    }

    반환 base_path + "/";
#elif defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
    만약 (len == 0) {
        반환 {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    만약 (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    반환 base_path + L"\\";
#아니면
    반환 {};
#endif
}

static fs::path backend_filename_prefix() {
#ifdef _WIN32
    반환 fs::u8path("ggml-");
#아니면
    반환 fs::u8path("libggml-");
#endif
}

static fs::path backend_filename_extension() {
#ifdef _WIN32
    반환 fs::u8path(".dll");
#아니면
    반환 fs::u8path(".so");
#endif
}

static ggml_backend_reg_t ggml_backend_load_best(const char * name, bool silent, const char * user_search_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    const fs::path name_path = fs::u8path(name);
    const fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    const fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> search_paths;
    만약 (user_search_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        search_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        search_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        const fs::path cwd = fs::current_path(cwd_ec);
        만약 (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } 아니면 {
            search_paths.push_back(cwd);
        }
    } 아니면 {
        search_paths.push_back(fs::u8path(user_search_path));
    }

    정수 best_score = 0;
    fs::path best_path;
    std::error_code ec;

    위해 (const auto & search_path : search_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(search_path, dir_ec);

        위해 (const fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            const auto & entry = *dir_it;
            만약 (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                만약 (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    만약 (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    만약 (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        만약 (score_fn) {
                            정수 s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            만약 (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } 아니면 {
                            만약 (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        만약 (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(search_path).c_str(), dir_ec.message().c_str());
        }
    }

    만약 (best_score == 0) {
        // try to load the base backend
        위해 (const auto & search_path : search_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = search_path / filename;
            만약 (std::error_code ec; fs::exists(path, ec)) {
                반환 get_reg().load_backend(path, silent);
            } 아니면 {
                만약 (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        반환 nullptr;
    }

    반환 get_reg().load_backend(best_path, silent);
}

void ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

void ggml_backend_load_all_from_path(const char * dir_path) {
#ifdef NDEBUG
    bool silent = true;
#아니면
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
    const char * backend_path = std::getenv("GGML_BACKEND_PATH");
    만약 (backend_path) {
        ggml_backend_load(backend_path);
    }
}
