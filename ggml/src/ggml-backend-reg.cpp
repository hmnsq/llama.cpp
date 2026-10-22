#અન્તર્ભાવ "ggml-backend-impl.h"
#અન્તર્ભાવ "ggml-backend.h"
#અન્તર્ભાવ "ggml-backend-dl.h"
#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <cstring>
#અન્તર્ભાવ <filesystem>
#અન્તર્ભાવ <memory>
#અન્તર્ભાવ <string>
#અન્તર્ભાવ <type_traits>
#અન્તર્ભાવ <vector>
#અન્તર્ભાવ <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#અથવાજો defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#નહીંતર
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#અન્તર્ભાવ "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#અન્તર્ભાવ "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#અન્તર્ભાવ "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#અન્તર્ભાવ "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#અન્તર્ભાવ "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#અન્તર્ભાવ "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#અન્તર્ભાવ "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#અન્તર્ભાવ "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#અન્તર્ભાવ "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#અન્તર્ભાવ "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#અન્તર્ભાવ "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#અન્તર્ભાવ "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#અન્તર્ભાવ "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#અન્તર્ભાવ "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#અન્તર્ભાવ "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#અન્તર્ભાવ "ggml-et.h"
#endif

નામાકાશ fs = std::filesystem;

સ્થિર std::string path_str(અચલ fs::path & path) {
    પ્રયાસ {
#જો defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        અચલ std::u8string u8str = path.u8string();
        પાછા std::string(reinterpret_cast<અચલ અક્ષર *>(u8str.data()), u8str.size());
#નહીંતર
        // C++17: u8string() returns std::string
        પાછા path.u8string();
#endif
    } પકડો (...) {
        પાછા std::string();
    }
}

સંરચના ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

સંરચના ggml_backend_registry {
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
    જો (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } નહીંતર {
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
        માટે (auto & entry : backends) {
            જો (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    શૂન્ય register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        જો (!reg) {
            પાછા;
        }

        માટે (auto & entry : backends) {
            જો (entry.reg == reg) {
                પાછા;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ગણતરી(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        માટે (size_t i = 0; i < ggml_backend_reg_dev_ગણતરી(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    શૂન્ય register_device(ggml_backend_dev_t device) {
        માટે (auto & dev : devices) {
            જો (dev == device) {
                પાછા;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(અચલ fs::path & path, તાર્કિક silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        જો (!handle) {
            જો (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            પાછા nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        જો (score_fn && score_fn() == 0) {
            જો (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            પાછા nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        જો (!backend_init_fn) {
            જો (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            પાછા nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        જો (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            જો (!silent) {
                જો (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } નહીંતર {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            પાછા nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        પાછા reg;
    }

    શૂન્ય unload_backend(ggml_backend_reg_t reg, તાર્કિક silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](અચલ ggml_backend_reg_entry & entry) { પાછા entry.reg == reg; });

        જો (it == backends.end()) {
            જો (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            પાછા;
        }

        જો (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { પાછા ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

સ્થિર ggml_backend_registry & get_reg() {
    સ્થિર ggml_backend_registry reg;
    પાછા reg;
}

// Internal API
શૂન્ય ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

શૂન્ય ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
સ્થિર તાર્કિક striequals(અચલ અક્ષર * a, અચલ અક્ષર * b) {
    માટે (; *a && *b; a++, b++) {
        જો (std::tolower(*a) != std::tolower(*b)) {
            પાછા ખોટું;
        }
    }
    પાછા *a == *b;
}

size_t ggml_backend_reg_ગણતરી() {
    પાછા get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_ગણતરી());
    પાછા get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(અચલ અક્ષર * name) {
    માટે (size_t i = 0; i < ggml_backend_reg_ગણતરી(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        જો (striequals(ggml_backend_reg_name(reg), name)) {
            પાછા reg;
        }
    }
    પાછા nullptr;
}

// Device enumeration
size_t ggml_backend_dev_ગણતરી() {
    પાછા get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_ગણતરી());
    પાછા get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(અચલ અક્ષર * name) {
    માટે (size_t i = 0; i < ggml_backend_dev_ગણતરી(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        જો (striequals(ggml_backend_dev_name(dev), name)) {
            પાછા dev;
        }
    }
    પાછા nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(ગણના ggml_backend_dev_type type) {
    માટે (size_t i = 0; i < ggml_backend_dev_ગણતરી(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        જો (ggml_backend_dev_type(dev) == type) {
            પાછા dev;
        }
    }
    પાછા nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(અચલ અક્ષર * name, અચલ અક્ષર * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    જો (!dev) {
        પાછા nullptr;
    }
    પાછા ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(ગણના ggml_backend_dev_type type, અચલ અક્ષર * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    જો (!dev) {
        પાછા nullptr;
    }
    પાછા ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(શૂન્ય) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    જો (!dev) {
        પાછા nullptr;
    }
    પાછા ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(અચલ અક્ષર * path) {
    પાછા get_reg().load_backend(path, ખોટું);
}

શૂન્ય ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, સાચું);
}

સ્થિર fs::path get_executable_path() {
#જો defined(__APPLE__)
    // get executable path
    std::vector<અક્ષર> path;
    uint32_t size;
    જ્યારે (સાચું) {
        size = path.size();
        જો (_NSGetExecutablePath(path.data(), &size) == 0) {
            તોડો;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    જો (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    પાછા base_path + "/";
#અથવાજો defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<અક્ષર> path(1024);
    જ્યારે (સાચું) {
        // get executable path
#    જો defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        જો (len == -1) {
            તોડો;
        }
        જો (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            જો (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            તોડો;
        }
        path.resize(path.size() * 2);
    }

    પાછા base_path + "/";
#અથવાજો defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(ખાલી, path.data(), path.size());
    જો (len == 0) {
        પાછા {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    જો (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    પાછા base_path + L"\\";
#નહીંતર
    પાછા {};
#endif
}

સ્થિર fs::path backend_filename_prefix() {
#ifdef _WIN32
    પાછા fs::u8path("ggml-");
#નહીંતર
    પાછા fs::u8path("libggml-");
#endif
}

સ્થિર fs::path backend_filename_extension() {
#ifdef _WIN32
    પાછા fs::u8path(".dll");
#નહીંતર
    પાછા fs::u8path(".so");
#endif
}

સ્થિર ggml_backend_reg_t ggml_backend_load_best(અચલ અક્ષર * name, તાર્કિક silent, અચલ અક્ષર * user_શોધ_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    અચલ fs::path name_path = fs::u8path(name);
    અચલ fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    અચલ fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> શોધ_paths;
    જો (user_શોધ_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        શોધ_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        શોધ_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        અચલ fs::path cwd = fs::current_path(cwd_ec);
        જો (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } નહીંતર {
            શોધ_paths.push_back(cwd);
        }
    } નહીંતર {
        શોધ_paths.push_back(fs::u8path(user_શોધ_path));
    }

    પૂર્ણાંક best_score = 0;
    fs::path best_path;
    std::error_code ec;

    માટે (અચલ auto & શોધ_path : શોધ_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(શોધ_path, dir_ec);

        માટે (અચલ fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            અચલ auto & entry = *dir_it;
            જો (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                જો (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    જો (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    જો (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        જો (score_fn) {
                            પૂર્ણાંક s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            જો (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } નહીંતર {
                            જો (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        જો (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(શોધ_path).c_str(), dir_ec.message().c_str());
        }
    }

    જો (best_score == 0) {
        // try to load the base backend
        માટે (અચલ auto & શોધ_path : શોધ_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = શોધ_path / filename;
            જો (std::error_code ec; fs::exists(path, ec)) {
                પાછા get_reg().load_backend(path, silent);
            } નહીંતર {
                જો (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        પાછા nullptr;
    }

    પાછા get_reg().load_backend(best_path, silent);
}

શૂન્ય ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

શૂન્ય ggml_backend_load_all_from_path(અચલ અક્ષર * dir_path) {
#ifdef NDEBUG
    તાર્કિક silent = સાચું;
#નહીંતર
    તાર્કિક silent = ખોટું;
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
    અચલ અક્ષર * backend_path = std::getenv("GGML_BACKEND_PATH");
    જો (backend_path) {
        ggml_backend_load(backend_path);
    }
}
