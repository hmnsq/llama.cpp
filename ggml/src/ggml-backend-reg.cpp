#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-backend-dl.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstring>
#अन्तर्भाव <filesystem>
#अन्तर्भाव <memory>
#अन्तर्भाव <string>
#अन्तर्भाव <type_traits>
#अन्तर्भाव <vector>
#अन्तर्भाव <cctype>

#ifdef _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#नोचेद्यदि defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#अञ्ञथा
#    include <dlfcn.h>
#    include <unistd.h>
#endif

// Backend registry
#ifdef GGML_USE_CPU
#अन्तर्भाव "ggml-cpu.h"
#endif

#ifdef GGML_USE_CUDA
#अन्तर्भाव "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#अन्तर्भाव "ggml-metal.h"
#endif

#ifdef GGML_USE_SYCL
#अन्तर्भाव "ggml-sycl.h"
#endif

#ifdef GGML_USE_VULKAN
#अन्तर्भाव "ggml-vulkan.h"
#endif

#ifdef GGML_USE_WEBGPU
#अन्तर्भाव "ggml-webgpu.h"
#endif

#ifdef GGML_USE_ZDNN
#अन्तर्भाव "ggml-zdnn.h"
#endif

#ifdef GGML_USE_OPENCL
#अन्तर्भाव "ggml-opencl.h"
#endif

#ifdef GGML_USE_HEXAGON
#अन्तर्भाव "ggml-hexagon.h"
#endif

#ifdef GGML_USE_BLAS
#अन्तर्भाव "ggml-blas.h"
#endif

#ifdef GGML_USE_RPC
#अन्तर्भाव "ggml-rpc.h"
#endif

#ifdef GGML_USE_VIRTGPU_FRONTEND
#अन्तर्भाव "ggml-virtgpu.h"
#endif

#ifdef GGML_USE_CANN
#अन्तर्भाव "ggml-cann.h"
#endif

#ifdef GGML_USE_ZENDNN
#अन्तर्भाव "ggml-zendnn.h"
#endif

#ifdef GGML_USE_OPENVINO
#अन्तर्भाव "ggml-openvino.h"
#endif

#ifdef GGML_USE_ET
#अन्तर्भाव "ggml-et.h"
#endif

नामाकाश fs = std::filesystem;

स्थिर std::string path_str(ध्रुव fs::path & path) {
    यत्न {
#सचे defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ध्रुव std::u8string u8str = path.u8string();
        पटिनिवत्त std::string(reinterpret_cast<ध्रुव वर्ण *>(u8str.data()), u8str.size());
#अञ्ञथा
        // C++17: u8string() returns std::string
        पटिनिवत्त path.u8string();
#endif
    } गृहाण (...) {
        पटिनिवत्त std::string();
    }
}

संरचना ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

संरचना ggml_backend_registry {
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
    सचे (getenv("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } अञ्ञथा {
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
        अत्थाय (auto & entry : backends) {
            सचे (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    शून्य register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        सचे (!reg) {
            पटिनिवत्त;
        }

        अत्थाय (auto & entry : backends) {
            सचे (entry.reg == reg) {
                पटिनिवत्त;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_गणनम्(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        अत्थाय (size_t i = 0; i < ggml_backend_reg_dev_गणनम्(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    शून्य register_device(ggml_backend_dev_t device) {
        अत्थाय (auto & dev : devices) {
            सचे (dev == device) {
                पटिनिवत्त;
            }
        }

#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(ध्रुव fs::path & path, तार्किक silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        सचे (!handle) {
            सचे (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            पटिनिवत्त nullptr;
        }

        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        सचे (score_fn && score_fn() == 0) {
            सचे (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            पटिनिवत्त nullptr;
        }

        auto backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        सचे (!backend_init_fn) {
            सचे (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            पटिनिवत्त nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        सचे (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            सचे (!silent) {
                सचे (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } अञ्ञथा {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            पटिनिवत्त nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        पटिनिवत्त reg;
    }

    शून्य unload_backend(ggml_backend_reg_t reg, तार्किक silent) {
        auto it = std::find_if(backends.begin(), backends.end(),
                               [reg](ध्रुव ggml_backend_reg_entry & entry) { पटिनिवत्त entry.reg == reg; });

        सचे (it == backends.end()) {
            सचे (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            पटिनिवत्त;
        }

        सचे (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { पटिनिवत्त ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

स्थिर ggml_backend_registry & get_reg() {
    स्थिर ggml_backend_registry reg;
    पटिनिवत्त reg;
}

// Internal API
शून्य ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

शून्य ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
स्थिर तार्किक striequals(ध्रुव वर्ण * a, ध्रुव वर्ण * b) {
    अत्थाय (; *a && *b; a++, b++) {
        सचे (std::tolower(*a) != std::tolower(*b)) {
            पटिनिवत्त मिच्छा;
        }
    }
    पटिनिवत्त *a == *b;
}

size_t ggml_backend_reg_गणनम्() {
    पटिनिवत्त get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_reg_गणनम्());
    पटिनिवत्त get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ध्रुव वर्ण * name) {
    अत्थाय (size_t i = 0; i < ggml_backend_reg_गणनम्(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        सचे (striequals(ggml_backend_reg_name(reg), name)) {
            पटिनिवत्त reg;
        }
    }
    पटिनिवत्त nullptr;
}

// Device enumeration
size_t ggml_backend_dev_गणनम्() {
    पटिनिवत्त get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t index) {
    GGML_ASSERT(index < ggml_backend_dev_गणनम्());
    पटिनिवत्त get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ध्रुव वर्ण * name) {
    अत्थाय (size_t i = 0; i < ggml_backend_dev_गणनम्(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        सचे (striequals(ggml_backend_dev_name(dev), name)) {
            पटिनिवत्त dev;
        }
    }
    पटिनिवत्त nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(गणना ggml_backend_dev_type type) {
    अत्थाय (size_t i = 0; i < ggml_backend_dev_गणनम्(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        सचे (ggml_backend_dev_type(dev) == type) {
            पटिनिवत्त dev;
        }
    }
    पटिनिवत्त nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ध्रुव वर्ण * name, ध्रुव वर्ण * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    सचे (!dev) {
        पटिनिवत्त nullptr;
    }
    पटिनिवत्त ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(गणना ggml_backend_dev_type type, ध्रुव वर्ण * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    सचे (!dev) {
        पटिनिवत्त nullptr;
    }
    पटिनिवत्त ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(शून्य) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    सचे (!dev) {
        पटिनिवत्त nullptr;
    }
    पटिनिवत्त ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(ध्रुव वर्ण * path) {
    पटिनिवत्त get_reg().load_backend(path, मिच्छा);
}

शून्य ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, सच्च);
}

स्थिर fs::path get_executable_path() {
#सचे defined(__APPLE__)
    // get executable path
    std::vector<वर्ण> path;
    uint32_t size;
    याव (सच्च) {
        size = path.size();
        सचे (_NSGetExecutablePath(path.data(), &size) == 0) {
            भङ्ग;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    auto last_slash = base_path.find_last_of('/');
    सचे (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    पटिनिवत्त base_path + "/";
#नोचेद्यदि defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<वर्ण> path(1024);
    याव (सच्च) {
        // get executable path
#    सचे defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        सचे (len == -1) {
            भङ्ग;
        }
        सचे (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            auto last_slash = base_path.find_last_of('/');
            सचे (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            भङ्ग;
        }
        path.resize(path.size() * 2);
    }

    पटिनिवत्त base_path + "/";
#नोचेद्यदि defined(_WIN32)
    std::vector<wchar_t> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(रिक्त, path.data(), path.size());
    सचे (len == 0) {
        पटिनिवत्त {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    auto last_slash = base_path.find_last_of('\\');
    सचे (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    पटिनिवत्त base_path + L"\\";
#अञ्ञथा
    पटिनिवत्त {};
#endif
}

स्थिर fs::path backend_filename_prefix() {
#ifdef _WIN32
    पटिनिवत्त fs::u8path("ggml-");
#अञ्ञथा
    पटिनिवत्त fs::u8path("libggml-");
#endif
}

स्थिर fs::path backend_filename_extension() {
#ifdef _WIN32
    पटिनिवत्त fs::u8path(".dll");
#अञ्ञथा
    पटिनिवत्त fs::u8path(".so");
#endif
}

स्थिर ggml_backend_reg_t ggml_backend_load_best(ध्रुव वर्ण * name, तार्किक silent, ध्रुव वर्ण * user_अन्वेषणम्_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ध्रुव fs::path name_path = fs::u8path(name);
    ध्रुव fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    ध्रुव fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> अन्वेषणम्_paths;
    सचे (user_अन्वेषणम्_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        अन्वेषणम्_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        अन्वेषणम्_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        ध्रुव fs::path cwd = fs::current_path(cwd_ec);
        सचे (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } अञ्ञथा {
            अन्वेषणम्_paths.push_back(cwd);
        }
    } अञ्ञथा {
        अन्वेषणम्_paths.push_back(fs::u8path(user_अन्वेषणम्_path));
    }

    पूर्णाङ्क best_score = 0;
    fs::path best_path;
    std::error_code ec;

    अत्थाय (ध्रुव auto & अन्वेषणम्_path : अन्वेषणम्_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(अन्वेषणम्_path, dir_ec);

        अत्थाय (ध्रुव fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ध्रुव auto & entry = *dir_it;
            सचे (entry.is_regular_file(ec)) {
                auto filename = entry.path().filename();
                auto ext = entry.path().extension();
                सचे (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    सचे (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    सचे (handle) {
                        auto score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        सचे (score_fn) {
                            पूर्णाङ्क s = score_fn();
#ifndef NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            सचे (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } अञ्ञथा {
                            सचे (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        सचे (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(अन्वेषणम्_path).c_str(), dir_ec.message().c_str());
        }
    }

    सचे (best_score == 0) {
        // try to load the base backend
        अत्थाय (ध्रुव auto & अन्वेषणम्_path : अन्वेषणम्_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = अन्वेषणम्_path / filename;
            सचे (std::error_code ec; fs::exists(path, ec)) {
                पटिनिवत्त get_reg().load_backend(path, silent);
            } अञ्ञथा {
                सचे (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        पटिनिवत्त nullptr;
    }

    पटिनिवत्त get_reg().load_backend(best_path, silent);
}

शून्य ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

शून्य ggml_backend_load_all_from_path(ध्रुव वर्ण * dir_path) {
#ifdef NDEBUG
    तार्किक silent = सच्च;
#अञ्ञथा
    तार्किक silent = मिच्छा;
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
    ध्रुव वर्ण * backend_path = std::getenv("GGML_BACKEND_PATH");
    सचे (backend_path) {
        ggml_backend_load(backend_path);
    }
}
