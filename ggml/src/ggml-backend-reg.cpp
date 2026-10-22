#समावेश "ggml-backend-impl.h"
#समावेश "ggml-backend.h"
#समावेश "ggml-backend-dl.h"
#समावेश "ggml-impl.h"
#समावेश <algorithm>
#समावेश <cstring>
#समावेश <filesystem>
#समावेश <memory>
#समावेश <string>
#समावेश <type_traits>
#समावेश <vector>
#समावेश <cctype>

#अगर_घोषित _WIN32
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#या_अगर defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#पूर्ण_अगर

// Backend registry
#अगर_घोषित GGML_USE_CPU
#समावेश "ggml-cpu.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CUDA
#समावेश "ggml-cuda.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_METAL
#समावेश "ggml-metal.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_SYCL
#समावेश "ggml-sycl.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_VULKAN
#समावेश "ggml-vulkan.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_WEBGPU
#समावेश "ggml-webgpu.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_ZDNN
#समावेश "ggml-zdnn.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_OPENCL
#समावेश "ggml-opencl.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_HEXAGON
#समावेश "ggml-hexagon.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_BLAS
#समावेश "ggml-blas.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_RPC
#समावेश "ggml-rpc.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_VIRTGPU_FRONTEND
#समावेश "ggml-virtgpu.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_CANN
#समावेश "ggml-cann.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_ZENDNN
#समावेश "ggml-zendnn.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_OPENVINO
#समावेश "ggml-openvino.h"
#पूर्ण_अगर

#अगर_घोषित GGML_USE_ET
#समावेश "ggml-et.h"
#पूर्ण_अगर

namespace fs = std::filesystem;

अटल std::string path_str(स्थिर fs::path & path) {
    प्रयास {
#अगर defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        स्थिर std::u8string u8str = path.u8string();
        वापस std::string(reinterpret_cast<स्थिर अक्षर *>(u8str.तथ्य()), u8str.size());
#अन्यथा
        // C++17: u8string() returns std::string
        वापस path.u8string();
#पूर्ण_अगर
    } पकड़ो (...) {
        वापस std::string();
    }
}

काष्ठा ggml_backend_reg_entry {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

काष्ठा ggml_backend_registry {
    std::vector<ggml_backend_reg_entry> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#अगर_घोषित GGML_USE_CUDA
        register_backend(ggml_backend_cuda_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_METAL
        register_backend(ggml_backend_metal_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_SYCL
        register_backend(ggml_backend_sycl_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_VULKAN
    // Add runtime disable check
    अगर (दो_पर्या("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } अन्यथा {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#पूर्ण_अगर
#अगर_घोषित GGML_USE_WEBGPU
        register_backend(ggml_backend_webgpu_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_ZDNN
        register_backend(ggml_backend_zdnn_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_VIRTGPU_FRONTEND
        register_backend(ggml_backend_virtgpu_reg());
#पूर्ण_अगर

#अगर_घोषित GGML_USE_OPENCL
        register_backend(ggml_backend_opencl_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_ZENDNN
        register_backend(ggml_backend_zendnn_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_HEXAGON
        register_backend(ggml_backend_hexagon_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_CANN
        register_backend(ggml_backend_cann_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_BLAS
        register_backend(ggml_backend_blas_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_RPC
        register_backend(ggml_backend_rpc_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_OPENVINO
        register_backend(ggml_backend_openvino_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_ET
        register_backend(ggml_backend_et_reg());
#पूर्ण_अगर
#अगर_घोषित GGML_USE_CPU
        register_backend(ggml_backend_cpu_reg());
#पूर्ण_अगर
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        क्रम (स्वतः & entry : backends) {
            अगर (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    व्योम register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        अगर (!reg) {
            वापस;
        }

        क्रम (स्वतः & entry : backends) {
            अगर (entry.reg == reg) {
                वापस;
            }
        }

#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_गिनती(reg));
#पूर्ण_अगर
        backends.push_back({ reg, std::move(handle) });
        क्रम (माप_प्रकार i = 0; i < ggml_backend_reg_dev_गिनती(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    व्योम register_device(ggml_backend_dev_t device) {
        क्रम (स्वतः & dev : devices) {
            अगर (dev == device) {
                वापस;
            }
        }

#अगर_अघोषित न_संशोधन
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#पूर्ण_अगर
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(स्थिर fs::path & path, bool silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        अगर (!handle) {
            अगर (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            वापस nullptr;
        }

        स्वतः score_fn = (ggml_backend_score_t) dl_get_sym(handle.लो(), "ggml_backend_score");
        अगर (score_fn && score_fn() == 0) {
            अगर (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            वापस nullptr;
        }

        स्वतः backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.लो(), "ggml_backend_init");
        अगर (!backend_init_fn) {
            अगर (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            वापस nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        अगर (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            अगर (!silent) {
                अगर (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } अन्यथा {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            वापस nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        वापस reg;
    }

    व्योम unload_backend(ggml_backend_reg_t reg, bool silent) {
        स्वतः it = std::find_if(backends.begin(), backends.end(),
                               [reg](स्थिर ggml_backend_reg_entry & entry) { वापस entry.reg == reg; });

        अगर (it == backends.end()) {
            अगर (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            वापस;
        }

        अगर (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { वापस ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

अटल ggml_backend_registry & get_reg() {
    अटल ggml_backend_registry reg;
    वापस reg;
}

// Internal API
व्योम ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

व्योम ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
अटल bool striequals(स्थिर अक्षर * a, स्थिर अक्षर * b) {
    क्रम (; *a && *b; a++, b++) {
        अगर (std::छोटे(*a) != std::छोटे(*b)) {
            वापस false;
        }
    }
    वापस *a == *b;
}

माप_प्रकार ggml_backend_reg_गिनती() {
    वापस get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक < ggml_backend_reg_गिनती());
    वापस get_reg().backends[सूचकांक].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(स्थिर अक्षर * name) {
    क्रम (माप_प्रकार i = 0; i < ggml_backend_reg_गिनती(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        अगर (striequals(ggml_backend_reg_name(reg), name)) {
            वापस reg;
        }
    }
    वापस nullptr;
}

// Device enumeration
माप_प्रकार ggml_backend_dev_गिनती() {
    वापस get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(माप_प्रकार सूचकांक) {
    GGML_ASSERT(सूचकांक < ggml_backend_dev_गिनती());
    वापस get_reg().devices[सूचकांक];
}

ggml_backend_dev_t ggml_backend_dev_by_name(स्थिर अक्षर * name) {
    क्रम (माप_प्रकार i = 0; i < ggml_backend_dev_गिनती(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        अगर (striequals(ggml_backend_dev_name(dev), name)) {
            वापस dev;
        }
    }
    वापस nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(क्रमागत ggml_backend_dev_type type) {
    क्रम (माप_प्रकार i = 0; i < ggml_backend_dev_गिनती(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        अगर (ggml_backend_dev_type(dev) == type) {
            वापस dev;
        }
    }
    वापस nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(स्थिर अक्षर * name, स्थिर अक्षर * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    अगर (!dev) {
        वापस nullptr;
    }
    वापस ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(क्रमागत ggml_backend_dev_type type, स्थिर अक्षर * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    अगर (!dev) {
        वापस nullptr;
    }
    वापस ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(व्योम) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    अगर (!dev) {
        वापस nullptr;
    }
    वापस ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(स्थिर अक्षर * path) {
    वापस get_reg().load_backend(path, false);
}

व्योम ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, true);
}

अटल fs::path get_executable_path() {
#अगर defined(__APPLE__)
    // get executable path
    std::vector<अक्षर> path;
    uint32_t size;
    जबतक (true) {
        size = path.size();
        अगर (_NSGetExecutablePath(path.तथ्य(), &size) == 0) {
            अवरोध;
        }
        path.resize(size);
    }
    std::string base_path(path.तथ्य(), size);
    // remove executable name
    स्वतः last_slash = base_path.find_last_of('/');
    अगर (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    वापस base_path + "/";
#या_अगर defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<अक्षर> path(1024);
    जबतक (true) {
        // get executable path
#    अगर defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.तथ्य(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.तथ्य(), path.size());
#    endif
        अगर (len == -1) {
            अवरोध;
        }
        अगर (len < (ssize_t) path.size()) {
            base_path = std::string(path.तथ्य(), len);
            // remove executable name
            स्वतः last_slash = base_path.find_last_of('/');
            अगर (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            अवरोध;
        }
        path.resize(path.size() * 2);
    }

    वापस base_path + "/";
#या_अगर defined(_WIN32)
    std::vector<ब_अक्षर_प्रकार> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(शून्य, path.तथ्य(), path.size());
    अगर (len == 0) {
        वापस {};
    }
    std::wstring base_path(path.तथ्य(), len);
    // remove executable name
    स्वतः last_slash = base_path.find_last_of('\\');
    अगर (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    वापस base_path + L"\\";
#अन्यथा
    वापस {};
#पूर्ण_अगर
}

अटल fs::path backend_filename_prefix() {
#अगर_घोषित _WIN32
    वापस fs::u8path("ggml-");
#अन्यथा
    वापस fs::u8path("libggml-");
#पूर्ण_अगर
}

अटल fs::path backend_filename_extension() {
#अगर_घोषित _WIN32
    वापस fs::u8path(".dll");
#अन्यथा
    वापस fs::u8path(".so");
#पूर्ण_अगर
}

अटल ggml_backend_reg_t ggml_backend_load_best(स्थिर अक्षर * name, bool silent, स्थिर अक्षर * user_खोज_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    स्थिर fs::path name_path = fs::u8path(name);
    स्थिर fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    स्थिर fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> खोज_paths;
    अगर (user_खोज_path == nullptr) {
#अगर_घोषित GGML_BACKEND_DIR
        खोज_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#पूर्ण_अगर
        // default search paths: executable directory, current directory
        खोज_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        स्थिर fs::path cwd = fs::current_path(cwd_ec);
        अगर (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } अन्यथा {
            खोज_paths.push_back(cwd);
        }
    } अन्यथा {
        खोज_paths.push_back(fs::u8path(user_खोज_path));
    }

    पूर्णांक best_score = 0;
    fs::path best_path;
    std::error_code ec;

    क्रम (स्थिर स्वतः & खोज_path : खोज_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(खोज_path, dir_ec);

        क्रम (स्थिर fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            स्थिर स्वतः & entry = *dir_it;
            अगर (entry.is_regular_file(ec)) {
                स्वतः filename = entry.path().filename();
                स्वतः ext = entry.path().extension();
                अगर (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    अगर (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    अगर (handle) {
                        स्वतः score_fn = (ggml_backend_score_t) dl_get_sym(handle.लो(), "ggml_backend_score");
                        अगर (score_fn) {
                            पूर्णांक s = score_fn();
#अगर_अघोषित न_संशोधन
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#पूर्ण_अगर
                            अगर (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } अन्यथा {
                            अगर (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        अगर (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(खोज_path).c_str(), dir_ec.message().c_str());
        }
    }

    अगर (best_score == 0) {
        // try to load the base backend
        क्रम (स्थिर स्वतः & खोज_path : खोज_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = खोज_path / filename;
            अगर (std::error_code ec; fs::exists(path, ec)) {
                वापस get_reg().load_backend(path, silent);
            } अन्यथा {
                अगर (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        वापस nullptr;
    }

    वापस get_reg().load_backend(best_path, silent);
}

व्योम ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

व्योम ggml_backend_load_all_from_path(स्थिर अक्षर * dir_path) {
#अगर_घोषित न_संशोधन
    bool silent = true;
#अन्यथा
    bool silent = false;
#पूर्ण_अगर

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
    स्थिर अक्षर * backend_path = std::दो_पर्या("GGML_BACKEND_PATH");
    अगर (backend_path) {
        ggml_backend_load(backend_path);
    }
}
