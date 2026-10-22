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
#किंवाजर defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#अन्यथा
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

स्थिर std::string path_str(स्थिरांक fs::path & path) {
    प्रयत्न {
#जर defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        स्थिरांक std::u8string u8str = path.u8string();
        परत std::string(reinterpret_cast<स्थिरांक अक्षर *>(u8str.data()), u8str.size());
#अन्यथा
        // C++17: u8string() returns std::string
        परत path.u8string();
#endif
    } पकडा (...) {
        परत std::string();
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
    जर (दो_पर्या("GGML_DISABLE_VULKAN") == nullptr) {
        register_backend(ggml_backend_vk_reg());
    } अन्यथा {
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
        साठी (स्वतः & entry : backends) {
            जर (entry.handle) {
                entry.handle.release(); // NOLINT
            }
        }
    }

    शून्य register_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
        जर (!reg) {
            परत;
        }

        साठी (स्वतः & entry : backends) {
            जर (entry.reg == reg) {
                परत;
            }
        }

#ifndef न_संशोधन
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_मोजणी(reg));
#endif
        backends.push_back({ reg, std::move(handle) });
        साठी (माप_प्रकार i = 0; i < ggml_backend_reg_dev_मोजणी(reg); i++) {
            register_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    शून्य register_device(ggml_backend_dev_t device) {
        साठी (स्वतः & dev : devices) {
            जर (dev == device) {
                परत;
            }
        }

#ifndef न_संशोधन
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#endif
        devices.push_back(device);
    }

    ggml_backend_reg_t load_backend(स्थिरांक fs::path & path, तार्किक silent) {
        dl_handle_ptr handle { dl_load_library(path) };
        जर (!handle) {
            जर (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(path).c_str(), dl_error());
            }
            परत nullptr;
        }

        स्वतः score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        जर (score_fn && score_fn() == 0) {
            जर (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, path_str(path).c_str());
            }
            परत nullptr;
        }

        स्वतः backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        जर (!backend_init_fn) {
            जर (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, path_str(path).c_str());
            }
            परत nullptr;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        जर (!reg || reg->api_version != GGML_BACKEND_API_VERSION) {
            जर (!silent) {
                जर (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, path_str(path).c_str());
                } अन्यथा {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, path_str(path).c_str(), reg->api_version, GGML_BACKEND_API_VERSION);
                }
            }
            परत nullptr;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), path_str(path).c_str());

        register_backend(reg, std::move(handle));

        परत reg;
    }

    शून्य unload_backend(ggml_backend_reg_t reg, तार्किक silent) {
        स्वतः it = std::find_if(backends.begin(), backends.end(),
                               [reg](स्थिरांक ggml_backend_reg_entry & entry) { परत entry.reg == reg; });

        जर (it == backends.end()) {
            जर (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            परत;
        }

        जर (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { परत ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

स्थिर ggml_backend_registry & get_reg() {
    स्थिर ggml_backend_registry reg;
    परत reg;
}

// Internal API
शून्य ggml_backend_register(ggml_backend_reg_t reg) {
    get_reg().register_backend(reg);
}

शून्य ggml_backend_device_register(ggml_backend_dev_t device) {
    get_reg().register_device(device);
}

// Backend (reg) enumeration
स्थिर तार्किक striequals(स्थिरांक अक्षर * a, स्थिरांक अक्षर * b) {
    साठी (; *a && *b; a++, b++) {
        जर (std::छोटे(*a) != std::छोटे(*b)) {
            परत असत्य;
        }
    }
    परत *a == *b;
}

माप_प्रकार ggml_backend_reg_मोजणी() {
    परत get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(माप_प्रकार index) {
    GGML_ASSERT(index < ggml_backend_reg_मोजणी());
    परत get_reg().backends[index].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(स्थिरांक अक्षर * name) {
    साठी (माप_प्रकार i = 0; i < ggml_backend_reg_मोजणी(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        जर (striequals(ggml_backend_reg_name(reg), name)) {
            परत reg;
        }
    }
    परत nullptr;
}

// Device enumeration
माप_प्रकार ggml_backend_dev_मोजणी() {
    परत get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(माप_प्रकार index) {
    GGML_ASSERT(index < ggml_backend_dev_मोजणी());
    परत get_reg().devices[index];
}

ggml_backend_dev_t ggml_backend_dev_by_name(स्थिरांक अक्षर * name) {
    साठी (माप_प्रकार i = 0; i < ggml_backend_dev_मोजणी(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        जर (striequals(ggml_backend_dev_name(dev), name)) {
            परत dev;
        }
    }
    परत nullptr;
}

ggml_backend_dev_t ggml_backend_dev_by_type(गणना ggml_backend_dev_type type) {
    साठी (माप_प्रकार i = 0; i < ggml_backend_dev_मोजणी(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        जर (ggml_backend_dev_type(dev) == type) {
            परत dev;
        }
    }
    परत nullptr;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(स्थिरांक अक्षर * name, स्थिरांक अक्षर * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    जर (!dev) {
        परत nullptr;
    }
    परत ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(गणना ggml_backend_dev_type type, स्थिरांक अक्षर * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    जर (!dev) {
        परत nullptr;
    }
    परत ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(शून्य) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    जर (!dev) {
        परत nullptr;
    }
    परत ggml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_load(स्थिरांक अक्षर * path) {
    परत get_reg().load_backend(path, असत्य);
}

शून्य ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, सत्य);
}

स्थिर fs::path get_executable_path() {
#जर defined(__APPLE__)
    // get executable path
    std::vector<अक्षर> path;
    uint32_t size;
    पर्यंत (सत्य) {
        size = path.size();
        जर (_NSGetExecutablePath(path.data(), &size) == 0) {
            थांबा;
        }
        path.resize(size);
    }
    std::string base_path(path.data(), size);
    // remove executable name
    स्वतः last_slash = base_path.find_last_of('/');
    जर (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    परत base_path + "/";
#किंवाजर defined(__linux__) || defined(__FreeBSD__)
    std::string base_path = ".";
    std::vector<अक्षर> path(1024);
    पर्यंत (सत्य) {
        // get executable path
#    जर defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
#    endif
        जर (len == -1) {
            थांबा;
        }
        जर (len < (ssize_t) path.size()) {
            base_path = std::string(path.data(), len);
            // remove executable name
            स्वतः last_slash = base_path.find_last_of('/');
            जर (last_slash != std::string::npos) {
                base_path = base_path.substr(0, last_slash);
            }
            थांबा;
        }
        path.resize(path.size() * 2);
    }

    परत base_path + "/";
#किंवाजर defined(_WIN32)
    std::vector<ब_अक्षर_प्रकार> path(MAX_PATH);
    DWORD len = GetModuleFileNameW(रिक्त, path.data(), path.size());
    जर (len == 0) {
        परत {};
    }
    std::wstring base_path(path.data(), len);
    // remove executable name
    स्वतः last_slash = base_path.find_last_of('\\');
    जर (last_slash != std::string::npos) {
        base_path = base_path.substr(0, last_slash);
    }
    परत base_path + L"\\";
#अन्यथा
    परत {};
#endif
}

स्थिर fs::path backend_filename_prefix() {
#ifdef _WIN32
    परत fs::u8path("ggml-");
#अन्यथा
    परत fs::u8path("libggml-");
#endif
}

स्थिर fs::path backend_filename_extension() {
#ifdef _WIN32
    परत fs::u8path(".dll");
#अन्यथा
    परत fs::u8path(".so");
#endif
}

स्थिर ggml_backend_reg_t ggml_backend_load_best(स्थिरांक अक्षर * name, तार्किक silent, स्थिरांक अक्षर * user_शोध_path) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    स्थिरांक fs::path name_path = fs::u8path(name);
    स्थिरांक fs::path file_prefix = backend_filename_prefix().native() + name_path.native() + fs::u8path("-").native();
    स्थिरांक fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> शोध_paths;
    जर (user_शोध_path == nullptr) {
#ifdef GGML_BACKEND_DIR
        शोध_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#endif
        // default search paths: executable directory, current directory
        शोध_paths.push_back(get_executable_path());
        std::error_code cwd_ec;
        स्थिरांक fs::path cwd = fs::current_path(cwd_ec);
        जर (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.message().c_str());
        } अन्यथा {
            शोध_paths.push_back(cwd);
        }
    } अन्यथा {
        शोध_paths.push_back(fs::u8path(user_शोध_path));
    }

    पूर्णांक best_score = 0;
    fs::path best_path;
    std::error_code ec;

    साठी (स्थिरांक स्वतः & शोध_path : शोध_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(शोध_path, dir_ec);

        साठी (स्थिरांक fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            स्थिरांक स्वतः & entry = *dir_it;
            जर (entry.is_regular_file(ec)) {
                स्वतः filename = entry.path().filename();
                स्वतः ext = entry.path().extension();
                जर (filename.native().find(file_prefix) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_load_library(entry) };
                    जर (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, path_str(entry.path()).c_str(), dl_error());
                    }
                    जर (handle) {
                        स्वतः score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        जर (score_fn) {
                            पूर्णांक s = score_fn();
#ifndef न_संशोधन
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, path_str(entry.path()).c_str(), s);
#endif
                            जर (s > best_score) {
                                best_score = s;
                                best_path = entry.path();
                            }
                        } अन्यथा {
                            जर (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, path_str(entry.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        जर (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, path_str(शोध_path).c_str(), dir_ec.message().c_str());
        }
    }

    जर (best_score == 0) {
        // try to load the base backend
        साठी (स्थिरांक स्वतः & शोध_path : शोध_paths) {
            fs::path filename = backend_filename_prefix().native() + name_path.native() + backend_filename_extension().native();
            fs::path path = शोध_path / filename;
            जर (std::error_code ec; fs::exists(path, ec)) {
                परत get_reg().load_backend(path, silent);
            } अन्यथा {
                जर (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, path_str(path).c_str(), ec.message().c_str());
                }
            }
        }
        परत nullptr;
    }

    परत get_reg().load_backend(best_path, silent);
}

शून्य ggml_backend_load_all() {
    ggml_backend_load_all_from_path(nullptr);
}

शून्य ggml_backend_load_all_from_path(स्थिरांक अक्षर * dir_path) {
#ifdef न_संशोधन
    तार्किक silent = सत्य;
#अन्यथा
    तार्किक silent = असत्य;
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
    स्थिरांक अक्षर * backend_path = std::दो_पर्या("GGML_BACKEND_PATH");
    जर (backend_path) {
        ggml_backend_load(backend_path);
    }
}
