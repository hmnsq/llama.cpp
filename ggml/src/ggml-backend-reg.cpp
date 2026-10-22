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

#यदि_परिभाषितम् _WIN32
#    define WIN32_LEAN_AND_माध्यम्
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#नोचेद्यदि defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#अन्यथा
#    include <dlfcn.h>
#    include <unistd.h>
#यदि_अन्तः

// Backend registry
#यदि_परिभाषितम् GGML_USE_CPU
#अन्तर्भाव "ggml-cpu.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CUDA
#अन्तर्भाव "ggml-cuda.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_METAL
#अन्तर्भाव "ggml-metal.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_SYCL
#अन्तर्भाव "ggml-sycl.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_VULKAN
#अन्तर्भाव "ggml-vulkan.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_WEBGPU
#अन्तर्भाव "ggml-webgpu.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_ZDNN
#अन्तर्भाव "ggml-zdnn.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_OPENCL
#अन्तर्भाव "ggml-opencl.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_HEXAGON
#अन्तर्भाव "ggml-hexagon.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_BLAS
#अन्तर्भाव "ggml-blas.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_RPC
#अन्तर्भाव "ggml-rpc.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_VIRTGPU_FRONTEND
#अन्तर्भाव "ggml-virtgpu.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_CANN
#अन्तर्भाव "ggml-cann.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_ZENDNN
#अन्तर्भाव "ggml-zendnn.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_OPENVINO
#अन्तर्भाव "ggml-openvino.h"
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_ET
#अन्तर्भाव "ggml-et.h"
#यदि_अन्तः

नामाकाश fs = std::filesystem;

स्थिर std::string मार्गः_str(ध्रुव fs::path & path) {
    यत्न {
#यदि defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ध्रुव std::u8string u8str = path.u8string();
        प्रत्यागम std::string(reinterpret_cast<ध्रुव वर्ण *>(u8str.दत्त()), u8str.size());
#अन्यथा
        // C++17: u8string() returns std::string
        प्रत्यागम path.u8string();
#यदि_अन्तः
    } गृहाण (...) {
        प्रत्यागम std::string();
    }
}

संरचना ggml_backend_reg_प्रविष्टिः {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

संरचना ggml_backend_registry {
    std::vector<ggml_backend_reg_प्रविष्टिः> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#यदि_परिभाषितम् GGML_USE_CUDA
        पञ्जिकाकोशः_backend(ggml_backend_cuda_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_METAL
        पञ्जिकाकोशः_backend(ggml_backend_metal_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_SYCL
        पञ्जिकाकोशः_backend(ggml_backend_sycl_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_VULKAN
    // Add runtime disable check
    यदि (getenv("GGML_DISABLE_VULKAN") == शून्यसूचकः) {
        पञ्जिकाकोशः_backend(ggml_backend_vk_reg());
    } अन्यथा {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_WEBGPU
        पञ्जिकाकोशः_backend(ggml_backend_webgpu_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_ZDNN
        पञ्जिकाकोशः_backend(ggml_backend_zdnn_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_VIRTGPU_FRONTEND
        पञ्जिकाकोशः_backend(ggml_backend_virtgpu_reg());
#यदि_अन्तः

#यदि_परिभाषितम् GGML_USE_OPENCL
        पञ्जिकाकोशः_backend(ggml_backend_opencl_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_ZENDNN
        पञ्जिकाकोशः_backend(ggml_backend_zendnn_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_HEXAGON
        पञ्जिकाकोशः_backend(ggml_backend_hexagon_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_CANN
        पञ्जिकाकोशः_backend(ggml_backend_cann_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_BLAS
        पञ्जिकाकोशः_backend(ggml_backend_blas_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_RPC
        पञ्जिकाकोशः_backend(ggml_backend_rpc_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_OPENVINO
        पञ्जिकाकोशः_backend(ggml_backend_openvino_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_ET
        पञ्जिकाकोशः_backend(ggml_backend_et_reg());
#यदि_अन्तः
#यदि_परिभाषितम् GGML_USE_CPU
        पञ्जिकाकोशः_backend(ggml_backend_cpu_reg());
#यदि_अन्तः
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        प्रति (स्वचालित & प्रविष्टिः : backends) {
            यदि (प्रविष्टिः.handle) {
                प्रविष्टिः.handle.release(); // NOLINT
            }
        }
    }

    शून्य पञ्जिकाकोशः_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = शून्यसूचकः) {
        यदि (!reg) {
            प्रत्यागम;
        }

        प्रति (स्वचालित & प्रविष्टिः : backends) {
            यदि (प्रविष्टिः.reg == reg) {
                प्रत्यागम;
            }
        }

#यदि_अपरिभाषितम् NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_गणनम्(reg));
#यदि_अन्तः
        backends.push_back({ reg, std::move(handle) });
        प्रति (size_t i = 0; i < ggml_backend_reg_dev_गणनम्(reg); i++) {
            पञ्जिकाकोशः_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    शून्य पञ्जिकाकोशः_device(ggml_backend_dev_t device) {
        प्रति (स्वचालित & dev : devices) {
            यदि (dev == device) {
                प्रत्यागम;
            }
        }

#यदि_अपरिभाषितम् NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#यदि_अन्तः
        devices.push_back(device);
    }

    ggml_backend_reg_t आरोपय_backend(ध्रुव fs::path & path, तार्किक silent) {
        dl_handle_ptr handle { dl_आरोपय_library(path) };
        यदि (!handle) {
            यदि (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, मार्गः_str(path).c_str(), dl_error());
            }
            प्रत्यागम शून्यसूचकः;
        }

        स्वचालित score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        यदि (score_fn && score_fn() == 0) {
            यदि (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, मार्गः_str(path).c_str());
            }
            प्रत्यागम शून्यसूचकः;
        }

        स्वचालित backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        यदि (!backend_init_fn) {
            यदि (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, मार्गः_str(path).c_str());
            }
            प्रत्यागम शून्यसूचकः;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        यदि (!reg || reg->api_संस्करणम् != GGML_BACKEND_API_संस्करणम्) {
            यदि (!silent) {
                यदि (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, मार्गः_str(path).c_str());
                } अन्यथा {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, मार्गः_str(path).c_str(), reg->api_संस्करणम्, GGML_BACKEND_API_संस्करणम्);
                }
            }
            प्रत्यागम शून्यसूचकः;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), मार्गः_str(path).c_str());

        पञ्जिकाकोशः_backend(reg, std::move(handle));

        प्रत्यागम reg;
    }

    शून्य unload_backend(ggml_backend_reg_t reg, तार्किक silent) {
        स्वचालित it = std::find_if(backends.begin(), backends.end(),
                               [reg](ध्रुव ggml_backend_reg_प्रविष्टिः & प्रविष्टिः) { प्रत्यागम प्रविष्टिः.reg == reg; });

        यदि (it == backends.end()) {
            यदि (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            प्रत्यागम;
        }

        यदि (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { प्रत्यागम ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

स्थिर ggml_backend_registry & get_reg() {
    स्थिर ggml_backend_registry reg;
    प्रत्यागम reg;
}

// Internal API
शून्य ggml_backend_पञ्जिकाकोशः(ggml_backend_reg_t reg) {
    get_reg().पञ्जिकाकोशः_backend(reg);
}

शून्य ggml_backend_device_पञ्जिकाकोशः(ggml_backend_dev_t device) {
    get_reg().पञ्जिकाकोशः_device(device);
}

// Backend (reg) enumeration
स्थिर तार्किक striequals(ध्रुव वर्ण * a, ध्रुव वर्ण * b) {
    प्रति (; *a && *b; a++, b++) {
        यदि (std::tolower(*a) != std::tolower(*b)) {
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम *a == *b;
}

size_t ggml_backend_reg_गणनम्() {
    प्रत्यागम get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t सूचकांक) {
    GGML_ASSERT(सूचकांक < ggml_backend_reg_गणनम्());
    प्रत्यागम get_reg().backends[सूचकांक].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ध्रुव वर्ण * name) {
    प्रति (size_t i = 0; i < ggml_backend_reg_गणनम्(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        यदि (striequals(ggml_backend_reg_name(reg), name)) {
            प्रत्यागम reg;
        }
    }
    प्रत्यागम शून्यसूचकः;
}

// Device enumeration
size_t ggml_backend_dev_गणनम्() {
    प्रत्यागम get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t सूचकांक) {
    GGML_ASSERT(सूचकांक < ggml_backend_dev_गणनम्());
    प्रत्यागम get_reg().devices[सूचकांक];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ध्रुव वर्ण * name) {
    प्रति (size_t i = 0; i < ggml_backend_dev_गणनम्(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        यदि (striequals(ggml_backend_dev_name(dev), name)) {
            प्रत्यागम dev;
        }
    }
    प्रत्यागम शून्यसूचकः;
}

ggml_backend_dev_t ggml_backend_dev_by_type(गणना ggml_backend_dev_type type) {
    प्रति (size_t i = 0; i < ggml_backend_dev_गणनम्(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        यदि (ggml_backend_dev_type(dev) == type) {
            प्रत्यागम dev;
        }
    }
    प्रत्यागम शून्यसूचकः;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ध्रुव वर्ण * name, ध्रुव वर्ण * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    यदि (!dev) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(गणना ggml_backend_dev_type type, ध्रुव वर्ण * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    यदि (!dev) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_best(शून्य) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    यदि (!dev) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम ggml_backend_dev_init(dev, शून्यसूचकः);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_आरोपय(ध्रुव वर्ण * path) {
    प्रत्यागम get_reg().आरोपय_backend(path, असत्य);
}

शून्य ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, सत्य);
}

स्थिर fs::path get_executable_मार्गः() {
#यदि defined(__APPLE__)
    // get executable path
    std::vector<वर्ण> path;
    uint32_t size;
    यावत् (सत्य) {
        size = path.size();
        यदि (NSGet_Executable_मार्गः(path.दत्त(), &size) == 0) {
            भङ्ग;
        }
        path.resize(size);
    }
    std::string आधारः_मार्गः(path.दत्त(), size);
    // remove executable name
    स्वचालित चरमः_slash = आधारः_मार्गः.find_last_of('/');
    यदि (चरमः_slash != std::string::npos) {
        आधारः_मार्गः = आधारः_मार्गः.substr(0, चरमः_slash);
    }
    प्रत्यागम आधारः_मार्गः + "/";
#नोचेद्यदि defined(__linux__) || defined(__FreeBSD__)
    std::string आधारः_मार्गः = ".";
    std::vector<वर्ण> path(1024);
    यावत् (सत्य) {
        // get executable path
#    यदि defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.दत्त(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.दत्त(), path.size());
#    endif
        यदि (len == -1) {
            भङ्ग;
        }
        यदि (len < (ssize_t) path.size()) {
            आधारः_मार्गः = std::string(path.दत्त(), len);
            // remove executable name
            स्वचालित चरमः_slash = आधारः_मार्गः.find_last_of('/');
            यदि (चरमः_slash != std::string::npos) {
                आधारः_मार्गः = आधारः_मार्गः.substr(0, चरमः_slash);
            }
            भङ्ग;
        }
        path.resize(path.size() * 2);
    }

    प्रत्यागम आधारः_मार्गः + "/";
#नोचेद्यदि defined(_WIN32)
    std::vector<wchar_t> path(MAX_मार्गः);
    DWORD len = GetModuleFileNameW(रिक्त, path.दत्त(), path.size());
    यदि (len == 0) {
        प्रत्यागम {};
    }
    std::wstring आधारः_मार्गः(path.दत्त(), len);
    // remove executable name
    स्वचालित चरमः_slash = आधारः_मार्गः.find_last_of('\\');
    यदि (चरमः_slash != std::string::npos) {
        आधारः_मार्गः = आधारः_मार्गः.substr(0, चरमः_slash);
    }
    प्रत्यागम आधारः_मार्गः + L"\\";
#अन्यथा
    प्रत्यागम {};
#यदि_अन्तः
}

स्थिर fs::path backend_filename_उपसर्गः() {
#यदि_परिभाषितम् _WIN32
    प्रत्यागम fs::u8path("ggml-");
#अन्यथा
    प्रत्यागम fs::u8path("libggml-");
#यदि_अन्तः
}

स्थिर fs::path backend_filename_extension() {
#यदि_परिभाषितम् _WIN32
    प्रत्यागम fs::u8path(".dll");
#अन्यथा
    प्रत्यागम fs::u8path(".so");
#यदि_अन्तः
}

स्थिर ggml_backend_reg_t ggml_backend_आरोपय_best(ध्रुव वर्ण * name, तार्किक silent, ध्रुव वर्ण * user_अन्वेषणम्_मार्गः) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ध्रुव fs::path name_मार्गः = fs::u8path(name);
    ध्रुव fs::path file_उपसर्गः = backend_filename_उपसर्गः().native() + name_मार्गः.native() + fs::u8path("-").native();
    ध्रुव fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> अन्वेषणम्_paths;
    यदि (user_अन्वेषणम्_मार्गः == शून्यसूचकः) {
#यदि_परिभाषितम् GGML_BACKEND_DIR
        अन्वेषणम्_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#यदि_अन्तः
        // default search paths: executable directory, current directory
        अन्वेषणम्_paths.push_back(get_executable_मार्गः());
        std::error_code cwd_ec;
        ध्रुव fs::path cwd = fs::current_path(cwd_ec);
        यदि (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.सन्देशः().c_str());
        } अन्यथा {
            अन्वेषणम्_paths.push_back(cwd);
        }
    } अन्यथा {
        अन्वेषणम्_paths.push_back(fs::u8path(user_अन्वेषणम्_मार्गः));
    }

    पूर्णाङ्क best_score = 0;
    fs::path best_मार्गः;
    std::error_code ec;

    प्रति (ध्रुव स्वचालित & अन्वेषणम्_मार्गः : अन्वेषणम्_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(अन्वेषणम्_मार्गः, dir_ec);

        प्रति (ध्रुव fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ध्रुव स्वचालित & प्रविष्टिः = *dir_it;
            यदि (प्रविष्टिः.is_regular_file(ec)) {
                स्वचालित filename = प्रविष्टिः.path().filename();
                स्वचालित ext = प्रविष्टिः.path().extension();
                यदि (filename.native().find(file_उपसर्गः) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_आरोपय_library(प्रविष्टिः) };
                    यदि (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, मार्गः_str(प्रविष्टिः.path()).c_str(), dl_error());
                    }
                    यदि (handle) {
                        स्वचालित score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        यदि (score_fn) {
                            पूर्णाङ्क s = score_fn();
#यदि_अपरिभाषितम् NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, मार्गः_str(प्रविष्टिः.path()).c_str(), s);
#यदि_अन्तः
                            यदि (s > best_score) {
                                best_score = s;
                                best_मार्गः = प्रविष्टिः.path();
                            }
                        } अन्यथा {
                            यदि (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, मार्गः_str(प्रविष्टिः.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        यदि (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, मार्गः_str(अन्वेषणम्_मार्गः).c_str(), dir_ec.सन्देशः().c_str());
        }
    }

    यदि (best_score == 0) {
        // try to load the base backend
        प्रति (ध्रुव स्वचालित & अन्वेषणम्_मार्गः : अन्वेषणम्_paths) {
            fs::path filename = backend_filename_उपसर्गः().native() + name_मार्गः.native() + backend_filename_extension().native();
            fs::path path = अन्वेषणम्_मार्गः / filename;
            यदि (std::error_code ec; fs::exists(path, ec)) {
                प्रत्यागम get_reg().आरोपय_backend(path, silent);
            } अन्यथा {
                यदि (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, मार्गः_str(path).c_str(), ec.सन्देशः().c_str());
                }
            }
        }
        प्रत्यागम शून्यसूचकः;
    }

    प्रत्यागम get_reg().आरोपय_backend(best_मार्गः, silent);
}

शून्य ggml_backend_आरोपय_all() {
    ggml_backend_आरोपय_all_from_मार्गः(शून्यसूचकः);
}

शून्य ggml_backend_आरोपय_all_from_मार्गः(ध्रुव वर्ण * dir_मार्गः) {
#यदि_परिभाषितम् NDEBUG
    तार्किक silent = सत्य;
#अन्यथा
    तार्किक silent = असत्य;
#यदि_अन्तः

    ggml_backend_आरोपय_best("blas", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("zendnn", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("cann", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("cuda", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("hip", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("metal", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("rpc", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("sycl", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("vulkan", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("virtgpu", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("opencl", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("hexagon", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("musa", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("openvino", silent, dir_मार्गः);
    ggml_backend_आरोपय_best("cpu", silent, dir_मार्गः);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ध्रुव वर्ण * backend_मार्गः = std::getenv("GGML_BACKEND_PATH");
    यदि (backend_मार्गः) {
        ggml_backend_आरोपय(backend_मार्गः);
    }
}
