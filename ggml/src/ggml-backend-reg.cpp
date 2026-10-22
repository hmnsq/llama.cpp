#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-backend-dl.h"
#ضمن "ggml-impl.h"
#ضمن <algorithm>
#ضمن <cstring>
#ضمن <filesystem>
#ضمن <memory>
#ضمن <string>
#ضمن <type_traits>
#ضمن <vector>
#ضمن <cctype>

#جيڪڏهن_وصف_ٿيل _WIN32
#    define WIN32_LEAN_AND_سراسري
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#وإلاإذا defined(__APPLE__)
#    include <mach-o/dyld.h>
#    include <dlfcn.h>
#وإلا
#    include <dlfcn.h>
#    include <unistd.h>
#جيڪڏهن_پورو

// Backend registry
#جيڪڏهن_وصف_ٿيل GGML_USE_CPU
#ضمن "ggml-cpu.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CUDA
#ضمن "ggml-cuda.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_METAL
#ضمن "ggml-metal.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_SYCL
#ضمن "ggml-sycl.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_VULKAN
#ضمن "ggml-vulkan.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_WEBGPU
#ضمن "ggml-webgpu.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_ZDNN
#ضمن "ggml-zdnn.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_OPENCL
#ضمن "ggml-opencl.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_HEXAGON
#ضمن "ggml-hexagon.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_BLAS
#ضمن "ggml-blas.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_RPC
#ضمن "ggml-rpc.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_VIRTGPU_FRONTEND
#ضمن "ggml-virtgpu.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_CANN
#ضمن "ggml-cann.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_ZENDNN
#ضمن "ggml-zendnn.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_OPENVINO
#ضمن "ggml-openvino.h"
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_ET
#ضمن "ggml-et.h"
#جيڪڏهن_پورو

نطاق fs = std::filesystem;

ساكن std::string رستو_str(ثابت fs::path & path) {
    حاول {
#إذا defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        أرجع std::string(reinterpret_cast<ثابت حرف *>(u8str.معلومات()), u8str.size());
#وإلا
        // C++17: u8string() returns std::string
        أرجع path.u8string();
#جيڪڏهن_پورو
    } امسك (...) {
        أرجع std::string();
    }
}

بنية ggml_backend_reg_داخلا {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

بنية ggml_backend_registry {
    std::vector<ggml_backend_reg_داخلا> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#جيڪڏهن_وصف_ٿيل GGML_USE_CUDA
        رجسٽر_backend(ggml_backend_cuda_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_METAL
        رجسٽر_backend(ggml_backend_metal_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_SYCL
        رجسٽر_backend(ggml_backend_sycl_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_VULKAN
    // Add runtime disable check
    إذا (getenv("GGML_DISABLE_VULKAN") == خالي_اشارو) {
        رجسٽر_backend(ggml_backend_vk_reg());
    } وإلا {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_WEBGPU
        رجسٽر_backend(ggml_backend_webgpu_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_ZDNN
        رجسٽر_backend(ggml_backend_zdnn_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_VIRTGPU_FRONTEND
        رجسٽر_backend(ggml_backend_virtgpu_reg());
#جيڪڏهن_پورو

#جيڪڏهن_وصف_ٿيل GGML_USE_OPENCL
        رجسٽر_backend(ggml_backend_opencl_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_ZENDNN
        رجسٽر_backend(ggml_backend_zendnn_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_HEXAGON
        رجسٽر_backend(ggml_backend_hexagon_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_CANN
        رجسٽر_backend(ggml_backend_cann_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_BLAS
        رجسٽر_backend(ggml_backend_blas_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_RPC
        رجسٽر_backend(ggml_backend_rpc_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_OPENVINO
        رجسٽر_backend(ggml_backend_openvino_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_ET
        رجسٽر_backend(ggml_backend_et_reg());
#جيڪڏهن_پورو
#جيڪڏهن_وصف_ٿيل GGML_USE_CPU
        رجسٽر_backend(ggml_backend_cpu_reg());
#جيڪڏهن_پورو
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        لكل (خودڪار & داخلا : backends) {
            إذا (داخلا.handle) {
                داخلا.handle.release(); // NOLINT
            }
        }
    }

    فراغ رجسٽر_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = خالي_اشارو) {
        إذا (!reg) {
            أرجع;
        }

        لكل (خودڪار & داخلا : backends) {
            إذا (داخلا.reg == reg) {
                أرجع;
            }
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_ڳڻپ(reg));
#جيڪڏهن_پورو
        backends.push_back({ reg, std::move(handle) });
        لكل (size_t i = 0; i < ggml_backend_reg_dev_ڳڻپ(reg); i++) {
            رجسٽر_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    فراغ رجسٽر_device(ggml_backend_dev_t device) {
        لكل (خودڪار & dev : devices) {
            إذا (dev == device) {
                أرجع;
            }
        }

#جيڪڏهن_وصف_نه_ٿيل NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#جيڪڏهن_پورو
        devices.push_back(device);
    }

    ggml_backend_reg_t ڀريو_backend(ثابت fs::path & path, منطقي silent) {
        dl_handle_ptr handle { dl_ڀريو_library(path) };
        إذا (!handle) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, رستو_str(path).c_str(), dl_error());
            }
            أرجع خالي_اشارو;
        }

        خودڪار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        إذا (score_fn && score_fn() == 0) {
            إذا (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, رستو_str(path).c_str());
            }
            أرجع خالي_اشارو;
        }

        خودڪار backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        إذا (!backend_init_fn) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, رستو_str(path).c_str());
            }
            أرجع خالي_اشارو;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        إذا (!reg || reg->api_نسخو != GGML_BACKEND_API_نسخو) {
            إذا (!silent) {
                إذا (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, رستو_str(path).c_str());
                } وإلا {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, رستو_str(path).c_str(), reg->api_نسخو, GGML_BACKEND_API_نسخو);
                }
            }
            أرجع خالي_اشارو;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), رستو_str(path).c_str());

        رجسٽر_backend(reg, std::move(handle));

        أرجع reg;
    }

    فراغ unload_backend(ggml_backend_reg_t reg, منطقي silent) {
        خودڪار it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_داخلا & داخلا) { أرجع داخلا.reg == reg; });

        إذا (it == backends.end()) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: backend not found\n", __func__);
            }
            أرجع;
        }

        إذا (!silent) {
            GGML_LOG_DEBUG("%s: unloading %s backend\n", __func__, ggml_backend_reg_name(reg));
        }

        // remove devices
        devices.erase(
            std::remove_if(devices.begin(), devices.end(),
                            [reg](ggml_backend_dev_t dev) { أرجع ggml_backend_dev_backend_reg(dev) == reg; }),
            devices.end());

        // remove backend
        backends.erase(it);
    }
};

ساكن ggml_backend_registry & get_reg() {
    ساكن ggml_backend_registry reg;
    أرجع reg;
}

// Internal API
فراغ ggml_backend_رجسٽر(ggml_backend_reg_t reg) {
    get_reg().رجسٽر_backend(reg);
}

فراغ ggml_backend_device_رجسٽر(ggml_backend_dev_t device) {
    get_reg().رجسٽر_device(device);
}

// Backend (reg) enumeration
ساكن منطقي striequals(ثابت حرف * a, ثابت حرف * b) {
    لكل (; *a && *b; a++, b++) {
        إذا (std::tolower(*a) != std::tolower(*b)) {
            أرجع خطأ;
        }
    }
    أرجع *a == *b;
}

size_t ggml_backend_reg_ڳڻپ() {
    أرجع get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t اشاريو) {
    GGML_ASSERT(اشاريو < ggml_backend_reg_ڳڻپ());
    أرجع get_reg().backends[اشاريو].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_reg_ڳڻپ(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        إذا (striequals(ggml_backend_reg_name(reg), name)) {
            أرجع reg;
        }
    }
    أرجع خالي_اشارو;
}

// Device enumeration
size_t ggml_backend_dev_ڳڻپ() {
    أرجع get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t اشاريو) {
    GGML_ASSERT(اشاريو < ggml_backend_dev_ڳڻپ());
    أرجع get_reg().devices[اشاريو];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_dev_ڳڻپ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (striequals(ggml_backend_dev_name(dev), name)) {
            أرجع dev;
        }
    }
    أرجع خالي_اشارو;
}

ggml_backend_dev_t ggml_backend_dev_by_type(تعداد ggml_backend_dev_type type) {
    لكل (size_t i = 0; i < ggml_backend_dev_ڳڻپ(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (ggml_backend_dev_type(dev) == type) {
            أرجع dev;
        }
    }
    أرجع خالي_اشارو;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    إذا (!dev) {
        أرجع خالي_اشارو;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(تعداد ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    إذا (!dev) {
        أرجع خالي_اشارو;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_بهترين(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    إذا (!dev) {
        أرجع خالي_اشارو;
    }
    أرجع ggml_backend_dev_init(dev, خالي_اشارو);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_ڀريو(ثابت حرف * path) {
    أرجع get_reg().ڀريو_backend(path, خطأ);
}

فراغ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, سچ);
}

ساكن fs::path get_executable_رستو() {
#إذا defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    طالما (سچ) {
        size = path.size();
        إذا (NSGet_Executable_رستو(path.معلومات(), &size) == 0) {
            اكسر;
        }
        path.resize(size);
    }
    std::string بنياد_رستو(path.معلومات(), size);
    // remove executable name
    خودڪار آخري_slash = بنياد_رستو.find_last_of('/');
    إذا (آخري_slash != std::string::npos) {
        بنياد_رستو = بنياد_رستو.substr(0, آخري_slash);
    }
    أرجع بنياد_رستو + "/";
#وإلاإذا defined(__linux__) || defined(__FreeBSD__)
    std::string بنياد_رستو = ".";
    std::vector<حرف> path(1024);
    طالما (سچ) {
        // get executable path
#    إذا defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.معلومات(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.معلومات(), path.size());
#    endif
        إذا (len == -1) {
            اكسر;
        }
        إذا (len < (ssize_t) path.size()) {
            بنياد_رستو = std::string(path.معلومات(), len);
            // remove executable name
            خودڪار آخري_slash = بنياد_رستو.find_last_of('/');
            إذا (آخري_slash != std::string::npos) {
                بنياد_رستو = بنياد_رستو.substr(0, آخري_slash);
            }
            اكسر;
        }
        path.resize(path.size() * 2);
    }

    أرجع بنياد_رستو + "/";
#وإلاإذا defined(_WIN32)
    std::vector<wchar_t> path(MAX_رستو);
    DWORD len = GetModuleFileNameW(عدم, path.معلومات(), path.size());
    إذا (len == 0) {
        أرجع {};
    }
    std::wstring بنياد_رستو(path.معلومات(), len);
    // remove executable name
    خودڪار آخري_slash = بنياد_رستو.find_last_of('\\');
    إذا (آخري_slash != std::string::npos) {
        بنياد_رستو = بنياد_رستو.substr(0, آخري_slash);
    }
    أرجع بنياد_رستو + L"\\";
#وإلا
    أرجع {};
#جيڪڏهن_پورو
}

ساكن fs::path backend_filename_اڳياڙي() {
#جيڪڏهن_وصف_ٿيل _WIN32
    أرجع fs::u8path("ggml-");
#وإلا
    أرجع fs::u8path("libggml-");
#جيڪڏهن_پورو
}

ساكن fs::path backend_filename_extension() {
#جيڪڏهن_وصف_ٿيل _WIN32
    أرجع fs::u8path(".dll");
#وإلا
    أرجع fs::u8path(".so");
#جيڪڏهن_پورو
}

ساكن ggml_backend_reg_t ggml_backend_ڀريو_بهترين(ثابت حرف * name, منطقي silent, ثابت حرف * user_ڳولا_رستو) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_رستو = fs::u8path(name);
    ثابت fs::path file_اڳياڙي = backend_filename_اڳياڙي().native() + name_رستو.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> ڳولا_paths;
    إذا (user_ڳولا_رستو == خالي_اشارو) {
#جيڪڏهن_وصف_ٿيل GGML_BACKEND_DIR
        ڳولا_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#جيڪڏهن_پورو
        // default search paths: executable directory, current directory
        ڳولا_paths.push_back(get_executable_رستو());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        إذا (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.پيغام().c_str());
        } وإلا {
            ڳولا_paths.push_back(cwd);
        }
    } وإلا {
        ڳولا_paths.push_back(fs::u8path(user_ڳولا_رستو));
    }

    صحيح بهترين_score = 0;
    fs::path بهترين_رستو;
    std::error_code ec;

    لكل (ثابت خودڪار & ڳولا_رستو : ڳولا_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(ڳولا_رستو, dir_ec);

        لكل (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت خودڪار & داخلا = *dir_it;
            إذا (داخلا.is_regular_file(ec)) {
                خودڪار filename = داخلا.path().filename();
                خودڪار ext = داخلا.path().extension();
                إذا (filename.native().find(file_اڳياڙي) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_ڀريو_library(داخلا) };
                    إذا (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, رستو_str(داخلا.path()).c_str(), dl_error());
                    }
                    إذا (handle) {
                        خودڪار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        إذا (score_fn) {
                            صحيح s = score_fn();
#جيڪڏهن_وصف_نه_ٿيل NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, رستو_str(داخلا.path()).c_str(), s);
#جيڪڏهن_پورو
                            إذا (s > بهترين_score) {
                                بهترين_score = s;
                                بهترين_رستو = داخلا.path();
                            }
                        } وإلا {
                            إذا (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, رستو_str(داخلا.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        إذا (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, رستو_str(ڳولا_رستو).c_str(), dir_ec.پيغام().c_str());
        }
    }

    إذا (بهترين_score == 0) {
        // try to load the base backend
        لكل (ثابت خودڪار & ڳولا_رستو : ڳولا_paths) {
            fs::path filename = backend_filename_اڳياڙي().native() + name_رستو.native() + backend_filename_extension().native();
            fs::path path = ڳولا_رستو / filename;
            إذا (std::error_code ec; fs::exists(path, ec)) {
                أرجع get_reg().ڀريو_backend(path, silent);
            } وإلا {
                إذا (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, رستو_str(path).c_str(), ec.پيغام().c_str());
                }
            }
        }
        أرجع خالي_اشارو;
    }

    أرجع get_reg().ڀريو_backend(بهترين_رستو, silent);
}

فراغ ggml_backend_ڀريو_all() {
    ggml_backend_ڀريو_all_from_رستو(خالي_اشارو);
}

فراغ ggml_backend_ڀريو_all_from_رستو(ثابت حرف * dir_رستو) {
#جيڪڏهن_وصف_ٿيل NDEBUG
    منطقي silent = سچ;
#وإلا
    منطقي silent = خطأ;
#جيڪڏهن_پورو

    ggml_backend_ڀريو_بهترين("blas", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("zendnn", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("cann", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("cuda", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("hip", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("metal", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("rpc", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("sycl", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("vulkan", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("virtgpu", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("opencl", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("hexagon", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("musa", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("openvino", silent, dir_رستو);
    ggml_backend_ڀريو_بهترين("cpu", silent, dir_رستو);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ثابت حرف * backend_رستو = std::getenv("GGML_BACKEND_PATH");
    إذا (backend_رستو) {
        ggml_backend_ڀريو(backend_رستو);
    }
}
