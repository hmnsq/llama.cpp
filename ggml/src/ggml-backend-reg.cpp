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

#اگر_تعریف_شده _WIN32
#    define WIN32_LEAN_AND_میانگین
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
#پایان_اگر

// Backend registry
#اگر_تعریف_شده GGML_USE_CPU
#ضمن "ggml-cpu.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_CUDA
#ضمن "ggml-cuda.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_METAL
#ضمن "ggml-metal.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_SYCL
#ضمن "ggml-sycl.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_VULKAN
#ضمن "ggml-vulkan.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_WEBGPU
#ضمن "ggml-webgpu.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_ZDNN
#ضمن "ggml-zdnn.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_OPENCL
#ضمن "ggml-opencl.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_HEXAGON
#ضمن "ggml-hexagon.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_BLAS
#ضمن "ggml-blas.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_RPC
#ضمن "ggml-rpc.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_VIRTGPU_FRONTEND
#ضمن "ggml-virtgpu.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_CANN
#ضمن "ggml-cann.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_ZENDNN
#ضمن "ggml-zendnn.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_OPENVINO
#ضمن "ggml-openvino.h"
#پایان_اگر

#اگر_تعریف_شده GGML_USE_ET
#ضمن "ggml-et.h"
#پایان_اگر

نطاق fs = std::filesystem;

ساكن std::string مسیر_str(ثابت fs::path & path) {
    حاول {
#إذا defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        أرجع std::string(reinterpret_cast<ثابت حرف *>(u8str.داده()), u8str.size());
#وإلا
        // C++17: u8string() returns std::string
        أرجع path.u8string();
#پایان_اگر
    } امسك (...) {
        أرجع std::string();
    }
}

بنية ggml_backend_reg_مدخل {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

بنية ggml_backend_registry {
    std::vector<ggml_backend_reg_مدخل> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#اگر_تعریف_شده GGML_USE_CUDA
        ثبات_backend(ggml_backend_cuda_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_METAL
        ثبات_backend(ggml_backend_metal_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_SYCL
        ثبات_backend(ggml_backend_sycl_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_VULKAN
    // Add runtime disable check
    إذا (getenv("GGML_DISABLE_VULKAN") == اشاره_تهی) {
        ثبات_backend(ggml_backend_vk_reg());
    } وإلا {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#پایان_اگر
#اگر_تعریف_شده GGML_USE_WEBGPU
        ثبات_backend(ggml_backend_webgpu_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_ZDNN
        ثبات_backend(ggml_backend_zdnn_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_VIRTGPU_FRONTEND
        ثبات_backend(ggml_backend_virtgpu_reg());
#پایان_اگر

#اگر_تعریف_شده GGML_USE_OPENCL
        ثبات_backend(ggml_backend_opencl_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_ZENDNN
        ثبات_backend(ggml_backend_zendnn_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_HEXAGON
        ثبات_backend(ggml_backend_hexagon_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_CANN
        ثبات_backend(ggml_backend_cann_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_BLAS
        ثبات_backend(ggml_backend_blas_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_RPC
        ثبات_backend(ggml_backend_rpc_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_OPENVINO
        ثبات_backend(ggml_backend_openvino_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_ET
        ثبات_backend(ggml_backend_et_reg());
#پایان_اگر
#اگر_تعریف_شده GGML_USE_CPU
        ثبات_backend(ggml_backend_cpu_reg());
#پایان_اگر
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        لكل (خودکار & مدخل : backends) {
            إذا (مدخل.handle) {
                مدخل.handle.release(); // NOLINT
            }
        }
    }

    فراغ ثبات_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = اشاره_تهی) {
        إذا (!reg) {
            أرجع;
        }

        لكل (خودکار & مدخل : backends) {
            إذا (مدخل.reg == reg) {
                أرجع;
            }
        }

#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_شمار(reg));
#پایان_اگر
        backends.push_back({ reg, std::move(handle) });
        لكل (size_t i = 0; i < ggml_backend_reg_dev_شمار(reg); i++) {
            ثبات_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    فراغ ثبات_device(ggml_backend_dev_t device) {
        لكل (خودکار & dev : devices) {
            إذا (dev == device) {
                أرجع;
            }
        }

#اگر_تعریف_نشده NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#پایان_اگر
        devices.push_back(device);
    }

    ggml_backend_reg_t بارگیری_backend(ثابت fs::path & path, منطقي silent) {
        dl_handle_ptr handle { dl_بارگیری_library(path) };
        إذا (!handle) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, مسیر_str(path).c_str(), dl_error());
            }
            أرجع اشاره_تهی;
        }

        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        إذا (score_fn && score_fn() == 0) {
            إذا (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, مسیر_str(path).c_str());
            }
            أرجع اشاره_تهی;
        }

        خودکار backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        إذا (!backend_init_fn) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, مسیر_str(path).c_str());
            }
            أرجع اشاره_تهی;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        إذا (!reg || reg->api_نسخه != GGML_BACKEND_API_نسخه) {
            إذا (!silent) {
                إذا (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, مسیر_str(path).c_str());
                } وإلا {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, مسیر_str(path).c_str(), reg->api_نسخه, GGML_BACKEND_API_نسخه);
                }
            }
            أرجع اشاره_تهی;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), مسیر_str(path).c_str());

        ثبات_backend(reg, std::move(handle));

        أرجع reg;
    }

    فراغ unload_backend(ggml_backend_reg_t reg, منطقي silent) {
        خودکار it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_مدخل & مدخل) { أرجع مدخل.reg == reg; });

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
فراغ ggml_backend_ثبات(ggml_backend_reg_t reg) {
    get_reg().ثبات_backend(reg);
}

فراغ ggml_backend_device_ثبات(ggml_backend_dev_t device) {
    get_reg().ثبات_device(device);
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

size_t ggml_backend_reg_شمار() {
    أرجع get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t شاخص) {
    GGML_ASSERT(شاخص < ggml_backend_reg_شمار());
    أرجع get_reg().backends[شاخص].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_reg_شمار(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        إذا (striequals(ggml_backend_reg_name(reg), name)) {
            أرجع reg;
        }
    }
    أرجع اشاره_تهی;
}

// Device enumeration
size_t ggml_backend_dev_شمار() {
    أرجع get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t شاخص) {
    GGML_ASSERT(شاخص < ggml_backend_dev_شمار());
    أرجع get_reg().devices[شاخص];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_dev_شمار(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (striequals(ggml_backend_dev_name(dev), name)) {
            أرجع dev;
        }
    }
    أرجع اشاره_تهی;
}

ggml_backend_dev_t ggml_backend_dev_by_type(تعداد ggml_backend_dev_type type) {
    لكل (size_t i = 0; i < ggml_backend_dev_شمار(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (ggml_backend_dev_type(dev) == type) {
            أرجع dev;
        }
    }
    أرجع اشاره_تهی;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    إذا (!dev) {
        أرجع اشاره_تهی;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(تعداد ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    إذا (!dev) {
        أرجع اشاره_تهی;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_بهترین(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    إذا (!dev) {
        أرجع اشاره_تهی;
    }
    أرجع ggml_backend_dev_init(dev, اشاره_تهی);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_بارگیری(ثابت حرف * path) {
    أرجع get_reg().بارگیری_backend(path, خطأ);
}

فراغ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, درست);
}

ساكن fs::path get_executable_مسیر() {
#إذا defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    طالما (درست) {
        size = path.size();
        إذا (NSGet_Executable_مسیر(path.داده(), &size) == 0) {
            اكسر;
        }
        path.resize(size);
    }
    std::string پایه_مسیر(path.داده(), size);
    // remove executable name
    خودکار آخرین_slash = پایه_مسیر.find_last_of('/');
    إذا (آخرین_slash != std::string::npos) {
        پایه_مسیر = پایه_مسیر.substr(0, آخرین_slash);
    }
    أرجع پایه_مسیر + "/";
#وإلاإذا defined(__linux__) || defined(__FreeBSD__)
    std::string پایه_مسیر = ".";
    std::vector<حرف> path(1024);
    طالما (درست) {
        // get executable path
#    إذا defined(__linux__)
        ssize_t len = readlink("/proc/self/exe", path.داده(), path.size());
#    elif defined(__FreeBSD__)
        ssize_t len = readlink("/proc/curproc/file", path.داده(), path.size());
#    endif
        إذا (len == -1) {
            اكسر;
        }
        إذا (len < (ssize_t) path.size()) {
            پایه_مسیر = std::string(path.داده(), len);
            // remove executable name
            خودکار آخرین_slash = پایه_مسیر.find_last_of('/');
            إذا (آخرین_slash != std::string::npos) {
                پایه_مسیر = پایه_مسیر.substr(0, آخرین_slash);
            }
            اكسر;
        }
        path.resize(path.size() * 2);
    }

    أرجع پایه_مسیر + "/";
#وإلاإذا defined(_WIN32)
    std::vector<wchar_t> path(MAX_مسیر);
    DWORD len = GetModuleFileNameW(عدم, path.داده(), path.size());
    إذا (len == 0) {
        أرجع {};
    }
    std::wstring پایه_مسیر(path.داده(), len);
    // remove executable name
    خودکار آخرین_slash = پایه_مسیر.find_last_of('\\');
    إذا (آخرین_slash != std::string::npos) {
        پایه_مسیر = پایه_مسیر.substr(0, آخرین_slash);
    }
    أرجع پایه_مسیر + L"\\";
#وإلا
    أرجع {};
#پایان_اگر
}

ساكن fs::path backend_filename_پیشوند() {
#اگر_تعریف_شده _WIN32
    أرجع fs::u8path("ggml-");
#وإلا
    أرجع fs::u8path("libggml-");
#پایان_اگر
}

ساكن fs::path backend_filename_extension() {
#اگر_تعریف_شده _WIN32
    أرجع fs::u8path(".dll");
#وإلا
    أرجع fs::u8path(".so");
#پایان_اگر
}

ساكن ggml_backend_reg_t ggml_backend_بارگیری_بهترین(ثابت حرف * name, منطقي silent, ثابت حرف * user_جستجو_مسیر) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_مسیر = fs::u8path(name);
    ثابت fs::path file_پیشوند = backend_filename_پیشوند().native() + name_مسیر.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> جستجو_paths;
    إذا (user_جستجو_مسیر == اشاره_تهی) {
#اگر_تعریف_شده GGML_BACKEND_DIR
        جستجو_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#پایان_اگر
        // default search paths: executable directory, current directory
        جستجو_paths.push_back(get_executable_مسیر());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        إذا (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.پیام().c_str());
        } وإلا {
            جستجو_paths.push_back(cwd);
        }
    } وإلا {
        جستجو_paths.push_back(fs::u8path(user_جستجو_مسیر));
    }

    صحيح بهترین_score = 0;
    fs::path بهترین_مسیر;
    std::error_code ec;

    لكل (ثابت خودکار & جستجو_مسیر : جستجو_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(جستجو_مسیر, dir_ec);

        لكل (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت خودکار & مدخل = *dir_it;
            إذا (مدخل.is_regular_file(ec)) {
                خودکار filename = مدخل.path().filename();
                خودکار ext = مدخل.path().extension();
                إذا (filename.native().find(file_پیشوند) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_بارگیری_library(مدخل) };
                    إذا (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, مسیر_str(مدخل.path()).c_str(), dl_error());
                    }
                    إذا (handle) {
                        خودکار score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        إذا (score_fn) {
                            صحيح s = score_fn();
#اگر_تعریف_نشده NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, مسیر_str(مدخل.path()).c_str(), s);
#پایان_اگر
                            إذا (s > بهترین_score) {
                                بهترین_score = s;
                                بهترین_مسیر = مدخل.path();
                            }
                        } وإلا {
                            إذا (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, مسیر_str(مدخل.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        إذا (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, مسیر_str(جستجو_مسیر).c_str(), dir_ec.پیام().c_str());
        }
    }

    إذا (بهترین_score == 0) {
        // try to load the base backend
        لكل (ثابت خودکار & جستجو_مسیر : جستجو_paths) {
            fs::path filename = backend_filename_پیشوند().native() + name_مسیر.native() + backend_filename_extension().native();
            fs::path path = جستجو_مسیر / filename;
            إذا (std::error_code ec; fs::exists(path, ec)) {
                أرجع get_reg().بارگیری_backend(path, silent);
            } وإلا {
                إذا (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, مسیر_str(path).c_str(), ec.پیام().c_str());
                }
            }
        }
        أرجع اشاره_تهی;
    }

    أرجع get_reg().بارگیری_backend(بهترین_مسیر, silent);
}

فراغ ggml_backend_بارگیری_all() {
    ggml_backend_بارگیری_all_from_مسیر(اشاره_تهی);
}

فراغ ggml_backend_بارگیری_all_from_مسیر(ثابت حرف * dir_مسیر) {
#اگر_تعریف_شده NDEBUG
    منطقي silent = درست;
#وإلا
    منطقي silent = خطأ;
#پایان_اگر

    ggml_backend_بارگیری_بهترین("blas", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("zendnn", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("cann", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("cuda", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("hip", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("metal", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("rpc", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("sycl", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("vulkan", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("virtgpu", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("opencl", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("hexagon", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("musa", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("openvino", silent, dir_مسیر);
    ggml_backend_بارگیری_بهترین("cpu", silent, dir_مسیر);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ثابت حرف * backend_مسیر = std::getenv("GGML_BACKEND_PATH");
    إذا (backend_مسیر) {
        ggml_backend_بارگیری(backend_مسیر);
    }
}
