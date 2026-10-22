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

#که_تعریف_شوی _WIN32
#    define WIN32_LEAN_AND_منځنۍ
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
#د_که_پای

// Backend registry
#که_تعریف_شوی GGML_USE_CPU
#ضمن "ggml-cpu.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_CUDA
#ضمن "ggml-cuda.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_METAL
#ضمن "ggml-metal.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_SYCL
#ضمن "ggml-sycl.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_VULKAN
#ضمن "ggml-vulkan.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_WEBGPU
#ضمن "ggml-webgpu.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_ZDNN
#ضمن "ggml-zdnn.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_OPENCL
#ضمن "ggml-opencl.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_HEXAGON
#ضمن "ggml-hexagon.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_BLAS
#ضمن "ggml-blas.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_RPC
#ضمن "ggml-rpc.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_VIRTGPU_FRONTEND
#ضمن "ggml-virtgpu.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_CANN
#ضمن "ggml-cann.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_ZENDNN
#ضمن "ggml-zendnn.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_OPENVINO
#ضمن "ggml-openvino.h"
#د_که_پای

#که_تعریف_شوی GGML_USE_ET
#ضمن "ggml-et.h"
#د_که_پای

نطاق fs = std::filesystem;

ساكن std::string لار_str(ثابت fs::path & path) {
    حاول {
#إذا defined(__cpp_lib_char8_t)
        // C++20 and later: u8string() returns std::u8string
        ثابت std::u8string u8str = path.u8string();
        أرجع std::string(reinterpret_cast<ثابت حرف *>(u8str.معلومات()), u8str.size());
#وإلا
        // C++17: u8string() returns std::string
        أرجع path.u8string();
#د_که_پای
    } امسك (...) {
        أرجع std::string();
    }
}

بنية ggml_backend_reg_ثبت {
    ggml_backend_reg_t reg;
    dl_handle_ptr handle;
};

بنية ggml_backend_registry {
    std::vector<ggml_backend_reg_ثبت> backends;
    std::vector<ggml_backend_dev_t> devices;

    ggml_backend_registry() {
#که_تعریف_شوی GGML_USE_CUDA
        ثبتوونکی_backend(ggml_backend_cuda_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_METAL
        ثبتوونکی_backend(ggml_backend_metal_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_SYCL
        ثبتوونکی_backend(ggml_backend_sycl_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_VULKAN
    // Add runtime disable check
    إذا (getenv("GGML_DISABLE_VULKAN") == تش_اشاره) {
        ثبتوونکی_backend(ggml_backend_vk_reg());
    } وإلا {
        GGML_LOG_DEBUG("Vulkan backend disabled by GGML_DISABLE_VULKAN environment variable\n");
    }
#د_که_پای
#که_تعریف_شوی GGML_USE_WEBGPU
        ثبتوونکی_backend(ggml_backend_webgpu_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_ZDNN
        ثبتوونکی_backend(ggml_backend_zdnn_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_VIRTGPU_FRONTEND
        ثبتوونکی_backend(ggml_backend_virtgpu_reg());
#د_که_پای

#که_تعریف_شوی GGML_USE_OPENCL
        ثبتوونکی_backend(ggml_backend_opencl_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_ZENDNN
        ثبتوونکی_backend(ggml_backend_zendnn_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_HEXAGON
        ثبتوونکی_backend(ggml_backend_hexagon_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_CANN
        ثبتوونکی_backend(ggml_backend_cann_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_BLAS
        ثبتوونکی_backend(ggml_backend_blas_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_RPC
        ثبتوونکی_backend(ggml_backend_rpc_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_OPENVINO
        ثبتوونکی_backend(ggml_backend_openvino_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_ET
        ثبتوونکی_backend(ggml_backend_et_reg());
#د_که_پای
#که_تعریف_شوی GGML_USE_CPU
        ثبتوونکی_backend(ggml_backend_cpu_reg());
#د_که_پای
    }

    ~ggml_backend_registry() {
        // FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
        // since backend threads may still be running and accessing resources from the dynamic library
        لكل (خپلکاری & ثبت : backends) {
            إذا (ثبت.handle) {
                ثبت.handle.release(); // NOLINT
            }
        }
    }

    فراغ ثبتوونکی_backend(ggml_backend_reg_t reg, dl_handle_ptr handle = تش_اشاره) {
        إذا (!reg) {
            أرجع;
        }

        لكل (خپلکاری & ثبت : backends) {
            إذا (ثبت.reg == reg) {
                أرجع;
            }
        }

#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n",
            __func__, ggml_backend_reg_name(reg), ggml_backend_reg_dev_شمېر(reg));
#د_که_پای
        backends.push_back({ reg, std::move(handle) });
        لكل (size_t i = 0; i < ggml_backend_reg_dev_شمېر(reg); i++) {
            ثبتوونکی_device(ggml_backend_reg_dev_get(reg, i));
        }
    }

    فراغ ثبتوونکی_device(ggml_backend_dev_t device) {
        لكل (خپلکاری & dev : devices) {
            إذا (dev == device) {
                أرجع;
            }
        }

#که_نه_تعریف_شوی NDEBUG
        GGML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, ggml_backend_dev_name(device), ggml_backend_dev_description(device));
#د_که_پای
        devices.push_back(device);
    }

    ggml_backend_reg_t بارول_backend(ثابت fs::path & path, منطقي silent) {
        dl_handle_ptr handle { dl_بارول_library(path) };
        إذا (!handle) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, لار_str(path).c_str(), dl_error());
            }
            أرجع تش_اشاره;
        }

        خپلکاری score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
        إذا (score_fn && score_fn() == 0) {
            إذا (!silent) {
                GGML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, لار_str(path).c_str());
            }
            أرجع تش_اشاره;
        }

        خپلکاری backend_init_fn = (ggml_backend_init_t) dl_get_sym(handle.get(), "ggml_backend_init");
        إذا (!backend_init_fn) {
            إذا (!silent) {
                GGML_LOG_ERROR("%s: failed to find ggml_backend_init in %s\n", __func__, لار_str(path).c_str());
            }
            أرجع تش_اشاره;
        }

        ggml_backend_reg_t reg = backend_init_fn();
        إذا (!reg || reg->api_نسخه != GGML_BACKEND_API_نسخه) {
            إذا (!silent) {
                إذا (!reg) {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: ggml_backend_init returned NULL\n",
                        __func__, لار_str(path).c_str());
                } وإلا {
                    GGML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n",
                        __func__, لار_str(path).c_str(), reg->api_نسخه, GGML_BACKEND_API_نسخه);
                }
            }
            أرجع تش_اشاره;
        }

        GGML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, ggml_backend_reg_name(reg), لار_str(path).c_str());

        ثبتوونکی_backend(reg, std::move(handle));

        أرجع reg;
    }

    فراغ unload_backend(ggml_backend_reg_t reg, منطقي silent) {
        خپلکاری it = std::find_if(backends.begin(), backends.end(),
                               [reg](ثابت ggml_backend_reg_ثبت & ثبت) { أرجع ثبت.reg == reg; });

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
فراغ ggml_backend_ثبتوونکی(ggml_backend_reg_t reg) {
    get_reg().ثبتوونکی_backend(reg);
}

فراغ ggml_backend_device_ثبتوونکی(ggml_backend_dev_t device) {
    get_reg().ثبتوونکی_device(device);
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

size_t ggml_backend_reg_شمېر() {
    أرجع get_reg().backends.size();
}

ggml_backend_reg_t ggml_backend_reg_get(size_t شاخص) {
    GGML_ASSERT(شاخص < ggml_backend_reg_شمېر());
    أرجع get_reg().backends[شاخص].reg;
}

ggml_backend_reg_t ggml_backend_reg_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_reg_شمېر(); i++) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);
        إذا (striequals(ggml_backend_reg_name(reg), name)) {
            أرجع reg;
        }
    }
    أرجع تش_اشاره;
}

// Device enumeration
size_t ggml_backend_dev_شمېر() {
    أرجع get_reg().devices.size();
}

ggml_backend_dev_t ggml_backend_dev_get(size_t شاخص) {
    GGML_ASSERT(شاخص < ggml_backend_dev_شمېر());
    أرجع get_reg().devices[شاخص];
}

ggml_backend_dev_t ggml_backend_dev_by_name(ثابت حرف * name) {
    لكل (size_t i = 0; i < ggml_backend_dev_شمېر(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (striequals(ggml_backend_dev_name(dev), name)) {
            أرجع dev;
        }
    }
    أرجع تش_اشاره;
}

ggml_backend_dev_t ggml_backend_dev_by_type(تعداد ggml_backend_dev_type type) {
    لكل (size_t i = 0; i < ggml_backend_dev_شمېر(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        إذا (ggml_backend_dev_type(dev) == type) {
            أرجع dev;
        }
    }
    أرجع تش_اشاره;
}

// Convenience functions
ggml_backend_t ggml_backend_init_by_name(ثابت حرف * name, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(name);
    إذا (!dev) {
        أرجع تش_اشاره;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_by_type(تعداد ggml_backend_dev_type type, ثابت حرف * params) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    إذا (!dev) {
        أرجع تش_اشاره;
    }
    أرجع ggml_backend_dev_init(dev, params);
}

ggml_backend_t ggml_backend_init_غوره(فراغ) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    dev = dev ? dev : ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    إذا (!dev) {
        أرجع تش_اشاره;
    }
    أرجع ggml_backend_dev_init(dev, تش_اشاره);
}

// Dynamic loading
ggml_backend_reg_t ggml_backend_بارول(ثابت حرف * path) {
    أرجع get_reg().بارول_backend(path, خطأ);
}

فراغ ggml_backend_unload(ggml_backend_reg_t reg) {
    get_reg().unload_backend(reg, ریښتیا);
}

ساكن fs::path get_executable_لار() {
#إذا defined(__APPLE__)
    // get executable path
    std::vector<حرف> path;
    uint32_t size;
    طالما (ریښتیا) {
        size = path.size();
        إذا (NSGet_Executable_لار(path.معلومات(), &size) == 0) {
            اكسر;
        }
        path.resize(size);
    }
    std::string بنسټ_لار(path.معلومات(), size);
    // remove executable name
    خپلکاری وروستی_slash = بنسټ_لار.find_last_of('/');
    إذا (وروستی_slash != std::string::npos) {
        بنسټ_لار = بنسټ_لار.substr(0, وروستی_slash);
    }
    أرجع بنسټ_لار + "/";
#وإلاإذا defined(__linux__) || defined(__FreeBSD__)
    std::string بنسټ_لار = ".";
    std::vector<حرف> path(1024);
    طالما (ریښتیا) {
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
            بنسټ_لار = std::string(path.معلومات(), len);
            // remove executable name
            خپلکاری وروستی_slash = بنسټ_لار.find_last_of('/');
            إذا (وروستی_slash != std::string::npos) {
                بنسټ_لار = بنسټ_لار.substr(0, وروستی_slash);
            }
            اكسر;
        }
        path.resize(path.size() * 2);
    }

    أرجع بنسټ_لار + "/";
#وإلاإذا defined(_WIN32)
    std::vector<wchar_t> path(MAX_لار);
    DWORD len = GetModuleFileNameW(عدم, path.معلومات(), path.size());
    إذا (len == 0) {
        أرجع {};
    }
    std::wstring بنسټ_لار(path.معلومات(), len);
    // remove executable name
    خپلکاری وروستی_slash = بنسټ_لار.find_last_of('\\');
    إذا (وروستی_slash != std::string::npos) {
        بنسټ_لار = بنسټ_لار.substr(0, وروستی_slash);
    }
    أرجع بنسټ_لار + L"\\";
#وإلا
    أرجع {};
#د_که_پای
}

ساكن fs::path backend_filename_مختاړی() {
#که_تعریف_شوی _WIN32
    أرجع fs::u8path("ggml-");
#وإلا
    أرجع fs::u8path("libggml-");
#د_که_پای
}

ساكن fs::path backend_filename_extension() {
#که_تعریف_شوی _WIN32
    أرجع fs::u8path(".dll");
#وإلا
    أرجع fs::u8path(".so");
#د_که_پای
}

ساكن ggml_backend_reg_t ggml_backend_بارول_غوره(ثابت حرف * name, منطقي silent, ثابت حرف * user_پلټنه_لار) {
    // enumerate all the files that match [lib]ggml-name-*.[so|dll] in the search paths
    ثابت fs::path name_لار = fs::u8path(name);
    ثابت fs::path file_مختاړی = backend_filename_مختاړی().native() + name_لار.native() + fs::u8path("-").native();
    ثابت fs::path file_extension = backend_filename_extension();

    std::vector<fs::path> پلټنه_paths;
    إذا (user_پلټنه_لار == تش_اشاره) {
#که_تعریف_شوی GGML_BACKEND_DIR
        پلټنه_paths.push_back(fs::u8path(GGML_BACKEND_DIR));
#د_که_پای
        // default search paths: executable directory, current directory
        پلټنه_paths.push_back(get_executable_لار());
        std::error_code cwd_ec;
        ثابت fs::path cwd = fs::current_path(cwd_ec);
        إذا (cwd_ec) {
            GGML_LOG_DEBUG("%s: failed to get current directory: %s\n", __func__, cwd_ec.پیغام().c_str());
        } وإلا {
            پلټنه_paths.push_back(cwd);
        }
    } وإلا {
        پلټنه_paths.push_back(fs::u8path(user_پلټنه_لار));
    }

    صحيح غوره_score = 0;
    fs::path غوره_لار;
    std::error_code ec;

    لكل (ثابت خپلکاری & پلټنه_لار : پلټنه_paths) {
        std::error_code dir_ec;
        fs::directory_iterator dir_it(پلټنه_لار, dir_ec);

        لكل (ثابت fs::directory_iterator end; dir_it != end; dir_it.increment(dir_ec)) {
            ثابت خپلکاری & ثبت = *dir_it;
            إذا (ثبت.is_regular_file(ec)) {
                خپلکاری filename = ثبت.path().filename();
                خپلکاری ext = ثبت.path().extension();
                إذا (filename.native().find(file_مختاړی) == 0 && ext == file_extension) {
                    dl_handle_ptr handle { dl_بارول_library(ثبت) };
                    إذا (!handle && !silent) {
                        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, لار_str(ثبت.path()).c_str(), dl_error());
                    }
                    إذا (handle) {
                        خپلکاری score_fn = (ggml_backend_score_t) dl_get_sym(handle.get(), "ggml_backend_score");
                        إذا (score_fn) {
                            صحيح s = score_fn();
#که_نه_تعریف_شوی NDEBUG
                            GGML_LOG_DEBUG("%s: %s score: %d\n", __func__, لار_str(ثبت.path()).c_str(), s);
#د_که_پای
                            إذا (s > غوره_score) {
                                غوره_score = s;
                                غوره_لار = ثبت.path();
                            }
                        } وإلا {
                            إذا (!silent) {
                                GGML_LOG_INFO("%s: failed to find ggml_backend_score in %s\n", __func__, لار_str(ثبت.path()).c_str());
                            }
                        }
                    }
                }
            }
        }

        إذا (dir_ec) {
            GGML_LOG_DEBUG("%s: failed to enumerate %s: %s\n", __func__, لار_str(پلټنه_لار).c_str(), dir_ec.پیغام().c_str());
        }
    }

    إذا (غوره_score == 0) {
        // try to load the base backend
        لكل (ثابت خپلکاری & پلټنه_لار : پلټنه_paths) {
            fs::path filename = backend_filename_مختاړی().native() + name_لار.native() + backend_filename_extension().native();
            fs::path path = پلټنه_لار / filename;
            إذا (std::error_code ec; fs::exists(path, ec)) {
                أرجع get_reg().بارول_backend(path, silent);
            } وإلا {
                إذا (ec) {
                    GGML_LOG_DEBUG("%s: posix_stat(%s) failure, error-message: %s\n", __func__, لار_str(path).c_str(), ec.پیغام().c_str());
                }
            }
        }
        أرجع تش_اشاره;
    }

    أرجع get_reg().بارول_backend(غوره_لار, silent);
}

فراغ ggml_backend_بارول_all() {
    ggml_backend_بارول_all_from_لار(تش_اشاره);
}

فراغ ggml_backend_بارول_all_from_لار(ثابت حرف * dir_لار) {
#که_تعریف_شوی NDEBUG
    منطقي silent = ریښتیا;
#وإلا
    منطقي silent = خطأ;
#د_که_پای

    ggml_backend_بارول_غوره("blas", silent, dir_لار);
    ggml_backend_بارول_غوره("zendnn", silent, dir_لار);
    ggml_backend_بارول_غوره("cann", silent, dir_لار);
    ggml_backend_بارول_غوره("cuda", silent, dir_لار);
    ggml_backend_بارول_غوره("hip", silent, dir_لار);
    ggml_backend_بارول_غوره("metal", silent, dir_لار);
    ggml_backend_بارول_غوره("rpc", silent, dir_لار);
    ggml_backend_بارول_غوره("sycl", silent, dir_لار);
    ggml_backend_بارول_غوره("vulkan", silent, dir_لار);
    ggml_backend_بارول_غوره("virtgpu", silent, dir_لار);
    ggml_backend_بارول_غوره("opencl", silent, dir_لار);
    ggml_backend_بارول_غوره("hexagon", silent, dir_لار);
    ggml_backend_بارول_غوره("musa", silent, dir_لار);
    ggml_backend_بارول_غوره("openvino", silent, dir_لار);
    ggml_backend_بارول_غوره("cpu", silent, dir_لار);
    // check the environment variable GGML_BACKEND_PATH to load an out-of-tree backend
    ثابت حرف * backend_لار = std::getenv("GGML_BACKEND_PATH");
    إذا (backend_لار) {
        ggml_backend_بارول(backend_لار);
    }
}
